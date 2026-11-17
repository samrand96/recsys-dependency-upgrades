"""Synthetic positive controls (protocol/synthetic_controls.md).

Each fault is applied to the new side's saved inference artifacts of a held-out base case,
evaluated by the frozen evaluator, labelled by the full reference, and then scored by every
frozen detector. Results go to results/synthetic/ and never into the runs table.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from silentrank import paths
from silentrank.orchestrator import evaluation, ledger, manifests
from silentrank.stats.paired import ANALYSIS_SEED, paired_user_bootstrap
from silentrank.stats.verdict import VERDICT_TOLERANCES, classify_verdict
from silentrank.worker.ranking import native_topk
from silentrank.worker.tracing import CallTrace

from .panels import PANEL_SIZES, PANEL_STRATEGIES, build_panels

FAULTS = ("F1", "F2", "F3", "F4")
FAULT_NAMES = {
    "F1": "swapped item-id mapping for 1% of items",
    "F2": "seen-item mask dropped",
    "F3": "scores rounded to 2 decimals",
    "F4": "reversed sort for 1% of users",
}
SCORE_TOLERANCE = 1e-6
LENIENT_OVERLAP = 0.99
PRIMARY = ("activity", 300, 0.005)
_COPIED = ("user_ids.npy", "item_ids.npy", "seen_mask.npy")


# ---------------------------------------------------------------- base cases

def base_cases() -> list[str]:
    """Held-out ranking cases with an accepted inference A/A and a completed A/B."""
    heldout = {entry["id"] for entry in manifests.load_all() if entry.get("split") == "heldout"}
    cases = []
    for path in sorted((paths.results_dir() / "compare").glob("*.json")):
        if path.stem.split("-", 1)[0] not in heldout or "-P0" in path.stem:
            continue
        inference = json.loads(path.read_text(encoding="utf-8")).get("modes", {}).get("inference", {})
        if inference.get("AA", {}).get("status") == "ok" and inference.get("AB", {}).get("status") == "ok":
            cases.append(path.stem)
    return cases


def _inference_runs(case_id: str) -> tuple[Path, Path, Path, dict]:
    report = json.loads((paths.results_dir() / "compare" / f"{case_id}.json").read_text(encoding="utf-8"))
    inference = report["modes"]["inference"]
    old_id = inference["AA"]["runs"][0]["old"]
    repeat_id = inference["AA"]["runs"][0]["new"]
    new_id = inference["AB"]["runs"][0]["new"]
    if inference["AB"]["runs"][0]["old"] != old_id:
        raise ValueError(f"A/A and A/B must share one baseline for {case_id}")
    with ledger.reader() as con:
        rows = dict(con.execute("SELECT run_id, artifact_dir FROM runs WHERE run_id IN (?, ?, ?)",
                                [old_id, repeat_id, new_id]).fetchall())
    return Path(rows[old_id]), Path(rows[repeat_id]), Path(rows[new_id]), {
        "old": old_id, "old_repeat": repeat_id, "new": new_id, "reference_verdicts": inference.get("verdicts", [])}


def _dataset(directory: Path) -> Path:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    manifest = json.loads((paths.repo_root() / "data" / "manifest.json").read_text(encoding="utf-8"))
    if meta["inputs"]["dataset_sha256"] != manifest["prepared"]["sha256"]:
        raise ValueError("synthetic controls use the primary MovieLens 1M split only")
    return paths.repo_root() / "data" / manifest["prepared"]["filename"]


# ---------------------------------------------------------------- faults

def _rng() -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(ANALYSIS_SEED))


def _rebuild(scores: np.ndarray, mask: np.ndarray | None, item_ids: np.ndarray, *,
             ascending: np.ndarray | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Native top-10/top-20 lists, as worker/ranking.native_topk builds them."""
    trace = CallTrace(None)
    lists = {}
    for k in (10, 20):
        items = np.full((len(scores), k), -1, dtype=np.int64)
        values = np.full((len(scores), k), np.nan, dtype=scores.dtype)
        for row in range(len(scores)):
            eligible = np.arange(scores.shape[1]) if mask is None else np.flatnonzero(~mask[row])
            row_scores = scores[row, eligible]
            key = -row_scores if ascending is not None and ascending[row] else row_scores
            chosen = eligible[native_topk(key, k, trace)]
            items[row, :chosen.size] = item_ids[chosen]
            values[row, :chosen.size] = scores[row, chosen]
        lists[k] = (items, values)
    return lists


def apply_fault(fault: str, scores: np.ndarray, mask: np.ndarray, item_ids: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    """Return faulted scores, native lists and a description of what was changed."""
    rng = _rng()
    if fault == "F1":
        count = math.ceil(0.01 * scores.shape[1])
        columns = np.sort(rng.choice(scores.shape[1], size=count, replace=False))
        faulted = scores.copy()
        faulted[:, columns] = scores[:, np.roll(columns, -1)]
        return faulted, _rebuild(faulted, mask, item_ids), {"swapped_item_ids": item_ids[columns].tolist()}
    if fault == "F2":
        return scores.copy(), _rebuild(scores, None, item_ids), {}
    if fault == "F3":
        faulted = np.round(scores, 2)
        return faulted, _rebuild(faulted, mask, item_ids), {}
    if fault == "F4":
        count = math.ceil(0.01 * len(scores))
        rows = np.sort(rng.choice(len(scores), size=count, replace=False))
        ascending = np.zeros(len(scores), dtype=bool)
        ascending[rows] = True
        return scores.copy(), _rebuild(scores, mask, item_ids, ascending=ascending), {"reversed_rows": rows.tolist()}
    raise ValueError(f"unknown fault {fault!r}")


def _write_faulted(source: Path, target: Path, scores: np.ndarray, lists: dict) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in _COPIED:
        shutil.copyfile(source / name, target / name)
    np.save(target / "scores.npy", scores, allow_pickle=False)
    for k, (items, values) in lists.items():
        suffix = "10" if k == 10 else ""
        np.save(target / f"topk_native{suffix}.npy", items, allow_pickle=False)
        np.save(target / f"scores_topk{suffix}.npy", values, allow_pickle=False)


# ---------------------------------------------------------------- detection

def _ndcg(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["user", "activity", "ndcg10"])
    return table["user"].to_numpy(), table["activity"].to_numpy(), table["ndcg10"].to_numpy(zero_copy_only=False)


def _violating_rows(directory: Path, users: np.ndarray) -> np.ndarray:
    violations = json.loads((directory / "contracts.json").read_text(encoding="utf-8"))
    return np.isin(users, [entry["user"] for entry in violations])


def _decide(loss: np.ndarray | None, violating: np.ndarray, positions: np.ndarray) -> dict[float, str]:
    if bool(violating[positions].any()):
        return {delta: "contract_failure" for delta in VERDICT_TOLERANCES}
    interval = paired_user_bootstrap(loss[positions])
    return {delta: classify_verdict(interval.low, interval.high, delta) for delta in VERDICT_TOLERANCES}


def _baselines(old: Path, other: Path, mask: np.ndarray) -> dict[str, bool]:
    """The frozen baselines, computed with the same diagnostics as the real comparisons.

    ``comparisons._ranking_diagnostics`` scales scores before squaring, so sentinel values
    (implicit's -DBL_MAX for unscorable items) cannot overflow into a spurious alarm.
    """
    from silentrank.orchestrator.comparisons import _ranking_diagnostics, _relative_norm

    a_lists = np.load(old / "topk_native10.npy", allow_pickle=False)
    b_lists = np.load(other / "topk_native10.npy", allow_pickle=False)
    overlaps = []
    for row_a, row_b, row_mask in zip(a_lists, b_lists, mask):
        size = min(10, int(np.count_nonzero(~row_mask)))
        overlaps.append(len(set(row_a[row_a >= 0]) & set(row_b[row_b >= 0])) / size)
    try:
        diagnostics = _ranking_diagnostics({"directory": old}, {"directory": other})
        relative = _relative_norm(diagnostics["score_baseline_sum_squares"], diagnostics["score_difference_sum_squares"])
        nonfinite = False
    except ValueError as exc:
        if "nonfinite" not in str(exc):
            raise
        relative, nonfinite = None, True  # a non-finite eligible score is itself a detectable score change
    return {
        "exact_snapshot": bool(not np.array_equal(a_lists, b_lists)),
        "score_tolerance": bool(relative is None or relative > SCORE_TOLERANCE),
        "overlap_strict": bool(min(overlaps) < 1.0),
        "overlap_lenient": bool(float(np.mean(overlaps)) < LENIENT_OVERLAP),
        "score_rel_l2": relative,
        "nonfinite_scores": nonfinite,
    }


def _detect(old: Path, other: Path, users, activity, old_loss_base, other_metrics: Path, panels, mask) -> dict:
    _, _, other_ndcg = _ndcg(other_metrics)
    violating = _violating_rows(other, users)
    loss = np.where(np.isnan(other_ndcg), np.nan, old_loss_base - other_ndcg)
    everyone = np.arange(len(users), dtype=np.int64)
    result = {"reference": _decide(loss, violating, everyone), "contract_violations": int(violating.sum()),
              "full_loss_mean": None if violating.any() else float(np.mean(loss)),
              "panels": {f"{p.strategy}:{len(p.positions)}": _decide(loss, violating, p.positions) for p in panels},
              "baselines": _baselines(old, other, mask)}
    return result


# ---------------------------------------------------------------- driver

def run_case(case_id: str) -> dict:
    old_dir, repeat_dir, new_dir, ids = _inference_runs(case_id)
    dataset = _dataset(old_dir)
    users, activity, old_ndcg = _ndcg(paths.results_dir() / "metrics" / f"{ids['old']}.parquet")
    scores = np.load(old_dir / "scores.npy", allow_pickle=False)
    mask = np.load(old_dir / "seen_mask.npy", allow_pickle=False)
    panels = build_panels(activity, scores, mask, PANEL_SIZES)
    metrics_dir = paths.results_dir() / "metrics"
    record = {"case": case_id, "runs": ids, "faults": {},
              "aa": _detect(old_dir, repeat_dir, users, activity, old_ndcg, metrics_dir / f"{ids['old_repeat']}.parquet", panels, mask),
              "ab": _detect(old_dir, new_dir, users, activity, old_ndcg, metrics_dir / f"{ids['new']}.parquet", panels, mask)}
    new_scores = np.load(new_dir / "scores.npy", allow_pickle=False)
    item_ids = np.load(new_dir / "item_ids.npy", allow_pickle=False)
    root = paths.results_dir() / "synthetic" / case_id
    for fault in FAULTS:
        faulted, lists, detail = apply_fault(fault, new_scores, mask, item_ids)
        target = root / fault
        _write_faulted(new_dir, target, faulted, lists)
        evaluated = evaluation.evaluate(target, dataset, "test", target / "metrics.parquet")
        record["faults"][fault] = {"detail": detail, "violations": evaluated["violations"],
                                   **_detect(old_dir, target, users, activity, old_ndcg, target / "metrics.parquet", panels, mask)}
        for name in ("scores.npy",):  # keep lists, metrics and contracts; drop the large score copy
            (target / name).unlink()
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _alarm(decision: dict[float, str], delta: float) -> bool:
    return decision[delta] in {"harmful", "contract_failure"}


def summarize(records: list[dict]) -> dict:
    """Recall on synthetic positives, alarms on real A/A and benign A/B, and cost per detector."""
    detectors = [f"{s}:{n}" for n in PANEL_SIZES for s in PANEL_STRATEGIES]
    baselines = ["exact_snapshot", "score_tolerance", "overlap_strict", "overlap_lenient"]
    n_users = 1007
    rows = []
    for delta in VERDICT_TOLERANCES:
        positives = [(r["case"], f, r["faults"][f]) for r in records for f in FAULTS if _alarm(r["faults"][f]["reference"], delta)]
        benign = [r for r in records if r["ab"]["reference"][delta] == "equivalent"]
        for name in detectors + baselines + ["full_check"]:
            def hit(entry):
                if name == "full_check":
                    return _alarm(entry["reference"], delta)
                if name in baselines:
                    return bool(entry["baselines"][name])
                return _alarm(entry["panels"][name], delta)
            size = int(name.split(":")[1]) if ":" in name else n_users
            per_fault = {}
            for fault in FAULTS:
                fault_positives = [entry for _, f, entry in positives if f == fault]
                per_fault[fault] = {"positives": len(fault_positives),
                                    "detected": sum(hit(entry) for entry in fault_positives)}
            rows.append({
                "delta": delta, "detector": name,
                "paired_user_requests": 2 * size,
                "synthetic_positives": len(positives),
                "detected": sum(hit(entry) for _, _, entry in positives),
                "recall": (sum(hit(entry) for _, _, entry in positives) / len(positives)) if positives else None,
                "per_fault": per_fault,
                "aa_controls": len(records), "aa_alarms": sum(hit(r["aa"]) for r in records),
                "benign_ab_cases": len(benign), "benign_ab_alarms": sum(hit(r["ab"]) for r in benign),
            })
    labels = {fault: {str(delta): {
        verdict: sum(r["faults"][fault]["reference"][delta] == verdict for r in records)
        for verdict in ("contract_failure", "harmful", "equivalent", "improved", "inconclusive")}
        for delta in VERDICT_TOLERANCES} for fault in FAULTS}
    return {"n_base_cases": len(records), "primary": dict(zip(("strategy", "size", "delta"), PRIMARY)),
            "reference_labels": labels, "rows": rows}


def run_all(case_ids: list[str] | None = None) -> dict:
    cases = case_ids or base_cases()
    for case_id in cases:
        if not re.fullmatch(r"[\w-]+", case_id, flags=re.ASCII):
            raise ValueError(f"invalid case id {case_id!r}")
    root = paths.results_dir() / "synthetic"
    records = []
    for case_id in cases:
        existing = root / case_id / "result.json"
        records.append(json.loads(existing.read_text(encoding="utf-8")) if existing.is_file() else run_case(case_id))
    summary = summarize(records)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
