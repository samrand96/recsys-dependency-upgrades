"""Synthetic positive controls, analysis version 2 (protocol/synthetic_controls_v2.md).

Version 1 (synthetic.py, results/synthetic/) rebuilt every user's list with the reference
ranker, so users outside a fault's declared scope could change too. Version 2 keeps each
user's saved native list unless the fault changes that user's reference ranking, adds a
no-fault reconstruction control (F0), pins the base-case cohort, and serialises tolerance
keys so cached records can be resumed. Version 1 records are never modified.

Outputs go to results/synthetic_v2/ and never into the runs table.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from silentrank import paths
from silentrank.orchestrator import evaluation
from silentrank.stats.verdict import VERDICT_TOLERANCES

from . import synthetic as v1
from .panels import PANEL_SIZES, PANEL_STRATEGIES, build_panels

ANALYSIS_VERSION = 2
FAULTS = ("F1", "F2", "F3", "F4")
CONTROL = "F0"
OUTPUT = "synthetic_v2"
_NATIVE = {10: ("topk_native10.npy", "scores_topk10.npy"), 20: ("topk_native.npy", "scores_topk.npy")}


def cohort() -> list[str]:
    """The pinned base cases: the 97 held-out MovieLens 1M cases of analysis version 1."""
    path = paths.repo_root() / "protocol" / "synthetic_cohort_v1.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    if len(cases) != len(set(cases)) or not all(re.fullmatch(r"[\w-]+", c, flags=re.ASCII) for c in cases):
        raise ValueError("the pinned cohort is malformed")
    return cases


def _native(directory: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {k: tuple(np.load(directory / name, allow_pickle=False) for name in names) for k, names in _NATIVE.items()}


def _values(scores: np.ndarray, items: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    """Scores of the listed items; padding (-1) stays NaN."""
    column = {int(item): index for index, item in enumerate(item_ids)}
    values = np.full(items.shape, np.nan, dtype=scores.dtype)
    for row in range(items.shape[0]):
        for position, item in enumerate(items[row]):
            if item >= 0:
                values[row, position] = scores[row, column[int(item)]]
    return values


def faithful_rebuild(scores: np.ndarray, mask: np.ndarray | None, item_ids: np.ndarray,
                     native: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Top-K by descending score, with exact ties resolved in the saved native order.

    Items in the user's native list rank among equal scores by their native position; all
    other items follow in catalogue order. With unchanged scores and mask this reproduces
    every native list that is consistent with the scores.
    """
    column = {int(item): index for index, item in enumerate(item_ids)}
    lists = {}
    for k in (10, 20):
        items = np.full((len(scores), k), -1, dtype=np.int64)
        values = np.full((len(scores), k), np.nan, dtype=scores.dtype)
        for row in range(len(scores)):
            prior = np.arange(k, k + scores.shape[1], dtype=np.int64)
            for position, item in enumerate(native[k][0][row]):
                if item >= 0:
                    prior[column[int(item)]] = position
            eligible = np.arange(scores.shape[1]) if mask is None else np.flatnonzero(~mask[row])
            order = np.lexsort((prior[eligible], -scores[row, eligible]))[:k]
            chosen = eligible[order]
            items[row, :chosen.size] = item_ids[chosen]
            values[row, :chosen.size] = scores[row, chosen]
        lists[k] = (items, values)
    return lists


def reconstruction_control(scores: np.ndarray, mask: np.ndarray, item_ids: np.ndarray) -> dict:
    """F0: the version-1 rebuild with no fault. Measures what reconstruction alone changes."""
    return v1._rebuild(scores, mask, item_ids)


def isolated_fault(fault: str, scores: np.ndarray, mask: np.ndarray, item_ids: np.ndarray,
                   native: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, dict, dict]:
    """Apply one fault so that only users whose reference ranking it changes get a new list.

    For every user and cutoff, a reference ranker is run on the unchanged inputs (R0) and
    on the faulted inputs (R1). Where R1 equals R0 the saved native list is kept; elsewhere
    the list becomes R1. F4 replaces exactly the reversed rows. Listed scores are always
    read from the faulted score matrix.
    """
    rng = v1._rng()
    faulted, fault_mask, ascending, detail = scores, mask, None, {}
    if fault == "F1":
        count = math.ceil(0.01 * scores.shape[1])
        columns = np.sort(rng.choice(scores.shape[1], size=count, replace=False))
        faulted = scores.copy()
        faulted[:, columns] = scores[:, np.roll(columns, -1)]
        detail["swapped_item_ids"] = item_ids[columns].tolist()
    elif fault == "F2":
        fault_mask = None
    elif fault == "F3":
        faulted = np.round(scores, 2)
    elif fault == "F4":
        count = math.ceil(0.01 * len(scores))
        rows = np.sort(rng.choice(len(scores), size=count, replace=False))
        ascending = np.zeros(len(scores), dtype=bool)
        ascending[rows] = True
        detail["reversed_rows"] = rows.tolist()
    else:
        raise ValueError(f"unknown fault {fault!r}")

    if fault in ("F1", "F2"):
        # These faults do not create ties by themselves, so ties are resolved in the saved
        # native order: the reconstruction then adds no tie reordering of its own.
        before = faithful_rebuild(scores, mask, item_ids, native)
        after = faithful_rebuild(faulted, fault_mask, item_ids, native)
        detail["faithful_mismatch_rows"] = int(np.any(before[10][0] != native[10][0], axis=1).sum())
    else:
        # F3 creates ties by rounding and F4 reverses the order: how the reference ranker
        # orders the result is part of the declared fault.
        before = v1._rebuild(scores, mask, item_ids)
        after = v1._rebuild(faulted, fault_mask, item_ids, ascending=ascending)
    lists, rebuilt = {}, {}
    for k in (10, 20):
        changed = ascending.copy() if ascending is not None else np.any(before[k][0] != after[k][0], axis=1)
        items = native[k][0].copy()
        items[changed] = after[k][0][changed]
        if np.any(items[~changed] != native[k][0][~changed]):
            raise AssertionError("a list outside the fault's footprint was modified")
        lists[k] = (items, _values(faulted, items, item_ids))
        rebuilt[k] = int(changed.sum())
    detail["rows_rebuilt"] = {str(k): n for k, n in rebuilt.items()}
    detail["users_changed_top10"] = int(np.any(lists[10][0] != native[10][0], axis=1).sum())
    return faulted, lists, detail


def run_case(case_id: str) -> dict:
    old_dir, repeat_dir, new_dir, ids = v1._inference_runs(case_id)
    dataset = v1._dataset(old_dir)
    users, activity, old_ndcg = v1._ndcg(paths.results_dir() / "metrics" / f"{ids['old']}.parquet")
    scores = np.load(old_dir / "scores.npy", allow_pickle=False)
    mask = np.load(old_dir / "seen_mask.npy", allow_pickle=False)
    panels = build_panels(activity, scores, mask, PANEL_SIZES)
    metrics_dir = paths.results_dir() / "metrics"
    record = {"analysis_version": ANALYSIS_VERSION, "case": case_id, "runs": ids, "faults": {},
              "aa": v1._detect(old_dir, repeat_dir, users, activity, old_ndcg, metrics_dir / f"{ids['old_repeat']}.parquet", panels, mask),
              "ab": v1._detect(old_dir, new_dir, users, activity, old_ndcg, metrics_dir / f"{ids['new']}.parquet", panels, mask)}
    new_scores = np.load(new_dir / "scores.npy", allow_pickle=False)
    item_ids = np.load(new_dir / "item_ids.npy", allow_pickle=False)
    native = _native(new_dir)
    root = paths.results_dir() / OUTPUT / case_id
    for fault in (CONTROL,) + FAULTS:
        if fault == CONTROL:
            faulted, lists = new_scores, reconstruction_control(new_scores, mask, item_ids)
            detail = {"users_changed_top10": int(np.any(lists[10][0] != native[10][0], axis=1).sum())}
        else:
            faulted, lists, detail = isolated_fault(fault, new_scores, mask, item_ids, native)
        target = root / fault
        v1._write_faulted(new_dir, target, faulted, lists)
        evaluated = evaluation.evaluate(target, dataset, "test", target / "metrics.parquet")
        record["faults"][fault] = {"detail": detail, "violations": evaluated["violations"],
                                   **v1._detect(old_dir, target, users, activity, old_ndcg, target / "metrics.parquet", panels, mask)}
        (target / "scores.npy").unlink()  # keep lists, metrics and contracts; drop the large score copy
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(_dump(record), indent=2) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------- serialisation

def _dump(value):
    """JSON form: tolerance keys become their repr strings."""
    if isinstance(value, dict):
        return {(repr(k) if isinstance(k, float) else k): _dump(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dump(v) for v in value]
    return value


_TOLERANCE_KEYS = {repr(delta): delta for delta in VERDICT_TOLERANCES}


def _load(value):
    """Inverse of _dump: a dict keyed only by tolerance strings gets its float keys back."""
    if isinstance(value, dict):
        if value and all(k in _TOLERANCE_KEYS for k in value):
            return {_TOLERANCE_KEYS[k]: _load(v) for k, v in value.items()}
        return {k: _load(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_load(v) for v in value]
    return value


def load_record(path: Path) -> dict:
    return _load(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- summary

def _pipeline(case_id: str) -> str:
    return case_id.split("-", 1)[1]


def summarize(records: list[dict]) -> dict:
    """Reference labels, detector recall and alarms; the F0 control is reported apart."""
    detectors = [f"{s}:{n}" for n in PANEL_SIZES for s in PANEL_STRATEGIES]
    baselines = ["exact_snapshot", "score_tolerance", "overlap_strict", "overlap_lenient"]
    n_users = 1007
    rows = []
    for delta in VERDICT_TOLERANCES:
        positives = [(r["case"], f, r["faults"][f]) for r in records for f in FAULTS if v1._alarm(r["faults"][f]["reference"], delta)]
        benign = [r for r in records if r["ab"]["reference"][delta] == "equivalent"]
        for name in detectors + baselines + ["full_check"]:
            def hit(entry):
                if name == "full_check":
                    return v1._alarm(entry["reference"], delta)
                if name in baselines:
                    return bool(entry["baselines"][name])
                return v1._alarm(entry["panels"][name], delta)
            size = int(name.split(":")[1]) if ":" in name else n_users
            per_fault = {fault: {"positives": sum(f == fault for _, f, _ in positives),
                                 "detected": sum(hit(e) for _, f, e in positives if f == fault)} for fault in FAULTS}
            per_pipeline = {}
            for case, _, entry in positives:
                slot = per_pipeline.setdefault(_pipeline(case), {"positives": 0, "detected": 0})
                slot["positives"] += 1
                slot["detected"] += int(hit(entry))
            detected = sum(hit(e) for _, _, e in positives)
            rows.append({
                "delta": delta, "detector": name, "paired_user_requests": 2 * size,
                "synthetic_positives": len(positives), "detected": detected,
                "recall": detected / len(positives) if positives else None,
                "per_fault": per_fault, "per_pipeline": dict(sorted(per_pipeline.items())),
                "aa_controls": len(records), "aa_alarms": sum(hit(r["aa"]) for r in records),
                "benign_ab_cases": len(benign), "benign_ab_alarms": sum(hit(r["ab"]) for r in benign),
                "control_alarms": sum(hit(r["faults"][CONTROL]) for r in records),
            })
    verdicts = ("contract_failure", "harmful", "equivalent", "improved", "inconclusive")
    labels = {fault: {repr(delta): {v: sum(r["faults"][fault]["reference"][delta] == v for r in records) for v in verdicts}
                      for delta in VERDICT_TOLERANCES} for fault in (CONTROL,) + FAULTS}
    footprint = {}
    for fault in (CONTROL,) + FAULTS:
        changed = [r["faults"][fault]["detail"]["users_changed_top10"] for r in records]
        losses = [r["faults"][fault]["full_loss_mean"] for r in records if r["faults"][fault]["full_loss_mean"] is not None]
        footprint[fault] = {"cases_with_changed_lists": sum(c > 0 for c in changed),
                            "median_users_changed": float(np.median(changed)), "max_users_changed": int(max(changed)),
                            "median_full_loss": float(np.median(losses)) if losses else None,
                            "max_full_loss": float(max(losses)) if losses else None,
                            "cases_with_loss": len(losses)}
    return {"analysis_version": ANALYSIS_VERSION, "n_base_cases": len(records),
            "primary": dict(zip(("strategy", "size", "delta"), v1.PRIMARY)),
            "reference_labels": labels, "footprint": footprint, "rows": rows}


def run_all(case_ids: list[str] | None = None, shard: tuple[int, int] | None = None) -> dict | None:
    """Run or resume the pinned cohort. With a shard (index, count) only that slice is run."""
    pinned = cohort()
    cases = case_ids or pinned
    unknown = sorted(set(cases) - set(pinned))
    if unknown:
        raise ValueError(f"not in the pinned cohort: {unknown}")
    if shard:
        cases = cases[shard[0]::shard[1]]
    root = paths.results_dir() / OUTPUT
    records = []
    for case_id in cases:
        existing = root / case_id / "result.json"
        records.append(load_record(existing) if existing.is_file() else _load(_dump(run_case(case_id))))
    if shard or case_ids:
        return None
    summary = summarize(records)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", help="index/count, for example 0/4")
    args = parser.parse_args()
    part = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    result = run_all(shard=part)
    print(json.dumps({"done": True, "shard": args.shard, "summary": bool(result)}))
