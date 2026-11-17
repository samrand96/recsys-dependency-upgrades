"""Run development-only probes against saved inference controls and full audits."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from silentrank import paths
from silentrank.orchestrator import ledger
from silentrank.stats.paired import ANALYSIS_SEED, paired_user_bootstrap
from silentrank.stats.verdict import VERDICT_TOLERANCES, classify_verdict

from .panels import PANEL_SIZES, PANEL_STRATEGIES, build_panels


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(run_id: str) -> dict:
    table = pq.read_table(paths.results_dir() / "metrics" / f"{run_id}.parquet",
                          columns=["user", "activity", "ndcg10"])
    return {key: table[key].to_numpy() for key in table.column_names}


def _run_records(case_id: str) -> tuple[str, dict[str, dict]]:
    with ledger.reader() as con:
        case = con.execute(
            "SELECT t.collection FROM cases c JOIN transitions t ON t.transition_id = c.transition_id "
            "WHERE c.case_id = ?", [case_id],
        ).fetchone()
        cursor = con.execute(
            """SELECT run_id, artifact_dir, status, wall_s, cpu_s, peak_rss_mb,
                      (SELECT max(notes) FROM costs WHERE costs.run_id = runs.run_id
                        AND phase = 'inference:test') AS timing_notes
               FROM runs WHERE case_id = ?""",
            [case_id],
        )
        names = [column[0] for column in cursor.description]
        runs = {row[0]: dict(zip(names, row)) for row in cursor.fetchall()}
    if case is None:
        raise ValueError(f"probe comparison {case_id} has no linked transition collection")
    return case[0], runs


def _load_pair(comparison: dict, baseline: dict, runs: dict) -> dict:
    pair, = comparison["runs"]
    alternative = runs[pair["new"]]
    if alternative["status"] == "contract_fail":
        contract_path = Path(alternative["artifact_dir"]) / "contracts.json"
        if contract_path.is_file():
            violations = _read_json(contract_path)
            users = [violation.get("user") for violation in violations]
            evidence_path = contract_path
        else:
            users = []
            evidence_path = contract_path.with_name("meta.json")
        user_ids = [user for user in users if user is not None]
        affected = np.isin(baseline["user"], user_ids)
        unlocalized = not users or any(user is None for user in users) or not np.isin(user_ids, baseline["user"]).all()
        return {"comparison": comparison, "run": alternative, "violations": affected, "loss": None,
                "contract_evidence": str(evidence_path), "unlocalized_contract": unlocalized}
    candidate = _metrics(pair["new"])
    if any(not np.array_equal(baseline[key], candidate[key]) for key in ("user", "activity")):
        raise ValueError("probe comparison requires matching user and activity arrays")
    return {"comparison": comparison, "run": alternative, "violations": None,
            "loss": baseline["ndcg10"] - candidate["ndcg10"]}


def _probe_outcome(pair: dict, positions: np.ndarray) -> dict:
    violations = pair["violations"]
    interval = paired_user_bootstrap(pair["loss"][positions]) if violations is None else None
    contract_failure = bool(violations[positions].any()) if violations is not None else False
    result = {
        "comparison_id": pair["comparison"]["comparison_id"],
        "interval": asdict(interval) if interval else None,
        "verdicts": [
            {"delta": delta, "verdict": classify_verdict(interval.low if interval else None,
                                                        interval.high if interval else None, delta,
                                                        contract_failure=contract_failure)}
            for delta in VERDICT_TOLERANCES
        ],
    }
    if violations is not None:
        observation = "observed" if contract_failure else "not_observed"
        if not contract_failure and pair["unlocalized_contract"]:
            observation = "unlocalized_review"
        result.update(contract_observation=observation, contract_evidence=pair["contract_evidence"])
    return result


def _reference(inference: dict) -> dict:
    ab = inference["AB"]
    behavior = ab.get("behavior", {})
    return {
        "comparison_id": ab.get("comparison_id"),
        "verdicts": inference.get("verdicts", []),
        "smoke_failure": ab["status"] == "run_failed",
        "exact_native_snapshot_changed": (
            "S7" in ab["runs"][0]["changed_stages"] if ab["status"] == "ok" else None
        ),
        "overlap10_mean": behavior.get("overlap10_mean"),
        "score_maxabs": behavior.get("score_maxabs"),
        "score_rel_l2": behavior.get("score_rel_l2"),
    }


def _case_report(case_id: str, path: Path) -> dict:
    comparison = _read_json(path)
    inference = comparison["modes"].get("inference", {})
    report = {"case": case_id, "comparison_artifact": str(path)}
    if not inference.get("AB") or inference["AA"]["status"] != "ok":
        return {**report, "status": "no_accepted_inference_control"}
    collection, runs = _run_records(case_id)
    if case_id.endswith(("-sensitivity", "-ml20m")):
        return {**report, "status": "separate_sensitivity_split"}
    baseline_id = inference["AA"]["runs"][0]["old"]
    if baseline_id != inference["AB"]["runs"][0]["old"]:
        raise ValueError(f"A/A and A/B must share one baseline for {case_id}")
    baseline = _metrics(baseline_id)
    old = runs[baseline_id]
    directory = Path(old["artifact_dir"])
    users = np.load(directory / "user_ids.npy", allow_pickle=False)
    if not np.array_equal(users, baseline["user"]):
        raise ValueError(f"baseline score rows and metric users differ for {case_id}")
    scores = np.load(directory / "scores.npy", mmap_mode="r", allow_pickle=False)
    seen = np.load(directory / "seen_mask.npy", mmap_mode="r", allow_pickle=False)

    start = time.perf_counter()
    panels = build_panels(baseline["activity"], scores, seen, PANEL_SIZES)
    construction_s = time.perf_counter() - start
    pairs = {"AA": _load_pair(inference["AA"], baseline, runs)}
    if inference["AB"]["status"] == "ok":
        pairs["AB"] = _load_pair(inference["AB"], baseline, runs)
    results = []
    for panel in panels:
        start = time.perf_counter()
        outcomes = {kind: _probe_outcome(pair, panel.positions) for kind, pair in pairs.items()}
        activity_counts = np.bincount(baseline["activity"][panel.positions], minlength=3)
        results.append({
            "strategy": panel.strategy,
            "size": len(panel.positions),
            "user_indices": users[panel.positions].tolist(),
            "activity_counts": dict(zip(("low", "medium", "high"), activity_counts.tolist())),
            "undefined_margins": panel.undefined_margins,
            "paired_user_score_requests_proxy": 2 * len(panel.positions),
            "analysis_wall_s": time.perf_counter() - start,
            "pairs": outcomes,
        })
    return {**report, "status": "analyzed", "collection": collection, "n_users": len(users),
            "baseline": old, "full_run_costs": {kind: pair["run"] for kind, pair in pairs.items()},
            "full_reference": _reference(inference), "panel_construction_wall_s": construction_s,
            "baseline_score_rows_scanned_for_margins": len(users),
            "unavailable_sizes": [size for size in PANEL_SIZES if size > len(users)], "panels": results}


def _verdict(verdicts: list[dict], delta: float) -> str | None:
    return next((entry["verdict"] for entry in verdicts if entry["delta"] == delta), None)


def _summary(cases: list[dict]) -> dict:
    collections = sorted({case["collection"] for case in cases})
    summary = {}
    for collection in collections:
        collection_cases = [case for case in cases if case["collection"] == collection]
        rows = []
        for size in PANEL_SIZES:
            for strategy in PANEL_STRATEGIES:
                panels = [(case, panel) for case in collection_cases for panel in case["panels"]
                          if panel["size"] == size and panel["strategy"] == strategy]
                ab_cases = sum("AB" in panel["pairs"] for _, panel in panels)
                for delta in VERDICT_TOLERANCES:
                    controls = [_verdict(panel["pairs"]["AA"]["verdicts"], delta) for _, panel in panels]
                    harmful = [panel for case, panel in panels
                               if _verdict(case["full_reference"]["verdicts"], delta) == "harmful"]
                    hits = sum(_verdict(panel["pairs"]["AB"]["verdicts"], delta) == "harmful" for panel in harmful)
                    alarms = sum(verdict in {"harmful", "contract_failure"} for verdict in controls)
                    rows.append({"size": size, "strategy": strategy, "delta": delta,
                                 "paired_user_score_requests_proxy": 2 * size, "ab_cases": ab_cases,
                                 "aa_controls": len(controls), "aa_false_alarms": alarms,
                                 "aa_false_alarm_rate": alarms / len(controls) if controls else None,
                                 "full_reference_supported_harms": len(harmful), "supported_harms_detected": hits,
                                 "harm_recall": hits / len(harmful) if harmful else None})
        summary[collection] = rows
    return summary


def analyze_probes(case_ids: list[str] | None = None) -> dict:
    """Analyze saved comparisons without running or retuning a recommender."""
    root = paths.results_dir()
    compare_dir = root / "compare"
    selected = sorted(set(case_ids)) if case_ids is not None else sorted(path.stem for path in compare_dir.glob("*.json"))
    cases = []
    for case_id in selected:
        if not re.fullmatch(r"[\w-]+", case_id, flags=re.ASCII):
            raise ValueError(f"invalid pilot case ID {case_id!r}")
        path = compare_dir / f"{case_id}.json"
        cases.append(_case_report(case_id, path) if path.is_file() else {"case": case_id, "status": "no_comparison"})

    analyzed = [case for case in cases if case["status"] == "analyzed"]
    directory = root / "probes"
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "pilot.json"
    report = {
        "status": "ok" if analyzed else "no_eligible_comparisons",
        "artifact": str(artifact), "split": "development", "mode": "inference",
        "selection": {"rng": "PCG64", "seed": ANALYSIS_SEED, "sizes": list(PANEL_SIZES),
                      "uniform": "sampling without replacement",
                      "activity": "proportional allocation by largest remainder across training activity groups",
                      "score_margin": "smallest normalized K=10 gap on baseline eligible scores, ties by fixed random priority"},
        "limits": [
            "The full labeled audit includes panel users. Intervals describe selected users and these cases are development evidence.",
            "A/A false alarms are conditional on accepted deterministic inference controls.",
            "Each panel request proxy counts one ranking request per user per side. The saved full runs were scored before this analysis.",
            "The score-margin policy scans full baseline scores. Panel-only runtime and realized savings were not measured.",
            "Run costs here cover saved inference runs. Environment builds, training, and validation tuning are outside this comparison.",
            "Timing notes record concurrent activity where logged. Contended runs provide elapsed observations with limited throughput comparability.",
            "A contract alert needs a violation linked to a selected user. Structural or unlocalized failures require review.",
            "Overlap and score tolerance are reported as continuous diagnostics. No detection threshold was frozen for those baselines.",
        ],
        "cases": cases, "summary_by_collection": _summary(analyzed),
    }
    with TemporaryDirectory(dir=directory) as work:
        staged = Path(work) / artifact.name
        staged.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.replace(artifact)
    return report
