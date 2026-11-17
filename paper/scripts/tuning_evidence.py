"""Recover tested settings, validation grids and baseline quality from saved runs."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from silentrank.orchestrator.comparisons import _intervals
from silentrank.stats.paired import ACTIVITY_CONFIDENCE_LEVEL, CONFIDENCE_LEVEL
from silentrank.stats.verdict import VERDICT_TOLERANCES, classify_verdict

METRICS = ("ndcg10", "recall10", "ndcg20")
GROUPS = ("low", "medium", "high")


def verify_quality(case, mode, pairs, comparison, verdicts):
    """Recompute saved quality summaries from aligned baseline and upgraded metrics."""
    first = pairs[0][0]
    for pair in pairs:
        for side in pair:
            if any(not np.array_equal(side[key], first[key]) for key in ("user", "activity")):
                raise ValueError(f"Misaligned evaluation users or activity groups: {case}/{mode}")

    losses = {metric: np.stack([old[metric] - new[metric] for old, new in pairs])
              for metric in METRICS}
    quality = {metric: _intervals(values, mode, CONFIDENCE_LEVEL) for metric, values in losses.items()}
    groups = {}
    for number, group in enumerate(GROUPS):
        mask = first["activity"] == number
        count = int(mask.sum())
        groups[group] = {"n_users": count}
        if count:
            groups[group].update(_intervals(losses["ndcg10"][:, mask], mode, ACTIVITY_CONFIDENCE_LEVEL))
    quality["ndcg10"]["activity_groups"] = groups

    def check(actual, expected, path):
        if isinstance(actual, dict):
            if actual.keys() == expected.keys():
                for key, value in actual.items():
                    check(value, expected[key], f"{path}.{key}")
                return
        elif isinstance(actual, float):
            if np.isclose(actual, expected, rtol=1e-12, atol=1e-15):
                return
        elif actual == expected:
            return
        raise ValueError(f"Saved summary differs from recomputed metrics: {case}/{mode}/{path}")

    check(quality, comparison["quality"], "quality")
    interval = quality["ndcg10"]["primary"]
    calculated = [{"delta": delta, "verdict": classify_verdict(interval["low"], interval["high"], delta)}
                  for delta in VERDICT_TOLERANCES]
    if verdicts != calculated:
        raise ValueError(f"Saved verdicts differ from recomputed metrics: {case}/{mode}")


def audit(root: Path, con, primary: dict, reports: dict, configs: dict) -> dict:
    inputs = {}

    def read(path, reader):
        inputs[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return reader(path)

    def metadata(run_id):
        return read(root / "results" / "runs" / run_id / "meta.json",
                    lambda path: json.loads(path.read_text(encoding="utf-8")))["inputs"]

    def mean_ndcg(run_id):
        return read(root / "results" / "metrics" / f"{run_id}.parquet",
                    lambda path: float(pq.read_table(path, columns=["ndcg10"])["ndcg10"].to_numpy().mean()))

    def test_metrics(run_id):
        def columns(path):
            table = pq.read_table(path, columns=["user", "activity", *METRICS])
            return {name: table[name].to_numpy() for name in table.column_names}

        return read(root / "results" / "metrics" / f"{run_id}.parquet", columns)

    by_case = defaultdict(list)
    for case, run_id in con.execute("""SELECT case_id, run_id FROM runs
        WHERE mode = 'retrain' AND status = 'ok' AND side = 'old'
        ORDER BY case_id, run_id""").fetchall():
        by_case[case].append(run_id)

    cases = {}
    for case, outcome in sorted(primary.items()):
        cfg = configs[case]
        parameter = next((p for p in ("factors", "similarity", "lamb", "neighbors")
                          if cfg.get(f"{p}_grid")), None)
        if parameter is None:
            continue
        grid = cfg[f"{parameter}_grid"]
        tested = {}
        quality = {}
        for mode, report in reports[case].get("modes", {}).items():
            ab = report.get("AB", {})
            if ab.get("status") != "ok":
                continue
            pairs = ab.get("runs", [])
            aligned = []
            for pair in pairs:
                for side in ("old", "new"):
                    run_id = pair[side]
                    meta = metadata(run_id)
                    if (meta["split"], meta["case"], meta["mode"], meta["side"], meta["seed"]) != (
                        "test", case, mode, side, pair["seed"]
                    ):
                        raise ValueError(f"Test metadata mismatch: {case}, {run_id}")
                    tested[run_id] = meta[parameter]
                aligned.append((test_metrics(pair["old"]), test_metrics(pair["new"])))
            if aligned:
                verify_quality(case, mode, aligned, ab, report["verdicts"])
                means = [float(old["ndcg10"].mean()) for old, _ in aligned]
                quality[mode] = {
                    "baseline_ndcg10": sum(means) / len(means),
                    "baseline_runs": [p["old"] for p in pairs],
                    "quality_intervals_and_verdicts_verified": True,
                }

        trials = {value: [] for value in grid}
        for run_id in by_case[case]:
            meta = metadata(run_id)
            if meta["split"] == "validation" and meta["seed"] == 17 and meta["env_id"] == cfg["old"]:
                trials[meta[parameter]].append({"run_id": run_id, "mean_ndcg10": mean_ndcg(run_id)})
        scores = {value: {trial["mean_ndcg10"] for trial in entries} for value, entries in trials.items()}
        complete = all(len(values) == 1 for values in scores.values())
        winner = max(grid, key=lambda value: next(iter(scores[value]))) if complete else None
        actual = set(tested.values())
        status = ("conflicting_test_settings" if len(actual) > 1 else
                  "no_successful_comparison" if not actual else
                  "incomplete_or_conflicting_grid" if not complete else
                  "winner_mismatch" if actual != {winner} else "verified")
        cases[case] = {
            "pipeline": outcome["pipeline"], "baseline_environment": cfg["old"],
            "parameter": parameter, "validation_seed": 17,
            "validation": [{"value": v, "trials": trials[v]} for v in grid],
            "validation_winner": winner, "tested_runs": tested,
            "selected": next(iter(actual)) if len(actual) == 1 else None,
            "quality": quality, "status": status,
        }
    return {"verified_quality_comparisons": sum(len(case["quality"]) for case in cases.values()),
            "cases": cases, "inputs_sha256": inputs}
