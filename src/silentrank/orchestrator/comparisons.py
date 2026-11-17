"""Compare completed pilot runs and record protocol-v0 decisions."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from silentrank import paths
from silentrank.orchestrator import ledger, manifests, pilot
from silentrank.stats.paired import (
    ACTIVITY_CONFIDENCE_LEVEL,
    CONFIDENCE_LEVEL,
    PILOT_SEEDS,
    paired_seed_user_bootstrap,
    paired_user_bootstrap,
    seed_t_interval,
)
from silentrank.stats.verdict import VERDICT_TOLERANCES, classify_verdict

_METRICS = ("ndcg10", "recall10", "ndcg20")
_ACTIVITY = ("low", "medium", "high")
_VALID_OUTPUT = {"ok", "contract_fail"}


def _rule(case_id: str) -> str:
    """Development cases are judged under protocol v0, held-out cases under protocol v1."""
    transition = case_id.split("-", 1)[0]
    row = next(((entry["split"],) for entry in manifests.load_all() if entry["id"] == transition), None)
    return "protocol-v1" if row and row[0] == "heldout" else "protocol-v0"


def _load_test_runs(case_id: str, config: dict) -> list[dict]:
    runs = []
    for row in pilot._run_rows(case_id):
        if row["side"] not in {"old", "old_repeat", "new"}:
            continue
        directory = Path(row["artifact_dir"])
        inputs = pilot._read_json(directory / "meta.json")["inputs"]
        if inputs["split"] != "test":
            continue
        expected_env = config["new" if row["side"] == "new" else "old"]
        if inputs["env_id"] != expected_env:
            continue
        if any(inputs[key] != row[key] for key in ("mode", "side", "seed", "evaluator_version")):
            raise ValueError(f"run metadata disagrees with ledger for {row['run_id']}")
        if inputs["case"] != case_id:
            raise ValueError(f"run metadata has the wrong case for {row['run_id']}")
        if row["mode"] == "inference" and row["status"] in _VALID_OUTPUT and not inputs["source_model"]:
            raise ValueError(f"inference run {row['run_id']} has no frozen model identity")
        if (config["pipeline"] == "implicit_als" and row["mode"] == "retrain" and row["status"] in _VALID_OUTPUT
                and not inputs.get("initial_model")):
            raise ValueError(f"ALS run {row['run_id']} has no frozen initial-factor identity")
        runs.append({**row, "inputs": inputs, "directory": directory})
    return runs


def _cohort(run: dict) -> tuple:
    inputs = run["inputs"]
    return (run["host_id"], inputs["worker_sha256"], inputs["evaluator_version"],
            inputs["dataset_sha256"], inputs["image"], inputs.get("neighbors"), inputs.get("factors"),
            inputs.get("similarity"), inputs.get("lamb"))


def _latest_matching(runs: list[dict], mode: str, keys: list[tuple[str, int]]) -> list[dict] | None:
    cohorts = defaultdict(dict)
    last = {}
    for index, run in enumerate(runs):
        key = (run["side"], run["seed"])
        if run["mode"] != mode or key not in keys or not run["host_id"]:
            continue
        inputs = run["inputs"]
        signature = (*_cohort(run), inputs["source_model"], inputs.get("initial_model"))
        if key not in cohorts[signature] or run["repeat"] >= cohorts[signature][key]["repeat"]:
            cohorts[signature][key] = run
            last[signature] = index
    complete = [signature for signature, rows in cohorts.items() if all(key in rows for key in keys)]
    if not complete:
        return None
    selected = cohorts[max(complete, key=last.__getitem__)]
    return [selected[key] for key in keys]


def _changed_stages(old: dict, new: dict) -> list[str]:
    a = dict(pilot._stage_values(old["run_id"]))
    b = dict(pilot._stage_values(new["run_id"]))
    if not a or not b:
        raise ValueError("completed runs must have stage hashes")
    return sorted(stage for stage in a.keys() | b.keys() if a.get(stage) != b.get(stage))


def _npy(run: dict, name: str) -> np.ndarray:
    return np.load(run["directory"] / f"{name}.npy", mmap_mode="r", allow_pickle=False)


def _metrics(run: dict) -> dict[str, np.ndarray]:
    path = paths.results_dir() / "metrics" / f"{run['run_id']}.parquet"
    table = pq.read_table(path, columns=["user", "activity", *_METRICS])
    values = {name: table[name].to_numpy() for name in table.column_names}
    if not np.array_equal(values["user"], _npy(run, "user_ids")):
        raise ValueError(f"metrics and rankings have different users in {run['run_id']}")
    return values


def _crossing_tie(scores: np.ndarray, k: int) -> int:
    if scores.size <= k:
        return 0
    cut = scores.size - k
    threshold = np.partition(scores, cut)[cut]
    equal = int(np.count_nonzero(scores == threshold))
    return equal if equal > 1 and np.count_nonzero(scores > threshold) + equal > k else 0


def _ranking_diagnostics(old: dict, new: dict) -> dict:
    users = _npy(old, "user_ids")
    mask = _npy(old, "seen_mask")
    for name, reference in (("user_ids", users), ("item_ids", _npy(old, "item_ids")), ("seen_mask", mask)):
        if not np.array_equal(reference, _npy(new, name)):
            raise ValueError(f"comparison requires identical {name}")

    old_ranks, new_ranks = (_npy(run, "topk_native10") for run in (old, new))
    old_scores, new_scores = (_npy(run, "scores") for run in (old, new))
    canonical_equal = np.array_equal(_npy(old, "topk_canonical"), _npy(new, "topk_canonical"))
    score_maxabs, score_scale, baseline_ss, diff_ss = 0.0, 0.0, 0.0, 0.0
    overlap_total, shared_items, moved_items, positions_moved = 0.0, 0, 0, 0
    ties = {side: {k: Counter() for k in (10, 20)} for side in ("old", "new")}
    for row, excluded in enumerate(mask):
        eligible = ~excluded
        a = {int(item): rank for rank, item in enumerate(old_ranks[row]) if item != -1}
        b = {int(item): rank for rank, item in enumerate(new_ranks[row]) if item != -1}
        common = a.keys() & b.keys()
        denominator = min(10, int(np.count_nonzero(eligible)))
        if denominator == 0:
            raise ValueError("an evaluation user has no eligible candidates")
        overlap_total += len(common) / denominator
        shared_items += len(common)
        shifts = [abs(a[item] - b[item]) for item in common]
        moved_items += sum(shift > 0 for shift in shifts)
        positions_moved += sum(shifts)

        a_scores = old_scores[row, eligible].astype(np.float64, copy=False)
        b_scores = new_scores[row, eligible].astype(np.float64, copy=False)
        if not np.isfinite(a_scores).all() or not np.isfinite(b_scores).all():
            raise ValueError("validated rankings contain nonfinite eligible scores")
        with np.errstate(over="ignore"):
            difference = a_scores - b_scores
        score_maxabs = max(score_maxabs, float(np.max(np.abs(difference))))
        scale = max(score_scale, float(np.max(np.abs(a_scores))), float(np.max(np.abs(b_scores))))
        if scale:
            if scale > score_scale:
                ratio = score_scale / scale
                baseline_ss *= ratio * ratio
                diff_ss *= ratio * ratio
                score_scale = scale
            a_scaled, b_scaled = a_scores / score_scale, b_scores / score_scale
            finite_difference = np.isfinite(difference)
            delta = a_scaled - b_scaled
            delta[finite_difference] = difference[finite_difference] / score_scale
            baseline_ss += float(a_scaled @ a_scaled)
            diff_ss += float(delta @ delta)
        for side, scores in (("old", a_scores), ("new", b_scores)):
            for k in (10, 20):
                size = _crossing_tie(scores, k)
                if size:
                    ties[side][k][size] += 1

    count = len(users)
    if not count:
        raise ValueError("completed rankings contain no evaluation users")
    return {
        "n_users": count,
        "overlap10_mean": overlap_total / count,
        "users_changed_frac": float(np.count_nonzero(np.any(old_ranks != new_ranks, axis=1)) / count),
        "shared_items10": shared_items,
        "shared_items_changed_position10": moved_items,
        "shared_items_total_absolute_shift10": positions_moved,
        "canonical_equal": bool(canonical_equal),
        "score_maxabs": score_maxabs if np.isfinite(score_maxabs) else None,
        "score_norm_scale": score_scale,
        "score_baseline_sum_squares": baseline_ss,
        "score_difference_sum_squares": diff_ss,
        "crossing_ties": {
            side: {f"k{k}": {"users": sum(counts.values()), "tie_sizes": dict(sorted(counts.items()))}
                   for k, counts in by_k.items()}
            for side, by_k in ties.items()
        },
    }


def _relative_norm(baseline_ss: float, diff_ss: float) -> float | None:
    if baseline_ss == 0:
        return 0.0 if diff_ss == 0 else None
    relative = diff_ss ** 0.5 / baseline_ss ** 0.5
    return relative if np.isfinite(relative) else None


def _behavior(pairs: list[dict]) -> dict:
    scale = max(pair["score_norm_scale"] for pair in pairs)
    weights = [(pair["score_norm_scale"] / scale) ** 2 if scale else 0.0 for pair in pairs]
    baseline_ss = sum(weight * pair["score_baseline_sum_squares"] for weight, pair in zip(weights, pairs))
    difference_ss = sum(weight * pair["score_difference_sum_squares"] for weight, pair in zip(weights, pairs))
    maximums = [pair["score_maxabs"] for pair in pairs]
    return {
        "overlap10_mean": float(np.mean([pair["overlap10_mean"] for pair in pairs])),
        "users_changed_frac": float(np.mean([pair["users_changed_frac"] for pair in pairs])),
        "canonical_equal": all(pair["canonical_equal"] for pair in pairs),
        "score_maxabs": max(maximums) if all(value is not None for value in maximums) else None,
        "score_rel_l2": _relative_norm(baseline_ss, difference_ss),
    }


def _intervals(losses: np.ndarray, mode: str, level: float) -> dict:
    if mode == "inference":
        return {"primary": asdict(paired_user_bootstrap(losses[0], level=level))}
    return {"primary": asdict(seed_t_interval(losses, level=level)),
            "sensitivity": asdict(paired_seed_user_bootstrap(losses, level=level))}


def _quality(pairs: list[tuple[dict, dict]], mode: str) -> dict:
    aligned = [(_metrics(old), _metrics(new)) for old, new in pairs]
    reference = aligned[0][0]
    for pair in aligned:
        for metrics in pair:
            if any(not np.array_equal(metrics[key], reference[key]) for key in ("user", "activity")):
                raise ValueError("paired metrics must have identical user order and activity groups")
    losses = {key: np.stack([a[key] - b[key] for a, b in aligned]) for key in _METRICS}
    quality = {key: _intervals(values, mode, CONFIDENCE_LEVEL) for key, values in losses.items()}
    groups = {}
    for number, group in enumerate(_ACTIVITY):
        selected = reference["activity"] == number
        count = int(np.count_nonzero(selected))
        groups[group] = {"n_users": count}
        if count:
            groups[group].update(_intervals(losses["ndcg10"][:, selected], mode, ACTIVITY_CONFIDENCE_LEVEL))
    quality["ndcg10"]["activity_groups"] = groups
    return quality


def _comparison(case_id: str, pairs: list[tuple[dict, dict]], mode: str, kind: str) -> dict:
    runs = [{"seed": old["seed"], "old": old["run_id"], "new": new["run_id"],
             "old_status": old["status"], "new_status": new["status"]} for old, new in pairs]
    result = {"mode": mode, "kind": kind, "runs": runs, "status": "ok"}
    if any(run["status"] not in _VALID_OUTPUT for pair in pairs for run in pair):
        result["status"] = "run_failed"
        return result

    result["comparison_id"] = pilot._token({"case": case_id, "mode": mode, "kind": kind, "runs": runs, "rule": _rule(case_id)})
    for details, (old, new) in zip(runs, pairs):
        details["changed_stages"] = _changed_stages(old, new)
    if mode == "prep":
        return result

    result["contract_failure"] = any(run["status"] == "contract_fail" for pair in pairs for run in pair)
    reference = next((run for pair in pairs for run in pair if run["status"] == "ok"), None)
    result["n_users"] = len(_npy(reference, "user_ids")) if reference else None
    result["quality"] = None
    for details, (old, new) in zip(runs, pairs):
        failing_runs = [run for run in (old, new) if run["status"] == "contract_fail"]
        if failing_runs:
            evidence = []
            for run in failing_runs:
                path = run["directory"] / "contracts.json"
                evidence.append(str(path if path.is_file() else run["directory"] / "meta.json"))
            details["contracts"] = evidence
        else:
            details.update(_ranking_diagnostics(old, new))
    if not result["contract_failure"]:
        result["behavior"] = _behavior(runs)
        result["quality"] = _quality(pairs, mode)
    return result


def _retraining_pairs(runs: list[dict], old: dict, new: dict) -> list[tuple[dict, dict]]:
    locks = {"old": old["inputs"]["lock_sha256"], "new": new["inputs"]["lock_sha256"]}
    comparable = [
        run for run in runs
        if run["mode"] == "retrain" and _cohort(run) == _cohort(old)
        and run["inputs"]["lock_sha256"] == locks.get(run["side"])
    ]
    pairs = []
    for seed in PILOT_SEEDS:
        pair = _latest_matching(comparable, "retrain", [("old", seed), ("new", seed)])
        if pair is None:
            return []
        pairs.append((pair[0], pair[1]))
    return pairs


def _mode_verdicts(result: dict) -> list[dict]:
    if result["status"] != "ok" or result["mode"] == "prep":
        return []
    interval = result["quality"]["ndcg10"]["primary"] if result["quality"] else {}
    return [
        {"delta": delta, "verdict": classify_verdict(interval.get("low"), interval.get("high"), delta,
                                                   contract_failure=result["contract_failure"])}
        for delta in VERDICT_TOLERANCES
    ]


def _store_comparisons(case_id: str, results: list[dict]) -> None:
    with ledger.writer() as con:
        con.execute("BEGIN TRANSACTION")
        for result in results:
            if "comparison_id" not in result:
                continue
            runs = result["runs"]
            behavior = result.get("behavior", {})
            quality = result.get("quality")
            interval = quality["ndcg10"]["primary"] if quality else {}
            # For a seed aggregate, the JSON contains all run pairs and the ledger references the first.
            con.execute(
                """INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (comparison_id) DO UPDATE SET score_maxabs = excluded.score_maxabs,
                       score_rel_l2 = excluded.score_rel_l2""",
                [result["comparison_id"], case_id, runs[0]["old"], runs[0]["new"], result["kind"],
                 result.get("n_users"), len(runs), behavior.get("overlap10_mean"), behavior.get("users_changed_frac"),
                 behavior.get("canonical_equal"), behavior.get("score_maxabs"), behavior.get("score_rel_l2"),
                 interval.get("estimate"), interval.get("low"), interval.get("high"), interval.get("method"),
                 pilot._now()],
            )
        con.execute("COMMIT")


def _save_report(case_id: str, report: dict) -> None:
    directory = paths.results_dir() / "compare"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{case_id}.json"
    report["artifact"] = str(target)
    with TemporaryDirectory(dir=directory) as work:
        staged = Path(work) / target.name
        staged.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.replace(target)


def compare_case(case_id: str) -> dict:
    """Record comparisons for the latest comparable test runs of the case."""
    if not re.fullmatch(r"[\w-]+", case_id, flags=re.ASCII):
        raise ValueError("invalid pilot case ID")
    with ledger.reader() as con:
        row = con.execute("SELECT config_json, applicable FROM cases WHERE case_id = ?", [case_id]).fetchone()
        if row is None:
            raise ValueError(f"no runs registered for {case_id}; run the case first")
        config, applicable = json.loads(row[0]), row[1]
        suite_rows = con.execute(
            "SELECT env_id, package, status, n_pass, n_fail, log_path FROM testsuite "
            "WHERE env_id IN (?, ?) ORDER BY env_id, package", [config["old"], config["new"]],
        ).fetchall()
    runs = _load_test_runs(case_id, config)
    report = {"case": case_id, "status": "incomplete", "modes": {}, "rule_version": _rule(case_id)}
    report["upstream_tests"] = [
        dict(zip(("env_id", "package", "status", "n_pass", "n_fail", "log_path"), row))
        for row in suite_rows
    ]
    report["test_run_attempts"] = [
        {key: run[key] for key in ("run_id", "side", "mode", "seed", "repeat", "status")} for run in runs
    ]
    modes, comparisons = report["modes"], []
    if applicable is False:
        report["status"] = "not_applicable"
    elif config["pipeline"] == "prep_pandas":
        pair = _latest_matching(runs, "prep", [("old", 0), ("new", 0)])
        if pair:
            ab = _comparison(case_id, [tuple(pair)], "prep", "AB")
            modes["prep"] = {"AB": ab, "status": ab["status"]}
            report["status"] = ab["status"]
            comparisons.append(ab)
        else:
            modes["prep"] = {"status": "incomplete"}
    else:
        cohort = _latest_matching(runs, "inference", [("old", 0), ("old_repeat", 0), ("new", 0)])
        if cohort:
            old, duplicate, new = cohort
            aa = _comparison(case_id, [(old, duplicate)], "inference", "AA")
            comparisons.append(aa)
            modes["inference"] = {"AA": aa, "status": "control_failed"}
            try:
                if old["inputs"]["lock_sha256"] != duplicate["inputs"]["lock_sha256"]:
                    raise ValueError("A/A control must use the same environment lock")
                pilot._require_clean_control(old, duplicate)
                if aa["quality"]["ndcg10"]["primary"]["estimate"] != 0:
                    raise ValueError("A/A control has nonzero NDCG loss")
            except ValueError as exc:
                report.update(status="control_failed", reason=str(exc))
            else:
                ab = _comparison(case_id, [(old, new)], "inference", "AB")
                modes["inference"].update(AB=ab, status=ab["status"], verdicts=_mode_verdicts(ab))
                comparisons.append(ab)
                report["status"] = ab["status"]
                if ab["status"] == "ok":
                    pairs = _retraining_pairs(runs, old, new)
                    if pairs:
                        rb = _comparison(case_id, pairs, "retrain", "AB")
                        modes["retrain"] = {"AB": rb, "status": rb["status"], "verdicts": _mode_verdicts(rb)}
                        comparisons.append(rb)
                    else:
                        status = "incomplete" if any(run["mode"] == "retrain" for run in runs) else "not_run"
                        modes["retrain"] = {"status": status, "required_seeds": list(PILOT_SEEDS)}
        else:
            modes["inference"] = {"status": "incomplete"}

    _store_comparisons(case_id, comparisons)
    _save_report(case_id, report)
    return report


def verdict_case(case_id: str) -> dict:
    """Refresh mode-specific decisions and record inference verdicts in the ledger."""
    report = compare_case(case_id)
    verdicts = report["modes"].get("inference", {}).get("verdicts", [])
    if verdicts:
        with ledger.writer() as con:
            con.executemany(
                """INSERT INTO verdicts VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (case_id, delta) DO UPDATE SET verdict = excluded.verdict,
                     rule_version = excluded.rule_version, decided_at = excluded.decided_at""",
                [[case_id, row["delta"], row["verdict"], _rule(case_id), pilot._now()] for row in verdicts],
            )
    return report
