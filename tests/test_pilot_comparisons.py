import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from silentrank.orchestrator import comparisons, evaluation, ledger, pilot
from silentrank.stats.paired import ACTIVITY_CONFIDENCE_LEVEL, PILOT_SEEDS


@pytest.fixture
def comparison_runs(tmp_path, tmp_ledger, monkeypatch):
    monkeypatch.setenv("SILENTRANK_ROOT", str(tmp_path))
    results = tmp_path / "results"
    (results / "metrics").mkdir(parents=True)
    users = np.arange(3, dtype=np.int64)
    catalog = np.arange(100, 121, dtype=np.int64)

    def register(case, pipeline):
        config = {"pipeline": pipeline, "old": "old-env", "new": "new-env"}
        with ledger.writer() as con:
            con.execute("INSERT INTO cases (case_id, config_json, applicable) VALUES (?, ?, TRUE)",
                        [case, json.dumps(config)])

    def add(case, run, side, mode, *, seed=0, repeat=0, host="host", status="ok", split="test",
            source="frozen-model", initial=None, changed=False, structural=False):
        env_id = "new-env" if side == "new" else "old-env"
        inputs = {"case": case, "side": side, "mode": mode, "seed": seed, "split": split,
                  "env_id": env_id, "source_model": source if mode == "inference" else None,
                  "initial_model": initial, "dataset_sha256": "dataset", "worker_sha256": "worker",
                  "evaluator_version": "evaluator", "image": "image", "lock_sha256": env_id,
                  "neighbors": None, "factors": 16 if case.endswith("als") else None}
        directory = results / "runs" / run
        directory.mkdir(parents=True)
        meta = {"inputs": inputs, "error": "shape mismatch" if structural else None}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        stages = {"S0": "data", "S1": "mapping-changed" if changed and mode == "prep" else "mapping"}
        if mode != "prep" and status == "ok":
            canonical = np.tile(catalog[:20], (3, 1))
            native = canonical[:, :10].copy()
            if changed:
                native[:, :2] = native[:, :2][:, ::-1]
            scores = np.full((3, 21), 1.0 if changed else 0.0, dtype=np.float64)
            arrays = {"user_ids": users, "item_ids": catalog, "seen_mask": np.zeros_like(scores, dtype=bool),
                      "topk_native10": native, "topk_canonical": canonical, "scores": scores}
            for name, values in arrays.items():
                np.save(directory / f"{name}.npy", values, allow_pickle=False)
            metric = [0.25 if changed else 0.5] * len(users)
            table = pa.table({"user": users, "activity": users,
                              "ndcg10": metric, "recall10": metric, "ndcg20": metric})
            pq.write_table(table, results / "metrics" / f"{run}.parquet")
            stages["S7"] = evaluation.array_hash(native)["canonical_sha256"]
        if status == "contract_fail" and not structural:
            (directory / "contracts.json").write_text('[{"codes":["duplicate_items"]}]', encoding="utf-8")
        with ledger.writer() as con:
            con.execute(
                """INSERT INTO runs (run_id, case_id, side, mode, seed, repeat, status, host_id,
                                     artifact_dir, started_at, evaluator_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [run, case, side, mode, seed, repeat, status, host, str(directory), pilot._now(), "evaluator"],
            )
            con.executemany("INSERT INTO stage_hashes VALUES (?, ?, ?, NULL, NULL)",
                            [(run, name, digest) for name, digest in stages.items()])

    return register, add


def test_score_diagnostics_accept_implicit_missing_score_sentinel(comparison_runs, tmp_path):
    register, add = comparison_runs
    case = "N1-P2-cosine"
    register(case, "implicit_cosine")
    add(case, "baseline", "old", "inference")
    add(case, "control", "old_repeat", "inference")
    add(case, "upgrade", "new", "inference", changed=True)
    for name, value in (("baseline", 1.0), ("control", 1.0), ("upgrade", 2.0)):
        path = tmp_path / "results" / "runs" / name / "scores.npy"
        scores = np.full((3, 21), -np.finfo(np.float64).max)
        scores[:, 0] = value
        np.save(path, scores, allow_pickle=False)

    behavior = comparisons.verdict_case(case)["modes"]["inference"]["AB"]["behavior"]
    assert behavior["score_maxabs"] == 1.0
    assert behavior["score_rel_l2"] == pytest.approx(0.0)


def test_compare_matches_runs_and_requires_all_seed_pairs(comparison_runs):
    register, add = comparison_runs
    case = "N1-P2-als"
    register(case, "implicit_als")
    add(case, "baseline", "old", "inference")
    add(case, "control", "old_repeat", "inference")
    add(case, "upgrade", "new", "inference", changed=True)
    add(case, "different-host", "new", "inference", host="other", repeat=1)
    add(case, "different-model", "new", "inference", source="different", repeat=2)
    add(case, "validation", "old", "retrain", seed=PILOT_SEEDS[0], split="validation")
    for seed in PILOT_SEEDS:
        add(case, f"old{seed}", "old", "retrain", seed=seed, initial=f"factors{seed}")
        initial = "wrong" if seed == PILOT_SEEDS[-1] else f"factors{seed}"
        add(case, f"new{seed}", "new", "retrain", seed=seed, initial=initial, changed=True)

    incomplete = comparisons.compare_case(case)
    assert incomplete["modes"]["retrain"]["status"] == "incomplete"
    assert "validation" not in {attempt["run_id"] for attempt in incomplete["test_run_attempts"]}
    add(case, "new-last-retry", "new", "retrain", seed=PILOT_SEEDS[-1], repeat=1,
        initial=f"factors{PILOT_SEEDS[-1]}", changed=True)

    report = comparisons.verdict_case(case)
    inference = report["modes"]["inference"]
    retrain = report["modes"]["retrain"]
    assert inference["AA"]["runs"][0]["changed_stages"] == []
    assert inference["AA"]["quality"]["ndcg10"]["primary"]["estimate"] == 0
    assert inference["AB"]["runs"][0]["new"] == "upgrade"
    assert inference["AB"]["runs"][0]["crossing_ties"]["old"]["k10"] == {
        "users": 3, "tie_sizes": {21: 3},
    }
    assert inference["AB"]["behavior"] == {
        "overlap10_mean": 1.0, "users_changed_frac": 1.0, "canonical_equal": True,
        "score_maxabs": 1.0, "score_rel_l2": None,
    }
    assert retrain["AB"]["runs"][-1]["new"] == "new-last-retry"
    interval = retrain["AB"]["quality"]["ndcg10"]
    assert interval["primary"]["n_seeds"] == 5
    assert interval["sensitivity"]["estimate"] == 0.25
    assert interval["activity_groups"]["low"]["primary"]["level"] == ACTIVITY_CONFIDENCE_LEVEL
    assert {row["verdict"] for mode in (inference, retrain) for row in mode["verdicts"]} == {"harmful"}

    with ledger.reader() as con:
        assert con.execute("SELECT kind, n_seeds FROM comparisons ORDER BY kind, n_seeds").fetchall() == [
            ("AA", 1), ("AB", 1), ("AB", 5),
        ]
        assert con.execute("SELECT DISTINCT verdict FROM verdicts").fetchall() == [("harmful",)]


@pytest.mark.parametrize("structural", [False, True])
def test_contract_failure_has_precedence_over_quality(comparison_runs, structural):
    register, add = comparison_runs
    case = "N1-P1-float32"
    register(case, "ref_itemknn")
    add(case, "baseline", "old", "inference")
    add(case, "control", "old_repeat", "inference")
    add(case, "upgrade", "new", "inference", status="contract_fail", structural=structural)

    result = comparisons.verdict_case(case)["modes"]["inference"]

    assert result["AB"]["quality"] is None
    evidence = result["AB"]["runs"][0]["contracts"][0]
    assert evidence.endswith("meta.json" if structural else "contracts.json")
    assert {row["verdict"] for row in result["verdicts"]} == {"contract_failure"}


def test_changed_control_blocks_ab_comparison(comparison_runs):
    register, add = comparison_runs
    case = "N1-P1-float32"
    register(case, "ref_itemknn")
    add(case, "baseline", "old", "inference")
    add(case, "control", "old_repeat", "inference", changed=True)
    add(case, "upgrade", "new", "inference", changed=True)

    report = comparisons.compare_case(case)

    assert report["status"] == "control_failed"
    assert "AB" not in report["modes"]["inference"]


def test_preprocessing_comparison_uses_only_stage_hashes(comparison_runs):
    register, add = comparison_runs
    case = "D1-P0"
    register(case, "prep_pandas")
    add(case, "baseline", "old", "prep")
    add(case, "upgrade", "new", "prep", changed=True)

    report = comparisons.verdict_case(case)

    assert report["modes"]["prep"]["AB"]["runs"][0]["changed_stages"] == ["S1"]
    assert "quality" not in report["modes"]["prep"]["AB"]
    with ledger.reader() as con:
        assert con.execute("SELECT COUNT(*) FROM verdicts").fetchone() == (0,)
