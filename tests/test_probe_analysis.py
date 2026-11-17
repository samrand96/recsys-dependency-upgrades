import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from silentrank.detect.analysis import analyze_probes
from silentrank.detect.panels import build_panels, normalized_margins
from silentrank.orchestrator import ledger
from silentrank.stats.verdict import VERDICT_TOLERANCES


def test_score_margins_exclude_seen_items_and_normalize_by_score_range():
    scores = np.array([[5.0, 4.0, 1.0], [10.0, 8.0, 2.0], [3.0, 3.0, 3.0], [2.0, 1.0, 0.0]])
    seen = np.array([[True, False, False], [True, False, False], [False, False, False], [True, True, False]])

    np.testing.assert_array_equal(normalized_margins(scores, seen, k=1), [1.0, 1.0, 0.0, np.inf])


def test_margin_is_finite_when_score_range_exceeds_float64():
    extreme = np.finfo(np.float64).max
    scores = np.array([[extreme, 0.0, -extreme]])
    np.testing.assert_array_equal(normalized_margins(scores, np.zeros_like(scores, dtype=bool), k=1), [0.5])


def test_probe_selection_is_fixed_and_activity_allocation_is_proportional():
    activity = np.repeat([0, 1, 2], [223, 277, 507])
    scores = np.tile(np.arange(12, dtype=np.float64), (len(activity), 1))
    scores[42, 1:3] = 1.0  # Ties across the K=10 boundary.
    seen = np.zeros_like(scores, dtype=bool)

    panels = build_panels(activity, scores, seen)
    repeated = build_panels(activity, scores, seen)
    assert len(panels) == len(repeated) == 9
    for first, second in zip(panels, repeated):
        assert first.strategy == second.strategy
        np.testing.assert_array_equal(first.positions, second.positions)
        assert len(np.unique(first.positions)) == len(first.positions)
    stratified = next(panel for panel in panels if panel.strategy == "activity" and len(panel.positions) == 100)
    assert np.bincount(activity[stratified.positions]).tolist() == [22, 28, 50]
    for panel in panels:
        if panel.strategy == "score_margin":
            assert 42 in panel.positions


@pytest.mark.parametrize(
    "upgraded_ndcg, verdict, expected_recall, contract_style",
    [(0.5, "equivalent", None, None), (0.4, "harmful", 1.0, None),
     (None, "contract_failure", None, "global"), (None, "contract_failure", None, "historical"),
     (None, "contract_failure", None, "localized")],
)
def test_probe_analysis_links_audit_runs_and_reports_recall_only_for_supported_harm(
    tmp_path, tmp_ledger, monkeypatch, upgraded_ndcg, verdict, expected_recall, contract_style,
):
    monkeypatch.setenv("SILENTRANK_ROOT", str(tmp_path))
    case_id = "N1-P1-float32"
    results = tmp_path / "results"
    (results / "compare").mkdir(parents=True)
    (results / "metrics").mkdir()
    users = np.arange(100, dtype=np.int64)
    activity = np.repeat([0, 1, 2], [25, 25, 50])
    with ledger.writer() as con:
        con.execute("INSERT INTO transitions (transition_id, collection) VALUES ('N1', 'systematic_single')")
        con.execute("INSERT INTO cases (case_id, transition_id) VALUES (?, 'N1')", [case_id])
        for run in ("baseline", "duplicate", "upgraded"):
            artifact = results / "runs" / run
            artifact.mkdir(parents=True)
            ndcg = upgraded_ndcg if run == "upgraded" else 0.5
            if ndcg is None:
                status = "contract_fail"
                if contract_style == "historical":
                    (artifact / "meta.json").write_text('{"error":"shape mismatch"}', encoding="utf-8")
                else:
                    user = 0 if contract_style == "localized" else None
                    (artifact / "contracts.json").write_text(json.dumps([{"user": user}]), encoding="utf-8")
            else:
                status = "ok"
                table = pa.table({"user": users, "activity": activity, "ndcg10": np.full(100, ndcg)})
                pq.write_table(table, results / "metrics" / f"{run}.parquet")
            con.execute("INSERT INTO runs (run_id, case_id, artifact_dir, status, wall_s) VALUES (?, ?, ?, ?, 1.0)",
                        [run, case_id, str(artifact), status])
            con.execute("INSERT INTO costs VALUES (?, 'inference:test', 1.0, NULL, 'Concurrent screening')", [run])
    baseline = results / "runs" / "baseline"
    scores = np.ones((100, 11), dtype=np.float64)
    np.save(baseline / "user_ids.npy", users)
    np.save(baseline / "scores.npy", scores)
    np.save(baseline / "seen_mask.npy", np.zeros_like(scores, dtype=bool))
    comparisons = {}
    for kind, alternative in (("AA", "duplicate"), ("AB", "upgraded")):
        comparisons[kind] = {
            "status": "ok", "comparison_id": kind,
            "runs": [{"old": "baseline", "new": alternative, "changed_stages": []}],
        }
    comparisons["verdicts"] = [{"delta": delta, "verdict": verdict} for delta in VERDICT_TOLERANCES]
    comparison_path = results / "compare" / f"{case_id}.json"
    comparison_path.write_text(json.dumps({"modes": {"inference": comparisons}}), encoding="utf-8")

    report = analyze_probes()

    assert report["status"] == "ok"
    assert Path(report["artifact"]).is_file()
    case, = report["cases"]
    assert case["unavailable_sizes"] == [300, 1000]
    assert case["full_reference"]["comparison_id"] == "AB"
    assert case["baseline"]["timing_notes"] == "Concurrent screening"
    assert case["full_run_costs"]["AB"]["timing_notes"] == "Concurrent screening"
    assert all(panel["pairs"]["AA"]["comparison_id"] == "AA" for panel in case["panels"])
    assert all(panel["paired_user_score_requests_proxy"] == 200 for panel in case["panels"])
    if contract_style:
        observation = "observed" if contract_style == "localized" else "unlocalized_review"
        panel_verdict = "contract_failure" if contract_style == "localized" else "inconclusive"
        for panel in case["panels"]:
            outcome = panel["pairs"]["AB"]
            assert outcome["contract_observation"] == observation
            assert outcome["contract_evidence"].endswith("meta.json" if contract_style == "historical" else "contracts.json")
            assert {entry["verdict"] for entry in outcome["verdicts"]} == {panel_verdict}
    for row in report["summary_by_collection"]["systematic_single"]:
        if row["size"] == 100:
            assert row["full_reference_supported_harms"] == int(verdict == "harmful")
            assert row["harm_recall"] == expected_recall
        else:
            assert row["full_reference_supported_harms"] == 0
            assert row["harm_recall"] is None
        assert row["aa_false_alarm_rate"] == (0 if row["size"] == 100 else None)
