"""The four declared synthetic faults change exactly what protocol/synthetic_controls.md says."""

import json

import numpy as np

from silentrank.detect import synthetic


def _fixture(users=200, items=400, seed=3):
    rng = np.random.default_rng(seed)
    scores = rng.random((users, items))
    mask = rng.random((users, items)) < 0.1
    item_ids = np.arange(1000, 1000 + items, dtype=np.int64)
    return scores, mask, item_ids


def test_f1_rotates_one_percent_of_item_columns():
    scores, mask, item_ids = _fixture()
    faulted, lists, detail = synthetic.apply_fault("F1", scores, mask, item_ids)
    changed = np.flatnonzero(np.any(faulted != scores, axis=0))
    assert len(detail["swapped_item_ids"]) == 4 == len(changed)
    assert sorted(item_ids[changed].tolist()) == sorted(detail["swapped_item_ids"])
    assert not np.isin(lists[10][0], item_ids[np.flatnonzero(mask[0])])[0].any()


def test_f2_ignores_the_seen_mask():
    scores, mask, item_ids = _fixture()
    _, lists, _ = synthetic.apply_fault("F2", scores, mask, item_ids)
    seen_returned = sum(np.isin(lists[10][0][row], item_ids[mask[row]]).any() for row in range(len(scores)))
    assert seen_returned > 0
    top = lists[10][0][0]
    assert top[0] == item_ids[np.argmax(scores[0])]


def test_f3_rounds_scores_to_two_decimals():
    scores, mask, item_ids = _fixture()
    faulted, lists, _ = synthetic.apply_fault("F3", scores, mask, item_ids)
    assert np.array_equal(faulted, np.round(scores, 2))
    assert np.all(np.diff(lists[20][1], axis=1) <= 0)


def test_f4_reverses_one_percent_of_users_only():
    scores, mask, item_ids = _fixture()
    _, lists, detail = synthetic.apply_fault("F4", scores, mask, item_ids)
    _, normal, _ = synthetic.apply_fault("F2", scores, np.zeros_like(mask), item_ids)  # unmasked reference shape only
    reversed_rows = detail["reversed_rows"]
    assert len(reversed_rows) == 2
    for row in range(len(scores)):
        values = lists[10][1][row]
        assert np.all(np.diff(values) >= 0) if row in reversed_rows else np.all(np.diff(values) <= 0)


def test_fault_selection_is_fixed():
    scores, mask, item_ids = _fixture()
    assert synthetic.apply_fault("F1", scores, mask, item_ids)[2] == synthetic.apply_fault("F1", scores, mask, item_ids)[2]
    assert synthetic.apply_fault("F4", scores, mask, item_ids)[2] == synthetic.apply_fault("F4", scores, mask, item_ids)[2]


def test_summary_counts_only_reference_positives():
    decision = lambda verdict: {d: verdict for d in (0.002, 0.005, 0.01)}
    entry = lambda ref, panel, base: {"reference": decision(ref), "panels": {f"{s}:{n}": decision(panel)
                                      for n in (100, 300, 1000) for s in ("uniform", "activity", "score_margin")},
                                      "baselines": dict.fromkeys(("exact_snapshot", "score_tolerance", "overlap_strict",
                                                                  "overlap_lenient"), base)}
    record = {"case": "X", "aa": entry("equivalent", "equivalent", False), "ab": entry("equivalent", "equivalent", True),
              "faults": {"F1": entry("harmful", "harmful", True), "F2": entry("contract_failure", "equivalent", True),
                         "F3": entry("equivalent", "harmful", True), "F4": entry("inconclusive", "equivalent", False)}}
    summary = synthetic.summarize([record])
    row = next(r for r in summary["rows"] if r["detector"] == "activity:300" and r["delta"] == 0.005)
    assert row["synthetic_positives"] == 2 and row["detected"] == 1 and row["recall"] == 0.5
    snap = next(r for r in summary["rows"] if r["detector"] == "exact_snapshot" and r["delta"] == 0.005)
    assert snap["benign_ab_alarms"] == 1 and snap["aa_alarms"] == 0


# ---- analysis version 2: fault isolation, cohort and cached records

def _native_fixture():
    from silentrank.detect import synthetic_v2

    scores, mask, item_ids = _fixture()
    scores[:, :5] = 0.75  # exact ties, so a non-reference tie order is possible
    reference = synthetic._rebuild(scores, mask, item_ids)
    native = {k: (items.copy(), values.copy()) for k, (items, values) in reference.items()}
    for k in (10, 20):  # a "native" adapter that orders some rows differently from the reference ranker
        native[k][0][::7] = native[k][0][::7, ::-1]
        native[k][1][::7] = native[k][1][::7, ::-1]
    return synthetic_v2, scores, mask, item_ids, native


def test_v2_f4_changes_only_the_reversed_users():
    v2, scores, mask, item_ids, native = _native_fixture()
    _, lists, detail = v2.isolated_fault("F4", scores, mask, item_ids, native)
    rows = np.array(detail["reversed_rows"])
    untouched = np.setdiff1d(np.arange(len(scores)), rows)
    for k in (10, 20):
        assert np.array_equal(lists[k][0][untouched], native[k][0][untouched])
    assert detail["users_changed_top10"] <= len(rows) == 2


def test_v2_f1_and_f3_keep_users_whose_reference_ranking_is_unchanged():
    v2, scores, mask, item_ids, native = _native_fixture()
    for fault in ("F1", "F3"):
        faulted, lists, _ = v2.isolated_fault(fault, scores, mask, item_ids, native)
        rebuild = (lambda s: v2.faithful_rebuild(s, mask, item_ids, native)) if fault == "F1" else                   (lambda s: synthetic._rebuild(s, mask, item_ids))
        before, after = rebuild(scores), rebuild(faulted)
        same = ~np.any(before[10][0] != after[10][0], axis=1)
        assert same.any() or fault == "F3"
        assert np.array_equal(lists[10][0][same], native[10][0][same])
        assert np.array_equal(lists[10][0][~same], after[10][0][~same])


def test_v2_faithful_rebuild_reproduces_score_consistent_native_lists():
    v2, scores, mask, item_ids, _ = _native_fixture()
    native = synthetic._rebuild(scores, mask, item_ids)  # consistent with the scores, arbitrary tie order
    rebuilt = v2.faithful_rebuild(scores, mask, item_ids, native)
    for k in (10, 20):
        assert np.array_equal(rebuilt[k][0], native[k][0])


def test_v2_listed_scores_come_from_the_faulted_matrix():
    v2, scores, mask, item_ids, native = _native_fixture()
    faulted, lists, _ = v2.isolated_fault("F3", scores, mask, item_ids, native)
    column = {int(item): i for i, item in enumerate(item_ids)}
    for row in (0, 7, 50):
        expected = [faulted[row, column[int(item)]] for item in lists[10][0][row]]
        assert np.array_equal(lists[10][1][row], expected)


def test_v2_records_survive_a_json_round_trip():
    from silentrank.detect import synthetic_v2 as v2

    decision = {0.002: "harmful", 0.005: "equivalent", 0.01: "equivalent"}
    record = {"case": "X", "faults": {"F1": {"reference": decision, "panels": {"uniform:100": decision}}}}
    restored = v2._load(json.loads(json.dumps(v2._dump(record))))
    assert restored == record and restored["faults"]["F1"]["reference"][0.005] == "equivalent"


def test_v2_cohort_is_pinned():
    from silentrank.detect import synthetic_v2 as v2

    cases = v2.cohort()
    assert len(cases) == 97 and cases == sorted(cases) and not any("ml20m" in c or "-P7" in c for c in cases)
