import numpy as np
import pytest

from silentrank.evaluator.canonical import canonical_topk


def test_all_items_tied_at_boundary_are_considered_in_external_id_order():
    ids = np.array([900, 111, 20, 7, 1], dtype=np.int64)
    scores = np.array([1.0, 0.5, 0.5, 0.5, 0.0], dtype=np.float32)
    shuffled = [1, 0, 4, 3, 2]

    for ranking in (
        canonical_topk(ids, scores, 2),
        canonical_topk(ids[shuffled], scores[shuffled], 2),
    ):
        np.testing.assert_array_equal(ranking.item_ids, [900, 7])
        np.testing.assert_array_equal(ranking.scores, [1.0, 0.5])
        assert ranking.scores.dtype == np.float32
        assert ranking.boundary_tie_size == 3
        assert ranking.boundary_tie_crosses_cutoff


def test_scores_just_apart_are_distinct_and_short_rankings_have_no_boundary():
    higher = np.nextafter(np.float64(1.0), np.float64(2.0))
    ranking = canonical_topk([3, 9], [1.0, higher], 1)

    np.testing.assert_array_equal(ranking.item_ids, [9])
    assert ranking.boundary_tie_size == 0
    assert not ranking.boundary_tie_crosses_cutoff

    short = canonical_topk([3, 9], [1.0, 1.0], 3)
    np.testing.assert_array_equal(short.item_ids, [3, 9])
    assert short.boundary_tie_size == 0
    assert not short.boundary_tie_crosses_cutoff


def test_tie_block_can_end_at_the_boundary_without_crossing_it():
    ranking = canonical_topk([3, 2, 1], [1.0, 1.0, 0.0], 2)
    assert ranking.boundary_tie_size == 2
    assert not ranking.boundary_tie_crosses_cutoff


@pytest.mark.parametrize(
    ("ids", "scores", "message"),
    [
        ([1], [1.0, 2.0], "same length"),
        ([1, 1], [1.0, 2.0], "unique"),
        ([1], [np.inf], "finite"),
        ([1], [np.nan], "finite"),
    ],
)
def test_canonical_ranking_requires_aligned_unique_candidates_and_finite_scores(
    ids, scores, message
):
    with pytest.raises(ValueError, match=message):
        canonical_topk(ids, scores, 2)
