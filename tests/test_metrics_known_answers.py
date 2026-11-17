from math import log2

import pytest

from silentrank.evaluator.metrics import ndcg_at_k, recall_at_k


def test_binary_quality_at_cutoff():
    ranking = [1, 3, 2, 4]
    relevant = [3, 4, 5]
    discount_rank_2 = 1 / log2(3)

    assert ndcg_at_k(ranking, relevant, 2) == pytest.approx(
        discount_rank_2 / (1 + discount_rank_2)
    )
    assert recall_at_k(ranking, relevant, 2) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranking, relevant, 4) == pytest.approx(
        (discount_rank_2 + 1 / log2(5)) / (1 + discount_rank_2 + 1 / 2)
    )
    assert recall_at_k(ranking, relevant, 4) == pytest.approx(2 / 3)


def test_perfect_and_reversed_rankings():
    ranking = [7, 2, 5, 3]
    relevant = [7, 2]

    assert ndcg_at_k(ranking, relevant, 2) == 1.0
    assert recall_at_k(ranking, relevant, 2) == 1.0
    assert ndcg_at_k(ranking[::-1], relevant, 2) == 0.0
    assert recall_at_k(ranking[::-1], relevant, 2) == 0.0


def test_short_list_uses_all_eligible_relevant_items():
    assert ndcg_at_k([1], [1, 2, 2], 10) == pytest.approx(1 / (1 + 1 / log2(3)))
    assert recall_at_k([1], [1, 2, 2], 10) == 0.5


@pytest.mark.parametrize("metric", [ndcg_at_k, recall_at_k])
def test_metrics_reject_an_ineligible_user_and_duplicate_recommendations(metric):
    with pytest.raises(ValueError, match="eligible relevant item"):
        metric([1, 2], [], 2)
    with pytest.raises(ValueError, match="duplicate items"):
        metric([1, 1], [1, 2], 2)
