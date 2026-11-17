import numpy as np
import pytest

from silentrank.evaluator.contracts import check_topk


@pytest.mark.parametrize(
    ("ranking", "scores", "violations"),
    [
        ([11, 14], [0.9, 0.5], set()),
        ([11, 11], [0.9, 0.5], {"duplicate_items"}),
        ([13, 14], [0.9, 0.5], {"seen_items"}),
        ([99, 14], [0.9, 0.5], {"invalid_item_ids"}),
        ([11, 14], [np.nan, np.inf], {"nonfinite_scores"}),
        ([11], [0.9], {"wrong_length"}),
        ([11, 14, 13], [0.9, 0.5, 0.3], {"wrong_length", "seen_items"}),
        ([11, 14], [0.9], {"score_alignment"}),
        ([11, 14], None, {"invalid_scores"}),
        ([11, None], [0.9, 0.5], {"invalid_item_ids"}),
    ],
)
def test_topk_contracts(ranking, scores, violations):
    actual = check_topk(ranking, scores, catalog=[11, 13, 14], seen=[13, 21], k=2)
    assert set(actual) == violations


def test_short_and_empty_lists_are_valid_when_eligible_items_are_exhausted():
    assert check_topk([11], [0.9], catalog=[11, 14], seen=[14], k=10) == ()
    assert check_topk([], [], catalog=[11, 14], seen=[11, 14], k=10) == ()
