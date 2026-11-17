"""Independent binary relevance metrics for a single user's native ranking."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._inputs import cutoff, item_ids


def _hits_at_k(
    ranked_items: ArrayLike, relevant_items: ArrayLike, k: int
) -> tuple[NDArray[np.bool_], int]:
    top = item_ids(ranked_items, "ranked_items")[:k]
    relevant = np.unique(item_ids(relevant_items, "relevant_items"))
    if not relevant.size:
        raise ValueError("quality metrics require at least one eligible relevant item")
    if np.unique(top).size != top.size:
        raise ValueError("ranked_items contains duplicate items in the metric cutoff")
    return np.isin(top, relevant), relevant.size


def ndcg_at_k(ranked_items: ArrayLike, relevant_items: ArrayLike, k: int) -> float:
    """Binary NDCG with log2 discount and an ideal list of eligible positives.

    Use the same item-ID space for both arrays. The relevance set must already
    exclude seen and out-of-catalog items.
    """
    k = cutoff(k)
    hits, n_relevant = _hits_at_k(ranked_items, relevant_items, k)
    ideal_size = min(k, n_relevant)
    positions = np.arange(2, max(hits.size, ideal_size) + 2, dtype=np.float64)
    discounts = 1.0 / np.log2(positions)
    dcg = np.sum(discounts[:hits.size][hits], dtype=np.float64)
    ideal_dcg = np.sum(discounts[:ideal_size], dtype=np.float64)
    return float(dcg / ideal_dcg)


def recall_at_k(ranked_items: ArrayLike, relevant_items: ArrayLike, k: int) -> float:
    """Fraction of all eligible positives retrieved in the first K positions."""
    hits, n_relevant = _hits_at_k(ranked_items, relevant_items, cutoff(k))
    return float(np.count_nonzero(hits) / n_relevant)
