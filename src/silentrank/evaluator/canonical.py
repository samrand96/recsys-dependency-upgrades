"""Deterministic diagnostic ranking over all eligible candidates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._inputs import cutoff, item_ids, score_vector


@dataclass(frozen=True)
class CanonicalRanking:
    item_ids: NDArray[np.int64]
    scores: NDArray[np.floating]
    boundary_tie_size: int
    boundary_tie_crosses_cutoff: bool


def canonical_topk(
    external_ids: ArrayLike, scores: ArrayLike, k: int
) -> CanonicalRanking:
    """Sort by descending score, then ascending external ID for exact ties.

    Pass every eligible candidate with its finite score in the same input
    order. The score of the Kth item identifies its tie block. The tie size is
    zero if the Kth item does not exist or its score is unique.
    """
    k = cutoff(k)
    ids = item_ids(external_ids, "external_ids")
    values = score_vector(scores)
    if ids.size != values.size:
        raise ValueError("external_ids and scores must have the same length")
    if np.unique(ids).size != ids.size:
        raise ValueError("external_ids must be unique")
    if not np.isfinite(values).all():
        raise ValueError("eligible candidate scores must be finite")

    # Sort the full eligible universe so every candidate tied at K is considered.
    order = np.lexsort((ids, -values))
    top = order[:k]
    tie_size = 0
    crosses_cutoff = False
    if ids.size >= k:
        threshold = values[top[-1]]
        count = int(np.count_nonzero(values == threshold))
        if count > 1:
            tie_size = count
            crosses_cutoff = ids.size > k and bool(values[order[k]] == threshold)
    return CanonicalRanking(ids[top], values[top], tie_size, crosses_cutoff)
