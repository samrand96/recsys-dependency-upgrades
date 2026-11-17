"""Contracts for native lists against the frozen catalogue and seen mask."""

import numpy as np
from numpy.typing import ArrayLike

from ._inputs import cutoff, item_ids, score_vector


def check_topk(
    ranked_items: ArrayLike,
    scores: ArrayLike,
    *,
    catalog: ArrayLike,
    seen: ArrayLike,
    k: int,
) -> tuple[str, ...]:
    """Return violation codes for one user, or an empty tuple for a valid list.

    The ranking, frozen training catalogue and seen items use the same ID
    space. The seen mask may also contain IDs outside the catalogue. A score
    must be supplied for each returned item.
    """
    k = cutoff(k)
    catalog_ids = np.unique(item_ids(catalog, "catalog"))
    seen_ids = item_ids(seen, "seen")
    eligible_count = np.count_nonzero(~np.isin(catalog_ids, seen_ids))
    violations = []

    try:
        returned = item_ids(ranked_items, "ranked_items")
    except ValueError:
        violations.append("invalid_item_ids")
        returned = None

    if returned is not None:
        if returned.size != min(k, eligible_count):
            violations.append("wrong_length")
        if np.unique(returned).size != returned.size:
            violations.append("duplicate_items")
        if not np.isin(returned, catalog_ids).all():
            violations.append("invalid_item_ids")
        if np.isin(returned, seen_ids).any():
            violations.append("seen_items")

    try:
        values = score_vector(scores)
    except ValueError:
        violations.append("invalid_scores")
    else:
        if returned is not None and values.size != returned.size:
            violations.append("score_alignment")
        if not np.isfinite(values).all():
            violations.append("nonfinite_scores")

    return tuple(violations)
