"""Validate the item IDs, score vectors and cutoffs used by the evaluator."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def item_ids(values: ArrayLike, name: str) -> NDArray[np.int64]:
    ids = np.asarray(values)
    if ids.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector of item IDs")
    if ids.size and ids.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer item IDs")
    if ids.size and ids.dtype.kind == "u" and int(ids.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} contains an item ID outside the int64 range")
    return ids.astype(np.int64, copy=False)


def score_vector(values: ArrayLike) -> NDArray[np.floating]:
    scores = np.asarray(values)
    if scores.ndim != 1 or scores.dtype.kind != "f":
        raise ValueError("scores must be a one-dimensional floating-point vector")
    return scores


def cutoff(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    return int(k)
