"""Fixed user selection rules using only training activity and baseline scores."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from silentrank.stats.paired import ANALYSIS_SEED

PANEL_SIZES = (100, 300, 1_000)
PANEL_STRATEGIES = ("uniform", "activity", "score_margin")


@dataclass(frozen=True)
class Panel:
    strategy: str
    positions: NDArray[np.int64]
    undefined_margins: int = 0


def normalized_margins(scores: np.ndarray, seen: np.ndarray, k: int = 10) -> NDArray[np.float64]:
    """Compute the K/K+1 eligible score gap divided by the user's score range.

    A constant score row has margin zero. With at most K eligible items the
    margin is undefined and sorts after every defined margin.
    """
    if scores.ndim != 2 or seen.shape != scores.shape or seen.dtype != np.bool_:
        raise ValueError("scores and the Boolean seen mask must have the same matrix shape")
    if k < 1:
        raise ValueError("k must be positive")
    margins = np.full(scores.shape[0], np.inf, dtype=np.float64)
    for user, mask in enumerate(seen):
        eligible = scores[user, ~mask].astype(np.float64, copy=False)
        if not np.isfinite(eligible).all():
            raise ValueError("baseline eligible scores must be finite")
        if len(eligible) <= k:
            continue
        cut = len(eligible) - k
        threshold = np.partition(eligible, (cut - 1, cut))
        lower, upper = float(eligible.min()), float(eligible.max())
        scale = max(abs(lower), abs(upper))
        if lower == upper:
            margins[user] = 0.0
        elif lower < 0 < upper:
            margins[user] = (threshold[cut] / scale - threshold[cut - 1] / scale) / (upper / scale - lower / scale)
        else:
            margins[user] = (threshold[cut] - threshold[cut - 1]) / (upper - lower)
    return margins


def _stratified(order: np.ndarray, activity: np.ndarray, size: int) -> NDArray[np.int64]:
    groups, counts = np.unique(activity, return_counts=True)
    quotas, remainder = np.divmod(counts * size, len(order))
    priority = np.argsort(-remainder, kind="stable")
    quotas[priority[:size - int(quotas.sum())]] += 1
    return np.concatenate([order[activity[order] == group][:quota] for group, quota in zip(groups, quotas)])


def build_panels(
    activity: np.ndarray, scores: np.ndarray, seen: np.ndarray, sizes: tuple[int, ...] = PANEL_SIZES,
) -> list[Panel]:
    """Select all available sizes without replacement, using one fixed random priority."""
    if activity.ndim != 1 or len(activity) != len(scores) or not len(activity):
        raise ValueError("one activity group is required for each baseline user")
    if any(size < 1 for size in sizes):
        raise ValueError("panel sizes must be positive")
    margins = normalized_margins(scores, seen)
    rng = np.random.Generator(np.random.PCG64(ANALYSIS_SEED))
    order = rng.permutation(np.arange(len(activity), dtype=np.int64))
    margin_order = order[np.argsort(margins[order], kind="stable")]
    panels = []
    for size in sizes:
        if size > len(order):
            continue
        selected = margin_order[:size]
        panels.extend((
            Panel("uniform", np.sort(order[:size])),
            Panel("activity", np.sort(_stratified(order, activity, size))),
            Panel("score_margin", np.sort(selected), int(np.count_nonzero(np.isinf(margins[selected])))),
        ))
    return panels
