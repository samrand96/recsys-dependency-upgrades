"""Paired intervals for protocol v0's old-minus-new per-user losses."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t

CONFIDENCE_LEVEL = 0.95
ACTIVITY_CONFIDENCE_LEVEL = 1 - 0.05 / 3
BOOTSTRAP_REPLICATES = 10_000
ANALYSIS_SEED = 713
PILOT_SEEDS = (17, 29, 43, 59, 71)


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    level: float
    method: str
    n_users: int
    n_seeds: int


def _paired_losses(losses: ArrayLike, ndim: int) -> NDArray[np.float64]:
    values = np.asarray(losses)
    if values.ndim != ndim or not values.size:
        shape = "a nonempty vector of users" if ndim == 1 else "a nonempty seeds × users matrix"
        raise ValueError(f"losses must be {shape}")
    if values.dtype.kind not in "iuf":
        raise ValueError("losses must be real numbers")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("losses must be finite")
    return values


def _tail_probability(level: float) -> float:
    if not 0 < level < 1:
        raise ValueError("confidence level must be between 0 and 1")
    return (1 - level) / 2


def _bootstrap_interval(
    losses: NDArray[np.float64],
    *,
    level: float,
    replicates: int,
    seed: int,
    resample_seeds: bool,
    method: str,
) -> Interval:
    tail = _tail_probability(level)
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, (int, np.integer))
        or replicates < 1
    ):
        raise ValueError("bootstrap replicates must be a positive integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    n_seeds, n_users = losses.shape
    if np.all(losses == losses[0, 0]):
        value = float(losses[0, 0])
        return Interval(value, value, value, level, method, n_users, n_seeds)

    means = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        if resample_seeds:
            seeds = rng.integers(n_seeds, size=n_seeds, dtype=np.int64)
            users = rng.integers(n_users, size=n_users, dtype=np.int64)
            # One user draw is shared across all sampled seed rows.
            sampled = losses[np.ix_(seeds, users)]
        else:
            users = rng.integers(n_users, size=n_users, dtype=np.int64)
            sampled = losses[0, users]
        means[i] = sampled.mean()

    low, high = np.quantile(means, [tail, 1 - tail], method="linear")
    return Interval(float(losses.mean()), float(low), float(high), level, method, n_users, n_seeds)


def paired_user_bootstrap(
    losses: ArrayLike,
    *,
    level: float = CONFIDENCE_LEVEL,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = ANALYSIS_SEED,
) -> Interval:
    """Percentile interval from resampling paired user differences."""
    values = _paired_losses(losses, ndim=1)
    return _bootstrap_interval(
        values[None, :],
        level=level,
        replicates=replicates,
        seed=seed,
        resample_seeds=False,
        method="paired_user_bootstrap",
    )


def seed_t_interval(losses: ArrayLike, *, level: float = CONFIDENCE_LEVEL) -> Interval:
    """Student t interval across the mean paired losses of independent seeds.

    Rows are seed pairs and columns are the same users in the same order.
    """
    values = _paired_losses(losses, ndim=2)
    n_seeds, n_users = values.shape
    if n_seeds < 2:
        raise ValueError("the t interval requires at least two paired seeds")
    tail = _tail_probability(level)
    method = "paired_seed_t"
    if np.all(values == values[0, 0]):
        value = float(values[0, 0])
        return Interval(value, value, value, level, method, n_users, n_seeds)

    seed_means = values.mean(axis=1)
    if np.all(seed_means == seed_means[0]):
        value = float(seed_means[0])
        return Interval(value, value, value, level, method, n_users, n_seeds)

    estimate = float(seed_means.mean())
    margin = float(t.isf(tail, n_seeds - 1) * seed_means.std(ddof=1) / np.sqrt(n_seeds))
    return Interval(estimate, estimate - margin, estimate + margin, level, method, n_users, n_seeds)


def paired_seed_user_bootstrap(
    losses: ArrayLike,
    *,
    level: float = CONFIDENCE_LEVEL,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = ANALYSIS_SEED,
) -> Interval:
    """Resample paired seeds, then a shared set of paired users per replicate."""
    values = _paired_losses(losses, ndim=2)
    return _bootstrap_interval(
        values,
        level=level,
        replicates=replicates,
        seed=seed,
        resample_seeds=True,
        method="paired_seed_user_bootstrap",
    )
