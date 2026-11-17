from math import sqrt

import numpy as np
import pytest

from silentrank.stats.paired import (
    ACTIVITY_CONFIDENCE_LEVEL,
    ANALYSIS_SEED,
    BOOTSTRAP_REPLICATES,
    CONFIDENCE_LEVEL,
    PILOT_SEEDS,
    paired_seed_user_bootstrap,
    paired_user_bootstrap,
    seed_t_interval,
)


def test_paired_user_bootstrap_uses_pcg64_and_percentiles():
    losses = np.array([-0.25, 0.0, 0.5], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(713))
    samples = rng.integers(3, size=(11, 3), dtype=np.int64)
    expected = np.quantile(losses[samples].mean(axis=1), [0.1, 0.9])

    result = paired_user_bootstrap(losses, level=0.8, replicates=11)

    assert result.estimate == pytest.approx(1 / 12)
    assert (result.low, result.high) == pytest.approx(expected)
    assert (result.n_users, result.n_seeds) == (3, 1)
    assert result.method == "paired_user_bootstrap"
    assert result == paired_user_bootstrap(losses, level=0.8, replicates=11)


def test_seed_t_uses_seed_level_means_and_seed_degrees_of_freedom():
    losses = np.array(
        [[-0.15, -0.05], [-0.1, 0.0], [-0.05, 0.05], [0.0, 0.1], [0.05, 0.15]],
        dtype=np.float64,
    )
    margin = 2.7764451051977987 * sqrt(0.00125)

    result = seed_t_interval(losses)
    repeated_users = seed_t_interval(np.tile(losses, (1, 10)))

    assert result.estimate == pytest.approx(0.0)
    assert (result.low, result.high) == pytest.approx((-margin, margin))
    assert (repeated_users.low, repeated_users.high) == pytest.approx((-margin, margin))
    assert (result.n_users, result.n_seeds) == (2, 5)
    assert result.method == "paired_seed_t"
    assert seed_t_interval(losses, level=ACTIVITY_CONFIDENCE_LEVEL).high > result.high


def test_seed_user_bootstrap_shares_the_user_draw_across_seeds():
    # Each endpoint has probability 1/8 with shared users and 1/16 with
    # independent draws per seed, so the central 80% separates the procedures.
    result = paired_seed_user_bootstrap([[0.0, 1.0], [1.0, 0.0]], level=0.8)

    assert (result.estimate, result.low, result.high) == (0.5, 0.0, 1.0)
    assert (result.n_users, result.n_seeds) == (2, 2)
    assert result.method == "paired_seed_user_bootstrap"


def test_degenerate_samples_have_exact_point_intervals():
    users = paired_user_bootstrap([0.1])
    seeds = seed_t_interval([[0.0, 0.5], [0.5, 0.0]])
    constant_seeds = seed_t_interval(np.full((5, 7), 0.1, dtype=np.float64))
    crossed = paired_seed_user_bootstrap([[0.1, 0.1], [0.1, 0.1]])

    assert (users.estimate, users.low, users.high) == (0.1, 0.1, 0.1)
    assert (seeds.estimate, seeds.low, seeds.high) == (0.25, 0.25, 0.25)
    assert (constant_seeds.estimate, constant_seeds.low, constant_seeds.high) == (0.1, 0.1, 0.1)
    assert (crossed.estimate, crossed.low, crossed.high) == (0.1, 0.1, 0.1)


def test_protocol_defaults():
    assert CONFIDENCE_LEVEL == 0.95
    assert ACTIVITY_CONFIDENCE_LEVEL == 1 - 0.05 / 3
    assert ANALYSIS_SEED == 713
    assert BOOTSTRAP_REPLICATES == 10_000
    assert PILOT_SEEDS == (17, 29, 43, 59, 71)


@pytest.mark.parametrize(
    ("function", "losses", "kwargs", "message"),
    [
        (paired_user_bootstrap, [], {}, "nonempty vector"),
        (paired_user_bootstrap, [np.nan], {}, "finite"),
        (paired_user_bootstrap, [0.5j], {}, "real numbers"),
        (paired_user_bootstrap, [0.0], {"replicates": 0}, "positive integer"),
        (seed_t_interval, [[0.0]], {}, "at least two"),
        (seed_t_interval, [[0.0], [0.0]], {"level": 1}, "confidence level"),
        (paired_seed_user_bootstrap, [0.0, 0.0], {}, "seeds × users matrix"),
    ],
)
def test_invalid_interval_inputs_are_rejected(function, losses, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function(losses, **kwargs)
