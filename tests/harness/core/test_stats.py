"""Tests for goaly.harness.core.stats."""

import itertools
import math
import random
import statistics
from collections.abc import Sequence
from fractions import Fraction

import pytest

from goaly.harness.core.stats import (
    BOOTSTRAP_RESAMPLES,
    Interval,
    MeanStd,
    SignTest,
    bootstrap_ci,
    mean_std,
    seed_means,
    sign_test,
)


def scaled_mean(values: Sequence[float]) -> float:
    return 100 * statistics.fmean(values)


def outcomes(wins: int, losses: int, ties: int = 0) -> tuple[dict[str, float], dict[str, float]]:
    """A baseline and a candidate, keyed by task, with these paired outcomes."""
    pairs = [(0.0, 1.0)] * wins + [(1.0, 0.0)] * losses + [(0.5, 0.5)] * ties
    baseline = {f"t{index}": base for index, (base, _) in enumerate(pairs)}
    candidate = {f"t{index}": cand for index, (_, cand) in enumerate(pairs)}
    return baseline, candidate


def upper_tail(wins: int, losses: int) -> float:
    """P(X >= wins) for X ~ Binomial(wins + losses, 1/2), in exact arithmetic."""
    trials = wins + losses
    tail = sum(math.comb(trials, k) for k in range(wins, trials + 1))
    return float(Fraction(tail, 2**trials))


# mean_std.


def test_mean_std_uses_the_sample_standard_deviation() -> None:
    assert mean_std([1.0, 2.0, 3.0, 4.0]) == MeanStd(mean=2.5, std=statistics.stdev([1, 2, 3, 4]))


def test_mean_std_of_one_value_has_no_standard_deviation() -> None:
    assert mean_std([88.3]) == MeanStd(mean=88.3, std=None)


def test_mean_std_needs_a_value() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        mean_std([])


# seed_means.


def test_seed_means_averages_each_key_over_its_seeds() -> None:
    means = seed_means({"a": [1.0, 0.5], "b": [0.25, 0.25], "c": [0.0, 0.75]})

    assert means == {"a": 0.75, "b": 0.25, "c": 0.375}


def test_seed_means_of_three_seeds_is_exact_for_equal_true_means() -> None:
    # Credits are multiples of 0.25, so equal true means give equal floats whatever the order.
    means = seed_means({"a": [0.0, 0.0, 1.0], "b": [0.25, 0.5, 0.25], "c": [1.0, 0.0, 0.0]})

    assert means["a"] == means["b"] == means["c"] == 1 / 3


def test_seed_means_of_no_keys_is_empty() -> None:
    assert seed_means({}) == {}


def test_seed_means_rejects_a_key_without_seeds() -> None:
    with pytest.raises(ValueError, match="'b' has no values"):
        seed_means({"a": [1.0], "b": []})


def test_seed_means_rejects_keys_with_different_seed_counts() -> None:
    with pytest.raises(ValueError, match="'b' has 1 and 'a' has 2"):
        seed_means({"a": [1.0, 0.0], "b": [1.0]})


# bootstrap_ci.


def test_bootstrap_resamples_the_items_with_replacement_10000_times_by_default() -> None:
    sizes: list[int] = []
    drawn: set[float] = set()

    def record(sample: Sequence[float]) -> float:
        sizes.append(len(sample))
        drawn.update(sample)
        return statistics.fmean(sample)

    bootstrap_ci([1.0, 2.0, 3.0], record, seed=7)

    assert BOOTSTRAP_RESAMPLES == 10_000
    assert len(sizes) == BOOTSTRAP_RESAMPLES
    assert set(sizes) == {3}
    assert drawn == {1.0, 2.0, 3.0}


def test_bootstrap_ci_is_the_95_percent_percentile_interval() -> None:
    items = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0]
    # The reference: the same resamples, and their 2.5th and 97.5th percentiles, interpolated
    # linearly between order statistics (statistics.quantiles' "inclusive" method).
    rng = random.Random(20261014)  # noqa: S311 - seeded for reproducibility, not for secrecy.
    values = [scaled_mean(rng.choices(items, k=len(items))) for _ in range(2_000)]
    cuts = statistics.quantiles(values, n=40, method="inclusive")

    interval = bootstrap_ci(items, scaled_mean, seed=20261014, resamples=2_000)

    assert interval.low == pytest.approx(cuts[0], rel=1e-12)
    assert interval.high == pytest.approx(cuts[-1], rel=1e-12)


def test_bootstrap_ci_interpolates_between_order_statistics() -> None:
    calls = itertools.count()  # The statistic's values are 0, 1, ..., 100.

    interval = bootstrap_ci([1.0], lambda _: float(next(calls)), seed=0, resamples=101)

    assert interval.low == pytest.approx(2.5)
    assert interval.high == pytest.approx(97.5)


def test_bootstrap_ci_takes_any_confidence_level() -> None:
    calls = itertools.count()

    interval = bootstrap_ci(
        [1.0], lambda _: float(next(calls)), seed=0, resamples=11, confidence=0.8
    )

    assert interval.low == pytest.approx(1.0)
    assert interval.high == pytest.approx(9.0)


def test_bootstrap_ci_is_deterministic_for_a_seed() -> None:
    items = [float(value) for value in range(30)]

    first = bootstrap_ci(items, statistics.fmean, seed=1, resamples=1_000)

    assert bootstrap_ci(items, statistics.fmean, seed=1, resamples=1_000) == first
    assert bootstrap_ci(items, statistics.fmean, seed=2, resamples=1_000) != first


def test_bootstrap_ci_of_the_mean_covers_the_mean_with_the_expected_width() -> None:
    items = [float(value) for value in range(100)]  # Mean 49.5, standard error about 2.9.

    interval = bootstrap_ci(items, statistics.fmean, seed=11)

    assert 42.5 < interval.low < 45.0
    assert 54.0 < interval.high < 56.5


def test_bootstrap_ci_of_identical_items_is_a_point() -> None:
    assert bootstrap_ci([0.75] * 5, scaled_mean, seed=0, resamples=50) == Interval(75.0, 75.0)


@pytest.mark.parametrize(
    ("items", "resamples", "confidence", "message"),
    [
        ([], 10, 0.95, "at least one item"),
        ([1.0], 0, 0.95, "at least one resample"),
        ([1.0], 10, 0.0, "between 0 and 1"),
        ([1.0], 10, 1.0, "between 0 and 1"),
    ],
)
def test_bootstrap_ci_rejects_bad_arguments(
    items: list[float], resamples: int, confidence: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_ci(items, statistics.fmean, seed=0, resamples=resamples, confidence=confidence)


# sign_test.


def test_sign_test_counts_wins_losses_and_ties_by_key() -> None:
    baseline = {"a": 0.0, "b": 1.0, "c": 0.5, "d": 0.25}
    candidate = {"d": 0.75, "c": 0.5, "b": 0.0, "a": 1.0}  # Paired by key, not by position.

    result = sign_test(baseline, candidate)

    assert (result.wins, result.losses, result.ties) == (2, 1, 1)


@pytest.mark.parametrize(
    ("wins", "losses", "p_value"),
    [
        (5, 0, 1 / 32),
        (4, 1, 6 / 32),
        (7, 2, 46 / 512),  # The smallest win count of 9 that passes p <= 0.1.
        (6, 3, 130 / 512),
        (3, 3, 42 / 64),
        (0, 3, 1.0),
        (1, 0, 0.5),
    ],
)
def test_sign_test_is_the_exact_one_sided_binomial(wins: int, losses: int, p_value: float) -> None:
    baseline, candidate = outcomes(wins, losses)

    assert sign_test(baseline, candidate) == SignTest(wins, losses, 0, p_value)


def test_sign_test_drops_ties() -> None:
    baseline, candidate = outcomes(5, 0, ties=10)

    assert sign_test(baseline, candidate) == SignTest(wins=5, losses=0, ties=10, p_value=1 / 32)


def test_sign_test_with_only_ties_has_p_value_1() -> None:
    baseline, candidate = outcomes(0, 0, ties=3)

    assert sign_test(baseline, candidate) == SignTest(wins=0, losses=0, ties=3, p_value=1.0)


def test_sign_test_stays_exact_for_many_tasks() -> None:
    baseline, candidate = outcomes(200, 38)

    p_value = sign_test(baseline, candidate).p_value

    assert p_value == upper_tail(200, 38)
    assert 0.0 < p_value < 1e-25


def test_sign_test_needs_the_same_keys() -> None:
    with pytest.raises(ValueError, match="the same keys"):
        sign_test({"a": 1.0, "b": 0.0}, {"a": 1.0, "c": 0.0})
