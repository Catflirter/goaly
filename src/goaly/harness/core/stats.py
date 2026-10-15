"""Statistics for comparing versions over tasks and seeds (spec §6.3, §8.3).

Nothing here knows a benchmark: the values are per-task scores keyed by task, such as credits.

- :func:`seed_means`: a task's value for a version is its mean over that version's seeds.
- :func:`mean_std`: a metric's mean and standard deviation over seeds.
- :func:`bootstrap_ci`: a percentile bootstrap confidence interval that resamples tasks.
- :func:`sign_test`: paired win, loss and tie counts, and the exact one-sided sign test.
"""

import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "Interval",
    "MeanStd",
    "SignTest",
    "bootstrap_ci",
    "mean_std",
    "seed_means",
    "sign_test",
]

BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class MeanStd:
    """A mean, with the sample standard deviation (n - 1), which is None for a single value."""

    mean: float
    std: float | None


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval."""

    low: float
    high: float


@dataclass(frozen=True, slots=True)
class SignTest:
    """A candidate compared with a baseline, key by key, and the one-sided sign test's p-value."""

    wins: int  # Keys where the candidate's value is higher.
    losses: int  # Keys where it is lower.
    ties: int  # Keys where the two are equal; the test drops them.
    p_value: float


def mean_std(values: Sequence[float]) -> MeanStd:
    """Return the mean of ``values`` and their sample standard deviation, if there are two."""
    if not values:
        raise ValueError("mean_std needs at least one value")
    return MeanStd(statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else None)


def seed_means[K](values: Mapping[K, Sequence[float]]) -> dict[K, float]:
    """Return each key's mean over its seeds; ``values`` maps a key to its value in each seed.

    Every key needs one value per seed, so the same number of values. fmean's sum has no
    rounding error when it is representable, as for credits (multiples of 0.25), and it divides
    once: equal true means then give equal floats, whatever the order of the values, so a
    paired comparison sees exact ties.
    """
    first: tuple[K, int] | None = None
    for key, seeds in values.items():
        if not seeds:
            raise ValueError(f"{key!r} has no values")
        if first is None:
            first = (key, len(seeds))
        elif len(seeds) != first[1]:
            raise ValueError(
                f"every key needs one value per seed, but {key!r} has {len(seeds)}"
                f" and {first[0]!r} has {first[1]}"
            )
    return {key: statistics.fmean(seeds) for key, seeds in values.items()}


def bootstrap_ci[T](
    items: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
) -> Interval:
    """Return the percentile bootstrap interval of ``statistic(items)``.

    Each resample draws ``len(items)`` items with replacement from ``random.Random(seed)``, so
    the same arguments always give the same interval. The bounds are the statistic's
    (1 - confidence) / 2 and (1 + confidence) / 2 quantiles over the resamples, interpolated
    linearly between order statistics. To compare two versions, pass each task's pair or
    difference as one item, so that a resample keeps the pair together.
    """
    if not items:
        raise ValueError("bootstrap_ci needs at least one item")
    if resamples < 1:
        raise ValueError("bootstrap_ci needs at least one resample")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, not {confidence}")
    rng = random.Random(seed)  # noqa: S311 - seeded for reproducibility, not for secrecy.
    values = sorted(statistic(rng.choices(items, k=len(items))) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return Interval(_quantile(values, tail), _quantile(values, 1 - tail))


def sign_test[K](baseline: Mapping[K, float], candidate: Mapping[K, float]) -> SignTest:
    """Compare ``candidate`` with ``baseline`` key by key, with the exact one-sided sign test.

    Ties are dropped. Under the null hypothesis each untied key is a fair coin flip, so the
    p-value is P(X ≥ wins) for X ~ Binomial(wins + losses, 1/2), and 1 when every key ties.
    """
    if baseline.keys() != candidate.keys():
        raise ValueError("the baseline and the candidate need the same keys")
    wins = sum(candidate[key] > value for key, value in baseline.items())
    losses = sum(candidate[key] < value for key, value in baseline.items())
    trials = wins + losses
    tail = sum(math.comb(trials, k) for k in range(wins, trials + 1))
    # The tail is an exact integer, and integer true division rounds once.
    return SignTest(wins, losses, len(baseline) - trials, tail / 2**trials)


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Return the ``fraction`` quantile of the sorted ``ordered``, interpolated linearly."""
    position = fraction * (len(ordered) - 1)
    below = math.floor(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (position - below) * (ordered[above] - ordered[below])
