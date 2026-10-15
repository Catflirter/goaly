"""Tests for goaly.harness.core.budgets: the budgets' config, their checks, and the error streak."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from goaly.contracts.gen.trajectory_pb2 import StopReason
from goaly.harness.core.budgets import Budgets, BudgetUse, ErrorStreak

STEPS = StopReason.STOP_REASON_STEP_BUDGET_EXHAUSTED
TIME = StopReason.STOP_REASON_TIME_BUDGET_EXHAUSTED
TOKENS = StopReason.STOP_REASON_TOKEN_BUDGET_EXHAUSTED
COST = StopReason.STOP_REASON_COST_BUDGET_EXHAUSTED


def budgets(**changes: Any) -> Budgets:
    return Budgets.model_validate({"steps": 30, "wall_clock_s": 600.0, **changes})


def use(
    steps: int = 0, elapsed_s: float = 0.0, tokens: int = 0, cost_usd: float = 0.0
) -> BudgetUse:
    return BudgetUse(steps=steps, elapsed_s=elapsed_s, tokens=tokens, cost_usd=cost_usd)


def test_v0_effisql_budgets_read_from_config() -> None:
    v0 = Budgets.model_validate({"steps": 30, "wall_clock_s": 600})

    assert v0 == Budgets(steps=30, wall_clock_s=600.0)
    assert v0.tokens is None
    assert v0.cost_usd is None
    assert (v0.remind_after_errors, v0.stop_after_errors) == (3, 5)  # Spec §5.1.


@pytest.mark.parametrize(
    "config",
    [
        {"wall_clock_s": 600},  # The step budget has no default.
        {"steps": 30},  # Nor does the wall clock.
        {"steps": 0, "wall_clock_s": 600},
        {"steps": 30, "wall_clock_s": 0},
        {"steps": 30, "wall_clock_s": -1},
        {"steps": 30, "wall_clock_s": math.inf},
        {"steps": 30, "wall_clock_s": math.nan},
        {"steps": 30, "wall_clock_s": 600, "tokens": 0},
        {"steps": 30, "wall_clock_s": 600, "cost_usd": 0},
        {"steps": 30, "wall_clock_s": 600, "cost_usd": math.inf},
        {"steps": 30, "wall_clock_s": 600, "remind_after_errors": 0},
        {"steps": 30, "wall_clock_s": 600, "remind_after_errors": 5},  # Not before the stop.
        {"steps": 30, "wall_clock_s": 600, "stop_after_errors": 3},
        {"steps": 30, "wall_clock_s": 600, "max_steps": 30},  # Unknown fields are typos.
    ],
)
def test_invalid_budgets_are_rejected(config: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Budgets.model_validate(config)


def test_budgets_are_frozen() -> None:
    with pytest.raises(ValidationError):
        budgets().steps = 31  # type: ignore[misc]  # Frozen, as the test shows.


def test_nothing_is_exhausted_below_every_budget() -> None:
    limits = budgets(tokens=10_000, cost_usd=1.0)

    assert limits.exhausted(use(steps=29, elapsed_s=599.9, tokens=9_999, cost_usd=0.99)) is None


@pytest.mark.parametrize(
    ("spent", "reason"),
    [
        (use(steps=30), STEPS),
        (use(elapsed_s=600.0), TIME),
        (use(tokens=10_000), TOKENS),
        (use(cost_usd=1.0), COST),
        (use(steps=31, tokens=20_000), STEPS),  # Past the budget counts too.
    ],
)
def test_a_budget_is_exhausted_once_it_is_all_spent(
    spent: BudgetUse, reason: StopReason.ValueType
) -> None:
    assert budgets(tokens=10_000, cost_usd=1.0).exhausted(spent) == reason


def test_the_first_budget_in_stop_reason_order_is_named() -> None:
    limits = budgets(tokens=10, cost_usd=0.1)

    everything = use(steps=30, elapsed_s=600.0, tokens=10, cost_usd=0.1)
    assert limits.exhausted(everything) == STEPS
    assert limits.exhausted(use(elapsed_s=600.0, tokens=10, cost_usd=0.1)) == TIME
    assert limits.exhausted(use(tokens=10, cost_usd=0.1)) == TOKENS


def test_no_token_or_cost_budget_means_no_cap() -> None:
    assert budgets().exhausted(use(tokens=10**12, cost_usd=1e6)) is None


def test_an_error_streak_counts_errors_in_a_row_and_a_success_resets_it() -> None:
    streak = ErrorStreak(budgets())

    for error in (True, True, False, True):
        streak.add(error=error)

    assert streak.count == 1
    assert not streak.exhausted


def test_an_error_streak_reminds_once_when_it_reaches_the_reminder() -> None:
    streak = ErrorStreak(budgets())
    reminders = []
    for _ in range(4):
        streak.add(error=True)
        reminders.append(streak.take_reminder())

    assert reminders == [False, False, True, False]


def test_a_new_streak_gets_a_new_reminder() -> None:
    streak = ErrorStreak(budgets())
    for _ in range(3):
        streak.add(error=True)
    assert streak.take_reminder()

    streak.add(error=False)
    for _ in range(3):
        streak.add(error=True)

    assert streak.take_reminder()


def test_an_error_streak_is_exhausted_at_the_stop() -> None:
    streak = ErrorStreak(budgets(remind_after_errors=1, stop_after_errors=2))

    streak.add(error=True)
    assert not streak.exhausted
    streak.add(error=True)
    assert streak.exhausted
