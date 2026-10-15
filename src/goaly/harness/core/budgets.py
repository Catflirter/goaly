"""The agent loop's budgets (spec §5.1, §5.4): what a task may spend, and the error streak.

A pack's config.yaml sets the budgets, and v0 Effi-SQL allows 30 steps or 10 minutes. When one
runs out, the loop stops and the pack's fallback answers.

The error rules are core's, which RSI cannot edit, so they are constants here rather than
config (spec §5.1, §8.5). :class:`ErrorStreak` counts the model's failed replies in a row: the
loop reminds the model of the tool schemas after REMIND_AFTER_ERRORS of them, and stops after
STOP_AFTER_ERRORS. Of one reply's calls, no more than CALLS_PER_REPLY run.
"""

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from goaly.contracts.gen.trajectory_pb2 import StopReason

__all__ = [
    "CALLS_PER_REPLY",
    "REMIND_AFTER_ERRORS",
    "STOP_AFTER_ERRORS",
    "BudgetUse",
    "Budgets",
    "ErrorStreak",
]

# Failed replies in a row after which the model is reminded of the tool schemas.
REMIND_AFTER_ERRORS: Final = 3
# Failed replies in a row after which the loop stops; after the reminder, so the model always
# gets replies to act on it.
STOP_AFTER_ERRORS: Final = 5
# How many of one reply's calls run; the rest are answered with an error.
CALLS_PER_REPLY: Final = 16


@dataclass(frozen=True, slots=True)
class BudgetUse:
    """What a task has spent so far: model calls, seconds, tokens and dollars."""

    # Model calls that got a reply.
    steps: int
    # Seconds since the task's first loop started.
    elapsed_s: float
    # Input plus output tokens, summed over the task's model calls (llm_call.proto's Usage).
    tokens: int
    # The cost the gateway reported for those calls; 0 if it reported none.
    cost_usd: float


class Budgets(BaseModel):
    """What one task may spend before the loop stops it and the pack's fallback answers.

    ``steps`` caps the model calls that get a reply, and ``wall_clock_s`` the seconds from the
    start of the task's first loop. ``tokens`` and ``cost_usd`` cap the input plus output tokens
    and the reported cost of every model call of the task; None means no cap. Like the step
    budget, they are checked before each call, so a reply can take a task past them, by its own
    tokens and cost, before the loop stops.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    steps: PositiveInt
    wall_clock_s: PositiveFloat
    tokens: PositiveInt | None = None
    cost_usd: PositiveFloat | None = None

    def exhausted(self, use: BudgetUse) -> StopReason.ValueType | None:
        """Return the stop reason of the first budget ``use`` has used up, or None.

        A budget is used up once nothing of it is left. The budgets are checked in the order of
        their stop reasons: steps, wall clock, tokens, cost.
        """
        if use.steps >= self.steps:
            return StopReason.STOP_REASON_STEP_BUDGET_EXHAUSTED
        if use.elapsed_s >= self.wall_clock_s:
            return StopReason.STOP_REASON_TIME_BUDGET_EXHAUSTED
        if self.tokens is not None and use.tokens >= self.tokens:
            return StopReason.STOP_REASON_TOKEN_BUDGET_EXHAUSTED
        if self.cost_usd is not None and use.cost_usd >= self.cost_usd:
            return StopReason.STOP_REASON_COST_BUDGET_EXHAUSTED
        return None


class ErrorStreak:
    """Counts the model's failed replies in a row (spec §5.1); a reply that works ends it.

    The loop decides which replies failed: those none of whose calls gave a result that is no
    error. The reminder comes once a streak, at REMIND_AFTER_ERRORS, before the stop at
    STOP_AFTER_ERRORS.
    """

    def __init__(self) -> None:
        self.count = 0
        self._reminded = False

    def add(self, *, error: bool) -> None:
        """Count one more reply: one that failed lengthens the streak, any other ends it."""
        if error:
            self.count += 1
        else:
            self.count = 0
            self._reminded = False

    def take_reminder(self) -> bool:
        """Whether the model is due its reminder of the tool schemas: once a streak."""
        if self._reminded or self.count < REMIND_AFTER_ERRORS:
            return False
        self._reminded = True
        return True

    @property
    def exhausted(self) -> bool:
        """Whether the streak has reached the stop."""
        return self.count >= STOP_AFTER_ERRORS
