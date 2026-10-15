"""The agent loop's budgets (spec §5.1, §5.4): what a task may spend, and the error streak.

A pack's config.yaml sets the budgets, and v0 Effi-SQL allows 30 steps or 10 minutes. When one
runs out, the loop stops and the pack's fallback answers. :class:`ErrorStreak` counts the model's
errors in a row: the loop reminds the model of the tool schemas after 3, and stops after 5.
"""

from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, model_validator

from goaly.contracts.gen.trajectory_pb2 import StopReason

__all__ = ["BudgetUse", "Budgets", "ErrorStreak"]


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
    and the reported cost of every model call of the task; None means no cap. After
    ``remind_after_errors`` errors in a row the loop reminds the model of the tool schemas, and
    after ``stop_after_errors`` it stops. A reply's tool calls past ``calls_per_reply`` do not
    run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    steps: PositiveInt
    wall_clock_s: PositiveFloat
    tokens: PositiveInt | None = None
    cost_usd: PositiveFloat | None = None
    remind_after_errors: PositiveInt = 3
    stop_after_errors: PositiveInt = 5
    calls_per_reply: PositiveInt = 16

    @model_validator(mode="after")
    def _remind_before_stopping(self) -> Self:
        if self.remind_after_errors >= self.stop_after_errors:
            raise ValueError(
                f"remind_after_errors ({self.remind_after_errors}) must be below "
                f"stop_after_errors ({self.stop_after_errors})"
            )
        return self

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
    """Counts the model's errors in a row (spec §5.1); a call that works ends the streak.

    The loop decides what an error is: a call that failed or repeats the one before it, a reply
    that calls no tool, or a reply cut off at the output limit.
    """

    def __init__(self, budgets: Budgets) -> None:
        self._remind_after = budgets.remind_after_errors
        self._stop_after = budgets.stop_after_errors
        self.count = 0
        self._reminded = False

    def add(self, *, error: bool) -> None:
        """Count one more outcome: an error lengthens the streak, anything else ends it."""
        if error:
            self.count += 1
        else:
            self.count = 0
            self._reminded = False

    def take_reminder(self) -> bool:
        """Whether the model is due its reminder of the tool schemas: once a streak."""
        if self._reminded or self.count < self._remind_after:
            return False
        self._reminded = True
        return True

    @property
    def exhausted(self) -> bool:
        """Whether the streak has reached the stop."""
        return self.count >= self._stop_after
