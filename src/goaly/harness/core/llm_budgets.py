"""The LLM concurrency budgets, and each process's share of them (spec §6.2).

A model's budget is how many of its requests may be in flight across all of our processes. Each
phase of the schedule, such as the baseline, the RSI run and the final evaluation, divides the
budgets among the processes that run at the same time. A process enforces its share with one
semaphore per model in the LLM client, whose adaptive limiting may lower it but never raises it.

A process gets exactly the shares its phase gives it, and nothing falls back to a model's whole
budget: a process the phase does not name has no share, and a model its phase gives it no share
of is one it must not call.

The budgets are config, read from a TOML file, so changing them needs no code change. The file
lives outside harness/core/, so changing it leaves the harness hash and the evaluation cache as
they are. A share is not in the run's config hash either: it limits how fast a run goes, not
what it finds, so an interrupted run may resume under another share.
"""

import tomllib
from collections import Counter
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError, model_validator

__all__ = ["BudgetsError", "LlmBudgets", "load_budgets"]


class BudgetsError(ValueError):
    """The budgets file is missing or invalid, or gives a process no share."""


class LlmBudgets(BaseModel):
    """The budgets and the phases' shares of them, as the budgets file holds them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Requests in flight per model, across all of our processes.
    budgets: dict[str, PositiveInt]
    # For each phase, each process's share of each model's budget.
    phases: dict[str, dict[str, dict[str, PositiveInt]]]

    @model_validator(mode="after")
    def _shares_fit_the_budgets(self) -> Self:
        for phase, processes in self.phases.items():
            given: Counter[str] = Counter()
            for process, shares in processes.items():
                for model, share in shares.items():
                    if model not in self.budgets:
                        raise ValueError(
                            f"phase {phase!r} gives {process!r} a share of {model!r},"
                            " which has no budget"
                        )
                    given[model] += share
            for model, total in given.items():
                if total > self.budgets[model]:
                    raise ValueError(
                        f"phase {phase!r} gives out {total} of {model!r},"
                        f" whose budget is {self.budgets[model]}"
                    )
        return self

    def share(self, phase: str, process: str) -> dict[str, int]:
        """Return the share of each model's budget that ``phase`` gives ``process``.

        Raise BudgetsError if there is no such phase, or the phase gives the process no share.
        """
        if phase not in self.phases:
            raise BudgetsError(
                f"no phase {phase!r}; the phases are: {', '.join(sorted(self.phases))}"
            )
        processes = self.phases[phase]
        if process not in processes:
            raise BudgetsError(
                f"phase {phase!r} gives no share to {process!r}; it gives shares to:"
                f" {', '.join(sorted(processes))}"
            )
        return dict(processes[process])


def load_budgets(path: Path) -> LlmBudgets:
    """Read and check the budgets file at ``path``; raise BudgetsError, naming it, if invalid."""
    try:
        with path.open("rb") as file:
            return LlmBudgets.model_validate(tomllib.load(file))
    except OSError as error:
        raise BudgetsError(f"cannot read the budgets file {path}: {error}") from error
    except (tomllib.TOMLDecodeError, ValidationError) as error:
        raise BudgetsError(f"the budgets file {path} is invalid: {error}") from error
