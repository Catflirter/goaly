"""``goaly-smoke``: the one command the host runs for the implementer's sandbox (spec §8.8).

The agent runs ``goaly-smoke <instance_id>`` to try its pack, as it stands, on one train task.
The sandbox never runs it: SandboxEnvironment hands it to a :class:`SmokeService` on the host,
which, for each call:

1. refuses any instance that is not in train, and any call after the limit (2 per hypothesis);
2. pins the pack as it stands into a commit (Workspace.pin), since a candidate is evaluated only
   from a commit;
3. runs the step-3 guards on that commit, through the runner;
4. runs the candidate on the one task, through the runner: one seed, a solve container, then a
   host-side grade (spec §6.1), ahead of every phase on the grading instances (spec §6.2);
5. tells the agent the score, the stop reason, the step count, the cost and the last tool error,
   and records the call as a SmokeTest of the ImplementerRun.

The runner is a seam: T21 fills it with T11's two-phase pipeline.
"""

import logging
import re
import shlex
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Final, Protocol

from goaly.contracts.gen.experiment_pb2 import SmokeTest
from goaly.rsi.workspace import UnsafePackError, Workspace, Worktree

__all__ = ["SMOKE_COMMAND", "SmokeOutcome", "SmokeRunner", "SmokeService", "mentions_smoke"]

SMOKE_COMMAND: Final = "goaly-smoke"
_MENTION: Final = re.compile(rf"(?<![\w-]){re.escape(SMOKE_COMMAND)}(?![\w-])")
_USAGE: Final = f"Run it on its own, as `{SMOKE_COMMAND} <instance_id>`, with one train task's ID."

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmokeOutcome:
    """How the candidate did on the one task."""

    # The smoke run that holds the task's result and trajectory.
    run_id: str
    # The task's score as its benchmark defines it, for Effi-SQL the credit; None if grading
    # failed.
    score: float | None
    stop_reason: str
    steps: int
    cost_usd: float
    # The last error a tool returned to the executor; empty if none did.
    last_tool_error: str


class SmokeRunner(Protocol):
    """Runs a pinned candidate's guards and its 1-task smoke run (T21 wires the real one).

    Both are called from the implementer's worker thread, and may block. The runner raises for an
    infrastructure failure, such as goaly.rsi.sandbox.SandboxError, which aborts the experiment;
    whatever the candidate does is an outcome.
    """

    def guard(self, commit: str) -> str | None:
        """Run the step-3 guards (spec §8.2) on ``commit``; return why one failed, or None."""

    def run(self, commit: str, instance_id: str) -> SmokeOutcome:
        """Run ``commit`` on the train task ``instance_id``, with one seed, through the
        isolated two-phase path; its grading must go ahead of every phase on the grading
        instances, waiting at most for the evaluation already running (GradingPriority.SMOKE).
        """


def mentions_smoke(command: str) -> bool:
    """True if ``command`` names goaly-smoke, so that the host, not the sandbox, answers it."""
    return _MENTION.search(command) is not None


class SmokeService:
    """Answers one implementer run's goaly-smoke calls, and keeps them as SmokeTests."""

    def __init__(
        self,
        runner: SmokeRunner,
        workspace: Workspace,
        worktree: Worktree,
        *,
        train_ids: Collection[str],
        limit: int,
    ) -> None:
        self._runner = runner
        self._workspace = workspace
        self._worktree = worktree
        self._train_ids = frozenset(train_ids)
        self._limit = limit
        self._runs = 0
        self.tests: list[SmokeTest] = []

    def handle(self, command: str) -> dict[str, Any]:
        """Answer a command that mentions goaly-smoke, as a command's output for the agent."""
        try:
            words = shlex.split(command)
        except ValueError:
            words = []
        if len(words) != 2 or words[0] != SMOKE_COMMAND:
            return _output(f"{SMOKE_COMMAND}: not run. {_USAGE}", returncode=2)
        instance_id = words[1]
        if instance_id not in self._train_ids:
            return self._refuse(
                instance_id,
                f"{instance_id!r} is not a train task; goaly-smoke runs train tasks only",
            )
        if self._runs >= self._limit:
            return self._refuse(
                instance_id, f"the limit of {self._limit} smoke runs per hypothesis is reached"
            )
        self._runs += 1
        try:
            commit = self._workspace.pin(self._worktree)
        except UnsafePackError as error:
            return self._refuse(instance_id, f"the pack cannot be run: {error}")
        failure = self._runner.guard(commit)
        if failure is not None:
            return self._refuse(instance_id, f"a guard failed: {failure}")
        outcome = self._runner.run(commit, instance_id)
        detail = _detail(outcome)
        test = SmokeTest(instance_id=instance_id, run_id=outcome.run_id, detail=detail)
        if outcome.score is not None:
            test.score = outcome.score
        self.tests.append(test)
        score = "none (grading failed)" if outcome.score is None else f"{outcome.score:g}"
        _logger.info(
            "ran a smoke test",
            extra={
                "experiment_id": self._worktree.experiment_id,
                "instance_id": instance_id,
                "run_id": outcome.run_id,
                "score": outcome.score,
            },
        )
        return _output(f"{SMOKE_COMMAND} {instance_id}: score {score}; {detail}\n", returncode=0)

    def _refuse(self, instance_id: str, reason: str) -> dict[str, Any]:
        self.tests.append(SmokeTest(instance_id=instance_id, detail=f"refused: {reason}"))
        return _output(f"{SMOKE_COMMAND}: refused: {reason}.\n", returncode=1)


def _detail(outcome: SmokeOutcome) -> str:
    parts = [
        f"stop reason {outcome.stop_reason}",
        f"{outcome.steps} steps",
        f"cost ${outcome.cost_usd:.4f}",
        f"last tool error: {outcome.last_tool_error or 'none'}",
    ]
    return ", ".join(parts)


def _output(text: str, *, returncode: int) -> dict[str, Any]:
    return {"output": text, "returncode": returncode, "exception_info": ""}
