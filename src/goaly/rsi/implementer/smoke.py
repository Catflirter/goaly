"""``goaly-smoke``: the one command the host runs for the implementer's sandbox (spec §8.8).

The agent runs ``goaly-smoke <instance_id>`` to try its pack, as it stands, on one train task.
The sandbox never runs it: SandboxEnvironment hands it to a :class:`SmokeService` on the host,
which, for each call:

1. refuses any instance that is not in train, and any call after the limit (2 per hypothesis);
2. stops and removes the sandbox, so that nothing can change the worktree, then pins the packs as
   they stand to a commit no branch holds (Workspace.pin), and runs the step-3 guards on it;
3. if the guards pass, runs that commit on the one task: one seed, a solve container, then a
   host-side grade (spec §6.1), ahead of every phase on the grading instances (spec §6.2);
4. starts a fresh sandbox on the same worktree, whatever the outcome, so the agent continues
   where it was, with an empty ``/tmp``;
5. tells the agent the score, the stop reason, the step count, the cost and the last tool error,
   and records the call as a SmokeTest of the ImplementerRun.

The runner is a seam: T21 fills it with T11's two-phase pipeline.
"""

import logging
import shlex
import threading
from collections.abc import Callable, Collection
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Final, Protocol

from goaly.contracts.gen.experiment_pb2 import SmokeTest
from goaly.rsi.workspace import UnsafePackError, Workspace, Worktree

__all__ = [
    "SMOKE_COMMAND",
    "Pause",
    "SmokeOutcome",
    "SmokeRunner",
    "SmokeService",
    "is_smoke_call",
]

SMOKE_COMMAND: Final = "goaly-smoke"
_USAGE: Final = f"Run it on its own, as `{SMOKE_COMMAND} <instance_id>`, with one train task's ID."
_RESTARTED: Final = (
    "The sandbox was stopped for the smoke run and started afresh: /tmp is empty, and nothing "
    "you started in the background runs any more."
)

# Stops and removes the sandbox for the block, and starts a fresh one after it, whatever happens.
type Pause = Callable[[], AbstractContextManager[None]]

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

    Both are called from the implementer's worker thread, while the implementer's sandbox is
    stopped, and may block. The runner raises for an infrastructure failure, such as
    goaly.rsi.sandbox.SandboxError, which aborts the experiment; whatever the candidate does is an
    outcome.
    """

    def guard(self, commit: str) -> str | None:
        """Run the step-3 guards (spec §8.2) on ``commit``; return why one failed, or None."""

    def run(self, commit: str, instance_id: str, stop: threading.Event) -> SmokeOutcome:
        """Run ``commit`` on the train task ``instance_id``, with one seed, through the
        isolated two-phase path.

        Its grading must take goaly-smoke's priority on T10's grading scheduler
        (goaly.bench.effisql.grading_pool, Priority.SMOKE), which serves it ahead of every phase
        and makes it wait at most for the evaluation already running. Once ``stop`` is set, the
        run is no longer wanted: the runner should end it as soon as it can.
        """


def is_smoke_call(command: str) -> bool:
    """True if ``command`` runs goaly-smoke, its first word, so that the host answers it.

    Any other command, one that only mentions goaly-smoke included, runs in the sandbox, where no
    such program exists.
    """
    try:
        words = shlex.split(command)
    except ValueError:  # Unbalanced quotes: the sandbox's bash says what is wrong.
        return False
    return bool(words) and words[0] == SMOKE_COMMAND


class SmokeService:
    """Answers one implementer run's goaly-smoke calls, and keeps them as SmokeTests.

    ``stop`` is the run's stop event: once it is set, no smoke run starts, and a smoke run in
    progress is told to end.
    """

    def __init__(
        self,
        runner: SmokeRunner,
        workspace: Workspace,
        worktree: Worktree,
        *,
        train_ids: Collection[str],
        limit: int,
        stop: threading.Event,
    ) -> None:
        self._runner = runner
        self._workspace = workspace
        self._worktree = worktree
        self._train_ids = frozenset(train_ids)
        self._limit = limit
        self._stop = stop
        self._runs = 0
        self.tests: list[SmokeTest] = []

    def handle(self, command: str, pause: Pause) -> dict[str, Any]:
        """Answer a goaly-smoke call (see is_smoke_call), as a command's output for the agent.

        ``pause`` stops the sandbox for the pin and the smoke run, and starts a fresh one after
        them; a call that is refused never stops it.
        """
        words = shlex.split(command)
        if len(words) != 2:
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
        if self._stop.is_set():
            return self._refuse(instance_id, "the run is stopping")
        self._runs += 1
        with pause():
            return self._smoke(instance_id)

    def _smoke(self, instance_id: str) -> dict[str, Any]:
        """Pin, guard and run, while the sandbox is stopped."""
        try:
            commit = self._workspace.pin(self._worktree)
        except UnsafePackError as error:
            return self._refuse(instance_id, f"the pack cannot be run: {error}", restarted=True)
        failure = self._runner.guard(commit)
        if failure is not None:
            return self._refuse(instance_id, f"a guard failed: {failure}", restarted=True)
        if self._stop.is_set():
            return self._refuse(instance_id, "the run is stopping", restarted=True)
        outcome = self._runner.run(commit, instance_id, self._stop)
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
        text = f"{SMOKE_COMMAND} {instance_id}: score {score}; {detail}\n{_RESTARTED}\n"
        return _output(text, returncode=0)

    def _refuse(self, instance_id: str, reason: str, *, restarted: bool = False) -> dict[str, Any]:
        self.tests.append(SmokeTest(instance_id=instance_id, detail=f"refused: {reason}"))
        text = f"{SMOKE_COMMAND}: refused: {reason}.\n"
        return _output(text + (f"{_RESTARTED}\n" if restarted else ""), returncode=1)


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
