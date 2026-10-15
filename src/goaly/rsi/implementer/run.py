"""One implementer run: a hypothesis in, at most one commit out (spec §8.2 step 2, §8.8).

:meth:`Implementer.implement` runs mini-swe-agent's agent on the lane's implementer thread pool,
never on the event loop's thread. In that thread it:

1. starts the sandbox on the experiment's worktree and runs the agent in it, the trajectory
   saved after every step to ``runs/rsi/<lane>/<experiment_id>/implementer.traj.json``;
2. removes the sandbox, so nothing can change the worktree any more;
3. maps how the run ended to an ImplementerExit;
4. if the agent submitted, commits its change once through T18's commit helper, with the
   Experiment-Id and Hypothesis-Id trailers; a submission that changed nothing is NO_CHANGES.

Only IMPLEMENTER_EXIT_SUBMITTED comes with a commit. A limit, repeated format errors or a crash
reject the candidate; a gateway or Docker failure (INFRASTRUCTURE) and a stopped run
(INTERRUPTED) abort the experiment instead. Errors of the gateway client after its retries, of
Docker, and of git on the host are all caught and mapped; implement() raises only
asyncio.CancelledError, after it has told the run to stop.
"""

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable, Collection, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from goaly.contracts.gen.experiment_pb2 import (
    ChangeKind,
    Hypothesis,
    ImplementerExit,
    ImplementerRun,
)
from goaly.harness.core.llm_errors import (
    ClientError,
    ConnectionFailed,
    ContextOverflow,
    InvalidResponse,
    LlmError,
    RateLimited,
    ServerError,
    TimedOut,
)
from goaly.rsi.commit import OutsidePackError, commit_candidate
from goaly.rsi.git import GitError
from goaly.rsi.guards.allowlist import ALLOWED_CORE_MODULES, ALLOWED_STDLIB
from goaly.rsi.implementer import msa
from goaly.rsi.implementer.config import ImplementerConfig, load_config
from goaly.rsi.implementer.environment import HostCommand, SandboxEnvironment
from goaly.rsi.implementer.model import (
    EXIT_COST_LIMIT,
    EXIT_INTERRUPTED,
    EXIT_STEP_LIMIT,
    EXIT_TIME_LIMIT,
    GlmModel,
    SyncLlm,
)
from goaly.rsi.implementer.smoke import SmokeRunner, SmokeService
from goaly.rsi.lanes import lane_info
from goaly.rsi.sandbox import SandboxConfig, SandboxError
from goaly.rsi.workspace import UnsafePackError, Workspace, WorkspaceError, Worktree

__all__ = [
    "TRAJECTORY_NAME",
    "EnvironmentFactory",
    "Implementation",
    "Implementer",
    "ModelFactory",
    "exit_for_error",
    "exit_for_status",
    "trajectory_path",
]

TRAJECTORY_NAME: Final = "implementer.traj.json"

_E = ImplementerExit
# How the agent's exit message names the ending, for each ending that is not an error.
_EXITS_BY_STATUS: Final[Mapping[str, ImplementerExit.ValueType]] = {
    "Submitted": _E.IMPLEMENTER_EXIT_SUBMITTED,
    EXIT_STEP_LIMIT: _E.IMPLEMENTER_EXIT_STEP_LIMIT,
    EXIT_COST_LIMIT: _E.IMPLEMENTER_EXIT_COST_LIMIT,
    EXIT_TIME_LIMIT: _E.IMPLEMENTER_EXIT_TIME_LIMIT,
    "RepeatedFormatError": _E.IMPLEMENTER_EXIT_FORMAT_ERRORS,
    EXIT_INTERRUPTED: _E.IMPLEMENTER_EXIT_INTERRUPTED,
}
# The gateway failed after the client's retries: nothing to do with the hypothesis (T02a-5).
# Docker, and goaly's own worktree and git operations, fail the same way.
_INFRASTRUCTURE_ERRORS: Final = (
    *(RateLimited, ServerError, TimedOut, ConnectionFailed),
    *(SandboxError, WorkspaceError, GitError),
)
# The run's own request failed: its context overflowed, or the gateway refused or garbled it.
_CRASH_ERRORS: Final = (ContextOverflow, InvalidResponse, ClientError)

_logger = logging.getLogger(__name__)

# Starts the sandbox for a worktree, with goaly-smoke's host side if it is on; leaving the context
# removes it, and must wait until it is gone.
type EnvironmentFactory = Callable[
    [Worktree, HostCommand | None], AbstractContextManager[msa.Environment]
]
type ModelFactory = Callable[[str, threading.Event], msa.Model]


@dataclass(frozen=True)
class Implementation:
    """What an implementer run left: its record, and its commit if it made one."""

    run: ImplementerRun
    # The candidate commit, on the experiment's branch; None unless run.exit is SUBMITTED.
    commit: str | None


def trajectory_path(worktree: Worktree) -> PurePosixPath:
    """The run's trajectory, relative to the runs directory, as ImplementerRun records it."""
    name = lane_info(worktree.lane).name
    return PurePosixPath("rsi", name, worktree.experiment_id, TRAJECTORY_NAME)


def exit_for_status(status: str) -> ImplementerExit.ValueType:
    """Map the exit status of the agent's last message to an ImplementerExit; CRASHED if unknown.

    SUBMITTED means only that the agent finished: whether it changed anything shows at the commit.
    """
    return _EXITS_BY_STATUS.get(status, _E.IMPLEMENTER_EXIT_CRASHED)


def exit_for_error(error: BaseException) -> ImplementerExit.ValueType:
    """Map an error that ended a run to an ImplementerExit (T02a-5's mapping).

    The gateway after its retries, Docker, and goaly's own worktree or git operations are
    INFRASTRUCTURE; a call cancelled by the client's loop is INTERRUPTED; anything else, such as
    a context overflow or a request the gateway refused, is CRASHED.
    """
    if isinstance(error, _INFRASTRUCTURE_ERRORS):
        return _E.IMPLEMENTER_EXIT_INFRASTRUCTURE
    if isinstance(error, concurrent.futures.CancelledError):
        return _E.IMPLEMENTER_EXIT_INTERRUPTED
    return _E.IMPLEMENTER_EXIT_CRASHED


class Implementer:
    """Builds candidates for one lane process: one :meth:`implement` per hypothesis.

    ``llm`` is the lane's LlmClient, on the lane's event loop. ``pool`` is the lane's implementer
    thread pool, of k threads (spec §6.2); runs never use the loop's default executor, which file
    and git work share. ``image`` is the sandbox image (goaly.rsi.sandbox.ensure_image), and
    ``runs_dir`` the runs directory. With ``smoke``, the agent may run goaly-smoke on the tasks of
    ``train_ids``, at most ``limits.smoke_limit`` times; without it, goaly-smoke is off and the
    prompt does not offer it. ``make_environment`` and ``make_model`` replace the sandbox and
    GlmModel, for tests and dry runs.
    """

    def __init__(
        self,
        llm: SyncLlm,
        workspace: Workspace,
        *,
        image: str,
        runs_dir: Path,
        pool: concurrent.futures.ThreadPoolExecutor,
        config: ImplementerConfig | None = None,
        sandbox: SandboxConfig = SandboxConfig(),  # noqa: B008  (frozen, so one default is safe)
        make_environment: EnvironmentFactory | None = None,
        make_model: ModelFactory | None = None,
        smoke: SmokeRunner | None = None,
        train_ids: Collection[str] = (),
    ) -> None:
        self._llm = llm
        self._smoke = smoke
        self._train_ids = frozenset(train_ids)
        self._workspace = workspace
        self._image = image
        self._runs_dir = runs_dir
        self._pool = pool
        self._config = load_config() if config is None else config
        self._sandbox = sandbox
        self._make_environment = make_environment or self._sandbox_environment
        self._make_model = make_model or self._glm_model

    async def implement(self, hypothesis: Hypothesis, worktree: Worktree) -> Implementation:
        """Build ``hypothesis`` in the experiment's ``worktree``, on the implementer pool.

        Cancelling it tells the run to stop at its next step, where it ends as INTERRUPTED and
        commits nothing; the run's thread then removes its sandbox on its own.
        """
        stop = threading.Event()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._pool, self.run, hypothesis, worktree, stop)
        except asyncio.CancelledError:
            stop.set()
            raise

    def run(
        self, hypothesis: Hypothesis, worktree: Worktree, stop: threading.Event
    ) -> Implementation:
        """Run the agent and commit its change, in the calling thread; implement() calls it on
        the pool. It never raises for how the run ends: the result records that.
        """
        started = time.monotonic()
        relative = trajectory_path(worktree)
        output_path = self._runs_dir / relative
        record = ImplementerRun()
        commit = None
        agent: msa.DefaultAgent | None = None
        smoke = self._smoke_service(worktree)
        try:
            host = None if smoke is None else smoke.handle
            with self._make_environment(worktree, host) as environment:
                model = self._make_model(worktree.experiment_id, stop)
                agent = msa.make_agent(
                    model,
                    environment,
                    system_template=self._config.agent.system_template,
                    instance_template=self._config.agent.instance_template,
                    max_consecutive_format_errors=self._config.limits.max_consecutive_format_errors,
                    output_path=output_path,
                )
                result = agent.run(hypothesis.change, **self._template_vars(hypothesis, worktree))
            # The sandbox is gone: nothing changes the worktree any more.
            record.exit = exit_for_status(result.get("exit_status", ""))
            if record.exit == _E.IMPLEMENTER_EXIT_SUBMITTED:
                record.summary = result.get("submission", "").strip()
                if stop.is_set():
                    record.exit = _E.IMPLEMENTER_EXIT_INTERRUPTED
                else:
                    commit = commit_candidate(
                        self._workspace,
                        worktree,
                        hypothesis_id=hypothesis.hypothesis_id,
                        summary=record.summary,
                    )
                    if commit is None:
                        record.exit = _E.IMPLEMENTER_EXIT_NO_CHANGES
        except (OutsidePackError, UnsafePackError) as error:
            # The sandbox should make both impossible; the candidate is rejected all the same.
            record.exit = _E.IMPLEMENTER_EXIT_CRASHED
            _log_error(worktree, error)
        except Exception as error:  # noqa: BLE001  (every ending is mapped and recorded)
            record.exit = exit_for_error(error)
            _log_error(worktree, error)
        if smoke is not None:
            record.smoke_tests.extend(smoke.tests)
        if agent is not None:
            record.step_count = _steps(agent.messages)
            record.trajectory_path = str(relative) if output_path.exists() else ""
        record.wall_time_s = time.monotonic() - started
        _logger.info(
            "the implementer finished",
            extra={
                "experiment_id": worktree.experiment_id,
                "hypothesis_id": hypothesis.hypothesis_id,
                "exit": ImplementerExit.Name(record.exit),
                "steps": record.step_count,
                "commit": commit,
            },
        )
        return Implementation(record, commit)

    def _template_vars(self, hypothesis: Hypothesis, worktree: Worktree) -> dict[str, Any]:
        return {
            **self._config.limits.model_dump(),
            "hypothesis": {
                "hypothesis_id": hypothesis.hypothesis_id,
                "failure_pattern": hypothesis.failure_pattern,
                "target_instance_ids": list(hypothesis.target_instance_ids),
                "change_kind": ChangeKind.Name(hypothesis.change_kind),
                "change": hypothesis.change,
                "expected_effect": hypothesis.expected_effect,
                "risk": hypothesis.risk,
            },
            "lane": lane_info(worktree.lane).title,
            "worktree": str(worktree.path),
            "pack_dirs": [str(path) for path in lane_info(worktree.lane).pack_dirs],
            "allowed_stdlib": sorted(ALLOWED_STDLIB),
            "allowed_core_modules": sorted(ALLOWED_CORE_MODULES),
            "smoke": self._smoke is not None and self._config.limits.smoke_limit > 0,
        }

    def _smoke_service(self, worktree: Worktree) -> SmokeService | None:
        """goaly-smoke's host side for one run, or None if goaly-smoke is off."""
        if self._smoke is None or self._config.limits.smoke_limit == 0:
            return None
        return SmokeService(
            self._smoke,
            self._workspace,
            worktree,
            train_ids=self._train_ids,
            limit=self._config.limits.smoke_limit,
        )

    def _sandbox_environment(
        self, worktree: Worktree, smoke: HostCommand | None
    ) -> SandboxEnvironment:
        return SandboxEnvironment(
            image=self._image,
            worktree=worktree,
            command_timeout_s=self._config.limits.command_timeout_s,
            sandbox=self._sandbox,
            smoke=smoke,
        )

    def _glm_model(self, experiment_id: str, stop: threading.Event) -> GlmModel:
        return GlmModel(
            self._llm,
            experiment_id=experiment_id,
            limits=self._config.limits,
            templates=self._config.model,
            stop=stop,
        )


def _steps(messages: list[dict[str, Any]]) -> int:
    """How many model replies the run got: its steps, with or without a valid bash call."""
    return sum(
        1
        for message in messages
        if "actions" in message.get("extra", {})
        or message.get("extra", {}).get("interrupt_type") == "FormatError"
    )


def _log_error(worktree: Worktree, error: BaseException) -> None:
    level = logging.WARNING if isinstance(error, LlmError | SandboxError) else logging.ERROR
    _logger.log(
        level,
        "the implementer's run failed",
        exc_info=error,
        extra={"experiment_id": worktree.experiment_id, "error": type(error).__name__},
    )
