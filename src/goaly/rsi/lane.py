"""An RSI lane: rounds back to back until a stopping rule holds (spec §8.0, §8.4, §8.7).

:meth:`Lane.run` is the whole of a lane process's work, and nothing in it waits for a person:

1. **Resume** (spec §8.7): remove leftover sandbox containers and worktrees, make the best tag and
   the ledger agree, and abort the experiments interrupted before their commit, whose hypotheses
   go back in the queue. An unfinished round resumes from its records; the round number and the
   rounds without an accepted change are recomputed from the ledger and the round log.
2. **Rounds**, one at a time. A round that accepts nothing counts toward the stopping rule if it
   decided a candidate or its analysis stayed invalid; a round the infrastructure stopped does
   not count, and the lane waits ``infrastructure_backoff_s`` before the next.
3. **Stop** (spec §8.4) after ``max_rounds_without_accept`` rounds in a row with no accepted
   change, or at the deadline, an absolute time from config that a restart never moves. At the
   deadline the running round is cancelled, its pending experiments recorded as aborted, and the
   best kept. Either way the lane writes its stop record (LANE_EVENT_KIND_STOPPED), calls the
   results-PR hook and then the notifier hook (T22 implements both), and returns 0.

A restart after the deadline stops at once: it aborts what is pending and runs the stop steps.
The hooks must be idempotent, as T22's are: the results PR is found by the ledger's PR number.
"""

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from goaly.contracts.gen.experiment_pb2 import (
    LANE_EVENT_KIND_STARTED,
    LANE_EVENT_KIND_STOPPED,
    ExperimentRecord,
    Lane,
    LaneEvent,
    LaneEventKind,
)
from goaly.rsi.lanes import lane_info
from goaly.rsi.locks import file_lock
from goaly.rsi.probe import ProbeReading
from goaly.rsi.resume import abort_experiments, reconcile_best, remove_worktrees
from goaly.rsi.round import RoundConfig, RoundDeps, RoundOutcome, RoundRunner

__all__ = [
    "DEADLINE",
    "NO_ACCEPTED_CHANGE",
    "LaneConfig",
    "LaneHooks",
    "LaneRunner",
    "LaneState",
    "RoundEntry",
    "RoundLog",
    "StopInfo",
]

DEADLINE: Final = "deadline"
NO_ACCEPTED_CHANGE: Final = "no accepted change"

_logger = logging.getLogger(__name__)


class LaneConfig(BaseModel):
    """A lane's config (spec §6.2, §8.2, §8.4). The deadline is the freeze time, absolute."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deadline: AwareDatetime
    max_rounds_without_accept: int = Field(default=4, ge=1)
    infrastructure_backoff_s: float = Field(default=120.0, ge=0)
    task_concurrency: int = Field(default=16, ge=1)  # Tasks the evaluator runs at once.
    implementer_threads: int = Field(default=3, ge=1)  # The implementer pool, k threads.
    hook_timeout_s: float = Field(default=300.0, gt=0)  # Each stop hook's limit (spec §8.0).
    # How long a stopping lane waits for its implementer threads before it exits anyway.
    shutdown_timeout_s: float = Field(default=60.0, ge=0)
    round: RoundConfig = RoundConfig()


@dataclass(frozen=True, slots=True)
class StopInfo:
    """Why and when a lane stopped, for the results-PR and notifier hooks."""

    lane: Lane.ValueType
    rule: str  # DEADLINE or NO_ACCEPTED_CHANGE.
    detail: str
    best: str | None  # The best commit the lane kept.
    stopped_at: datetime
    aborted: tuple[str, ...] = ()  # The experiments the stop aborted.


type Hook = Callable[[StopInfo], Awaitable[None]]


async def _nothing(_: StopInfo) -> None:
    return None


async def _after_nothing(_: RoundOutcome) -> None:
    return None


async def _no_probe() -> list[ProbeReading]:
    return []


@dataclass(frozen=True)
class LaneHooks:
    """What the lane calls besides its rounds (T22). A stopping lane calls the final results PR,
    then the notification; each must be idempotent. ``after_round`` follows every finished
    round, as for a results PR that is due. ``running`` is entered for the lane's whole life, as
    for its heartbeat. An exception in a hook is logged, never raised.
    """

    results_pr: Hook = _nothing
    notify: Hook = _nothing
    # The identity probes of the grading instances, once per round before its validations.
    identity_probe: Callable[[], Awaitable[Sequence[ProbeReading]]] = _no_probe
    after_round: Callable[[RoundOutcome], Awaitable[None]] = _after_nothing
    running: Callable[[], AbstractAsyncContextManager[None]] = contextlib.nullcontext
    remove_containers: Callable[[], object] = lambda: 0  # Leftover sandboxes, at a start.


class RoundEntry(BaseModel):
    """One line of the round log: a round started, or how it finished."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round: int = Field(ge=1)
    status: str  # "started" or "finished".
    accepted: str | None = None
    decided: int = 0
    aborted: int = 0
    skipped: str = ""
    infrastructure: bool = False
    probes: tuple[ProbeReading, ...] = ()  # A started round's identity probes (spec §6.2).
    at: AwareDatetime


@dataclass
class LaneState:
    """What the round log and the ledger say about a lane's rounds (spec §8.7)."""

    next_round: int
    unfinished: int | None  # A round that started and never finished.
    without_accept: int  # Rounds in a row, up to now, that accepted nothing and count.


class RoundLog:
    """``rounds-<lane>.jsonl`` in the ledger directory: when each round started and finished.

    It records what the experiments cannot: a round that built nothing, because its analysis
    stayed invalid or the infrastructure failed. Each line is appended and synced under a lock.
    """

    def __init__(self, directory: Path, lane: Lane.ValueType, *, lock_timeout_s: float = 60.0):
        name = lane_info(lane).name
        self.path = directory / f"rounds-{name}.jsonl"
        self._lock = directory / f".rounds-{name}.lock"
        self._lock_timeout_s = lock_timeout_s

    def entries(self) -> list[RoundEntry]:
        if not self.path.exists():
            return []
        with file_lock(self._lock, shared=True, timeout_s=self._lock_timeout_s):
            lines = self.path.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines:
            with contextlib.suppress(ValueError):  # A torn last line, after a crash.
                entries.append(RoundEntry.model_validate_json(line))
        return entries

    def append(self, entry: RoundEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with (
            file_lock(self._lock, timeout_s=self._lock_timeout_s),
            self.path.open("ab") as file,
        ):
            if file.tell() and not self._ends_with_newline():
                file.write(b"\n")  # Cut off a torn last line.
            file.write(entry.model_dump_json().encode() + b"\n")
            file.flush()
            os.fsync(file.fileno())

    def state(self, ledger_rounds: Sequence[int]) -> LaneState:
        """The lane's state, from the log and the rounds the ledger's experiments ran in."""
        entries = self.entries()  # One read, so that one state comes of it.
        started: dict[int, RoundEntry] = {}
        streak = 0
        for entry in entries:
            if entry.status == "started":
                started[entry.round] = entry
                continue
            started.pop(entry.round, None)
            if entry.accepted:
                streak = 0
            elif (entry.skipped and not entry.infrastructure) or entry.decided:
                streak += 1
        last = max([*ledger_rounds, *(e.round for e in entries)], default=0)
        unfinished = max(started) if started else None
        return LaneState(next_round=last + 1, unfinished=unfinished, without_accept=streak)

    def _ends_with_newline(self) -> bool:
        with self.path.open("rb") as file:
            file.seek(-1, os.SEEK_END)
            return file.read(1) == b"\n"


class LaneRunner:
    """Runs one lane until a stopping rule holds: :meth:`run`."""

    def __init__(
        self,
        deps: RoundDeps,
        config: LaneConfig,
        *,
        hooks: LaneHooks = LaneHooks(),  # noqa: B008  (frozen, so one default is safe)
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._deps = deps
        self._config = config
        self._hooks = hooks
        self._sleep = sleep
        self._log = RoundLog(deps.ledger.directory, deps.lane)
        self._rounds = RoundRunner(deps, config.round)

    async def run(self) -> int:
        """Run rounds until a rule stops the lane; return 0, its exit code when stopped by rule.

        Raises goaly.rsi.resume.LaneStateError if the ledger and the best tag disagree in a way
        no crash explains, and lets any other unexpected error through: the supervisor counts
        the lane as crashed and restarts it from the ledger.
        """
        async with self._hooks.running():
            return await self._run()

    async def _run(self) -> int:
        await self._resume()
        if self._past_deadline():
            return await self._stop(DEADLINE, "the deadline had passed when the lane started")
        await self._record_event(LANE_EVENT_KIND_STARTED, "", deadline=True)
        while True:
            if self._past_deadline():  # A round that ended past the deadline starts no other.
                return await self._stop(DEADLINE, "the deadline came as a round ended")
            state = await asyncio.to_thread(self._state)
            limit = self._config.max_rounds_without_accept
            if state.without_accept >= limit:
                detail = f"{state.without_accept} rounds in a row accepted no change"
                return await self._stop(NO_ACCEPTED_CHANGE, detail)
            probes = await self._probe() if state.unfinished is None else []
            round_no, resume = await asyncio.to_thread(self._next_round, state, probes)
            outcome = await self._until_deadline(self._rounds.run(round_no, resume=resume))
            if outcome is None:
                return await self._stop(DEADLINE, f"the deadline came during round {round_no}")
            await asyncio.to_thread(self._log.append, self._finished(outcome))
            _logger.info(
                "round %d finished: accepted %s, %d decided, %d aborted%s",
                round_no,
                outcome.accepted or "nothing",
                outcome.decided,
                outcome.aborted,
                f", skipped: {outcome.skipped}" if outcome.skipped else "",
            )
            try:
                await self._hooks.after_round(outcome)
            except Exception:
                _logger.exception("the after-round hook failed")
            if outcome.infrastructure or not (outcome.decided or outcome.skipped):
                # Nothing was learned: wait, so a long outage does not spin through rounds.
                remaining = (self._config.deadline - self._deps.clock()).total_seconds()
                await self._sleep(max(0.0, min(self._config.infrastructure_backoff_s, remaining)))
                if self._past_deadline():
                    return await self._stop(DEADLINE, "the deadline came during a backoff")

    # Resuming.

    async def _resume(self) -> None:
        deps = self._deps
        await asyncio.to_thread(self._hooks.remove_containers)
        await asyncio.to_thread(remove_worktrees, deps.workspace, deps.lane)
        await asyncio.to_thread(self._reconcile)
        await asyncio.to_thread(
            abort_experiments,
            deps.ledger,
            deps.queue,
            deps.lane,
            "interrupted before its commit; the lane restarted",
            now=deps.clock(),
            uncommitted_only=True,
            usage=deps.usage,
            max_requeues=self._config.round.max_requeues,
        )

    def _reconcile(self) -> str:
        deps = self._deps
        return reconcile_best(
            deps.ledger, deps.workspace, deps.lane, v0_commit=deps.v0_commit, now=deps.clock()
        )

    def _state(self) -> LaneState:
        rounds = [r.round for r in self._deps.ledger.latest_experiments(self._deps.lane).values()]
        return self._log.state(rounds)

    def _next_round(
        self, state: LaneState, probes: Sequence[ProbeReading]
    ) -> tuple[int, list[ExperimentRecord]]:
        if state.unfinished is not None:
            records = [
                record
                for record in self._deps.ledger.latest_experiments(self._deps.lane).values()
                if record.round == state.unfinished
            ]
            _logger.warning("resuming round %d with %d experiments", state.unfinished, len(records))
            return state.unfinished, records
        self._log.append(
            RoundEntry(
                round=state.next_round,
                status="started",
                probes=tuple(probes),
                at=self._deps.clock(),
            )
        )
        return state.next_round, []

    def _finished(self, outcome: RoundOutcome) -> RoundEntry:
        return RoundEntry(
            round=outcome.round,
            status="finished",
            accepted=outcome.accepted,
            decided=outcome.decided,
            aborted=outcome.aborted,
            skipped=outcome.skipped,
            infrastructure=outcome.infrastructure,
            at=self._deps.clock(),
        )

    async def _probe(self) -> list[ProbeReading]:
        """The round's identity probes (goaly.rsi.probe); a failing hook gives none."""
        try:
            return list(await self._until_deadline(self._hooks.identity_probe()) or [])
        except Exception:
            _logger.exception("the identity probes failed")
            return []

    # The deadline.

    def _past_deadline(self) -> bool:
        return self._deps.clock() >= self._config.deadline

    async def _until_deadline[T](self, work: Awaitable[T]) -> T | None:
        """Await ``work`` until the deadline; at the deadline cancel it and return None."""
        task = asyncio.ensure_future(work)
        try:
            remaining = (self._config.deadline - self._deps.clock()).total_seconds()
            done, _ = await asyncio.wait({task}, timeout=max(remaining, 0.0))
            if task in done:
                return task.result()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return None
        finally:
            if not task.done():  # The lane itself was cancelled: the work goes with it.
                task.cancel()

    # Stopping.

    async def _stop(self, rule: str, detail: str) -> int:
        deps = self._deps
        aborted: list[ExperimentRecord] = []
        # The stop record and the final results PR are the deliverables, so a failing step
        # before them is logged and the stop goes on.
        if rule == DEADLINE:
            try:
                # A round cancelled between moving the tag and recording it left the tag ahead.
                await asyncio.to_thread(self._reconcile)
                aborted = await asyncio.to_thread(
                    abort_experiments,
                    deps.ledger,
                    deps.queue,
                    deps.lane,
                    "the lane's deadline came first",
                    now=deps.clock(),
                    uncommitted_only=False,
                    usage=deps.usage,
                    max_requeues=self._config.round.max_requeues,
                )
            except Exception:
                _logger.exception("could not settle the experiments at the deadline")
        best = None
        try:
            best = await asyncio.to_thread(deps.workspace.best_commit, deps.lane)
            await self._record_event(LANE_EVENT_KIND_STOPPED, f"{rule}: {detail}")
        except Exception:
            _logger.exception("could not write the stop record")
        info = StopInfo(
            lane=deps.lane,
            rule=rule,
            detail=detail,
            best=best,
            stopped_at=deps.clock(),
            aborted=tuple(record.experiment_id for record in aborted),
        )
        _logger.warning("the lane stopped: %s: %s; the best is %s", rule, detail, best)
        for name, hook in (
            ("results PR", self._hooks.results_pr),
            ("notifier", self._hooks.notify),
        ):
            try:
                async with asyncio.timeout(self._config.hook_timeout_s):
                    await hook(info)
            except Exception:  # A timeout too: TimeoutError is an Exception.
                _logger.exception("the %s hook failed", name)
        return 0

    async def _record_event(
        self, kind: LaneEventKind.ValueType, detail: str, *, deadline: bool = False
    ) -> None:
        event = LaneEvent(lane=self._deps.lane, kind=kind, detail=detail)
        event.occurred_at.CopyFrom(_timestamp(self._deps.clock()))
        if deadline:
            event.deadline_at.CopyFrom(_timestamp(self._config.deadline))
        await asyncio.to_thread(self._deps.ledger.record_lane_event, event)


def _timestamp(moment: datetime) -> Timestamp:
    stamp = Timestamp()
    stamp.FromDatetime(moment)
    return stamp
