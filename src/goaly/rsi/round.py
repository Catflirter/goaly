"""One RSI round (spec §8.2), run by code alone: no step waits for a person (spec §8.0).

:meth:`RoundRunner.run` takes a round from its hypotheses to its decision:

1. **Hypotheses.** The round builds the first k of the lane's hypothesis queue. When fewer than
   k are queued it first asks the analyst (T19), whose 3 hypotheses join the back of the queue.
   An analysis that stays invalid skips the round if nothing is queued; a gateway failure
   (LlmError) is infrastructure, and the round is aborted.
2. **Implement** each one in its own worktree from the best, concurrently (T20).
3. **Guards** on the commit's snapshot, after the implementer's sandbox has stopped (T18).
4. **Screen** on up to 15 train failures the hypothesis targets plus 5 train tasks the best
   solves, 1 seed; 5. **validate** on every val task, 1 seed; 6. **decide**: the largest gain
   wins, and the others are rejected as outranked and queued again, to be rebuilt on the new
   best; 7. **confirm** the winner with 1 more seed, then move ``best/<lane>``.
8. **Record** every step in the ledger as it happens, and a lesson for each decided candidate.

Every evaluation goes through the :class:`Evaluator` seam: T11's isolated two-phase path on the
commit's snapshot. Rules come from goaly.rsi.decision.

**Outcomes.** An implementer that ends on a limit, a crash or no change rejects its candidate
(IMPLEMENTATION_FAILED), and so does a failed guard. An infrastructure failure, such as Docker,
the gateway, git or an evaluation that failed or came back incomplete, aborts the experiment
(DECISION_ABORTED) and queues its hypothesis again.

A winner that fails its confirmation leaves the best as it was. The candidates it outranked
were already queued again, and are rebuilt on the unchanged best next round, rather than
confirmed in turn: a design choice, since the spec is silent and a rebuild costs no val run.

**Resuming** (spec §8.7). The ledger holds each experiment's latest state, and a pending record's
steps so far all passed. Given the round's records, :meth:`RoundRunner.run` resumes each pending
experiment at its next step, and starts no new one, so no experiment is duplicated. The lane
first aborts the pending experiments that had no commit yet (goaly.rsi.resume).
"""

import asyncio
import enum
import functools
import hashlib
import json
import logging
import secrets
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final, Protocol

from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from goaly.bench.effisql.grading import Mode
from goaly.bench.effisql.grading_pool import Priority
from goaly.bench.effisql.metrics import (
    SCORED_STATUSES,
    GradeTable,
    Scoring,
    solved,
    task_credits,
)
from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    DECISION_PENDING,
    DECISION_REJECTED,
    IMPLEMENTER_EXIT_INFRASTRUCTURE,
    IMPLEMENTER_EXIT_INTERRUPTED,
    IMPLEMENTER_EXIT_SUBMITTED,
    REJECTION_REASON_GUARD_FAILED,
    REJECTION_REASON_IMPLEMENTATION_FAILED,
    REJECTION_REASON_OUTRANKED,
    DiffStats,
    ExperimentRecord,
    GuardResult,
    Hypothesis,
    ImplementerExit,
    ImplementerRun,
    Lane,
    RejectionReason,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose, Usage
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_ENVIRONMENT,
    TASK_ERROR_TYPE_LLM,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import (
    SPLIT_TRAIN,
    SPLIT_VAL,
    Split,
)
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.harness.core.llm import LlmClient
from goaly.harness.core.llm_errors import LlmError
from goaly.rsi import decision
from goaly.rsi.analyst import Analysis, AnalysisSkipped, AnalystInput, analyze
from goaly.rsi.decision import CostStats, DecisionConfig, Measured, Verdict
from goaly.rsi.git import run_git
from goaly.rsi.hypothesis_queue import HypothesisQueue
from goaly.rsi.lanes import lane_info
from goaly.rsi.ledger import Ledger
from goaly.rsi.lesson_writer import write_lesson
from goaly.rsi.lessons import LessonStore
from goaly.rsi.resume import (
    MAX_REQUEUES,
    UsageSource,
    abort_record,
    finish,
    merge_usage,
    timestamp,
)
from goaly.rsi.structured import StructuredOutputError
from goaly.rsi.workspace import Workspace, Worktree

__all__ = [
    "AnalystInputBuilder",
    "EvalPhase",
    "Evaluation",
    "EvaluationRequest",
    "Evaluator",
    "GuardRunner",
    "Implementation",
    "Implementer",
    "RoundConfig",
    "RoundDeps",
    "RoundOutcome",
    "RoundRunner",
    "lesson_id",
    "tasks_digest",
]

_logger = logging.getLogger(__name__)

# A task the infrastructure failed, by the container's own word, until the host counts them.
_INFRA_ERRORS = frozenset({TASK_ERROR_TYPE_LLM, TASK_ERROR_TYPE_ENVIRONMENT})

type _Key = tuple[str, Split.ValueType, tuple[str, ...], tuple[int, ...]]
_ACCEPT_RETRIES_S: Final = (1.0, 2.0, 4.0, 8.0, 16.0)
_REASON_CHARS: Final = 300  # How much of an infrastructure error an abort reason keeps.


class EvalPhase(enum.StrEnum):
    """Why an evaluation runs, which sets its grading's priority (spec §6.2)."""

    SMOKE = "smoke"  # goaly-smoke, for the implementer: graded ahead of every phase.
    SCREEN = "screen"
    VALIDATION = "validation"
    CONFIRMATION = "confirmation"
    REFERENCE = "reference"  # The best's or v0's results, which the evaluator's cache holds.

    @property
    def priority(self) -> Priority:
        """The grading pool's priority (T10): goaly-smoke first, every other phase as RSI."""
        return Priority.SMOKE if self is EvalPhase.SMOKE else Priority.RSI


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Evaluate the pack at ``commit`` on ``instance_ids`` of ``split``, in each of ``seeds``.

    Everything here is known on the host, and the evaluation takes its expectations from it,
    never from the solve container's files (ruling V27): the tasks and seeds it must grade,
    every one of them, the split, the grading mode and the snapshot's identity, its commit and
    tree.
    """

    phase: EvalPhase
    commit: str
    tree: str
    split: Split.ValueType
    instance_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    run_id: str
    mode: Mode = Mode.DEV  # RSI grades in dev mode; official mode is for the final runs.
    # The pack's tree at ``commit``: the harness identity, which the host run's harness hash
    # must match, as T06's planner computes it from the snapshot's pack directory.
    pack_tree: str = ""
    # sha256 of the split, the task IDs and the seeds: the expected set, which the host run's
    # plan must match exactly. The run's config hash and the dataset's task digests are the
    # host planner's, computed from the snapshot and the dataset, never from solve/ (V27).
    tasks_digest: str = ""

    @property
    def priority(self) -> Priority:
        return self.phase.priority


@dataclass(frozen=True, slots=True)
class Evaluation:
    """An evaluation's graded results, one per task and seed, its cost statistics, and the
    tasks and seeds the infrastructure failed.

    ``stats`` must be host-verified (V27, controller ruling): tokens and cost are the LLM
    proxy's own run totals divided by the tasks evaluated, and wall time is the host's clock
    around the solve segments, a cached task bringing the host numbers stored with it. The
    solve container's per-task numbers are informational only and never reach a decision.

    ``infra_failed`` names the (task, seed) pairs the infrastructure failed, as the host saw it:
    proxy records whose retries ran out, and environment failures. None until the host can name
    them; the round then takes the results' LLM and environment errors instead. The round drops
    a failed task from every version it compares, and aborts beyond ``max_infra_share``.
    """

    results: Sequence[TaskResult]
    stats: CostStats
    infra_failed: Collection[tuple[str, int]] | None = None


class Evaluator(Protocol):
    """T11's isolated two-phase evaluation (spec §6.1), behind one seam."""

    async def evaluate(self, request: EvaluationRequest) -> Evaluation:
        """Solve in the sandbox container from the commit's snapshot, grade on the host with
        the request's priority, and return every expected task and seed. Tasks the host-side
        cache holds are not solved again. Raise on an infrastructure failure.
        """

    def known_train(self, commit: str) -> tuple[Sequence[TaskResult], Sequence[Trajectory]]:
        """Return the graded train results the commit already has, and their trajectories."""


class Implementation(Protocol):
    """What an implementer run left (T20's Implementation)."""

    @property
    def run(self) -> ImplementerRun: ...
    @property
    def commit(self) -> str | None: ...


class Implementer(Protocol):
    """T20's implementer, on the lane's k-thread pool."""

    async def implement(self, hypothesis: Hypothesis, worktree: Worktree) -> Implementation: ...


# Runs T18's guards: (experiment_id, parent, candidate) -> results. Raises on Docker failure.
type GuardRunner = Callable[[str, str, str], list[GuardResult]]
# Builds the analyst's input from the best's train results and trajectories and the queue.
type AnalystInputBuilder = Callable[
    [Sequence[TaskResult], Sequence[Trajectory], Sequence[Hypothesis]], AnalystInput
]


class RoundConfig(BaseModel):
    """A round's sizes and seeds, from config (spec §8.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(default=3, ge=1)
    screen_failures: int = Field(default=15, ge=1)
    screen_solved: int = Field(default=5, ge=0)
    screen_seed: int = 0
    validation_seed: int = 0
    confirmation_seed: int = 1
    decision: DecisionConfig = DecisionConfig()
    # Above this share of tasks failed by the infrastructure, an evaluation aborts its experiment.
    max_infra_share: float = Field(default=0.2, ge=0, le=1)
    max_requeues: int = Field(default=MAX_REQUEUES, ge=0)

    @model_validator(mode="after")
    def _check_seeds(self) -> "RoundConfig":
        if self.validation_seed == self.confirmation_seed:
            raise ValueError("the confirmation needs a seed other than the validation's")
        return self

    @property
    def val_seeds(self) -> tuple[int, int]:
        """The seeds a version's val credit averages over: v0's and every best's."""
        return (self.validation_seed, self.confirmation_seed)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class RoundDeps:
    """What a round works with: the lane's stores, agents and evaluator."""

    lane: Lane.ValueType
    ledger: Ledger
    lessons: LessonStore
    queue: HypothesisQueue
    workspace: Workspace
    llm: LlmClient
    implementer: Implementer
    guards: GuardRunner
    evaluator: Evaluator
    analyst_input: AnalystInputBuilder
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    v0_commit: str
    usage: UsageSource = field(default=lambda _: {})
    clock: Callable[[], datetime] = _utc_now


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    """How a round ended."""

    round: int
    accepted: str | None  # The accepted experiment, if any.
    decided: int  # Experiments accepted or rejected; the rest were aborted.
    aborted: int
    skipped: str = ""  # Why the round built nothing, if it did not.
    infrastructure: bool = False  # Whether it built nothing because the infrastructure failed.


def lesson_id(experiment_id: str) -> str:
    """The ID of an experiment's lesson, so that a resumed lane never writes it twice."""
    return f"lesson-{experiment_id}"


class _AbortedError(Exception):
    """An experiment cannot go on for a reason that says nothing about its hypothesis."""


class _RoundAbortedError(Exception):
    """The round could not start for a reason that says nothing about any hypothesis."""


class RoundRunner:
    """Runs a lane's rounds, one :meth:`run` at a time."""

    def __init__(self, deps: RoundDeps, config: RoundConfig) -> None:
        self._deps = deps
        self._config = config
        self._info = lane_info(deps.lane)
        # The round's evaluations, by what they evaluate: the k candidates share the best's and
        # v0's references, and the confirmation reuses the validation's.
        self._evaluations: dict[_Key, asyncio.Future[Evaluation]] = {}

    async def run(self, round_no: int, *, resume: Sequence[ExperimentRecord] = ()) -> RoundOutcome:
        """Run round ``round_no``, or resume it from its experiments' latest records."""
        try:
            return await self._run(round_no, resume)
        finally:
            pending = list(self._evaluations.values())
            self._evaluations.clear()
            for evaluation in pending:
                evaluation.cancel()  # Nothing keeps evaluating after the round, as at a deadline.
            await asyncio.gather(*pending, return_exceptions=True)

    async def _run(self, round_no: int, resume: Sequence[ExperimentRecord]) -> RoundOutcome:
        best = await asyncio.to_thread(self._best)
        records = list(resume)
        if not records:
            try:
                records, skipped = await self._start(round_no, best)
            except _RoundAbortedError as error:
                _logger.warning("round %d aborted: %s", round_no, error)
                return RoundOutcome(round_no, None, 0, 0, str(error), infrastructure=True)
            if skipped:
                _logger.warning("round %d skipped: %s", round_no, skipped)
                return RoundOutcome(round_no, None, 0, 0, skipped)
        await asyncio.gather(*(self._through_validation(record) for record in records))
        await self._decide(records)
        for record in records:
            await self._lesson(record)
        accepted = [r.experiment_id for r in records if r.decision == DECISION_ACCEPTED]
        aborted = sum(record.decision == DECISION_ABORTED for record in records)
        decided = len(records) - aborted
        return RoundOutcome(round_no, accepted[0] if accepted else None, decided, aborted)

    # Step 1: hypotheses.

    async def _start(self, round_no: int, best: str) -> tuple[list[ExperimentRecord], str]:
        queue = self._deps.queue
        k = self._config.k
        analyst_usage = Usage()
        if len(await asyncio.to_thread(queue.pending)) < k:
            analysis = await self._analyze(round_no, best)
            analyst_usage = analysis.usage  # Charged to this round's experiments, if any.
            if isinstance(analysis, AnalysisSkipped):
                if not await asyncio.to_thread(queue.pending):
                    return [], analysis.reason
            else:
                await finish(functools.partial(queue.push, analysis.hypotheses))
        chosen = (await asyncio.to_thread(queue.pending))[:k]
        shares = _split_usage(analyst_usage, len(chosen))
        records = []
        for index, (hypothesis, share) in enumerate(zip(chosen, shares, strict=True), 1):
            record = ExperimentRecord(
                experiment_id=f"{self._info.name}-r{round_no}-{index}-{secrets.token_hex(3)}",
                lane=self._deps.lane,
                round=round_no,
                parent_version=best,
                hypothesis=hypothesis,
                decision=DECISION_PENDING,
                thresholds=self._config.decision.thresholds(),
                started_at=self._now(),
            )
            if share.input_tokens or share.output_tokens or share.HasField("cost_usd"):
                record.usage_by_purpose.add(
                    purpose=CallPurpose.CALL_PURPOSE_RSI_ANALYST, usage=share
                )
            await self._save(record)
            records.append(record)
        await finish(functools.partial(queue.remove, [h.hypothesis_id for h in chosen]))
        return records, ""

    async def _analyze(self, round_no: int, best: str) -> Analysis | AnalysisSkipped:
        """The round's analysis, or its skip. An LlmError aborts the round."""
        results, trajectories = await asyncio.to_thread(self._train_history, best)
        queued = await asyncio.to_thread(self._deps.queue.pending)
        analyst_input = self._deps.analyst_input(results, trajectories, queued)
        round_id = f"{self._info.name}-r{round_no}-analyst"
        try:
            analysis = await analyze(self._deps.llm, analyst_input, round_id=round_id)
        except LlmError as error:
            raise _RoundAbortedError(f"the analyst's call failed: {error}") from error
        return analysis

    # Steps 2-5, for one experiment.

    async def _through_validation(self, record: ExperimentRecord) -> None:
        best = record.parent_version  # The best when the round started, as it still is.
        try:
            if record.decision == DECISION_PENDING:
                await self._implement(record)
            if record.decision == DECISION_PENDING and not record.guards:
                await self._guard(record)
            if record.decision == DECISION_PENDING and not record.HasField("screening"):
                await self._screen(record, best)
            if record.decision == DECISION_PENDING and not record.HasField("validation"):
                await self._validate(record, best)
        except _AbortedError as error:
            await self._abort(record, str(error))
        except Exception as error:  # Docker, git or the evaluator: infrastructure.
            _logger.exception("experiment %s failed", record.experiment_id)
            await self._abort(record, _infrastructure(error))

    async def _implement(self, record: ExperimentRecord) -> None:
        if record.candidate_commit:
            return
        workspace = self._deps.workspace
        worktree = await asyncio.to_thread(workspace.create, self._deps.lane, record.experiment_id)
        implementation = await self._deps.implementer.implement(record.hypothesis, worktree)
        # The branch keeps the commit. A cancelled run leaves its worktree, which the lane
        # removes when it next starts, once no sandbox can still be using it.
        await asyncio.to_thread(workspace.remove, worktree)
        record.implementer.CopyFrom(implementation.run)
        exit_ = implementation.run.exit
        if exit_ in (IMPLEMENTER_EXIT_INFRASTRUCTURE, IMPLEMENTER_EXIT_INTERRUPTED):
            raise _AbortedError(f"the implementer ended as {_exit_name(exit_)}")
        if exit_ != IMPLEMENTER_EXIT_SUBMITTED or implementation.commit is None:
            await self._reject(
                record,
                [REJECTION_REASON_IMPLEMENTATION_FAILED],
                f"not implemented: the implementer ended as {_exit_name(exit_)}",
            )
            return
        record.candidate_commit = implementation.commit
        record.diff_stats.CopyFrom(await asyncio.to_thread(self._diff_stats, record))
        await self._save(record)

    async def _guard(self, record: ExperimentRecord) -> None:
        results = await asyncio.to_thread(
            self._deps.guards,
            record.experiment_id,
            record.parent_version,
            record.candidate_commit,
        )
        record.guards.extend(results)
        failed = [result for result in results if not result.passed]
        if failed or not results:
            await self._reject(record, [REJECTION_REASON_GUARD_FAILED], "a guard failed")
        else:
            await self._save(record)

    async def _screen(self, record: ExperimentRecord, best: str) -> None:
        tasks = await asyncio.to_thread(self._screen_tasks, record, best)
        if not tasks:  # Setup or infrastructure: v0's train results are missing.
            raise _AbortedError("no train results to screen on")
        seeds = (self._config.screen_seed,)
        theirs, v0, ours = await asyncio.gather(
            self._measure(
                EvalPhase.REFERENCE,
                best,
                SPLIT_TRAIN,
                tasks,
                seeds,
                f"{record.experiment_id}-screen-best",
            ),
            self._measure(
                EvalPhase.REFERENCE,
                self._deps.v0_commit,
                SPLIT_TRAIN,
                tasks,
                seeds,
                f"{record.experiment_id}-screen-v0",
            ),
            self._measure(
                EvalPhase.SCREEN,
                record.candidate_commit,
                SPLIT_TRAIN,
                tasks,
                seeds,
                f"{record.experiment_id}-screen",
            ),
        )
        (theirs_m, v0_m, ours_m), dropped = self._comparable(
            "screen", tasks, [theirs], [v0], [ours]
        )
        config = _config_of(record)
        verdict = decision.screen(SPLIT_TRAIN, theirs_m, ours_m, v0=v0_m, config=config)
        await self._apply(record, "screening", verdict, dropped)

    async def _validate(self, record: ExperimentRecord, best: str) -> None:
        theirs, v0, ours = await asyncio.gather(
            self._reference_val(best, record.experiment_id, "best"),
            self._reference_val(self._deps.v0_commit, record.experiment_id, "v0"),
            self._candidate_results(record, self._config.validation_seed, EvalPhase.VALIDATION),
        )
        (theirs_m, v0_m, ours_m), dropped = self._comparable(
            "validation", self._deps.val_ids, [theirs], [v0], [ours]
        )
        verdict = decision.validate(SPLIT_VAL, theirs_m, ours_m, v0=v0_m, config=_config_of(record))
        await self._apply(record, "validation", verdict, dropped)

    # Steps 6-7.

    async def _decide(self, records: Sequence[ExperimentRecord]) -> None:
        passing = {
            record.experiment_id: record
            for record in records
            if record.decision == DECISION_PENDING and record.HasField("validation")
        }
        if not passing:
            return
        winner, outranked = decision.rank(
            {name: record.validation.delta for name, record in passing.items()}
        )
        for name in outranked:
            loser = passing[name]
            # Queued first, so a crash before the rejection is saved loses no hypothesis.
            push = functools.partial(self._deps.queue.push, [loser.hypothesis])
            await finish(push)
            await self._reject(
                loser,
                [REJECTION_REASON_OUTRANKED],
                f"outranked by {winner}, whose validation gain is larger",
            )
        record = passing[winner]
        try:
            await self._confirm(record)
        except Exception as error:
            tag = await asyncio.to_thread(self._deps.workspace.best_commit, self._deps.lane)
            if tag == record.candidate_commit:
                # The tag moved, so the candidate is the best: never abort it. The lane fails,
                # and its restart records the acceptance (goaly.rsi.resume.reconcile_best).
                raise
            if isinstance(error, _AbortedError):
                await self._abort(record, str(error))
            else:  # Infrastructure, as in _through_validation.
                _logger.exception("experiment %s failed", winner)
                await self._abort(record, _infrastructure(error))

    async def _confirm(self, record: ExperimentRecord) -> None:
        best = record.parent_version
        if not record.HasField("confirmation"):
            seeds = self._config.val_seeds
            theirs, v0, first, second = await asyncio.gather(
                self._reference_val(best, record.experiment_id, "best"),
                self._reference_val(self._deps.v0_commit, record.experiment_id, "v0"),
                self._candidate_results(record, seeds[0], EvalPhase.VALIDATION),
                self._candidate_results(record, seeds[1], EvalPhase.CONFIRMATION),
            )
            (theirs_m, v0_m, ours_m), dropped = self._comparable(
                "confirmation", self._deps.val_ids, [theirs], [v0], [first, second]
            )
            verdict = decision.confirm(
                SPLIT_VAL,
                theirs_m,
                ours_m,
                v0=v0_m,
                config=_config_of(record),  # The thresholds recorded when the round began.
                validation=record.validation,
            )
            await self._apply(record, "confirmation", verdict, dropped)
            if record.decision != DECISION_PENDING:
                return
        # A pending record whose confirmation passed: the tag moves first, so after a crash the
        # lane finds the tag on this commit and records the acceptance (goaly.rsi.resume).
        # Run to the end even if the round is cancelled meanwhile, as at the deadline.
        await finish(lambda: _accept(self._deps, record))

    # Step 8.

    async def _lesson(self, record: ExperimentRecord) -> None:
        if record.decision not in (DECISION_ACCEPTED, DECISION_REJECTED):
            return
        name = lesson_id(record.experiment_id)
        if await asyncio.to_thread(self._deps.lessons.get, name) is not None:
            return
        try:
            reply = await write_lesson(self._deps.llm, record)
        except (LlmError, StructuredOutputError) as error:
            _logger.warning("no lesson for %s: %s", record.experiment_id, error)
            return
        await finish(
            functools.partial(
                self._deps.lessons.add,
                name,
                self._deps.lane,
                reply.value.lesson,
                [record.experiment_id],
            )
        )
        self._add_usage(record)
        await self._save(record)

    # Helpers.

    def _best(self) -> str:
        best = self._deps.workspace.best_commit(self._deps.lane)
        if best is None:
            raise RuntimeError(f"the lane has no {self._info.best_tag} tag")
        return best

    def _lineage(self, best: str) -> list[str]:
        """The best and the versions it grew from, newest first, ending with v0."""
        accepted = [
            record.candidate_commit
            for record in self._deps.ledger.latest_experiments(self._deps.lane).values()
            if record.decision == DECISION_ACCEPTED
        ]
        chain = [best, *reversed(accepted), self._deps.v0_commit]
        return list(dict.fromkeys(chain))

    def _train_history(self, best: str) -> tuple[list[TaskResult], list[Trajectory]]:
        """The best's train results, a task the best never ran taking its newest ancestor's."""
        train = set(self._deps.train_ids)
        results: dict[str, list[TaskResult]] = {}
        trajectories: list[Trajectory] = []
        for commit in self._lineage(best):
            known, paths = self._deps.evaluator.known_train(commit)
            new = {r.instance_id for r in known if r.instance_id in train} - results.keys()
            for result in known:
                if result.instance_id in new:
                    results.setdefault(result.instance_id, []).append(result)
            trajectories.extend(t for t in paths if t.instance_id in new)
        return [r for listed in results.values() for r in listed], trajectories

    def _screen_tasks(self, record: ExperimentRecord, best: str) -> tuple[str, ...]:
        """Up to 15 train failures the hypothesis targets, plus 5 train tasks the best solves."""
        results = _first_seed(self._train_history(best)[0])
        if not results:
            return ()
        table = GradeTable.from_results(results)
        credits = task_credits(table, scoring=Scoring.CHECKED)
        solved_tasks = solved(table, scoring=Scoring.CHECKED)
        targets = [
            task
            for task in dict.fromkeys(record.hypothesis.target_instance_ids)
            if task in credits and task not in solved_tasks
        ][: self._config.screen_failures]
        # The solved tasks are the best's own when it has run enough of them; otherwise its
        # ancestors' (the train history's fallback), which is harmless for scoring, since the
        # best is measured again on them as the reference. They are drawn by a hash of the
        # experiment, so a resume draws the same.
        own = {r.instance_id for r in self._deps.evaluator.known_train(best)[0]}
        if len(own & solved_tasks) >= self._config.screen_solved:
            solved_tasks = frozenset(own & solved_tasks)
        draw = sorted(solved_tasks, key=lambda task: _hash(record.experiment_id, task))
        return (*targets, *draw[: self._config.screen_solved])

    async def _measure(
        self,
        phase: EvalPhase,
        commit: str,
        split: Split.ValueType,
        tasks: Sequence[str],
        seeds: Sequence[int],
        run_id: str,
    ) -> Evaluation:
        return await self._evaluate(phase, commit, split, tasks, seeds, run_id)

    def _comparable(
        self, name: str, tasks: Sequence[str], *versions: Sequence[Evaluation]
    ) -> tuple[list[Measured], int]:
        """Measure each version, given as its evaluations, on the tasks the infrastructure
        failed in none of them, and return how many tasks were dropped. Beyond
        ``max_infra_share`` of the tasks, the experiment aborts.
        """
        failed = {task for version in versions for part in version for task in _failed(part)}
        if failed and len(failed) > self._config.max_infra_share * len(tasks):
            raise _AbortedError(
                f"the {name}: the infrastructure failed {len(failed)} of {len(tasks)} tasks"
            )
        measured = []
        for version in versions:
            results = [r for part in version for r in part.results if r.instance_id not in failed]
            stats = CostStats.combine([part.stats for part in version])
            measured.append(Measured.from_results(results, stats))
        return measured, len(failed)

    async def _evaluate(
        self,
        phase: EvalPhase,
        commit: str,
        split: Split.ValueType,
        tasks: Sequence[str],
        seeds: Sequence[int],
        run_id: str,
    ) -> Evaluation:
        key: _Key = (commit, split, tuple(tasks), tuple(seeds))
        shared = self._evaluations.get(key)
        if shared is None or (shared.done() and (shared.cancelled() or shared.exception())):
            # Kept before its first await, so a concurrent request for the same key shares it.
            shared = asyncio.ensure_future(self._request(phase, key, run_id))
            self._evaluations[key] = shared
        evaluation = await asyncio.shield(shared)  # One waiter's cancellation spares the rest.
        results = list(evaluation.results)
        expected = {(task, seed) for task in tasks for seed in seeds}
        got = [(result.instance_id, result.seed) for result in results]
        if len(got) != len(set(got)) or set(got) != expected:
            missing, extra = len(expected - set(got)), len(set(got) - expected)
            raise _AbortedError(
                f"the {phase} evaluation {run_id} returned the wrong results: {missing} missing,"
                f" {extra} unexpected, {len(got) - len(set(got))} repeated"
            )
        try:
            GradeTable.from_results(results)  # Every result graded and scored, once.
        except ValueError as error:
            raise _AbortedError(
                f"the {phase} evaluation {run_id} is not usable: {error}"
            ) from error
        return evaluation

    async def _reference_val(self, commit: str, experiment_id: str, who: str) -> Evaluation:
        run_id = f"{experiment_id}-val-{who}"
        return await self._measure(
            EvalPhase.REFERENCE,
            commit,
            SPLIT_VAL,
            self._deps.val_ids,
            self._config.val_seeds,
            run_id,
        )

    async def _candidate_results(
        self, record: ExperimentRecord, seed: int, phase: EvalPhase
    ) -> Evaluation:
        run_id = f"{record.experiment_id}-{phase}"
        return await self._evaluate(
            phase, record.candidate_commit, SPLIT_VAL, self._deps.val_ids, (seed,), run_id
        )

    async def _apply(
        self, record: ExperimentRecord, phase_field: str, verdict: Verdict, dropped: int = 0
    ) -> None:
        getattr(record, phase_field).CopyFrom(verdict.phase)
        text = decision.describe(phase_field, verdict)
        if dropped:
            text += f"; {dropped} tasks left out: the infrastructure failed them"
        record.decision_reason = "; ".join(filter(None, [record.decision_reason, text]))
        if verdict.passed:
            await self._save(record)
        else:
            await self._reject(record, verdict.reasons, "")

    async def _reject(
        self, record: ExperimentRecord, reasons: Sequence[RejectionReason.ValueType], text: str
    ) -> None:
        record.decision = DECISION_REJECTED
        record.rejection_reasons.extend(reasons)
        if text:
            record.decision_reason = "; ".join(filter(None, [record.decision_reason, text]))
        record.finished_at.CopyFrom(self._now())
        self._add_usage(record)
        await self._save(record)

    async def _abort(self, record: ExperimentRecord, reason: str) -> None:
        deps = self._deps
        usage = deps.usage(record.experiment_id)
        await finish(
            lambda: abort_record(
                deps.ledger,
                deps.queue,
                record,
                reason,
                now=deps.clock(),
                usage=usage,
                max_requeues=self._config.max_requeues,
            )
        )

    def _add_usage(self, record: ExperimentRecord) -> None:
        merge_usage(record, self._deps.usage(record.experiment_id))

    def _diff_stats(self, record: ExperimentRecord) -> DiffStats:
        output = run_git(
            ["diff", "--numstat", record.parent_version, record.candidate_commit],
            cwd=self._deps.workspace.repo,
        )
        stats = DiffStats()
        for line in output.splitlines():
            added, removed, _ = line.split("\t", 2)
            stats.files_changed += 1
            stats.insertions += int(added) if added.isdigit() else 0
            stats.deletions += int(removed) if removed.isdigit() else 0
        return stats

    async def _request(self, phase: EvalPhase, key: _Key, run_id: str) -> Evaluation:
        commit, split, tasks, seeds = key
        tree, pack_tree = await asyncio.to_thread(self._trees, commit)
        request = EvaluationRequest(
            phase,
            commit,
            tree,
            split,
            tasks,
            seeds,
            run_id,
            pack_tree=pack_tree,
            tasks_digest=tasks_digest(split, tasks, seeds),
        )
        return await self._deps.evaluator.evaluate(request)

    def _trees(self, commit: str) -> tuple[str, str]:
        """The commit's tree and its pack directory's tree."""
        pack = self._info.pack_dirs[0]
        trees = run_git(
            ["rev-parse", f"{commit}^{{tree}}", f"{commit}:{pack}"], cwd=self._deps.workspace.repo
        ).split()
        return trees[0], trees[1]

    async def _save(self, record: ExperimentRecord) -> None:
        # Every ledger write runs to its end, even if the round is cancelled meanwhile.
        await finish(lambda: self._deps.ledger.record_experiment(record))

    def _now(self) -> Timestamp:
        return timestamp(self._deps.clock())


def _accept(deps: RoundDeps, record: ExperimentRecord) -> None:
    """Move the best tag to the record's commit, then record it accepted; in one thread, so a
    cancelled round cannot stop it between the two.
    """
    deps.workspace.set_best(deps.lane, record.candidate_commit, expected=record.parent_version)
    record.decision = DECISION_ACCEPTED
    record.finished_at.CopyFrom(timestamp(deps.clock()))
    merge_usage(record, deps.usage(record.experiment_id))
    # The tag has moved: the candidate is the best, so the record must say so. A failed write is
    # retried with backoff; one still failing propagates, and reconcile_best records it later.
    for pause in (*_ACCEPT_RETRIES_S, None):
        try:
            deps.ledger.record_experiment(record)
            return
        except Exception:
            if pause is None:
                raise
            _logger.warning("retrying the acceptance of %s", record.experiment_id, exc_info=True)
            time.sleep(pause)


def tasks_digest(split: Split.ValueType, tasks: Sequence[str], seeds: Sequence[int]) -> str:
    """sha256 of the split, the tasks and the seeds an evaluation expects, as the request says."""
    text = json.dumps([Split.Name(split), sorted(tasks), sorted(seeds)], separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _split_usage(usage: Usage, parts: int) -> list[Usage]:
    """Split ``usage`` evenly, the token remainders going to the first part (ExperimentRecord)."""
    if parts == 0:
        return []
    shares = []
    for index in range(parts):
        share = Usage()
        for name in ("input_tokens", "output_tokens"):
            total = getattr(usage, name)
            setattr(share, name, total // parts + (total % parts if index == 0 else 0))
        if usage.HasField("reasoning_tokens"):
            total = usage.reasoning_tokens
            share.reasoning_tokens = total // parts + (total % parts if index == 0 else 0)
        if usage.HasField("cost_usd"):
            share.cost_usd = usage.cost_usd / parts
        shares.append(share)
    return shares


def _first_seed(results: Sequence[TaskResult]) -> list[TaskResult]:
    """One scored seed per task, the lowest, so that tasks run in different seeds, or by
    different versions, still form one table. A grade that says nothing is left out.
    """
    chosen: dict[str, TaskResult] = {}
    for result in sorted(results, key=lambda r: r.seed):
        if result.grade.status in SCORED_STATUSES:  # T07's: the statuses its tables score.
            chosen.setdefault(result.instance_id, result)
    return list(chosen.values())


def _failed(evaluation: Evaluation) -> set[str]:
    """The tasks the infrastructure failed: the host's account, or the results' own until then."""
    if evaluation.infra_failed is not None:
        return {task for task, _ in evaluation.infra_failed}
    return {r.instance_id for r in evaluation.results if r.error_type in _INFRA_ERRORS}


def _config_of(record: ExperimentRecord) -> DecisionConfig:
    """The decision thresholds the record carries, set when its round began."""
    return DecisionConfig.from_thresholds(record.thresholds)


def _infrastructure(error: BaseException) -> str:
    """An abort reason: the error's type and the start of its first line, never raw output."""
    text = str(error).strip()
    first = text.splitlines()[0] if text else ""
    return f"infrastructure failure: {type(error).__name__}: {first[:_REASON_CHARS]}"


def _hash(experiment_id: str, task: str) -> str:
    return hashlib.sha256(f"{experiment_id}\0{task}".encode()).hexdigest()


def _exit_name(exit_: ImplementerExit.ValueType) -> str:
    return ImplementerExit.Name(exit_)
