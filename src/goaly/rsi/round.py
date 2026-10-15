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

**Resuming** (spec §8.7). The ledger holds each experiment's latest state, and a pending record's
steps so far all passed. Given the round's records, :meth:`RoundRunner.run` resumes each pending
experiment at its next step, and starts no new one, so no experiment is duplicated. The lane
first aborts the pending experiments that had no commit yet (goaly.rsi.resume).
"""

import asyncio
import enum
import hashlib
import logging
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from goaly.bench.effisql.grading import Mode
from goaly.bench.effisql.grading_pool import Priority
from goaly.bench.effisql.metrics import GradeTable, Scoring, solved, task_credits
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
    REJECTION_REASON_NO_SCREEN_GAIN,
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
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_PREDICTION_FAILED,
    GRADE_STATUS_TIMEOUT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    Split,
)
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.harness.core.llm import LlmClient
from goaly.harness.core.llm_errors import LlmError
from goaly.rsi import decision
from goaly.rsi.analyst import Analysis, AnalystInput, analyze
from goaly.rsi.decision import CostStats, DecisionConfig, Measured, Verdict
from goaly.rsi.git import run_git
from goaly.rsi.hypothesis_queue import HypothesisQueue
from goaly.rsi.lanes import lane_info
from goaly.rsi.ledger import Ledger
from goaly.rsi.lesson_writer import write_lesson
from goaly.rsi.lessons import LessonStore
from goaly.rsi.resume import UsageSource, abort_record, merge_usage
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
]

_logger = logging.getLogger(__name__)

_SCORED = frozenset({GRADE_STATUS_GRADED, GRADE_STATUS_PREDICTION_FAILED, GRADE_STATUS_TIMEOUT})


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

    @property
    def priority(self) -> Priority:
        return self.phase.priority


@dataclass(frozen=True, slots=True)
class Evaluation:
    """An evaluation's graded results, one per task and seed, and its cost statistics, which
    come from the LLM proxy's records and the host's clock (V27), cached tasks included.

    The host attributes each proxy record to its request's (instance_id, seed) tag, and the
    cache entry and the graded results.jsonl line keep the attributed usage and host wall
    time, so a cached task brings its original numbers. The tags are the container's, but the
    run total is the proxy's own: the evaluator flags a sum that differs from it by over 1%.
    Numbers it cannot give are left out, and the rules that need them are skipped (decision).
    """

    results: Sequence[TaskResult]
    stats: CostStats


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


class RoundRunner:
    """Runs a lane's rounds, one :meth:`run` at a time."""

    def __init__(self, deps: RoundDeps, config: RoundConfig) -> None:
        self._deps = deps
        self._config = config
        self._info = lane_info(deps.lane)

    async def run(self, round_no: int, *, resume: Sequence[ExperimentRecord] = ()) -> RoundOutcome:
        """Run round ``round_no``, or resume it from its experiments' latest records."""
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
            if isinstance(analysis, str):
                if not await asyncio.to_thread(queue.pending):
                    return [], analysis
            else:
                analyst_usage = analysis.usage
                await asyncio.to_thread(queue.push, analysis.hypotheses)
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
        await asyncio.to_thread(queue.remove, [h.hypothesis_id for h in chosen])
        return records, ""

    async def _analyze(self, round_no: int, best: str) -> Analysis | str:
        """The round's analysis, or why it was skipped. An LlmError aborts the round."""
        results, trajectories = await asyncio.to_thread(self._train_history, best)
        queued = await asyncio.to_thread(self._deps.queue.pending)
        analyst_input = self._deps.analyst_input(results, trajectories, queued)
        round_id = f"{self._info.name}-r{round_no}-analyst"
        try:
            analysis = await analyze(self._deps.llm, analyst_input, round_id=round_id)
        except LlmError as error:
            raise _RoundAbortedError(f"the analyst's call failed: {error}") from error
        return analysis if isinstance(analysis, Analysis) else analysis.reason

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
            await self._abort(record, f"infrastructure failure: {type(error).__name__}: {error}")

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
        if not tasks:
            await self._reject(
                record, [REJECTION_REASON_NO_SCREEN_GAIN], "screen: no train task to screen on"
            )
            return
        seeds = (self._config.screen_seed,)
        theirs, ours = await asyncio.gather(
            self._measure(
                EvalPhase.REFERENCE,
                best,
                SPLIT_TRAIN,
                tasks,
                seeds,
                f"{record.experiment_id}-screen-best",
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
        await self._apply(record, "screening", decision.screen(SPLIT_TRAIN, theirs, ours))

    async def _validate(self, record: ExperimentRecord, best: str) -> None:
        theirs, v0, ours = await asyncio.gather(
            self._reference_val(best, record.experiment_id, "best"),
            self._reference_val(self._deps.v0_commit, record.experiment_id, "v0"),
            self._candidate_val(record, self._config.validation_seed, EvalPhase.VALIDATION),
        )
        verdict = decision.validate(SPLIT_VAL, theirs, ours, v0=v0, config=self._config.decision)
        await self._apply(record, "validation", verdict)

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
            await self._reject(
                loser,
                [REJECTION_REASON_OUTRANKED],
                f"outranked by {winner}, whose validation gain is larger",
            )
            await asyncio.to_thread(self._deps.queue.push, [loser.hypothesis])
        try:
            await self._confirm(passing[winner])
        except _AbortedError as error:
            await self._abort(passing[winner], str(error))
        except Exception as error:  # Infrastructure, as in _through_validation.
            _logger.exception("experiment %s failed", winner)
            await self._abort(
                passing[winner], f"infrastructure failure: {type(error).__name__}: {error}"
            )

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
            ours = Measured.from_results(
                [*first.results, *second.results], CostStats.combine([first.stats, second.stats])
            )
            guardrail = record.validation.effective_min_delta > self._config.decision.min_delta
            verdict = decision.confirm(
                SPLIT_VAL, theirs, ours, v0=v0, config=self._config.decision, guardrail=guardrail
            )
            await self._apply(record, "confirmation", verdict)
            if record.decision != DECISION_PENDING:
                return
        # A pending record whose confirmation passed: the tag moves first, so after a crash the
        # lane finds the tag on this commit and records the acceptance (goaly.rsi.resume).
        await asyncio.to_thread(shielded_accept, self._deps, record)

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
        await asyncio.to_thread(
            self._deps.lessons.add,
            name,
            self._deps.lane,
            reply.value.lesson,
            [record.experiment_id],
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
        # The solved tasks are drawn by a hash of the experiment, so a resume draws the same.
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
    ) -> Measured:
        evaluation = await self._evaluate(phase, commit, split, tasks, seeds, run_id)
        return Measured.from_results(evaluation.results, evaluation.stats)

    async def _evaluate(
        self,
        phase: EvalPhase,
        commit: str,
        split: Split.ValueType,
        tasks: Sequence[str],
        seeds: Sequence[int],
        run_id: str,
    ) -> Evaluation:
        tree = await asyncio.to_thread(self._tree, commit)
        request = EvaluationRequest(phase, commit, tree, split, tuple(tasks), tuple(seeds), run_id)
        evaluation = await self._deps.evaluator.evaluate(request)
        results = list(evaluation.results)
        got = {(result.instance_id, result.seed) for result in results}
        missing = {(task, seed) for task in tasks for seed in seeds} - got
        if missing:
            raise _AbortedError(
                f"the {phase} evaluation {run_id} lacks {len(missing)} task results"
            )
        try:
            GradeTable.from_results(results)  # Every result graded and scored, once.
        except ValueError as error:
            raise _AbortedError(
                f"the {phase} evaluation {run_id} is not usable: {error}"
            ) from error
        return evaluation

    async def _reference_val(self, commit: str, experiment_id: str, who: str) -> Measured:
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

    async def _candidate_val(
        self, record: ExperimentRecord, seed: int, phase: EvalPhase
    ) -> Measured:
        evaluation = await self._candidate_results(record, seed, phase)
        return Measured.from_results(evaluation.results, evaluation.stats)

    async def _apply(self, record: ExperimentRecord, phase_field: str, verdict: Verdict) -> None:
        getattr(record, phase_field).CopyFrom(verdict.phase)
        text = decision.describe(phase_field, verdict)
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
        await asyncio.to_thread(
            abort_record, deps.ledger, deps.queue, record, reason, now=deps.clock(), usage=usage
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

    def _tree(self, commit: str) -> str:
        return run_git(["rev-parse", f"{commit}^{{tree}}"], cwd=self._deps.workspace.repo).strip()

    async def _save(self, record: ExperimentRecord) -> None:
        await asyncio.to_thread(self._deps.ledger.record_experiment, record)

    def _now(self) -> Timestamp:
        return _timestamp(self._deps.clock())


class _RoundAbortedError(Exception):
    """The round could not start for a reason that says nothing about any hypothesis."""


def shielded_accept(deps: RoundDeps, record: ExperimentRecord) -> None:
    """Move the best tag to the record's commit, then record it accepted; in one thread, so a
    cancelled round cannot stop it between the two.
    """
    deps.workspace.set_best(deps.lane, record.candidate_commit, expected=record.parent_version)
    record.decision = DECISION_ACCEPTED
    record.finished_at.CopyFrom(_timestamp(deps.clock()))
    merge_usage(record, deps.usage(record.experiment_id))
    deps.ledger.record_experiment(record)


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
        if result.grade.status in _SCORED:
            chosen.setdefault(result.instance_id, result)
    return list(chosen.values())


def _hash(experiment_id: str, task: str) -> str:
    return hashlib.sha256(f"{experiment_id}\0{task}".encode()).hexdigest()


def _exit_name(exit_: ImplementerExit.ValueType) -> str:
    return ImplementerExit.Name(exit_)


def _timestamp(moment: datetime) -> Timestamp:
    stamp = Timestamp()
    stamp.FromDatetime(moment)
    return stamp
