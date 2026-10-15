"""The parallel runner (spec §6.1, §6.2): it drives a benchmark adapter and a pack's policy over
many tasks at once, and records every task.

A task is one instance with one seed. Each task:

1. builds the agent's input, and takes the task's evaluation from the cache if it is there;
2. waits for one of the run's task slots, which bound how many tasks solve at once;
3. provisions its environment, enters its context, and calls the policy's solve;
4. frees its slot, names its instance in the prediction, and is graded, except on SPLIT_EXTERNAL.
   Grading waits for the adapter's own grading pool, not for a slot;
5. writes its trajectory, then its result, and caches a clean evaluation.

A task that fails still gets both records, with its error type, and its empty answer is graded:
a failure in one task never affects the others. Only a failure of the run's own machinery, such
as a records file that cannot be written, stops the run. Cancellation always propagates: the
tasks in flight are cancelled, their environments closed, and every records file keeps only
whole lines, so the run can be resumed.

Once every task is recorded, the official prediction files are written from the trajectories,
whole: one per seed, with one record per instance, in the benchmark's order.
"""

import asyncio
import fcntl
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol, Self

from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_ENVIRONMENT,
    TASK_ERROR_TYPE_NONE,
    TASK_ERROR_TYPE_POLICY_CRASH,
    TASK_ERROR_TYPE_TIMEOUT,
    TaskErrorType,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADER_ERROR,
    GRADE_STATUS_TIMEOUT,
    SPLIT_EXTERNAL,
    Grade,
    Prediction,
    Split,
    TaskInput,
)
from goaly.contracts.gen.trajectory_pb2 import (
    STOP_REASON_CANCELLED,
    STOP_REASON_CRASHED,
    STOP_REASON_TIME_BUDGET_EXHAUSTED,
    STOP_REASON_UNSPECIFIED,
    StopReason,
    Trajectory,
)
from goaly.contracts.interfaces import Benchmark, Environment, Instance, Policy
from goaly.contracts.jsonl import read_jsonl, to_json_line
from goaly.harness.core.atomic import write_text_atomically
from goaly.harness.core.eval_cache import CachedEvaluation, CacheKey, EvalCache, task_digest
from goaly.harness.core.jsonl_writer import JsonlWriter
from goaly.harness.core.logs import log_context
from goaly.harness.core.manifest import read_manifest, write_manifest

__all__ = [
    "DEFAULT_MAX_TASKS",
    "ContextFactory",
    "ErrorClassifier",
    "RunDirectoryError",
    "RunFiles",
    "RunIdentity",
    "RunLock",
    "RunSummary",
    "Runner",
    "TaskContext",
    "UsageError",
    "classify_error",
    "error_classifier",
    "lock_run_dir",
]

_logger = logging.getLogger(__name__)

# How many tasks solve at once by default: the Effi-SQL lane's figure (spec §6.2).
DEFAULT_MAX_TASKS = 16
# How much of an error message a record keeps.
_MESSAGE_LIMIT = 2_000
# The manifest's fields only the benchmark's adapter knows (Benchmark.manifest_fields).
_ADAPTER_FIELDS = ("dataset_revision", "dataset_notes", "environment_settings", "dataset_sha256")
# What a resumed run must share with the run it resumes: anything that changes its results, so
# it never resumes on another dataset or in another grading mode.
_MUST_MATCH = (
    "benchmark",
    "split",
    "seeds",
    "instance_ids",
    "harness_hash",
    "config_hash",
    *_ADAPTER_FIELDS,
    "gold_sha256",
    "git_commit",
    "uv_lock_sha256",
    "models",
)

type _Key = tuple[str, int]  # (instance ID, seed)


class RunDirectoryError(Exception):
    """The run directory cannot be used: it holds another run, or another process uses it."""


class UsageError(ValueError):
    """The caller asked for a run that cannot run as asked; the message says what to change.

    `goaly run` exits with its usage status for this error, and for any other ValueError exits
    as a failed run, with the traceback logged: such an error is a bug to debug.
    """


@dataclass(frozen=True)
class RunFiles:
    """Where a run keeps its records, in its directory ``root``."""

    root: Path

    @property
    def manifest(self) -> Path:
        """The RunManifest, as indented JSON."""
        return self.root / "manifest.json"

    @property
    def results(self) -> Path:
        """One TaskResult per task: whoever reads it, a later line for a task wins."""
        return self.root / "results.jsonl"

    @property
    def trajectories(self) -> Path:
        """One Trajectory per task; as in the results, a later line for a task wins."""
        return self.root / "trajectories.jsonl"

    @property
    def llm_calls(self) -> Path:
        """The LLM call log, which the LLM client writes through the wiring, not the runner."""
        return self.root / "llm_calls.jsonl"

    @property
    def lock(self) -> Path:
        """Locked while a process runs in the directory."""
        return self.root / ".lock"

    def official(self, seed: int) -> Path:
        """The official prediction file of replicate ``seed``: one record per instance."""
        return self.root / "official" / f"seed-{seed}.jsonl"


class RunLock:
    """The lock that keeps every other process out of a run directory, until it is released.

    lock_run_dir() takes it. It is a context manager, and release() releases it; releasing it
    again does nothing. A process that dies releases it too.
    """

    def __init__(self, files: RunFiles, handle: BinaryIO) -> None:
        self._files = files
        self._handle = handle

    @property
    def files(self) -> RunFiles:
        """The run directory it locks."""
        return self._files

    @property
    def released(self) -> bool:
        """Whether it was released, so that it keeps no other process out any more."""
        return self._handle.closed

    def release(self) -> None:
        """Release the lock."""
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def lock_run_dir(files: RunFiles) -> RunLock:
    """Create the run directory if needed, and lock it for this process.

    Raise RunDirectoryError if another process holds its lock. It opens a file, so run it in a
    worker thread from the event loop.
    """
    try:
        files.root.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, NotADirectoryError) as error:  # It, or a parent, is a file.
        raise RunDirectoryError(f"{files.root} cannot be a run directory: {error}") from error
    handle = files.lock.open("ab")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RunDirectoryError(f"another process is running in {files.root}") from None
    return RunLock(files, handle)


@dataclass(frozen=True)
class RunIdentity:
    """The run a task belongs to, as its context factory receives it."""

    run_id: str
    # The benchmark adapter's name.
    benchmark: str
    split: Split.ValueType


class TaskContext(Protocol):
    """What the runner needs of a task's context, the policy's CtxT: what the agent did."""

    def trajectory(self) -> Trajectory:
        """Return the agent's record of the task so far: its chats, steps, usage, stop reason.

        The runner calls it once, when solve returns or raises, and copies the result. The
        runner fills in the identity, the task, the prediction, the grade and any error itself,
        and takes the task's usage, step count and stop reason from it.
        """


# Makes a task's context: (instance, seed, run) -> its async context manager.
type ContextFactory[InstanceT, CtxT] = Callable[
    [InstanceT, int, RunIdentity], AbstractAsyncContextManager[CtxT]
]
# Says what kind of error ended a task, from the exception that ended it.
type ErrorClassifier = Callable[[Exception], TaskErrorType.ValueType]


def error_classifier(
    rules: Sequence[tuple[type[Exception], TaskErrorType.ValueType]] = (),
) -> ErrorClassifier:
    """Return a classifier that says what kind of error an exception is, by its type.

    An EnvironmentFailure is TASK_ERROR_TYPE_ENVIRONMENT. Then ``rules``, pairs of an exception
    type and its error type, apply in order, and anything else is TASK_ERROR_TYPE_POLICY_CRASH,
    the only other crash result.proto has. A rule matches the exception itself or, in an
    exception group, any exception at any depth: a pack's own TaskGroup can wrap an
    EnvironmentFailure. The wiring adds, say, the LLM client's error.
    """
    ordered = [(EnvironmentFailure, TASK_ERROR_TYPE_ENVIRONMENT), *rules]

    def classify(error: Exception) -> TaskErrorType.ValueType:
        for kind, error_type in ordered:
            if _holds(error, kind):
                return error_type
        return TASK_ERROR_TYPE_POLICY_CRASH

    return classify


_DEFAULT_CLASSIFIER = error_classifier()


def classify_error(error: Exception) -> TaskErrorType.ValueType:
    """The default classifier: an EnvironmentFailure, even in a group, else a policy crash."""
    return _DEFAULT_CLASSIFIER(error)


@dataclass(frozen=True)
class RunSummary:
    """What a call of Runner.run did."""

    run_id: str
    # Every task of the run: each instance once per seed.
    tasks: int
    # Recorded by an earlier call that was interrupted, so not run again.
    resumed: int
    # Recorded from the evaluation cache.
    cached: int
    # Solved now.
    ran: int
    # How many of the tasks this call recorded ended with each error type, by its name.
    errors: Mapping[str, int]


class Runner[InstanceT: Instance, EnvT: Environment, CtxT: TaskContext]:
    """Runs a pack's policy on a benchmark's tasks, many at once, and records every task.

    ``contexts`` makes each task's context, which the policy gets. ``classify`` says what kind
    of error ended a task. With a ``cache``, a task's evaluation is reused from it, and a clean
    one is put in it; runs on SPLIT_EXTERNAL bypass it. At most ``max_tasks`` tasks solve at
    once, and a task that is not done ``task_timeout_s`` seconds after it got its slot is
    cancelled and recorded as TASK_ERROR_TYPE_TIMEOUT.
    """

    def __init__(
        self,
        benchmark: Benchmark[InstanceT, EnvT],
        policy: Policy[EnvT, CtxT],
        contexts: ContextFactory[InstanceT, CtxT],
        *,
        classify: ErrorClassifier = classify_error,
        cache: EvalCache | None = None,
        max_tasks: int = DEFAULT_MAX_TASKS,
        task_timeout_s: float | None = None,
    ) -> None:
        if max_tasks < 1:
            raise UsageError(f"max_tasks must be at least 1: {max_tasks}")
        if task_timeout_s is not None and not 0 < task_timeout_s < math.inf:
            raise UsageError(f"task_timeout_s must be a positive, finite number: {task_timeout_s}")
        self._benchmark = benchmark
        self._policy = policy
        self._contexts = contexts
        self._classify = classify
        self._cache = cache
        self._max_tasks = max_tasks
        self._task_timeout_s = task_timeout_s

    async def run(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        split: Split.ValueType,
        seeds: Sequence[int],
        instance_ids: Collection[str] | None = None,
        resume: bool = False,
        retry_errors: bool = False,
        lock: RunLock | None = None,
    ) -> RunSummary:
        """Run each instance of ``split`` once per seed, in ``run_dir``, and record every task.

        ``manifest`` holds the caller's part of the run's manifest: its run_id, the hashes, the
        provenance, the models and the command line. The runner adds the benchmark, the split,
        the seeds, the instance IDs and the times, takes the dataset's fields and the
        environment's settings from the benchmark's manifest_fields(), which alone sets them, and
        writes the manifest to the run directory.
        ``instance_ids`` restricts the run to those instances.

        With ``resume``, a run directory that holds an interrupted run continues it: the tasks
        without a result run, under the stored manifest, which must match this one in all that
        changes results. The run keeps its stored run_id, which the summary and every task's
        RunIdentity carry; the caller's is ignored. With ``retry_errors`` too, a task whose last
        result is an error, such as a transient environment or LLM failure, runs again, and its
        new records supersede the old ones. A directory with no run starts one. Without
        ``resume``, a directory that holds a run raises RunDirectoryError, as does one another
        process runs in.

        The run holds the directory's lock throughout. A caller that holds it already, as
        `goaly run` does from before it opens the harness, passes it as ``lock``.
        """
        if not manifest.run_id:
            raise UsageError("the manifest has no run_id")
        if self._cache is not None and not manifest.harness_hash:
            raise UsageError("the evaluation cache needs the manifest's harness_hash")
        seeds = _checked_seeds(seeds)
        files = RunFiles(run_dir)
        if lock is not None and lock.files != files:
            raise UsageError(f"the lock given is on {lock.files.root}, not on {run_dir}")
        if lock is not None and lock.released:
            raise UsageError(f"the lock given on {run_dir} is released already")
        own_lock = None if lock is not None else await asyncio.to_thread(lock_run_dir, files)
        try:
            plan = await asyncio.to_thread(
                self._plan, files, manifest, split, seeds, instance_ids, resume, retry_errors
            )
            run = await self._run_tasks(files, plan)
            await asyncio.to_thread(self._finish, files, plan)
        finally:
            if own_lock is not None:
                own_lock.release()
        summary = RunSummary(
            run_id=plan.manifest.run_id,
            tasks=len(plan.instances) * len(plan.seeds),
            resumed=len(plan.done),
            cached=run.cached,
            ran=run.ran,
            errors=dict(run.errors),
        )
        _logger.info("finished run %s: %s", summary.run_id, summary)
        return summary

    # Planning and finishing, in worker threads.

    def _plan(
        self,
        files: RunFiles,
        template: RunManifest,
        split: Split.ValueType,
        seeds: list[int],
        instance_ids: Collection[str] | None,
        resume: bool,
        retry_errors: bool,
    ) -> "_Plan[InstanceT]":
        instances = self._instances(split, instance_ids)
        manifest = RunManifest()
        manifest.CopyFrom(template)
        _take_adapter_fields(manifest, self._benchmark.manifest_fields())
        manifest.benchmark = self._benchmark.name
        manifest.split = split
        manifest.seeds[:] = seeds
        manifest.instance_ids[:] = [inst.instance_id for inst in instances]
        manifest.ClearField("started_at")
        manifest.ClearField("finished_at")
        keys = {(inst.instance_id, seed) for inst in instances for seed in seeds}
        if files.manifest.exists():
            stored = read_manifest(files.manifest)
            if not resume:
                raise RunDirectoryError(
                    f"{files.root} already holds the run {stored.run_id}: resume it,"
                    " or choose another directory"
                )
            different = [
                name for name in _MUST_MATCH if getattr(stored, name) != getattr(manifest, name)
            ]
            if different:
                raise RunDirectoryError(
                    f"the run {stored.run_id} in {files.root} does not match this one in:"
                    f" {', '.join(different)}"
                )
            recorded = _recorded(files, keys)
            done = {
                key
                for key, result in recorded.items()
                if not (retry_errors and result.error_type != TASK_ERROR_TYPE_NONE)
            }
            if template.run_id != stored.run_id:
                _logger.info(
                    "resuming run %s, not %s: a resumed run keeps its run ID",
                    stored.run_id,
                    template.run_id,
                )
            _logger.info(
                "resuming run %s: %d of %d tasks recorded", stored.run_id, len(done), len(keys)
            )
            return _Plan(stored, instances, seeds, done)
        for path in (files.results, files.trajectories):
            if path.exists():
                raise RunDirectoryError(f"{path} exists, but {files.manifest} does not")
        manifest.started_at.GetCurrentTime()
        write_manifest(files.manifest, manifest)
        _logger.info("started run %s: %d tasks", manifest.run_id, len(keys))
        return _Plan(manifest, instances, seeds, set())

    def _instances(
        self, split: Split.ValueType, instance_ids: Collection[str] | None
    ) -> list[InstanceT]:
        """The instances of ``split``, or those of them in ``instance_ids``, in their order."""
        instances = list(self._benchmark.load_instances(split))
        known: set[str] = set()
        for inst in instances:
            if inst.instance_id in known:
                raise ValueError(f"the benchmark has two instances {inst.instance_id!r}")
            known.add(inst.instance_id)
        if instance_ids is None:
            return instances
        unknown = sorted(set(instance_ids) - known)
        if unknown:
            raise UsageError(f"{Split.Name(split)} has no instance {', '.join(map(repr, unknown))}")
        return [inst for inst in instances if inst.instance_id in instance_ids]

    def _finish(self, files: RunFiles, plan: "_Plan[InstanceT]") -> None:
        """Write the official prediction files from the trajectories, then finish the manifest."""
        predictions = _predictions(files)
        for seed in plan.seeds:
            lines = []
            for inst in plan.instances:
                key = (inst.instance_id, seed)
                if key not in predictions:
                    raise RunDirectoryError(f"{files.trajectories} has no trajectory of {key}")
                prediction = predictions[key]
                prediction.instance_id = inst.instance_id
                record = self._benchmark.to_official(inst, prediction)
                lines.append(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                )
            path = files.official(seed)
            path.parent.mkdir(exist_ok=True)
            write_text_atomically(path, "".join(f"{line}\n" for line in lines))
        plan.manifest.finished_at.GetCurrentTime()
        write_manifest(files.manifest, plan.manifest)

    # The tasks.

    async def _run_tasks(self, files: RunFiles, plan: "_Plan[InstanceT]") -> "_Run":
        identity = RunIdentity(
            run_id=plan.manifest.run_id,
            benchmark=plan.manifest.benchmark,
            split=plan.manifest.split,
        )
        async with (
            JsonlWriter[TaskResult](files.results) as results,
            JsonlWriter[Trajectory](files.trajectories) as trajectories,
        ):
            run = _Run(
                identity=identity,
                harness_hash=plan.manifest.harness_hash,
                results=results,
                trajectories=trajectories,
                slots=asyncio.Semaphore(self._max_tasks),
            )
            async with asyncio.TaskGroup() as group:
                for seed in plan.seeds:  # A whole replicate first, then the next.
                    for inst in plan.instances:
                        if (inst.instance_id, seed) not in plan.done:
                            name = f"task {inst.instance_id} seed {seed}"
                            group.create_task(self._task(run, inst, seed), name=name)
        return run

    async def _task(self, run: "_Run", inst: InstanceT, seed: int) -> None:
        # Every line the task's code logs names the task, the pack's and the adapter's too.
        with log_context(run_id=run.identity.run_id, instance_id=inst.instance_id, seed=seed):
            await self._task_records(run, inst, seed)

    async def _task_records(self, run: "_Run", inst: InstanceT, seed: int) -> None:
        key = None
        task: TaskInput | None = None
        try:
            task = self._benchmark.agent_input(inst)
        except BaseException as error:  # noqa: BLE001 - The adapter's failure ends only this task.
            _reraise_unless_the_tasks_own(error, _cancel_requests(), "building the agent's input")
            attempt = _Attempt()
            self._fail(attempt, error, inst, seed)
        else:
            key = self._cache_key(run, inst, seed, task)
            if key is not None:
                cached = await self._cache_get(key)
                if cached is not None:
                    await run.record(cached.result, cached.trajectory, cached=True)
                    return
            attempt = await self._attempt(run, inst, seed, task)
        prediction = self._submission(attempt, inst, seed)
        grade = await self._grade(inst, prediction) if run.graded else None
        result, trajectory = self._records(run, inst, seed, task, attempt, prediction, grade)
        await run.record(result, trajectory, cached=False)
        if key is not None and _clean(result):
            await self._cache_put(key, result, trajectory)

    async def _attempt(
        self, run: "_Run", inst: InstanceT, seed: int, task: TaskInput
    ) -> "_Attempt":
        """Provision the task's environment and context, and solve it, in one of the slots.

        The slot's work runs in a task of its own, and so does solve within it. Python 3.12's
        TaskGroup leaves a cancel request on its task when a child fails while the group exits,
        though nothing cancelled that task; adapter, wiring or pack code that handles such a
        failure leaves it there, never on this task, which compares cancel requests to tell an
        outer cancellation from a task's own failure.
        """
        async with run.slots:
            attempt = _Attempt(started_at=_now(), slot_taken=time.monotonic())
            cancel_requests = _cancel_requests()
            deadline = asyncio.timeout(self._task_timeout_s)
            try:
                async with deadline:
                    await asyncio.create_task(
                        self._provision_and_solve(attempt, run, inst, seed, task)
                    )
            except BaseException as error:  # noqa: BLE001 - The task's own failures; see _fail.
                _reraise_unless_the_tasks_own(error, cancel_requests, "solving the task")
                if attempt.solve_started_at is None:  # Provisioning took this long, and failed.
                    attempt.provision_s = time.monotonic() - attempt.slot_taken
                if deadline.expired():
                    self._time_out(attempt, inst, seed)
                else:
                    self._fail(attempt, error, inst, seed)
            else:
                _propagate_cancellation(cancel_requests, None, "solving the task")
                if deadline.expired():
                    self._time_out(attempt, inst, seed)  # It answered too late.
        return attempt

    async def _provision_and_solve(
        self, attempt: "_Attempt", run: "_Run", inst: InstanceT, seed: int, task: TaskInput
    ) -> None:
        async with (
            self._benchmark.provision(inst) as env,
            self._contexts(inst, seed, run.identity) as ctx,
        ):
            attempt.provision_s = time.monotonic() - attempt.slot_taken
            await self._solve(attempt, task, env, ctx)

    async def _solve(self, attempt: "_Attempt", task: TaskInput, env: EnvT, ctx: CtxT) -> None:
        attempt.solve_started_at = _now()
        start = time.monotonic()
        trajectory_error = None
        try:
            # In a task of its own too, so that a cancel request the policy leaves on it never
            # reaches the code that closes the context and the environment.
            attempt.prediction = await asyncio.create_task(self._policy.solve(task, env, ctx))
        finally:
            attempt.wall_time_s = time.monotonic() - start
            attempt.finished_at = _now()
            attempt.trajectory, trajectory_error = _trajectory_of(ctx)
        if trajectory_error is not None:
            raise trajectory_error

    def _submission(self, attempt: "_Attempt", inst: InstanceT, seed: int) -> Prediction:
        """The prediction the task submits, named for its instance: the policy's, or the
        benchmark's empty answer if there is none or the adapter cannot write it.

        The empty answer's official form is written here too, task by task, so that an adapter
        that breaks that contract stops the run at its first such task, not after the last one.
        """
        if attempt.prediction is not None:
            prediction = Prediction()
            prediction.CopyFrom(attempt.prediction)
            prediction.instance_id = inst.instance_id
            try:
                self._benchmark.to_official(inst, prediction)
            except BaseException as error:  # noqa: BLE001 - It ends only this task; see _fail.
                _reraise_unless_the_tasks_own(error, _cancel_requests(), "writing the prediction")
                rejected = f"; it was {to_json_line(prediction)}"  # Kept, for debugging.
                self._fail(
                    attempt, error, inst, seed, "the prediction has no official form: ", rejected
                )
            else:
                return prediction
        empty = Prediction(instance_id=inst.instance_id)
        try:
            self._benchmark.to_official(inst, empty)
        except Exception as error:
            raise RuntimeError(
                f"the benchmark cannot write the empty answer of {inst.instance_id!r},"
                f" which every task needs: {_describe(error)}"
            ) from error
        return empty

    async def _grade(self, inst: InstanceT, prediction: Prediction) -> Grade:
        cancel_requests = _cancel_requests()
        try:
            # In a task of its own, as solving is: see _attempt.
            grade = await asyncio.create_task(self._benchmark.grade(inst, prediction))
        except BaseException as error:
            _reraise_unless_the_tasks_own(error, cancel_requests, "grading")
            _logger.warning(
                "grading %s raised an exception",
                inst.instance_id,
                exc_info=error,
                extra={"instance_id": inst.instance_id},
            )
            feedback = _cut(f"The grader failed: {_describe(error)}")
            return Grade(
                instance_id=inst.instance_id, status=GRADE_STATUS_GRADER_ERROR, feedback=feedback
            )
        _propagate_cancellation(cancel_requests, None, "grading")
        return grade

    def _records(
        self,
        run: "_Run",
        inst: InstanceT,
        seed: int,
        task: TaskInput | None,
        attempt: "_Attempt",
        prediction: Prediction,
        grade: Grade | None,
    ) -> tuple[TaskResult, Trajectory]:
        trajectory = attempt.trajectory
        trajectory.run_id = run.identity.run_id
        trajectory.instance_id = inst.instance_id
        trajectory.seed = seed
        _set(trajectory, "task", task)
        # The prediction submitted: absent when the task has none, so its file has the empty one.
        _set(trajectory, "prediction", None if attempt.prediction is None else prediction)
        _set(trajectory, "grade", grade)
        if attempt.error_type != TASK_ERROR_TYPE_NONE:
            trajectory.stop_reason = attempt.stop_reason
            trajectory.error_message = attempt.error_message
        if attempt.solve_started_at is not None:
            if not trajectory.HasField("started_at"):
                trajectory.started_at.CopyFrom(attempt.solve_started_at)
            if not trajectory.HasField("finished_at") and attempt.finished_at is not None:
                trajectory.finished_at.CopyFrom(attempt.finished_at)
        result = TaskResult(
            run_id=run.identity.run_id,
            instance_id=inst.instance_id,
            seed=seed,
            wall_time_s=attempt.wall_time_s,
            provision_s=attempt.provision_s,
            usage=trajectory.usage,
            step_count=len(trajectory.steps),
            stop_reason=trajectory.stop_reason,
            error_type=attempt.error_type,
            error_message=attempt.error_message,
        )
        _set(result, "grade", grade)
        _set(result, "started_at", attempt.started_at)
        _set(result, "finished_at", attempt.finished_at)
        return result, trajectory

    # Failures.

    def _fail(
        self,
        attempt: "_Attempt",
        error: BaseException,
        inst: InstanceT,
        seed: int,
        context: str = "",
        detail: str = "",
    ) -> None:
        error_type = self._classify_own(error)
        attempt.fail(error_type, _cut(context + _describe(error) + detail), STOP_REASON_CRASHED)
        _logger.warning(
            "task %s, seed %d, failed: %s",
            inst.instance_id,
            seed,
            attempt.error_message,
            exc_info=error,
            extra={"instance_id": inst.instance_id, "seed": seed},
        )

    def _classify_own(self, error: BaseException) -> TaskErrorType.ValueType:
        """Classify an error of the task's own code, which may be a stray CancelledError, raised
        though nothing cancelled the run, or a BaseExceptionGroup holding one."""
        exception = error.subgroup(Exception) if isinstance(error, BaseExceptionGroup) else error
        if not isinstance(exception, Exception):
            return TASK_ERROR_TYPE_POLICY_CRASH  # Only cancellations, from within the task.
        try:
            return self._classify(exception)
        except Exception:
            _logger.exception("the error classifier raised; recording a policy crash")
            return TASK_ERROR_TYPE_POLICY_CRASH

    def _time_out(self, attempt: "_Attempt", inst: InstanceT, seed: int) -> None:
        message = f"the task ran past its time limit of {self._task_timeout_s:g} s"
        attempt.fail(TASK_ERROR_TYPE_TIMEOUT, message, STOP_REASON_CANCELLED)
        _logger.warning(
            "task %s, seed %d: %s",
            inst.instance_id,
            seed,
            message,
            extra={"instance_id": inst.instance_id, "seed": seed},
        )

    # The evaluation cache.

    def _cache_key(
        self, run: "_Run", inst: InstanceT, seed: int, task: TaskInput
    ) -> CacheKey | None:
        if self._cache is None or not run.graded:
            return None  # Organizer IDs on SPLIT_EXTERNAL may collide with ours.
        return CacheKey(
            harness_hash=run.harness_hash,
            benchmark=run.identity.benchmark,
            instance_id=inst.instance_id,
            seed=seed,
            task_digest=task_digest(task),
        )

    async def _cache_get(self, key: CacheKey) -> CachedEvaluation | None:
        assert self._cache is not None  # noqa: S101 - _cache_key returns a key only with a cache.
        return await self._cache.get(key)

    async def _cache_put(self, key: CacheKey, result: TaskResult, trajectory: Trajectory) -> None:
        assert self._cache is not None  # noqa: S101 - As above.
        try:
            await self._cache.put(key, CachedEvaluation(result, trajectory))
        except OSError as error:
            _logger.warning(
                "could not cache the evaluation of %s, seed %d: %s",
                key.instance_id,
                key.seed,
                error,
            )


@dataclass(frozen=True)
class _Plan[InstanceT]:
    """What a call of Runner.run runs: the manifest, the instances, the seeds, and the tasks
    recorded already, by an earlier call."""

    manifest: RunManifest
    instances: list[InstanceT]
    seeds: list[int]
    done: set[_Key]


@dataclass
class _Run:
    """The state of one call of Runner.run, which its tasks share."""

    identity: RunIdentity
    harness_hash: str
    results: JsonlWriter[TaskResult]
    trajectories: JsonlWriter[Trajectory]
    slots: asyncio.Semaphore
    cached: int = 0
    ran: int = 0
    errors: Counter[str] = field(default_factory=Counter)

    @property
    def graded(self) -> bool:
        """Whether its tasks are graded: SPLIT_EXTERNAL has no gold answers."""
        return self.identity.split != SPLIT_EXTERNAL

    async def record(self, result: TaskResult, trajectory: Trajectory, *, cached: bool) -> None:
        """Write a task's records: the trajectory first, so a result always has its trajectory."""
        result.run_id = trajectory.run_id = self.identity.run_id
        await self.trajectories.write(trajectory)
        await self.results.write(result)
        if cached:
            self.cached += 1
        else:
            self.ran += 1
        if result.error_type != TASK_ERROR_TYPE_NONE:
            self.errors[TaskErrorType.Name(result.error_type)] += 1


@dataclass
class _Attempt:
    """How solving a task went."""

    trajectory: Trajectory = field(default_factory=Trajectory)
    prediction: Prediction | None = None
    error_type: TaskErrorType.ValueType = TASK_ERROR_TYPE_NONE
    error_message: str = ""
    stop_reason: StopReason.ValueType = STOP_REASON_UNSPECIFIED
    # When the task got its slot; None if it never did.
    started_at: Timestamp | None = None
    # When solve started; None if it never did.
    solve_started_at: Timestamp | None = None
    # When the policy returned, or the task failed.
    finished_at: Timestamp | None = None
    provision_s: float = 0.0
    wall_time_s: float = 0.0
    # When the task got its slot, as time.monotonic() counts.
    slot_taken: float = 0.0

    def fail(
        self, error_type: TaskErrorType.ValueType, message: str, stop_reason: StopReason.ValueType
    ) -> None:
        """Record an error that ended the task; it then has no prediction."""
        self.prediction = None
        self.error_type = error_type
        self.error_message = message
        self.stop_reason = stop_reason
        self.finished_at = _now()


def _checked_seeds(seeds: Sequence[int]) -> list[int]:
    if not seeds:
        raise UsageError("a run needs at least one seed")
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise UsageError(f"the seeds must be distinct replicate indices, 0 or more: {list(seeds)}")
    return list(seeds)


def _take_adapter_fields(manifest: RunManifest, fields: RunManifest) -> None:
    """Set ``manifest``'s adapter fields from ``fields``, the adapter's partial manifest, and
    nothing else from it: whatever the caller set for them, the adapter alone sets them."""
    for name in _ADAPTER_FIELDS:
        manifest.ClearField(name)  # type: ignore[arg-type]  # The names are the manifest's fields.
    manifest.dataset_revision = fields.dataset_revision
    manifest.dataset_notes.extend(fields.dataset_notes)
    manifest.environment_settings.update(fields.environment_settings)
    manifest.dataset_sha256 = fields.dataset_sha256


def _recorded(files: RunFiles, keys: set[_Key]) -> dict[_Key, TaskResult]:
    """The last result of each task of ``keys`` the run's results file records. A torn last line
    is skipped, and the next write cuts it off. A run directory holds one run, so every line is
    the run's."""
    if not files.results.exists():  # Interrupted before its records files were created.
        return {}
    last = {
        (result.instance_id, result.seed): result
        for result in read_jsonl(files.results, TaskResult)
    }
    return {key: result for key, result in last.items() if key in keys}


def _predictions(files: RunFiles) -> dict[_Key, Prediction]:
    """Each task's submitted prediction, from its last trajectory line; empty if it had none.

    Each is a copy: a sub-message of a trajectory would keep the whole trajectory in memory.
    """
    predictions: dict[_Key, Prediction] = {}
    for trajectory in read_jsonl(files.trajectories, Trajectory):
        prediction = Prediction()
        prediction.CopyFrom(trajectory.prediction)
        predictions[(trajectory.instance_id, trajectory.seed)] = prediction
    return predictions


def _clean(result: TaskResult) -> bool:
    """Whether a result is worth caching: a final outcome of the harness itself.

    That is no error, and a grade that means something. An outcome that depends on the load is
    not final either: the wall-clock budget running out, whose time includes the gateway's waits
    and backoffs, or the grader's own time limit.
    """
    return (
        result.error_type == TASK_ERROR_TYPE_NONE
        and result.stop_reason != STOP_REASON_TIME_BUDGET_EXHAUSTED
        and result.HasField("grade")
        and result.grade.status not in (GRADE_STATUS_GRADER_ERROR, GRADE_STATUS_TIMEOUT)
    )


def _trajectory_of(ctx: TaskContext) -> tuple[Trajectory, BaseException | None]:
    """A copy of ``ctx``'s trajectory or, if it cannot give one, an empty one and the error."""
    trajectory = Trajectory()
    try:
        trajectory.CopyFrom(ctx.trajectory())
    except BaseException as error:
        if _ends_the_process(error):
            raise
        return Trajectory(), error
    return trajectory, None


def _set(message: TaskResult | Trajectory, name: str, value: object) -> None:
    """Set the message field ``name`` to a copy of ``value``, or clear it for None."""
    if value is None:
        message.ClearField(name)  # type: ignore[arg-type]  # The callers name message fields.
    else:
        getattr(message, name).CopyFrom(value)


def _holds(error: BaseException, kind: type[Exception]) -> bool:
    """Whether ``error`` is a ``kind`` or, if it is a group, holds one at any depth."""
    if isinstance(error, BaseExceptionGroup):
        return error.subgroup(kind) is not None
    return isinstance(error, kind)


def _describe(error: BaseException) -> str:
    """Name ``error``'s type and give its message; for a group, each exception's inside it."""
    if isinstance(error, BaseExceptionGroup):
        inside = "; ".join(_describe(leaf) for leaf in error.exceptions)
        return f"{type(error).__name__}: {error.message}: [{inside}]"
    try:
        message = str(error)
    except Exception:  # noqa: BLE001 - A broken __str__ must not take the run down.
        message = ""
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _cut(text: str) -> str:
    if len(text) <= _MESSAGE_LIMIT:
        return text
    return f"{text[:_MESSAGE_LIMIT]} [{len(text) - _MESSAGE_LIMIT} characters cut]"


def _now() -> Timestamp:
    now = Timestamp()
    now.GetCurrentTime()
    return now


def _cancel_requests() -> int:
    """How many requests to cancel the current task are pending, as Task.cancelling counts."""
    task = asyncio.current_task()
    return 0 if task is None else task.cancelling()


def _reraise_unless_the_tasks_own(error: BaseException, before: int, what: str) -> None:
    """Re-raise ``error``, caught from the task's code, unless it is the task's own failure.

    An outer cancellation, which raised the current task's cancel requests above ``before``,
    propagates as a cancellation, even if the code turned it into ``error``; so does a
    KeyboardInterrupt or a SystemExit, alone or in a group. Anything else is the task's own: an
    exception, a CancelledError the code raised while nothing cancelled the run, or a
    BaseExceptionGroup of those. It fails that task alone, never the run.
    """
    _propagate_cancellation(before, error, what)
    if _ends_the_process(error):
        raise error


def _ends_the_process(error: BaseException) -> bool:
    """Whether ``error`` is, or holds at any depth, a KeyboardInterrupt or a SystemExit."""
    if isinstance(error, BaseExceptionGroup):
        return error.subgroup((KeyboardInterrupt, SystemExit)) is not None
    return isinstance(error, (KeyboardInterrupt, SystemExit))


def _propagate_cancellation(before: int, error: BaseException | None, what: str) -> None:
    """Raise CancelledError if the current task was cancelled since it had ``before`` requests.

    Adapter, wiring or pack code ``what`` ran turned that cancellation into ``error``, or
    swallowed it if ``error`` is None. The cancellation propagates all the same, and is logged.
    """
    if _cancel_requests() <= before:
        return
    if error is None:
        _logger.warning("%s swallowed a cancellation, which propagates", what)
    else:
        _logger.warning(
            "%s turned a cancellation into %s, which propagates as a cancellation",
            what,
            _describe(error),
        )
    raise asyncio.CancelledError from error
