"""The host side of a two-phase run (spec §6.1, V27): plan it, take the evaluations the cache holds,
have the benchmark's launcher solve the rest in the solve container, and grade every task.

The solve container runs candidate pack code, so the host trusts nothing it wrote for anything
that decides a score:

- The run's expectations are the host's. It plans the run from the benchmark alone, which loads
  no pack code: the instances, the seeds, the split and the grading mode. It also computes the
  harness hash from the pack's files and the agent's input from the dataset. All of these go in
  the manifest, which only the host writes, and key the evaluation cache. A later grade phase
  takes them from that manifest.
- The container writes only the solve directory (RunFiles.solve). Of it, the host reads each
  task's prediction and nothing else that matters. It grades every task it expects: a task with
  no prediction, with two, or with one that does not parse gets an error and credit 0. A task
  is never "graded already" by anything the container wrote: only the host writes results.jsonl.
- Only the host reads and writes the evaluation cache. Its hits are taken before the container
  starts, so the container never solves them.
- A task's usage, its tokens and cost, comes from the records of the host's LLM proxy, which
  carry the task their call claimed, and its wall time is the span of those records. The run's
  totals are exact; the split among the tasks is the container's claim, and usage.json says how
  much of the total the tasks account for. The container's own result and trajectory of a task
  are kept, for people and the analyst, but its error type, stop reason and step count are its
  claims too: they decide nothing but whether a result is cached, and so only that harness's
  own cache entries.
"""

import asyncio
import json
import logging
from collections import Counter
from collections.abc import AsyncIterator, Collection, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord, Usage
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_NONE,
    TASK_ERROR_TYPE_POLICY_CRASH,
    TaskErrorType,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADER_ERROR,
    GRADE_STATUS_PREDICTION_FAILED,
    SPLIT_EXTERNAL,
    Grade,
    Prediction,
    Split,
    TaskInput,
)
from goaly.contracts.gen.trajectory_pb2 import STOP_REASON_CRASHED, Trajectory
from goaly.contracts.interfaces import Benchmark, Environment, Instance
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core.atomic import write_text_atomically
from goaly.harness.core.eval_cache import (
    CachedEvaluation,
    CacheKey,
    EvalCache,
    evaluation_key,
    put_evaluation,
)
from goaly.harness.core.isolation import (
    SolveLauncher,
    SolveRecord,
    SolveRequest,
    SolveTasks,
    read_solve_records,
    write_solve_tasks,
)
from goaly.harness.core.jsonl_writer import JsonlWriter
from goaly.harness.core.manifest import read_manifest, write_manifest
from goaly.harness.core.runner import (
    GRADING_TASK,
    RunDirectoryError,
    RunFiles,
    RunLock,
    RunPlan,
    RunSummary,
    UsageError,
    cacheable,
    differing_adapter_fields,
    grade_prediction,
    last_records,
    lock_run_dir,
    plan_run,
)

__all__ = ["GRADING_TASK", "USAGE_TOLERANCE", "HostRunner", "SolveOptions"]

_logger = logging.getLogger(__name__)

# How far the tasks' usage may fall short of the proxy's total, as a fraction, before the host
# warns that the solve phase's calls are not all accounted for.
USAGE_TOLERANCE = 0.01

type _Key = tuple[str, int]


@dataclass(frozen=True)
class SolveOptions:
    """What the host's caller says of a run's solve phase, for the launcher's SolveRequest."""

    # The registered harness the container runs, and the checkout it runs it from: the host's
    # own or --harness-checkout's.
    harness: str
    checkout: Path
    # The pack's directory within ``checkout``, which the host hashed; None for no pack.
    pack_dir: Path | None
    # The organizers' task file (--instances), or None.
    instances_file: Path | None
    llm_shares: Mapping[str, int]
    max_tasks: int
    task_timeout_s: float | None
    log_level: str


@dataclass
class _Tally:
    """What one call of the host's run or grade did, for its summary."""

    cached: int = 0
    graded: int = 0
    errors: Counter[str] = field(default_factory=Counter)

    def count(self, result: TaskResult) -> None:
        self.graded += 1
        if result.error_type != TASK_ERROR_TYPE_NONE:
            self.errors[TaskErrorType.Name(result.error_type)] += 1


class _Writers:
    """The host's results and trajectories files, which only it writes."""

    def __init__(self, results: JsonlWriter[TaskResult], trajectories: JsonlWriter[Trajectory]):
        self._results = results
        self._trajectories = trajectories

    async def record(self, run_id: str, result: TaskResult, trajectory: Trajectory) -> None:
        """Write a task's records: the trajectory first, so a result always has its trajectory."""
        result.run_id = trajectory.run_id = run_id
        await self._trajectories.write(trajectory)
        await self._results.write(result)


class HostRunner[InstanceT: Instance, EnvT: Environment]:
    """The host side of a two-phase run of ``benchmark``: see the module's docstring.

    ``launcher`` runs the isolated solve phase; ``cache`` is the evaluation cache, which only
    the host reads and writes.
    """

    def __init__(
        self,
        benchmark: Benchmark[InstanceT, EnvT],
        *,
        launcher: SolveLauncher | None,
        cache: EvalCache | None = None,
    ) -> None:
        self._benchmark = benchmark
        self._launcher = launcher
        self._cache = cache

    async def run(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        split: Split.ValueType,
        seeds: list[int],
        instance_ids: Collection[str] | None,
        solve: SolveOptions,
        resume: bool = False,
        retry_errors: bool = False,
        lock: RunLock | None = None,
        grading: bool = True,
    ) -> RunSummary:
        """Run the run's solve phase in the container, then grade it, as `goaly run` does.

        It plans the run as Runner.run does, with the same ``manifest``, ``resume`` and
        ``retry_errors``. The tasks the host has recorded already are done, except one whose
        grade is a GRADER_ERROR, which is graded again. The rest are taken from the cache, or
        solved, then graded, and the run finishes with its official files. Without ``grading``
        (`--phase solve`), it stops once they are solved, for a later grade(); except on
        SPLIT_EXTERNAL, which has no grade, so that the run is recorded and finished at once.
        """
        if self._launcher is None:
            raise UsageError(
                f"the benchmark {self._benchmark.name!r} has no isolated solve phase:"
                " run it with --in-process"
            )
        launcher = self._launcher
        if self._cache is not None:
            _check_cacheable(manifest.harness_hash)
        files = RunFiles(run_dir)
        async with _locked(files, lock):
            plan = await asyncio.to_thread(
                plan_run,
                self._benchmark,
                files,
                manifest,
                split=split,
                seeds=seeds,
                instance_ids=instance_ids,
                resume=resume,
                retry_errors=retry_errors,
                grading=False,  # Only the host writes this results file: no line is ungraded.
            )
            await asyncio.to_thread(files.solve.root.mkdir, exist_ok=True)
            pending = await asyncio.to_thread(_pending, files, plan)
            tasks = {key: self._benchmark.agent_input(inst) for key, inst in pending.items()}
            finishing = grading or plan.manifest.split == SPLIT_EXTERNAL
            tally = _Tally()
            async with _writers(files) as writers:
                to_solve = await self._take_cached(plan, pending, tasks, writers, tally)
                if to_solve:
                    request = await asyncio.to_thread(
                        _request, files, plan, to_solve, solve, retry_errors
                    )
                    _logger.info(
                        "solving %d tasks of run %s in the solve container",
                        len(to_solve),
                        plan.manifest.run_id,
                    )
                    await launcher(request)
                if finishing:
                    await self._collect(files, plan, to_solve, tasks, writers, tally)
            if finishing:
                await asyncio.to_thread(self._finish, files, plan, tally)
        return _summary(plan, len(pending), tally)

    async def grade(
        self, run_dir: Path, *, lock: RunLock | None = None, retry_errors: bool = False
    ) -> RunSummary:
        """Grade every task of the solved run in ``run_dir`` that the host has not, and finish it.

        The run's expectations come from its manifest, which only the host writes. Raise
        RunDirectoryError if the directory holds no run, a run of another benchmark, another
        grading mode or another dataset, a run that was not solved in isolation, or one on
        SPLIT_EXTERNAL, whose solve phase finishes it. With ``retry_errors``, a task whose result
        is an error is graded again from its solve record.
        """
        files = RunFiles(run_dir)
        if not await asyncio.to_thread(files.manifest.exists):  # Before the lock creates it.
            raise RunDirectoryError(f"{run_dir} holds no run to grade")
        async with _locked(files, lock):
            plan = await asyncio.to_thread(self._stored_plan, files, retry_errors)
            pending = await asyncio.to_thread(_pending, files, plan)
            tasks = {key: self._benchmark.agent_input(inst) for key, inst in pending.items()}
            tally = _Tally()
            async with _writers(files) as writers:
                await self._collect(files, plan, list(pending), tasks, writers, tally)
            await asyncio.to_thread(self._finish, files, plan, tally)
        return _summary(plan, len(pending), tally)

    def _stored_plan(self, files: RunFiles, retry_errors: bool) -> RunPlan[InstanceT]:
        """The plan of the run in ``files``, from the manifest the host wrote for it."""
        manifest = read_manifest(files.manifest)
        where = f"the run {manifest.run_id} in {files.root}"
        if manifest.benchmark != self._benchmark.name:
            raise RunDirectoryError(
                f"{where} is of the benchmark {manifest.benchmark!r}, not {self._benchmark.name!r}"
            )
        different = differing_adapter_fields(manifest, self._benchmark.manifest_fields())
        if different:
            raise RunDirectoryError(
                f"{where} has another dataset or grading mode than this benchmark, in:"
                f" {', '.join(different)}"
            )
        if manifest.split == SPLIT_EXTERNAL:
            raise RunDirectoryError(f"{where} is on SPLIT_EXTERNAL, which its solve phase finishes")
        if not files.solve.root.is_dir():
            raise RunDirectoryError(f"{where} was not solved in isolation: it has no solve phase")
        if self._cache is not None:
            _check_cacheable(manifest.harness_hash)
        by_id = {inst.instance_id: inst for inst in self._benchmark.load_instances(manifest.split)}
        unknown = [instance_id for instance_id in manifest.instance_ids if instance_id not in by_id]
        if unknown:
            raise RunDirectoryError(
                f"the benchmark has no instance {', '.join(map(repr, unknown))} of {where}"
            )
        instances = [by_id[instance_id] for instance_id in manifest.instance_ids]
        seeds = list(manifest.seeds)
        keys = {(inst.instance_id, seed) for inst in instances for seed in seeds}
        done = {
            key
            for key, result in _host_results(files, keys).items()
            if not (retry_errors and result.error_type != TASK_ERROR_TYPE_NONE)
        }
        return RunPlan(manifest, instances, seeds, done)

    # The cache, before the container starts.

    async def _take_cached(
        self,
        plan: RunPlan[InstanceT],
        pending: Mapping[_Key, InstanceT],
        tasks: Mapping[_Key, TaskInput],
        writers: _Writers,
        tally: _Tally,
    ) -> list[_Key]:
        """Record each pending task the cache holds; return the others, which need solving."""
        if self._cache is None or plan.manifest.split == SPLIT_EXTERNAL:
            return list(pending)
        to_solve = []
        for key in pending:
            hit = await self._cache.get(_key(plan, key, tasks[key]))
            if hit is None:
                to_solve.append(key)
                continue
            await writers.record(plan.manifest.run_id, hit.result, hit.trajectory)
            tally.cached += 1
        return to_solve

    # Collecting and grading, after the container.

    async def _collect(
        self,
        files: RunFiles,
        plan: RunPlan[InstanceT],
        keys: list[_Key],
        tasks: Mapping[_Key, TaskInput],
        writers: _Writers,
        tally: _Tally,
    ) -> None:
        """Grade each task of ``keys`` from its solve record, and record it on the host."""
        if not keys:
            return
        records = await asyncio.to_thread(read_solve_records, files.solve.root, keys)
        usage = await asyncio.to_thread(_attributed, files, plan.manifest.run_id, set(keys))
        by_id = {inst.instance_id: inst for inst in plan.instances}
        async with asyncio.TaskGroup() as group:
            for key in keys:
                group.create_task(
                    self._record(
                        plan,
                        by_id[key[0]],
                        key,
                        tasks[key],
                        records[key],
                        usage.get(key),
                        writers,
                        tally,
                    ),
                    name=f"grading {key[0]} seed {key[1]}",
                )

    async def _record(
        self,
        plan: RunPlan[InstanceT],
        inst: InstanceT,
        key: _Key,
        task: TaskInput,
        record: SolveRecord,
        usage: "_TaskUsage | None",
        writers: _Writers,
        tally: _Tally,
    ) -> None:
        result, trajectory = _host_records(plan.manifest.run_id, key, task, record, usage)
        if plan.manifest.split != SPLIT_EXTERNAL:
            if record.prediction is None:
                grade = Grade(
                    instance_id=key[0],
                    score=0.0,
                    correct=False,
                    status=GRADE_STATUS_PREDICTION_FAILED,
                    feedback=f"Not graded: {record.problem}.",
                )
            else:
                grade = await grade_prediction(
                    self._benchmark, inst, record.prediction, seed=key[1]
                )
            result.grade.CopyFrom(grade)
            trajectory.grade.CopyFrom(grade)
            if self._cache is not None and cacheable(result):
                # Before the records, so that an interrupt between the two loses no entry.
                evaluation = CachedEvaluation(result, trajectory)
                await put_evaluation(self._cache, _key(plan, key, task), evaluation)
        await writers.record(plan.manifest.run_id, result, trajectory)
        tally.count(result)

    # Finishing.

    def _finish(self, files: RunFiles, plan: RunPlan[InstanceT], tally: _Tally) -> None:
        """Once the host has recorded every task: write the official files and the usage, then
        the manifest's finished_at, which a later call that records nothing leaves as it is."""
        trajectories = (
            last_records(files.trajectories, Trajectory) if files.trajectories.exists() else {}
        )
        missing = [key for key in plan.keys if key not in trajectories]
        if missing:
            _logger.warning(
                "run %s is not finished: the host has not recorded %d of its tasks",
                plan.manifest.run_id,
                len(missing),
            )
            return
        for seed in plan.seeds:
            lines = []
            for inst in plan.instances:
                prediction = Prediction()
                prediction.CopyFrom(trajectories[(inst.instance_id, seed)].prediction)
                prediction.instance_id = inst.instance_id
                record = self._benchmark.to_official(inst, prediction)
                lines.append(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                )
            path = files.official(seed)
            path.parent.mkdir(exist_ok=True)
            write_text_atomically(path, "".join(f"{line}\n" for line in lines))
        _write_usage(files, plan)
        if tally.graded or tally.cached or not plan.manifest.HasField("finished_at"):
            plan.manifest.finished_at.GetCurrentTime()
            write_manifest(files.manifest, plan.manifest)


def _summary[I: Instance](plan: RunPlan[I], pending: int, tally: _Tally) -> RunSummary:
    summary = RunSummary(
        run_id=plan.manifest.run_id,
        tasks=len(plan.keys),
        resumed=len(plan.keys) - pending,
        cached=tally.cached,
        ran=tally.graded,
        errors=dict(tally.errors),
    )
    _logger.info("host side of run %s: %s", summary.run_id, summary)
    return summary


@asynccontextmanager
async def _writers(files: RunFiles) -> AsyncIterator[_Writers]:
    async with (
        JsonlWriter[TaskResult](files.results) as results,
        JsonlWriter[Trajectory](files.trajectories) as trajectories,
    ):
        yield _Writers(results, trajectories)


@asynccontextmanager
async def _locked(files: RunFiles, lock: RunLock | None) -> AsyncIterator[None]:
    """Hold ``files``' lock: the caller's ``lock``, or one taken for the call."""
    if lock is not None and lock.files != files:
        raise UsageError(f"the lock given is on {lock.files.root}, not on {files.root}")
    if lock is not None and lock.released:
        raise UsageError(f"the lock given on {files.root} is released already")
    own = None if lock is not None else await asyncio.to_thread(lock_run_dir, files)
    try:
        yield
    finally:
        if own is not None:
            own.release()


def _key[I: Instance](plan: RunPlan[I], key: _Key, task: TaskInput) -> CacheKey:
    return evaluation_key(
        harness_hash=plan.manifest.harness_hash,
        benchmark=plan.manifest.benchmark,
        instance_id=key[0],
        seed=key[1],
        task=task,
    )


def _pending[I: Instance](files: RunFiles, plan: RunPlan[I]) -> dict[_Key, I]:
    """The run's tasks the host has still to record, in order. A task whose grade is a
    GRADER_ERROR, which says nothing of its prediction, is one of them."""
    recorded = _host_results(files, plan.keys)
    pending = {}
    for seed in plan.seeds:
        for inst in plan.instances:
            key = (inst.instance_id, seed)
            result = recorded.get(key)
            regrade = result is not None and result.grade.status == GRADE_STATUS_GRADER_ERROR
            if key not in plan.done or regrade:
                pending[key] = inst
    return pending


def _host_results(files: RunFiles, keys: Collection[_Key]) -> dict[_Key, TaskResult]:
    if not files.results.exists():
        return {}
    last = last_records(files.results, TaskResult)
    return {key: result for key, result in last.items() if key in keys}


def _host_records(
    run_id: str, key: _Key, task: TaskInput, record: SolveRecord, usage: "_TaskUsage | None"
) -> tuple[TaskResult, Trajectory]:
    """A task's result and trajectory on the host: the container's, for what it claims, with
    the host's run, task, prediction and usage. A task with no valid prediction is an error."""
    result, trajectory = TaskResult(), Trajectory()
    if record.result is not None:
        result.CopyFrom(record.result)
    if record.trajectory is not None:
        trajectory.CopyFrom(record.trajectory)
    instance_id, seed = key
    result.run_id, result.instance_id, result.seed = run_id, instance_id, seed
    trajectory.run_id, trajectory.instance_id, trajectory.seed = run_id, instance_id, seed
    result.ClearField("grade")
    trajectory.ClearField("grade")
    trajectory.task.CopyFrom(task)
    trajectory.prediction.CopyFrom(
        record.prediction if record.prediction is not None else Prediction(instance_id=instance_id)
    )
    if record.prediction is None:
        result.error_type = TASK_ERROR_TYPE_POLICY_CRASH
        result.error_message = trajectory.error_message = record.problem
        result.stop_reason = trajectory.stop_reason = STOP_REASON_CRASHED
    found = _TaskUsage() if usage is None else usage
    result.usage.CopyFrom(found.usage)
    trajectory.usage.CopyFrom(found.usage)
    result.wall_time_s = found.wall_time_s
    return result, trajectory


def _request[I: Instance](
    files: RunFiles, plan: RunPlan[I], keys: list[_Key], solve: SolveOptions, retry_errors: bool
) -> SolveRequest:
    """Write the tasks file and the launch's start, and return the launcher's request."""
    write_solve_tasks(
        files.solve.tasks,
        SolveTasks(
            instance_ids=tuple(plan.manifest.instance_ids),
            seeds=tuple(plan.manifest.seeds),
            tasks=tuple(keys),
        ),
    )
    with files.launches.open("a", encoding="utf-8") as launches:
        launches.write(json.dumps({"started_at": datetime.now(UTC).isoformat()}) + "\n")
    return SolveRequest(
        run_id=plan.manifest.run_id,
        benchmark=plan.manifest.benchmark,
        harness=solve.harness,
        checkout=solve.checkout,
        pack_dir=solve.pack_dir,
        split=plan.manifest.split,
        instances_file=solve.instances_file,
        solve_dir=files.solve.root,
        inputs_dir=files.inputs,
        llm_calls=files.llm_calls,
        tasks=tuple(keys),
        llm_shares=dict(solve.llm_shares),
        max_tasks=solve.max_tasks,
        task_timeout_s=solve.task_timeout_s,
        resume=files.solve.manifest.exists(),
        retry_errors=retry_errors,
        log_level=solve.log_level,
    )


def _check_cacheable(harness_hash: str) -> None:
    """Refuse, before any task runs, a harness hash the cache cannot key an entry with."""
    try:
        EvalCache(Path(), "check").path(CacheKey(harness_hash, "", "", 0, ""))
    except ValueError as error:
        raise UsageError(f"the evaluation cache cannot key this run: {error}") from error


# Usage, from the proxy's records.


@dataclass
class _TaskUsage:
    """What a task's calls through the proxy took: the usage of all of them, and the span of
    those of the last solve container that made any, the wall time of its final attempt."""

    usage: Usage = field(default_factory=Usage)
    wall_time_s: float = 0.0


def _attributed(files: RunFiles, run_id: str, keys: set[_Key]) -> dict[_Key, _TaskUsage]:
    """Each task's usage and wall time, from the proxy's records whose call claimed it."""
    starts = _launch_starts(files)
    found: dict[_Key, _TaskUsage] = {}
    spans: dict[_Key, dict[int, tuple[float, float]]] = {}
    for call in _calls(files, run_id):
        key = (call.instance_id, call.seed)
        if key not in keys:
            continue
        _add(found.setdefault(key, _TaskUsage()).usage, call.usage)
        begin = call.started_at.ToDatetime(tzinfo=UTC).timestamp()
        end = begin + max(call.latency_s, 0.0)
        launch = sum(1 for start in starts if start <= begin)  # Which container made it.
        first, last = spans.setdefault(key, {}).get(launch, (begin, end))
        spans[key][launch] = (min(first, begin), max(last, end))
    for key, by_launch in spans.items():
        first, last = by_launch[max(by_launch)]
        found[key].wall_time_s = last - first
    return found


def _write_usage[I: Instance](files: RunFiles, plan: RunPlan[I]) -> None:
    """Write usage.json: the proxy's totals for the run, and how much of them its tasks account
    for. Warn when they fall short by more than USAGE_TOLERANCE."""
    if not files.llm_calls.exists():
        return  # No solve phase called the proxy: every task came from the cache.
    calls = _calls(files, plan.manifest.run_id)
    total, attributed = Usage(), Usage()
    attributed_calls = 0
    keys = plan.keys
    for call in calls:
        _add(total, call.usage)
        if (call.instance_id, call.seed) in keys:
            _add(attributed, call.usage)
            attributed_calls += 1
    tokens = (
        total.input_tokens + total.output_tokens,
        attributed.input_tokens + attributed.output_tokens,
    )
    shortfall = max(_shortfall(*tokens), _shortfall(total.cost_usd, attributed.cost_usd))
    report = {
        "proxy_total": _usage_fields(total),
        "attributed": _usage_fields(attributed),
        "calls": {"total": len(calls), "attributed": attributed_calls},
        "unattributed_fraction": shortfall,
    }
    write_text_atomically(files.usage, json.dumps(report, indent=2) + "\n")
    if shortfall > USAGE_TOLERANCE:
        _logger.warning(
            "the tasks of run %s account for only %.1f%% of its usage through the proxy:"
            " %d of %d calls claimed one of them; see %s",
            plan.manifest.run_id,
            100 * (1 - shortfall),
            attributed_calls,
            len(calls),
            files.usage,
        )


def _calls(files: RunFiles, run_id: str) -> list[LlmCallRecord]:
    """The proxy's records of the run's calls; none if it made none."""
    if not files.llm_calls.exists():
        return []
    return [call for call in read_jsonl(files.llm_calls, LlmCallRecord) if call.run_id == run_id]


def _launch_starts(files: RunFiles) -> list[float]:
    """When the host started each of the run's solve containers, in order, as POSIX times."""
    if not files.launches.exists():
        return []
    lines = files.launches.read_text(encoding="utf-8").splitlines()
    return [
        datetime.fromisoformat(json.loads(line)["started_at"]).timestamp()
        for line in lines
        if line.strip()
    ]


def _add(total: Usage, usage: Usage) -> None:
    total.input_tokens += usage.input_tokens
    total.output_tokens += usage.output_tokens
    if usage.HasField("reasoning_tokens"):
        total.reasoning_tokens = total.reasoning_tokens + usage.reasoning_tokens
    if usage.HasField("cost_usd"):
        total.cost_usd = total.cost_usd + usage.cost_usd


def _shortfall(total: float, attributed: float) -> float:
    return 0.0 if total <= 0 else max(0.0, (total - attributed) / total)


def _usage_fields(usage: Usage) -> dict[str, object]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens if usage.HasField("reasoning_tokens") else None,
        "cost_usd": usage.cost_usd if usage.HasField("cost_usd") else None,
    }
