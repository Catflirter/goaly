"""The grade phase of a two-phase run (spec V24): grade every task its solve phase recorded.

The solve phase runs the pack's policy, candidate code that may run in a container, and records
each task ungraded (Runner.run with grading=False). The grade phase runs only the benchmark's
grader, imports no pack code, and needs no LLM. It reads the run directory the solve phase
wrote: the manifest, and each task's last result line and last trajectory line, whose
prediction the official files submitted. Each task whose last result has no grade is graded,
and a graded copy of that result is appended to results.jsonl. A later line for a task wins
(result.proto), so each task ends with exactly one graded line, which supersedes its ungraded
one. An interrupted grade phase resumes the same way: a task graded already is not graded
again. Once every task is graded, the manifest gets its finished_at.

The grade phase writes the evaluation cache, which holds graded evaluations only: a clean one,
with the grade in its trajectory too, so a later run, in either phase, takes it from there.
"""

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import TASK_ERROR_TYPE_NONE, TaskErrorType, TaskResult
from goaly.contracts.gen.task_pb2 import SPLIT_EXTERNAL, Prediction
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.contracts.interfaces import Benchmark, Environment, Instance
from goaly.harness.core.eval_cache import CachedEvaluation, CacheKey, EvalCache, task_digest
from goaly.harness.core.jsonl_writer import JsonlWriter
from goaly.harness.core.manifest import read_manifest, write_manifest
from goaly.harness.core.runner import (
    RunDirectoryError,
    RunFiles,
    RunLock,
    RunSummary,
    cacheable,
    grade_prediction,
    last_records,
    lock_run_dir,
)

__all__ = ["Grader"]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Ungraded[InstanceT]:
    """A task the solve phase recorded and nothing graded yet."""

    inst: InstanceT
    result: TaskResult
    trajectory: Trajectory


@dataclass(frozen=True)
class _Plan[InstanceT]:
    manifest: RunManifest
    tasks: int
    graded: int  # Graded already: by an earlier grade phase, or taken from the cache.
    ungraded: list[_Ungraded[InstanceT]]


class Grader[InstanceT: Instance, EnvT: Environment]:
    """Grades the tasks of a solved run with ``benchmark``'s grader.

    With a ``cache``, a clean graded evaluation goes into it.
    """

    def __init__(self, benchmark: Benchmark[InstanceT, EnvT], *, cache: EvalCache | None = None):
        self._benchmark = benchmark
        self._cache = cache

    async def grade(self, run_dir: Path, *, lock: RunLock | None = None) -> RunSummary:
        """Grade every task of the run in ``run_dir`` that has no grade, and finish the run.

        Raise RunDirectoryError if the directory holds no run, a run of another benchmark, one
        on SPLIT_EXTERNAL, which has no gold answers, or one whose solve phase did not record
        every task. The grade phase holds the directory's lock throughout; a caller that holds
        it already passes it as ``lock``.
        """
        files = RunFiles(run_dir)
        if lock is not None and lock.files != files:
            raise ValueError(f"the lock given is on {lock.files.root}, not on {run_dir}")
        if not await asyncio.to_thread(files.manifest.exists):  # Before the lock creates it.
            raise RunDirectoryError(f"{run_dir} holds no run to grade")
        own_lock = None if lock is not None else await asyncio.to_thread(lock_run_dir, files)
        try:
            plan = await asyncio.to_thread(self._plan, files)
            errors = await self._grade_all(files, plan)
            plan.manifest.finished_at.GetCurrentTime()
            await asyncio.to_thread(write_manifest, files.manifest, plan.manifest)
        finally:
            if own_lock is not None:
                own_lock.release()
        summary = RunSummary(
            run_id=plan.manifest.run_id,
            tasks=plan.tasks,
            resumed=plan.graded,
            cached=0,
            ran=len(plan.ungraded),
            errors=dict(errors),
        )
        _logger.info("graded run %s: %s", summary.run_id, summary)
        return summary

    def _plan(self, files: RunFiles) -> _Plan[InstanceT]:
        manifest = read_manifest(files.manifest)
        where = f"the run {manifest.run_id} in {files.root}"
        if manifest.benchmark != self._benchmark.name:
            raise RunDirectoryError(
                f"{where} is of the benchmark {manifest.benchmark!r}, not {self._benchmark.name!r}"
            )
        if manifest.split == SPLIT_EXTERNAL:
            raise RunDirectoryError(f"{where} is on SPLIT_EXTERNAL, which has no gold answers")
        if self._cache is not None and not manifest.harness_hash:
            raise ValueError(f"the evaluation cache needs the harness_hash {where} lacks")
        by_id = {inst.instance_id: inst for inst in self._benchmark.load_instances(manifest.split)}
        unknown = [instance_id for instance_id in manifest.instance_ids if instance_id not in by_id]
        if unknown:
            raise RunDirectoryError(
                f"the benchmark has no instance {', '.join(map(repr, unknown))}"
            )
        # Either may be missing if the solve phase stopped before it recorded anything.
        results = last_records(files.results, TaskResult) if files.results.exists() else {}
        trajectories = (
            last_records(files.trajectories, Trajectory) if files.trajectories.exists() else {}
        )
        ungraded = []
        graded = 0
        for seed in manifest.seeds:
            for instance_id in manifest.instance_ids:
                key = (instance_id, seed)
                if key not in results or key not in trajectories:
                    raise RunDirectoryError(
                        f"{where} has no record of {key}: resume its solve phase first"
                    )
                if results[key].HasField("grade"):
                    graded += 1
                else:
                    ungraded.append(_Ungraded(by_id[instance_id], results[key], trajectories[key]))
        tasks = len(manifest.instance_ids) * len(manifest.seeds)
        return _Plan(manifest, tasks, graded, ungraded)

    async def _grade_all(self, files: RunFiles, plan: _Plan[InstanceT]) -> Counter[str]:
        errors: Counter[str] = Counter()
        async with JsonlWriter[TaskResult](files.results) as results, asyncio.TaskGroup() as group:
            for task in plan.ungraded:
                name = f"grading {task.result.instance_id} seed {task.result.seed}"
                group.create_task(self._grade(task, plan.manifest, results, errors), name=name)
        return errors

    async def _grade(
        self,
        task: _Ungraded[InstanceT],
        manifest: RunManifest,
        results: JsonlWriter[TaskResult],
        errors: Counter[str],
    ) -> None:
        prediction = Prediction()
        prediction.CopyFrom(task.trajectory.prediction)  # Empty, the empty answer, if it had none.
        prediction.instance_id = task.inst.instance_id
        grade = await grade_prediction(self._benchmark, task.inst, prediction)
        result = TaskResult()
        result.CopyFrom(task.result)
        result.grade.CopyFrom(grade)
        await results.write(result)
        if result.error_type != TASK_ERROR_TYPE_NONE:
            errors[TaskErrorType.Name(result.error_type)] += 1
        if self._cache is not None and cacheable(result):
            trajectory = Trajectory()
            trajectory.CopyFrom(task.trajectory)
            trajectory.grade.CopyFrom(grade)
            key = CacheKey(
                harness_hash=manifest.harness_hash,
                benchmark=manifest.benchmark,
                instance_id=result.instance_id,
                seed=result.seed,
                task_digest=task_digest(trajectory.task),
            )
            try:
                await self._cache.put(key, CachedEvaluation(result, trajectory))
            except OSError as error:
                _logger.warning(
                    "could not cache the evaluation of %s, seed %d: %s",
                    key.instance_id,
                    key.seed,
                    error,
                )
