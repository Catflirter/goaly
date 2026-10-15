"""TEMPORARY: a thin host driver for the isolated Effi-SQL run, until T06's host pipeline lands.

The controller approved it on 2026-10-15 as a contingency for T11's end-to-end run. Delete this
module, and `python -m goaly.bench.effisql.interim`, once T06's host pipeline (core planning,
the launcher call and host grading behind the V27 hook) is in. It keeps V27 throughout:

- the host plans the run: the tasks and seeds it expects, its manifest and its inputs;
- the solve container writes only ``<run>/solve/``, through the real EffiSqlLauncher;
- the host grades every expected (task, seed) from that directory, as untrusted input: a
  missing, duplicate or malformed record gets POLICY_CRASH and credit 0;
- the run's usage comes from the proxy's own records, written host-side to llm_calls.jsonl,
  which carry each call's task from the client's X-Goaly-Call-Tags; the summary shows them
  beside the container's own counts.

No evaluation cache: every task solves.
"""

import argparse
import asyncio
import json
import logging
import os
import stat
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from google.protobuf import json_format

from goaly.bench.effisql.benchmark import BENCHMARK_NAME, EffiSqlBenchmark
from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.harness import EffiSqlWiring, load_pack, model_settings, open_host
from goaly.bench.effisql.launcher import EffiSqlLauncher, SolveContainerError
from goaly.bench.effisql.pg_instances import confine_to_python_cpus
from goaly.contracts.gen.llm_call_pb2 import (
    LlmCallRecord,
)
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_NONE,
    TASK_ERROR_TYPE_POLICY_CRASH,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_PREDICTION_FAILED,
    SPLIT_HELDOUT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    EffiSqlGrade,
    Grade,
    GradeStatus,
    Prediction,
    Split,
)
from goaly.contracts.gen.trajectory_pb2 import StopReason, Trajectory
from goaly.contracts.jsonl import from_json_line, to_json_line
from goaly.harness.core.atomic import write_text_atomically
from goaly.harness.core.logs import configure_logging
from goaly.harness.core.manifest import (
    harness_hash,
    new_run_id,
    provenance,
    read_manifest,
    write_manifest,
)
from goaly.harness.core.registry import HostBenchmark, HostOptions
from goaly.harness.core.runner import RunFiles, lock_run_dir

__all__ = ["InterimRequest", "SolveRequest", "main", "run_interim"]

_logger = logging.getLogger(__name__)

_SPLITS = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "heldout": SPLIT_HELDOUT}
_PACK = Path("src/goaly/harness/packs/effisql")
_RECORDS_LIMIT = 512 * 1024 * 1024  # The largest records file read from the container.


@dataclass(frozen=True)
class SolveRequest:
    """T06's SolveRequest as the interface describes it, with the CLI's arguments of today."""

    run_id: str
    pack_dir: Path
    solve_dir: Path
    inputs_dir: Path
    llm_calls: Path
    tasks: tuple[tuple[str, int], ...]
    split: Split.ValueType
    budget_phase: str
    budget_share: str
    max_tasks: int
    instances_file: Path | None = None
    task_timeout_s: float | None = None
    benchmark: str = BENCHMARK_NAME
    harness: str = "v0"

    def goaly_args(
        self, *, solve_dir: Path, inputs_dir: Path, instances_file: Path | None = None
    ) -> list[str]:
        ids = sorted({instance_id for instance_id, _ in self.tasks}, key=_numeric)
        seeds = sorted({seed for _, seed in self.tasks})
        where = (
            ["--instances", str(instances_file)]
            if instances_file is not None
            else ["--split", Split.Name(self.split).removeprefix("SPLIT_").lower()]
        )
        args = [
            "run", "--bench", self.benchmark, "--harness", self.harness, "--phase", "solve",
            "--in-container", "--inputs", str(inputs_dir), "--out", str(solve_dir), *where,
            "--only", *ids, "--first-seed", str(seeds[0]), "--seeds", str(len(seeds)),
            "--run-id", self.run_id, "--budget-phase", self.budget_phase,
            "--budget-share", self.budget_share, "--max-tasks", str(self.max_tasks),
        ]  # fmt: skip
        if self.task_timeout_s is not None:
            args += ["--task-timeout", f"{self.task_timeout_s:g}"]
        return args


def _numeric(instance_id: str) -> tuple[int, str]:
    return (int(instance_id), "") if instance_id.isdigit() else (1 << 62, instance_id)


@dataclass(frozen=True)
class InterimRequest:
    out: Path
    split: Split.ValueType
    only: Sequence[str]
    seeds: Sequence[int] = (0,)
    budget_phase: str = "baseline"
    budget_share: str = BENCHMARK_NAME
    max_tasks: int = 16
    harness_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4])
    run_id: str | None = None
    # Grade a run this driver solved before, without solving it again.
    grade_only: bool = False


@dataclass(frozen=True)
class TaskSummary:
    instance_id: str
    seed: int
    status: str
    credit: float
    matched: bool
    speedup: float | None
    fallback: bool
    steps: int
    stop_reason: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    wall_time_s: float
    error: str


async def run_interim(
    request: InterimRequest,
    wiring: EffiSqlWiring | None = None,
    launcher: EffiSqlLauncher | None = None,
) -> dict[str, object]:
    """Run ``request`` through the isolated pipeline and return its summary."""
    wiring = EffiSqlWiring() if wiring is None else wiring
    files = RunFiles(request.out)
    run_id = request.run_id or new_run_id()
    started = time.monotonic()
    llm_calls = request.out / "llm_calls.jsonl"
    with await asyncio.to_thread(lock_run_dir, files):
        if request.grade_only:  # Grade a run solved before, as its host manifest names it.
            if not files.manifest.exists() or files.results.exists():
                raise RuntimeError(f"{request.out} holds no solved run left to grade")
            run_id = read_manifest(files.manifest).run_id
        elif files.manifest.exists():
            raise RuntimeError(f"{request.out} holds a run already")
        options = HostOptions(files, None, request.split, request.budget_phase)
        if launcher is None:
            launcher = EffiSqlLauncher(paths=wiring.paths)
        async with open_host(options, wiring, launcher) as host:
            if not isinstance(host, HostBenchmark):
                raise TypeError("Effi-SQL's host side is a HostBenchmark")
            benchmark = host.benchmark
            if not isinstance(benchmark, EffiSqlBenchmark):
                raise TypeError("Effi-SQL's host side holds an EffiSqlBenchmark")
            wanted = set(request.only)
            instances = [
                i for i in benchmark.load_instances(request.split) if i.instance_id in wanted
            ]
            if len(instances) != len(wanted):
                raise ValueError(f"not all of {sorted(wanted)} are in the split")
            pack_dir = request.harness_dir / _PACK
            if not request.grade_only:
                await asyncio.to_thread(
                    _write_manifest, files, run_id, request, benchmark, instances, pack_dir
                )
            tasks = tuple((i.instance_id, seed) for seed in request.seeds for i in instances)
            solve = SolveRequest(
                run_id=run_id,
                pack_dir=pack_dir,
                solve_dir=request.out / "solve",
                inputs_dir=request.out / "inputs",
                llm_calls=llm_calls,
                tasks=tasks,
                split=request.split,
                budget_phase=request.budget_phase,
                budget_share=request.budget_share,
                max_tasks=request.max_tasks,
            )
            failure: SolveContainerError | None = None
            if not request.grade_only:
                solve.solve_dir.mkdir(parents=True, exist_ok=True)
                try:
                    await launcher(solve)
                except SolveContainerError as error:
                    failure = error
                    _logger.exception("the solve container failed; grading what it left")
            summaries = await _grade(request, benchmark, instances, run_id)
    proxy = _proxy_totals(llm_calls)
    summary = {
        "run_id": run_id,
        "wall_s": round(time.monotonic() - started, 1),
        "container_failure": None if failure is None else str(failure),
        "proxy": proxy,
        "tasks": [s.__dict__ for s in summaries],
        "mean_credit": sum(s.credit for s in summaries) / max(len(summaries), 1),
    }
    write_text_atomically(request.out / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def _write_manifest(
    files: RunFiles,
    run_id: str,
    request: InterimRequest,
    benchmark: EffiSqlBenchmark,
    instances: list[EffiSqlInstance],
    pack_dir: Path,
) -> None:
    config, _ = load_pack(pack_dir)
    manifest = provenance(request.harness_dir)
    manifest.run_id = run_id
    manifest.benchmark = BENCHMARK_NAME
    manifest.split = request.split
    manifest.seeds.extend(request.seeds)
    manifest.instance_ids.extend(i.instance_id for i in instances)
    manifest.harness_hash = harness_hash(pack_dir)
    manifest.models.append(model_settings(config.model))
    manifest.command_line[:] = sys.argv
    manifest.MergeFrom(benchmark.manifest_fields())
    manifest.environment_settings["driver"] = "interim (T11 contingency)"
    manifest.started_at.GetCurrentTime()
    write_manifest(files.manifest, manifest)


def _read_untrusted(path: Path) -> bytes | None:
    """``path``'s bytes, a file the container wrote: None if it is absent.

    The container may have planted a symlink or a special file, so the file is opened with
    O_NOFOLLOW and must be a regular file no larger than the records limit (T18's #48 M8);
    otherwise this raises ValueError, which fails every task the file was to answer.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as error:  # ELOOP: a symlink.
        raise ValueError(f"{path} is not a file the host reads: {error.strerror}") from None
    with os.fdopen(fd, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _RECORDS_LIMIT:
            raise ValueError(f"{path} is not a regular records file within the limit")
        return file.read(_RECORDS_LIMIT + 1)


def _lines[M: (Trajectory, TaskResult)](path: Path, kind: type[M]) -> list[M | None]:
    try:
        raw = _read_untrusted(path)
    except ValueError:
        _logger.exception("an untrusted records file was refused")
        return [None]
    if raw is None:
        return []
    parsed: list[M | None] = []
    for line in raw.split(b"\n"):
        if line.strip():
            try:
                parsed.append(from_json_line(line.decode("utf-8"), kind))
            except (UnicodeDecodeError, json_format.ParseError, ValueError):
                parsed.append(None)
    return parsed


async def _grade(
    request: InterimRequest,
    benchmark: EffiSqlBenchmark,
    instances: list[EffiSqlInstance],
    run_id: str,
) -> list[TaskSummary]:
    solve = request.out / "solve"
    trajectories = await asyncio.to_thread(_lines, solve / "trajectories.jsonl", Trajectory)
    results = await asyncio.to_thread(_lines, solve / "results.jsonl", TaskResult)
    malformed = None in trajectories or None in results
    graded: list[tuple[TaskResult, Trajectory | None]] = []

    async def grade_one(inst: EffiSqlInstance, seed: int) -> None:
        instance_id = inst.instance_id
        mine = [
            t
            for t in trajectories
            if t is not None and (t.instance_id, t.seed) == (instance_id, seed)
        ]
        theirs = [
            r for r in results if r is not None and (r.instance_id, r.seed) == (instance_id, seed)
        ]
        result = TaskResult(run_id=run_id, instance_id=instance_id, seed=seed)
        problem = None
        if not mine or not theirs:
            problem = "malformed" if malformed else "missing"
        elif len(mine) > 1 or len(theirs) > 1:
            problem = "duplicate"
        elif mine[0].run_id != run_id or theirs[0].run_id != run_id:
            problem = "malformed: of another run"
        elif mine[0].prediction.WhichOneof("benchmark") not in (None, "effisql"):
            problem = "malformed: not an Effi-SQL prediction"
        if problem is not None:
            result.error_type = TASK_ERROR_TYPE_POLICY_CRASH
            result.error_message = f"no valid solve record: {problem}"
            result.grade.CopyFrom(
                Grade(
                    instance_id=instance_id,
                    status=GRADE_STATUS_PREDICTION_FAILED,
                    feedback=f"the solve phase left no valid record for this task: {problem}",
                    effisql=EffiSqlGrade(),
                )
            )
            graded.append((result, None))
            return
        claimed = theirs[0]
        result.wall_time_s = claimed.wall_time_s
        result.step_count = claimed.step_count
        result.stop_reason = claimed.stop_reason
        result.error_type = claimed.error_type
        result.error_message = claimed.error_message[:2000]
        result.usage.CopyFrom(claimed.usage)
        prediction = Prediction()
        prediction.CopyFrom(mine[0].prediction)
        prediction.instance_id = instance_id
        result.grade.CopyFrom(await benchmark.grade(inst, prediction))
        graded.append((result, mine[0]))

    async with asyncio.TaskGroup() as group:
        for seed in request.seeds:
            for inst in instances:
                group.create_task(grade_one(inst, seed))
    order = {
        (i.instance_id, s): n
        for n, (s, i) in enumerate((s, i) for s in request.seeds for i in instances)
    }
    graded.sort(key=lambda pair: order[pair[0].instance_id, pair[0].seed])
    await asyncio.to_thread(_write_records, request, benchmark, instances, graded)
    return [_summary(result, trajectory) for result, trajectory in graded]


def _write_records(
    request: InterimRequest,
    benchmark: EffiSqlBenchmark,
    instances: list[EffiSqlInstance],
    graded: list[tuple[TaskResult, Trajectory | None]],
) -> None:
    out = request.out
    write_text_atomically(out / "results.jsonl", "".join(to_json_line(r) + "\n" for r, _ in graded))
    write_text_atomically(
        out / "trajectories.jsonl",
        "".join(to_json_line(t) + "\n" for _, t in graded if t is not None),
    )
    for seed in request.seeds:
        accepted = {
            r.instance_id: t.prediction for r, t in graded if r.seed == seed and t is not None
        }
        lines = [
            json.dumps(
                benchmark.to_official(i, accepted.get(i.instance_id, Prediction())),
                ensure_ascii=False,
            )
            for i in instances
        ]
        (out / "official").mkdir(exist_ok=True)
        write_text_atomically(
            out / "official" / f"seed-{seed}.jsonl", "".join(f"{x}\n" for x in lines)
        )


def _summary(result: TaskResult, trajectory: Trajectory | None) -> TaskSummary:
    grade = result.grade
    effisql = grade.effisql
    return TaskSummary(
        instance_id=result.instance_id,
        seed=result.seed,
        status=GradeStatus.Name(grade.status),
        credit=effisql.credit,
        matched=effisql.matched,
        speedup=effisql.speedup if effisql.HasField("speedup") else None,
        fallback=trajectory is not None and trajectory.prediction.fallback,
        steps=result.step_count,
        stop_reason=StopReason.Name(result.stop_reason),
        tokens_in=result.usage.input_tokens,
        tokens_out=result.usage.output_tokens,
        cost_usd=result.usage.cost_usd,
        wall_time_s=round(result.wall_time_s, 1),
        error="" if result.error_type == TASK_ERROR_TYPE_NONE else result.error_message[:300],
    )


def _call_records(path: Path) -> list[LlmCallRecord | None]:
    if not path.exists():
        return []
    records: list[LlmCallRecord | None] = []
    for line in path.read_bytes().split(b"\n"):
        if line.strip():
            try:
                records.append(from_json_line(line.decode("utf-8"), LlmCallRecord))
            except (UnicodeDecodeError, json_format.ParseError, ValueError):
                records.append(None)
    return records


def _proxy_totals(path: Path) -> dict[str, float]:
    records = [r for r in _call_records(path) if r is not None]
    return {
        "calls": len(records),
        "input_tokens": sum(r.usage.input_tokens for r in records),
        "output_tokens": sum(r.usage.output_tokens for r in records),
        "cost_usd": round(sum(r.usage.cost_usd for r in records), 6),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TEMPORARY isolated Effi-SQL run (T11).")
    parser.add_argument("--split", choices=sorted(_SPLITS), required=True)
    parser.add_argument("--only", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--budget-phase", default="baseline")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grade-only", action="store_true", help="grade a solved run in --out")
    args = parser.parse_args(argv)
    configure_logging("INFO")
    confine_to_python_cpus()
    request = InterimRequest(
        out=args.out,
        split=_SPLITS[args.split],
        only=args.only,
        seeds=args.seeds,
        budget_phase=args.budget_phase,
        grade_only=args.grade_only,
    )
    summary = asyncio.run(run_interim(request))
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
