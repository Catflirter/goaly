"""T21's evaluator seam for the Effi-SQL lane: candidate commits, evaluated on the isolated path.

rsi/lane-effisql.toml names :func:`make_evaluator`. Each evaluation:

1. materialises the commit read-only, with ``git archive``, as a snapshot under the runs
   directory; only the snapshot's pack runs, in the solve container, while every host step runs
   this checkout's trusted code;
2. runs the host command ``goaly run --bench effisql --phase all`` for the request's split, tasks
   and seeds, under ``<runs>/<run_id>``, as a subprocess, so the lane's event loop never blocks;
3. returns the graded results, the host's last line for each (task, seed), and their cost
   statistics, taken from the usage the host attributed from the proxy's records. A per-task
   sum that differs from the proxy's total by more than 1% is logged as a WARNING.

Evaluations run at once share one budget of ``task_concurrency`` tasks. An infrastructure
failure, such as a host command that fails, raises; T21 aborts the experiment.

bench may not import goaly.rsi, so the request is read by its attributes, and the result types
are T21's own classes, which it passes in.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from google.protobuf import json_format

from goaly.bench.effisql.benchmark import BENCHMARK_NAME
from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grading import BaseCache, Mode
from goaly.bench.effisql.grading_pool import GradingInstance, Priority
from goaly.bench.effisql.grading_service import EffiSqlGrader
from goaly.bench.effisql.pg_settings import PgEnvConfig
from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_ENVIRONMENT,
    TASK_ERROR_TYPE_LLM,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import GRADE_STATUS_GRADER_ERROR, SPLIT_TRAIN, Split
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.contracts.jsonl import from_json_line
from goaly.harness.core.manifest import checkout_harness_hash, read_manifest

__all__ = [
    "EffiSqlEvaluator",
    "ProbeGrader",
    "ProxyTotals",
    "TaskBudget",
    "infra_failed",
    "make_evaluator",
]

_logger = logging.getLogger(__name__)

# The pack's place in a checkout, which T06's host hashes, with core, for --harness-checkout.
_PACK = Path("src/goaly/harness/packs/effisql")
# The host CLI's option naming the checkout whose pack the solve container runs (T06).
_CHECKOUT_OPTION = "--harness-checkout"
# How far the per-task usage may drift from the proxy's total before it is flagged.
_DRIFT = 0.01
# How long a cancelled host command has to stop its container and proxy before it is killed.
_TERMINATE_GRACE_S = 60.0

# Runs the host command: (argv, working directory) -> its exit status and its output's end.
type RunCommand = Callable[[Sequence[str], Path], Any]


class EvaluationRequestLike(Protocol):
    """What the evaluator reads of T21's EvaluationRequest."""

    @property
    def commit(self) -> str: ...

    @property
    def tree(self) -> str: ...

    @property
    def split(self) -> Split.ValueType: ...

    @property
    def instance_ids(self) -> tuple[str, ...]: ...

    @property
    def seeds(self) -> tuple[int, ...]: ...

    @property
    def run_id(self) -> str: ...

    @property
    def priority(self) -> object: ...


class TaskBudget:
    """A budget of tasks that solve at once, shared by every evaluation of the lane."""

    def __init__(self, tasks: int) -> None:
        if tasks < 1:
            raise ValueError("the task budget is at least 1")
        self.total = tasks
        self._free = tasks
        self._changed = asyncio.Condition()

    async def acquire(self, wanted: int) -> int:
        """Take up to ``wanted`` tasks of the budget, at least one, and return how many."""
        take = max(1, min(wanted, self.total))
        async with self._changed:
            await self._changed.wait_for(lambda: self._free >= take)
            self._free -= take
        return take

    async def release(self, taken: int) -> None:
        async with self._changed:
            self._free += taken
            self._changed.notify_all()


async def _run_command(argv: Sequence[str], cwd: Path) -> tuple[int, str]:
    """Run the host command; if cancelled, SIGTERM it, which stops its proxy and container, and
    wait for it to end before the cancellation goes on."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await process.communicate()
    except asyncio.CancelledError:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.shield(asyncio.wait_for(process.wait(), _TERMINATE_GRACE_S))
            except TimeoutError:
                process.kill()
                await asyncio.shield(process.wait())
        raise
    return process.returncode or 0, output.decode("utf-8", "replace")[-20_000:]


@dataclass(frozen=True)
class ProxyTotals:
    """The proxy's own totals for a run: input plus output tokens, and the cost."""

    tokens: int
    cost_usd: float


def _proxy_totals(usage_file: Path, calls: Sequence[LlmCallRecord]) -> ProxyTotals:
    """The run's proxy totals from the host's usage.json, or from its call records without it."""
    try:
        total = json.loads(usage_file.read_text(encoding="utf-8"))["proxy_total"]
        return ProxyTotals(
            int(total["input_tokens"]) + int(total["output_tokens"]), float(total["cost_usd"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        tokens = sum(c.usage.input_tokens + c.usage.output_tokens for c in calls)
        return ProxyTotals(tokens, sum(c.usage.cost_usd for c in calls))


# The host's error types that are the infrastructure's, not the candidate's (T21's ruling).
_INFRA_ERRORS = frozenset({TASK_ERROR_TYPE_LLM, TASK_ERROR_TYPE_ENVIRONMENT})


def infra_failed(results: Sequence[TaskResult]) -> frozenset[tuple[str, int]]:
    """The (task, seed) pairs the infrastructure failed, by the host's graded results: an LLM
    gateway outage after the retries, an environment failure, or a grader error."""
    return frozenset(
        (r.instance_id, r.seed)
        for r in results
        if r.error_type in _INFRA_ERRORS or r.grade.status == GRADE_STATUS_GRADER_ERROR
    )


class ProbeGrader:
    """T21's identity probes (§6.2): a grader made for each probe and closed right after.

    T10 allows one grader per instance at a time, and the host run each evaluation starts makes
    its own, so the evaluator holds none: a probe waits for its instance's lock as any grading
    does, and T21 runs them once a round.
    """

    def __init__(self, cache_root: Path) -> None:
        self._cache_root = cache_root
        self._lock = asyncio.Lock()

    async def identity_probe(
        self, inst: EffiSqlInstance, *, instance: GradingInstance, priority: Priority = Priority.RSI
    ) -> float:
        async with self._lock:
            grader = await asyncio.to_thread(
                EffiSqlGrader.on_instances, (instance,), cache_root=self._cache_root
            )
            try:
                return await grader.identity_probe(inst, instance=instance, priority=priority)
            finally:
                await asyncio.to_thread(grader.close)

    def identity_reference(self, inst: EffiSqlInstance) -> float | None:
        """The identity τ the setup measured for ``inst``, from the base cache; no grader."""
        entry = BaseCache(self._cache_root / Mode.DEV.value).get(inst)
        return None if entry is None else entry.identity_speedup


@dataclass
class EffiSqlEvaluator:
    """T21's Evaluator for the Effi-SQL lane; see the module docstring."""

    repo: Path
    runs_dir: Path
    budget: TaskBudget
    evaluation_type: Callable[..., Any]
    cost_stats_type: Callable[..., Any]
    run_command: RunCommand = _run_command
    python: str = sys.executable
    grading_cache_root: Path = field(default_factory=lambda: PgEnvConfig().data_root / "grader")

    def __post_init__(self) -> None:
        self._probes = ProbeGrader(self.grading_cache_root)

    async def evaluate(self, request: EvaluationRequestLike) -> Any:
        """Evaluate ``request``'s commit on its tasks and seeds; raise on an infrastructure
        failure."""
        snapshot = await asyncio.to_thread(self.snapshot, request.commit, request.tree)
        run_dir = self.runs_dir / request.run_id
        tasks = len(request.instance_ids) * len(request.seeds)
        taken = await self.budget.acquire(tasks)
        try:
            argv = self.command(request, snapshot, run_dir, taken)
            status, output = await self.run_command(argv, self.repo)
        finally:
            await self.budget.release(taken)
        if status != 0:
            raise RuntimeError(f"the host run {request.run_id} exited {status}: {output[-2000:]}")
        results = await asyncio.to_thread(_last_results, run_dir / "results.jsonl")
        wanted = {(i, s) for i in request.instance_ids for s in request.seeds}
        chosen = [results[key] for key in sorted(wanted) if key in results]
        calls = await asyncio.to_thread(_records, run_dir / "llm_calls.jsonl", LlmCallRecord)
        mine = [c for c in calls if c.run_id == request.run_id]
        totals = await asyncio.to_thread(_proxy_totals, run_dir / "usage.json", mine)
        stats = self.cost_stats(chosen, mine, totals)
        return self.evaluation_type(results=chosen, stats=stats, infra_failed=infra_failed(chosen))

    def command(
        self, request: EvaluationRequestLike, snapshot: Path, run_dir: Path, max_tasks: int
    ) -> list[str]:
        """The host command for ``request``: `goaly run --phase all` on the snapshot's pack."""
        seeds = sorted(request.seeds)
        if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
            raise ValueError(f"the seeds must be contiguous: {seeds}")
        split = Split.Name(request.split).removeprefix("SPLIT_").lower()
        phase = "rsi"
        return [
            self.python, "-m", "goaly", "run", "--bench", BENCHMARK_NAME,
            _CHECKOUT_OPTION, str(snapshot), "--split", split, "--only", *request.instance_ids,
            "--first-seed", str(seeds[0]), "--seeds", str(len(seeds)),
            "--run-id", request.run_id, "--budget-phase", phase,
            "--grading-priority", _priority_name(request.priority),
            "--max-tasks", str(max_tasks), "--out", str(run_dir),
        ]  # fmt: skip

    def cost_stats(
        self,
        results: Sequence[TaskResult],
        calls: Sequence[LlmCallRecord],
        totals: ProxyTotals,
    ) -> Any:
        """The means per (task, seed), host-verified (the controller's ruling for T21).

        Tokens and cost are the proxy's own totals for the run (usage.json), which cover every
        task solved now; a task the cache answered, which made no call, brings the host numbers
        stored with it. Wall time is the host's, per task. The container's own counts are never
        used. A per-task sum that differs from the proxy's total by more than 1% is a WARNING.
        """
        count = len(results)
        if not count:
            return self.cost_stats_type(
                mean_tokens=0.0, mean_cost_usd=None, mean_wall_time_s=0.0, count=0
            )
        solved = {(c.instance_id, c.seed) for c in calls}
        cached = [r for r in results if (r.instance_id, r.seed) not in solved]
        fresh = [r for r in results if (r.instance_id, r.seed) in solved]
        tokens = totals.tokens + sum(r.usage.input_tokens + r.usage.output_tokens for r in cached)
        cost = totals.cost_usd + sum(r.usage.cost_usd for r in cached)
        attributed = sum(r.usage.input_tokens + r.usage.output_tokens for r in fresh)
        if totals.tokens and abs(attributed - totals.tokens) > _DRIFT * totals.tokens:
            _logger.warning(
                "the tasks' tokens differ from the proxy's total by more than 1%%",
                extra={"tasks": attributed, "proxy": totals.tokens},
            )
        priced = totals.cost_usd > 0 or any(r.usage.cost_usd for r in cached)
        return self.cost_stats_type(
            mean_tokens=tokens / count,
            mean_cost_usd=cost / count if priced else None,
            mean_wall_time_s=sum(r.wall_time_s for r in results) / count,
            count=count,
        )

    @property
    def grader(self) -> ProbeGrader:
        """T21's identity probes, on a grader that exists only while one runs (the ruling)."""
        return self._probes

    async def aclose(self) -> None:
        """Release what the evaluator holds; it starts no pool of its own yet."""

    def snapshot(self, commit: str, tree: str) -> Path:
        """The commit's files, read-only, under ``<runs>/snapshots/<commit>``; made once.

        Its tree must be ``tree``, the identity the host computed.
        """
        target = self.runs_dir / "snapshots" / commit
        if target.is_dir():
            return target
        actual = _git(self.repo, "rev-parse", f"{commit}^{{tree}}").strip()
        if actual != tree:
            raise RuntimeError(f"commit {commit} has tree {actual}, not {tree}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{commit[:12]}-", dir=target.parent))
        try:
            archive = staging / "archive.tar"
            _git(self.repo, "archive", "--format=tar", f"--output={archive}", commit)
            with tarfile.open(archive) as tar:
                tar.extractall(staging / "tree", filter="data")
            archive.unlink()
            try:
                (staging / "tree").rename(target)  # Before it is read-only, which refuses it.
            except OSError:
                if not target.is_dir():
                    raise
                return target  # Another evaluation made the same snapshot meanwhile.
            _read_only(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return target

    def known_train(self, commit: str) -> tuple[Sequence[TaskResult], Sequence[Trajectory]]:
        """The graded train results and trajectories the runs directory holds for ``commit``.

        A run counts if its manifest is of the Effi-SQL train split, finished, and has the
        harness hash of the commit's pack: v0's T12 train run among them. Later runs win.
        """
        snapshot = self.runs_dir / "snapshots" / commit
        if not snapshot.is_dir():
            try:
                tree = _git(self.repo, "rev-parse", f"{commit}^{{tree}}").strip()
                snapshot = self.snapshot(commit, tree)
            except (RuntimeError, OSError):
                return [], []
        wanted = checkout_harness_hash(snapshot, _PACK)  # As the host hashes --harness-checkout.
        results: dict[tuple[str, int], TaskResult] = {}
        trajectories: dict[tuple[str, int], Trajectory] = {}
        for manifest_path in sorted(self.runs_dir.glob("*/manifest.json")):
            manifest = _manifest(manifest_path)
            if (
                manifest is None
                or manifest.benchmark != BENCHMARK_NAME
                or manifest.split != SPLIT_TRAIN
                or manifest.harness_hash != wanted
                or not manifest.HasField("finished_at")
            ):
                continue
            run = manifest_path.parent
            results.update(
                (key, r)
                for key, r in _last_results(run / "results.jsonl").items()
                if r.HasField("grade")
            )
            for t in _records(run / "trajectories.jsonl", Trajectory):
                trajectories[t.instance_id, t.seed] = t
        keys = sorted(results)
        return [results[k] for k in keys], [trajectories[k] for k in keys if k in trajectories]


def make_evaluator(
    *,
    repo: Path,
    runs_dir: Path,
    task_concurrency: int,
    evaluation_type: Callable[..., Any],
    cost_stats_type: Callable[..., Any],
) -> EffiSqlEvaluator:
    """T21's factory, called once per lane process (rsi/lane-effisql.toml's ``evaluator``)."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    return EffiSqlEvaluator(
        repo=repo,
        runs_dir=runs_dir,
        budget=TaskBudget(task_concurrency),
        evaluation_type=evaluation_type,
        cost_stats_type=cost_stats_type,
    )


def _priority_name(priority: object) -> str:
    """The request's grading priority as `--grading-priority` takes it: RSI's if it has none."""
    if isinstance(priority, Priority):
        return priority.name.lower()
    return Priority.RSI.name.lower()


def _git(repo: Path, *args: str) -> str:
    import subprocess

    completed = subprocess.run(  # noqa: S603 - fixed git arguments, a commit from the ledger.
        ["git", "-C", str(repo), *args],  # noqa: S607 - git on PATH, as everywhere in goaly.
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {completed.stderr.strip()[-500:]}")
    return completed.stdout


def _read_only(root: Path) -> None:
    for directory, _, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            path.chmod(path.stat().st_mode & 0o555)
    for directory, _, _ in os.walk(root, topdown=False):
        Path(directory).chmod(0o555)


def _records[M: (TaskResult, Trajectory, LlmCallRecord)](path: Path, kind: type[M]) -> list[M]:
    """The whole lines of ``path`` read as ``kind``; a torn or bad line is skipped."""
    if not path.exists():
        return []
    found = []
    for line in path.read_bytes().split(b"\n"):
        if line.strip():
            try:
                found.append(from_json_line(line.decode("utf-8"), kind))
            except (UnicodeDecodeError, json_format.ParseError, ValueError):
                continue
    return found


def _last_results(path: Path) -> dict[tuple[str, int], TaskResult]:
    """Each (task, seed)'s last result line: a later line wins (result.proto)."""
    return {(r.instance_id, r.seed): r for r in _records(path, TaskResult)}


def _manifest(path: Path) -> RunManifest | None:
    try:
        return read_manifest(path)
    except (OSError, ValueError, json_format.ParseError):
        return None
