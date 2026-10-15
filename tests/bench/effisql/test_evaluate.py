"""Tests for goaly.bench.effisql.evaluate, T21's evaluator seam, with a fake host command."""

import asyncio
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.evaluate import (
    EffiSqlEvaluator,
    TaskBudget,
    _run_command,  # The child's lifetime, tested directly.
    infra_failed,
    make_evaluator,
)
from goaly.bench.effisql.grading_pool import GRADING_INSTANCES, Priority
from goaly.bench.effisql.grading_service import EffiSqlGrader
from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord, Usage
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_ENVIRONMENT,
    TASK_ERROR_TYPE_LLM,
    TASK_ERROR_TYPE_NONE,
    TASK_ERROR_TYPE_POLICY_CRASH,
    TaskErrorType,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    SPLIT_TRAIN,
    SPLIT_VAL,
    Grade,
    GradeStatus,
    Split,
)
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.contracts.jsonl import to_json_line
from goaly.harness.core.manifest import content_hash, write_manifest
from tests.bench.effisql.synthetic import task

pytestmark = pytest.mark.anyio


@dataclass(frozen=True)
class Evaluation:  # T21's goaly.rsi.round.Evaluation, as T21 passes it in.
    results: Sequence[TaskResult]
    stats: object
    infra_failed: object = None


@dataclass(frozen=True)
class CostStats:  # T21's goaly.rsi.decision.CostStats.
    mean_tokens: float
    mean_cost_usd: float | None
    mean_wall_time_s: float
    count: int = 1


@dataclass(frozen=True)
class Request:  # T21's EvaluationRequest, as the evaluator reads it.
    commit: str
    tree: str
    split: Split.ValueType
    instance_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    run_id: str
    priority: object = None


async def test_a_smoke_request_grades_first(repo: Path, tmp_path: Path) -> None:
    commit, tree = head(repo)
    host = FakeHost([], [])
    request = Request(commit, tree, SPLIT_TRAIN, ("1",), (0,), "smoke-1", priority=Priority.SMOKE)

    await evaluator(repo, tmp_path, host).evaluate(request)

    argv = host.argv[0]
    assert argv[argv.index("--grading-priority") + 1] == "smoke"


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed git arguments in a test repo.
        ["git", "-C", str(repo), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    pack = root / "src" / "goaly" / "harness" / "packs" / "effisql"
    core = root / "src" / "goaly" / "harness" / "core"
    pack.mkdir(parents=True)
    core.mkdir(parents=True)
    (pack / "policy.py").write_text("x = 1\n")
    (core / "agent.py").write_text("y = 2\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "-c", "user.email=t@e", "-c", "user.name=t", "add", ".")
    git(root, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "-m", "c")
    return root


def head(repo: Path) -> tuple[str, str]:
    return git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "HEAD^{tree}")


def result(instance_id: str, seed: int, tokens: int, cost: float, wall: float) -> TaskResult:
    return TaskResult(
        run_id="run-1",
        instance_id=instance_id,
        seed=seed,
        wall_time_s=wall,
        usage=Usage(input_tokens=tokens, output_tokens=0, cost_usd=cost),
        grade=Grade(instance_id=instance_id, score=0.5),
    )


class FakeHost:
    """The host command: it writes the run's results and the proxy's records."""

    def __init__(
        self,
        results: Sequence[TaskResult],
        calls: Sequence[LlmCallRecord],
        status: int = 0,
        usage: dict[str, float] | None = None,
        sent: Sequence[tuple[str, int]] | None = None,
        wall_s: float = 40.0,
    ):
        self.usage = usage
        self.sent = sent
        self.wall_s = wall_s
        self.results = results
        self.calls = calls
        self.status = status
        self.argv: list[list[str]] = []
        self.running = 0
        self.peak = 0

    async def __call__(self, argv: Sequence[str], cwd: Path) -> tuple[int, str]:
        self.argv.append(list(argv))
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.jsonl").write_text("".join(to_json_line(r) + "\n" for r in self.results))
        (out / "llm_calls.jsonl").write_text("".join(to_json_line(c) + "\n" for c in self.calls))
        if self.usage is not None:
            (out / "usage.json").write_text(json.dumps({"proxy_total": self.usage}))
        sent = self.sent
        if sent is None:  # By default the host sent every requested task to the container.
            only = argv[argv.index("--only") + 1 : argv.index("--first-seed")]
            first = int(argv[argv.index("--first-seed") + 1])
            seeds = range(first, first + int(argv[argv.index("--seeds") + 1]))
            sent = [(i, s) for i in only for s in seeds]
        max_tasks = int(argv[argv.index("--max-tasks") + 1])
        run = {"tasks": [list(t) for t in sent], "wall_s": self.wall_s, "max_tasks": max_tasks}
        run |= {"started_at": 0.0, "exit_code": 0, "timed_out": False}
        (out / "solve-runs.jsonl").write_text(json.dumps(run) + "\n")
        return self.status, "host output"


def evaluator(repo: Path, tmp_path: Path, host: FakeHost, budget: int = 16) -> EffiSqlEvaluator:
    made = make_evaluator(
        repo=repo,
        runs_dir=tmp_path / "runs",
        task_concurrency=budget,
        evaluation_type=Evaluation,
        cost_stats_type=CostStats,
    )
    made.run_command = host
    return made


def calls(*tokens: int, run_id: str = "run-1") -> list[LlmCallRecord]:
    return [LlmCallRecord(run_id=run_id, usage=Usage(input_tokens=t)) for t in tokens]


def tagged(instance_id: str, tokens: int, cost: float) -> LlmCallRecord:
    """A proxy record of one of the run's calls, tagged with its task (X-Goaly-Call-Tags)."""
    return LlmCallRecord(
        run_id="run-1",
        instance_id=instance_id,
        seed=0,
        http_status=200,
        usage=Usage(input_tokens=tokens, cost_usd=cost),
    )


async def test_it_runs_the_host_command_on_a_read_only_snapshot(repo: Path, tmp_path: Path) -> None:
    commit, tree = head(repo)
    host = FakeHost(
        [
            result("1", 0, 100, 0.01, 10),
            result("1", 0, 300, 0.03, 30),
            result("2", 0, 200, 0.02, 20),
        ],
        [tagged("1", 300, 0.03), tagged("2", 200, 0.02)],
    )
    request = Request(commit, tree, SPLIT_TRAIN, ("1", "2"), (0,), "run-1")

    evaluation = await evaluator(repo, tmp_path, host).evaluate(request)

    (argv,) = host.argv
    snapshot = tmp_path / "runs" / "snapshots" / commit
    assert argv[argv.index("--harness-checkout") + 1] == str(snapshot)
    assert argv[argv.index("--split") + 1] == "train"
    assert argv[argv.index("--only") + 1 : argv.index("--first-seed")] == ["1", "2"]
    assert argv[argv.index("--seeds") + 1] == "1"
    assert argv[argv.index("--max-tasks") + 1] == "2"
    assert argv[argv.index("--grading-priority") + 1] == "rsi"
    assert argv[argv.index("--out") + 1] == str(tmp_path / "runs" / "run-1")
    assert (snapshot / "src" / "goaly" / "harness" / "packs" / "effisql" / "policy.py").exists()
    with pytest.raises(PermissionError):
        (snapshot / "new").write_text("x")
    assert [(r.instance_id, r.usage.input_tokens) for r in evaluation.results] == [
        ("1", 300),
        ("2", 200),
    ]
    stats = evaluation.stats
    # The proxy's 500 tokens over the 2 tasks sent; 40 s of host clock / 2 tasks x 2 at once.
    assert (stats.mean_tokens, stats.mean_wall_time_s, stats.count) == (250.0, 40.0, 2)
    assert stats.mean_cost_usd == pytest.approx(0.025)


async def test_a_failed_host_run_raises(repo: Path, tmp_path: Path) -> None:
    commit, tree = head(repo)
    host = FakeHost([], [], status=1)

    with pytest.raises(RuntimeError, match="exited 1"):
        await evaluator(repo, tmp_path, host).evaluate(
            Request(commit, tree, SPLIT_VAL, ("1",), (0,), "run-2")
        )


async def test_a_wrong_tree_is_refused(repo: Path, tmp_path: Path) -> None:
    commit, _ = head(repo)

    with pytest.raises(RuntimeError, match="has tree"):
        await evaluator(repo, tmp_path, FakeHost([], [])).evaluate(
            Request(commit, "0" * 40, SPLIT_TRAIN, ("1",), (0,), "run-3")
        )


async def test_seeds_must_be_contiguous(repo: Path, tmp_path: Path) -> None:
    commit, tree = head(repo)

    with pytest.raises(ValueError, match="contiguous"):
        await evaluator(repo, tmp_path, FakeHost([], [])).evaluate(
            Request(commit, tree, SPLIT_TRAIN, ("1",), (0, 2), "run-4")
        )


async def test_concurrent_evaluations_share_the_task_budget(repo: Path, tmp_path: Path) -> None:
    commit, tree = head(repo)
    host = FakeHost([], [])
    made = evaluator(repo, tmp_path, host, budget=3)

    await asyncio.gather(
        *(
            made.evaluate(Request(commit, tree, SPLIT_TRAIN, ("1", "2"), (0,), f"r{n}"))
            for n in range(3)
        )
    )

    assert host.peak == 1  # Each takes 2 of 3 tasks: no two at once.


async def test_the_budget_never_takes_more_than_it_has() -> None:
    budget = TaskBudget(4)

    assert await budget.acquire(10) == 4
    await budget.release(4)
    assert await budget.acquire(0) == 1


def test_known_train_finds_the_commits_finished_train_runs(repo: Path, tmp_path: Path) -> None:
    commit, tree = head(repo)
    made = evaluator(repo, tmp_path, FakeHost([], []))
    snapshot = made.snapshot(commit, tree)
    pack = snapshot / "src" / "goaly" / "harness" / "packs" / "effisql"
    wanted = content_hash({"core": snapshot / "src" / "goaly" / "harness" / "core", "pack": pack})

    def run(name: str, split: Split.ValueType, digest: str, finished: bool) -> None:
        root = tmp_path / "runs" / name
        root.mkdir(parents=True)
        manifest = RunManifest(benchmark="effisql", split=split, harness_hash=digest)
        if finished:
            manifest.finished_at.GetCurrentTime()
        write_manifest(root / "manifest.json", manifest)
        (root / "results.jsonl").write_text(to_json_line(result("1", 0, 1, 0, 1)) + "\n")
        (root / "trajectories.jsonl").write_text(
            to_json_line(Trajectory(instance_id="1", seed=0)) + "\n"
        )

    run("v0-train", SPLIT_TRAIN, wanted, finished=True)
    run("val", SPLIT_VAL, wanted, finished=True)
    run("other", SPLIT_TRAIN, "f" * 64, finished=True)
    run("unfinished", SPLIT_TRAIN, wanted, finished=False)

    results, trajectories = made.known_train(commit)

    assert [(r.instance_id, r.seed) for r in results] == [("1", 0)]
    assert [(t.instance_id, t.seed) for t in trajectories] == [("1", 0)]
    assert made.known_train("0" * 40) == ([], [])


async def test_a_cached_task_brings_its_stored_host_numbers(repo: Path, tmp_path: Path) -> None:
    commit, tree = head(repo)
    # Task 1 solved now (the proxy saw it); task 2 came from the cache (no call was made).
    host = FakeHost(
        [result("1", 0, 300, 0.03, 30), result("2", 0, 100, 0.01, 10)],
        [tagged("1", 300, 0.03)],
        sent=[("1", 0)],  # The host took task 2 from the cache before the launch.
        wall_s=12.0,
    )

    evaluation = await evaluator(repo, tmp_path, host).evaluate(
        Request(commit, tree, SPLIT_TRAIN, ("1", "2"), (0,), "run-1")
    )

    stats = evaluation.stats
    assert stats.mean_tokens == 200.0  # (300 from the proxy + 100 stored) / 2
    assert stats.mean_cost_usd == pytest.approx(0.02)
    assert stats.mean_wall_time_s == 11.0  # (12 s of host clock + 10 s stored) / 2
    assert evaluation.infra_failed == frozenset()  # The host saw no infrastructure failure.


async def test_a_cancelled_evaluation_stops_its_host_command(tmp_path: Path) -> None:
    started = tmp_path / "started"
    script = f"import pathlib, time; pathlib.Path({str(started)!r}).touch(); time.sleep(60)"
    running = asyncio.create_task(_run_command([sys.executable, "-c", script], tmp_path))
    while not started.exists():
        await asyncio.sleep(0.05)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


async def test_the_proxy_totals_come_from_usage_json(repo: Path, tmp_path: Path) -> None:
    commit, tree = head(repo)
    usage = {"input_tokens": 900, "output_tokens": 100, "reasoning_tokens": 50, "cost_usd": 0.5}
    host = FakeHost([result("1", 0, 300, 0.03, 30)], [tagged("1", 300, 0.03)], usage=usage)

    evaluation = await evaluator(repo, tmp_path, host).evaluate(
        Request(commit, tree, SPLIT_TRAIN, ("1",), (0,), "run-1")
    )

    assert evaluation.stats.mean_tokens == 1000.0  # The proxy's total, not the call log's.
    assert evaluation.stats.mean_cost_usd == pytest.approx(0.5)


def _result(
    instance_id: str,
    error: TaskErrorType.ValueType = TASK_ERROR_TYPE_NONE,
    status: GradeStatus.ValueType = GRADE_STATUS_GRADED,
) -> TaskResult:
    return TaskResult(instance_id=instance_id, seed=0, error_type=error, grade=Grade(status=status))


def _call(instance_id: str, http_status: int, at_s: int) -> LlmCallRecord:
    call = LlmCallRecord(run_id="run-1", instance_id=instance_id, seed=0, http_status=http_status)
    call.started_at.FromSeconds(at_s)
    return call


def test_forged_container_errors_are_not_infrastructure_failures() -> None:
    # The container claims LLM and environment failures, with no host evidence for either.
    forged = [_result("1", TASK_ERROR_TYPE_LLM), _result("2", TASK_ERROR_TYPE_ENVIRONMENT)]
    calls = [_call("1", 200, 1), _call("2", 200, 1)]

    assert infra_failed(forged, calls) == frozenset()


def test_infra_failures_rest_on_the_proxy_the_launcher_and_the_grader() -> None:
    results = [
        _result("1", TASK_ERROR_TYPE_LLM),  # Its last call got a 503 at the proxy.
        _result("2"),  # A 502 that a retry recovered from: no failure.
        _result("3", status=GRADE_STATUS_GRADER_ERROR),  # The host's own grade.
        _result("4", TASK_ERROR_TYPE_POLICY_CRASH),  # No record, but the container exited 0.
        _result("5", TASK_ERROR_TYPE_LLM),  # The proxy saw no reply: a timeout.
    ]
    calls = [
        _call("1", 200, 1),
        _call("1", 503, 2),
        _call("2", 502, 1),
        _call("2", 200, 2),
        _call("5", 0, 1),
        _call("9", 503, 1),  # Another task's call: not this evaluation's.
    ]

    assert infra_failed(results, calls) == {("1", 0), ("3", 0), ("5", 0)}
    assert infra_failed(results, calls, container_failed=True) == {
        ("1", 0),
        ("3", 0),
        ("4", 0),
        ("5", 0),
    }


async def test_a_probe_makes_a_grader_and_closes_it(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[FakeProbeGrader] = []

    def on_instances(instances: object, *, cache_root: Path) -> FakeProbeGrader:
        made.append(FakeProbeGrader(instances))
        return made[-1]

    monkeypatch.setattr(EffiSqlGrader, "on_instances", on_instances)
    grader = evaluator(repo, tmp_path, FakeHost([], [])).grader
    inst = EffiSqlInstance.model_validate(task(id=5))
    instance = GRADING_INSTANCES[1]

    tau = await grader.identity_probe(inst, instance=instance, priority=Priority.SMOKE)

    assert tau == 1.01
    assert [(g.instances, g.probed, g.closed) for g in made] == [
        ((instance,), [("5", instance, Priority.SMOKE)], True)
    ]
    assert grader.identity_reference(inst) is None  # No base cache here; no grader started.
    assert len(made) == 1


class FakeProbeGrader:
    def __init__(self, instances: object) -> None:
        self.instances = instances
        self.probed: list[tuple[str, object, object]] = []
        self.closed = False

    async def identity_probe(
        self, inst: EffiSqlInstance, *, instance: object, priority: object
    ) -> float:
        self.probed.append((inst.instance_id, instance, priority))
        return 1.01

    def close(self) -> None:
        self.closed = True


async def test_a_forged_tag_cannot_make_a_task_look_cached(repo: Path, tmp_path: Path) -> None:
    commit, tree = head(repo)
    # The container tagged every call as task 1; the host sent both tasks.
    host = FakeHost(
        [result("1", 0, 900, 0.09, 1), result("2", 0, 1, 0.0, 1)],
        [tagged("1", 600, 0.06)],
        wall_s=10.0,
    )

    evaluation = await evaluator(repo, tmp_path, host).evaluate(
        Request(commit, tree, SPLIT_TRAIN, ("1", "2"), (0,), "run-1")
    )

    stats = evaluation.stats
    assert stats.mean_tokens == 300.0  # The proxy's 600 over both sent tasks, tags aside.
    assert stats.mean_wall_time_s == 10.0  # 10 s / 2 tasks x 2 at once; claims aside.
