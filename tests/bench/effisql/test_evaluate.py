"""Tests for goaly.bench.effisql.evaluate, T21's evaluator seam, with a fake host command."""

import asyncio
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from goaly.bench.effisql.evaluate import EffiSqlEvaluator, TaskBudget, make_evaluator
from goaly.bench.effisql.grading_pool import Priority
from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord, Usage
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import SPLIT_TRAIN, SPLIT_VAL, Grade, Split
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.contracts.jsonl import to_json_line
from goaly.harness.core.manifest import content_hash, write_manifest

pytestmark = pytest.mark.anyio


@dataclass(frozen=True)
class Evaluation:  # T21's goaly.rsi.round.Evaluation, as T21 passes it in.
    results: Sequence[TaskResult]
    stats: object


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
        self, results: Sequence[TaskResult], calls: Sequence[LlmCallRecord], status: int = 0
    ):
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


async def test_it_runs_the_host_command_on_a_read_only_snapshot(repo: Path, tmp_path: Path) -> None:
    commit, tree = head(repo)
    host = FakeHost(
        [
            result("1", 0, 100, 0.01, 10),
            result("1", 0, 300, 0.03, 30),
            result("2", 0, 200, 0.02, 20),
        ],
        calls(300, 200),
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
    assert (stats.mean_tokens, stats.mean_wall_time_s, stats.count) == (250.0, 25.0, 2)
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


async def test_a_drift_from_the_proxy_total_is_flagged(
    repo: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    commit, tree = head(repo)
    host = FakeHost(
        [result("1", 0, 100, 0.0, 5)], [*calls(150, run_id="run-5"), *calls(9, run_id="x")]
    )

    with caplog.at_level(logging.WARNING):
        evaluation = await evaluator(repo, tmp_path, host).evaluate(
            Request(commit, tree, SPLIT_TRAIN, ("1",), (0,), "run-5")
        )

    assert "differ from the proxy's total" in caplog.text
    assert evaluation.stats.mean_cost_usd is None  # Nothing was priced.


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
