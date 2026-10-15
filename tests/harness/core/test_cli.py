"""Tests for goaly.harness.core.cli: `goaly run`, end to end with the toy benchmark (spec §6.4)."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Coroutine, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from goaly.contracts.gen.manifest_pb2 import ModelSettings
from goaly.contracts.gen.result_pb2 import TASK_ERROR_TYPE_TIMEOUT, TaskResult
from goaly.contracts.gen.task_pb2 import SPLIT_EXTERNAL, SPLIT_TRAIN, SPLIT_VAL
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core import cli
from goaly.harness.core.manifest import CORE_DIR, git_state, harness_hash, read_manifest
from goaly.harness.core.registry import (
    Dataset,
    Grading,
    GradingOptions,
    Harness,
    HarnessOptions,
    Registry,
)
from goaly.harness.core.runner import RunFiles, lock_run_dir
from tests.fakes import TOY_DATASET_SHA256, FakeSqlEnvironment, ToyInstance
from tests.harness.core.runner_doubles import (
    ScriptedBenchmark,
    ScriptedPolicy,
    StubContext,
    StubContexts,
    base_sql,
    sleeping,
    toy_instances,
)

BUDGETS = """
[budgets]
"qwen3.6-35b-a3b" = 16

[phases.test]
toy = { "qwen3.6-35b-a3b" = 4 }
quiet = {}
"""

type ToyHarness = Harness[ToyInstance, FakeSqlEnvironment, StubContext]


@dataclass
class ToyFactory:
    """The toy harness's factory, as the composition root would register it.

    It reads an ``--instances`` file of toy tasks, one JSON object a line, and records the
    options it got and whether its harness was closed.
    """

    policy: ScriptedPolicy = field(default_factory=ScriptedPolicy)
    pack_dir: Path | None = None
    name: str = "toy"
    cache_namespace: str = "toy-dev"
    options: list[HarnessOptions] = field(default_factory=list)
    benchmarks: list[ScriptedBenchmark] = field(default_factory=list)
    closed: bool = False

    @asynccontextmanager
    async def __call__(self, options: HarnessOptions) -> AsyncIterator[ToyHarness]:
        self.options.append(options)
        external = None
        if options.instances_file is not None:
            lines = options.instances_file.read_text(encoding="utf-8").splitlines()
            external = [
                ToyInstance(record["id"], SPLIT_EXTERNAL, record["base_sql"], gold_sql=None)
                for record in map(json.loads, lines)
            ]
        instances = [*toy_instances(3), ToyInstance("v0", SPLIT_VAL, base_sql(9), "SELECT 9")]
        benchmark = ScriptedBenchmark(instances, external=external)
        benchmark.name = self.name
        self.benchmarks.append(benchmark)
        try:
            yield Harness(
                benchmark=benchmark,
                policy=self.policy,
                contexts=StubContexts(),
                pack_dir=self.pack_dir,
                models=(ModelSettings(model="qwen3.6-35b-a3b", temperature=0.6, thinking=True),),
                dataset=Dataset(revision="rev-1", notes=("toy data",), gold_sha256="g" * 64),
                cache_namespace=self.cache_namespace,
            )
        finally:
            self.closed = True


@dataclass
class ToyGradingFactory:
    """The toy benchmark alone, for the grade phase: it opens no pack."""

    name: str = "toy"
    cache_namespace: str = "toy-dev"
    benchmarks: list[ScriptedBenchmark] = field(default_factory=list)

    @asynccontextmanager
    async def __call__(
        self, options: GradingOptions
    ) -> AsyncIterator[Grading[ToyInstance, FakeSqlEnvironment]]:
        instances = [*toy_instances(3), ToyInstance("v0", SPLIT_VAL, base_sql(9), "SELECT 9")]
        benchmark = ScriptedBenchmark(instances)
        benchmark.name = self.name
        self.benchmarks.append(benchmark)
        yield Grading(benchmark=benchmark, cache_namespace=self.cache_namespace)


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """main() configures logging for the process: put it back after each test."""
    root, quiet = logging.getLogger(), logging.getLogger("pydantic_settings")
    handlers, level, quiet_level = list(root.handlers), root.level, quiet.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)
    quiet.setLevel(quiet_level)


@pytest.fixture
def budgets(tmp_path: Path) -> Path:
    path = tmp_path / "llm_budgets.toml"
    path.write_text(BUDGETS, encoding="utf-8")
    return path


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "prompt.md").write_text("Make it fast.\n", encoding="utf-8")
    return pack


@pytest.fixture
def factory(pack: Path) -> ToyFactory:
    return ToyFactory(pack_dir=pack)


@pytest.fixture
def grading() -> ToyGradingFactory:
    return ToyGradingFactory()


@pytest.fixture
def registry(factory: ToyFactory, grading: ToyGradingFactory) -> Registry:
    registry = Registry()
    registry.add("toy", "v0", factory)
    registry.add_benchmark("toy", grading)
    return registry


@dataclass
class Outcome:
    status: int
    stdout: str
    stderr: str

    @property
    def summary(self) -> dict[str, Any]:
        lines = self.stdout.splitlines()
        assert len(lines) == 1, self.stdout
        summary: dict[str, Any] = json.loads(lines[0])
        return summary


def goaly(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, *argv: str
) -> Outcome:
    status = cli.main(["run", *argv], registry, budgets_file=budgets)
    captured = capsys.readouterr()
    return Outcome(status, captured.out, captured.err)


def results(run_dir: Path) -> list[TaskResult]:
    return list(read_jsonl(RunFiles(run_dir).results, TaskResult))


def official(run_dir: Path, seed: int = 0) -> list[dict[str, Any]]:
    lines = RunFiles(run_dir).official(seed).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# End to end.


def test_runs_the_toy_benchmark_end_to_end(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    factory: ToyFactory,
    pack: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "runs" / "train"
    argv = [
        "--bench",
        "toy",
        "--split",
        "train",
        "--seeds",
        "2",
        "--budget-phase",
        "test",
        "--out",
        str(out),
    ]

    outcome = goaly(capsys, registry, budgets, *argv)

    assert outcome.status == 0, outcome.stderr
    summary = outcome.summary
    assert summary == {
        "run_id": summary["run_id"],
        "phase": "all",
        "run_dir": str(out),
        "tasks": 6,
        "resumed": 0,
        "cached": 0,
        "ran": 6,
        "errors": {},
    }
    found = results(out)
    assert sorted((result.instance_id, result.seed) for result in found) == [
        (f"t{n}", seed) for n in range(3) for seed in (0, 1)
    ]
    assert all(result.grade.score == 0.5 for result in found)
    for seed in (0, 1):
        assert official(out, seed) == [
            {"instance_id": f"t{n}", "sql": base_sql(n)} for n in range(3)
        ]
    manifest = read_manifest(RunFiles(out).manifest)
    assert manifest.run_id == summary["run_id"]
    assert (manifest.benchmark, manifest.split) == ("toy", SPLIT_TRAIN)
    assert list(manifest.seeds) == [0, 1]
    assert list(manifest.instance_ids) == ["t0", "t1", "t2"]
    assert manifest.harness_hash == harness_hash(pack)
    assert len(manifest.config_hash) == 64
    assert list(manifest.command_line) == ["goaly", "run", *argv]
    # The benchmark's own fields (Benchmark.manifest_fields) win over the harness's Dataset.
    assert manifest.dataset_revision == "toy-revision-1"
    assert list(manifest.dataset_notes) == ["the toy tasks are held in memory"]
    assert manifest.dataset_sha256 == TOY_DATASET_SHA256
    assert dict(manifest.environment_settings) == {"grading_mode": "dev"}
    assert manifest.gold_sha256 == "g" * 64
    assert [model.model for model in manifest.models] == ["qwen3.6-35b-a3b"]
    assert manifest.machine.hostname
    state = git_state(CORE_DIR)
    assert manifest.git_commit == ("" if state is None else state.commit)
    assert manifest.HasField("started_at")
    assert manifest.HasField("finished_at")
    (options,) = factory.options
    assert options.files == RunFiles(out)
    assert options.instances_file is None
    assert options.llm_shares == {"qwen3.6-35b-a3b": 4}
    assert options.split == SPLIT_TRAIN
    assert options.budget_phase == "test"
    assert factory.closed


def test_runs_the_organizers_task_file_ungraded(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    factory: ToyFactory,
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "heldout.jsonl"
    lines = [
        '{"id": "x1", "base_sql": "SELECT 1 FROM t"}',
        '{"id": "x2", "base_sql": "SELECT 2 FROM t"}',
    ]
    tasks.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    out = tmp_path / "heldout"

    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--instances",
        str(tasks),
        "--budget-phase",
        "test",
        "--out",
        str(out),
    )

    assert outcome.status == 0, outcome.stderr
    assert factory.options[0].instances_file == tasks
    assert not any(result.HasField("grade") for result in results(out))
    assert factory.benchmarks[0].graded == []
    assert official(out) == [
        {"instance_id": "x1", "sql": "SELECT 1 FROM t"},
        {"instance_id": "x2", "sql": "SELECT 2 FROM t"},
    ]
    assert read_manifest(RunFiles(out).manifest).split == SPLIT_EXTERNAL


def test_a_changed_task_file_is_another_run(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    tasks = tmp_path / "heldout.jsonl"
    tasks.write_text('{"id": "x1", "base_sql": "SELECT 1 FROM t"}\n', encoding="utf-8")
    out = tmp_path / "heldout"
    argv = [
        "--bench",
        "toy",
        "--instances",
        str(tasks),
        "--budget-phase",
        "test",
        "--out",
        str(out),
    ]
    goaly(capsys, registry, budgets, *argv)

    # The same task IDs, with another query: resuming would mix two task files in one run.
    tasks.write_text('{"id": "x1", "base_sql": "SELECT 2 FROM t"}\n', encoding="utf-8")
    resumed = goaly(capsys, registry, budgets, *argv, "--resume")

    assert resumed.status == cli.USAGE
    assert "config_hash" in resumed.stderr


def test_runs_the_chosen_instances_seeds_and_options(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    factory: ToyFactory,
    tmp_path: Path,
) -> None:
    factory.policy = ScriptedPolicy({"t2": sleeping(10)})
    out = tmp_path / "run"

    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--harness",
        "v0",
        "--split",
        "train",
        "--only",
        "t2",
        "t0",
        "--seeds",
        "2",
        "--first-seed",
        "3",
        "--budget-phase",
        "test",
        "--out",
        str(out),
        "--max-tasks",
        "1",
        "--task-timeout",
        "0.5",
        "--run-id",
        "confirm-1",
    )

    assert outcome.status == 0, outcome.stderr
    assert outcome.summary["run_id"] == "confirm-1"
    keys = sorted((result.instance_id, result.seed) for result in results(out))
    assert keys == [("t0", 3), ("t0", 4), ("t2", 3), ("t2", 4)]
    timed_out = [result for result in results(out) if result.instance_id == "t2"]
    assert all(result.error_type == TASK_ERROR_TYPE_TIMEOUT for result in timed_out)
    assert outcome.summary["errors"] == {"TASK_ERROR_TYPE_TIMEOUT": 2}
    assert factory.benchmarks[0].most_running == 1
    manifest = read_manifest(RunFiles(out).manifest)
    assert (manifest.run_id, list(manifest.seeds)) == ("confirm-1", [3, 4])


def test_a_val_run_takes_the_val_split(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--split",
        "val",
        "--budget-phase",
        "test",
        "--out",
        str(tmp_path / "v"),
    )

    assert outcome.status == 0, outcome.stderr
    assert [result.instance_id for result in results(tmp_path / "v")] == ["v0"]


def test_resume_continues_the_run_and_a_new_run_refuses_its_directory(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    argv = ["--bench", "toy", "--split", "train", "--budget-phase", "test", "--out", str(out)]
    first = goaly(capsys, registry, budgets, *argv)
    files = RunFiles(out)
    lines = files.results.read_text(encoding="utf-8").splitlines(keepends=True)
    files.results.write_text("".join(lines[:-1]), encoding="utf-8")  # The last never got there.

    again = goaly(capsys, registry, budgets, *argv)
    resumed = goaly(capsys, registry, budgets, *argv, "--resume")

    assert again.status == cli.USAGE
    assert "already holds the run" in again.stderr
    assert resumed.status == 0, resumed.stderr
    assert resumed.summary["run_id"] == first.summary["run_id"]
    assert (resumed.summary["resumed"], resumed.summary["ran"]) == (2, 1)
    assert len(results(out)) == 3


def test_a_resumed_run_must_use_the_same_options(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    argv = ["--bench", "toy", "--split", "train", "--budget-phase", "test", "--out", str(out)]
    goaly(capsys, registry, budgets, *argv)

    other = goaly(capsys, registry, budgets, *argv, "--resume", "--task-timeout", "60")

    assert other.status == cli.USAGE
    assert "config_hash" in other.stderr


def test_the_cache_makes_a_second_run_free(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    common = ["--bench", "toy", "--split", "train", "--budget-phase", "test"]
    cache = ["--cache", str(tmp_path / "cache")]
    goaly(capsys, registry, budgets, *common, *cache, "--out", str(tmp_path / "a"))

    second = goaly(capsys, registry, budgets, *common, *cache, "--out", str(tmp_path / "b"))

    assert second.status == 0, second.stderr
    assert (second.summary["cached"], second.summary["ran"]) == (3, 0)
    hashes = {read_manifest(RunFiles(tmp_path / name).manifest).config_hash for name in ("a", "b")}
    assert len(hashes) == 1  # The output and cache directories are not settings of the run.


def test_a_harness_without_a_grading_namespace_cannot_use_the_cache(
    capsys: pytest.CaptureFixture[str], budgets: Path, tmp_path: Path
) -> None:
    registry = Registry()
    registry.add("toy", "v0", ToyFactory(cache_namespace=""))

    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--split",
        "train",
        "--budget-phase",
        "test",
        "--out",
        str(tmp_path / "r"),
        "--cache",
        str(tmp_path / "cache"),
    )

    assert outcome.status == cli.USAGE
    assert "names no grading namespace" in outcome.stderr


def test_a_run_directory_another_process_uses_is_refused_before_the_harness_opens(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    factory: ToyFactory,
    tmp_path: Path,
) -> None:
    out = tmp_path / "run"
    with lock_run_dir(RunFiles(out)):  # Another process's run, as far as the lock goes.
        outcome = goaly(
            capsys,
            registry,
            budgets,
            "--bench",
            "toy",
            "--split",
            "train",
            "--budget-phase",
            "test",
            "--out",
            str(out),
        )

    assert outcome.status == cli.USAGE
    assert "another process is running in" in outcome.stderr
    assert factory.options == []  # Nothing of the harness touched the directory.


# Runs in two phases (spec V24).


def solve_and_grade(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, out: Path, *more: str
) -> tuple[Outcome, Outcome]:
    common = ["--bench", "toy", "--out", str(out)]
    solved = goaly(
        capsys,
        registry,
        budgets,
        *common,
        "--phase",
        "solve",
        "--split",
        "train",
        "--budget-phase",
        "test",
        *more,
    )
    graded = goaly(capsys, registry, budgets, *common, "--phase", "grade", *more)
    return solved, graded


def test_a_run_solves_then_grades_and_the_grade_phase_opens_no_pack(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    factory: ToyFactory,
    grading: ToyGradingFactory,
    tmp_path: Path,
) -> None:
    out = tmp_path / "run"

    solved, graded = solve_and_grade(capsys, registry, budgets, out)

    assert solved.status == 0, solved.stderr
    assert graded.status == 0, graded.stderr
    assert (solved.summary["phase"], graded.summary["phase"]) == ("solve", "grade")
    assert (solved.summary["ran"], graded.summary["ran"]) == (3, 3)
    assert graded.summary["run_id"] == solved.summary["run_id"]
    assert len(factory.options) == 1  # The harness, and so the pack, opened for solving only.
    assert factory.benchmarks[0].graded == []
    assert len(grading.benchmarks[0].graded) == 3
    found = results(out)
    assert len(found) == 6  # Each task's ungraded line, then its graded one.
    assert all(result.HasField("grade") for result in found[3:])
    assert official(out) == [{"instance_id": f"t{n}", "sql": base_sql(n)} for n in range(3)]
    assert read_manifest(RunFiles(out).manifest).HasField("finished_at")


def test_the_grade_phase_fills_the_cache_for_the_next_solve_phase(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    cache = ["--cache", str(tmp_path / "cache")]
    solve_and_grade(capsys, registry, budgets, tmp_path / "a", *cache)

    again, _ = solve_and_grade(capsys, registry, budgets, tmp_path / "b", *cache)

    assert (again.summary["cached"], again.summary["ran"]) == (3, 0)


@pytest.mark.parametrize(
    ("argv", "options"),
    [
        (["--split", "train"], "--split"),
        (
            ["--seeds", "2", "--resume", "--budget-phase", "test"],
            "--seeds, --resume, --budget-phase",
        ),
        (["--harness", "v0", "--max-tasks", "2"], "--harness, --max-tasks"),
    ],
)
def test_the_grade_phase_takes_its_settings_from_the_run(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    tmp_path: Path,
    argv: list[str],
    options: str,
) -> None:
    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--phase",
        "grade",
        "--out",
        str(tmp_path / "r"),
        *argv,
    )

    assert outcome.status == cli.USAGE
    assert f"--phase grade takes these from the run it grades: {options}" in outcome.stderr


def test_the_grade_phase_needs_a_run_to_grade(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    out = tmp_path / "missing"

    outcome = goaly(
        capsys, registry, budgets, "--bench", "toy", "--phase", "grade", "--out", str(out)
    )

    assert outcome.status == cli.USAGE
    assert "holds no run to grade" in outcome.stderr
    assert not out.exists()


def test_a_benchmark_registered_without_a_grading_cannot_grade(
    capsys: pytest.CaptureFixture[str], factory: ToyFactory, budgets: Path, tmp_path: Path
) -> None:
    registry = Registry()
    registry.add("toy", "v0", factory)

    outcome = goaly(
        capsys, registry, budgets, "--bench", "toy", "--phase", "grade", "--out", str(tmp_path)
    )

    assert outcome.status == cli.USAGE
    assert "no benchmark 'toy' to grade with; none is registered" in outcome.stderr


def test_a_grading_without_a_namespace_cannot_use_the_cache(
    capsys: pytest.CaptureFixture[str], factory: ToyFactory, budgets: Path, tmp_path: Path
) -> None:
    registry = Registry()
    registry.add("toy", "v0", factory)
    registry.add_benchmark("toy", ToyGradingFactory(cache_namespace=""))
    out = tmp_path / "run"

    _, graded = solve_and_grade(capsys, registry, budgets, out, "--cache", str(tmp_path / "c"))

    assert graded.status == cli.USAGE
    assert "the benchmark 'toy' names no grading namespace for --cache" in graded.stderr


def test_a_grading_of_another_benchmark_is_refused(
    capsys: pytest.CaptureFixture[str], factory: ToyFactory, budgets: Path, tmp_path: Path
) -> None:
    registry = Registry()
    registry.add("toy", "v0", factory)
    registry.add_benchmark("toy", ToyGradingFactory(name="not-toy"))

    _, graded = solve_and_grade(capsys, registry, budgets, tmp_path / "run")

    assert graded.status == cli.FAILED
    assert "is of 'not-toy'" in graded.stderr


def test_an_interrupted_grade_phase_says_to_run_it_again(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(main: Coroutine[Any, Any, object]) -> object:
        main.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(asyncio, "run", interrupted)

    outcome = goaly(
        capsys, registry, budgets, "--bench", "toy", "--phase", "grade", "--out", str(tmp_path)
    )

    assert outcome.status == cli.INTERRUPTED
    assert "run the same command again" in outcome.stderr


# The concurrency share.


def test_the_share_is_the_one_the_phase_gives_the_process(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    factory: ToyFactory,
    tmp_path: Path,
) -> None:
    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--split",
        "train",
        "--budget-phase",
        "test",
        "--budget-share",
        "quiet",
        "--out",
        str(tmp_path / "run"),
    )

    assert outcome.status == 0, outcome.stderr
    assert factory.options[0].llm_shares == {}


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--budget-phase", "rsi"], "no phase 'rsi'; the phases are: test"),
        (
            ["--budget-phase", "test", "--budget-share", "background"],
            "gives no share to 'background'",
        ),
    ],
)
def test_a_process_without_a_share_does_not_run(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    factory: ToyFactory,
    tmp_path: Path,
    argv: list[str],
    message: str,
) -> None:
    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--split",
        "train",
        "--out",
        str(tmp_path / "run"),
        *argv,
    )

    assert outcome.status == cli.USAGE
    assert message in outcome.stderr
    assert factory.options == []
    assert not (tmp_path / "run").exists()


def test_a_budgets_file_that_is_invalid_stops_the_run(
    capsys: pytest.CaptureFixture[str], registry: Registry, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.toml"
    broken.write_text("[budgets]\nm = 0\n[phases]\n", encoding="utf-8")

    outcome = goaly(
        capsys,
        registry,
        broken,
        "--bench",
        "toy",
        "--split",
        "train",
        "--budget-phase",
        "test",
        "--out",
        str(tmp_path / "r"),
    )

    assert outcome.status == cli.USAGE
    assert "broken.toml" in outcome.stderr


# Usage errors.


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--bench", "swe"], "no benchmark 'swe'; the benchmarks are: toy"),
        (["--bench", "toy", "--harness", "v9"], "'toy' has no harness 'v9'"),
        (["--bench", "toy", "--only", "t9"], "SPLIT_TRAIN has no instance 't9'"),
    ],
    ids=["benchmark", "harness", "instance"],
)
def test_an_unknown_name_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    tmp_path: Path,
    argv: list[str],
    message: str,
) -> None:
    outcome = goaly(
        capsys,
        registry,
        budgets,
        *argv,
        "--split",
        "train",
        "--budget-phase",
        "test",
        "--out",
        str(tmp_path / "run"),
    )

    assert outcome.status == cli.USAGE
    assert f"goaly run: error: {message}" in outcome.stderr


def test_a_missing_task_file_is_a_usage_error(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--instances",
        str(tmp_path / "missing.jsonl"),
        "--budget-phase",
        "test",
        "--out",
        str(tmp_path / "run"),
    )

    assert outcome.status == cli.USAGE
    assert "no such task file" in outcome.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["--split", "train", "--instances", "tasks.jsonl"],
        ["--split", "test"],
        ["--split", "train", "--seeds", "0"],
        ["--split", "train", "--first-seed", "-1"],
        ["--split", "train", "--max-tasks", "0"],
        ["--split", "train", "--task-timeout", "0"],
        ["--split", "train", "--task-timeout", "nan"],
        ["--split", "train", "--seeds", "two"],
        ["--split", "train", "--task-timeout", "soon"],
    ],
    ids=[
        "both",
        "split",
        "seeds",
        "first seed",
        "max tasks",
        "timeout",
        "nan",
        "not a count",
        "not seconds",
    ],
)
def test_arguments_out_of_range_are_refused(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    tmp_path: Path,
    argv: list[str],
) -> None:
    with pytest.raises(SystemExit) as refused:
        goaly(
            capsys,
            registry,
            budgets,
            "--bench",
            "toy",
            "--budget-phase",
            "test",
            "--out",
            str(tmp_path / "r"),
            *argv,
        )

    assert refused.value.code == cli.USAGE


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--split", "train"], "--phase all needs --budget-phase"),
        (["--phase", "solve", "--split", "train"], "--phase solve needs --budget-phase"),
        (["--budget-phase", "test"], "--phase all needs --split or --instances"),
    ],
    ids=["budget phase", "budget phase to solve", "tasks"],
)
def test_a_phase_that_solves_needs_its_tasks_and_its_budget(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    tmp_path: Path,
    argv: list[str],
    message: str,
) -> None:
    outcome = goaly(
        capsys, registry, budgets, "--bench", "toy", "--out", str(tmp_path / "r"), *argv
    )

    assert outcome.status == cli.USAGE
    assert message in outcome.stderr


def test_a_harness_of_another_benchmark_is_refused(
    capsys: pytest.CaptureFixture[str], budgets: Path, tmp_path: Path
) -> None:
    registry = Registry()
    registry.add("toy", "v0", ToyFactory(name="not-toy"))

    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--split",
        "train",
        "--budget-phase",
        "test",
        "--out",
        str(tmp_path / "r"),
    )

    assert outcome.status == cli.FAILED
    assert "runs the benchmark 'not-toy'" in outcome.stderr


# Failures and interruptions.


def test_a_run_that_fails_exits_1_and_logs_why(
    capsys: pytest.CaptureFixture[str], budgets: Path, tmp_path: Path
) -> None:
    def broken(options: HarnessOptions) -> AbstractAsyncContextManager[ToyHarness]:
        raise RuntimeError("the gateway key is missing")

    registry = Registry()
    registry.add("toy", "v0", broken)

    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--split",
        "train",
        "--budget-phase",
        "test",
        "--out",
        str(tmp_path / "r"),
    )

    assert outcome.status == cli.FAILED
    assert "goaly run: the run failed: RuntimeError: the gateway key is missing" in outcome.stderr
    logged = [json.loads(line) for line in outcome.stderr.splitlines() if line.startswith("{")]
    assert any("Traceback" in entry.get("exception", "") for entry in logged)


def test_an_interrupted_run_says_how_to_continue(
    capsys: pytest.CaptureFixture[str],
    registry: Registry,
    budgets: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(main: Coroutine[Any, Any, object]) -> object:
        main.close()  # As if Ctrl-C came first: it never runs, so close it unawaited.
        raise KeyboardInterrupt

    monkeypatch.setattr(asyncio, "run", interrupted)

    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--split",
        "train",
        "--budget-phase",
        "test",
        "--out",
        str(tmp_path / "r"),
    )

    assert outcome.status == cli.INTERRUPTED
    assert "--resume" in outcome.stderr


# Logging.


def test_logs_are_json_lines_on_stderr_at_the_level_asked(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    outcome = goaly(
        capsys,
        registry,
        budgets,
        "--bench",
        "toy",
        "--split",
        "train",
        "--budget-phase",
        "test",
        "--out",
        str(tmp_path / "r"),
        "--log-level",
        "DEBUG",
    )

    assert outcome.status == 0
    logged = [json.loads(line) for line in outcome.stderr.splitlines()]
    assert any(entry["message"].startswith("finished run") for entry in logged)
    assert logging.getLogger().level == logging.DEBUG
    # pydantic-settings logs the raw settings, the API key among them, at DEBUG.
    assert not logging.getLogger("pydantic_settings").isEnabledFor(logging.DEBUG)
