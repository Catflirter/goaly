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
from goaly.harness.core.registry import Dataset, Harness, HarnessOptions, Registry
from goaly.harness.core.runner import RunFiles, lock_run_dir
from tests.fakes import FakeSqlEnvironment, ToyInstance
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
def registry(factory: ToyFactory) -> Registry:
    registry = Registry()
    registry.add("toy", "v0", factory)
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
        "--phase",
        "test",
        "--out",
        str(out),
    ]

    outcome = goaly(capsys, registry, budgets, *argv)

    assert outcome.status == 0, outcome.stderr
    summary = outcome.summary
    assert summary == {
        "run_id": summary["run_id"],
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
    assert manifest.dataset_revision == "rev-1"
    assert list(manifest.dataset_notes) == ["toy data"]
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
        "--phase",
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
    argv = ["--bench", "toy", "--instances", str(tasks), "--phase", "test", "--out", str(out)]
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
        "--phase",
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
        "--phase",
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
    argv = ["--bench", "toy", "--split", "train", "--phase", "test", "--out", str(out)]
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
    argv = ["--bench", "toy", "--split", "train", "--phase", "test", "--out", str(out)]
    goaly(capsys, registry, budgets, *argv)

    other = goaly(capsys, registry, budgets, *argv, "--resume", "--task-timeout", "60")

    assert other.status == cli.USAGE
    assert "config_hash" in other.stderr


def test_the_cache_makes_a_second_run_free(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    common = ["--bench", "toy", "--split", "train", "--phase", "test"]
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
        "--phase",
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
            "--phase",
            "test",
            "--out",
            str(out),
        )

    assert outcome.status == cli.USAGE
    assert "another process is running in" in outcome.stderr
    assert factory.options == []  # Nothing of the harness touched the directory.


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
        "--phase",
        "test",
        "--share",
        "quiet",
        "--out",
        str(tmp_path / "run"),
    )

    assert outcome.status == 0, outcome.stderr
    assert factory.options[0].llm_shares == {}


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--phase", "rsi"], "no phase 'rsi'; the phases are: test"),
        (["--phase", "test", "--share", "background"], "gives no share to 'background'"),
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
        "--phase",
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
        "--phase",
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
        "--phase",
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
        [],
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
        "neither",
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
            "--phase",
            "test",
            "--out",
            str(tmp_path / "r"),
            *argv,
        )

    assert refused.value.code == cli.USAGE


def test_the_phase_is_required(
    capsys: pytest.CaptureFixture[str], registry: Registry, budgets: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as refused:
        goaly(capsys, registry, budgets, "--bench", "toy", "--split", "train", "--out", "r")

    assert refused.value.code == cli.USAGE
    assert "--phase" in capsys.readouterr().err


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
        "--phase",
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
        "--phase",
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
        "--phase",
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
        "--phase",
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
