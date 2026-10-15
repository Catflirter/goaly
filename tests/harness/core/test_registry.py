"""Tests for goaly.harness.core.registry: the benchmarks and harnesses `goaly run` can run."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from goaly.contracts.gen.manifest_pb2 import ModelSettings, RunManifest
from goaly.contracts.gen.result_pb2 import TASK_ERROR_TYPE_LLM, TaskResult
from goaly.contracts.gen.task_pb2 import SPLIT_TRAIN
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core.eval_cache import EvalCache
from goaly.harness.core.grading import SolveOptions
from goaly.harness.core.isolation import SolveRequest
from goaly.harness.core.registry import (
    Harness,
    HarnessOptions,
    HostBenchmark,
    HostOptions,
    RegisteredHarness,
    Registry,
    RegistryError,
    RunnableHarness,
    RunnableHost,
)
from goaly.harness.core.runner import RunFiles, UsageError, error_classifier
from tests.fakes import FakeSqlEnvironment, ToyInstance
from tests.harness.core.runner_doubles import (
    ScriptedBenchmark,
    ScriptedPolicy,
    StubContext,
    StubContexts,
    raising,
    toy_instances,
)

pytestmark = pytest.mark.anyio

type ToyHarness = Harness[ToyInstance, FakeSqlEnvironment, StubContext]


class GatewayDownError(Exception):
    """Stands in for the LLM client's error after its retries."""


def toy_harness(policy: ScriptedPolicy | None = None) -> ToyHarness:
    return Harness(
        benchmark=ScriptedBenchmark(toy_instances(2)),
        policy=ScriptedPolicy() if policy is None else policy,
        contexts=StubContexts(),
        classify=error_classifier([(GatewayDownError, TASK_ERROR_TYPE_LLM)]),
        gold_sha256="g" * 64,
    )


@asynccontextmanager
async def toy(options: HarnessOptions) -> AsyncIterator[ToyHarness]:
    yield toy_harness()


@asynccontextmanager
async def other(options: HarnessOptions) -> AsyncIterator[ToyHarness]:
    yield toy_harness()


def options(tmp_path: Path) -> HarnessOptions:
    return HarnessOptions(
        files=RunFiles(tmp_path),
        instances_file=None,
        llm_shares={},
        split=SPLIT_TRAIN,
        budget_phase="test",
    )


# The registry.


def test_finds_a_harness_by_benchmark_and_name() -> None:
    registry = Registry()
    registry.add("toy", "v0", toy)
    registry.add("toy", "v1", other)

    assert registry.harness("toy", "v0") == RegisteredHarness("v0", toy)
    assert registry.harness("toy", "v1") == RegisteredHarness("v1", other)
    assert registry.benchmarks() == ["toy"]
    assert registry.harnesses("toy") == ["v0", "v1"]


def test_a_benchmarks_only_harness_needs_no_name() -> None:
    registry = Registry()
    registry.add("toy", "v0", toy)

    assert registry.harness("toy", None) == RegisteredHarness("v0", toy)


def test_a_harness_is_registered_with_what_the_host_knows_of_it(tmp_path: Path) -> None:
    registry = Registry()
    models = [ModelSettings(model="qwen3.6-35b-a3b", temperature=0.6)]

    registry.add("toy", "v0", toy, pack_dir=tmp_path / "pack", models=models)

    registered = registry.harness("toy", "v0")
    assert registered.pack_dir == tmp_path / "pack"
    assert registered.models == (ModelSettings(model="qwen3.6-35b-a3b", temperature=0.6),)
    models.clear()  # The registry keeps its own copy.
    assert len(registered.models) == 1


def test_a_benchmark_with_several_harnesses_needs_a_name() -> None:
    registry = Registry()
    registry.add("toy", "v1", other)
    registry.add("toy", "v0", toy)

    with pytest.raises(RegistryError, match="has several harnesses: v0, v1"):
        registry.harness("toy", None)


def test_an_unknown_name_lists_the_names_there_are() -> None:
    registry = Registry()
    registry.add("toy", "v0", toy)
    registry.add("effisql", "v0", other)

    with pytest.raises(RegistryError, match="no benchmark 'swe'; the benchmarks are: effisql, toy"):
        registry.harness("swe", None)
    with pytest.raises(RegistryError, match="'toy' has no harness 'v9'; its harnesses are: v0"):
        registry.harness("toy", "v9")
    with pytest.raises(RegistryError, match="no benchmark 'swe'"):
        registry.harnesses("swe")


def test_an_empty_registry_says_so() -> None:
    with pytest.raises(RegistryError, match="no benchmark 'toy'; none is registered"):
        Registry().harness("toy", None)


def test_a_name_is_registered_once() -> None:
    registry = Registry()
    registry.add("toy", "v0", toy)

    with pytest.raises(ValueError, match="'toy' already has a harness 'v0'"):
        registry.add("toy", "v0", other)
    assert registry.harness("toy", "v0").factory is toy


# Host sides, for planning and grading on the host.


@asynccontextmanager
async def toy_host(
    options: HostOptions,
) -> AsyncIterator[HostBenchmark[ToyInstance, FakeSqlEnvironment]]:
    yield HostBenchmark(
        benchmark=ScriptedBenchmark(toy_instances(2)),
        cache_namespace="toy-dev",
        gold_sha256="g" * 64,
    )


def test_finds_a_benchmarks_host_side_apart_from_its_harnesses() -> None:
    registry = Registry()
    registry.add_benchmark("toy", toy_host)

    assert registry.host("toy") is toy_host
    assert registry.benchmarks() == []  # It has no harness.


def test_a_benchmark_has_one_host_side() -> None:
    registry = Registry()
    registry.add_benchmark("toy", toy_host)

    with pytest.raises(ValueError, match="'toy' already has a host side"):
        registry.add_benchmark("toy", toy_host)


def test_an_unknown_host_side_lists_those_there_are() -> None:
    registry = Registry()

    with pytest.raises(RegistryError, match="host side for the benchmark 'toy'; none has one"):
        registry.host("toy")
    registry.add_benchmark("effisql", toy_host)
    with pytest.raises(RegistryError, match="the benchmarks that have one are: effisql"):
        registry.host("toy")


def solve_options(tmp_path: Path) -> SolveOptions:
    return SolveOptions(
        harness="v0",
        checkout=tmp_path,
        pack_dir=None,
        instances_file=None,
        llm_shares={},
        max_tasks=2,
        task_timeout_s=None,
        log_level="INFO",
    )


async def test_a_host_side_solves_through_its_launcher_and_grades(tmp_path: Path) -> None:
    requests: list[SolveRequest] = []

    async def launcher(request: SolveRequest) -> None:  # The solve container, as far as it goes.
        requests.append(request)
        runner = toy_harness().runner(cache=None, max_tasks=2, task_timeout_s=None)
        await runner.run(
            request.solve_dir,
            RunManifest(run_id=request.run_id, harness_hash="h"),
            split=request.split,
            seeds=[0],
            grading=False,
            tasks=request.tasks,
        )

    host = HostBenchmark(
        benchmark=ScriptedBenchmark(toy_instances(2)),
        cache_namespace="toy-dev",
        gold_sha256="g" * 64,
        launcher=launcher,
    )
    runnable: RunnableHost = host

    summary = await runnable.host_runner(cache=None).run(
        tmp_path / "run",
        RunManifest(run_id="r", harness_hash="h"),
        split=SPLIT_TRAIN,
        seeds=[0],
        instance_ids=None,
        solve=solve_options(tmp_path),
    )

    assert (runnable.benchmark_name, runnable.cache_namespace) == ("toy", "toy-dev")
    assert runnable.gold_sha256 == "g" * 64
    assert [request.tasks for request in requests] == [(("t0", 0), ("t1", 0))]
    assert summary.ran == 2
    assert all(
        result.HasField("grade")
        for result in read_jsonl(RunFiles(tmp_path / "run").results, TaskResult)
    )


async def test_a_host_side_without_a_launcher_runs_only_in_process(tmp_path: Path) -> None:
    opening = HostOptions(
        files=RunFiles(tmp_path), instances_file=None, split=SPLIT_TRAIN, budget_phase=None
    )
    async with toy_host(opening) as host:
        runnable: RunnableHost = host
        with pytest.raises(UsageError, match="run it with --in-process"):
            await runnable.host_runner(cache=None).run(
                tmp_path / "run",
                RunManifest(run_id="r", harness_hash="h"),
                split=SPLIT_TRAIN,
                seeds=[0],
                instance_ids=None,
                solve=solve_options(tmp_path),
            )

    assert runnable.launcher is None
    assert not (tmp_path / "run").exists()


# Harnesses.


async def test_a_harness_builds_its_runner(tmp_path: Path) -> None:
    policy = ScriptedPolicy({"t1": raising(GatewayDownError("502 after 5 retries"))})
    harness = toy_harness(policy)
    cache = EvalCache(tmp_path / "cache", "toy-dev")

    runner = harness.runner(cache=cache, max_tasks=1, task_timeout_s=30.0)
    summary = await runner.run(
        tmp_path / "run", RunManifest(run_id="r", harness_hash="h"), split=SPLIT_TRAIN, seeds=[0]
    )

    assert summary.ran == 2
    assert summary.errors == {"TASK_ERROR_TYPE_LLM": 1}  # The harness's own classifier.
    results = list(read_jsonl(RunFiles(tmp_path / "run").results, TaskResult))
    assert len(results) == 2
    assert list((tmp_path / "cache").rglob("*.json"))  # The clean task went into the cache.


def test_a_harness_names_its_benchmark() -> None:
    assert toy_harness().benchmark_name == "toy"


async def test_a_factory_yields_a_runnable_harness(tmp_path: Path) -> None:
    async with toy(options(tmp_path)) as harness:
        runnable: RunnableHarness = harness

    assert runnable.gold_sha256 == "g" * 64


def test_a_harness_without_gold_files_outside_the_dataset_has_no_gold_hash() -> None:
    harness: ToyHarness = Harness(
        benchmark=ScriptedBenchmark(toy_instances(1)),
        policy=ScriptedPolicy(),
        contexts=StubContexts(),
    )

    assert harness.gold_sha256 == ""
