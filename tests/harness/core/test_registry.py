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
from goaly.harness.core.registry import (
    Dataset,
    Harness,
    HarnessOptions,
    Registry,
    RegistryError,
    RunnableHarness,
)
from goaly.harness.core.runner import RunFiles, error_classifier
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
        classify=error_classifier({GatewayDownError: TASK_ERROR_TYPE_LLM}),
        models=(ModelSettings(model="qwen3.6-35b-a3b", temperature=0.6),),
        dataset=Dataset(revision="rev-a", notes=("dropped 1 task",), gold_sha256="g" * 64),
    )


@asynccontextmanager
async def toy(options: HarnessOptions) -> AsyncIterator[ToyHarness]:
    yield toy_harness()


@asynccontextmanager
async def other(options: HarnessOptions) -> AsyncIterator[ToyHarness]:
    yield toy_harness()


def options(tmp_path: Path) -> HarnessOptions:
    return HarnessOptions(files=RunFiles(tmp_path), instances_file=None, llm_shares={})


# The registry.


def test_finds_a_harness_by_benchmark_and_name() -> None:
    registry = Registry()
    registry.add("toy", "v0", toy)
    registry.add("toy", "v1", other)

    assert registry.factory("toy", "v0") is toy
    assert registry.factory("toy", "v1") is other
    assert registry.benchmarks() == ["toy"]
    assert registry.harnesses("toy") == ["v0", "v1"]


def test_a_benchmarks_only_harness_needs_no_name() -> None:
    registry = Registry()
    registry.add("toy", "v0", toy)

    assert registry.factory("toy", None) is toy


def test_a_benchmark_with_several_harnesses_needs_a_name() -> None:
    registry = Registry()
    registry.add("toy", "v1", other)
    registry.add("toy", "v0", toy)

    with pytest.raises(RegistryError, match="has several harnesses: v0, v1"):
        registry.factory("toy", None)


def test_an_unknown_name_lists_the_names_there_are() -> None:
    registry = Registry()
    registry.add("toy", "v0", toy)
    registry.add("effisql", "v0", other)

    with pytest.raises(RegistryError, match="no benchmark 'swe'; the benchmarks are: effisql, toy"):
        registry.factory("swe", None)
    with pytest.raises(RegistryError, match="'toy' has no harness 'v9'; its harnesses are: v0"):
        registry.factory("toy", "v9")
    with pytest.raises(RegistryError, match="no benchmark 'swe'"):
        registry.harnesses("swe")


def test_an_empty_registry_says_so() -> None:
    with pytest.raises(RegistryError, match="no benchmark 'toy'; none is registered"):
        Registry().factory("toy", None)


def test_a_name_is_registered_once() -> None:
    registry = Registry()
    registry.add("toy", "v0", toy)

    with pytest.raises(ValueError, match="'toy' already has a harness 'v0'"):
        registry.add("toy", "v0", other)
    assert registry.factory("toy", "v0") is toy


# Harnesses.


async def test_a_harness_builds_its_runner(tmp_path: Path) -> None:
    policy = ScriptedPolicy({"t1": raising(GatewayDownError("502 after 5 retries"))})
    harness = toy_harness(policy)
    cache = EvalCache(tmp_path / "cache")

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

    assert runnable.dataset == Dataset(
        revision="rev-a", notes=("dropped 1 task",), gold_sha256="g" * 64
    )
    assert list(runnable.models) == [ModelSettings(model="qwen3.6-35b-a3b", temperature=0.6)]
    assert runnable.pack_dir is None


def test_a_dataset_records_nothing_by_default() -> None:
    assert Dataset() == Dataset(revision="", notes=(), gold_sha256="")
