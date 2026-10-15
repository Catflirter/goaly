"""Tests for goaly.harness.core.runner: the parallel runner (spec §6.1, §6.2)."""

import asyncio
import fcntl
import json
import logging
import math
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from google.protobuf.message import Message
from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.llm_call_pb2 import Usage
from goaly.contracts.gen.manifest_pb2 import ModelSettings, RunManifest
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_ENVIRONMENT,
    TASK_ERROR_TYPE_LLM,
    TASK_ERROR_TYPE_NONE,
    TASK_ERROR_TYPE_POLICY_CRASH,
    TASK_ERROR_TYPE_TIMEOUT,
    TaskErrorType,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    GRADE_STATUS_TIMEOUT,
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
    SPLIT_VAL,
    Grade,
    Prediction,
    Split,
    TaskInput,
)
from goaly.contracts.gen.trajectory_pb2 import (
    STOP_REASON_CANCELLED,
    STOP_REASON_CRASHED,
    STOP_REASON_SUBMITTED,
    STOP_REASON_TIME_BUDGET_EXHAUSTED,
    Trajectory,
)
from goaly.contracts.interfaces import Environment, JsonValue
from goaly.contracts.jsonl import append_jsonl, read_jsonl
from goaly.harness.core import jsonl_writer
from goaly.harness.core.eval_cache import EvalCache
from goaly.harness.core.jsonl_writer import JsonlWriterError
from goaly.harness.core.manifest import read_manifest
from goaly.harness.core.runner import (
    RunDirectoryError,
    RunFiles,
    RunIdentity,
    Runner,
    RunSummary,
    TaskContext,
    classify_error,
    error_classifier,
    lock_run_dir,
)
from tests.fakes import TOY_DATASET_SHA256, FakeSqlEnvironment, ToyInstance
from tests.harness.core.runner_doubles import (
    Action,
    ScriptedBenchmark,
    ScriptedPolicy,
    StubContext,
    StubContexts,
    answering,
    base_sql,
    cancels_itself,
    raising,
    sleeping,
    toy_instances,
    waiting,
)

pytestmark = pytest.mark.anyio

# Guards each wait that would hang if the runner lost a task.
GUARD_S = 10
# The grading identity of the toy's runs, which share the evaluation cache.
CACHE_NAMESPACE = "toy-dev"

type Key = tuple[str, int]
type ToyRunner = Runner[ToyInstance, FakeSqlEnvironment, StubContext]


def template(run_id: str = "run-1", **fields: str) -> RunManifest:
    """The caller's part of a manifest, as the CLI builds it."""
    manifest = RunManifest(
        run_id=run_id,
        git_commit="0" * 40,
        harness_hash="harness-a",
        config_hash="config-a",
        dataset_revision="rev-a",
        command_line=["goaly", "run"],
    )
    for name, value in fields.items():
        setattr(manifest, name, value)
    return manifest


def with_models(*models: ModelSettings) -> RunManifest:
    manifest = template()
    manifest.models.extend(models)
    return manifest


def make_runner(
    benchmark: ScriptedBenchmark,
    policy: ScriptedPolicy | None = None,
    contexts: StubContexts | None = None,
    *,
    cache: EvalCache | None = None,
    max_tasks: int = 4,
    task_timeout_s: float | None = None,
) -> ToyRunner:
    return Runner(
        benchmark,
        ScriptedPolicy() if policy is None else policy,
        StubContexts() if contexts is None else contexts,
        cache=cache,
        max_tasks=max_tasks,
        task_timeout_s=task_timeout_s,
    )


async def run(
    runner: ToyRunner,
    run_dir: Path,
    *,
    manifest: RunManifest | None = None,
    split: Split.ValueType = SPLIT_TRAIN,
    seeds: Sequence[int] = (0,),
    instance_ids: Sequence[str] | None = None,
    resume: bool = False,
) -> RunSummary:
    async with asyncio.timeout(GUARD_S):
        return await runner.run(
            run_dir,
            template() if manifest is None else manifest,
            split=split,
            seeds=seeds,
            instance_ids=instance_ids,
            resume=resume,
        )


def results(run_dir: Path) -> dict[Key, TaskResult]:
    """The run's results by task, each recorded exactly once."""
    found: dict[Key, TaskResult] = {}
    for result in read_jsonl(RunFiles(run_dir).results, TaskResult):
        key = (result.instance_id, result.seed)
        assert key not in found, f"{key} recorded twice"
        found[key] = result
    return found


def results_last(run_dir: Path) -> dict[Key, TaskResult]:
    """The run's last result of each task, as readers take it: a later line replaces an earlier."""
    return {
        (result.instance_id, result.seed): result
        for result in read_jsonl(RunFiles(run_dir).results, TaskResult)
    }


def trajectories(run_dir: Path) -> dict[Key, Trajectory]:
    """The run's trajectories by task; a later line replaces an earlier one."""
    return {
        (trajectory.instance_id, trajectory.seed): trajectory
        for trajectory in read_jsonl(RunFiles(run_dir).trajectories, Trajectory)
    }


def official(run_dir: Path, seed: int = 0) -> list[dict[str, JsonValue]]:
    lines = RunFiles(run_dir).official(seed).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def base_answer(n: int) -> dict[str, JsonValue]:
    """The official record of toy task ``n`` with its base query unchanged: the empty answer."""
    return {"instance_id": f"t{n}", "sql": base_sql(n)}


async def wait_until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(GUARD_S):
        while not condition():
            await asyncio.sleep(0.005)


# A whole run.


async def test_runs_every_task_and_seed_once_and_writes_every_record(tmp_path: Path) -> None:
    benchmark = ScriptedBenchmark(toy_instances(3))
    policy = ScriptedPolicy()

    summary = await run(make_runner(benchmark, policy), tmp_path / "run", seeds=[0, 1])

    keys = {(f"t{n}", seed) for n in range(3) for seed in (0, 1)}
    assert sorted(policy.solved) == sorted(keys)
    found = results(tmp_path / "run")
    assert set(found) == keys
    for result in found.values():
        assert result.run_id == "run-1"
        assert result.error_type == TASK_ERROR_TYPE_NONE
        assert result.grade.status == GRADE_STATUS_GRADED
        assert result.grade.score == 0.5  # The toy's credit for the base query unchanged.
    assert set(trajectories(tmp_path / "run")) == keys
    for seed in (0, 1):
        assert official(tmp_path / "run", seed) == [base_answer(n) for n in range(3)]
    assert summary == RunSummary(run_id="run-1", tasks=6, resumed=0, cached=0, ran=6, errors={})


async def test_writes_the_manifest_at_the_start_and_finishes_it_at_the_end(
    tmp_path: Path,
) -> None:
    files = RunFiles(tmp_path / "run")
    gate = asyncio.Event()
    benchmark = ScriptedBenchmark(toy_instances(2))
    running = asyncio.create_task(
        run(make_runner(benchmark, ScriptedPolicy({"t1": waiting(gate)})), files.root, seeds=[0, 2])
    )
    await wait_until(lambda: files.manifest.exists())

    started = read_manifest(files.manifest)
    gate.set()
    await running

    finished = read_manifest(files.manifest)
    assert not started.HasField("finished_at")
    assert finished.started_at == started.started_at
    assert finished.finished_at.ToDatetime() >= finished.started_at.ToDatetime()
    expected = template()
    expected.ClearField("dataset_revision")  # The adapter's, which the caller's never overrides.
    expected.MergeFrom(
        RunManifest(benchmark="toy", split=SPLIT_TRAIN, seeds=[0, 2], instance_ids=["t0", "t1"])
    )
    expected.MergeFrom(benchmark.manifest_fields())
    expected.started_at.CopyFrom(started.started_at)
    assert started == expected


class OddManifest(ScriptedBenchmark):
    """Its manifest fields set more than the four that are an adapter's to set."""

    def manifest_fields(self) -> RunManifest:
        fields = super().manifest_fields()
        fields.run_id = "the adapter's"
        fields.harness_hash = "the adapter's"
        return fields


async def test_the_adapter_alone_sets_its_four_manifest_fields(tmp_path: Path) -> None:
    caller = template()
    caller.dataset_notes.append("the caller's note")
    caller.dataset_sha256 = "c" * 64
    caller.environment_settings["grading_mode"] = "official"

    await run(make_runner(OddManifest(toy_instances(1))), tmp_path, manifest=caller)

    manifest = read_manifest(RunFiles(tmp_path).manifest)
    assert manifest.dataset_revision == "toy-revision-1"
    assert list(manifest.dataset_notes) == ["the toy tasks are held in memory"]
    assert manifest.dataset_sha256 == TOY_DATASET_SHA256
    assert dict(manifest.environment_settings) == {"grading_mode": "dev"}
    assert (manifest.run_id, manifest.harness_hash) == ("run-1", "harness-a")  # The caller's.


async def test_a_result_and_its_trajectory_describe_the_task(tmp_path: Path) -> None:
    benchmark = ScriptedBenchmark(toy_instances(1), provision_delay_s=0.5)

    await run(make_runner(benchmark, ScriptedPolicy(delay_s=0.02)), tmp_path / "run", seeds=[3])

    result = results(tmp_path / "run")[("t0", 3)]
    two_steps = Usage(input_tokens=200, output_tokens=40, cost_usd=0.002)
    assert result.usage == two_steps
    assert result.step_count == 2
    assert result.stop_reason == STOP_REASON_SUBMITTED
    assert result.error_message == ""
    # The policy's solving time leaves out the provisioning, which has its own field.
    assert result.wall_time_s >= 0.02
    assert result.provision_s >= 0.5
    assert result.wall_time_s < result.provision_s  # Only a stall of 0.48 s could break it.
    assert result.started_at.ToNanoseconds() < result.finished_at.ToNanoseconds()
    trajectory = trajectories(tmp_path / "run")[("t0", 3)]
    assert (trajectory.run_id, trajectory.instance_id, trajectory.seed) == ("run-1", "t0", 3)
    assert trajectory.task == benchmark.agent_input(benchmark.instances[0])
    assert trajectory.prediction.instance_id == "t0"
    assert trajectory.prediction.effisql.sql == "SELECT 0 FROM slow"
    assert trajectory.grade == result.grade
    assert trajectory.usage == two_steps
    assert len(trajectory.steps) == 2
    assert trajectory.stop_reason == STOP_REASON_SUBMITTED
    assert trajectory.HasField("started_at")
    assert trajectory.HasField("finished_at")


async def test_the_trajectory_keeps_the_times_its_context_recorded(tmp_path: Path) -> None:
    started, finished = Timestamp(seconds=1_000), Timestamp(seconds=2_000)

    def own_times(ctx: StubContext) -> None:
        ctx.recorded.started_at.CopyFrom(started)
        ctx.recorded.finished_at.CopyFrom(finished)

    contexts = StubContexts(setup=own_times)

    await run(make_runner(ScriptedBenchmark(toy_instances(1)), contexts=contexts), tmp_path)

    trajectory = trajectories(tmp_path)[("t0", 0)]
    assert (trajectory.started_at, trajectory.finished_at) == (started, finished)


async def test_the_prediction_names_its_instance_before_it_is_graded(tmp_path: Path) -> None:
    benchmark = ScriptedBenchmark(toy_instances(2))

    await run(make_runner(benchmark), tmp_path / "run")

    assert sorted(prediction.instance_id for prediction in benchmark.graded) == ["t0", "t1"]


async def test_each_context_is_made_for_its_task_and_run(tmp_path: Path) -> None:
    contexts = StubContexts()

    await run(make_runner(ScriptedBenchmark(toy_instances(2)), contexts=contexts), tmp_path / "r")

    identity = RunIdentity(run_id="run-1", benchmark="toy", split=SPLIT_TRAIN)
    assert sorted((ctx.instance_id, ctx.seed) for ctx in contexts.made) == [("t0", 0), ("t1", 0)]
    assert all(ctx.run == identity and ctx.exited for ctx in contexts.made)


async def test_runs_only_the_instances_asked_for_in_the_benchmark_order(tmp_path: Path) -> None:
    policy = ScriptedPolicy()

    await run(
        make_runner(ScriptedBenchmark(toy_instances(4)), policy),
        tmp_path / "run",
        instance_ids=["t3", "t1"],
    )

    assert sorted(policy.solved) == [("t1", 0), ("t3", 0)]
    assert official(tmp_path / "run") == [base_answer(1), base_answer(3)]
    assert read_manifest(RunFiles(tmp_path / "run").manifest).instance_ids == ["t1", "t3"]


# Crash isolation (R4): a failure in one task never affects the others.


async def test_a_task_that_raises_is_recorded_and_the_others_finish(tmp_path: Path) -> None:
    benchmark = ScriptedBenchmark(toy_instances(3))
    policy = ScriptedPolicy({"t1": raising(ValueError("boom"))})

    summary = await run(make_runner(benchmark, policy), tmp_path / "run")

    found = results(tmp_path / "run")
    assert found[("t0", 0)].error_type == found[("t2", 0)].error_type == TASK_ERROR_TYPE_NONE
    crashed = found[("t1", 0)]
    assert crashed.error_type == TASK_ERROR_TYPE_POLICY_CRASH
    assert crashed.error_message == "ValueError: boom"
    assert crashed.stop_reason == STOP_REASON_CRASHED
    assert crashed.step_count == 2  # What the agent did before it crashed still counts.
    # A crashed task is graded as the benchmark's empty answer, and submits it.
    assert crashed.grade.score == 0.5
    graded = {prediction.instance_id: prediction for prediction in benchmark.graded}
    assert graded["t1"].WhichOneof("benchmark") is None
    assert official(tmp_path / "run") == [base_answer(0), base_answer(1), base_answer(2)]
    trajectory = trajectories(tmp_path / "run")[("t1", 0)]
    assert not trajectory.HasField("prediction")
    assert trajectory.error_message == "ValueError: boom"
    assert trajectory.stop_reason == STOP_REASON_CRASHED
    assert summary.errors == {"TASK_ERROR_TYPE_POLICY_CRASH": 1}


@pytest.mark.parametrize(
    "error",
    [
        EnvironmentFailure("the database died"),
        ExceptionGroup("tools", [EnvironmentFailure("the database died")]),
        ExceptionGroup(
            "outer", [ValueError("a"), ExceptionGroup("inner", [EnvironmentFailure("b")])]
        ),
    ],
    ids=["plain", "in a group", "deep in a group"],
)
async def test_an_environment_failure_in_the_policy_is_an_environment_error(
    tmp_path: Path, error: Exception
) -> None:
    policy = ScriptedPolicy({"t0": raising(error)})

    await run(make_runner(ScriptedBenchmark(toy_instances(2)), policy), tmp_path / "run")

    found = results(tmp_path / "run")
    assert found[("t0", 0)].error_type == TASK_ERROR_TYPE_ENVIRONMENT
    assert found[("t0", 0)].stop_reason == STOP_REASON_CRASHED
    assert "EnvironmentFailure" in found[("t0", 0)].error_message
    assert found[("t1", 0)].error_type == TASK_ERROR_TYPE_NONE


async def test_a_task_whose_environment_cannot_be_provisioned_is_an_environment_error(
    tmp_path: Path,
) -> None:
    benchmark = ScriptedBenchmark(
        toy_instances(2), provision_errors={"t0": EnvironmentFailure("no connection")}
    )
    contexts = StubContexts()

    await run(make_runner(benchmark, contexts=contexts), tmp_path / "run")

    failed = results(tmp_path / "run")[("t0", 0)]
    assert failed.error_type == TASK_ERROR_TYPE_ENVIRONMENT
    assert failed.error_message == "EnvironmentFailure: no connection"
    assert failed.step_count == 0
    assert failed.grade.score == 0.5  # The empty answer, graded.
    assert [ctx.instance_id for ctx in contexts.made] == ["t1"]


async def test_an_instance_whose_input_cannot_be_built_is_recorded_as_a_crash(
    tmp_path: Path,
) -> None:
    benchmark = ScriptedBenchmark(toy_instances(2), input_errors={"t0": KeyError("base_plan")})
    contexts = StubContexts()

    await run(make_runner(benchmark, contexts=contexts), tmp_path / "run")

    failed = results(tmp_path / "run")[("t0", 0)]
    assert failed.error_type == TASK_ERROR_TYPE_POLICY_CRASH
    assert failed.error_message == "KeyError: 'base_plan'"
    assert not failed.HasField("started_at")  # It never got a slot.
    assert failed.HasField("finished_at")
    assert failed.grade.score == 0.5
    assert not trajectories(tmp_path / "run")[("t0", 0)].HasField("task")
    assert official(tmp_path / "run") == [base_answer(0), base_answer(1)]
    assert [ctx.instance_id for ctx in contexts.made] == ["t1"]


async def test_the_provisioning_time_is_recorded_when_provisioning_fails(tmp_path: Path) -> None:
    benchmark = ScriptedBenchmark(
        toy_instances(1), provision_delay_s=0.2, provision_errors={"t0": EnvironmentFailure("x")}
    )

    await run(make_runner(benchmark), tmp_path)

    assert results(tmp_path)[("t0", 0)].provision_s >= 0.2


async def test_a_crash_inside_grade_is_a_grader_error_not_a_task_error(tmp_path: Path) -> None:
    benchmark = ScriptedBenchmark(
        toy_instances(2), grade_errors={"t0": RuntimeError("grading instance down")}
    )

    summary = await run(make_runner(benchmark), tmp_path / "run")

    result = results(tmp_path / "run")[("t0", 0)]
    assert result.error_type == TASK_ERROR_TYPE_NONE
    assert result.grade.instance_id == "t0"
    assert result.grade.status == GRADE_STATUS_GRADER_ERROR
    assert not result.grade.correct
    assert "RuntimeError: grading instance down" in result.grade.feedback
    assert results(tmp_path / "run")[("t1", 0)].grade.status == GRADE_STATUS_GRADED
    assert summary.errors == {}


async def test_a_prediction_without_an_official_form_counts_as_a_crash(tmp_path: Path) -> None:
    benchmark = ScriptedBenchmark(
        toy_instances(2), official_errors={"t0": ValueError("cannot write a diff")}
    )
    policy = ScriptedPolicy({"t0": answering("SELECT 0")})

    await run(make_runner(benchmark, policy), tmp_path / "run")

    result = results(tmp_path / "run")[("t0", 0)]
    assert result.error_type == TASK_ERROR_TYPE_POLICY_CRASH
    assert "cannot write a diff" in result.error_message
    assert '"sql":"SELECT 0"' in result.error_message  # The rejected answer, kept for debugging.
    assert result.grade.score == 0.5  # Graded as the empty answer, which the file submits.
    assert official(tmp_path / "run") == [base_answer(0), base_answer(1)]
    assert not trajectories(tmp_path / "run")[("t0", 0)].HasField("prediction")


class NoEmptyAnswer(ScriptedBenchmark):
    """An adapter that breaks the contract: it cannot write the empty answer."""

    def to_official(self, inst: ToyInstance, prediction: Prediction) -> dict[str, JsonValue]:
        if not prediction.HasField("effisql"):
            raise ValueError("no base_sql in the record")
        return super().to_official(inst, prediction)


async def test_an_adapter_that_cannot_write_the_empty_answer_stops_the_run_early(
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy({"t0": raising(ValueError("boom"))})

    # Found as the first crashed task is recorded, not after every task has run.
    with pytest.raises(ExceptionGroup) as stopped:
        await run(make_runner(NoEmptyAnswer(toy_instances(1)), policy), tmp_path)

    assert stopped.group_contains(RuntimeError, match="cannot write the empty answer of 't0'")
    assert results(tmp_path) == {}  # The task is not recorded: a resume runs it again.


async def test_a_context_that_cannot_give_its_trajectory_fails_its_task(tmp_path: Path) -> None:
    def break_t0(ctx: StubContext) -> None:
        if ctx.instance_id == "t0":
            ctx.trajectory_error = RuntimeError("no trajectory")

    contexts = StubContexts(setup=break_t0)

    await run(make_runner(ScriptedBenchmark(toy_instances(2)), contexts=contexts), tmp_path / "r")

    failed = results(tmp_path / "r")[("t0", 0)]
    assert failed.error_type == TASK_ERROR_TYPE_POLICY_CRASH
    assert failed.error_message == "RuntimeError: no trajectory"
    assert failed.grade.score == 0.5
    assert results(tmp_path / "r")[("t1", 0)].error_type == TASK_ERROR_TYPE_NONE


async def test_when_both_fail_the_policys_error_is_the_one_recorded(tmp_path: Path) -> None:
    def break_trajectory(ctx: StubContext) -> None:
        ctx.trajectory_error = RuntimeError("no trajectory")

    policy = ScriptedPolicy({"t0": raising(EnvironmentFailure("gone"))})
    contexts = StubContexts(setup=break_trajectory)

    await run(make_runner(ScriptedBenchmark(toy_instances(1)), policy, contexts), tmp_path / "r")

    failed = results(tmp_path / "r")[("t0", 0)]
    assert failed.error_type == TASK_ERROR_TYPE_ENVIRONMENT
    assert failed.error_message == "EnvironmentFailure: gone"


async def test_the_error_classifier_is_injected(tmp_path: Path) -> None:
    policy = ScriptedPolicy({"t0": raising(GatewayDownError("502 after 5 retries"))})
    runner = Runner(
        ScriptedBenchmark(toy_instances(1)),
        policy,
        StubContexts(),
        classify=error_classifier([(GatewayDownError, TASK_ERROR_TYPE_LLM)]),
    )

    summary = await run(runner, tmp_path / "run")

    assert results(tmp_path / "run")[("t0", 0)].error_type == TASK_ERROR_TYPE_LLM
    assert summary.errors == {"TASK_ERROR_TYPE_LLM": 1}


async def test_a_classifier_that_raises_falls_back_to_a_policy_crash(tmp_path: Path) -> None:
    def broken(error: Exception) -> TaskErrorType.ValueType:
        raise KeyError("no rule")

    policy = ScriptedPolicy({"t0": raising(ValueError("boom"))})
    runner = Runner(ScriptedBenchmark(toy_instances(1)), policy, StubContexts(), classify=broken)

    await run(runner, tmp_path / "run")

    assert results(tmp_path / "run")[("t0", 0)].error_type == TASK_ERROR_TYPE_POLICY_CRASH


class UnprintableError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("cannot print")


async def test_an_error_that_cannot_be_printed_is_named_by_its_type(tmp_path: Path) -> None:
    policy = ScriptedPolicy({"t0": raising(UnprintableError())})

    await run(make_runner(ScriptedBenchmark(toy_instances(1)), policy), tmp_path)

    assert results(tmp_path)[("t0", 0)].error_message == "UnprintableError"


async def test_an_error_message_is_cut_short(tmp_path: Path) -> None:
    policy = ScriptedPolicy({"t0": raising(ValueError("x" * 10_000))})

    await run(make_runner(ScriptedBenchmark(toy_instances(1)), policy), tmp_path / "run")

    message = results(tmp_path / "run")[("t0", 0)].error_message
    assert message.startswith("ValueError: xxx")
    assert len(message) < 2_100
    assert message.endswith("characters cut]")


# No task vanishes (fix round G1): code that raises a CancelledError while nothing cancelled the
# run, or a BaseExceptionGroup, fails only its own task.


def stray() -> asyncio.CancelledError:
    return asyncio.CancelledError("cancelled from within")


async def test_every_task_is_recorded_whatever_its_code_raises(tmp_path: Path) -> None:
    def break_context(ctx: StubContext) -> None:
        if ctx.instance_id == "t3":
            raise stray()
        if ctx.instance_id == "t9":
            ctx.trajectory_error = stray()

    benchmark = ScriptedBenchmark(
        toy_instances(12),
        input_errors={"t0": stray()},
        provision_errors={"t1": stray()},
        grade_errors={"t2": stray(), "t8": BaseExceptionGroup("grading", [stray()])},
        official_errors={"t4": stray()},
    )
    policy = ScriptedPolicy(
        {
            "t5": raising(stray()),
            "t6": cancels_itself,
            "t7": raising(BaseExceptionGroup("samples", [stray(), ValueError("bad sample")])),
            "t10": raising(BaseExceptionGroup("tools", [stray(), EnvironmentFailure("gone")])),
        }
    )

    summary = await run(
        make_runner(benchmark, policy, StubContexts(setup=break_context)), tmp_path, seeds=(0, 1)
    )

    found = results(tmp_path)
    # One result per (task, seed), and the official files: the run finished.
    assert set(found) == {(f"t{n}", seed) for n in range(12) for seed in (0, 1)}
    assert len(official(tmp_path, 1)) == 12
    crashed = {"t0", "t1", "t3", "t4", "t5", "t6", "t7", "t9"}
    for (instance_id, _), result in found.items():
        if instance_id in crashed:
            assert result.error_type == TASK_ERROR_TYPE_POLICY_CRASH, instance_id
            assert result.stop_reason == STOP_REASON_CRASHED, instance_id
            assert "CancelledError" in result.error_message, instance_id
            assert result.grade.score == 0.5, instance_id  # Its empty answer, graded.
        elif instance_id in {"t2", "t8"}:
            assert result.error_type == TASK_ERROR_TYPE_NONE
            assert result.grade.status == GRADE_STATUS_GRADER_ERROR
            assert "CancelledError" in result.grade.feedback
        elif instance_id == "t10":
            assert result.error_type == TASK_ERROR_TYPE_ENVIRONMENT
        else:
            assert result.error_type == TASK_ERROR_TYPE_NONE
            assert result.grade.status == GRADE_STATUS_GRADED
    assert summary.ran == 24


async def test_a_system_exit_from_the_context_still_ends_the_run(tmp_path: Path) -> None:
    def exit_from_trajectory(ctx: StubContext) -> None:
        ctx.trajectory_error = BaseExceptionGroup("exit", [SystemExit(4)])

    runner = make_runner(
        ScriptedBenchmark(toy_instances(1)), contexts=StubContexts(setup=exit_from_trajectory)
    )

    with pytest.raises(BaseExceptionGroup) as stopped:
        await run(runner, tmp_path)

    assert stopped.group_contains(SystemExit)


async def test_a_system_exit_inside_a_group_still_ends_the_run(tmp_path: Path) -> None:
    policy = ScriptedPolicy({"t0": raising(BaseExceptionGroup("exit", [SystemExit(3)]))})

    with pytest.raises(BaseExceptionGroup) as stopped:
        await run(make_runner(ScriptedBenchmark(toy_instances(2)), policy), tmp_path)

    assert stopped.group_contains(SystemExit)
    assert not RunFiles(tmp_path).official(0).exists()


# The task time limit.


async def test_the_time_limit_cancels_a_task_and_closes_its_environment(tmp_path: Path) -> None:
    benchmark = ScriptedBenchmark(toy_instances(2))
    policy = ScriptedPolicy({"t0": sleeping(GUARD_S)})
    contexts = StubContexts()

    await run(make_runner(benchmark, policy, contexts, task_timeout_s=0.5), tmp_path / "run")

    timed_out = results(tmp_path / "run")[("t0", 0)]
    assert timed_out.error_type == TASK_ERROR_TYPE_TIMEOUT
    assert timed_out.stop_reason == STOP_REASON_CANCELLED
    assert "0.5 s" in timed_out.error_message
    assert timed_out.step_count == 2  # Its trajectory up to the cancellation is kept.
    assert timed_out.grade.score == 0.5
    assert all(env.closed for env in benchmark.environments)
    assert all(ctx.exited for ctx in contexts.made)
    assert results(tmp_path / "run")[("t1", 0)].error_type == TASK_ERROR_TYPE_NONE


async def test_a_timeout_error_of_the_policys_own_is_not_the_time_limit(tmp_path: Path) -> None:
    policy = ScriptedPolicy({"t0": raising(TimeoutError("statement timeout"))})

    await run(
        make_runner(ScriptedBenchmark(toy_instances(1)), policy, task_timeout_s=GUARD_S),
        tmp_path / "run",
    )

    result = results(tmp_path / "run")[("t0", 0)]
    assert result.error_type == TASK_ERROR_TYPE_POLICY_CRASH
    assert result.error_message == "TimeoutError: statement timeout"


# Pack code that bends cancellation: an outer cancellation still propagates.


def in_sleep(started: asyncio.Event, on_cancel: Callable[[], Prediction | None]) -> Action:
    """An action that sleeps until cancelled, then answers or raises as ``on_cancel`` does."""

    async def act(task: TaskInput, env: Environment, ctx: StubContext) -> Prediction | None:
        started.set()
        try:
            await asyncio.sleep(GUARD_S)
        except asyncio.CancelledError:
            return on_cancel()
        return None

    return act


def cleanup_fails() -> Prediction | None:
    raise RuntimeError("cleanup failed")


def keeps_going() -> Prediction | None:
    return None  # The default answer, as if nothing happened.


@pytest.mark.parametrize("on_cancel", [cleanup_fails, keeps_going], ids=["converts", "swallows"])
async def test_a_policy_cannot_turn_a_cancellation_of_the_run_into_a_record(
    tmp_path: Path, on_cancel: Callable[[], Prediction | None]
) -> None:
    started = asyncio.Event()
    policy = ScriptedPolicy({"t0": in_sleep(started, on_cancel)})
    running = asyncio.create_task(
        run(make_runner(ScriptedBenchmark(toy_instances(1)), policy), tmp_path)
    )
    async with asyncio.timeout(GUARD_S):
        await started.wait()

    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running
    assert results(tmp_path) == {}


@pytest.mark.parametrize("on_cancel", [cleanup_fails, keeps_going], ids=["converts", "swallows"])
async def test_a_policy_that_resists_its_time_limit_is_timed_out(
    tmp_path: Path, on_cancel: Callable[[], Prediction | None]
) -> None:
    policy = ScriptedPolicy({"t0": in_sleep(asyncio.Event(), on_cancel)})

    await run(
        make_runner(ScriptedBenchmark(toy_instances(1)), policy, task_timeout_s=0.1), tmp_path
    )

    result = results(tmp_path)[("t0", 0)]
    assert result.error_type == TASK_ERROR_TYPE_TIMEOUT
    assert result.stop_reason == STOP_REASON_CANCELLED
    assert result.grade.score == 0.5  # Its late answer does not count: the empty one does.


async def test_a_task_group_failing_inside_the_policy_is_a_crash_not_a_cancellation(
    tmp_path: Path,
) -> None:
    # Python 3.12's TaskGroup leaves a cancel request on its task when a child fails while the
    # group exits; it must not read as a cancellation of the run.
    async def failing_child() -> None:
        await asyncio.sleep(0)
        raise ValueError("a sample failed")

    async def best_of_n(task: TaskInput, env: Environment, ctx: StubContext) -> Prediction | None:
        async with asyncio.TaskGroup() as group:
            group.create_task(failing_child())
        return None

    policy = ScriptedPolicy({"t0": best_of_n})

    await run(make_runner(ScriptedBenchmark(toy_instances(2)), policy), tmp_path)

    found = results(tmp_path)
    assert found[("t0", 0)].error_type == TASK_ERROR_TYPE_POLICY_CRASH
    assert "ValueError: a sample failed" in found[("t0", 0)].error_message
    assert found[("t1", 0)].error_type == TASK_ERROR_TYPE_NONE


async def leave_a_phantom_cancel_request() -> None:
    """Handle a TaskGroup whose child fails while the group exits: Python 3.12 then leaves a
    cancel request on the current task, though nothing cancelled it."""

    async def failing_child() -> None:
        await asyncio.sleep(0)
        raise ValueError("a child failed")

    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(failing_child())
    except* ValueError:
        pass  # Handled: the code goes on as if nothing happened.


class PhantomBenchmark(ScriptedBenchmark):
    """Its provisioning or its grader handles a failing TaskGroup first."""

    def __init__(self, instances: Sequence[ToyInstance], *, in_provision: bool) -> None:
        super().__init__(instances)
        self.in_provision = in_provision

    @asynccontextmanager
    async def provision(self, inst: ToyInstance) -> AsyncIterator[FakeSqlEnvironment]:
        if self.in_provision:
            await leave_a_phantom_cancel_request()
        async with super().provision(inst) as env:
            yield env

    async def grade(self, inst: ToyInstance, prediction: Prediction) -> Grade:
        if not self.in_provision:
            await leave_a_phantom_cancel_request()
        return await super().grade(inst, prediction)


@pytest.mark.parametrize("in_provision", [True, False], ids=["provision", "grade"])
async def test_a_failing_task_group_in_the_adapter_does_not_lose_the_task(
    tmp_path: Path, in_provision: bool
) -> None:
    benchmark = PhantomBenchmark(toy_instances(2), in_provision=in_provision)

    summary = await run(make_runner(benchmark), tmp_path)

    found = results(tmp_path)
    assert set(found) == {("t0", 0), ("t1", 0)}
    assert all(result.grade.status == GRADE_STATUS_GRADED for result in found.values())
    assert summary.errors == {}


async def test_a_failing_task_group_in_the_context_factory_does_not_lose_the_task(
    tmp_path: Path,
) -> None:
    @asynccontextmanager
    async def contexts(
        inst: ToyInstance, seed: int, run: RunIdentity
    ) -> AsyncIterator[StubContext]:
        await leave_a_phantom_cancel_request()
        yield StubContext(inst, seed, run)

    runner = Runner(ScriptedBenchmark(toy_instances(2)), ScriptedPolicy(), contexts)

    await run(runner, tmp_path)

    assert set(results(tmp_path)) == {("t0", 0), ("t1", 0)}


async def test_a_cancel_request_the_policy_leaves_never_reaches_the_context(
    tmp_path: Path,
) -> None:
    seen: list[int] = []

    @asynccontextmanager
    async def contexts(
        inst: ToyInstance, seed: int, run: RunIdentity
    ) -> AsyncIterator[StubContext]:
        yield StubContext(inst, seed, run)
        task = asyncio.current_task()
        assert task is not None
        seen.append(task.cancelling())  # What the code that closes the context sees.

    async def best_of_n(task: TaskInput, env: Environment, ctx: StubContext) -> Prediction | None:
        await leave_a_phantom_cancel_request()
        return None

    runner = Runner(
        ScriptedBenchmark(toy_instances(1)), ScriptedPolicy({"t0": best_of_n}), contexts
    )

    await run(runner, tmp_path)

    assert seen == [0]
    assert results(tmp_path)[("t0", 0)].error_type == TASK_ERROR_TYPE_NONE


async def test_pack_code_that_converts_a_cancellation_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    started = asyncio.Event()
    policy = ScriptedPolicy({"t0": in_sleep(started, cleanup_fails)})
    running = asyncio.create_task(
        run(make_runner(ScriptedBenchmark(toy_instances(1)), policy), tmp_path)
    )
    async with asyncio.timeout(GUARD_S):
        await started.wait()

    with caplog.at_level(logging.WARNING, logger="goaly.harness.core.runner"):
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    assert "turned a cancellation into RuntimeError: cleanup failed" in caplog.text


class CancelResistingBenchmark(ScriptedBenchmark):
    """Its grader turns a cancellation into an error of its own, or swallows it."""

    def __init__(
        self, instances: Sequence[ToyInstance], grading: asyncio.Event, *, swallows: bool
    ) -> None:
        super().__init__(instances)
        self.grading = grading
        self.swallows = swallows

    async def grade(self, inst: ToyInstance, prediction: Prediction) -> Grade:
        self.grading.set()
        try:
            await asyncio.sleep(GUARD_S)
        except asyncio.CancelledError:
            if not self.swallows:
                raise RuntimeError("the grading instance went away") from None
        return await super().grade(inst, prediction)


@pytest.mark.parametrize("swallows", [False, True], ids=["converts", "swallows"])
async def test_a_grader_cannot_turn_a_cancellation_of_the_run_into_a_grade(
    tmp_path: Path, swallows: bool
) -> None:
    grading = asyncio.Event()
    benchmark = CancelResistingBenchmark(toy_instances(1), grading, swallows=swallows)
    running = asyncio.create_task(run(make_runner(benchmark), tmp_path))
    async with asyncio.timeout(GUARD_S):
        await grading.wait()

    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running
    assert results(tmp_path) == {}


# Concurrency.


async def test_at_most_max_tasks_run_at_once(tmp_path: Path) -> None:
    benchmark = ScriptedBenchmark(toy_instances(12))

    await run(make_runner(benchmark, ScriptedPolicy(delay_s=0.01), max_tasks=3), tmp_path / "run")

    assert benchmark.most_running == 3
    assert len(results(tmp_path / "run")) == 12


async def test_grading_holds_no_task_slot(tmp_path: Path) -> None:
    t0_grading, t1_grading = asyncio.Event(), asyncio.Event()
    benchmark = ScriptedBenchmark(
        toy_instances(2),
        grade_started={"t0": t0_grading, "t1": t1_grading},
        grade_waits={"t0": t1_grading, "t1": t0_grading},
    )

    # With one slot, the two gradings, which wait for each other, can only run side by side if
    # neither holds a slot; grading waits for the adapter's own pool instead.
    await run(make_runner(benchmark, max_tasks=1), tmp_path / "run")

    assert len(results(tmp_path / "run")) == 2


# Runs on SPLIT_EXTERNAL.


async def test_an_external_run_is_not_graded_but_every_task_gets_its_official_record(
    tmp_path: Path,
) -> None:
    # The organizers' tasks reuse our IDs and inputs, which the cache must never mistake for ours.
    cache = EvalCache(tmp_path / "cache", CACHE_NAMESPACE)
    await run(make_runner(ScriptedBenchmark(toy_instances(3)), cache=cache), tmp_path / "ours")
    organizers = [replace(inst, split=SPLIT_EXTERNAL, gold_sql=None) for inst in toy_instances(3)]
    benchmark = ScriptedBenchmark(toy_instances(3), external=organizers)
    policy = ScriptedPolicy({"t1": raising(ValueError("boom")), "t2": answering("SELECT 2")})

    await run(make_runner(benchmark, policy, cache=cache), tmp_path / "run", split=SPLIT_EXTERNAL)

    assert sorted(policy.solved) == [("t0", 0), ("t1", 0), ("t2", 0)]
    assert benchmark.graded == []
    found = results(tmp_path / "run")
    assert not any(result.HasField("grade") for result in found.values())
    assert found[("t1", 0)].error_type == TASK_ERROR_TYPE_POLICY_CRASH
    assert official(tmp_path / "run") == [
        base_answer(0),
        base_answer(1),
        {"instance_id": "t2", "sql": "SELECT 2"},
    ]


# The evaluation cache.


async def test_a_cached_evaluation_is_recorded_again_without_solving(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "cache", CACHE_NAMESPACE)
    policy = ScriptedPolicy({"t1": answering("SELECT 1")})
    await run(make_runner(ScriptedBenchmark(toy_instances(3)), policy, cache=cache), tmp_path / "a")

    again = ScriptedPolicy()
    benchmark = ScriptedBenchmark(toy_instances(3))
    summary = await run(
        make_runner(benchmark, again, cache=cache), tmp_path / "b", manifest=template("run-2")
    )

    assert again.solved == []
    assert benchmark.graded == []
    first, second = results(tmp_path / "a"), results(tmp_path / "b")
    for key, result in first.items():
        assert second[key].run_id == "run-2"
        second[key].run_id = result.run_id
        assert second[key] == result
    for key, trajectory in trajectories(tmp_path / "a").items():
        copy = trajectories(tmp_path / "b")[key]
        assert copy.run_id == "run-2"
        copy.run_id = trajectory.run_id
        assert copy == trajectory
    assert official(tmp_path / "b") == official(tmp_path / "a")
    assert official(tmp_path / "b")[1] == {"instance_id": "t1", "sql": "SELECT 1"}
    assert summary == RunSummary(run_id="run-2", tasks=3, resumed=0, cached=3, ran=0, errors={})


async def test_only_clean_evaluations_are_cached(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "cache", CACHE_NAMESPACE)
    benchmark = ScriptedBenchmark(
        toy_instances(3), grade_errors={"t1": RuntimeError("grader down")}
    )
    policy = ScriptedPolicy({"t0": raising(EnvironmentFailure("gone"))})
    await run(make_runner(benchmark, policy, cache=cache), tmp_path / "a")

    again = ScriptedPolicy()
    await run(
        make_runner(ScriptedBenchmark(toy_instances(3)), again, cache=cache),
        tmp_path / "b",
        manifest=template("run-2"),
    )

    # A task error or a grader error is no property of the harness: those tasks run again.
    assert sorted(again.solved) == [("t0", 0), ("t1", 0)]


class GradingTimesOut(ScriptedBenchmark):
    """Its grader gives up on t1 at its own time limit."""

    async def grade(self, inst: ToyInstance, prediction: Prediction) -> Grade:
        grade = await super().grade(inst, prediction)
        if inst.instance_id == "t1":
            grade = Grade(instance_id="t1", status=GRADE_STATUS_TIMEOUT, feedback="300 s")
        return grade


async def test_outcomes_that_depend_on_the_load_are_not_cached(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "cache", CACHE_NAMESPACE)

    def out_of_time(ctx: StubContext) -> None:
        if ctx.instance_id == "t0":  # The fallback answered: the wall-clock budget ran out.
            ctx.recorded.stop_reason = STOP_REASON_TIME_BUDGET_EXHAUSTED

    first = make_runner(
        GradingTimesOut(toy_instances(3)), contexts=StubContexts(setup=out_of_time), cache=cache
    )
    await run(first, tmp_path / "a")

    again = ScriptedPolicy()
    await run(
        make_runner(ScriptedBenchmark(toy_instances(3)), again, cache=cache),
        tmp_path / "b",
        manifest=template("run-2"),
    )

    # A slow gateway or a busy grading instance is no property of the harness.
    assert sorted(again.solved) == [("t0", 0), ("t1", 0)]


@pytest.mark.parametrize(
    ("manifest", "seeds"),
    [(template("run-2", harness_hash="harness-b"), (0,)), (template("run-2"), (1,))],
    ids=["another harness", "another seed"],
)
async def test_another_harness_or_seed_misses_the_cache(
    tmp_path: Path, manifest: RunManifest, seeds: Sequence[int]
) -> None:
    cache = EvalCache(tmp_path / "cache", CACHE_NAMESPACE)
    await run(make_runner(ScriptedBenchmark(toy_instances(2)), cache=cache), tmp_path / "a")

    again = ScriptedPolicy()
    await run(
        make_runner(ScriptedBenchmark(toy_instances(2)), again, cache=cache),
        tmp_path / "b",
        manifest=manifest,
        seeds=seeds,
    )

    assert len(again.solved) == 2


async def test_a_cache_that_cannot_be_written_does_not_fail_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    not_a_directory = tmp_path / "cache"
    not_a_directory.write_text("", encoding="utf-8")

    await run(
        make_runner(
            ScriptedBenchmark(toy_instances(2)), cache=EvalCache(not_a_directory, CACHE_NAMESPACE)
        ),
        tmp_path / "run",
    )

    assert len(results(tmp_path / "run")) == 2
    assert "could not cache" in caplog.text


# Resuming.


async def interrupt_after(runner: ToyRunner, run_dir: Path, recorded: int) -> None:
    """Start a run in ``run_dir``, and cancel it once ``recorded`` results are on disk."""
    files = RunFiles(run_dir)
    running = asyncio.create_task(run(runner, run_dir))
    await wait_until(
        lambda: files.results.exists() and len(files.results.read_bytes().splitlines()) >= recorded
    )
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


async def test_resume_runs_only_the_tasks_without_a_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gate = asyncio.Event()  # Never set: the run is interrupted while t2 and t3 wait.
    blocked = ScriptedPolicy({"t2": waiting(gate), "t3": waiting(gate)})
    await interrupt_after(make_runner(ScriptedBenchmark(toy_instances(4)), blocked), tmp_path, 2)
    assert set(results(tmp_path)) == {("t0", 0), ("t1", 0)}
    assert not RunFiles(tmp_path).official(0).exists()
    assert not read_manifest(RunFiles(tmp_path).manifest).HasField("finished_at")

    resumed = ScriptedPolicy()
    with caplog.at_level(logging.INFO, logger="goaly.harness.core.runner"):
        summary = await run(
            make_runner(ScriptedBenchmark(toy_instances(4)), resumed),
            tmp_path,
            manifest=template("ignored-on-resume"),
            resume=True,
        )

    assert sorted(resumed.solved) == [("t2", 0), ("t3", 0)]
    assert set(results(tmp_path)) == {(f"t{n}", 0) for n in range(4)}
    assert {result.run_id for result in results(tmp_path).values()} == {"run-1"}
    assert official(tmp_path) == [base_answer(n) for n in range(4)]
    manifest = read_manifest(RunFiles(tmp_path).manifest)
    assert manifest.run_id == "run-1"
    assert manifest.HasField("finished_at")
    assert summary == RunSummary(run_id="run-1", tasks=4, resumed=2, cached=0, ran=2, errors={})
    assert "resuming run run-1, not ignored-on-resume" in caplog.text


async def test_resume_runs_each_seed_of_a_task_that_lacks_a_result(tmp_path: Path) -> None:
    never = asyncio.Event()

    async def first_seed_hangs(task: TaskInput, env: Environment, ctx: StubContext) -> None:
        if ctx.seed == 0:
            await never.wait()

    files = RunFiles(tmp_path)
    blocked = make_runner(
        ScriptedBenchmark(toy_instances(2)), ScriptedPolicy({"t0": first_seed_hangs})
    )
    running = asyncio.create_task(run(blocked, tmp_path, seeds=(0, 1)))
    await wait_until(
        lambda: files.results.exists() and files.results.read_bytes().count(b"\n") == 3
    )
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert set(results(tmp_path)) == {("t1", 0), ("t0", 1), ("t1", 1)}

    resumed = ScriptedPolicy()
    await run(
        make_runner(ScriptedBenchmark(toy_instances(2)), resumed),
        tmp_path,
        seeds=(0, 1),
        resume=True,
    )

    # (t0, 1) has a result and (t0, 0) has not: the resume key is the instance and the seed.
    assert resumed.solved == [("t0", 0)]
    assert len(official(tmp_path, 0)) == len(official(tmp_path, 1)) == 2


async def test_resume_reruns_a_task_whose_result_line_was_torn(tmp_path: Path) -> None:
    await run(make_runner(ScriptedBenchmark(toy_instances(2))), tmp_path)
    files = RunFiles(tmp_path)
    whole = files.results.read_bytes()
    files.results.write_bytes(whole[: whole.rindex(b"\n", 0, len(whole) - 1) + 20])  # Torn.
    torn = next(iter(set(results(tmp_path)) ^ {("t0", 0), ("t1", 0)}))

    resumed = ScriptedPolicy()
    await run(make_runner(ScriptedBenchmark(toy_instances(2)), resumed), tmp_path, resume=True)

    assert resumed.solved == [torn]
    assert set(results(tmp_path)) == {("t0", 0), ("t1", 0)}
    assert official(tmp_path) == [base_answer(0), base_answer(1)]


async def test_resume_submits_the_rerun_answer_of_a_task_that_lost_its_result(
    tmp_path: Path,
) -> None:
    # A crash between a task's two writes leaves its trajectory without its result.
    await run(
        make_runner(ScriptedBenchmark(toy_instances(2)), ScriptedPolicy({"t0": answering("old")})),
        tmp_path,
    )
    files = RunFiles(tmp_path)
    lines = files.results.read_text(encoding="utf-8").splitlines(keepends=True)
    files.results.write_text(
        "".join(line for line in lines if '"t0"' not in line), encoding="utf-8"
    )

    resumed = ScriptedPolicy({"t0": answering("new")})
    await run(make_runner(ScriptedBenchmark(toy_instances(2)), resumed), tmp_path, resume=True)

    assert resumed.solved == [("t0", 0)]
    assert official(tmp_path)[0] == {"instance_id": "t0", "sql": "new"}


async def test_retry_errors_reruns_the_tasks_whose_last_result_is_an_error(
    tmp_path: Path,
) -> None:
    flaky = ScriptedPolicy(
        {
            "t0": raising(EnvironmentFailure("the database restarted")),
            "t1": raising(ValueError("x")),
        }
    )
    await run(make_runner(ScriptedBenchmark(toy_instances(3)), flaky), tmp_path)

    healthy = ScriptedPolicy({"t0": answering("SELECT 0"), "t1": answering("SELECT 1")})
    async with asyncio.timeout(GUARD_S):
        summary = await make_runner(ScriptedBenchmark(toy_instances(3)), healthy).run(
            tmp_path, template(), split=SPLIT_TRAIN, seeds=[0], resume=True, retry_errors=True
        )

    assert sorted(healthy.solved) == [("t0", 0), ("t1", 0)]
    assert {key: result.error_type for key, result in results_last(tmp_path).items()} == {
        ("t0", 0): TASK_ERROR_TYPE_NONE,
        ("t1", 0): TASK_ERROR_TYPE_NONE,
        ("t2", 0): TASK_ERROR_TYPE_NONE,
    }
    assert official(tmp_path)[:2] == [
        {"instance_id": "t0", "sql": "SELECT 0"},
        {"instance_id": "t1", "sql": "SELECT 1"},
    ]
    assert (summary.resumed, summary.ran) == (1, 2)


async def test_a_resume_without_retry_errors_keeps_the_errors(tmp_path: Path) -> None:
    flaky = ScriptedPolicy({"t0": raising(EnvironmentFailure("the database restarted"))})
    await run(make_runner(ScriptedBenchmark(toy_instances(2)), flaky), tmp_path)

    again = ScriptedPolicy()
    await run(make_runner(ScriptedBenchmark(toy_instances(2)), again), tmp_path, resume=True)

    assert again.solved == []


async def test_resuming_a_finished_run_only_writes_its_official_files_again(
    tmp_path: Path,
) -> None:
    await run(make_runner(ScriptedBenchmark(toy_instances(2))), tmp_path)
    RunFiles(tmp_path).official(0).unlink()

    resumed = ScriptedPolicy()
    summary = await run(
        make_runner(ScriptedBenchmark(toy_instances(2)), resumed), tmp_path, resume=True
    )

    assert resumed.solved == []
    assert official(tmp_path) == [base_answer(0), base_answer(1)]
    assert summary.resumed == 2


async def test_resume_reruns_every_task_of_a_run_without_records_files(tmp_path: Path) -> None:
    # As after an interruption between writing the manifest and creating the records files.
    await run(make_runner(ScriptedBenchmark(toy_instances(2))), tmp_path)
    files = RunFiles(tmp_path)
    files.results.unlink()
    files.trajectories.unlink()

    resumed = ScriptedPolicy()
    await run(make_runner(ScriptedBenchmark(toy_instances(2)), resumed), tmp_path, resume=True)

    assert sorted(resumed.solved) == [("t0", 0), ("t1", 0)]
    assert len(results(tmp_path)) == 2


async def test_finishing_needs_a_trajectory_of_every_task(tmp_path: Path) -> None:
    await run(make_runner(ScriptedBenchmark(toy_instances(2))), tmp_path)
    files = RunFiles(tmp_path)
    lines = files.trajectories.read_text(encoding="utf-8").splitlines(keepends=True)
    files.trajectories.write_text("".join(line for line in lines if '"t0"' not in line), "utf-8")

    with pytest.raises(RunDirectoryError, match="no trajectory"):
        await run(make_runner(ScriptedBenchmark(toy_instances(2))), tmp_path, resume=True)


async def test_resume_without_a_run_starts_one(tmp_path: Path) -> None:
    summary = await run(make_runner(ScriptedBenchmark(toy_instances(2))), tmp_path, resume=True)

    assert summary.ran == 2
    assert read_manifest(RunFiles(tmp_path).manifest).run_id == "run-1"


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"manifest": template(harness_hash="harness-b")}, "harness_hash"),
        ({"manifest": template(config_hash="config-b")}, "config_hash"),
        ({"grading_mode": "official"}, "environment_settings"),
        ({"dataset": "rev-b"}, "dataset_revision"),
        ({"manifest": template(git_commit="1" * 40)}, "git_commit"),
        ({"manifest": template(uv_lock_sha256="u" * 64)}, "uv_lock_sha256"),
        (
            {"manifest": with_models(ModelSettings(model="qwen3.6-35b-a3b", temperature=0))},
            "models",
        ),
        ({"seeds": (0, 1)}, "seeds"),
        ({"instance_ids": ["t0"]}, "instance_ids"),
        ({"split": SPLIT_VAL}, "split"),
    ],
    ids=[
        "harness",
        "config",
        "grading mode",
        "dataset",
        "commit",
        "uv.lock",
        "models",
        "seeds",
        "instances",
        "split",
    ],
)
async def test_resume_refuses_a_run_that_does_not_match(
    tmp_path: Path, change: dict[str, object], field: str
) -> None:
    instances = [*toy_instances(2), ToyInstance("v0", SPLIT_VAL, "SELECT 9 FROM slow", "SELECT 9")]
    await run(make_runner(ScriptedBenchmark(instances)), tmp_path)
    before = RunFiles(tmp_path).manifest.read_bytes()
    options = dict(change)
    resumed = ScriptedBenchmark(instances, grading_mode=str(options.pop("grading_mode", "dev")))
    if "dataset" in options:
        revision = str(options.pop("dataset"))
        resumed.manifest_fields = lambda: RunManifest(dataset_revision=revision)  # type: ignore[method-assign]

    with pytest.raises(RunDirectoryError, match=field):
        await run(make_runner(resumed), tmp_path, resume=True, **options)  # type: ignore[arg-type]

    assert RunFiles(tmp_path).manifest.read_bytes() == before


async def test_a_new_run_refuses_a_directory_that_holds_a_run(tmp_path: Path) -> None:
    await run(make_runner(ScriptedBenchmark(toy_instances(1))), tmp_path)

    with pytest.raises(RunDirectoryError, match="run-1"):
        await run(make_runner(ScriptedBenchmark(toy_instances(1))), tmp_path)


async def test_a_new_run_refuses_records_without_a_manifest(tmp_path: Path) -> None:
    RunFiles(tmp_path).results.write_text("", encoding="utf-8")

    with pytest.raises(RunDirectoryError, match=r"results\.jsonl"):
        await run(make_runner(ScriptedBenchmark(toy_instances(1))), tmp_path, resume=True)


# Cancellation and the run lock.


async def test_cancellation_propagates_and_leaves_only_whole_records(tmp_path: Path) -> None:
    gate = asyncio.Event()
    benchmark = ScriptedBenchmark(toy_instances(4))
    contexts = StubContexts()
    policy = ScriptedPolicy({f"t{n}": waiting(gate) for n in (1, 2, 3)})

    await interrupt_after(make_runner(benchmark, policy, contexts), tmp_path, 1)

    assert all(env.closed for env in benchmark.environments)
    assert all(ctx.exited for ctx in contexts.made)
    assert set(results(tmp_path)) == {("t0", 0)}  # read_jsonl raises on any torn line.
    assert set(trajectories(tmp_path)) == {("t0", 0)}
    assert not RunFiles(tmp_path).official(0).parent.exists()
    # The lock went with the run: the run can be resumed at once.
    await run(make_runner(ScriptedBenchmark(toy_instances(4))), tmp_path, resume=True)
    assert len(results(tmp_path)) == 4


async def test_a_task_whose_trajectory_cannot_be_written_gets_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_append(path: Path, message: Message) -> None:
        if isinstance(message, Trajectory):
            raise OSError("disk full")
        append_jsonl(path, message)

    monkeypatch.setattr(jsonl_writer, "append_jsonl", failing_append)

    # A records file that cannot be written stops the run: no task can be recorded.
    with pytest.raises(ExceptionGroup) as stopped:
        await run(make_runner(ScriptedBenchmark(toy_instances(2))), tmp_path)

    assert stopped.group_contains(JsonlWriterError, match="disk full")
    assert results(tmp_path) == {}  # A result is written only after its trajectory.
    assert not RunFiles(tmp_path).official(0).exists()


async def test_a_run_directory_in_use_is_refused(tmp_path: Path) -> None:
    files = RunFiles(tmp_path)
    with files.lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # As another process would.

        with pytest.raises(RunDirectoryError, match="another process"):
            await run(make_runner(ScriptedBenchmark(toy_instances(1))), tmp_path)

    assert not files.manifest.exists()


async def test_a_run_can_hold_the_lock_its_caller_took(tmp_path: Path) -> None:
    with lock_run_dir(RunFiles(tmp_path)) as lock:
        async with asyncio.timeout(GUARD_S):
            summary = await make_runner(ScriptedBenchmark(toy_instances(2))).run(
                tmp_path, template(), split=SPLIT_TRAIN, seeds=[0], lock=lock
            )
        # The caller's lock is the caller's to release: the run left it held.
        with pytest.raises(RunDirectoryError, match="another process"):
            lock_run_dir(RunFiles(tmp_path))

    assert summary.ran == 2
    lock_run_dir(RunFiles(tmp_path)).release()  # Released with the caller's block.


async def test_a_lock_on_another_directory_is_refused(tmp_path: Path) -> None:
    with (
        lock_run_dir(RunFiles(tmp_path / "other")) as lock,
        pytest.raises(ValueError, match="lock"),
    ):
        await make_runner(ScriptedBenchmark(toy_instances(1))).run(
            tmp_path / "run", template(), split=SPLIT_TRAIN, seeds=[0], lock=lock
        )


# Arguments the runner refuses.


async def test_an_unknown_instance_id_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="'t9'"):
        await run(make_runner(ScriptedBenchmark(toy_instances(2))), tmp_path, instance_ids=["t9"])

    assert not RunFiles(tmp_path).manifest.exists()


async def test_duplicate_instance_ids_from_the_benchmark_are_refused(tmp_path: Path) -> None:
    twice = toy_instances(2) + toy_instances(1)

    with pytest.raises(ValueError, match="'t0'"):
        await run(make_runner(ScriptedBenchmark(twice)), tmp_path)


async def test_a_run_needs_its_id_and_a_cache_needs_the_harness_hash(tmp_path: Path) -> None:
    runner = make_runner(
        ScriptedBenchmark(toy_instances(1)), cache=EvalCache(tmp_path / "c", CACHE_NAMESPACE)
    )

    with pytest.raises(ValueError, match="run_id"):
        await run(runner, tmp_path / "a", manifest=template(run_id=""))
    with pytest.raises(ValueError, match="harness_hash"):
        await run(runner, tmp_path / "b", manifest=template(harness_hash=""))

    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()


@pytest.mark.parametrize("seeds", [(), (0, 0), (-1,)], ids=["none", "repeated", "negative"])
async def test_seeds_must_be_distinct_replicate_indices(
    tmp_path: Path, seeds: Sequence[int]
) -> None:
    with pytest.raises(ValueError, match="seed"):
        await run(make_runner(ScriptedBenchmark(toy_instances(1))), tmp_path, seeds=seeds)


@pytest.mark.parametrize(
    ("max_tasks", "task_timeout_s"),
    [(0, None), (1, 0.0), (1, -1.0), (1, math.inf), (1, math.nan)],
)
def test_limits_must_be_positive(max_tasks: int, task_timeout_s: float | None) -> None:
    with pytest.raises(ValueError, match="must be"):
        make_runner(
            ScriptedBenchmark(toy_instances(1)), max_tasks=max_tasks, task_timeout_s=task_timeout_s
        )


# The run directory and error classification.


def test_run_files_lay_out_the_run_directory(tmp_path: Path) -> None:
    files = RunFiles(tmp_path)

    assert files.manifest == tmp_path / "manifest.json"
    assert files.results == tmp_path / "results.jsonl"
    assert files.trajectories == tmp_path / "trajectories.jsonl"
    assert files.llm_calls == tmp_path / "llm_calls.jsonl"
    assert files.official(2) == tmp_path / "official" / "seed-2.jsonl"
    assert files.lock == tmp_path / ".lock"


class GatewayDownError(Exception):
    """Stands in for the LLM client's error after its retries."""


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (EnvironmentFailure("x"), TASK_ERROR_TYPE_ENVIRONMENT),
        (ExceptionGroup("g", [EnvironmentFailure("x")]), TASK_ERROR_TYPE_ENVIRONMENT),
        (
            ExceptionGroup("g", [ValueError(), ExceptionGroup("h", [EnvironmentFailure()])]),
            TASK_ERROR_TYPE_ENVIRONMENT,
        ),
        (ValueError("x"), TASK_ERROR_TYPE_POLICY_CRASH),
        (ExceptionGroup("g", [ValueError("x")]), TASK_ERROR_TYPE_POLICY_CRASH),
        (GatewayDownError(), TASK_ERROR_TYPE_POLICY_CRASH),
    ],
)
def test_the_default_classifier(error: Exception, expected: int) -> None:
    assert classify_error(error) == expected


def test_rules_can_be_a_module_constant() -> None:
    # A module-level list of rules type-checks, as a dict keyed by one exception type would not.
    rules: list[tuple[type[Exception], TaskErrorType.ValueType]] = [
        (GatewayDownError, TASK_ERROR_TYPE_LLM)
    ]

    assert error_classifier(rules)(GatewayDownError()) == TASK_ERROR_TYPE_LLM


def test_extra_rules_come_after_the_environment_rule() -> None:
    classify = error_classifier([(GatewayDownError, TASK_ERROR_TYPE_LLM)])

    assert classify(GatewayDownError()) == TASK_ERROR_TYPE_LLM
    assert classify(ExceptionGroup("g", [GatewayDownError()])) == TASK_ERROR_TYPE_LLM
    both = ExceptionGroup("g", [GatewayDownError(), EnvironmentFailure()])
    assert classify(both) == TASK_ERROR_TYPE_ENVIRONMENT
    assert classify(ValueError()) == TASK_ERROR_TYPE_POLICY_CRASH


def _typing_checks(ctx: StubContext) -> None:
    """Never called: ``make typecheck`` proves that the stub context is a TaskContext."""
    _: TaskContext = ctx
