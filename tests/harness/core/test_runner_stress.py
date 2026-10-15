"""The runner's stress test (spec §6.2): 500 simulated tasks, with failures and latency injected.

Every task must be recorded exactly once, with the error it was scripted to end with, and no
more than the limit of tasks may run at once, whether the run goes through in one go or is
interrupted and resumed.
"""

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_ENVIRONMENT,
    TASK_ERROR_TYPE_NONE,
    TASK_ERROR_TYPE_POLICY_CRASH,
    TASK_ERROR_TYPE_TIMEOUT,
    TaskErrorType,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    SPLIT_TRAIN,
    GradeStatus,
)
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core.runner import RunFiles, Runner, RunSummary
from tests.fakes import FakeSqlEnvironment, ToyInstance
from tests.harness.core.runner_doubles import (
    Action,
    ScriptedBenchmark,
    ScriptedPolicy,
    StubContext,
    StubContexts,
    raising,
    sleeping,
    toy_instances,
)

pytestmark = pytest.mark.anyio

INSTANCES = 250
SEEDS = (0, 1)  # 500 tasks.
MAX_TASKS = 16
TIME_LIMIT_S = 0.05
GUARD_S = 60


@dataclass(frozen=True)
class Fate:
    """What a scripted instance does, and what its records must then say."""

    error_type: TaskErrorType.ValueType
    grade_status: GradeStatus.ValueType


@dataclass
class Script:
    """The failures and latencies of one stress run, drawn from a seeded generator."""

    provision_errors: dict[str, Exception]
    grade_errors: dict[str, Exception]
    official_errors: dict[str, Exception]
    actions: dict[str, Action]
    fates: dict[str, Fate]

    def benchmark(self) -> ScriptedBenchmark:
        return ScriptedBenchmark(
            toy_instances(INSTANCES),
            provision_errors=self.provision_errors,
            provision_delay_s=0.0005,
            grade_errors=self.grade_errors,
            official_errors=self.official_errors,
        )

    def policy(self) -> ScriptedPolicy:
        return ScriptedPolicy(self.actions)


def draw_script(rng: random.Random) -> Script:
    script = Script({}, {}, {}, {}, {})
    graded, failed = GRADE_STATUS_GRADED, GRADE_STATUS_GRADER_ERROR
    for n in range(INSTANCES):
        name = f"t{n}"
        roll = rng.random()
        if roll < 0.08:
            script.provision_errors[name] = EnvironmentFailure("no solving instance")
            script.fates[name] = Fate(TASK_ERROR_TYPE_ENVIRONMENT, graded)
        elif roll < 0.16:
            script.actions[name] = raising(ValueError("the policy broke"))
            script.fates[name] = Fate(TASK_ERROR_TYPE_POLICY_CRASH, graded)
        elif roll < 0.22:
            group = ExceptionGroup("samples", [EnvironmentFailure("the session died")])
            script.actions[name] = raising(group)
            script.fates[name] = Fate(TASK_ERROR_TYPE_ENVIRONMENT, graded)
        elif roll < 0.28:
            script.grade_errors[name] = RuntimeError("a grading instance went away")
            script.actions[name] = sleeping(rng.uniform(0, 0.003))
            script.fates[name] = Fate(TASK_ERROR_TYPE_NONE, failed)
        elif roll < 0.32:
            script.actions[name] = sleeping(10 * TIME_LIMIT_S)
            script.fates[name] = Fate(TASK_ERROR_TYPE_TIMEOUT, graded)
        elif roll < 0.36:
            script.official_errors[name] = ValueError("no official form")
            script.fates[name] = Fate(TASK_ERROR_TYPE_POLICY_CRASH, graded)
        else:
            script.actions[name] = sleeping(rng.uniform(0, 0.003))
            script.fates[name] = Fate(TASK_ERROR_TYPE_NONE, graded)
    return script


@pytest.fixture
def script() -> Script:
    return draw_script(random.Random(20261015))  # noqa: S311 - Test data, not cryptography.


def make_runner(
    benchmark: ScriptedBenchmark, policy: ScriptedPolicy
) -> Runner[ToyInstance, FakeSqlEnvironment, StubContext]:
    return Runner(
        benchmark, policy, StubContexts(), max_tasks=MAX_TASKS, task_timeout_s=TIME_LIMIT_S
    )


async def run(
    runner: Runner[ToyInstance, FakeSqlEnvironment, StubContext], run_dir: Path, *, resume: bool
) -> RunSummary:
    async with asyncio.timeout(GUARD_S):
        return await runner.run(
            run_dir, RunManifest(run_id="stress"), split=SPLIT_TRAIN, seeds=SEEDS, resume=resume
        )


def assert_recorded_exactly_once(run_dir: Path, script: Script) -> None:
    files = RunFiles(run_dir)
    keys = [(result.instance_id, result.seed) for result in read_jsonl(files.results, TaskResult)]
    expected = {(f"t{n}", seed) for n in range(INSTANCES) for seed in SEEDS}
    assert len(keys) == len(expected) == 500
    assert set(keys) == expected
    for result in read_jsonl(files.results, TaskResult):
        fate = script.fates[result.instance_id]
        assert (result.error_type, result.grade.status) == (fate.error_type, fate.grade_status), (
            result.instance_id
        )
    traced = {
        (trajectory.instance_id, trajectory.seed)
        for trajectory in read_jsonl(files.trajectories, Trajectory)
    }
    assert traced == expected
    for seed in SEEDS:
        lines = files.official(seed).read_text(encoding="utf-8").splitlines()
        assert len(lines) == INSTANCES


async def test_every_one_of_500_tasks_is_recorded_once_within_the_limit(
    tmp_path: Path, script: Script
) -> None:
    benchmark = script.benchmark()

    summary = await run(make_runner(benchmark, script.policy()), tmp_path, resume=False)

    assert_recorded_exactly_once(tmp_path, script)
    assert benchmark.most_running == MAX_TASKS
    assert all(env.closed for env in benchmark.environments)
    assert summary.tasks == summary.ran == 500
    expected_errors = sum(fate.error_type != TASK_ERROR_TYPE_NONE for fate in script.fates.values())
    assert sum(summary.errors.values()) == expected_errors * len(SEEDS)


async def test_an_interrupted_stress_run_resumes_to_exactly_once(
    tmp_path: Path, script: Script
) -> None:
    files = RunFiles(tmp_path)
    first, second = script.benchmark(), script.benchmark()
    running = asyncio.create_task(run(make_runner(first, script.policy()), tmp_path, resume=False))
    recorded: Callable[[], int] = lambda: (  # noqa: E731 - A one-line probe.
        len(files.results.read_bytes().splitlines()) if files.results.exists() else 0
    )
    async with asyncio.timeout(GUARD_S):
        while recorded() < 200:
            await asyncio.sleep(0.005)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    interrupted = {
        (result.instance_id, result.seed) for result in read_jsonl(files.results, TaskResult)
    }
    resumed_policy = script.policy()

    summary = await run(make_runner(second, resumed_policy), tmp_path, resume=True)

    assert_recorded_exactly_once(tmp_path, script)
    assert not interrupted & set(resumed_policy.solved)  # Nothing recorded is solved again.
    assert summary.resumed == len(interrupted)
    assert max(first.most_running, second.most_running) <= MAX_TASKS
    assert all(env.closed for env in first.environments + second.environments)
