"""The script of the runner's stress test (spec §6.2): 500 simulated tasks, with failures and
latency injected, drawn from a seeded generator.

It is a module of its own so that a child process can run it, to be killed with SIGKILL:
``python -m tests.harness.core.stress_script RUN_DIR`` runs the stress run in RUN_DIR.
"""

import asyncio
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_ENVIRONMENT,
    TASK_ERROR_TYPE_NONE,
    TASK_ERROR_TYPE_POLICY_CRASH,
    TASK_ERROR_TYPE_TIMEOUT,
    TaskErrorType,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    SPLIT_TRAIN,
    GradeStatus,
)
from goaly.harness.core.runner import Runner, RunSummary
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

INSTANCES = 250
SEEDS = (0, 1)  # 500 tasks.
MAX_TASKS = 16
# Far above any ordinary task's few milliseconds, so that a slow CI machine's stalls never
# time one out; the scripted time-outs sleep ten times as long.
TIME_LIMIT_S = 0.5
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


def stress_script() -> Script:
    """The script every stress run uses: the same failures and latencies, run after run."""
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


def main(argv: list[str]) -> None:
    """Run the stress run in the directory ``argv[0]``, as a child process a test may kill."""
    script = stress_script()
    runner = make_runner(script.benchmark(), script.policy())
    asyncio.run(run(runner, Path(argv[0]), resume=False))


if __name__ == "__main__":
    main(sys.argv[1:])
