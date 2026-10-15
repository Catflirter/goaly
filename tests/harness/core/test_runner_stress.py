"""The runner's stress test (spec §6.2): 500 simulated tasks, with failures and latency injected.

Every task must be recorded exactly once, with the error it was scripted to end with, and no
more than the limit of tasks may run at once, whether the run goes through in one go, is
interrupted and resumed, is killed with SIGKILL and resumed, or leaves torn lines. Each official
file must list every instance once, in order, with the answer its last trajectory submitted.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from goaly.contracts.gen.result_pb2 import TASK_ERROR_TYPE_NONE, TaskResult
from goaly.contracts.gen.task_pb2 import Prediction
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core.runner import RunFiles
from tests.harness.core.runner_doubles import toy_instances
from tests.harness.core.stress_script import (
    GUARD_S,
    INSTANCES,
    MAX_TASKS,
    SEEDS,
    Script,
    make_runner,
    run,
    stress_script,
)

pytestmark = pytest.mark.anyio

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def script() -> Script:
    return stress_script()


def recorded(files: RunFiles) -> int:
    return len(files.results.read_bytes().splitlines()) if files.results.exists() else 0


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
    last = {
        (trajectory.instance_id, trajectory.seed): trajectory
        for trajectory in read_jsonl(files.trajectories, Trajectory)
    }
    assert set(last) == expected
    # Each official file: every instance once, in order, with what its last trajectory submitted.
    benchmark = script.benchmark()
    for seed in SEEDS:
        lines = files.official(seed).read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == [
            benchmark.to_official(inst, submitted(last[(inst.instance_id, seed)]))
            for inst in toy_instances(INSTANCES)
        ]


def submitted(trajectory: Trajectory) -> Prediction:
    prediction = Prediction()
    prediction.CopyFrom(trajectory.prediction)
    prediction.instance_id = trajectory.instance_id
    return prediction


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
    async with asyncio.timeout(GUARD_S):
        while recorded(files) < 200:
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


async def test_a_stress_run_killed_with_sigkill_resumes_to_exactly_once(
    tmp_path: Path, script: Script
) -> None:
    files = RunFiles(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    # The child runs the same script; its arguments are fixed paths, not untrusted input.
    child = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "tests.harness.core.stress_script", str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
    )
    try:
        async with asyncio.timeout(GUARD_S):
            while recorded(files) < 150:
                assert child.poll() is None, "the child finished before it could be killed"
                await asyncio.sleep(0.005)
        child.send_signal(signal.SIGKILL)  # No cleanup at all: whatever was half-written stays.
    finally:
        child.wait(timeout=GUARD_S)
    assert child.returncode == -signal.SIGKILL

    summary = await run(make_runner(script.benchmark(), script.policy()), tmp_path, resume=True)

    assert_recorded_exactly_once(tmp_path, script)
    assert summary.resumed >= 150


async def test_torn_lines_left_by_a_crash_resume_to_exactly_once(
    tmp_path: Path, script: Script
) -> None:
    files = RunFiles(tmp_path)
    running = asyncio.create_task(
        run(make_runner(script.benchmark(), script.policy()), tmp_path, resume=False)
    )
    async with asyncio.timeout(GUARD_S):
        while recorded(files) < 120:
            await asyncio.sleep(0.005)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    whole_lines = recorded(files)  # The writers drained what was queued as the run stopped.
    # As a crash mid-write leaves them: the last result cut short, a partial trajectory line.
    whole = files.results.read_bytes()
    files.results.write_bytes(whole[: len(whole) - 25])
    with files.trajectories.open("ab") as trajectories:
        trajectories.write(b'{"run_id":"stress","instance_id":"t')

    summary = await run(make_runner(script.benchmark(), script.policy()), tmp_path, resume=True)

    assert_recorded_exactly_once(tmp_path, script)
    assert summary.resumed == whole_lines - 1  # The torn result does not count.
