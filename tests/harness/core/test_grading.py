"""Tests for goaly.harness.core.grading: the grade phase of a two-phase run (spec V24)."""

import asyncio
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_ENVIRONMENT,
    TASK_ERROR_TYPE_NONE,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
)
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core.eval_cache import EvalCache
from goaly.harness.core.grading import Grader
from goaly.harness.core.manifest import read_manifest
from goaly.harness.core.runner import RunDirectoryError, RunFiles, Runner, RunSummary, lock_run_dir
from tests.harness.core.runner_doubles import (
    ScriptedBenchmark,
    ScriptedPolicy,
    StubContexts,
    answering,
    raising,
    toy_instances,
)

pytestmark = pytest.mark.anyio

GUARD_S = 10
NAMESPACE = "toy-dev"


def manifest(run_id: str = "run-1") -> RunManifest:
    return RunManifest(run_id=run_id, harness_hash="harness-a")


async def solve(
    run_dir: Path,
    benchmark: ScriptedBenchmark,
    policy: ScriptedPolicy | None = None,
    *,
    cache: EvalCache | None = None,
    seeds: tuple[int, ...] = (0,),
) -> RunSummary:
    """The solve phase: every task recorded, ungraded."""
    runner = Runner(benchmark, policy or ScriptedPolicy(), StubContexts(), cache=cache)
    async with asyncio.timeout(GUARD_S):
        return await runner.run(run_dir, manifest(), split=SPLIT_TRAIN, seeds=seeds, grading=False)


def lines(run_dir: Path) -> list[TaskResult]:
    return list(read_jsonl(RunFiles(run_dir).results, TaskResult))


def last(run_dir: Path) -> dict[tuple[str, int], TaskResult]:
    return {(result.instance_id, result.seed): result for result in lines(run_dir)}


# Grading a solved run.


async def test_grades_every_task_and_appends_one_graded_line_each(tmp_path: Path) -> None:
    await solve(tmp_path, ScriptedBenchmark(toy_instances(3)), seeds=(0, 1))
    official = RunFiles(tmp_path).official(0).read_bytes()
    benchmark = ScriptedBenchmark(toy_instances(3))

    summary = await Grader(benchmark).grade(tmp_path)

    assert summary == RunSummary(run_id="run-1", tasks=6, resumed=0, cached=0, ran=6, errors={})
    graded_lines = Counter(
        (result.instance_id, result.seed) for result in lines(tmp_path) if result.HasField("grade")
    )
    assert set(graded_lines.values()) == {1}  # Exactly one graded line per task...
    assert len(graded_lines) == 6
    for result in last(tmp_path).values():  # ...and it is the last line, which wins.
        assert result.grade.status == GRADE_STATUS_GRADED
        assert result.grade.score == 0.5
        assert result.run_id == "run-1"
    assert sorted(prediction.instance_id for prediction in benchmark.graded) == sorted(
        f"t{n}" for n in range(3) for _ in range(2)
    )
    assert read_manifest(RunFiles(tmp_path).manifest).HasField("finished_at")
    assert RunFiles(tmp_path).official(0).read_bytes() == official  # Untouched.


async def test_a_graded_result_keeps_everything_the_solve_phase_recorded(tmp_path: Path) -> None:
    policy = ScriptedPolicy({"t1": raising(EnvironmentFailure("gone"))})
    await solve(tmp_path, ScriptedBenchmark(toy_instances(2)), policy)
    solved = last(tmp_path)

    summary = await Grader(ScriptedBenchmark(toy_instances(2))).grade(tmp_path)

    for key, result in last(tmp_path).items():
        expected = TaskResult()
        expected.CopyFrom(solved[key])
        expected.grade.CopyFrom(result.grade)
        assert result == expected
    assert last(tmp_path)[("t1", 0)].error_type == TASK_ERROR_TYPE_ENVIRONMENT
    assert summary.errors == {"TASK_ERROR_TYPE_ENVIRONMENT": 1}


async def test_a_task_without_a_prediction_is_graded_as_its_empty_answer(tmp_path: Path) -> None:
    policy = ScriptedPolicy({"t0": raising(ValueError("boom")), "t1": answering("SELECT 1")})
    await solve(tmp_path, ScriptedBenchmark(toy_instances(2)), policy)
    benchmark = ScriptedBenchmark(toy_instances(2))

    await Grader(benchmark).grade(tmp_path)

    graded = {prediction.instance_id: prediction for prediction in benchmark.graded}
    assert graded["t0"].WhichOneof("benchmark") is None
    assert graded["t1"].effisql.sql == "SELECT 1"
    assert last(tmp_path)[("t1", 0)].grade.score == 1.0  # The toy's gold answer.


async def test_a_grader_that_raises_gives_a_grader_error(tmp_path: Path) -> None:
    await solve(tmp_path, ScriptedBenchmark(toy_instances(2)))
    benchmark = ScriptedBenchmark(toy_instances(2), grade_errors={"t0": RuntimeError("down")})

    await Grader(benchmark).grade(tmp_path)

    found = last(tmp_path)
    assert found[("t0", 0)].grade.status == GRADE_STATUS_GRADER_ERROR
    assert "RuntimeError: down" in found[("t0", 0)].grade.feedback
    assert found[("t0", 0)].error_type == TASK_ERROR_TYPE_NONE
    assert found[("t1", 0)].grade.status == GRADE_STATUS_GRADED


# Resuming.


async def test_an_interrupted_grade_phase_resumes_with_the_tasks_left(tmp_path: Path) -> None:
    await solve(tmp_path, ScriptedBenchmark(toy_instances(3)))
    never = asyncio.Event()
    stuck = ScriptedBenchmark(toy_instances(3), grade_waits={"t1": never, "t2": never})
    grading = asyncio.create_task(Grader(stuck).grade(tmp_path))
    async with asyncio.timeout(GUARD_S):
        while not any(result.HasField("grade") for result in lines(tmp_path)):
            await asyncio.sleep(0.005)
    grading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await grading
    assert not read_manifest(RunFiles(tmp_path).manifest).HasField("finished_at")

    benchmark = ScriptedBenchmark(toy_instances(3))
    summary = await Grader(benchmark).grade(tmp_path)

    assert sorted(prediction.instance_id for prediction in benchmark.graded) == ["t1", "t2"]
    assert (summary.resumed, summary.ran) == (1, 2)
    graded = Counter(result.instance_id for result in lines(tmp_path) if result.HasField("grade"))
    assert graded == {"t0": 1, "t1": 1, "t2": 1}
    assert read_manifest(RunFiles(tmp_path).manifest).HasField("finished_at")


async def test_grading_a_graded_run_again_grades_nothing(tmp_path: Path) -> None:
    await solve(tmp_path, ScriptedBenchmark(toy_instances(2)))
    await Grader(ScriptedBenchmark(toy_instances(2))).grade(tmp_path)
    before = RunFiles(tmp_path).results.read_bytes()
    benchmark = ScriptedBenchmark(toy_instances(2))

    summary = await Grader(benchmark).grade(tmp_path)

    assert benchmark.graded == []
    assert (summary.resumed, summary.ran) == (2, 0)
    assert RunFiles(tmp_path).results.read_bytes() == before


# The evaluation cache: only the grade phase writes it.


async def test_clean_graded_evaluations_go_into_the_cache_for_later_solve_phases(
    tmp_path: Path,
) -> None:
    cache = EvalCache(tmp_path / "cache", NAMESPACE)
    policy = ScriptedPolicy({"t2": raising(ValueError("boom"))})
    await solve(tmp_path / "a", ScriptedBenchmark(toy_instances(3)), policy, cache=cache)
    benchmark = ScriptedBenchmark(toy_instances(3), grade_errors={"t1": RuntimeError("down")})

    await Grader(benchmark, cache=cache).grade(tmp_path / "a")

    # The next solve phase takes the clean one, graded, and solves the others again.
    again = ScriptedPolicy()
    summary = await solve(tmp_path / "b", ScriptedBenchmark(toy_instances(3)), again, cache=cache)
    assert sorted(again.solved) == [("t1", 0), ("t2", 0)]
    assert summary.cached == 1
    hit = last(tmp_path / "b")[("t0", 0)]
    assert hit.grade.status == GRADE_STATUS_GRADED
    assert hit.run_id == "run-1"


async def test_a_task_the_solve_phase_took_from_the_cache_is_not_graded_again(
    tmp_path: Path,
) -> None:
    cache = EvalCache(tmp_path / "cache", NAMESPACE)
    await solve(tmp_path / "a", ScriptedBenchmark(toy_instances(2)), cache=cache)
    await Grader(ScriptedBenchmark(toy_instances(2)), cache=cache).grade(tmp_path / "a")
    await solve(tmp_path / "b", ScriptedBenchmark(toy_instances(2)), cache=cache)
    benchmark = ScriptedBenchmark(toy_instances(2))

    summary = await Grader(benchmark, cache=cache).grade(tmp_path / "b")

    assert benchmark.graded == []
    assert summary.resumed == 2


async def test_a_cache_that_cannot_be_written_does_not_stop_the_grading(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    await solve(tmp_path / "run", ScriptedBenchmark(toy_instances(2)))
    not_a_directory = tmp_path / "cache"
    not_a_directory.write_text("", encoding="utf-8")
    grader = Grader(
        ScriptedBenchmark(toy_instances(2)), cache=EvalCache(not_a_directory, NAMESPACE)
    )

    summary = await grader.grade(tmp_path / "run")

    assert summary.ran == 2
    assert all(result.HasField("grade") for result in last(tmp_path / "run").values())
    assert "could not cache" in caplog.text


async def test_the_cache_needs_the_runs_harness_hash(tmp_path: Path) -> None:
    runner = Runner(ScriptedBenchmark(toy_instances(1)), ScriptedPolicy(), StubContexts())
    async with asyncio.timeout(GUARD_S):
        await runner.run(
            tmp_path, RunManifest(run_id="r"), split=SPLIT_TRAIN, seeds=[0], grading=False
        )
    grader = Grader(ScriptedBenchmark(toy_instances(1)), cache=EvalCache(tmp_path / "c", NAMESPACE))

    with pytest.raises(ValueError, match="harness_hash"):
        await grader.grade(tmp_path)


# Runs that cannot be graded.


async def test_a_directory_without_a_run_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunDirectoryError, match="holds no run"):
        await Grader(ScriptedBenchmark(toy_instances(1))).grade(tmp_path / "missing")

    assert not (tmp_path / "missing").exists()


async def test_a_run_whose_solve_phase_is_unfinished_is_refused(tmp_path: Path) -> None:
    await solve(tmp_path, ScriptedBenchmark(toy_instances(2)))
    files = RunFiles(tmp_path)
    kept = [
        line
        for line in files.results.read_text(encoding="utf-8").splitlines()
        if '"t1"' not in line
    ]
    files.results.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")

    with pytest.raises(RunDirectoryError, match="resume its solve phase"):
        await Grader(ScriptedBenchmark(toy_instances(2))).grade(tmp_path)


async def test_a_run_whose_solve_phase_recorded_nothing_is_refused(tmp_path: Path) -> None:
    await solve(tmp_path, ScriptedBenchmark(toy_instances(1)))
    RunFiles(tmp_path).results.unlink()
    RunFiles(tmp_path).trajectories.unlink()

    with pytest.raises(RunDirectoryError, match="resume its solve phase"):
        await Grader(ScriptedBenchmark(toy_instances(1))).grade(tmp_path)


async def test_a_run_of_another_benchmark_is_refused(tmp_path: Path) -> None:
    await solve(tmp_path, ScriptedBenchmark(toy_instances(1)))
    other = ScriptedBenchmark(toy_instances(1))
    other.name = "other"

    with pytest.raises(RunDirectoryError, match="'toy', not 'other'"):
        await Grader(other).grade(tmp_path)


async def test_a_run_on_split_external_has_nothing_to_grade(tmp_path: Path) -> None:
    organizers = [replace(inst, split=SPLIT_EXTERNAL, gold_sql=None) for inst in toy_instances(1)]
    benchmark = ScriptedBenchmark(toy_instances(1), external=organizers)
    runner = Runner(benchmark, ScriptedPolicy(), StubContexts())
    async with asyncio.timeout(GUARD_S):
        await runner.run(tmp_path, manifest(), split=SPLIT_EXTERNAL, seeds=[0], grading=False)

    with pytest.raises(RunDirectoryError, match="no gold answers"):
        await Grader(benchmark).grade(tmp_path)


async def test_a_run_naming_an_instance_the_benchmark_lacks_is_refused(tmp_path: Path) -> None:
    await solve(tmp_path, ScriptedBenchmark(toy_instances(2)))

    with pytest.raises(RunDirectoryError, match="'t1'"):
        await Grader(ScriptedBenchmark(toy_instances(1))).grade(tmp_path)


# The lock.


async def test_grading_needs_the_run_directory_to_itself(tmp_path: Path) -> None:
    await solve(tmp_path, ScriptedBenchmark(toy_instances(1)))

    with lock_run_dir(RunFiles(tmp_path)), pytest.raises(RunDirectoryError, match="another"):
        await Grader(ScriptedBenchmark(toy_instances(1))).grade(tmp_path)

    with lock_run_dir(RunFiles(tmp_path)) as lock:
        summary = await Grader(ScriptedBenchmark(toy_instances(1))).grade(tmp_path, lock=lock)
    assert summary.ran == 1


async def test_a_lock_on_another_directory_is_refused(tmp_path: Path) -> None:
    await solve(tmp_path / "run", ScriptedBenchmark(toy_instances(1)))

    with (
        lock_run_dir(RunFiles(tmp_path / "other")) as lock,
        pytest.raises(ValueError, match="lock"),
    ):
        await Grader(ScriptedBenchmark(toy_instances(1))).grade(tmp_path / "run", lock=lock)
