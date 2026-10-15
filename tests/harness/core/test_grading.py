"""Tests for goaly.harness.core.grading: the host side of a two-phase run (spec §6.1, V27)."""

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord, Usage
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_NONE,
    TASK_ERROR_TYPE_POLICY_CRASH,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    GRADE_STATUS_PREDICTION_FAILED,
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
    EffiSqlPrediction,
    Grade,
    Prediction,
)
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.contracts.jsonl import append_jsonl, read_jsonl, to_json_line
from goaly.harness.core.eval_cache import EvalCache
from goaly.harness.core.grading import GRADING_TASK, HostRunner, SolveOptions
from goaly.harness.core.isolation import SolveRequest, read_solve_tasks
from goaly.harness.core.manifest import read_manifest
from goaly.harness.core.runner import RunDirectoryError, RunFiles, UsageError, lock_run_dir
from tests.fakes import ToyInstance
from tests.harness.core.runner_doubles import (
    ScriptedBenchmark,
    ScriptedPolicy,
    StubContexts,
    answering,
    base_sql,
    raising,
    toy_instances,
)

pytestmark = pytest.mark.anyio

GUARD_S = 10
HASH = "a" * 64

type Key = tuple[str, int]


def manifest(run_id: str = "run-1") -> RunManifest:
    return RunManifest(run_id=run_id, harness_hash=HASH, config_hash="c" * 64)


def solve_options(tmp_path: Path) -> SolveOptions:
    return SolveOptions(
        harness="v0",
        checkout=tmp_path,
        pack_dir=None,
        instances_file=None,
        llm_shares={"qwen3.6-35b-a3b": 4},
        max_tasks=4,
        task_timeout_s=None,
        log_level="INFO",
    )


@dataclass
class Container:
    """The solve container, as far as the host can tell: it runs the toy's Runner in the solve
    directory, as `goaly run --in-container` would, then ``after`` may tamper with it."""

    policy: ScriptedPolicy = field(default_factory=ScriptedPolicy)
    instances: Sequence[ToyInstance] = field(default_factory=lambda: toy_instances(3))
    after: Callable[[SolveRequest], None] | None = None
    fails: bool = False
    requests: list[SolveRequest] = field(default_factory=list)

    async def __call__(self, request: SolveRequest) -> None:
        self.requests.append(request)
        if self.fails:
            raise RuntimeError("the solve container exited with status 1")
        tasks = read_solve_tasks(RunFiles(request.solve_dir).tasks)
        from goaly.harness.core.runner import Runner

        benchmark = ScriptedBenchmark(self.instances, external=self.instances)
        runner = Runner(benchmark, self.policy, StubContexts(), max_tasks=4)
        await runner.run(
            request.solve_dir,
            RunManifest(run_id=request.run_id, harness_hash="container-says"),
            split=request.split,
            seeds=list(tasks.seeds),
            instance_ids=list(tasks.instance_ids),
            resume=request.resume,
            retry_errors=request.retry_errors,
            grading=False,
            tasks=tasks.tasks,
        )
        if self.after is not None:
            self.after(request)


def host(
    container: Container | None,
    *,
    cache: EvalCache | None = None,
    benchmark: ScriptedBenchmark | None = None,
) -> HostRunner:  # type: ignore[type-arg]
    return HostRunner(
        ScriptedBenchmark(toy_instances(3)) if benchmark is None else benchmark,
        launcher=container,
        cache=cache,
    )


async def solve_and_grade(
    runner: HostRunner,  # type: ignore[type-arg]
    run_dir: Path,
    *,
    seeds: Sequence[int] = (0,),
    resume: bool = False,
    retry_errors: bool = False,
    grading: bool = True,
    split: int = SPLIT_TRAIN,
) -> object:
    async with asyncio.timeout(GUARD_S):
        return await runner.run(
            run_dir,
            manifest(),
            split=split,  # type: ignore[arg-type]
            seeds=list(seeds),
            instance_ids=None,
            solve=solve_options(run_dir.parent),
            resume=resume,
            retry_errors=retry_errors,
            grading=grading,
        )


def results(run_dir: Path) -> dict[Key, TaskResult]:
    return {(r.instance_id, r.seed): r for r in read_jsonl(RunFiles(run_dir).results, TaskResult)}


def trajectories(run_dir: Path) -> dict[Key, Trajectory]:
    found = read_jsonl(RunFiles(run_dir).trajectories, Trajectory)
    return {(t.instance_id, t.seed): t for t in found}


def official(run_dir: Path, seed: int = 0) -> list[dict[str, object]]:
    lines = RunFiles(run_dir).official(seed).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def predictions_file(request: SolveRequest, seed: int = 0) -> Path:
    return RunFiles(request.solve_dir).predictions(seed)


# Solving in the container, grading on the host.


async def test_the_host_grades_every_task_from_its_prediction(tmp_path: Path) -> None:
    container = Container(ScriptedPolicy({"t1": answering("SELECT 1")}))
    run_dir = tmp_path / "run"

    summary = await solve_and_grade(host(container), run_dir)

    found = results(run_dir)
    assert {key: result.grade.score for key, result in found.items()} == {
        ("t0", 0): 0.5,
        ("t1", 0): 1.0,  # Its gold answer.
        ("t2", 0): 0.5,
    }
    assert all(result.run_id == "run-1" for result in found.values())
    (request,) = container.requests
    assert request.tasks == (("t0", 0), ("t1", 0), ("t2", 0))
    assert request.solve_dir == RunFiles(run_dir).solve.root
    host_trajectory = trajectories(run_dir)[("t1", 0)]
    assert host_trajectory.grade.score == 1.0  # The host's own record carries its grade.
    assert host_trajectory.task.effisql.base_sql  # The host's agent input, not the container's.
    assert official(run_dir) == [
        {"instance_id": "t0", "sql": "SELECT 0 FROM slow"},
        {"instance_id": "t1", "sql": "SELECT 1"},
        {"instance_id": "t2", "sql": "SELECT 2 FROM slow"},
    ]
    stored = read_manifest(RunFiles(run_dir).manifest)
    assert stored.harness_hash == HASH  # The host's, never the container's.
    assert stored.HasField("finished_at")
    assert summary == pytest.approx(summary)  # A RunSummary.
    assert (summary.tasks, summary.ran, summary.cached) == (3, 3, 0)  # type: ignore[attr-defined]


async def test_a_forged_grade_in_the_solve_directory_is_never_taken(tmp_path: Path) -> None:
    def forge(request: SolveRequest) -> None:
        for n in range(3):
            forged = TaskResult(
                run_id=request.run_id,
                instance_id=f"t{n}",
                seed=0,
                grade=Grade(instance_id=f"t{n}", score=99.0, status=GRADE_STATUS_GRADED),
            )
            append_jsonl(RunFiles(request.solve_dir).results, forged)
        # A manifest that would shrink the run, and a task file that would skip a task.
        (request.solve_dir / "manifest.json").write_text("{}", encoding="utf-8")

    run_dir = tmp_path / "run"
    await solve_and_grade(host(Container(after=forge)), run_dir)

    assert {result.grade.score for result in results(run_dir).values()} == {0.5}
    assert list(read_manifest(RunFiles(run_dir).manifest).instance_ids) == ["t0", "t1", "t2"]


async def test_a_task_the_container_left_unsolved_gets_credit_0(tmp_path: Path) -> None:
    def drop(request: SolveRequest) -> None:
        path = predictions_file(request)
        kept = [line for line in path.read_text().splitlines() if '"t1"' not in line]
        path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")

    run_dir = tmp_path / "run"
    summary = await solve_and_grade(host(Container(after=drop)), run_dir)

    t1 = results(run_dir)[("t1", 0)]
    assert t1.error_type == TASK_ERROR_TYPE_POLICY_CRASH
    assert t1.error_message == "the solve phase left no prediction for this task"
    assert (t1.grade.score, t1.grade.status) == (0.0, GRADE_STATUS_PREDICTION_FAILED)
    assert summary.errors == {"TASK_ERROR_TYPE_POLICY_CRASH": 1}  # type: ignore[attr-defined]
    # Every task still gets its official record: the empty answer.
    assert official(run_dir)[1] == {"instance_id": "t1", "sql": "SELECT 1 FROM slow"}


@pytest.mark.parametrize(
    ("lines", "problem"),
    [
        (
            [to_json_line(Prediction(instance_id="t1"))] * 2,
            "the solve phase left 2 predictions for this task, not one",
        ),
        (
            ['{"instance_id": "t1", "effisql": {"sql": 5}}'],
            "the solve phase left a prediction for this task that does not parse",
        ),
        (
            ["not json", '{"no": "instance"}', "[1]"],
            "the solve phase left no prediction for this task",
        ),
    ],
    ids=["duplicate", "malformed", "unattributable"],
)
async def test_a_bad_prediction_record_is_an_error(
    tmp_path: Path, lines: list[str], problem: str
) -> None:
    def tamper(request: SolveRequest) -> None:
        path = predictions_file(request)
        kept = [line for line in path.read_text().splitlines() if '"t1"' not in line]
        path.write_text("".join(f"{line}\n" for line in [*kept, *lines]), encoding="utf-8")

    run_dir = tmp_path / "run"
    await solve_and_grade(host(Container(after=tamper)), run_dir)

    found = results(run_dir)
    assert found[("t1", 0)].error_message == problem
    assert found[("t1", 0)].grade.score == 0.0
    assert found[("t0", 0)].grade.score == 0.5  # The others are graded as ever.


async def test_an_overlong_line_is_skipped_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("goaly.harness.core.isolation.MAX_PREDICTION_BYTES", 200)

    def bloat(request: SolveRequest) -> None:
        path = predictions_file(request)
        huge = to_json_line(Prediction(instance_id="t2", effisql=EffiSqlPrediction(sql="x" * 900)))
        kept = [line for line in path.read_text().splitlines() if '"t2"' not in line]
        path.write_text("".join(f"{line}\n" for line in [*kept, huge]), encoding="utf-8")

    run_dir = tmp_path / "run"
    with caplog.at_level(logging.WARNING, logger="goaly.harness.core.isolation"):
        await solve_and_grade(host(Container(after=bloat)), run_dir)

    assert "longer than 200 bytes" in caplog.text
    assert results(run_dir)[("t2", 0)].error_type == TASK_ERROR_TYPE_POLICY_CRASH


async def test_a_container_that_fails_stops_the_run_and_resume_relaunches_it(
    tmp_path: Path,
) -> None:
    container = Container(fails=True)
    run_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="exited with status 1"):
        await solve_and_grade(host(container), run_dir)
    assert not read_manifest(RunFiles(run_dir).manifest).HasField("finished_at")

    container.fails = False
    summary = await solve_and_grade(host(container), run_dir, resume=True)

    assert summary.ran == 3  # type: ignore[attr-defined]
    assert len(container.requests) == 2
    assert len(RunFiles(run_dir).launches.read_text().splitlines()) == 2


async def test_retry_errors_has_the_container_solve_the_errors_again(tmp_path: Path) -> None:
    container = Container(ScriptedPolicy({"t1": raising(RuntimeError("502 after 5 retries"))}))
    run_dir = tmp_path / "run"
    first = await solve_and_grade(host(container), run_dir)
    container.policy = ScriptedPolicy()

    kept = await solve_and_grade(host(container), run_dir, resume=True)
    retried = await solve_and_grade(host(container), run_dir, resume=True, retry_errors=True)

    assert first.errors == {"TASK_ERROR_TYPE_POLICY_CRASH": 1}  # type: ignore[attr-defined]
    assert kept.ran == 0  # type: ignore[attr-defined]
    assert retried.ran == 1  # type: ignore[attr-defined]
    assert container.requests[-1].tasks == (("t1", 0),)
    assert container.requests[-1].retry_errors
    assert container.requests[-1].resume
    assert results(run_dir)[("t1", 0)].error_type == TASK_ERROR_TYPE_NONE


# The cache, on the host alone.


async def test_cache_hits_are_taken_before_the_container_starts(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "cache", "toy-dev")
    await solve_and_grade(host(Container(), cache=cache), tmp_path / "a")
    for entry in (tmp_path / "cache").rglob("*.json"):
        if '"instance_id":"t2"' in entry.read_text():
            entry.unlink()  # One task is not cached.
    container = Container()

    summary = await solve_and_grade(host(container, cache=cache), tmp_path / "b")

    (request,) = container.requests
    assert request.tasks == (("t2", 0),)  # The cached tasks are never sent to the container.
    assert (summary.cached, summary.ran) == (2, 1)  # type: ignore[attr-defined]
    assert {result.run_id for result in results(tmp_path / "b").values()} == {"run-1"}


async def test_a_fully_cached_run_starts_no_container(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "cache", "toy-dev")
    await solve_and_grade(host(Container(), cache=cache), tmp_path / "a")
    container = Container()

    summary = await solve_and_grade(host(container, cache=cache), tmp_path / "b")

    assert container.requests == []
    assert summary.cached == 3  # type: ignore[attr-defined]
    assert official(tmp_path / "b") == official(tmp_path / "a")
    assert not RunFiles(tmp_path / "b").usage.exists()  # No call went through the proxy.


async def test_a_harness_hash_the_cache_cannot_key_is_refused_first(tmp_path: Path) -> None:
    runner = host(Container(), cache=EvalCache(tmp_path / "cache", "toy-dev"))

    with pytest.raises(UsageError, match="cannot key this run"):
        await runner.run(
            tmp_path / "run",
            RunManifest(run_id="r", harness_hash="../escape"),
            split=SPLIT_TRAIN,
            seeds=[0],
            instance_ids=None,
            solve=solve_options(tmp_path),
        )

    assert not (tmp_path / "run").exists()


# Usage, from the proxy's records.


def call(
    run_id: str, key: Key, at: datetime, latency_s: float, tokens: int, cost: float
) -> LlmCallRecord:
    record = LlmCallRecord(
        run_id=run_id,
        instance_id=key[0],
        seed=key[1],
        model="qwen3.6-35b-a3b",
        latency_s=latency_s,
        usage=Usage(input_tokens=tokens, output_tokens=tokens // 2, cost_usd=cost),
    )
    record.started_at.FromDatetime(at)
    return record


async def test_usage_and_wall_time_come_from_the_proxys_records(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def proxy(request: SolveRequest) -> None:
        now = datetime.now(UTC)
        for record in (
            call(request.run_id, ("t0", 0), now, 2.0, 100, 0.10),
            call(request.run_id, ("t0", 0), now + timedelta(seconds=5), 1.0, 60, 0.06),
            call(request.run_id, ("t1", 0), now, 3.0, 40, 0.04),
            call(request.run_id, ("", 0), now, 1.0, 400, 0.40),  # Its tags were bad.
        ):
            append_jsonl(request.llm_calls, record)

    run_dir = tmp_path / "run"
    with caplog.at_level(logging.WARNING, logger="goaly.harness.core.grading"):
        await solve_and_grade(host(Container(after=proxy)), run_dir)

    t0 = results(run_dir)[("t0", 0)]
    assert (t0.usage.input_tokens, t0.usage.output_tokens) == (160, 80)
    assert t0.usage.cost_usd == pytest.approx(0.16)
    assert t0.wall_time_s == pytest.approx(6.0)  # The span of its calls.
    assert trajectories(run_dir)[("t0", 0)].usage == t0.usage
    assert results(run_dir)[("t2", 0)].usage == Usage()  # It made no call.
    report = json.loads(RunFiles(run_dir).usage.read_text())
    assert report["calls"] == {"total": 4, "attributed": 3}
    assert report["proxy_total"]["input_tokens"] == 600
    assert report["unattributed_fraction"] == pytest.approx(2 / 3)
    assert "account for only 33.3% of its usage" in caplog.text


async def test_a_retried_tasks_wall_time_is_its_last_attempts(tmp_path: Path) -> None:
    started = datetime.now(UTC)

    def proxy(request: SolveRequest) -> None:
        later = started + timedelta(hours=1) if len(container.requests) > 1 else started
        append_jsonl(request.llm_calls, call(request.run_id, ("t1", 0), later, 2.0, 10, 0.01))
        if len(container.requests) > 1:  # The launch's start, as the host wrote it, an hour on.
            launches = RunFiles(request.solve_dir.parent).launches
            lines = launches.read_text().splitlines()
            lines[-1] = json.dumps({"started_at": (started + timedelta(minutes=59)).isoformat()})
            launches.write_text("".join(f"{line}\n" for line in lines))

    container = Container(ScriptedPolicy({"t1": raising(RuntimeError("502"))}), after=proxy)
    run_dir = tmp_path / "run"
    await solve_and_grade(host(container), run_dir)
    container.policy = ScriptedPolicy()

    await solve_and_grade(host(container), run_dir, resume=True, retry_errors=True)

    t1 = results(run_dir)[("t1", 0)]
    assert t1.wall_time_s == pytest.approx(2.0)  # Not the hour between the two containers.
    assert t1.usage.input_tokens == 20  # Every attempt's usage counts.


# Grading a solved run later, from the host's manifest.


async def test_a_solve_phase_leaves_the_grading_to_a_later_grade(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    solved = await solve_and_grade(host(Container()), run_dir, grading=False)

    assert not RunFiles(run_dir).results.exists() or results(run_dir) == {}
    assert solved.ran == 0  # type: ignore[attr-defined]
    async with asyncio.timeout(GUARD_S):
        graded = await host(None).grade(run_dir)

    assert graded.ran == 3
    assert {result.grade.score for result in results(run_dir).values()} == {0.5}
    assert read_manifest(RunFiles(run_dir).manifest).HasField("finished_at")


async def test_grading_again_grades_only_grader_errors_and_keeps_finished_at(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    broken = ScriptedBenchmark(toy_instances(3), grade_errors={"t1": ConnectionError("db")})
    await solve_and_grade(host(Container(), benchmark=broken), run_dir)
    assert results(run_dir)[("t1", 0)].grade.status == GRADE_STATUS_GRADER_ERROR
    first = read_manifest(RunFiles(run_dir).manifest).finished_at

    regraded = await host(None).grade(run_dir)
    second = read_manifest(RunFiles(run_dir).manifest).finished_at
    again = await host(None).grade(run_dir)

    assert regraded.ran == 1
    assert results(run_dir)[("t1", 0)].grade.status == GRADE_STATUS_GRADED
    assert second != first  # It recorded a task.
    assert again.ran == 0
    assert read_manifest(RunFiles(run_dir).manifest).finished_at == second  # Nothing new.


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "holds no run to grade"),
        ("benchmark", "is of the benchmark 'toy', not 'other'"),
        ("mode", "another dataset or grading mode than this benchmark, in: environment_settings"),
        ("not isolated", "was not solved in isolation"),
        ("instances", "has no instance 't2'"),
    ],
)
async def test_grade_refuses_a_run_it_cannot_grade(
    tmp_path: Path, change: str, message: str
) -> None:
    run_dir = tmp_path / "run"
    if change != "missing":
        await solve_and_grade(host(Container()), run_dir, grading=False)
    benchmark = ScriptedBenchmark(toy_instances(3))
    if change == "benchmark":
        benchmark.name = "other"
    elif change == "mode":
        benchmark = ScriptedBenchmark(toy_instances(3), grading_mode="official")
    elif change == "not isolated":
        import shutil

        shutil.rmtree(RunFiles(run_dir).solve.root)
    elif change == "instances":
        benchmark = ScriptedBenchmark(toy_instances(2))

    with pytest.raises(RunDirectoryError, match=message):
        await host(None, benchmark=benchmark).grade(run_dir)


async def test_an_organizers_run_is_recorded_and_finished_without_a_grade(tmp_path: Path) -> None:
    from dataclasses import replace

    organizers = [replace(inst, split=SPLIT_EXTERNAL, gold_sql=None) for inst in toy_instances(2)]
    benchmark = ScriptedBenchmark(toy_instances(2), external=organizers)
    container = Container(instances=organizers)
    run_dir = tmp_path / "run"

    tasks_file = tmp_path / "heldout.jsonl"
    tasks_file.write_text("", encoding="utf-8")
    solve = replace(solve_options(tmp_path), instances_file=tasks_file)

    async with asyncio.timeout(GUARD_S):
        await HostRunner(benchmark, launcher=container).run(
            run_dir,
            manifest(),
            split=SPLIT_EXTERNAL,
            seeds=[0],
            instance_ids=None,
            solve=solve,
            grading=False,  # As the organizers' --phase solve: there is no grade phase.
        )

    assert not any(result.HasField("grade") for result in results(run_dir).values())
    assert len(official(run_dir)) == 2
    assert read_manifest(RunFiles(run_dir).manifest).HasField("finished_at")
    with pytest.raises(RunDirectoryError, match="SPLIT_EXTERNAL"):
        await HostRunner(benchmark, launcher=None).grade(run_dir)


async def test_a_grader_sees_which_task_it_grades(tmp_path: Path) -> None:
    seen: list[object] = []

    class Watching(ScriptedBenchmark):
        async def grade(self, inst: ToyInstance, prediction: Prediction) -> Grade:
            seen.append(GRADING_TASK.get())
            return await super().grade(inst, prediction)

    await solve_and_grade(
        host(Container(), benchmark=Watching(toy_instances(3))), tmp_path / "run", seeds=(0, 1)
    )

    assert sorted(seen) == [(f"t{n}", seed) for n in range(3) for seed in (0, 1)]  # type: ignore[type-var]
    assert GRADING_TASK.get() is None  # Outside a grade.


async def test_a_prediction_alone_is_enough_to_grade_a_task(tmp_path: Path) -> None:
    def keep_only_predictions(request: SolveRequest) -> None:
        solve = RunFiles(request.solve_dir)
        solve.results.unlink()
        solve.trajectories.unlink()
        RunFiles(request.solve_dir.parent).launches.unlink()

    run_dir = tmp_path / "run"
    await solve_and_grade(host(Container(after=keep_only_predictions)), run_dir)

    assert {result.grade.score for result in results(run_dir).values()} == {0.5}


async def test_a_seed_without_a_predictions_file_has_every_task_missing(tmp_path: Path) -> None:
    def drop_seed(request: SolveRequest) -> None:
        predictions_file(request, 1).unlink()

    run_dir = tmp_path / "run"
    await solve_and_grade(host(Container(after=drop_seed)), run_dir, seeds=(0, 1))

    found = results(run_dir)
    assert {found[(f"t{n}", 1)].error_type for n in range(3)} == {TASK_ERROR_TYPE_POLICY_CRASH}
    assert {found[(f"t{n}", 0)].error_type for n in range(3)} == {TASK_ERROR_TYPE_NONE}


async def test_a_run_whose_host_records_are_gone_is_not_finished_again(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run_dir = tmp_path / "run"
    await solve_and_grade(host(Container()), run_dir)
    RunFiles(run_dir).trajectories.unlink()
    RunFiles(run_dir).official(0).unlink()

    with caplog.at_level(logging.WARNING, logger="goaly.harness.core.grading"):
        await host(None).grade(run_dir)

    assert "is not finished: the host has not recorded 3 of its tasks" in caplog.text
    assert not RunFiles(run_dir).official(0).exists()


async def test_the_host_takes_only_its_own_lock(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    await solve_and_grade(host(Container()), run_dir, grading=False)
    released = lock_run_dir(RunFiles(run_dir))
    released.release()

    with (
        lock_run_dir(RunFiles(tmp_path / "other")) as other,
        pytest.raises(UsageError, match="the lock given is on"),
    ):
        await host(None).grade(run_dir, lock=other)
    with pytest.raises(UsageError, match="released already"):
        await host(None).grade(run_dir, lock=released)
    with lock_run_dir(RunFiles(run_dir)) as held:
        summary = await host(None).grade(run_dir, lock=held)

    assert summary.ran == 3


async def test_usage_sums_the_reasoning_tokens_and_the_costs_it_has(tmp_path: Path) -> None:
    def proxy(request: SolveRequest) -> None:
        now = datetime.now(UTC)
        thinking = call(request.run_id, ("t0", 0), now, 1.0, 10, 0.0)
        thinking.usage.reasoning_tokens = 4
        thinking.usage.ClearField("cost_usd")  # The gateway sent no cost header.
        append_jsonl(request.llm_calls, thinking)

    run_dir = tmp_path / "run"
    await solve_and_grade(host(Container(after=proxy)), run_dir)

    usage = results(run_dir)[("t0", 0)].usage
    assert usage.reasoning_tokens == 4
    assert not usage.HasField("cost_usd")
    report = json.loads(RunFiles(run_dir).usage.read_text())
    assert report["proxy_total"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "reasoning_tokens": 4,
        "cost_usd": None,
    }
    assert report["unattributed_fraction"] == 0.0


async def test_the_official_files_hold_what_the_host_graded(tmp_path: Path) -> None:
    def disagree(request: SolveRequest) -> None:  # Its trajectories claim another prediction.
        solve = RunFiles(request.solve_dir)
        for trajectory in list(read_jsonl(solve.trajectories, Trajectory)):
            trajectory.prediction.effisql.sql = "SELECT 666"
            append_jsonl(solve.trajectories, trajectory)

    run_dir = tmp_path / "run"
    await solve_and_grade(host(Container(after=disagree)), run_dir)

    assert {line["sql"] for line in official(run_dir)} == {base_sql(n) for n in range(3)}
    assert trajectories(run_dir)[("t0", 0)].prediction == Prediction(
        instance_id="t0", effisql=EffiSqlPrediction(sql=base_sql(0))
    )


async def test_an_organizers_run_keeps_no_grade_the_container_claims(tmp_path: Path) -> None:
    from dataclasses import replace

    organizers = [replace(inst, split=SPLIT_EXTERNAL, gold_sql=None) for inst in toy_instances(1)]

    def forge(request: SolveRequest) -> None:
        solve = RunFiles(request.solve_dir)
        (result,) = read_jsonl(solve.results, TaskResult)
        result.grade.CopyFrom(Grade(instance_id="t0", score=99.0, status=GRADE_STATUS_GRADED))
        append_jsonl(solve.results, result)

    tasks_file = tmp_path / "heldout.jsonl"
    tasks_file.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"
    async with asyncio.timeout(GUARD_S):
        await HostRunner(
            ScriptedBenchmark(toy_instances(1), external=organizers),
            launcher=Container(instances=organizers, after=forge),
        ).run(
            run_dir,
            manifest(),
            split=SPLIT_EXTERNAL,
            seeds=[0],
            instance_ids=None,
            solve=replace(solve_options(tmp_path), instances_file=tasks_file),
            grading=False,
        )

    assert not results(run_dir)[("t0", 0)].HasField("grade")
    assert not trajectories(run_dir)[("t0", 0)].HasField("grade")
