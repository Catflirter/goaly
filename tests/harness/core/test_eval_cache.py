"""Tests for goaly.harness.core.eval_cache: the evaluation cache (spec §6.1, §6.2)."""

import asyncio
import dataclasses
import hashlib
import json
import logging
import threading
from pathlib import Path

import pytest

from goaly.contracts.gen.llm_call_pb2 import Usage
from goaly.contracts.gen.result_pb2 import TASK_ERROR_TYPE_NONE, TaskResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    EffiSqlPrediction,
    EffiSqlTaskInput,
    Grade,
    Prediction,
    TaskInput,
)
from goaly.contracts.gen.trajectory_pb2 import STOP_REASON_SUBMITTED, Step, Trajectory
from goaly.contracts.jsonl import to_json_line
from goaly.harness.core.eval_cache import CachedEvaluation, CacheKey, EvalCache, task_digest

pytestmark = pytest.mark.anyio

TASK = TaskInput(effisql=EffiSqlTaskInput(db="tpch", base_sql="SELECT 1"))
KEY = CacheKey(
    harness_hash="0123abcd" * 8,
    benchmark="effisql",
    instance_id="17",
    seed=1,
    task_digest=task_digest(TASK),
)


def evaluation(key: CacheKey = KEY, *, sql: str = "SELECT 2") -> CachedEvaluation:
    grade = Grade(instance_id=key.instance_id, score=0.75, correct=True, status=GRADE_STATUS_GRADED)
    usage = Usage(input_tokens=1200, output_tokens=300, cost_usd=0.002)
    result = TaskResult(
        run_id="run-a",
        instance_id=key.instance_id,
        seed=key.seed,
        grade=grade,
        wall_time_s=12.5,
        usage=usage,
        step_count=2,
        stop_reason=STOP_REASON_SUBMITTED,
        error_type=TASK_ERROR_TYPE_NONE,
    )
    trajectory = Trajectory(
        run_id="run-a",
        instance_id=key.instance_id,
        seed=key.seed,
        task=TASK,
        steps=[Step(index=0, usage=usage), Step(index=1)],
        prediction=Prediction(instance_id=key.instance_id, effisql=EffiSqlPrediction(sql=sql)),
        grade=grade,
        stop_reason=STOP_REASON_SUBMITTED,
        usage=usage,
    )
    return CachedEvaluation(result=result, trajectory=trajectory)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """The cache's directory: tmp_path also holds the isolated HOME of conftest.py."""
    return tmp_path / "cache"


@pytest.fixture
def cache(root: Path) -> EvalCache:
    return EvalCache(root)


def every_file(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


# Hits and misses.


async def test_a_new_cache_misses_quietly(
    cache: EvalCache, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        assert await cache.get(KEY) is None

    assert caplog.records == []


async def test_returns_what_was_put(cache: EvalCache) -> None:
    stored = evaluation()

    await cache.put(KEY, stored)
    found = await cache.get(KEY)

    assert found == stored
    assert found is not None
    assert found.result is not stored.result  # Every hit is the caller's own copy.


async def test_put_copies_the_evaluation_before_it_waits(
    cache: EvalCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = threading.Event(), threading.Event()
    write = EvalCache._write

    def held_write(path: Path, key: CacheKey, stored: CachedEvaluation) -> None:
        started.set()
        release.wait(timeout=10)  # Until the caller has changed its messages.
        write(path, key, stored)

    monkeypatch.setattr(EvalCache, "_write", staticmethod(held_write))
    stored = evaluation()
    putting = asyncio.create_task(cache.put(KEY, stored))
    assert await asyncio.to_thread(started.wait, 10)

    stored.result.wall_time_s = 99.0
    stored.trajectory.steps.add(index=2)
    release.set()
    await putting

    assert await cache.get(KEY) == evaluation()


@pytest.mark.parametrize(
    "change",
    [
        {"harness_hash": "f" * 64},
        {"benchmark": "swefficiency"},
        {"instance_id": "18"},
        {"seed": 2},
        {"task_digest": task_digest(TaskInput(effisql=EffiSqlTaskInput(base_sql="SELECT 3")))},
    ],
)
async def test_every_part_of_the_key_tells_entries_apart(
    cache: EvalCache, change: dict[str, str | int]
) -> None:
    other = dataclasses.replace(KEY, **change)  # type: ignore[arg-type]  # Fields vary by case.
    await cache.put(KEY, evaluation())

    assert await cache.get(other) is None
    await cache.put(other, evaluation(other, sql="SELECT 3"))
    assert await cache.get(KEY) == evaluation()
    assert await cache.get(other) == evaluation(other, sql="SELECT 3")


async def test_a_later_put_replaces_the_entry(cache: EvalCache, root: Path) -> None:
    await cache.put(KEY, evaluation(sql="SELECT 2"))
    await cache.put(KEY, evaluation(sql="SELECT 3"))

    assert await cache.get(KEY) == evaluation(sql="SELECT 3")
    assert every_file(root) == [cache.path(KEY)]  # No temporary file is left behind.


async def test_concurrent_puts_of_one_key_leave_one_whole_entry(
    cache: EvalCache, root: Path
) -> None:
    candidates = [evaluation(sql=f"SELECT {n}") for n in range(20)]

    async with asyncio.TaskGroup() as group:
        for candidate in candidates:
            group.create_task(cache.put(KEY, candidate))

    assert await cache.get(KEY) in candidates
    assert every_file(root) == [cache.path(KEY)]


# The entry on disk.


async def test_an_entry_is_one_json_file_under_its_harness_hash(
    cache: EvalCache, root: Path
) -> None:
    stored = evaluation()

    await cache.put(KEY, stored)

    path = cache.path(KEY)
    assert path.parent == root / KEY.harness_hash
    assert path.suffix == ".json"
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.count("\n") == 1
    assert json.loads(text) == {
        "format": 1,
        "key": dataclasses.asdict(KEY),
        "result": json.loads(to_json_line(stored.result)),
        "trajectory": json.loads(to_json_line(stored.trajectory)),
    }


def test_entries_of_one_harness_have_distinct_file_names(cache: EvalCache) -> None:
    keys = [
        KEY,
        dataclasses.replace(KEY, benchmark="swefficiency"),
        dataclasses.replace(KEY, instance_id="1"),
        dataclasses.replace(KEY, instance_id="17", seed=11),
        dataclasses.replace(KEY, instance_id="171", seed=1),  # "17"+"11" vs "171"+"1".
        dataclasses.replace(KEY, task_digest="0" * 64),
        dataclasses.replace(KEY, instance_id="../../etc/passwd"),  # Any ID is a safe name.
    ]

    paths = [cache.path(key) for key in keys]

    assert len(set(paths)) == len(keys)
    assert all(path.parent == cache.path(KEY).parent for path in paths)


async def test_unknown_fields_are_ignored(cache: EvalCache) -> None:
    await cache.put(KEY, evaluation())
    path = cache.path(KEY)
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["added_later"] = True
    entry["result"]["field_added_later"] = 1
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    assert await cache.get(KEY) == evaluation()


# Entries that cannot be used are misses.


def corrupt_entries() -> dict[str, bytes]:
    whole = {
        "format": 1,
        "key": dataclasses.asdict(KEY),
        "result": json.loads(to_json_line(evaluation().result)),
        "trajectory": json.loads(to_json_line(evaluation().trajectory)),
    }

    def changed(**fields: object) -> bytes:
        return json.dumps({**whole, **fields}).encode()

    other_key = dataclasses.asdict(dataclasses.replace(KEY, seed=2))
    other_result = json.loads(to_json_line(evaluation(dataclasses.replace(KEY, seed=2)).result))
    return {
        "empty": b"",
        "not JSON": b"{not json",
        "not UTF-8": b'{"format": 1, "key": "\xff"}',
        "not an object": b"[1, 2]",
        "another format": changed(format=2),
        "no format": json.dumps({k: v for k, v in whole.items() if k != "format"}).encode(),
        "another key": changed(key=other_key),
        "no result": json.dumps({k: v for k, v in whole.items() if k != "result"}).encode(),
        "result not an object": changed(result=[1]),
        "result of the wrong type": changed(result={"seed": "not a number"}),
        "result of another seed": changed(result=other_result),
        "trajectory of another task": changed(trajectory={"instance_id": "18", "seed": 1}),
    }


@pytest.mark.parametrize("content", corrupt_entries().values(), ids=corrupt_entries().keys())
async def test_an_entry_that_cannot_be_used_is_a_miss_until_replaced(
    cache: EvalCache, content: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    path = cache.path(KEY)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    with caplog.at_level(logging.WARNING, logger="goaly.harness.core.eval_cache"):
        assert await cache.get(KEY) is None

    assert str(path) in caplog.text
    await cache.put(KEY, evaluation())
    assert await cache.get(KEY) == evaluation()


async def test_an_unreadable_entry_is_a_miss(
    cache: EvalCache, caplog: pytest.LogCaptureFixture
) -> None:
    cache.path(KEY).mkdir(parents=True)  # Reading a directory fails with an OSError.

    with caplog.at_level(logging.WARNING, logger="goaly.harness.core.eval_cache"):
        assert await cache.get(KEY) is None

    assert "IsADirectoryError" in caplog.text


# Keys and evaluations the cache refuses.


@pytest.mark.parametrize("harness_hash", ["", ".", "..", "../up", "a/b", "a b", "a\x00b"])
async def test_refuses_a_harness_hash_that_is_not_a_plain_name(
    cache: EvalCache, harness_hash: str
) -> None:
    key = dataclasses.replace(KEY, harness_hash=harness_hash)

    with pytest.raises(ValueError, match="harness hash"):
        await cache.get(key)
    with pytest.raises(ValueError, match="harness hash"):
        await cache.put(key, evaluation(key))


@pytest.mark.parametrize(
    "mismatch",
    [
        {"result_instance": "18"},
        {"result_seed": 2},
        {"trajectory_instance": "18"},
        {"trajectory_seed": 2},
    ],
)
async def test_refuses_an_evaluation_of_another_task_or_seed(
    cache: EvalCache, root: Path, mismatch: dict[str, str | int]
) -> None:
    stored = evaluation()
    for field, value in mismatch.items():
        message = stored.result if field.startswith("result") else stored.trajectory
        if field.endswith("instance"):
            message.instance_id = str(value)
        else:
            message.seed = int(value)

    with pytest.raises(ValueError, match="does not match"):
        await cache.put(KEY, stored)
    assert every_file(root) == []


# Task digests.


def test_a_task_digest_is_the_sha256_of_the_task_as_a_json_line() -> None:
    expected = hashlib.sha256(to_json_line(TASK).encode("utf-8")).hexdigest()

    assert task_digest(TASK) == expected


def test_a_task_digest_changes_with_anything_the_agent_sees() -> None:
    changed = TaskInput()
    changed.CopyFrom(TASK)
    changed.effisql.schema = "CREATE TABLE t (id int)"

    assert task_digest(changed) != task_digest(TASK)
    assert task_digest(TaskInput()) != task_digest(TASK)
