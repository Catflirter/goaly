"""Tests for goaly.harness.core.jsonl_writer: the single-writer JSONL sink (spec §6.2)."""

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from google.protobuf.message import Message

from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import Grade
from goaly.contracts.jsonl import append_jsonl, read_jsonl, to_json_line
from goaly.harness.core import jsonl_writer
from goaly.harness.core.jsonl_writer import JsonlWriter, JsonlWriterError

pytestmark = pytest.mark.anyio


def result(n: int) -> TaskResult:
    return TaskResult(run_id="run", instance_id=f"task-{n}", seed=n % 3)


def instance_ids(path: Path) -> list[str]:
    return [line.instance_id for line in read_jsonl(path, TaskResult)]


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "results.jsonl"


class BlockingAppend:
    """Stands in for append_jsonl: the first call blocks until ``release`` is set."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, path: Path, message: Message) -> None:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            self.release.wait(timeout=10)
        append_jsonl(path, message)


@pytest.fixture
def blocking_append(monkeypatch: pytest.MonkeyPatch) -> Iterator[BlockingAppend]:
    append = BlockingAppend()
    monkeypatch.setattr(jsonl_writer, "append_jsonl", append)
    yield append
    append.release.set()  # Never leave a worker thread blocked.


async def wait_for(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 10)


# Guards the tests that would hang if a waiter were never told how its line went.
GUARD_S = 10


# Writing.


async def test_writes_each_line_in_order_and_flushes_it(path: Path) -> None:
    async with JsonlWriter[TaskResult](path) as writer:
        await writer.write(result(1))
        # The line is on disk while the writer is still open: it was flushed as it was written.
        assert instance_ids(path) == ["task-1"]
        await writer.write(result(2))
        assert instance_ids(path) == ["task-1", "task-2"]


async def test_put_returns_at_once_and_closing_writes_every_queued_line(path: Path) -> None:
    async with JsonlWriter[TaskResult](path) as writer:
        for n in range(50):
            writer.put(result(n))

    assert instance_ids(path) == [f"task-{n}" for n in range(50)]


async def test_lines_are_written_in_the_order_they_were_queued(path: Path) -> None:
    async with JsonlWriter[TaskResult](path) as writer:
        writer.put(result(1))
        second = asyncio.create_task(writer.write(result(2)))
        await asyncio.sleep(0)  # write() has queued its line.
        writer.put(result(3))
        await second

    assert instance_ids(path) == ["task-1", "task-2", "task-3"]


async def test_lines_are_whole_json_lines_of_the_contract_format(path: Path) -> None:
    message = TaskResult(run_id="run", instance_id="naïve\n日本語", error_message="a\u2028b")

    async with JsonlWriter[TaskResult](path) as writer:
        await writer.write(message)

    assert path.read_text(encoding="utf-8") == to_json_line(message) + "\n"


async def test_many_concurrent_writers_each_get_exactly_one_line(path: Path) -> None:
    async with JsonlWriter[TaskResult](path) as writer, asyncio.TaskGroup() as group:
        for n in range(200):
            group.create_task(writer.write(result(n)))

    assert sorted(instance_ids(path)) == sorted(f"task-{n}" for n in range(200))


async def test_a_single_task_does_all_the_writing(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = threading.Lock()
    in_flight = 0
    most_in_flight = 0

    def counting_append(path: Path, message: Message) -> None:
        nonlocal in_flight, most_in_flight
        with lock:
            in_flight += 1
            most_in_flight = max(most_in_flight, in_flight)
        try:
            append_jsonl(path, message)
        finally:
            with lock:
                in_flight -= 1

    monkeypatch.setattr(jsonl_writer, "append_jsonl", counting_append)

    async with JsonlWriter[TaskResult](path) as writer, asyncio.TaskGroup() as group:
        for n in range(100):
            group.create_task(writer.write(result(n)))
            writer.put(result(1000 + n))

    assert most_in_flight == 1
    assert len(instance_ids(path)) == 200


async def test_queues_a_copy_so_the_caller_may_change_its_message(path: Path) -> None:
    message = result(1)

    async with JsonlWriter[TaskResult](path) as writer:
        writer.put(message)
        message.instance_id = "changed"

    assert instance_ids(path) == ["task-1"]


async def test_appends_to_an_existing_file_and_cuts_a_torn_tail(path: Path) -> None:
    append_jsonl(path, result(1))
    with path.open("ab") as file:
        file.write(b'{"run_id":"run","instance')  # A line cut short by a crash.

    async with JsonlWriter[TaskResult](path) as writer:
        await writer.write(result(2))

    assert instance_ids(path) == ["task-1", "task-2"]


async def test_entering_creates_the_file(path: Path) -> None:
    async with JsonlWriter[TaskResult](path):
        assert path.read_bytes() == b""


async def test_entering_fails_at_once_if_the_file_cannot_be_created(tmp_path: Path) -> None:
    writer = JsonlWriter[TaskResult](tmp_path / "missing" / "results.jsonl")

    with pytest.raises(FileNotFoundError):
        await writer.__aenter__()


# Cancellation.


async def test_a_cancelled_write_still_writes_its_line(
    path: Path, blocking_append: BlockingAppend
) -> None:
    async with JsonlWriter[TaskResult](path) as writer:
        writer.put(result(1))  # Holds the writer task until it is released.
        await wait_for(blocking_append.started)
        waiting = asyncio.create_task(writer.write(result(2)))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        blocking_append.release.set()

    assert instance_ids(path) == ["task-1", "task-2"]


async def test_cancelling_the_close_stops_the_writer_and_fails_its_waiters(
    path: Path, blocking_append: BlockingAppend
) -> None:
    writer = JsonlWriter[TaskResult](path)
    await writer.__aenter__()
    first = asyncio.create_task(writer.write(result(1)))
    await wait_for(blocking_append.started)
    second = asyncio.create_task(writer.write(result(2)))
    await asyncio.sleep(0)
    closing = asyncio.create_task(writer.aclose())
    await asyncio.sleep(0)

    closing.cancel()

    async with asyncio.timeout(GUARD_S):
        with pytest.raises(asyncio.CancelledError):
            await closing
        # Nobody waits forever: each waiter learns that its line may not have been written.
        with pytest.raises(JsonlWriterError, match="stopped before"):
            await first
        with pytest.raises(JsonlWriterError, match="stopped before"):
            await second
    blocking_append.release.set()


# Failures.


async def test_a_failed_line_fails_the_writer(path: Path) -> None:
    writer = JsonlWriter[TaskResult](path)
    await writer.__aenter__()
    await writer.write(result(1))
    path.unlink()
    path.mkdir()  # Every later append fails, as on a full disk.

    with pytest.raises(JsonlWriterError, match="cannot write") as failed:
        await writer.write(result(2))
    assert isinstance(failed.value.__cause__, IsADirectoryError)
    path.rmdir()  # The file could be written again, but the writer stays failed.
    with pytest.raises(JsonlWriterError, match="cannot write"):
        await writer.write(result(3))
    with pytest.raises(JsonlWriterError, match="cannot write"):
        writer.put(result(4))
    with pytest.raises(JsonlWriterError, match="cannot write"):
        await writer.aclose()
    assert not path.exists()


async def test_lines_queued_behind_a_failed_line_are_dropped(
    path: Path, blocking_append: BlockingAppend
) -> None:
    writer = JsonlWriter[TaskResult](path)
    await writer.__aenter__()
    first = asyncio.create_task(writer.write(result(1)))
    await wait_for(blocking_append.started)
    second = asyncio.create_task(writer.write(result(2)))
    writer.put(result(3))
    await asyncio.sleep(0)
    path.unlink()
    path.mkdir()
    blocking_append.release.set()

    async with asyncio.timeout(GUARD_S):
        for waiter in (first, second):
            with pytest.raises(JsonlWriterError, match="cannot write"):
                await waiter
    path.rmdir()
    with pytest.raises(JsonlWriterError, match="cannot write"):
        await writer.aclose()
    assert blocking_append.calls == 1  # Nothing was attempted after the failure.


async def put_a_line_that_fails(path: Path, *, then_raise: Exception | None = None) -> None:
    async with JsonlWriter[TaskResult](path) as writer:
        path.unlink()
        path.mkdir()
        writer.put(result(1))
        if then_raise is not None:
            raise then_raise


async def test_a_put_that_fails_is_reported_on_close(path: Path) -> None:
    with pytest.raises(JsonlWriterError, match="cannot write"):
        await put_a_line_that_fails(path)


async def test_closing_after_an_error_in_the_body_keeps_that_error(path: Path) -> None:
    with pytest.raises(ValueError, match="the body's own error"):
        await put_a_line_that_fails(path, then_raise=ValueError("the body's own error"))


# Lifecycle.


async def test_refuses_messages_before_it_is_entered(path: Path) -> None:
    writer = JsonlWriter[TaskResult](path)

    with pytest.raises(RuntimeError, match="not open"):
        writer.put(result(1))
    with pytest.raises(RuntimeError, match="not open"):
        await writer.write(result(1))


async def test_refuses_messages_once_closed_and_closing_again_does_nothing(path: Path) -> None:
    async with JsonlWriter[TaskResult](path) as writer:
        writer.put(result(1))
    await writer.aclose()

    with pytest.raises(RuntimeError, match="not open"):
        writer.put(result(2))
    with pytest.raises(RuntimeError, match="not open"):
        await writer.write(result(2))
    assert instance_ids(path) == ["task-1"]


async def test_cannot_be_entered_twice(path: Path) -> None:
    writer = JsonlWriter[TaskResult](path)
    async with writer:
        with pytest.raises(RuntimeError, match="already"):
            await writer.__aenter__()
    with pytest.raises(RuntimeError, match="already"):
        await writer.__aenter__()


def _typing_checks(writer: JsonlWriter[TaskResult]) -> None:
    """Never called: ``make typecheck`` fails if an ignore below is not needed."""
    writer.put(Grade())  # type: ignore[arg-type]  # A writer takes only its own message type.
