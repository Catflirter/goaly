"""The single writer of a JSONL file (spec §6.2).

Each per-run JSONL file, such as a run's results, trajectories or LLM call log, gets one
JsonlWriter. Any task on the event loop hands it messages, and the writer's one task appends them
to the file in order, through goaly.contracts.jsonl.append_jsonl in a worker thread. So each line
is flushed to the operating system as it is written: a crash of the process loses at most the
line being written, and the next write cuts off the torn tail that such a crash leaves. Lines
are not synced to disk, so a power cut can lose the last ones the system had not yet written.

A message that cannot be a JSON line fails its own caller, at once; it never reaches the shared
writer, whose other callers carry on.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from google.protobuf.message import Message

from goaly.contracts.jsonl import append_jsonl, to_json_line

__all__ = ["JsonlWriter", "JsonlWriterError"]

_logger = logging.getLogger(__name__)


class JsonlWriterError(RuntimeError):
    """A JsonlWriter cannot record a line: it failed, stopped before the line was written, or
    is not open. A caller that must never raise, such as a call recorder, catches this alone."""


@dataclass(frozen=True, eq=False)
class _Line:
    message: Message
    # Resolved once the line is written, for write(); None for put().
    written: asyncio.Future[None] | None


class JsonlWriter[M: Message]:
    """The one writer of the JSONL file at ``path``, which holds ``M`` messages.

    Use it as an async context manager, on one event loop. Entering creates the file if it does
    not exist, and starts the writer task. :meth:`put` and :meth:`write` queue a copy of a
    message, so the caller may change the message afterwards, and lines are written in the order
    they were queued. The queue is unbounded: put() never waits. Leaving the context writes every
    line still queued, even when the body was cancelled, then stops the task.

    A message that cannot be a JSON line, such as one with an invalid Timestamp, makes put() or
    write() raise what to_json_line raises, for that caller only. A line the file system cannot
    take (an OSError, say because the disk is full) fails the writer instead. It writes nothing
    more: the lines queued behind it are dropped, and from then on put(), write() and leaving the
    context raise JsonlWriterError. Leaving does not raise it over an exception that is already
    propagating, which it would hide. put() and write() also raise JsonlWriterError before the
    writer is entered and once it is closed.

    A writer stopped by cancellation stops waiting at once, but a thread cannot be stopped: an
    append already running in its worker thread still runs to its end, so its line may reach the
    file after aclose() has returned.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._queue: asyncio.Queue[_Line | None] = asyncio.Queue()  # None stops the writer task.
        self._task: asyncio.Task[None] | None = None
        self._entered = False
        self._closed = False
        self._joined = False  # Whether aclose() has awaited the writer task.
        self._failure: BaseException | None = None  # Why a line could not be written.

    async def __aenter__(self) -> Self:
        if self._entered:
            raise JsonlWriterError(f"the writer of {self.path} was already entered")
        if self._closed:
            raise JsonlWriterError(f"the writer of {self.path} is closed")
        self._entered = True  # Before the await: a second, concurrent entry is refused.
        # Creating the file now fails a path that cannot be written before anything is queued.
        await asyncio.to_thread(_create, self.path)
        self._task = asyncio.create_task(self._write_lines(), name=f"JsonlWriter({self.path})")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await self.aclose()
        except JsonlWriterError:
            if exc is None:
                raise
            # The body's exception goes on; the writer's failure was logged when it happened.

    def put(self, message: M) -> None:
        """Queue a copy of ``message`` and return at once; the writer task writes it later.

        Raise what to_json_line raises if the message cannot be a JSON line.
        """
        self._queue_line(message, None)

    async def write(self, message: M) -> None:
        """Queue a copy of ``message`` and wait until its line is written and flushed.

        Raise what to_json_line raises if the message cannot be a JSON line. A caller cancelled
        while it waits does not take the line back: it is still written.
        """
        written = asyncio.get_running_loop().create_future()
        self._queue_line(message, written)
        await written

    async def aclose(self) -> None:
        """Write every line still queued, then stop the writer task.

        Calling it again, even at the same time, waits until those lines are written, and once
        they are it does nothing. If the first call is cancelled, the writer stops at once and
        the lines still queued are not written; a later call then does not wait for them.
        Cancelling a later call stops only that call's wait.
        """
        self._closed = True
        if self._task is None:
            return  # Never entered.
        if not self._joined:
            self._joined = True
            self._queue.put_nowait(None)
            await self._task
        elif not self._task.done():
            await asyncio.wait([self._task])  # Neither raises nor passes on a cancellation.
        if self._failure is not None:
            raise self._error()

    def _queue_line(self, message: M, written: asyncio.Future[None] | None) -> None:
        if self._failure is not None:
            raise self._error()
        if self._task is None:
            raise JsonlWriterError(f"the writer of {self.path} has not been entered")
        if self._closed:
            raise JsonlWriterError(f"the writer of {self.path} is closed")
        copy = type(message)()
        copy.CopyFrom(message)
        to_json_line(copy)  # Raises here, for this caller alone, if it cannot be a JSON line.
        self._queue.put_nowait(_Line(copy, written))

    async def _write_lines(self) -> None:
        in_flight: _Line | None = None  # Taken from the queue, and not yet settled.
        try:
            while (line := await self._queue.get()) is not None:
                in_flight = line
                if self._failure is None:
                    await self._write_line(line)
                else:
                    self._settle(line, self._error())
                in_flight = None
        finally:
            # Cancelled: whatever is left will not be written, and nobody may wait for it forever.
            if in_flight is not None:
                unsure = f"the writer of {self.path} stopped while it wrote this line, which"
                self._settle(in_flight, JsonlWriterError(f"{unsure} may not have been written"))
            dropped = 0 if in_flight is None else 1
            while not self._queue.empty():
                if (queued := self._queue.get_nowait()) is not None:
                    dropped += 1
                    stopped = f"the writer of {self.path} stopped before it wrote this line"
                    self._settle(queued, JsonlWriterError(stopped))
            if dropped:
                _logger.warning(
                    "the writer of %s stopped with %d lines unwritten", self.path, dropped
                )

    async def _write_line(self, line: _Line) -> None:
        try:
            await asyncio.to_thread(append_jsonl, self.path, line.message)
        except OSError as error:  # The file system's failure fails the writer.
            _logger.exception("cannot write %s; it gets no more lines", self.path)
            self._failure = error
            self._settle(line, self._error())
        except Exception as error:  # Anything else fails this line alone, as to_json_line would.
            _logger.exception("cannot write a line to %s", self.path)
            failed = JsonlWriterError(f"cannot write this line to {self.path}: {error}")
            failed.__cause__ = error
            self._settle(line, failed)
        else:
            self._settle(line, None)

    @staticmethod
    def _settle(line: _Line, error: JsonlWriterError | None) -> None:
        # A future that is done already was cancelled with its caller, who no longer waits.
        if line.written is None or line.written.done():
            return
        if error is None:
            line.written.set_result(None)
        else:
            line.written.set_exception(error)

    def _error(self) -> JsonlWriterError:
        """A new error for the writer's failure, so that no traceback grows with each raise."""
        error = JsonlWriterError(f"cannot write {self.path}: {self._failure}")
        error.__cause__ = self._failure
        return error


def _create(path: Path) -> None:
    """Create the file at ``path`` if it does not exist, leaving any content as it is."""
    with path.open("ab"):
        pass
