"""The single writer of a JSONL file (spec §6.2).

Each per-run JSONL file, such as a run's results, trajectories or LLM call log, gets one
JsonlWriter. Any task on the event loop hands it messages, and the writer's one task appends them
to the file in order, through goaly.contracts.jsonl.append_jsonl in a worker thread. So each line
is flushed as it is written, a crash loses at most the line being written, and the next write
cuts off the torn tail that such a crash leaves.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from google.protobuf.message import Message

from goaly.contracts.jsonl import append_jsonl

__all__ = ["JsonlWriter", "JsonlWriterError"]

_logger = logging.getLogger(__name__)


class JsonlWriterError(Exception):
    """A JsonlWriter could not write a line, or stopped before it wrote one."""


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
    they were queued. Leaving writes every line still queued, then stops the task.

    The first line that cannot be written, say because the disk is full, fails the writer. It
    writes nothing more: the lines queued behind it are dropped, and from then on put(), write()
    and leaving the context raise JsonlWriterError. Leaving does not raise it over an exception
    that is already propagating, which it would hide.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._queue: asyncio.Queue[_Line | None] = asyncio.Queue()  # None stops the writer task.
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._failure: BaseException | None = None  # Why a line could not be written.

    async def __aenter__(self) -> Self:
        if self._task is not None or self._closed:
            raise RuntimeError(f"the writer of {self.path} was already entered")
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
        """Queue a copy of ``message`` and return at once; the writer task writes it later."""
        self._queue_line(message, None)

    async def write(self, message: M) -> None:
        """Queue a copy of ``message`` and wait until its line is written and flushed.

        A caller cancelled while it waits does not take the line back: it is still written.
        """
        written = asyncio.get_running_loop().create_future()
        self._queue_line(message, written)
        await written

    async def aclose(self) -> None:
        """Write every line still queued, then stop the writer task. Calling it again does nothing.

        If it is cancelled, the writer stops at once and the lines still queued are not written.
        """
        if not self._closed:
            self._closed = True
            if self._task is not None:
                self._queue.put_nowait(None)
        if self._task is not None:
            await self._task
        if self._failure is not None:
            raise self._error()

    def _queue_line(self, message: M, written: asyncio.Future[None] | None) -> None:
        if self._failure is not None:
            raise self._error()
        if self._task is None or self._closed:
            raise RuntimeError(f"the writer of {self.path} is not open")
        copy = type(message)()
        copy.CopyFrom(message)
        self._queue.put_nowait(_Line(copy, written))

    async def _write_lines(self) -> None:
        line: _Line | None = None
        try:
            while (line := await self._queue.get()) is not None:
                if self._failure is None:
                    try:
                        await asyncio.to_thread(append_jsonl, self.path, line.message)
                    except Exception as error:
                        _logger.exception("cannot write %s; it gets no more lines", self.path)
                        self._failure = error
                self._settle(line, None if self._failure is None else self._error())
        finally:
            # Cancelled: whatever is left will not be written, and nobody may wait for it forever.
            stranded: list[_Line | None] = [line]
            while not self._queue.empty():
                stranded.append(self._queue.get_nowait())
            for item in stranded:
                if item is not None:
                    stopped = f"the writer of {self.path} stopped before it wrote this line"
                    self._settle(item, JsonlWriterError(stopped))

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
