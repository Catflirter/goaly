"""A lane's hypothesis queue (spec §8.2, §8.7): hypotheses waiting for a later round.

A round builds its top hypotheses. The others, those outranked at the decision and those whose
experiment was interrupted before its commit, go back in the queue, which T21 decides. The
analyst sees the queue, so it does not propose the same idea again.

The queue is a FIFO of Hypothesis contracts, one JSONL file per lane in the ledger's directory,
``hypothesis_queue-<lane>.jsonl``. Every change takes the queue's exclusive file lock and replaces
the whole file durably: the new content goes to a temporary file that is synced, renamed over the
old one, and the directory synced too. Readers, which take the shared lock, see the old queue or
the new one, never part of one, and a crash leaves one of the two.
"""

import os
import secrets
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import (
    CHANGE_KIND_UNSPECIFIED,
    HYPOTHESIS_SOURCE_UNSPECIFIED,
    Hypothesis,
    Lane,
)
from goaly.contracts.jsonl import read_jsonl, to_json_line
from goaly.rsi.ids import check_id
from goaly.rsi.lanes import lane_info
from goaly.rsi.locks import file_lock

__all__ = ["HypothesisQueue"]


class HypothesisQueue:
    """The hypothesis queue of ``lane``, kept in ``directory``, which a write creates."""

    def __init__(self, directory: Path, lane: Lane.ValueType, *, lock_timeout_s: float = 60.0):
        name = lane_info(lane).name
        self.directory = directory
        self.path = directory / f"hypothesis_queue-{name}.jsonl"
        self._lock = directory / f".hypothesis_queue-{name}.lock"
        self._lock_timeout_s = lock_timeout_s

    def pending(self) -> list[Hypothesis]:
        """Return the queued hypotheses, oldest first."""
        if not self.directory.exists():
            return []
        with file_lock(self._lock, shared=True, timeout_s=self._lock_timeout_s):
            return self._read()

    def push(self, hypotheses: Iterable[Hypothesis]) -> None:
        """Queue ``hypotheses`` at the back, in order. One already queued keeps its place.

        Raises ValueError, changing nothing, if a hypothesis has an invalid ID, or no change
        kind or source.
        """
        new = list(hypotheses)
        for hypothesis in new:
            _check(hypothesis)
        with self._writing() as queue:
            queued = {hypothesis.hypothesis_id for hypothesis in queue}
            for hypothesis in new:
                if hypothesis.hypothesis_id not in queued:
                    queue.append(hypothesis)
                    queued.add(hypothesis.hypothesis_id)

    def remove(self, hypothesis_ids: Iterable[str]) -> list[Hypothesis]:
        """Take the hypotheses with these IDs out of the queue and return them, oldest first.

        An ID that is not queued is ignored.
        """
        wanted = set(hypothesis_ids)
        with self._writing() as queue:
            taken = [hypothesis for hypothesis in queue if hypothesis.hypothesis_id in wanted]
            queue[:] = [
                hypothesis for hypothesis in queue if hypothesis.hypothesis_id not in wanted
            ]
        return taken

    def _read(self) -> list[Hypothesis]:
        return list(read_jsonl(self.path, Hypothesis)) if self.path.exists() else []

    @contextmanager
    def _writing(self) -> Iterator[list[Hypothesis]]:
        """Lock the queue and yield it as a list; the list as the block leaves it is written."""
        _make_directory(self.directory)
        with file_lock(self._lock, timeout_s=self._lock_timeout_s):
            queue = self._read()
            before = list(queue)
            yield queue
            if queue != before:
                self._write(queue)

    def _write(self, queue: list[Hypothesis]) -> None:
        text = "".join(to_json_line(hypothesis) + "\n" for hypothesis in queue)
        temporary = self.path.with_name(f".{self.path.name}.{secrets.token_hex(8)}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as file:
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(self.path)
            _fsync(self.directory)
        finally:
            temporary.unlink(missing_ok=True)  # Already gone once it has replaced the queue.


def _check(hypothesis: Hypothesis) -> None:
    check_id(hypothesis.hypothesis_id, "hypothesis_id")
    if hypothesis.change_kind == CHANGE_KIND_UNSPECIFIED:
        raise ValueError(f"hypothesis {hypothesis.hypothesis_id} has no change kind")
    if hypothesis.source == HYPOTHESIS_SOURCE_UNSPECIFIED:
        raise ValueError(f"hypothesis {hypothesis.hypothesis_id} has no source")


def _make_directory(directory: Path) -> None:
    if directory.exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    _fsync(directory.parent)


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
