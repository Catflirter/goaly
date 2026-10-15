"""The RSI ledger (spec §8.6): the lanes' shared memory of experiments, lessons and events.

The ledger is one directory of JSONL files, shared by the lanes, which run as separate processes
(spec §6.2):

- ``experiments.jsonl``, one ExperimentRecord per line. A lane writes an experiment's record as it
  goes (spec §8.7), and the last line for an experiment_id wins.
- ``lessons.jsonl`` and the ``lessons.md`` generated from it (goaly.rsi.lessons).
- ``lane_events.jsonl``, one LaneEvent per line: what happens to a lane outside any one
  experiment, such as starts, restarts, stops, results PRs and notifications (spec §8.7).

Every write appends one line under the ledger's exclusive lock and syncs it to disk before the
lock is released, so a crash loses at most the line being written. Reads take the shared lock, so
they never see a line half written. A torn last line left by a crash is skipped, and the next
write cuts it off (goaly.contracts.jsonl).
"""

import os
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from google.protobuf.message import Message

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    DECISION_PENDING,
    DECISION_REJECTED,
    LANE_EVENT_KIND_RESULTS_PR,
    LANE_EVENT_KIND_UNSPECIFIED,
    ExperimentRecord,
    Lane,
    LaneEvent,
    LaneEventKind,
)
from goaly.contracts.jsonl import append_jsonl, read_jsonl
from goaly.rsi.lanes import lane_info
from goaly.rsi.locks import file_lock

__all__ = [
    "EXPERIMENTS",
    "LANE_EVENTS",
    "LESSONS",
    "LESSONS_MARKDOWN",
    "Ledger",
    "LedgerError",
    "LedgerFiles",
    "check_id",
    "default_ledger_dir",
]

EXPERIMENTS = "experiments.jsonl"
LESSONS = "lessons.jsonl"
LESSONS_MARKDOWN = "lessons.md"
LANE_EVENTS = "lane_events.jsonl"
_LOCK = ".lock"

# IDs name branches, worktrees, run directories and container labels, so they stay plain.
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_FINAL = frozenset({DECISION_ACCEPTED, DECISION_REJECTED, DECISION_ABORTED})


class LedgerError(Exception):
    """The ledger's records contradict each other, so it cannot be replayed safely."""


def default_ledger_dir(repo_root: Path) -> Path:
    """Return the ledger directory a lane uses unless config names another: ``rsi/ledger``."""
    return repo_root / "rsi" / "ledger"


def check_id(value: str, field: str) -> None:
    """Raise ValueError unless ``value`` is 1-64 letters, digits, ``_`` or ``-``, not starting
    with ``_`` or ``-``.
    """
    if not _ID.fullmatch(value):
        raise ValueError(
            f"{field} must be 1-64 letters, digits, '_' or '-', starting with a letter or a "
            f"digit; got {value!r}"
        )


class LedgerFiles:
    """The ledger's files while :meth:`Ledger.session` holds the ledger's lock."""

    def __init__(self, directory: Path, *, writable: bool, present: bool = True) -> None:
        self._directory = directory
        self._writable = writable
        self._present = present  # False if the ledger did not exist when the session began.

    def read[M: Message](self, name: str, message_type: type[M]) -> list[M]:
        """Return the messages in the file ``name``, or none if it does not exist."""
        path = self._directory / name
        if not self._present or not path.exists():
            return []
        return list(read_jsonl(path, message_type))

    def append(self, name: str, message: Message) -> None:
        """Append ``message`` to the file ``name`` as one line, and sync it to disk."""
        self._check_writable()
        path = self._directory / name
        created = not path.exists()
        append_jsonl(path, message)
        _fsync(path)
        if created:  # Make the new file's directory entry durable too.
            _fsync(self._directory)

    def replace_text(self, name: str, text: str) -> None:
        """Replace the file ``name`` with ``text``; readers see the old file or the new one."""
        self._check_writable()
        path = self._directory / name
        temporary = path.with_name(f".{name}.{secrets.token_hex(8)}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)  # Already gone once it has replaced path.

    def _check_writable(self) -> None:
        if not self._writable:
            raise RuntimeError("this ledger session is read-only; open it with write=True")


class Ledger:
    """The ledger in ``directory``, created by the first write."""

    def __init__(self, directory: Path, *, lock_timeout_s: float = 60.0) -> None:
        self.directory = directory
        self.lock_timeout_s = lock_timeout_s

    @contextmanager
    def session(self, *, write: bool = False) -> Iterator[LedgerFiles]:
        """Hold the ledger's lock, shared for reading or exclusive with ``write``, for a block.

        Raises goaly.rsi.locks.LockTimeoutError if another process holds it too long.
        """
        if not write and not self.directory.exists():
            yield LedgerFiles(self.directory, writable=False, present=False)  # Nothing to read.
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = self.directory / _LOCK
        with file_lock(lock, shared=not write, timeout_s=self.lock_timeout_s):
            yield LedgerFiles(self.directory, writable=write)

    def record_experiment(self, record: ExperimentRecord) -> None:
        """Append ``record`` as its experiment's latest state.

        Raises ValueError, writing nothing, if its experiment_id is not a valid ID, its lane or
        decision is unset or unknown, its round is below 1, or it is accepted without a
        candidate_commit.
        """
        _check_experiment(record)
        with self.session(write=True) as files:
            files.append(EXPERIMENTS, record)

    def experiments(self) -> list[ExperimentRecord]:
        """Return every line of experiments.jsonl, in order."""
        with self.session() as files:
            return files.read(EXPERIMENTS, ExperimentRecord)

    def latest_experiments(self, lane: Lane.ValueType | None = None) -> dict[str, ExperimentRecord]:
        """Return each experiment's latest record, of one ``lane`` or of all, keyed by
        experiment_id in the order the experiments first appear.
        """
        latest: dict[str, ExperimentRecord] = {}
        for record in self.experiments():
            latest[record.experiment_id] = record  # Replacing a value keeps its key's place.
        return {key: record for key, record in latest.items() if lane in (None, record.lane)}

    def pending_experiments(self, lane: Lane.ValueType) -> list[ExperimentRecord]:
        """Return the latest records of ``lane``'s experiments that are not decided or aborted.

        A record whose decision is unset counts as pending, so a resumed lane resolves it.
        """
        return [
            record
            for record in self.latest_experiments(lane).values()
            if record.decision not in _FINAL
        ]

    def best_version(self, lane: Lane.ValueType) -> str | None:
        """Return the commit of ``lane``'s best version, or None while it has accepted nothing.

        The best is the candidate accepted in the latest round, as each round accepts at most one
        (spec §8.2). Raises LedgerError if that round accepted more than one, or if the accepted
        record has no candidate_commit.
        """
        accepted = [
            record
            for record in self.latest_experiments(lane).values()
            if record.decision == DECISION_ACCEPTED
        ]
        if not accepted:
            return None
        last_round = max(record.round for record in accepted)
        newest = [record for record in accepted if record.round == last_round]
        if len(newest) > 1:
            names = ", ".join(record.experiment_id for record in newest)
            raise LedgerError(
                f"experiments {names} were all accepted in round {last_round}, which can accept "
                "only one"
            )
        best = newest[0]
        if not best.candidate_commit:
            raise LedgerError(f"experiment {best.experiment_id} is accepted but has no commit")
        return best.candidate_commit

    def record_lane_event(self, event: LaneEvent) -> None:
        """Append ``event`` to lane_events.jsonl.

        Raises ValueError, writing nothing, if its lane or kind is unset or unknown, it has no
        occurred_at, or it is a results PR without its pr_number, which keeps a restarted lane
        from opening the PR again (spec §8.7).
        """
        lane_info(event.lane)
        if event.kind == LANE_EVENT_KIND_UNSPECIFIED or event.kind not in LaneEventKind.values():
            raise ValueError(f"a lane event's kind must be set and known; got {event.kind}")
        if not event.HasField("occurred_at"):
            raise ValueError("a lane event needs its occurred_at")
        if event.kind == LANE_EVENT_KIND_RESULTS_PR and event.pr_number < 1:
            raise ValueError("a results PR event needs its pr_number")
        with self.session(write=True) as files:
            files.append(LANE_EVENTS, event)

    def lane_events(self, lane: Lane.ValueType | None = None) -> list[LaneEvent]:
        """Return the events of one ``lane``, or of all, in the order they were recorded."""
        with self.session() as files:
            events = files.read(LANE_EVENTS, LaneEvent)
        return [event for event in events if lane in (None, event.lane)]


def _check_experiment(record: ExperimentRecord) -> None:
    check_id(record.experiment_id, "experiment_id")
    lane_info(record.lane)
    if record.decision not in _FINAL | {DECISION_PENDING}:
        raise ValueError(
            f"experiment {record.experiment_id}: decision must be DECISION_PENDING or a final "
            f"decision; got {record.decision}"
        )
    if record.round < 1:
        raise ValueError(f"experiment {record.experiment_id}: round counts from 1")
    if record.decision == DECISION_ACCEPTED and not record.candidate_commit:
        raise ValueError(
            f"experiment {record.experiment_id}: an accepted record needs its candidate_commit"
        )


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
