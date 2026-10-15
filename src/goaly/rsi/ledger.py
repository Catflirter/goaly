"""The RSI ledger (spec §8.6): the lanes' shared memory of experiments, lessons and events.

The ledger is one directory of JSONL files, shared by the lanes, which run as separate processes
(spec §6.2):

- ``experiments.jsonl``, one ExperimentRecord per line. A lane writes an experiment's record as it
  goes (spec §8.7), and the last line for an experiment wins. An experiment ID belongs to one lane
  and one round for good, and a decided experiment never goes back to pending.
- ``lessons.jsonl`` and the ``lessons.md`` generated from it (goaly.rsi.lessons).
- ``lane_events.jsonl``, one LaneEvent per line: what happens to a lane outside any one
  experiment, such as starts, restarts, stops, results PRs and notifications (spec §8.7).

Every ID the ledger records matches goaly.rsi.ids.ID_PATTERN: 1 to 64 letters, digits, ``_`` or
``-``, starting with a letter or a digit.

Every write appends one line under the ledger's exclusive lock and syncs it to disk before the
lock is released, so a crash loses at most the line being written. Reads take the shared lock, so
they never see a line half written. A torn last line left by a crash is skipped, and the next
write cuts it off (goaly.contracts.jsonl).
"""

import logging
import os
import secrets
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import cast

from google.protobuf.message import Message

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    DECISION_PENDING,
    DECISION_REJECTED,
    LANE_EVENT_KIND_RESULTS_PR,
    LANE_EVENT_KIND_UNSPECIFIED,
    REJECTION_REASON_UNSPECIFIED,
    ExperimentRecord,
    Lane,
    LaneEvent,
    LaneEventKind,
    Lesson,
    RejectionReason,
)
from goaly.contracts.jsonl import append_jsonl, read_jsonl
from goaly.rsi.ids import check_id
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
    "default_ledger_dir",
]

EXPERIMENTS = "experiments.jsonl"
LESSONS = "lessons.jsonl"
LESSONS_MARKDOWN = "lessons.md"
LANE_EVENTS = "lane_events.jsonl"
_LOCK = ".lock"
_FILES: dict[str, type[Message]] = {
    EXPERIMENTS: ExperimentRecord,
    LESSONS: Lesson,
    LANE_EVENTS: LaneEvent,
}

_FINAL = frozenset({DECISION_ACCEPTED, DECISION_REJECTED, DECISION_ABORTED})

_logger = logging.getLogger(__name__)


class LedgerError(Exception):
    """The ledger's records contradict each other, so it cannot be replayed safely."""


def default_ledger_dir(repo_root: Path) -> Path:
    """Return the ledger directory a lane uses unless config names another: ``rsi/ledger``."""
    return repo_root / "rsi" / "ledger"


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
        """Append ``message`` to the ledger file ``name`` as one line, and sync it to disk.

        Raises ValueError, writing nothing, if ``name`` is not a ledger file, ``message`` is not
        of the file's type, or it is not a valid record: see Ledger.record_experiment and
        Ledger.record_lane_event, and a lesson needs valid IDs, a known lane and a text.
        """
        self._check_writable()
        expected = _FILES.get(name)
        if expected is None:
            raise ValueError(f"{name} is not a ledger file")
        if type(message) is not expected:
            raise ValueError(f"{name} holds {expected.__name__}, not {type(message).__name__}")
        if isinstance(message, ExperimentRecord):
            _check_experiment(message, self.read(EXPERIMENTS, ExperimentRecord))
        elif isinstance(message, LaneEvent):
            _check_lane_event(message)
        else:  # Its type matched the file's, so it is a Lesson.
            _check_lesson(cast(Lesson, message))
        path = self._directory / name
        created = not path.exists()
        append_jsonl(path, message)
        _fsync(path)
        if created:  # Make the new file's directory entry durable too.
            _fsync(self._directory)

    def replace_text(self, name: str, text: str) -> None:
        """Replace the generated page ``name`` with ``text``, durably: readers see the old page
        or the new one. Temporary files a crash left behind are removed first.
        """
        self._check_writable()
        if name != LESSONS_MARKDOWN:
            raise ValueError(f"{name} is not a ledger file to generate")
        path = self._directory / name
        for stale in self._directory.glob(f".{name}.*.tmp"):
            stale.unlink(missing_ok=True)
        temporary = path.with_name(f".{name}.{secrets.token_hex(8)}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as file:
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(path)
            _fsync(self._directory)
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

        Sessions are not reentrant: one opened inside another in the same process waits for the
        first and times out. Raises goaly.rsi.locks.LockTimeoutError if another process holds the
        lock too long. A read of a ledger whose lock file is missing and cannot be created, as in
        a directory without write access, goes ahead without the lock, so it can see a write in
        progress as a torn last line, which it skips.
        """
        if not write and not self.directory.exists():
            yield LedgerFiles(self.directory, writable=False, present=False)  # Nothing to read.
            return
        if write:
            _make_directory(self.directory)
        lock = self.directory / _LOCK
        with ExitStack() as held:
            try:
                held.enter_context(file_lock(lock, shared=not write, timeout_s=self.lock_timeout_s))
            except PermissionError:
                if write:
                    raise
                _logger.warning("reading the ledger without the lock", extra={"lock": str(lock)})
            yield LedgerFiles(self.directory, writable=write)

    def record_experiment(self, record: ExperimentRecord) -> None:
        """Append ``record`` as its experiment's latest state.

        Raises ValueError, writing nothing, if:

        - its experiment or hypothesis ID is not a valid ID, or its lane is unset or unknown;
        - its decision is unset or unknown, or its round is below 1;
        - it is accepted without a candidate_commit;
        - it is rejected without a rejection reason, gives a reason with another decision, or
          gives an unset or unknown reason;
        - its experiment ID already belongs to another lane, or to another round;
        - it would put a decided experiment back to pending.
        """
        with self.session(write=True) as files:
            files.append(EXPERIMENTS, record)

    def experiments(self) -> list[ExperimentRecord]:
        """Return every line of experiments.jsonl, in order."""
        with self.session() as files:
            return files.read(EXPERIMENTS, ExperimentRecord)

    def latest_experiments(self, lane: Lane.ValueType) -> dict[str, ExperimentRecord]:
        """Return the latest record of each of ``lane``'s experiments, keyed by experiment ID in
        the order the experiments first appear.

        Only the lane's own records count, so a hand-edited ledger that uses one ID in two lanes
        still replays each lane apart.
        """
        latest: dict[str, ExperimentRecord] = {}
        for record in self.experiments():
            if record.lane == lane:
                latest[record.experiment_id] = record  # Replacing a value keeps its key's place.
        return latest

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
        (spec §8.2). So write an experiment as accepted only once its confirmation has passed and
        it has become the best (spec §8.2 step 7). Raises LedgerError if that round accepted more
        than one, or if the accepted record has no candidate_commit.
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
        occurred_at, an experiment ID it names is not a valid ID, or it is a results PR without
        its pr_number, which keeps a restarted lane from opening the PR again (spec §8.7).
        """
        with self.session(write=True) as files:
            files.append(LANE_EVENTS, event)

    def lane_events(self, lane: Lane.ValueType | None = None) -> list[LaneEvent]:
        """Return the events of one ``lane``, or of all, in the order they were recorded."""
        with self.session() as files:
            events = files.read(LANE_EVENTS, LaneEvent)
        return [event for event in events if lane in (None, event.lane)]


def _check_experiment(record: ExperimentRecord, earlier: Sequence[ExperimentRecord]) -> None:
    name = record.experiment_id
    check_id(name, "experiment_id")
    check_id(record.hypothesis.hypothesis_id, "hypothesis.hypothesis_id")
    lane_info(record.lane)
    if record.decision not in _FINAL | {DECISION_PENDING}:
        raise ValueError(
            f"experiment {name}: decision must be DECISION_PENDING or a final decision; got "
            f"{record.decision}"
        )
    if record.round < 1:
        raise ValueError(f"experiment {name}: round counts from 1")
    if record.decision == DECISION_ACCEPTED and not record.candidate_commit:
        raise ValueError(f"experiment {name}: an accepted record needs its candidate_commit")
    _check_rejection_reasons(record)
    previous = [old for old in earlier if old.experiment_id == name]
    for old in previous:
        if old.lane != record.lane:
            raise ValueError(
                f"experiment {name} belongs to {Lane.Name(old.lane)}, not to "
                f"{Lane.Name(record.lane)}"
            )
    if previous:
        last = previous[-1]
        if last.round != record.round:
            raise ValueError(f"experiment {name} ran in round {last.round}, not {record.round}")
        if last.decision in _FINAL and record.decision == DECISION_PENDING:
            raise ValueError(f"experiment {name} is decided, so it cannot be pending again")


def _check_rejection_reasons(record: ExperimentRecord) -> None:
    name = record.experiment_id
    if record.decision == DECISION_REJECTED and not record.rejection_reasons:
        raise ValueError(f"experiment {name}: a rejected record needs its rejection_reasons")
    if record.decision != DECISION_REJECTED and record.rejection_reasons:
        raise ValueError(f"experiment {name}: only a rejected record has rejection_reasons")
    for reason in record.rejection_reasons:
        if reason == REJECTION_REASON_UNSPECIFIED or reason not in RejectionReason.values():
            raise ValueError(f"experiment {name}: rejection reason {reason} is unset or unknown")


def _check_lane_event(event: LaneEvent) -> None:
    lane_info(event.lane)
    if event.kind == LANE_EVENT_KIND_UNSPECIFIED or event.kind not in LaneEventKind.values():
        raise ValueError(f"a lane event's kind must be set and known; got {event.kind}")
    if not event.HasField("occurred_at"):
        raise ValueError("a lane event needs its occurred_at")
    if event.kind == LANE_EVENT_KIND_RESULTS_PR and event.pr_number < 1:
        raise ValueError("a results PR event needs its pr_number")
    for experiment_id in event.experiment_ids:
        check_id(experiment_id, "experiment_id")


def _check_lesson(lesson: Lesson) -> None:
    check_id(lesson.lesson_id, "lesson_id")
    lane_info(lesson.lane)
    if not lesson.text.strip():
        raise ValueError(f"lesson {lesson.lesson_id}: the text must not be blank")
    for experiment_id in [*lesson.experiment_ids, *lesson.helped_by, *lesson.hurt_by]:
        check_id(experiment_id, "experiment_id")


def _make_directory(directory: Path) -> None:
    """Create ``directory`` and its missing parents, making each new entry durable."""
    missing = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync(created.parent)


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
