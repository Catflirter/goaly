"""Tests for goaly.rsi.ledger: the experiments log, its lock and the replay helpers."""

import logging
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    DECISION_PENDING,
    DECISION_REJECTED,
    DECISION_UNSPECIFIED,
    LANE_EFFISQL,
    LANE_SWE,
    LANE_UNSPECIFIED,
    REJECTION_REASON_NO_SCREEN_GAIN,
    REJECTION_REASON_UNSPECIFIED,
    Decision,
    ExperimentRecord,
    Hypothesis,
    Lane,
    LaneEvent,
    LaneEventKind,
    Lesson,
    RejectionReason,
)
from goaly.contracts.jsonl import append_jsonl
from goaly.rsi.ids import ID_PATTERN
from goaly.rsi.ledger import Ledger, LedgerError, default_ledger_dir
from goaly.rsi.locks import LockTimeoutError, file_lock


def experiment(
    experiment_id: str,
    decision: Decision.ValueType = DECISION_PENDING,
    *,
    lane: Lane.ValueType = LANE_EFFISQL,
    round_number: int = 1,
    commit: str = "",
    reasons: Sequence[RejectionReason.ValueType] | None = None,
) -> ExperimentRecord:
    if decision == DECISION_ACCEPTED and not commit:
        commit = f"{experiment_id}-commit"
    if reasons is None:
        reasons = [REJECTION_REASON_NO_SCREEN_GAIN] if decision == DECISION_REJECTED else []
    return ExperimentRecord(
        experiment_id=experiment_id,
        lane=lane,
        round=round_number,
        decision=decision,
        candidate_commit=commit,
        rejection_reasons=reasons,
        hypothesis=Hypothesis(hypothesis_id=f"h-{experiment_id}", change="tighten the prompt"),
    )


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger")


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def raw_append(ledger: Ledger, record: ExperimentRecord) -> None:
    """Append ``record`` as a hand edit would, past the ledger's checks."""
    ledger.directory.mkdir(parents=True, exist_ok=True)
    append_jsonl(ledger.directory / "experiments.jsonl", record)


@pytest.fixture
def writable_again(ledger: Ledger) -> Iterator[None]:
    """Give the ledger directory its write permission back after the test."""
    yield
    ledger.directory.chmod(0o755)


# Writing and reading experiments.


def test_default_ledger_dir_is_rsi_ledger_in_the_repository() -> None:
    assert default_ledger_dir(Path("/repo")) == Path("/repo/rsi/ledger")


def test_records_read_back_in_the_order_written(ledger: Ledger) -> None:
    records = [experiment("e1"), experiment("e2"), experiment("e1", DECISION_REJECTED)]

    for record in records:
        ledger.record_experiment(record)

    assert ledger.experiments() == records
    assert len(lines(ledger.directory / "experiments.jsonl")) == 3


def test_a_ledger_that_was_never_written_is_empty_and_is_not_created(ledger: Ledger) -> None:
    assert ledger.experiments() == []
    assert ledger.latest_experiments(LANE_EFFISQL) == {}
    assert not ledger.directory.exists()


def test_a_torn_last_line_is_skipped_then_cut_off_by_the_next_write(
    ledger: Ledger, caplog: pytest.LogCaptureFixture
) -> None:
    ledger.record_experiment(experiment("e1"))
    path = ledger.directory / "experiments.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write('{"experiment_id": "e2", "la')  # A crash in the middle of a write.

    with caplog.at_level(logging.WARNING):
        assert ledger.experiments() == [experiment("e1")]
    assert "torn" in caplog.text

    ledger.record_experiment(experiment("e3"))

    assert ledger.experiments() == [experiment("e1"), experiment("e3")]
    assert len(lines(path)) == 2


def test_each_write_is_synced_to_disk(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[str] = []
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        synced.append(str(Path(f"/proc/self/fd/{descriptor}").readlink()))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    path = ledger.directory / "experiments.jsonl"

    ledger.record_experiment(experiment("e1"))
    # The first write creates the directory and the file, so their entries are synced too.
    assert synced == [str(tmp_path), str(path), str(ledger.directory)]

    synced.clear()
    ledger.record_experiment(experiment("e2"))
    assert synced == [str(path)]


@pytest.mark.parametrize(
    ("record", "problem"),
    [
        (experiment(""), "experiment_id"),
        (experiment("has space"), "experiment_id"),
        (experiment("../escape"), "experiment_id"),
        (experiment("-leading-dash"), "experiment_id"),
        (experiment("e" * 65), "experiment_id"),
        (
            ExperimentRecord(
                experiment_id="e1", lane=LANE_EFFISQL, round=1, decision=DECISION_PENDING
            ),
            "hypothesis",
        ),
        (experiment("e1", lane=LANE_UNSPECIFIED), "lane"),
        (experiment("e1", lane=Lane.ValueType(99)), "lane"),
        (experiment("e1", DECISION_UNSPECIFIED), "decision"),
        (experiment("e1", Decision.ValueType(99)), "decision"),
        (experiment("e1", round_number=0), "round"),
        (experiment("e1", DECISION_REJECTED, reasons=[]), "rejection_reasons"),
        (experiment("e1", DECISION_REJECTED, reasons=[REJECTION_REASON_UNSPECIFIED]), "reason"),
        (experiment("e1", DECISION_REJECTED, reasons=[RejectionReason.ValueType(99)]), "reason"),
        (experiment("e1", reasons=[REJECTION_REASON_NO_SCREEN_GAIN]), "rejection_reasons"),
        (
            experiment("e1", DECISION_ACCEPTED, reasons=[REJECTION_REASON_NO_SCREEN_GAIN]),
            "rejection_reasons",
        ),
    ],
)
def test_an_invalid_record_is_refused_and_nothing_is_written(
    ledger: Ledger, record: ExperimentRecord, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        ledger.record_experiment(record)

    assert not (ledger.directory / "experiments.jsonl").exists()


def test_an_accepted_record_needs_its_commit(ledger: Ledger) -> None:
    accepted = experiment("e1", DECISION_ACCEPTED)
    accepted.candidate_commit = ""

    with pytest.raises(ValueError, match="candidate_commit"):
        ledger.record_experiment(accepted)


def test_an_experiment_id_belongs_to_one_lane(ledger: Ledger) -> None:
    ledger.record_experiment(experiment("e1"))

    with pytest.raises(ValueError, match="e1 belongs to LANE_EFFISQL"):
        ledger.record_experiment(experiment("e1", lane=LANE_SWE))

    assert ledger.experiments() == [experiment("e1")]


def test_an_experiments_round_never_changes(ledger: Ledger) -> None:
    ledger.record_experiment(experiment("e1", round_number=2))

    with pytest.raises(ValueError, match="round"):
        ledger.record_experiment(experiment("e1", DECISION_REJECTED, round_number=3))


@pytest.mark.parametrize("final", [DECISION_ACCEPTED, DECISION_REJECTED, DECISION_ABORTED])
def test_a_decided_experiment_never_goes_back_to_pending(
    ledger: Ledger, final: Decision.ValueType
) -> None:
    ledger.record_experiment(experiment("e1"))
    ledger.record_experiment(experiment("e1", final))

    with pytest.raises(ValueError, match="pending"):
        ledger.record_experiment(experiment("e1"))

    assert len(ledger.experiments()) == 2


def test_a_write_waits_for_the_ledger_lock(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    directory.mkdir()
    ledger = Ledger(directory, lock_timeout_s=0.2)

    with file_lock(directory / ".lock"), pytest.raises(LockTimeoutError):
        ledger.record_experiment(experiment("e1"))

    assert not (directory / "experiments.jsonl").exists()


def test_a_read_waits_for_a_write_in_progress(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    directory.mkdir()
    ledger = Ledger(directory, lock_timeout_s=0.2)

    with file_lock(directory / ".lock"), pytest.raises(LockTimeoutError):
        ledger.experiments()


@pytest.mark.usefixtures("writable_again")
def test_a_ledger_without_write_access_can_still_be_read(
    ledger: Ledger, caplog: pytest.LogCaptureFixture
) -> None:
    ledger.record_experiment(experiment("e1"))
    ledger.directory.chmod(0o555)

    assert ledger.experiments() == [experiment("e1")]  # Under the shared lock.

    ledger.directory.chmod(0o755)
    (ledger.directory / ".lock").unlink()
    ledger.directory.chmod(0o555)
    with caplog.at_level(logging.WARNING):
        assert ledger.experiments() == [experiment("e1")]  # No lock file, so no lock.
    assert "without the lock" in caplog.text


def test_a_session_appends_only_valid_messages_of_each_files_type(ledger: Ledger) -> None:
    with ledger.session(write=True) as files:
        with pytest.raises(ValueError, match=r"experiments\.jsonl holds ExperimentRecord"):
            files.append("experiments.jsonl", Lesson(lesson_id="l1"))
        with pytest.raises(ValueError, match="not a ledger file"):
            files.append("other.jsonl", experiment("e1"))
        with pytest.raises(ValueError, match="experiment_id"):
            files.append("experiments.jsonl", experiment("bad id"))
        with pytest.raises(ValueError, match="not a ledger file"):
            files.replace_text("experiments.jsonl", "")

    assert not (ledger.directory / "experiments.jsonl").exists()


def test_a_read_session_refuses_writes(ledger: Ledger) -> None:
    ledger.record_experiment(experiment("e1"))

    with ledger.session() as files, pytest.raises(RuntimeError, match="read-only"):
        files.append("experiments.jsonl", experiment("e2"))

    assert ledger.experiments() == [experiment("e1")]


# Starts once stdin closes, then writes argv[3] records whose IDs start with argv[2].
WRITER = """
import sys
from pathlib import Path
from goaly.contracts.gen.experiment_pb2 import (
    DECISION_PENDING, LANE_EFFISQL, ExperimentRecord, Hypothesis,
)
from goaly.rsi.ledger import Ledger

ledger = Ledger(Path(sys.argv[1]))
prefix, count = sys.argv[2], int(sys.argv[3])
sys.stdin.read()
for n in range(count):
    ledger.record_experiment(ExperimentRecord(
        experiment_id=f"{prefix}-{n}", lane=LANE_EFFISQL, round=1, decision=DECISION_PENDING,
        hypothesis=Hypothesis(hypothesis_id="h1"), decision_reason="x" * 4000,
    ))
"""


def test_two_processes_writing_at_once_lose_and_tear_nothing(ledger: Ledger) -> None:
    count = 150
    writers = [
        subprocess.Popen(  # noqa: S603  (a fixed script and arguments built here)
            [sys.executable, "-c", WRITER, str(ledger.directory), prefix, str(count)],
            stdin=subprocess.PIPE,
        )
        for prefix in ("a", "b")
    ]
    for writer in writers:  # Release both at once.
        assert writer.stdin is not None
        writer.stdin.close()
    for writer in writers:
        assert writer.wait(timeout=120) == 0

    records = ledger.experiments()

    expected = {f"{prefix}-{n}" for prefix in ("a", "b") for n in range(count)}
    assert sorted(record.experiment_id for record in records) == sorted(expected)
    assert len(lines(ledger.directory / "experiments.jsonl")) == 2 * count
    # Each writer's records stay in the order it wrote them.
    for prefix in ("a", "b"):
        mine = [r.experiment_id for r in records if r.experiment_id.startswith(f"{prefix}-")]
        assert mine == [f"{prefix}-{n}" for n in range(count)]


def test_the_id_pattern_is_the_documented_one() -> None:
    assert ID_PATTERN.pattern == "[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
    assert ID_PATTERN.fullmatch("exp_2026-10-15_01")


# Replay helpers.


def test_the_latest_record_wins_and_experiments_keep_their_first_position(
    ledger: Ledger,
) -> None:
    for record in [
        experiment("e1"),
        experiment("e2"),
        experiment("e1", DECISION_REJECTED),
        experiment("s1", lane=LANE_SWE),
    ]:
        ledger.record_experiment(record)

    latest = ledger.latest_experiments(LANE_EFFISQL)

    assert list(latest) == ["e1", "e2"]
    assert latest["e1"] == experiment("e1", DECISION_REJECTED)
    assert list(ledger.latest_experiments(LANE_SWE)) == ["s1"]


def test_one_id_in_two_lanes_of_a_hand_edited_ledger_replays_apart(ledger: Ledger) -> None:
    raw_append(ledger, experiment("e1"))
    raw_append(ledger, experiment("e1", DECISION_ABORTED, lane=LANE_SWE))

    assert ledger.latest_experiments(LANE_EFFISQL) == {"e1": experiment("e1")}
    assert ledger.pending_experiments(LANE_EFFISQL) == [experiment("e1")]
    assert ledger.pending_experiments(LANE_SWE) == []


def test_pending_experiments_are_the_lanes_undecided_ones(ledger: Ledger) -> None:
    for record in [
        experiment("decided"),
        experiment("waiting"),
        experiment("aborted", DECISION_ABORTED),
        experiment("other-lane", lane=LANE_SWE),
        experiment("decided", DECISION_ACCEPTED),
        experiment("rejected", DECISION_REJECTED),
    ]:
        ledger.record_experiment(record)

    assert ledger.pending_experiments(LANE_EFFISQL) == [experiment("waiting")]
    assert ledger.pending_experiments(LANE_SWE) == [experiment("other-lane", lane=LANE_SWE)]


def test_a_record_with_no_decision_counts_as_pending(ledger: Ledger) -> None:
    # The ledger never writes one, but a resumed lane must still resolve a hand-edited record.
    undecided = ExperimentRecord(experiment_id="e1", lane=LANE_EFFISQL, round=1)
    raw_append(ledger, undecided)

    assert ledger.pending_experiments(LANE_EFFISQL) == [undecided]


def test_a_lane_with_no_accepted_experiment_has_no_best_version(ledger: Ledger) -> None:
    ledger.record_experiment(experiment("e1", DECISION_REJECTED))
    ledger.record_experiment(experiment("s1", DECISION_ACCEPTED, lane=LANE_SWE))

    assert ledger.best_version(LANE_EFFISQL) is None


def test_the_best_version_is_the_candidate_accepted_in_the_latest_round(ledger: Ledger) -> None:
    for record in [
        experiment("e1", DECISION_ACCEPTED, round_number=1, commit="c1"),
        experiment("e3", DECISION_ACCEPTED, round_number=3, commit="c3"),
        experiment("e4", DECISION_REJECTED, round_number=4),
        experiment("e5", round_number=5),
        # Writing an older experiment again does not make it the best.
        experiment("e1", DECISION_ACCEPTED, round_number=1, commit="c1"),
        experiment("s9", DECISION_ACCEPTED, lane=LANE_SWE, round_number=9, commit="s9"),
    ]:
        ledger.record_experiment(record)

    assert ledger.best_version(LANE_EFFISQL) == "c3"
    assert ledger.best_version(LANE_SWE) == "s9"


def test_an_acceptance_overwritten_by_a_later_record_no_longer_counts(ledger: Ledger) -> None:
    ledger.record_experiment(experiment("e1", DECISION_ACCEPTED, round_number=1, commit="c1"))
    ledger.record_experiment(experiment("e2", DECISION_ACCEPTED, round_number=2, commit="c2"))
    ledger.record_experiment(experiment("e2", DECISION_REJECTED, round_number=2))

    assert ledger.best_version(LANE_EFFISQL) == "c1"


def test_two_acceptances_in_the_latest_round_are_an_error(ledger: Ledger) -> None:
    ledger.record_experiment(experiment("e1", DECISION_ACCEPTED, round_number=2))
    ledger.record_experiment(experiment("e2", DECISION_ACCEPTED, round_number=2))

    with pytest.raises(LedgerError, match="round 2"):
        ledger.best_version(LANE_EFFISQL)


def test_an_accepted_record_without_a_commit_is_an_error(ledger: Ledger) -> None:
    raw_append(
        ledger,
        ExperimentRecord(
            experiment_id="e1", lane=LANE_EFFISQL, round=1, decision=DECISION_ACCEPTED
        ),
    )

    with pytest.raises(LedgerError, match="e1"):
        ledger.best_version(LANE_EFFISQL)


# Lane events.

AT = Timestamp(seconds=1_760_529_600)
STARTED = LaneEventKind.LANE_EVENT_KIND_STARTED


def test_lane_events_read_back_in_order(ledger: Ledger) -> None:
    events = [
        LaneEvent(lane=LANE_EFFISQL, occurred_at=AT, kind=STARTED),
        LaneEvent(lane=LANE_SWE, occurred_at=AT, kind=STARTED),
        LaneEvent(
            lane=LANE_EFFISQL,
            occurred_at=AT,
            kind=LaneEventKind.LANE_EVENT_KIND_RESULTS_PR,
            pr_number=21,
            pr_url="https://github.com/Catflirter/goaly/pull/21",
            head_commit="abc123",
            experiment_ids=["e1", "e4"],
        ),
        LaneEvent(
            lane=LANE_EFFISQL,
            occurred_at=AT,
            kind=LaneEventKind.LANE_EVENT_KIND_STOPPED,
            detail="4 rounds without a gain",
        ),
    ]

    for lane_event in events:
        ledger.record_lane_event(lane_event)

    assert ledger.lane_events() == events
    assert ledger.lane_events(LANE_EFFISQL) == [events[0], events[2], events[3]]
    assert ledger.lane_events(LANE_SWE) == [events[1]]
    assert len(lines(ledger.directory / "lane_events.jsonl")) == 4


def test_a_ledger_without_lane_events_has_none(ledger: Ledger) -> None:
    ledger.record_experiment(experiment("e1"))

    assert ledger.lane_events() == []


@pytest.mark.parametrize(
    ("lane_event", "problem"),
    [
        (LaneEvent(lane=LANE_UNSPECIFIED, occurred_at=AT, kind=STARTED), "lane"),
        (
            LaneEvent(
                lane=LANE_EFFISQL, occurred_at=AT, kind=LaneEventKind.LANE_EVENT_KIND_UNSPECIFIED
            ),
            "kind",
        ),
        (LaneEvent(lane=LANE_EFFISQL, occurred_at=AT, kind=LaneEventKind.ValueType(99)), "kind"),
        (LaneEvent(lane=LANE_EFFISQL, kind=STARTED), "occurred_at"),
        (
            LaneEvent(
                lane=LANE_EFFISQL,
                occurred_at=AT,
                kind=LaneEventKind.LANE_EVENT_KIND_RESULTS_PR,
                pr_url="https://github.com/Catflirter/goaly/pull/21",
            ),
            "pr_number",
        ),
        (
            LaneEvent(lane=LANE_EFFISQL, occurred_at=AT, kind=STARTED, experiment_ids=["a b"]),
            "experiment_id",
        ),
    ],
)
def test_an_invalid_lane_event_is_refused_and_nothing_is_written(
    ledger: Ledger, lane_event: LaneEvent, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        ledger.record_lane_event(lane_event)

    assert not (ledger.directory / "lane_events.jsonl").exists()


@pytest.mark.parametrize(
    ("lesson", "problem"),
    [
        (Lesson(lesson_id="l1", lane=LANE_EFFISQL, text=" "), "text"),
        (Lesson(lesson_id="bad id", lane=LANE_EFFISQL, text="t"), "lesson_id"),
        (Lesson(lesson_id="l1", lane=LANE_UNSPECIFIED, text="t"), "lane"),
        (Lesson(lesson_id="l1", lane=LANE_EFFISQL, text="t", hurt_by=["a b"]), "experiment_id"),
    ],
)
def test_a_session_refuses_an_invalid_lesson(ledger: Ledger, lesson: Lesson, problem: str) -> None:
    with ledger.session(write=True) as files, pytest.raises(ValueError, match=problem):
        files.append("lessons.jsonl", lesson)

    assert not (ledger.directory / "lessons.jsonl").exists()


@pytest.mark.usefixtures("writable_again")
def test_a_write_without_a_lock_file_it_can_create_fails(ledger: Ledger) -> None:
    ledger.record_experiment(experiment("e1"))
    (ledger.directory / ".lock").unlink()
    ledger.directory.chmod(0o555)

    with pytest.raises(PermissionError):
        ledger.record_experiment(experiment("e2"))
