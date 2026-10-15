"""Tests for goaly.rsi.lessons: the lessons store and the generated lessons.md."""

import os
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from google.protobuf import json_format
from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_PENDING,
    LANE_EFFISQL,
    LANE_SWE,
    LANE_UNSPECIFIED,
    ExperimentRecord,
    Hypothesis,
    Lane,
    Lesson,
)
from goaly.rsi import lessons
from goaly.rsi.ledger import Ledger
from goaly.rsi.lessons import LessonNotFoundError, LessonStore, render_markdown

START = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def timestamp(moment: datetime) -> Timestamp:
    stamp = Timestamp()
    stamp.FromDatetime(moment)
    return stamp


def ticking_clock() -> Callable[[], datetime]:
    """A clock that starts at START and moves one minute per reading."""
    moments = (START + timedelta(minutes=n) for n in range(10_000))
    return lambda: next(moments)


def record_experiments(ledger: Ledger, *experiment_ids: str, lane: Lane.ValueType) -> None:
    for experiment_id in experiment_ids:
        ledger.record_experiment(
            ExperimentRecord(
                experiment_id=experiment_id,
                lane=lane,
                round=1,
                decision=DECISION_PENDING,
                hypothesis=Hypothesis(hypothesis_id=f"h-{experiment_id}"),
            )
        )


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    """A ledger that holds the experiments the lessons below cite."""
    ledger = Ledger(tmp_path / "ledger")
    record_experiments(ledger, "e1", "e2", "e3", "e4", "e9", lane=LANE_EFFISQL)
    record_experiments(ledger, "s0", lane=LANE_SWE)
    return ledger


@pytest.fixture
def store(ledger: Ledger) -> LessonStore:
    return LessonStore(ledger, clock=ticking_clock())


def jsonl_lines(ledger: Ledger) -> int:
    return len((ledger.directory / "lessons.jsonl").read_text(encoding="utf-8").splitlines())


def markdown(ledger: Ledger) -> str:
    return (ledger.directory / "lessons.md").read_text(encoding="utf-8")


# Adding lessons.


def test_an_added_lesson_is_stored_with_its_times(store: LessonStore, ledger: Ledger) -> None:
    added = store.add("l1", LANE_EFFISQL, "  Read the plan first.\n", ["e1", "e2"])

    assert added == Lesson(
        lesson_id="l1",
        lane=LANE_EFFISQL,
        text="Read the plan first.",
        experiment_ids=["e1", "e2"],
        created_at=timestamp(START),
        updated_at=timestamp(START),
    )
    assert store.lessons() == [added]
    assert store.get("l1") == added
    assert jsonl_lines(ledger) == 1


def test_the_default_clock_records_the_current_utc_time(ledger: Ledger) -> None:
    before = datetime.now(UTC)
    lesson = LessonStore(ledger).add("l1", LANE_EFFISQL, "text")
    after = datetime.now(UTC)

    assert before <= lesson.created_at.ToDatetime(tzinfo=UTC) <= after


def test_lessons_keep_the_order_they_were_added_in(store: LessonStore) -> None:
    store.add("b", LANE_EFFISQL, "second id, added first")
    store.add("a", LANE_SWE, "first id, added second")
    store.add("c", LANE_EFFISQL, "third")

    assert [lesson.lesson_id for lesson in store.lessons()] == ["b", "a", "c"]
    assert [lesson.lesson_id for lesson in store.lessons(LANE_EFFISQL)] == ["b", "c"]


def test_an_unknown_lesson_is_none(store: LessonStore) -> None:
    assert store.get("nope") is None
    assert store.lessons() == []


@pytest.mark.parametrize(
    ("lesson_id", "lane", "text", "experiment_ids", "problem"),
    [
        ("", LANE_EFFISQL, "text", [], "lesson_id"),
        ("a b", LANE_EFFISQL, "text", [], "lesson_id"),
        ("l1", LANE_UNSPECIFIED, "text", [], "lane"),
        ("l1", LANE_EFFISQL, " \n ", [], "text"),
        ("l1", LANE_EFFISQL, "text", ["e1", "bad id"], "experiment_id"),
        # A lesson cites only experiments of its own lane that the ledger holds.
        ("l1", LANE_EFFISQL, "text", ["e1", "e404"], "e404"),
        ("l1", LANE_SWE, "text", ["e1"], "e1"),
    ],
)
def test_an_invalid_lesson_is_refused_and_nothing_is_written(
    store: LessonStore,
    ledger: Ledger,
    lesson_id: str,
    lane: Lane.ValueType,
    text: str,
    experiment_ids: list[str],
    problem: str,
) -> None:
    with pytest.raises(ValueError, match=problem):
        store.add(lesson_id, lane, text, experiment_ids)

    assert not (ledger.directory / "lessons.jsonl").exists()


def test_a_lesson_id_is_never_reused(store: LessonStore, ledger: Ledger) -> None:
    store.add("l1", LANE_EFFISQL, "first")

    with pytest.raises(ValueError, match="l1"):
        store.add("l1", LANE_EFFISQL, "second")

    assert jsonl_lines(ledger) == 1


def test_repeated_evidence_is_kept_once(store: LessonStore) -> None:
    lesson = store.add("l1", LANE_EFFISQL, "text", ["e1", "e2", "e1"])

    assert list(lesson.experiment_ids) == ["e1", "e2"]


def test_a_failed_render_stores_nothing(
    store: LessonStore, ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(lessons: object) -> str:
        raise RuntimeError("cannot render")

    monkeypatch.setattr(lessons, "render_markdown", fail)

    with pytest.raises(RuntimeError, match="cannot render"):
        store.add("l1", LANE_EFFISQL, "text")

    assert not (ledger.directory / "lessons.jsonl").exists()


# Helped and hurt counters.


def test_marking_appends_the_lessons_new_state(store: LessonStore, ledger: Ledger) -> None:
    store.add("l1", LANE_EFFISQL, "text", ["e1"])

    helped = store.mark_helped("l1", "e2")
    hurt = store.mark_hurt("l1", "e3")

    assert list(helped.helped_by) == ["e2"]
    assert list(hurt.helped_by) == ["e2"]
    assert list(hurt.hurt_by) == ["e3"]
    assert hurt.created_at == timestamp(START)
    assert hurt.updated_at == timestamp(START + timedelta(minutes=2))
    assert store.get("l1") == hurt
    assert jsonl_lines(ledger) == 3  # Every change is one more line; nothing is rewritten.


def test_marking_an_experiment_again_changes_nothing(store: LessonStore, ledger: Ledger) -> None:
    store.add("l1", LANE_EFFISQL, "text")
    first = store.mark_helped("l1", "e2")

    again = store.mark_helped("l1", "e2")

    assert again == first
    assert jsonl_lines(ledger) == 2


def test_a_retried_mark_regenerates_lessons_md(store: LessonStore, ledger: Ledger) -> None:
    store.add("l1", LANE_EFFISQL, "text")
    store.mark_helped("l1", "e2")
    (ledger.directory / "lessons.md").unlink()  # As if the lane crashed before writing it.

    store.mark_helped("l1", "e2")

    assert "`l1`: helped 1, hurt 0" in markdown(ledger)


def test_an_experiment_cannot_both_help_and_hurt_a_lesson(store: LessonStore) -> None:
    store.add("l1", LANE_EFFISQL, "text")
    store.mark_helped("l1", "e2")

    with pytest.raises(ValueError, match="e2"):
        store.mark_hurt("l1", "e2")


def test_marking_an_unknown_lesson_is_an_error(store: LessonStore) -> None:
    with pytest.raises(LessonNotFoundError, match="nope"):
        store.mark_helped("nope", "e1")


def test_marking_needs_a_valid_experiment_id(store: LessonStore) -> None:
    store.add("l1", LANE_EFFISQL, "text")

    with pytest.raises(ValueError, match="experiment_id"):
        store.mark_hurt("l1", "not valid")


@pytest.mark.parametrize("experiment_id", ["e404", "s0"])
def test_marking_needs_an_experiment_of_the_lessons_lane(
    store: LessonStore, ledger: Ledger, experiment_id: str
) -> None:
    store.add("l1", LANE_EFFISQL, "text")

    with pytest.raises(ValueError, match=experiment_id):
        store.mark_helped("l1", experiment_id)

    assert jsonl_lines(ledger) == 1


# Deleting lessons.


def test_a_deleted_lesson_is_skipped_but_its_lines_stay(store: LessonStore, ledger: Ledger) -> None:
    store.add("l1", LANE_EFFISQL, "Keep the identity fallback.", ["e1"])
    store.add("l2", LANE_EFFISQL, "Profile first.")
    store.mark_helped("l1", "e2")

    store.delete("l1")

    assert [lesson.lesson_id for lesson in store.lessons()] == ["l2"]
    assert store.get("l1") is None
    assert "`l1`" not in markdown(ledger)
    # Nothing is lost: the deletion is one more line, holding the lesson's last state.
    stored = (ledger.directory / "lessons.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(stored) == 4
    deleted = Lesson()
    json_format.Parse(stored[-1], deleted)
    assert deleted.text == "Keep the identity fallback."
    assert list(deleted.helped_by) == ["e2"]
    assert deleted.deleted_at == timestamp(START + timedelta(minutes=3))
    assert deleted.updated_at == deleted.deleted_at


def test_deleting_a_lesson_again_changes_nothing(store: LessonStore, ledger: Ledger) -> None:
    store.add("l1", LANE_EFFISQL, "text")
    store.delete("l1")

    store.delete("l1")

    assert jsonl_lines(ledger) == 2


def test_a_retried_delete_regenerates_lessons_md(store: LessonStore, ledger: Ledger) -> None:
    store.add("l1", LANE_EFFISQL, "text")
    store.delete("l1")
    (ledger.directory / "lessons.md").unlink()  # As if the lane crashed before writing it.

    store.delete("l1")

    assert "No lessons yet." in markdown(ledger)


def test_deleting_an_unknown_lesson_is_an_error(store: LessonStore) -> None:
    with pytest.raises(LessonNotFoundError, match="nope"):
        store.delete("nope")


def test_a_deleted_lesson_cannot_be_marked(store: LessonStore) -> None:
    store.add("l1", LANE_EFFISQL, "text")
    store.delete("l1")

    with pytest.raises(LessonNotFoundError, match="deleted"):
        store.mark_helped("l1", "e2")


def test_a_deleted_lessons_id_is_not_reused(store: LessonStore) -> None:
    store.add("l1", LANE_EFFISQL, "text")
    store.delete("l1")

    with pytest.raises(ValueError, match="l1"):
        store.add("l1", LANE_EFFISQL, "new text")


def test_render_markdown_skips_deleted_lessons() -> None:
    deleted = Lesson()
    deleted.CopyFrom(sample_lessons()[0])
    deleted.deleted_at.CopyFrom(timestamp(START))

    assert render_markdown([*sample_lessons()[1:], deleted]) == render_markdown(
        sample_lessons()[1:]
    )


# Marks argv[3] experiments, named after argv[2], as having helped lesson l1.
MARKER = """
import sys
from pathlib import Path
from goaly.rsi.ledger import Ledger
from goaly.rsi.lessons import LessonStore

store = LessonStore(Ledger(Path(sys.argv[1])))
sys.stdin.read()
for n in range(int(sys.argv[3])):
    store.mark_helped("l1", f"{sys.argv[2]}-{n}")
"""


def test_two_processes_marking_one_lesson_lose_no_update(
    store: LessonStore, ledger: Ledger
) -> None:
    count = 40
    marked = [f"{prefix}-{n}" for prefix in ("a", "b") for n in range(count)]
    record_experiments(ledger, *marked, lane=LANE_EFFISQL)
    store.add("l1", LANE_EFFISQL, "text")
    markers = [
        subprocess.Popen(  # noqa: S603  (a fixed script and arguments built here)
            [sys.executable, "-c", MARKER, str(ledger.directory), prefix, str(count)],
            stdin=subprocess.PIPE,
        )
        for prefix in ("a", "b")
    ]
    for marker in markers:
        assert marker.stdin is not None
        marker.stdin.close()
    for marker in markers:
        assert marker.wait(timeout=120) == 0

    lesson = store.get("l1")

    assert lesson is not None
    assert sorted(lesson.helped_by) == sorted(marked)


# lessons.md.

EXPECTED_MARKDOWN = """\
# RSI lessons

Generated from lessons.jsonl by goaly.rsi; do not edit.

## Effi-SQL

### `l2`: helped 1, hurt 1

> Profile the base query first.
>
> Then rewrite the slowest join.

- Cites: `e1`, `e2`
- Helped: `e3`
- Hurt: `e4`
- Added 2026-10-15 12:00 UTC, updated 2026-10-15 12:30 UTC

### `l1`: helped 0, hurt 0

> Keep the identity fallback.

- Cites: none
- Helped: none
- Hurt: none
- Added 2026-10-15 12:05 UTC, updated 2026-10-15 12:05 UTC

## SWE-fficiency

### `s1`: helped 0, hurt 0

> Run the workload twice.

- Cites: `s0`
- Helped: none
- Hurt: none
- Added 2026-10-15 11:00 UTC, updated 2026-10-15 11:00 UTC
"""


def sample_lessons() -> list[Lesson]:
    return [
        Lesson(
            lesson_id="s1",
            lane=LANE_SWE,
            text="Run the workload twice.",
            experiment_ids=["s0"],
            created_at=timestamp(START - timedelta(hours=1)),
            updated_at=timestamp(START - timedelta(hours=1)),
        ),
        Lesson(
            lesson_id="l1",
            lane=LANE_EFFISQL,
            text="Keep the identity fallback.",
            created_at=timestamp(START + timedelta(minutes=5)),
            updated_at=timestamp(START + timedelta(minutes=5)),
        ),
        Lesson(
            lesson_id="l2",
            lane=LANE_EFFISQL,
            text="Profile the base query first.  \n\nThen rewrite the slowest join.",
            experiment_ids=["e1", "e2"],
            helped_by=["e3"],
            hurt_by=["e4"],
            created_at=timestamp(START),
            updated_at=timestamp(START + timedelta(minutes=30)),
        ),
    ]


def test_render_markdown_groups_lessons_by_lane_oldest_first() -> None:
    assert render_markdown(sample_lessons()) == EXPECTED_MARKDOWN


def test_render_markdown_does_not_depend_on_the_input_order() -> None:
    assert render_markdown(reversed(sample_lessons())) == EXPECTED_MARKDOWN


def test_render_markdown_with_no_lessons() -> None:
    assert render_markdown([]) == (
        "# RSI lessons\n\nGenerated from lessons.jsonl by goaly.rsi; do not edit.\n\n"
        "No lessons yet.\n"
    )


def test_render_markdown_names_a_lane_it_does_not_know_by_number() -> None:
    unknown = Lesson(lesson_id="x1", lane=Lane.ValueType(99), text="From a newer lane.")

    assert "## Lane 99\n\n### `x1`: helped 0, hurt 0" in render_markdown([unknown])


def test_every_write_regenerates_lessons_md(store: LessonStore, ledger: Ledger) -> None:
    store.add("l1", LANE_EFFISQL, "Keep the identity fallback.")
    assert markdown(ledger) == render_markdown(store.lessons())

    store.mark_helped("l1", "e9")
    assert "helped 1, hurt 0" in markdown(ledger)
    assert markdown(ledger) == render_markdown(store.lessons())


def test_lessons_md_is_synced_and_stale_temporary_files_are_cleared(
    store: LessonStore, ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = ledger.directory / ".lessons.md.0123456789abcdef.tmp"
    stale.write_text("left by a crash\n", encoding="utf-8")
    synced: list[str] = []
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        synced.append(str(Path(f"/proc/self/fd/{descriptor}").readlink()))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)

    store.add("l1", LANE_EFFISQL, "text")

    assert not stale.exists()
    temporary = [n for n, path in enumerate(synced) if re.search(r"/\.lessons\.md\.\w+\.tmp", path)]
    assert len(temporary) == 1
    # The new page is synced before it replaces the old one, and the directory after.
    assert synced[temporary[0] + 1 :] == [str(ledger.directory)]


def test_write_markdown_rebuilds_a_missing_page(store: LessonStore, ledger: Ledger) -> None:
    store.add("l1", LANE_EFFISQL, "text")
    (ledger.directory / "lessons.md").unlink()

    store.write_markdown()

    assert markdown(ledger) == render_markdown(store.lessons())
