"""The lessons store (spec §8.6): lessons.jsonl in the ledger, and lessons.md generated from it.

lessons.jsonl is the source of truth. Lessons are added one at a time, as in ACE, and each one
counts the experiments in which it helped or hurt. Every change appends the lesson's full new
state as one line, under the ledger's lock, and the last line for a lesson_id wins. A deletion
too is such a line, with deleted_at set: no line is ever rewritten, so nothing is lost by
accident. lessons.md, for people to read, is regenerated after every change; counters kept in
hand-edited Markdown could not be updated reliably.
"""

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from itertools import groupby

from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.experiment_pb2 import Lane, Lesson
from goaly.rsi.lanes import lane_info
from goaly.rsi.ledger import LESSONS, LESSONS_MARKDOWN, Ledger, LedgerFiles, check_id

__all__ = ["LessonNotFoundError", "LessonStore", "render_markdown"]


class LessonNotFoundError(LookupError):
    """No standing lesson has the ID: there is none, or it was deleted."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class LessonStore:
    """The lessons in ``ledger``; ``clock`` gives the times for created_at and updated_at."""

    def __init__(self, ledger: Ledger, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._ledger = ledger
        self._clock = clock

    def add(
        self,
        lesson_id: str,
        lane: Lane.ValueType,
        text: str,
        experiment_ids: Iterable[str] = (),
    ) -> Lesson:
        """Add a lesson that cites ``experiment_ids`` as its evidence, and return it.

        The text is stripped, and evidence cited twice is kept once. Raises ValueError, writing
        nothing, for an invalid ID or lane, a blank text, or a lesson_id already used, even by a
        deleted lesson.
        """
        check_id(lesson_id, "lesson_id")
        lane_info(lane)
        text = text.strip()
        if not text:
            raise ValueError(f"lesson {lesson_id}: the text must not be blank")
        evidence = list(dict.fromkeys(experiment_ids))
        for experiment_id in evidence:
            check_id(experiment_id, "experiment_id")
        with self._ledger.session(write=True) as files:
            lessons = _latest(files)
            if lesson_id in lessons:
                raise ValueError(f"lesson_id {lesson_id} is already used; IDs are never reused")
            now = self._now()
            lesson = Lesson(
                lesson_id=lesson_id,
                lane=lane,
                text=text,
                experiment_ids=evidence,
                created_at=now,
                updated_at=now,
            )
            _write(files, lessons, lesson)
        return lesson

    def mark_helped(self, lesson_id: str, experiment_id: str) -> Lesson:
        """Record that the lesson helped in ``experiment_id``, and return the lesson.

        Recording an experiment again changes nothing, so a resumed lane cannot count it twice.
        Raises LessonNotFoundError for an unknown or deleted lesson, and ValueError if the
        experiment already hurt it.
        """
        return self._mark(lesson_id, experiment_id, helped=True)

    def mark_hurt(self, lesson_id: str, experiment_id: str) -> Lesson:
        """Record that the lesson hurt in ``experiment_id``, as :meth:`mark_helped` does."""
        return self._mark(lesson_id, experiment_id, helped=False)

    def delete(self, lesson_id: str) -> None:
        """Delete the lesson by appending its last state with deleted_at set.

        Its earlier lines stay, and deleting it again changes nothing. Raises
        LessonNotFoundError if no lesson has the ID.
        """
        with self._ledger.session(write=True) as files:
            lessons = _latest(files)
            if lesson_id not in lessons:
                raise LessonNotFoundError(f"no lesson has the ID {lesson_id}")
            lesson = lessons[lesson_id]
            if not _stands(lesson):
                return
            deleted = Lesson()
            deleted.CopyFrom(lesson)
            now = self._now()
            deleted.updated_at.CopyFrom(now)
            deleted.deleted_at.CopyFrom(now)
            _write(files, lessons, deleted)

    def lessons(self, lane: Lane.ValueType | None = None) -> list[Lesson]:
        """Return the standing lessons, of one ``lane`` or of all, in the order they were added."""
        with self._ledger.session() as files:
            lessons = _latest(files)
        return [
            lesson for lesson in lessons.values() if _stands(lesson) and lane in (None, lesson.lane)
        ]

    def get(self, lesson_id: str) -> Lesson | None:
        """Return the standing lesson with ``lesson_id``, or None if there is none."""
        with self._ledger.session() as files:
            lesson = _latest(files).get(lesson_id)
        return lesson if lesson is not None and _stands(lesson) else None

    def write_markdown(self) -> None:
        """Regenerate lessons.md from lessons.jsonl, as every change already does."""
        with self._ledger.session(write=True) as files:
            files.replace_text(LESSONS_MARKDOWN, render_markdown(_latest(files).values()))

    def _mark(self, lesson_id: str, experiment_id: str, *, helped: bool) -> Lesson:
        check_id(experiment_id, "experiment_id")
        with self._ledger.session(write=True) as files:
            lessons = _latest(files)
            if lesson_id not in lessons:
                raise LessonNotFoundError(f"no lesson has the ID {lesson_id}")
            lesson = lessons[lesson_id]
            if not _stands(lesson):
                raise LessonNotFoundError(f"lesson {lesson_id} was deleted")
            marked, other = (
                (lesson.helped_by, lesson.hurt_by) if helped else (lesson.hurt_by, lesson.helped_by)
            )
            if experiment_id in marked:
                return lesson
            if experiment_id in other:
                raise ValueError(
                    f"experiment {experiment_id} already {'hurt' if helped else 'helped'} "
                    f"lesson {lesson_id}; it cannot both help and hurt it"
                )
            changed = Lesson()
            changed.CopyFrom(lesson)
            (changed.helped_by if helped else changed.hurt_by).append(experiment_id)
            changed.updated_at.CopyFrom(self._now())
            _write(files, lessons, changed)
        return changed

    def _now(self) -> Timestamp:
        now = Timestamp()
        now.FromDatetime(self._clock())
        return now


def _latest(files: LedgerFiles) -> dict[str, Lesson]:
    """Each lesson's latest state, keyed by lesson_id in the order the lessons were added."""
    lessons: dict[str, Lesson] = {}
    for lesson in files.read(LESSONS, Lesson):
        lessons[lesson.lesson_id] = lesson  # Replacing a value keeps its key's place.
    return lessons


def _stands(lesson: Lesson) -> bool:
    return not lesson.HasField("deleted_at")


def _write(files: LedgerFiles, lessons: dict[str, Lesson], lesson: Lesson) -> None:
    """Append ``lesson``'s new state, then regenerate lessons.md from every lesson's latest."""
    files.append(LESSONS, lesson)
    lessons[lesson.lesson_id] = lesson
    files.replace_text(LESSONS_MARKDOWN, render_markdown(lessons.values()))


def render_markdown(lessons: Iterable[Lesson]) -> str:
    """Render ``lessons`` as lessons.md: grouped by lane, oldest first, deleted ones left out.

    The text depends only on the lessons given, not on their order.
    """
    blocks = ["# RSI lessons", "Generated from lessons.jsonl by goaly.rsi; do not edit."]
    ordered = sorted(
        filter(_stands, lessons),
        key=lambda lesson: (
            lesson.lane,
            lesson.created_at.seconds,
            lesson.created_at.nanos,
            lesson.lesson_id,
        ),
    )
    if not ordered:
        blocks.append("No lessons yet.")
    for lane, group in groupby(ordered, key=lambda lesson: lesson.lane):
        blocks.append(f"## {lane_info(lane).title}")
        for lesson in group:
            blocks.extend(_render_lesson(lesson))
    return "\n\n".join(blocks) + "\n"


def _render_lesson(lesson: Lesson) -> list[str]:
    heading = (
        f"### `{lesson.lesson_id}`: helped {len(lesson.helped_by)}, hurt {len(lesson.hurt_by)}"
    )
    quoted = "\n".join(
        f"> {line}" if line else ">" for line in (raw.rstrip() for raw in lesson.text.splitlines())
    )
    details = "\n".join(
        [
            f"- Cites: {_ids(lesson.experiment_ids)}",
            f"- Helped: {_ids(lesson.helped_by)}",
            f"- Hurt: {_ids(lesson.hurt_by)}",
            f"- Added {_when(lesson.created_at)}, updated {_when(lesson.updated_at)}",
        ]
    )
    return [heading, quoted, details]


def _ids(values: Iterable[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) or "none"


def _when(moment: Timestamp) -> str:
    return moment.ToDatetime(tzinfo=UTC).strftime("%Y-%m-%d %H:%M UTC")
