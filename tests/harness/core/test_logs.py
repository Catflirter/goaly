"""Tests for goaly.harness.core.logs."""

import io
import json
import logging
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from goaly.harness.core.logs import JsonFormatter, configure_logging

CORE_KEYS = ["timestamp", "level", "logger", "message"]


def reject_constant(name: str) -> None:
    raise ValueError(f"not valid JSON: {name}")


def parse_line(text: str) -> dict[str, Any]:
    """Parse one rendered record strictly: a single line holding one JSON object."""
    assert text.endswith("\n")
    assert len(text.splitlines()) == 1  # No line boundary of any kind, Unicode ones included.
    entry = json.loads(text, parse_constant=reject_constant)
    assert isinstance(entry, dict)
    return entry


@dataclass
class Capture:
    logger: logging.Logger
    stream: io.StringIO

    def entry(self) -> dict[str, Any]:
        return parse_line(self.stream.getvalue())


@pytest.fixture
def capture() -> Iterator[Capture]:
    """A logger that writes only to a StringIO, through JsonFormatter."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("goaly.tests.logs")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield Capture(logger, stream)
    logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


# JsonFormatter.


def test_starts_with_the_core_keys_in_order(capture: Capture) -> None:
    capture.logger.warning("disk %s is %d%% full", "/data", 91)

    entry = capture.entry()

    assert list(entry) == CORE_KEYS
    assert entry["level"] == "WARNING"
    assert entry["logger"] == "goaly.tests.logs"
    assert entry["message"] == "disk /data is 91% full"


def test_timestamp_is_utc_rfc3339_with_milliseconds() -> None:
    moment = datetime(2026, 10, 14, 21, 56, 12, 345678, tzinfo=UTC)
    record = logging.makeLogRecord({"msg": "tick", "created": moment.timestamp()})

    entry = parse_line(JsonFormatter().format(record) + "\n")

    assert entry["timestamp"] == "2026-10-14T21:56:12.345Z"


def test_live_timestamps_match_the_format(capture: Capture) -> None:
    capture.logger.info("tick")

    timestamp = capture.entry()["timestamp"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", timestamp)


def test_extras_follow_the_core_keys(capture: Capture) -> None:
    capture.logger.info("solved", extra={"task_id": "effisql-17", "steps": 12, "ok": True})

    entry = capture.entry()

    assert list(entry) == [*CORE_KEYS, "task_id", "steps", "ok"]
    assert (entry["task_id"], entry["steps"], entry["ok"]) == ("effisql-17", 12, True)


def test_extras_never_overwrite_the_core_keys(capture: Capture) -> None:
    capture.logger.info("real", extra={"timestamp": "fake", "level": "fake", "logger": "fake"})

    entry = capture.entry()

    assert entry["timestamp"] != "fake"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "goaly.tests.logs"
    assert entry["message"] == "real"


def test_renders_values_json_cannot_encode_with_str(capture: Capture) -> None:
    looped: list[object] = []
    looped.append(looped)
    marker = object()
    extra: dict[Any, object] = {
        "path": Path("/srv/runs"),
        "marker": marker,
        "score": math.nan,
        "limit": math.inf,
        "pairs": {(1, 2): "tuple key"},
        "nested": {"inner": {(3, 4): "deep"}},
        "looped": looped,
        "fine": [1, "two", None],
        (5, 6): "top-level tuple key",
    }

    capture.logger.info("odd values", extra=extra)

    entry = capture.entry()
    assert entry["path"] == "/srv/runs"
    assert entry["marker"] == str(marker)
    assert entry["score"] == "nan"
    assert entry["limit"] == "inf"
    assert entry["pairs"] == {"(1, 2)": "tuple key"}
    assert entry["nested"] == {"inner": {"(3, 4)": "deep"}}
    assert entry["looped"] == ["[[...]]"]
    assert entry["fine"] == [1, "two", None]
    assert entry["(5, 6)"] == "top-level tuple key"


@pytest.mark.parametrize(
    "message",
    ['she said "hi"', "first line\nsecond line\r\n", "tab\there", "naïve 日本語 \u2028 ✓"],
)
def test_any_message_stays_one_valid_json_line(capture: Capture, message: str) -> None:
    capture.logger.info(message)

    assert capture.entry()["message"] == message


def test_adds_the_exception_after_the_extras(capture: Capture) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        capture.logger.exception("step failed", extra={"step": 3})

    entry = capture.entry()

    assert list(entry) == [*CORE_KEYS, "step", "exception"]
    assert entry["exception"].startswith("Traceback (most recent call last):")
    assert entry["exception"].endswith("ValueError: boom")


def test_adds_the_stack_last(capture: Capture) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        capture.logger.exception("step failed", stack_info=True)

    entry = capture.entry()

    assert list(entry) == [*CORE_KEYS, "exception", "stack"]
    assert entry["stack"].startswith("Stack (most recent call last):")


def test_exception_and_stack_always_come_last(capture: Capture) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        capture.logger.exception("failed", extra={"exception": "mine", "after": 1})
    capture.logger.info("no stack requested", extra={"stack": "blue"})

    failed, plain = (parse_line(line + "\n") for line in capture.stream.getvalue().splitlines())

    assert list(failed) == [*CORE_KEYS, "after", "exception"]
    assert failed["exception"].endswith("ValueError: boom")
    assert plain["stack"] == "blue"


# configure_logging.


@pytest.fixture
def root() -> Iterator[logging.Logger]:
    """The root logger, with its handlers and level restored after the test."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    yield root
    for handler in list(root.handlers):
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_level)


def json_handlers(root: logging.Logger) -> list[logging.Handler]:
    return [handler for handler in root.handlers if isinstance(handler.formatter, JsonFormatter)]


def test_configure_logging_writes_json_lines_to_the_stream(root: logging.Logger) -> None:
    stream = io.StringIO()

    configure_logging(stream=stream)
    logging.getLogger("goaly.bench.runner").info("started", extra={"tasks": 5})

    entry = parse_line(stream.getvalue())
    assert entry["logger"] == "goaly.bench.runner"
    assert entry["message"] == "started"
    assert entry["tasks"] == 5


def test_configure_logging_defaults_to_stderr(
    root: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging()
    logging.getLogger("goaly").info("to stderr")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert parse_line(captured.err)["message"] == "to stderr"


@pytest.mark.parametrize("level", ["WARNING", logging.WARNING])
def test_configure_logging_sets_the_root_level(root: logging.Logger, level: int | str) -> None:
    stream = io.StringIO()

    configure_logging(level, stream=stream)
    logging.getLogger("goaly").info("dropped")
    logging.getLogger("goaly").warning("kept")

    assert root.level == logging.WARNING
    assert parse_line(stream.getvalue())["message"] == "kept"


def test_configure_logging_replaces_its_own_handler(root: logging.Logger) -> None:
    first, second = io.StringIO(), io.StringIO()

    configure_logging(stream=first)
    configure_logging(stream=second)
    logging.getLogger("goaly").info("once")

    assert len(json_handlers(root)) == 1
    assert first.getvalue() == ""
    assert parse_line(second.getvalue())["message"] == "once"


def test_configure_logging_rejects_an_unknown_level_and_changes_nothing(
    root: logging.Logger,
) -> None:
    configure_logging(stream=io.StringIO())
    handlers, level = list(root.handlers), root.level

    with pytest.raises(ValueError, match="NOPE"):
        configure_logging("NOPE")

    assert root.handlers == handlers
    assert root.level == level


def test_configure_logging_leaves_other_handlers_alone(root: logging.Logger) -> None:
    other = logging.NullHandler()
    root.addHandler(other)

    configure_logging(stream=io.StringIO())
    configure_logging(stream=io.StringIO())

    assert other in root.handlers
    assert len(json_handlers(root)) == 1
