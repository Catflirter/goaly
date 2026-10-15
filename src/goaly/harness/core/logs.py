"""Structured logging: every record becomes one line of JSON.

Call :func:`configure_logging` once at startup. Then log with the standard library and pass
structured fields through ``extra``::

    logging.getLogger(__name__).info("task solved", extra={"task_id": task_id, "steps": steps})

Fields that every record of a stretch of work shares, such as the run and the task that a
runner's task works on, go in a :func:`log_context` instead: every record logged inside it gets
them, from any code, in the asyncio tasks it starts and the threads that asyncio.to_thread runs.
"""

import json
import logging
import math
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TextIO

__all__ = ["ContextFilter", "JsonFormatter", "configure_logging", "log_context"]

# Every LogRecord has these attributes (plus `message` and `asctime` once a Formatter has run);
# any other attribute on a record came from `extra=`.
_RECORD_ATTRIBUTES = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

# Marks the handler that configure_logging installs, so a second call can replace it.
_HANDLER_NAME = "goaly-json"

# The fields of the current log context. A task or a thread started inside one copies it.
_CONTEXT: ContextVar[Mapping[str, object]] = ContextVar(
    "goaly_log_context", default=MappingProxyType({})
)


@contextmanager
def log_context(**fields: object) -> Iterator[None]:
    """Add ``fields`` to every record logged inside the block, through ContextFilter.

    The fields reach the asyncio tasks started inside the block, and the threads that
    asyncio.to_thread runs, for as long as those run. A nested context adds its fields to the
    outer one's, and wins on a name both have.
    """
    token = _CONTEXT.set(MappingProxyType({**_CONTEXT.get(), **fields}))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


class ContextFilter(logging.Filter):
    """Add the current log context's fields to each record it sees.

    A field the record has already, such as one logged through ``extra``, is kept as it is.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for name, value in _CONTEXT.get().items():
            if not hasattr(record, name):
                setattr(record, name, value)
        return True


class JsonFormatter(logging.Formatter):
    """Format a record as a single-line JSON object.

    Keys, in order: ``timestamp`` (UTC, RFC 3339, milliseconds), ``level``, ``logger``,
    ``message``, then each ``extra`` field, then ``exception`` and ``stack`` when present.
    The formatter's own keys win: an extra with the same name is dropped. Keys that are not
    strings are rendered with ``str()`` at every level, and so are values JSON cannot encode.
    """

    def format(self, record: logging.LogRecord) -> str:
        if record.exc_info and not record.exc_text:
            # Cache the traceback on the record, as logging.Formatter does.
            record.exc_text = self.formatException(record.exc_info)
        trailer: dict[str, object] = {}
        if record.exc_text:
            trailer["exception"] = record.exc_text
        if record.stack_info:
            trailer["stack"] = self.formatStack(record.stack_info)

        entry: dict[str, object] = {
            "timestamp": _utc_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RECORD_ATTRIBUTES and key not in entry and key not in trailer:
                entry[key] = value
        entry.update(trailer)
        return _to_json(entry)


def configure_logging(level: int | str = "INFO", *, stream: TextIO | None = None) -> None:
    """Send log records at ``level`` and above to ``stream`` (default: stderr) as JSON lines.

    Installs one handler on the root logger, with the log context's fields on each record. A
    later call replaces it, and handlers installed by anyone else are left alone. stderr is the
    default because stdout is for command output.
    """
    root = logging.getLogger()
    root.setLevel(level)  # First, so that an unknown level name changes nothing.
    for handler in list(root.handlers):
        if handler.name == _HANDLER_NAME:
            root.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.name = _HANDLER_NAME
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextFilter())
    root.addHandler(handler)


def _utc_timestamp(created: float) -> str:
    moment = datetime.fromtimestamp(created, tz=UTC)
    return moment.isoformat(timespec="milliseconds").removesuffix("+00:00") + "Z"


def _to_json(entry: dict[str, object]) -> str:
    try:
        return _dumps(entry)
    except (TypeError, ValueError):
        # A key that is not a string, NaN or infinity, or a reference cycle, at any depth.
        return _dumps(_jsonable(entry, frozenset()))


def _jsonable(value: object, seen: frozenset[int]) -> object:
    """Copy ``value`` into something json encodes as is; ``seen`` stops reference cycles."""
    if value is None or isinstance(value, str | int):  # bool is an int.
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict | list | tuple) and id(value) not in seen:
        inside = seen | {id(value)}
        if isinstance(value, dict):
            return {
                key if isinstance(key, str) else str(key): _jsonable(item, inside)
                for key, item in value.items()
            }
        return [_jsonable(item, inside) for item in value]
    return str(value)


def _dumps(value: object) -> str:
    # default=str renders unknown types; allow_nan=False rejects NaN and infinity, which are not
    # JSON. Newlines and non-ASCII text are escaped, so the output is always a single line.
    return json.dumps(value, default=str, allow_nan=False)
