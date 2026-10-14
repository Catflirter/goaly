"""Structured logging: every record becomes one line of JSON.

Call :func:`configure_logging` once at startup. Then log with the standard library and pass
structured fields through ``extra``::

    logging.getLogger(__name__).info("task solved", extra={"task_id": task_id, "steps": steps})
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TextIO

__all__ = ["JsonFormatter", "configure_logging"]

# Every LogRecord has these attributes (plus `message` and `asctime` once a Formatter has run);
# any other attribute on a record came from `extra=`.
_RECORD_ATTRIBUTES = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

# Marks the handler that configure_logging installs, so a second call can replace it.
_HANDLER_NAME = "goaly-json"


class JsonFormatter(logging.Formatter):
    """Format a record as a single-line JSON object.

    Keys, in order: ``timestamp`` (UTC, RFC 3339, milliseconds), ``level``, ``logger``,
    ``message``, then each ``extra`` field, then ``exception`` and ``stack`` when present.
    The formatter's own keys win: an extra with the same name is dropped. Values JSON cannot
    encode are rendered with ``str()``.
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

    Installs one handler on the root logger. A later call replaces it, and handlers installed
    by anyone else are left alone. stderr is the default because stdout is for command output.
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
    root.addHandler(handler)


def _utc_timestamp(created: float) -> str:
    moment = datetime.fromtimestamp(created, tz=UTC)
    return moment.isoformat(timespec="milliseconds").removesuffix("+00:00") + "Z"


def _to_json(entry: dict[str, object]) -> str:
    try:
        return _dumps(entry)
    except (TypeError, ValueError):
        # A value holds a non-string dict key, NaN or infinity, or a reference cycle.
        return _dumps({key: _encodable(value) for key, value in entry.items()})


def _encodable(value: object) -> object:
    try:
        _dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _dumps(value: object) -> str:
    # default=str renders unknown types; allow_nan=False rejects NaN and infinity, which are not
    # JSON. Newlines and non-ASCII text are escaped, so the output is always a single line.
    return json.dumps(value, default=str, allow_nan=False)
