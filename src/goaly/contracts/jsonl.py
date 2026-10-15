"""Read and write protobuf messages as JSON Lines (spec §3.3).

Each line holds one message in protobuf's standard JSON mapping, under the field names of the
.proto files. Fields without presence are always written, so a default such as
``"correct": false`` shows instead of disappearing. Reading ignores unknown fields, which keeps
files written before a field was removed (and reserved) readable.

The helpers neither lock nor fsync: a file must have one writer at a time (spec §6.2).
"""

import json
import secrets
from collections.abc import Iterable, Iterator
from pathlib import Path

from google.protobuf import json_format
from google.protobuf.message import Message

__all__ = [
    "JsonlError",
    "append_jsonl",
    "from_json_line",
    "read_jsonl",
    "to_json_line",
    "write_jsonl",
]

_JSON_WHITESPACE = " \t\n\r"


class JsonlError(ValueError):
    """A line of a JSONL file holds no valid message; the error text starts ``path:line:``."""

    def __init__(self, path: Path, line_number: int, reason: str) -> None:
        super().__init__(f"{path}:{line_number}: {reason}")
        self.path = path
        self.line_number = line_number


def to_json_line(message: Message) -> str:
    """Write ``message`` as one line of compact JSON, with no line break at the end.

    Raises ``json_format.SerializeToJsonError`` for a value JSON cannot hold, such as a NaN in a
    ``google.protobuf.Value``.
    """
    fields = json_format.MessageToDict(
        message, always_print_fields_with_no_presence=True, preserving_proto_field_name=True
    )
    # json escapes every control character, so the line holds no "\n". Other text stays as it
    # is, U+2028 and U+2029 included, so readers must split lines at "\n" alone.
    return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def from_json_line[M: Message](line: str, message_type: type[M]) -> M:
    """Read a line written by :func:`to_json_line` as a ``message_type``, ignoring unknown fields.

    Raises ``json_format.ParseError`` if the line is not a JSON object for ``message_type``.
    """
    # protobuf alone reads a JSON string or a list of strings as unknown field names, and
    # returns an empty message.
    if not line.lstrip(_JSON_WHITESPACE).startswith("{"):
        raise json_format.ParseError("expected a JSON object")
    return json_format.Parse(line, message_type(), ignore_unknown_fields=True)


def read_jsonl[M: Message](path: Path, message_type: type[M]) -> Iterator[M]:
    """Yield the messages in the JSONL file at ``path`` in order, skipping blank lines.

    Only a line feed ends a line, so a U+2028 inside a string never splits a record. A line that
    is not UTF-8, or not a valid ``message_type``, raises :class:`JsonlError` naming the file and
    the line number.
    """
    with path.open("rb") as file:
        for line_number, raw in enumerate(file, start=1):  # Binary files split at b"\n" only.
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                reason = f"not valid UTF-8 at byte {error.start}"
                raise JsonlError(path, line_number, reason) from error
            if not line.strip(_JSON_WHITESPACE):
                continue
            try:
                message = from_json_line(line, message_type)
            except json_format.ParseError as error:
                reason = f"not a valid {message_type.DESCRIPTOR.full_name}: {error}"
                raise JsonlError(path, line_number, reason) from error
            yield message


def write_jsonl(path: Path, messages: Iterable[Message]) -> None:
    """Write ``messages`` to ``path``, one per line, replacing the file atomically.

    The lines go to a new file in the same directory, which then replaces ``path``: readers see
    the old file or the new one, never part of one, and a failed write leaves ``path`` as it was.
    """
    # Created by open() rather than tempfile, so the file gets the usual permissions, not 0600.
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            for message in messages:
                file.write(to_json_line(message) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)  # Already gone once it has replaced path.


def append_jsonl(path: Path, message: Message) -> None:
    """Append ``message`` to ``path`` as one line, creating the file if needed, and flush it."""
    with path.open("a", encoding="utf-8") as file:  # Closing the file flushes it.
        file.write(to_json_line(message) + "\n")
