"""Read and write protobuf messages as JSON Lines (spec §3.3).

Each line holds one message in protobuf's standard JSON mapping, under the field names of the
.proto files. Fields without presence are always written, so a default such as
``"correct": false`` shows instead of disappearing. Reading ignores unknown fields, which keeps
files written before a field was removed (and reserved) readable.

On disk, field names and enum value names are the keys. So never rename a field or an enum
value, and when one is removed, reserve its name as well as its number.

A crash loses at most the line being written (spec §6.2): read_jsonl skips the torn tail it
leaves, and the next append_jsonl cuts it off. Nothing here locks or fsyncs, so a caller with
concurrent writers must hold its own lock across each call.
"""

import json
import logging
import os
import secrets
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO, Self, cast

from google.protobuf import json_format
from google.protobuf.descriptor import Descriptor, FieldDescriptor
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
_WELL_KNOWN_TYPES = "google.protobuf"  # Their JSON form is special: a string, a list or any JSON.
_READ_SIZE = 64 * 1024  # How far append_jsonl reads back at a time, looking for a line feed.

_logger = logging.getLogger(__name__)


class JsonlError(ValueError):
    """A line of a JSONL file holds no valid message; the error text starts ``path:line:``."""

    def __init__(self, path: Path, line_number: int, reason: str) -> None:
        super().__init__(f"{path}:{line_number}: {reason}")
        self.path = path
        self.line_number = line_number
        self.reason = reason

    def __reduce__(self) -> tuple[type[Self], tuple[Path, int, str]]:
        # Pickling would rebuild the error from args, which hold only the formatted text.
        return type(self), (self.path, self.line_number, self.reason)


def to_json_line(message: Message) -> str:
    """Write ``message`` as one line of compact JSON, with no line break at the end.

    Keys follow the order the .proto declares the fields in, at every level, so the lines of a
    file line up whichever fields are set. Raises ``json_format.SerializeToJsonError`` for a value
    JSON cannot hold, such as a NaN in a ``google.protobuf.Value``.
    """
    fields = json_format.MessageToDict(
        message, always_print_fields_with_no_presence=True, preserving_proto_field_name=True
    )
    descriptor = cast(Descriptor, message.DESCRIPTOR)  # The stubs also allow upb's own class.
    if descriptor.file.package != _WELL_KNOWN_TYPES:
        fields = _in_field_order(descriptor, fields)
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

    Only a line feed ends a line, so a U+2028 inside a string never splits a record. A last line
    with no line feed that does not parse is the torn tail of an interrupted write: it is skipped,
    with a warning. Any other line that is not UTF-8, or not a valid ``message_type``, raises
    :class:`JsonlError` naming the file and the line number.
    """
    with path.open("rb") as file:
        for line_number, raw in enumerate(file, start=1):  # Binary files split at b"\n" only.
            try:
                message = _parse_line(raw, message_type)
            except (UnicodeDecodeError, json_format.ParseError) as error:
                if not raw.endswith(b"\n"):  # Only the last line can lack one.
                    _logger.warning("%s:%d: skipped a torn last line", path, line_number)
                    return
                raise JsonlError(path, line_number, _reason(error, message_type)) from error
            if message is not None:
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
    """Append ``message`` to ``path`` as one line, creating the file if needed, and flush it.

    The message is serialized first, so an error leaves the file as it was. Every line these
    helpers write ends in a line feed, so a last line without one is the tail of an interrupted
    write. If it does not parse, it is cut off, as read_jsonl skips it; if it does, it lost only
    its line feed, which is put back. Nothing locks the file: a caller with concurrent writers
    must hold its lock across the call.
    """
    line = (to_json_line(message) + "\n").encode("utf-8")
    with path.open("a+b") as file:  # Every write goes to the end; closing the file flushes it.
        end = file.seek(0, os.SEEK_END)
        start = _start_of_last_line(file, end)
        if start < end:
            file.seek(start)
            if _parses(file.read(), type(message)):
                line = b"\n" + line
            else:
                file.truncate(start)
        file.write(line)


def _in_field_order(descriptor: Descriptor, fields: dict[str, Any]) -> dict[str, Any]:
    """Return ``fields``, a message's JSON object, with its keys in declaration order.

    protobuf writes the fields that are set first, ordered by number, and then the rest.
    """
    return {
        field.name: _value_in_field_order(field, fields[field.name])
        for field in descriptor.fields
        if field.name in fields
    }


def _value_in_field_order(field: FieldDescriptor, value: Any) -> Any:
    """Order the keys inside a field's JSON value; a well-known type's value stays as it is."""
    message = field.message_type
    if message is None or message.file.package == _WELL_KNOWN_TYPES:
        return value
    if message.GetOptions().map_entry:  # A map: its keys stay as they are, its values are ordered.
        entry_value = message.fields_by_name["value"]
        return {key: _value_in_field_order(entry_value, item) for key, item in value.items()}
    if field.is_repeated:
        return [_in_field_order(message, item) for item in value]
    return _in_field_order(message, value)


def _parse_line[M: Message](raw: bytes, message_type: type[M]) -> M | None:
    """Parse one line of a JSONL file, or return None if it is blank."""
    line = raw.decode("utf-8")
    return from_json_line(line, message_type) if line.strip(_JSON_WHITESPACE) else None


def _parses(raw: bytes, message_type: type[Message]) -> bool:
    try:
        _parse_line(raw, message_type)
    except (UnicodeDecodeError, json_format.ParseError):
        return False
    return True


def _reason(error: UnicodeDecodeError | json_format.ParseError, message_type: type[Message]) -> str:
    if isinstance(error, UnicodeDecodeError):
        return f"not valid UTF-8 at byte {error.start}"
    return f"not a valid {message_type.DESCRIPTOR.full_name}: {error}"


def _start_of_last_line(file: BinaryIO, end: int) -> int:
    """Return the offset just after the last line feed before ``end``, or 0 if there is none."""
    position = end
    while position > 0:
        start = max(0, position - _READ_SIZE)
        file.seek(start)
        newline = file.read(position - start).rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        position = start
    return 0
