"""Replacing a file atomically, for the files a run rewrites whole: cache entries, the manifest
and the official prediction files (spec §6.2)."""

import os
import secrets
from pathlib import Path

__all__ = ["write_text_atomically"]


def write_text_atomically(path: Path, text: str) -> None:
    """Replace ``path`` with ``text``, encoded as UTF-8 and otherwise written as it is.

    The text goes to a new temporary file in the same directory, which then replaces ``path``
    with one rename. A reader sees the old file or the new one, never part of one, and a failed
    write leaves ``path`` as it was. When writers race, the last rename wins. The file is synced
    to disk before the rename and the directory after it, so the new file survives a power cut.
    It blocks: run it in a worker thread from the event loop.
    """
    data = text.encode("utf-8")  # First: text that cannot be encoded leaves no temporary file.
    # Created by open() rather than tempfile, so the file gets the usual permissions, not 0600.
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)  # Already gone once it has replaced path.
    _sync_directory(path.parent)


def _sync_directory(directory: Path) -> None:
    """Sync ``directory`` to disk, which makes a rename inside it durable."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
