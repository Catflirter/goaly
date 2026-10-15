"""Tests for goaly.harness.core.atomic: replacing a file atomically."""

import os
import stat
from pathlib import Path, PurePath

import pytest

from goaly.harness.core.atomic import write_text_atomically


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    """An empty directory: tmp_path also holds the isolated HOME of conftest.py."""
    path = tmp_path / "out"
    path.mkdir()
    return path


def test_writes_the_text_as_utf8_bytes_unchanged(directory: Path) -> None:
    path = directory / "file.jsonl"

    write_text_atomically(path, "naïve 日本語\r\nline\n")

    assert path.read_bytes() == "naïve 日本語\r\nline\n".encode()
    assert list(directory.iterdir()) == [path]  # No temporary file is left behind.


def test_replaces_an_existing_file(directory: Path) -> None:
    path = directory / "file.json"
    path.write_text("old, and longer than the new text\n", encoding="utf-8")

    write_text_atomically(path, "new\n")

    assert path.read_text(encoding="utf-8") == "new\n"
    assert list(directory.iterdir()) == [path]


def test_renames_a_file_from_the_same_directory(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A rename replaces the file atomically only within one file system.
    renames: list[tuple[Path, Path]] = []
    replace = Path.replace

    def recording_replace(source: Path, target: str | PurePath) -> Path:
        renames.append((source, Path(target)))
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    path = directory / "file.json"

    write_text_atomically(path, "text")

    ((source, target),) = renames
    assert target == path
    assert source.parent == target.parent
    assert source.name.startswith(".file.json.")


def test_syncs_the_file_before_the_rename_and_the_directory_after_it(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    fsync, replace = os.fsync, Path.replace

    def recording_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(f"sync {kind}")
        fsync(descriptor)

    def recording_replace(source: Path, target: str | PurePath) -> Path:
        events.append("rename")
        return replace(source, target)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(Path, "replace", recording_replace)

    write_text_atomically(directory / "file.json", "text")

    assert events == ["sync file", "rename", "sync directory"]


def test_a_failed_write_leaves_the_file_as_it_was(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = directory / "file.json"
    path.write_text("old\n", encoding="utf-8")

    def failing_replace(source: Path, target: str | PurePath) -> Path:
        raise OSError("the rename failed")

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="the rename failed"):
        write_text_atomically(path, "new\n")

    assert path.read_text(encoding="utf-8") == "old\n"
    assert list(directory.iterdir()) == [path]


def test_text_that_cannot_be_encoded_changes_nothing(directory: Path) -> None:
    path = directory / "file.json"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(UnicodeEncodeError):
        write_text_atomically(path, "a lone surrogate: \ud800")

    assert path.read_text(encoding="utf-8") == "old\n"
    assert list(directory.iterdir()) == [path]


def test_a_written_file_gets_the_permissions_of_any_new_file(directory: Path) -> None:
    path, reference = directory / "file.json", directory / "reference"
    reference.touch()

    write_text_atomically(path, "text")

    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)
