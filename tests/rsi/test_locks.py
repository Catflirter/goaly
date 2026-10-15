"""Tests for goaly.rsi.locks: advisory file locks between RSI processes."""

import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from goaly.rsi.locks import LockTimeoutError, file_lock

# Holds the lock named by argv[1] until stdin closes, printing "locked" once it holds it.
HOLDER = """
import sys
from pathlib import Path
from goaly.rsi.locks import file_lock

with file_lock(Path(sys.argv[1]), shared=sys.argv[2] == "shared"):
    print("locked", flush=True)
    sys.stdin.read()
"""


@contextmanager
def lock_held_elsewhere(path: Path, *, shared: bool = False) -> Iterator[Callable[[], None]]:
    """Hold the lock on ``path`` in another process; the block gets a function releasing it."""
    with subprocess.Popen(  # noqa: S603  (a fixed script and arguments built here)
        [sys.executable, "-c", HOLDER, str(path), "shared" if shared else "exclusive"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as holder:
        assert holder.stdin is not None
        assert holder.stdout is not None
        assert holder.stdout.readline() == "locked\n"
        yield holder.stdin.close
    assert holder.returncode == 0


def test_an_exclusive_lock_times_out_while_another_process_holds_it(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    with lock_held_elsewhere(path):
        started = time.monotonic()
        with (
            pytest.raises(LockTimeoutError, match=re.escape(str(path))),
            file_lock(path, timeout_s=0.3),
        ):
            pytest.fail("the lock was taken while another process held it")
        assert time.monotonic() - started >= 0.3


def test_the_lock_is_free_again_once_its_holder_releases_it(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    with lock_held_elsewhere(path):
        pass

    with file_lock(path, timeout_s=0):
        pass


def test_a_waiting_lock_is_taken_once_its_holder_releases_it(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    with lock_held_elsewhere(path) as release:
        releaser = threading.Timer(0.3, release)
        started = time.monotonic()
        releaser.start()
        with file_lock(path, timeout_s=10):
            waited = time.monotonic() - started
        releaser.join()

    # Timing is loose on a busy machine: the release comes at 0.3 s, the retry within 64 ms.
    assert 0.25 <= waited < 10


def test_a_lock_is_released_when_its_block_raises(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    with pytest.raises(RuntimeError, match="boom"), file_lock(path):
        raise RuntimeError("boom")

    with file_lock(path, timeout_s=0):
        pass


def test_two_opens_in_one_process_exclude_each_other(tmp_path: Path) -> None:
    # The lock belongs to the open file, so two threads of one lane process exclude each other.
    path = tmp_path / "lock"
    with file_lock(path), pytest.raises(LockTimeoutError), file_lock(path, timeout_s=0):
        pass


def test_shared_locks_coexist(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    with lock_held_elsewhere(path, shared=True), file_lock(path, shared=True, timeout_s=0):
        pass


def test_a_shared_lock_keeps_out_an_exclusive_one(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    with (
        lock_held_elsewhere(path, shared=True),
        pytest.raises(LockTimeoutError),
        file_lock(path, timeout_s=0),
    ):
        pass


def test_an_exclusive_lock_keeps_out_a_shared_one(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    with (
        lock_held_elsewhere(path),
        pytest.raises(LockTimeoutError),
        file_lock(path, shared=True, timeout_s=0),
    ):
        pass


def test_the_lock_file_is_created_and_kept(tmp_path: Path) -> None:
    path = tmp_path / "lock"

    with file_lock(path):
        assert path.is_file()

    assert path.is_file()


def test_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError), file_lock(tmp_path / "missing" / "lock"):
        pass


@pytest.mark.parametrize(("shared", "access"), [(True, os.O_RDONLY), (False, os.O_RDWR)])
def test_an_exclusive_lock_opens_its_file_for_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared: bool, access: int
) -> None:
    """NFS emulates flock with POSIX locks, and an exclusive one needs a writable descriptor."""
    opened: list[int] = []
    real_open = os.open

    def spy(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        opened.append(flags)
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", spy)

    with file_lock(tmp_path / "lock", shared=shared):
        pass

    assert [flags & os.O_ACCMODE for flags in opened] == [access]
