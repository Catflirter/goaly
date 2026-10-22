"""One gpuctl write at a time (stretch S4 review r2-1; design spec section 10.4: gpu-down always wins).

`gpuctl apply`, `down` and `sync-peers` hold flock(2) on <repo>/.run/gpuctl.lock for the whole command. A second one
waits until the first ends and says so: a make gpu-down started while the autoscaler's `gpuctl apply` runs goes
right after it, then removes the override file and every GPU node. The read-only commands never take the lock.

A gpuctl write must never start another gpuctl write as its child: the child would wait for its parent's lock for
ever. apply runs join-node.sh with --skip-sync for that reason (it syncs the peers itself; test_infra pins it).

The kernel releases the lock when its holder exits, so there is never a stale lock; the file itself stays and is
always empty: it records nothing (spec section 5.4 keeps gpuctl free of local state).

The stretch autoscaler (scripts/autoscaler.py) takes the lock without waiting (try_lock) before it re-checks what its
tick read, writes the override file and runs `gpuctl apply`, and hands it to that apply as an inherited descriptor
named by GPUCTL_LOCK_FD. A descriptor of the same open file already holds the lock, so that apply runs at once. A
GPUCTL_LOCK_FD naming no descriptor, or one of another file, is ignored: the command then takes the lock itself.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

LOCK_FD_ENV = "GPUCTL_LOCK_FD"


def lock_path(root: Path) -> Path:
    return Path(root) / ".run" / "gpuctl.lock"


class LockError(Exception):
    """The lock file cannot be created or opened: nothing was changed."""


def _open(path: Path) -> int:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    except OSError as e:
        raise LockError(f"cannot open the gpuctl lock {path}: {e.strerror or e}") from None


def try_lock(path: Path) -> int | None:
    """The lock without waiting: an open descriptor holding it (close it to release), or None when it is held.
    LockError when the file cannot be opened."""
    fd = _open(Path(path))
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    return fd


def inherited(path: Path, env: Mapping[str, str]) -> bool:
    """True when env[GPUCTL_LOCK_FD] is a descriptor of this lock file that holds the lock (the caller's)."""
    value = env.get(LOCK_FD_ENV, "").strip()
    if not (value.isascii() and value.isdigit()):
        return False
    fd = int(value)
    try:
        held, wanted = os.fstat(fd), os.stat(path)
        if (held.st_dev, held.st_ino) != (wanted.st_dev, wanted.st_ino):
            return False
        # The same open file description already holds the lock, so this succeeds at once; another one of the same
        # file fails here (never waits), and the command then takes the lock through its own descriptor.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:  # BlockingIOError included
        return False
    return True


@contextmanager
def write_lock(path: Path, *, env: Mapping[str, str], waiting: Callable[[], None]) -> Iterator[None]:
    """Hold the lock for the block; when another write holds it, call waiting() once and wait for it."""
    path = Path(path)
    if inherited(path, env):
        yield
        return
    fd = _open(path)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            waiting()
            fcntl.flock(fd, fcntl.LOCK_EX)  # Ctrl-C stops the wait (KeyboardInterrupt): nothing has changed yet
        yield
    finally:
        os.close(fd)  # releases the lock
