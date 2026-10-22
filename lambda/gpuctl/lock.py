"""One gpuctl write at a time (stretch S4 review r2-1; design spec section 10.4: gpu-down always wins).

`gpuctl apply`, `down` and `sync-peers` hold flock(2) on <repo>/.run/gpuctl.lock for the whole command, including
an interactive one's y/N confirmation prompt (what was confirmed is what runs). A second one waits until the first
ends and says so, again every WAIT_REPEAT_S while it waits: a make gpu-down started while the autoscaler's
`gpuctl apply` runs goes right after it, then removes the override file and every GPU node. Ctrl-C stops the wait
(LockWaitInterrupted, a KeyboardInterrupt): nothing has changed then. The read-only commands never take the lock.

A gpuctl write must never start another gpuctl write as its child: the child would wait for its parent's lock for
ever. apply runs join-node.sh with --skip-sync for that reason (it syncs the peers itself; test_infra pins it).

The kernel releases the lock when its holder exits, so there is never a stale lock; the file itself stays and is
always empty: it records nothing (spec section 5.4 keeps gpuctl free of local state).

The stretch autoscaler (scripts/autoscaler.py) takes the lock without waiting (try_lock) before it re-checks what its
tick read, writes the override file and runs `gpuctl apply`, and hands it to that apply as an inherited descriptor
named by GPUCTL_LOCK_FD. A descriptor of the same open file already holds the lock, so that apply runs at once. A
GPUCTL_LOCK_FD that is set but is not that (no number, no open descriptor, another file, or the lock file through a
description that does not hold the lock) is refused at once (LockError, review S4-r3-1): waiting for the lock file
would wait for the parent that holds it, which in turn waits for this command, and every later make gpu-down would
wait behind both. An unset or empty GPUCTL_LOCK_FD means nothing is inherited: the command takes the lock itself.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

LOCK_FD_ENV = "GPUCTL_LOCK_FD"
WAIT_POLL_S = 0.2  # how often a waiting command tries the lock again
WAIT_REPEAT_S = 30.0  # how often a waiting command says again that it waits (review S4-r3-3)


def lock_path(root: Path) -> Path:
    return Path(root) / ".run" / "gpuctl.lock"


class LockError(Exception):
    """The lock file cannot be created or opened, or GPUCTL_LOCK_FD is not the caller's lock: nothing was changed."""


class LockWaitInterrupted(KeyboardInterrupt):
    """Ctrl-C while waiting for the lock: nothing was changed."""


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


def inherited_problem(path: Path, env: Mapping[str, str]) -> str | None:
    """Why env[GPUCTL_LOCK_FD] is not a descriptor of this lock file that holds the lock, or None when it is (or when
    it is unset or empty: nothing is inherited)."""
    value = env.get(LOCK_FD_ENV, "").strip()
    if not value:
        return None
    if not (value.isascii() and value.isdigit()):
        return "not a descriptor number"
    fd = int(value)
    try:
        held = os.fstat(fd)
    except OSError as e:
        return f"not an open descriptor in this process ({e.strerror or e}; the parent did not pass it through)"
    try:
        wanted = os.stat(path)
    except OSError as e:
        return f"cannot check it against {path}: {e.strerror or e}"
    if (held.st_dev, held.st_ino) != (wanted.st_dev, wanted.st_ino):
        return f"not a descriptor of {path} (another file: a parent with another repo root?)"
    try:
        # The same open file description already holds the lock, so this succeeds at once; another one of the same
        # file fails here (never waits)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:  # BlockingIOError included
        return f"a descriptor of {path} that does not hold the lock ({e.strerror or e})"
    return None


def inherited(path: Path, env: Mapping[str, str]) -> bool:
    """True when env[GPUCTL_LOCK_FD] is a descriptor of this lock file that holds the lock (the caller's)."""
    return bool(env.get(LOCK_FD_ENV, "").strip()) and inherited_problem(path, env) is None


@contextmanager
def write_lock(path: Path, *, env: Mapping[str, str], waiting: Callable[[float], None]) -> Iterator[None]:
    """Hold the lock for the block. When another write holds it, call waiting(0.0) at once and waiting(seconds
    waited so far) every WAIT_REPEAT_S, and wait for it; Ctrl-C then raises LockWaitInterrupted. LockError when
    GPUCTL_LOCK_FD is set but is not the caller's lock (never a wait for it)."""
    path = Path(path)
    if env.get(LOCK_FD_ENV, "").strip():
        problem = inherited_problem(path, env)
        if problem is not None:
            raise LockError(f"{LOCK_FD_ENV}={env[LOCK_FD_ENV].strip()} is {problem}: refusing rather than waiting for "
                            f"the lock its parent holds (a deadlock); unset {LOCK_FD_ENV} to wait for {path}")
        yield
        return
    fd = _open(path)
    try:
        _wait(fd, waiting)
        yield
    finally:
        os.close(fd)  # releases the lock


def _wait(fd: int, waiting: Callable[[float], None]) -> None:
    started = time.monotonic()
    said: float | None = None
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                pass
            waited = time.monotonic() - started
            if said is None:
                waiting(0.0)  # 0.0 exactly: the first time
                said = waited
            elif waited - said >= WAIT_REPEAT_S:
                waiting(max(waited, 1e-3))
                said = waited
            time.sleep(WAIT_POLL_S)
    except KeyboardInterrupt:
        raise LockWaitInterrupted from None
