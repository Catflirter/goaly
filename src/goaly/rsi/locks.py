"""Advisory file locks that serialize the RSI processes of one machine (spec §6.2).

The ledger takes one around every read and write, and the workspace one around every git
operation on the main repository. They are ``fcntl.flock`` locks: the kernel releases one when
its holder closes it or dies, so a crashed lane never leaves a lock behind.
"""

import fcntl
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["LockTimeoutError", "file_lock"]

# A waiting lock tries again after a pause that starts at 1 ms and doubles up to 64 ms, with
# jitter, so that waiters do not retry in step and none of them starves.
_FIRST_PAUSE_S = 0.001
_LONGEST_PAUSE_S = 0.064


class LockTimeoutError(TimeoutError):
    """Another holder kept the lock for longer than the timeout allowed."""


@contextmanager
def file_lock(path: Path, *, shared: bool = False, timeout_s: float = 60.0) -> Iterator[None]:
    """Hold the lock on ``path`` while the block runs: exclusive, or ``shared`` with other readers.

    The lock file is created if needed, but not its directory, and it is never removed, since
    removing it would let two processes lock two different files. It is opened read-only, which
    is all flock needs, so a reader can lock it without write access. The lock belongs to this
    open file, so two blocks in one process exclude each other too, and child processes do not
    inherit it. Waiting longer than ``timeout_s`` raises :class:`LockTimeoutError`; 0 tries once.
    Raises PermissionError if the file is missing and cannot be created.
    """
    descriptor = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        _acquire(descriptor, fcntl.LOCK_SH if shared else fcntl.LOCK_EX, path, timeout_s)
        yield
    finally:
        os.close(descriptor)  # Closing the only descriptor of the open file releases the lock.


def _acquire(descriptor: int, operation: int, path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    pause = _FIRST_PAUSE_S
    while True:
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockTimeoutError(
                    f"{path} is still locked by another holder after {timeout_s:g} s"
                ) from None
            time.sleep(min(remaining, random.uniform(pause / 2, pause)))  # noqa: S311  (jitter)
            pause = min(pause * 2, _LONGEST_PAUSE_S)
        else:
            return
