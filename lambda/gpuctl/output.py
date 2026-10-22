"""Best-effort output for gpuctl's progress lines and warnings (Task 21a review C0-1, C0-2, X0-1).

gpuctl flushes every line so `make gpu-apply | tee` shows progress as it happens (live finding L21-4). When the
reader of that pipe is gone (tee killed by the same Ctrl-C, `| head`, a demo filter that died) every write raises
BrokenPipeError, since Python ignores SIGPIPE. A progress line must never stop the work it reports: gpu-down must
still terminate, and a failed apply must still terminate what it launched and take the IPs out of the allow-lists.
"""

from __future__ import annotations

import io
import os
import sys
from typing import TextIO


def write(stream: TextIO, text: str) -> bool:
    """Write and flush `text`; False, and nothing raised, when the stream is gone.

    When that stream is the process's own stdout or stderr, /dev/null takes the place of its file descriptor, so
    later writes, and the interpreter's final flush at exit, neither fail nor change the exit code.
    """
    try:
        stream.write(text)
        stream.flush()
        return True
    except (OSError, ValueError):  # BrokenPipeError, EIO from a closed terminal; ValueError: the file was closed
        _replace_with_devnull(stream)
        return False


def _replace_with_devnull(stream: TextIO) -> None:
    if not any(stream is s for s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)):
        return
    try:
        fd = stream.fileno()
        null = os.open(os.devnull, os.O_WRONLY)
    except (OSError, ValueError, io.UnsupportedOperation):  # not a real file (a test's capture), or already closed
        return
    try:
        os.dup2(null, fd)
    except OSError:
        pass
    finally:
        os.close(null)
