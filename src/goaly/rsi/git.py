"""How goaly.rsi runs git: never a hook, a prompt, an editor or background maintenance.

The lanes run unattended (spec §8.0), so nothing git starts may wait for a person or run code
from the repository's hooks, and a command that hangs is stopped, with its children, after a
timeout.
"""

import os
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path

__all__ = ["GitError", "run_git"]

# Settings that override the repository's and the user's for every command.
_SETTINGS = (
    "core.hooksPath=/dev/null",  # No hook runs: not pre-commit, nor post-checkout.
    "core.fsmonitor=false",
    "gc.auto=0",  # No garbage collection while other lanes use the repository.
    "maintenance.auto=false",
    "commit.gpgSign=false",  # Signing could wait for a passphrase.
    "tag.gpgSign=false",
)
# The only GIT_* variables kept from the environment: who commits. Others, such as GIT_DIR or
# GIT_INDEX_FILE, could point git at another repository.
_KEPT = frozenset(
    {"GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"}
)


class GitError(Exception):
    """A git command failed or timed out; ``returncode`` is None after a timeout."""

    def __init__(self, args: Sequence[str], returncode: int | None, reason: str) -> None:
        super().__init__(f"git {' '.join(args)}: {reason.strip()}")
        self.returncode = returncode


def run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float = 120.0,
    input_text: str | None = None,
) -> str:
    """Run ``git <args>`` in ``cwd`` and return its standard output.

    Bytes that are not UTF-8, as in a path, come back as surrogate escapes. Raises GitError if
    git fails, or after ``timeout_s``, when git and every process it started are killed.
    """
    settings = [item for setting in _SETTINGS for item in ("-c", setting)]
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_") or name in _KEPT
    }
    environment.update(GIT_TERMINAL_PROMPT="0", GIT_EDITOR="true", GIT_PAGER="cat", LC_ALL="C")
    with subprocess.Popen(  # noqa: S603  (git, with arguments goaly.rsi builds)
        ["git", *settings, *args],  # noqa: S607  (git from PATH)
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL if input_text is None else subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,  # Its own process group, so a timeout can stop its children.
    ) as process:
        try:
            stdout, stderr = process.communicate(
                None if input_text is None else input_text.encode("utf-8"), timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise GitError(args, None, f"timed out after {timeout_s:g} s") from None
    if process.returncode != 0:
        raise GitError(args, process.returncode, stderr.decode("utf-8", "replace"))
    return stdout.decode("utf-8", "surrogateescape")
