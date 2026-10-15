"""How goaly.rsi runs git: never a hook, a prompt, an editor or background maintenance.

The lanes run unattended (spec §8.0), so nothing git starts may wait for a person or run code
from the repository's hooks, and a command that hangs is stopped, with its children, after a
timeout. git also ignores the user's and the system's config, never descends into a nested
repository, and reads no attributes or ignore rules but the repository's own: a candidate's pack
may hold anything the sandbox could write.

For T22, which pushes results branches: run_git drops the credentials a push needs, GIT_ASKPASS
and the user's config included, so a push must bring its own. best/<lane> can move to a commit
that does not descend from the remote's, so pushing it must force the tag.
"""

import contextlib
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = ["GitError", "run_git", "run_git_unless_no"]

# Settings that override the repository's for every command.
_SETTINGS = (
    "core.hooksPath=/dev/null",  # No hook runs: not pre-commit, nor post-checkout.
    "core.fsmonitor=false",
    "core.attributesFile=/dev/null",  # Attributes and ignore rules come only from the repository.
    "core.excludesFile=",
    "submodule.recurse=false",  # Never descend into a nested repository,
    "protocol.file.allow=never",  # nor clone one from a path.
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
# The only variable a caller may add: an index of its own, as Workspace.pin uses.
_CALLER_VARIABLES = frozenset({"GIT_INDEX_FILE"})
_REAP_S = 10.0  # How long to wait for a killed command's pipes to close.


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
    env: Mapping[str, str] | None = None,
) -> str:
    """Run ``git <args>`` in ``cwd`` and return its standard output.

    ``env`` adds variables once the environment is cleaned; only a GIT_INDEX_FILE of the
    caller's own is allowed, and the hardened variables are applied after it. Bytes that are not
    UTF-8, as in a path, come back as surrogate escapes. Raises ValueError for any other variable
    in ``env``, and GitError if git fails, or after ``timeout_s``, when git and every process it
    started are killed.
    """
    refused = sorted(set(env or {}) - _CALLER_VARIABLES)
    if refused:
        raise ValueError(f"run_git cannot set {', '.join(refused)}")
    settings = [item for setting in _SETTINGS for item in ("-c", setting)]
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_") or name in _KEPT
    }
    environment.update(env or {})
    environment.update(  # Last, so that nothing can undo them.
        GIT_CONFIG_NOSYSTEM="1",  # Neither the system's config nor the user's applies.
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_TERMINAL_PROMPT="0",
        GIT_EDITOR="true",
        GIT_PAGER="cat",
        LC_ALL="C",
    )
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
            _kill_group(process.pid)
            # Something outside the group may still hold the pipes; its output is then lost.
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.communicate(timeout=_REAP_S)
            raise GitError(args, None, f"timed out after {timeout_s:g} s") from None
    if process.returncode != 0:
        reason = stderr.strip() or stdout.strip()  # Some failures say why on stdout only.
        raise GitError(args, process.returncode, reason.decode("utf-8", "replace"))
    return stdout.decode("utf-8", "surrogateescape")


def run_git_unless_no(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float = 120.0,
) -> str | None:
    """Run a git command that answers no by exiting with 1, such as ``merge-base
    --is-ancestor`` or ``diff --quiet``: return None for no, and its output otherwise. Any other
    failure raises GitError, as :func:`run_git` does.
    """
    try:
        return run_git(args, cwd=cwd, timeout_s=timeout_s)
    except GitError as error:
        if error.returncode == 1:
            return None
        raise


def _kill_group(group: int) -> None:
    with contextlib.suppress(ProcessLookupError):  # Every process of it has already exited.
        os.killpg(group, signal.SIGKILL)
