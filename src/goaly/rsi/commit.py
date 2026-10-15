"""The commit helper (spec §8.8, "Finishing"): an implementer's work becomes one commit.

The sandbox cannot commit, so goaly.rsi commits on the host once the agent's run has ended and
its container is gone. It checks that every change lies inside the lane's two pack directories,
stages only those directories, and commits once, as ``rsi(<lane>): <summary>`` with
``Experiment-Id`` and ``Hypothesis-Id`` trailers. It never pushes: the results PRs do (spec §8.7).
"""

from pathlib import PurePosixPath

from goaly.rsi.git import run_git
from goaly.rsi.lanes import lane_info
from goaly.rsi.ledger import check_id
from goaly.rsi.workspace import Workspace, Worktree

__all__ = ["OutsidePackError", "commit_candidate"]

_SUBJECT_LIMIT = 72  # Characters, as git's own tools expect of a subject line.
_BODY_LIMIT = 4000


class OutsidePackError(Exception):
    """The worktree changed outside the lane's pack directories, so nothing was committed."""

    def __init__(self, paths: list[str]) -> None:
        super().__init__("changed outside the pack directories: " + ", ".join(paths))
        self.paths = paths


def commit_candidate(
    workspace: Workspace, worktree: Worktree, *, hypothesis_id: str, summary: str
) -> str | None:
    """Commit the worktree's changes to its lane's pack as one commit, and return the commit.

    ``summary`` is the agent's closing summary: its first line, cleaned and shortened, becomes
    the subject, and the whole of it the body. Returns None, committing nothing, if nothing
    changed. Raises OutsidePackError, committing nothing, if any tracked or untracked path
    outside the pack directories changed, and ValueError for an invalid hypothesis ID. Ignored
    files are never committed. Call it only once the sandbox has stopped, so that nothing changes
    the worktree meanwhile.
    """
    check_id(hypothesis_id, "hypothesis_id")
    info = lane_info(worktree.lane)
    message = _message(info.name, summary, worktree.experiment_id, hypothesis_id)
    with workspace.lock():
        changed = [PurePosixPath(path) for path in _changed_paths(worktree)]
        outside = sorted(
            str(path)
            for path in changed
            if not any(path.is_relative_to(pack) for pack in info.pack_dirs)
        )
        if outside:
            raise OutsidePackError(outside)
        if not changed:
            return None
        # git add fails on a pathspec that matches nothing, so stage only the pack directories
        # that exist or hold a change, such as a deleted file.
        staged = [
            str(pack)
            for pack in info.pack_dirs
            if (worktree.path / pack).exists() or any(path.is_relative_to(pack) for path in changed)
        ]
        run_git(["add", "--all", "--", *staged], cwd=worktree.path)
        run_git(
            ["commit", "--quiet", "--no-verify", "--cleanup=verbatim", "--file=-"],
            cwd=worktree.path,
            input_text=message,
        )
    return workspace.head(worktree)


def _changed_paths(worktree: Worktree) -> list[str]:
    """The paths that differ from HEAD: changed, deleted or untracked, but not ignored."""
    output = run_git(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames"],
        cwd=worktree.path,
    )
    # Each entry is "XY <path>"; with --no-renames, none is followed by a rename's source.
    return [entry[3:] for entry in output.split("\0") if entry]


def _message(lane: str, summary: str, experiment_id: str, hypothesis_id: str) -> str:
    text = _clean(summary)
    first_line = text.partition("\n")[0]
    prefix = f"rsi({lane}): "
    headline = _shorten(first_line, _SUBJECT_LIMIT - len(prefix))
    subject = prefix + (headline or f"implement hypothesis {hypothesis_id}")
    # The body holds the summary unless the subject already says all of it.
    body = text if text and text != headline else ""
    trailers = f"Experiment-Id: {experiment_id}\nHypothesis-Id: {hypothesis_id}"
    return "\n\n".join(part for part in (subject, body, trailers) if part) + "\n"


def _clean(text: str) -> str:
    """Keep printable characters and line breaks, and drop trailing spaces and blank edges."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    printable = ("".join(char for char in line if char.isprintable()).rstrip() for line in lines)
    return "\n".join(printable).strip()[:_BODY_LIMIT]


def _shorten(line: str, limit: int) -> str:
    """Return ``line``, or its words that fit in ``limit`` characters with "..." after them."""
    if len(line) <= limit:
        return line
    cut = line[: limit - 3]
    if line[limit - 3] != " " and " " in cut:  # Do not end in the middle of a word.
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "..."
