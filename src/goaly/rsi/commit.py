"""The commit helper (spec §8.8, "Finishing"): an implementer's work becomes one commit.

The sandbox cannot commit, so goaly.rsi commits on the host once the agent's run has ended and
its container is gone. It checks that every change lies inside the lane's two pack directories,
stages only those directories, and commits once, as ``rsi(<lane>): <summary>`` with
``Experiment-Id`` and ``Hypothesis-Id`` trailers. It never pushes: the results PRs do (spec §8.7).

What the sandbox wrote is checked before git runs in the worktree, and what git staged is
checked again before the commit: a pack holds only regular files and directories.
"""

from pathlib import PurePosixPath

from goaly.rsi.git import run_git, run_git_unless_no
from goaly.rsi.ids import check_id
from goaly.rsi.lanes import LaneInfo, lane_info
from goaly.rsi.workspace import (
    GIT_FILES,
    UnsafePackError,
    Workspace,
    WorkspaceError,
    Worktree,
    check_pack_files,
)

__all__ = ["OutsidePackError", "commit_candidate"]

_SUBJECT_LIMIT = 72  # Characters, as git's own tools expect of a subject line.
_BODY_LIMIT = 4000
_REGULAR = frozenset({"100644", "100755"})  # git's modes for a file; 120000 is a symlink, 160000
# a gitlink.


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
    changed. Nothing is committed when it raises:

    - WorkspaceError if ``worktree`` is not the registered worktree of its experiment, or is not
      on its experiment's branch;
    - UnsafePackError if a pack holds anything but regular files and directories (see
      goaly.rsi.workspace.check_pack_files), or if its best version did; this rejects the
      candidate;
    - OutsidePackError if any tracked or untracked path outside the pack directories changed;
    - ValueError for an invalid experiment or hypothesis ID.

    Ignored files are never committed. Call it only once the sandbox has stopped, so that
    nothing changes the worktree meanwhile.
    """
    check_id(worktree.experiment_id, "experiment_id")
    check_id(hypothesis_id, "hypothesis_id")
    info = lane_info(worktree.lane)
    message = _message(info.name, summary, worktree.experiment_id, hypothesis_id)
    with workspace.lock():
        workspace.check_registered(worktree)
        check_pack_files(worktree)  # Before any git command runs in the worktree.
        _check_modes(_tracked(worktree, info), "in its best version")
        _check_branch(worktree)
        changed = [PurePosixPath(path) for path in _changed_paths(worktree)]
        outside = sorted(
            str(path)
            for path in changed
            if not any(path.is_relative_to(pack) for pack in info.pack_dirs)
        )
        if outside:
            raise OutsidePackError(outside)
        # git add fails on a pathspec that matches nothing, so stage only the pack directories
        # that exist or hold a change, such as a deleted file.
        staged = [
            str(pack)
            for pack in info.pack_dirs
            if (worktree.path / pack).exists() or any(path.is_relative_to(pack) for path in changed)
        ]
        if not staged:  # Neither pack exists or changed; git add without a path adds all.
            return None
        run_git(["add", "--all", "--", *staged], cwd=worktree.path)
        try:
            _check_modes(_staged(worktree, info), "staged")
        except UnsafePackError:
            run_git(["reset", "--quiet", "--", *staged], cwd=worktree.path)
            raise
        if not _anything_staged(worktree):
            return None
        run_git(
            ["commit", "--quiet", "--no-verify", "--cleanup=verbatim", "--file=-"],
            cwd=worktree.path,
            input_text=message,
        )
    return workspace.head(worktree)


def _check_branch(worktree: Worktree) -> None:
    output = run_git_unless_no(["symbolic-ref", "--quiet", "HEAD"], cwd=worktree.path)
    head = "no branch" if output is None else output.strip()  # None: a detached HEAD.
    if head != f"refs/heads/{worktree.branch}":
        raise WorkspaceError(f"{worktree.path} is on {head}, not on {worktree.branch}")


def _tracked(worktree: Worktree, info: LaneInfo) -> dict[str, str]:
    """The mode of every path the packs hold in HEAD; ls-tree reads no file of the worktree."""
    output = run_git(
        ["ls-tree", "-r", "-z", "HEAD", "--", *map(str, info.pack_dirs)], cwd=worktree.path
    )
    entries = (entry.partition("\t") for entry in output.split("\0") if entry)
    return {path: meta.split(" ")[0] for meta, _, path in entries}  # "<mode> <type> <object>"


def _staged(worktree: Worktree, info: LaneInfo) -> dict[str, str]:
    """The mode of every path of the packs in the index."""
    output = run_git(
        ["ls-files", "--stage", "-z", "--", *map(str, info.pack_dirs)], cwd=worktree.path
    )
    entries = (entry.partition("\t") for entry in output.split("\0") if entry)
    return {path: meta.split(" ")[0] for meta, _, path in entries}  # "<mode> <object> <stage>"


def _check_modes(modes: dict[str, str], where: str) -> None:
    problems = []
    for path, mode in sorted(modes.items()):
        if mode == "160000":
            problems.append(f"{path}: a gitlink {where}")
        elif mode not in _REGULAR:
            problems.append(f"{path}: mode {mode} {where}")
        elif PurePosixPath(path).name in GIT_FILES:
            problems.append(f"{path}: git's own file {where}")
    if problems:
        raise UnsafePackError(problems)


def _anything_staged(worktree: Worktree) -> bool:
    # diff --quiet exits with 1 when the index differs from HEAD, so its "no" is a yes here.
    quiet = ["diff", "--cached", "--quiet", "--ignore-submodules=all"]
    return run_git_unless_no(quiet, cwd=worktree.path) is None


def _changed_paths(worktree: Worktree) -> list[str]:
    """The paths that differ from HEAD: changed, deleted or untracked, but not ignored."""
    output = run_git(
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
            "--ignore-submodules=all",
        ],
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
    """Keep printable characters and line breaks, with tabs as spaces, and drop trailing spaces
    and blank edges. A line that starts with ``---`` gets a space before it, so that tools that
    read a commit as a patch never take it for the end of the message.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").split("\n")
    printable = ("".join(char for char in line if char.isprintable()).rstrip() for line in lines)
    kept = "\n".join(printable).strip()[:_BODY_LIMIT].split("\n")
    return "\n".join(f" {line}" if line.startswith("---") else line for line in kept)


def _shorten(line: str, limit: int) -> str:
    """Return ``line``, or its words that fit in ``limit`` characters with "..." after them."""
    if len(line) <= limit:
        return line
    cut = line[: limit - 3]
    if line[limit - 3] != " " and " " in cut:  # Do not end in the middle of a word.
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "..."
