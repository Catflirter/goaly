"""GUARD_ALLOWED_PATHS (spec §8.2 step 3): a candidate changes only its lane's pack.

The diff between the candidate and the version it was built on may touch only the lane's two pack
directories, and every file it adds or changes there must be a regular file of a kind a pack
holds: no symlink, no gitlink, none of git's own files, and nothing Python could load in place of
source, such as a compiled extension or bytecode.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from goaly.contracts.gen.experiment_pb2 import GUARD_ALLOWED_PATHS, GuardResult, Lane
from goaly.rsi.git import run_git
from goaly.rsi.lanes import lane_info
from goaly.rsi.workspace import GIT_FILES

__all__ = ["ALLOWED_SUFFIXES", "Change", "changes", "check_allowed_paths"]

# The kinds of file a pack holds (spec §5.2): code, prompts and skills, config and test data.
ALLOWED_SUFFIXES = frozenset(
    {".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".sql", ".toml", ".csv"}
)
_REGULAR = frozenset({"100644", "100755"})
_DELETED = "D"


@dataclass(frozen=True)
class Change:
    """One path the diff touches: its status letter, such as A, M or D, and its new git mode."""

    path: str
    status: str
    mode: str


def changes(repo: Path, parent: str, candidate: str) -> list[Change]:
    """Return every path that differs between two commits of ``repo``.

    The diff is between two trees, so git reads no worktree. Submodules are not ignored, so a
    gitlink shows, with its mode.
    """
    output = run_git(
        ["diff", "--raw", "-z", "--no-renames", "--no-abbrev", parent, candidate, "--"],
        cwd=repo,
    )
    # Each change is ":<old mode> <new mode> <old object> <new object> <status>", then its path,
    # and the output ends with a NUL; the empty field after it has no partner, so zip drops it.
    fields = output.split("\0")
    found = []
    for meta, path in zip(fields[0::2], fields[1::2], strict=False):
        _, mode, _, _, status = meta.lstrip(":").split(" ")
        found.append(Change(path, status, mode))
    return found


def check_allowed_paths(
    repo: Path, lane: Lane.ValueType, *, parent: str, candidate: str
) -> GuardResult:
    """Check that ``candidate`` changes only regular pack files of the lane, since ``parent``."""
    info = lane_info(lane)
    found = changes(repo, parent, candidate)
    problems = [problem for change in found if (problem := _problem(change, info.pack_dirs))]
    if problems:
        return GuardResult(guard=GUARD_ALLOWED_PATHS, passed=False, detail="\n".join(problems))
    count = f"{len(found)} changed path{'' if len(found) == 1 else 's'}"
    return GuardResult(guard=GUARD_ALLOWED_PATHS, passed=True, detail=f"{count}, all in the pack")


def _problem(change: Change, packs: tuple[PurePosixPath, ...]) -> str | None:
    path = PurePosixPath(change.path)
    if not any(path.is_relative_to(pack) for pack in packs):
        return f"{change.path}: outside the lane's pack directories"
    if change.status == _DELETED:
        return None
    if change.mode not in _REGULAR:
        return f"{change.path}: mode {change.mode}, not a regular file"
    if path.name in GIT_FILES:
        return f"{change.path}: git's own file"
    if path.suffix not in ALLOWED_SUFFIXES:
        kind = f"{path.suffix} files" if path.suffix else "files without a suffix"
        return f"{change.path}: a pack holds no {kind}"
    return None
