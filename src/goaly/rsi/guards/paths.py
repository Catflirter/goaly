"""GUARD_ALLOWED_PATHS (spec §8.2 step 3): a candidate changes only its lane's pack.

The diff between the candidate and the version it was built on may touch only the lane's two pack
directories, and every file it adds or changes there must be a regular file of a kind a pack
holds: no symlink, no gitlink, none of git's own files in any path component, and nothing Python
could load in place of source, such as a compiled extension or bytecode. No file may be larger
than config allows, nor all of them together, so that a snapshot stays small.

Paths are the candidate's own choice, so a path that is not plain printable text is shown quoted
in the guard's report, where it cannot pass for another line.
"""

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from goaly.contracts.gen.experiment_pb2 import GUARD_ALLOWED_PATHS, GuardResult, Lane
from goaly.rsi.git import run_git
from goaly.rsi.lanes import lane_info
from goaly.rsi.workspace import mode_problem

__all__ = [
    "ALLOWED_SUFFIXES",
    "Change",
    "PathsConfig",
    "changes",
    "check_allowed_paths",
    "commit_of",
    "shown",
]

# The kinds of file a pack holds (spec §5.2): code, prompts and skills, config and test data.
ALLOWED_SUFFIXES = frozenset(
    {".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".sql", ".toml", ".csv"}
)
_DELETED = "D"


class PathsConfig(BaseModel):
    """The guard's size limits, from config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_file_bytes: int = Field(default=1024 * 1024, gt=0)  # Prompts and code are far smaller.
    max_total_bytes: int = Field(default=16 * 1024 * 1024, gt=0)  # Of every file added or changed.


@dataclass(frozen=True)
class Change:
    """One path the diff touches: its status letter, such as A, M or D, its new git mode and its
    new object.
    """

    path: str
    status: str
    mode: str
    object: str


def commit_of(repo: Path, revision: str) -> str:
    """Resolve ``revision`` to its commit, once, so that every later command names that commit.
    A revision that looks like an option is never read as one. Raises GitError if it names no
    commit.
    """
    return run_git(
        ["rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"], cwd=repo
    ).strip()


def changes(repo: Path, parent: str, candidate: str) -> list[Change]:
    """Return every path that differs between two commits of ``repo``.

    The diff is between two trees, so git reads no worktree. Submodules are never ignored,
    whatever the repository's config says, so a gitlink shows, with its mode.
    """
    output = run_git(
        [
            "diff",
            "--raw",
            "-z",
            "--no-renames",
            "--no-abbrev",
            "--ignore-submodules=none",
            commit_of(repo, parent),
            commit_of(repo, candidate),
            "--",
        ],
        cwd=repo,
    )
    # Each change is ":<old mode> <new mode> <old object> <new object> <status>", then its path,
    # and the output ends with a NUL; the empty field after it has no partner, so zip drops it.
    fields = output.split("\0")
    found = []
    for meta, path in zip(fields[0::2], fields[1::2], strict=False):
        _, mode, _, new_object, status = meta.lstrip(":").split(" ")
        found.append(Change(path, status, mode, new_object))
    return found


def check_allowed_paths(
    repo: Path,
    lane: Lane.ValueType,
    *,
    parent: str,
    candidate: str,
    config: PathsConfig = PathsConfig(),  # noqa: B008  (frozen, so one default is safe)
) -> GuardResult:
    """Check that ``candidate`` changes only regular pack files of the lane, since ``parent``,
    and none larger than ``config`` allows.
    """
    info = lane_info(lane)
    found = changes(repo, parent, candidate)
    problems = [problem for change in found if (problem := _problem(change, info.pack_dirs))]
    if not problems:
        problems = _size_problems(repo, found, config)
    if problems:
        return GuardResult(guard=GUARD_ALLOWED_PATHS, passed=False, detail="\n".join(problems))
    count = f"{len(found)} changed path{'' if len(found) == 1 else 's'}"
    return GuardResult(guard=GUARD_ALLOWED_PATHS, passed=True, detail=f"{count}, all in the pack")


def shown(path: str) -> str:
    """``path`` as a report shows it: as it is if plain printable text, quoted otherwise."""
    return path if path.isprintable() and path.strip() == path else json.dumps(path)


def _problem(change: Change, packs: tuple[PurePosixPath, ...]) -> str | None:
    path = PurePosixPath(change.path)
    name = shown(change.path)
    if not any(path.is_relative_to(pack) for pack in packs):
        return f"{name}: outside the lane's pack directories"
    if change.status == _DELETED:
        return None
    problem = mode_problem(change.path, change.mode)
    if problem is not None:
        return f"{name}: {problem}" + (", not a regular file" if "mode" in problem else "")
    if path.suffix not in ALLOWED_SUFFIXES:
        kind = f"{path.suffix} files" if path.suffix else "files without a suffix"
        return f"{name}: a pack holds no {shown(kind)}"
    return None


def _size_problems(repo: Path, found: list[Change], config: PathsConfig) -> list[str]:
    """The files larger than the limit, and the total if it is too: git's sizes, no reading."""
    added = [change for change in found if change.status != _DELETED]
    if not added:
        return []
    listed = "".join(f"{change.object}\n" for change in added)
    sizes = run_git(["cat-file", "--batch-check=%(objectsize)"], cwd=repo, input_text=listed)
    problems = []
    total = 0
    for change, size in zip(added, map(int, sizes.split()), strict=True):
        total += size
        if size > config.max_file_bytes:
            problems.append(
                f"{shown(change.path)}: {size} bytes, more than {config.max_file_bytes}"
            )
    if total > config.max_total_bytes:
        problems.append(f"the changed files hold {total} bytes, more than {config.max_total_bytes}")
    return problems
