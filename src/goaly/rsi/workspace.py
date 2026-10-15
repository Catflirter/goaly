"""Candidate worktrees, best tags and the repository lock (spec §6.2, §8.2 step 2, §8.7).

Each experiment builds its candidate in its own git worktree, ``<root>/<lane>/<experiment_id>``,
on the branch ``rsi/<lane>/<experiment_id>``, from the lane's best version: the tag
``best/<lane>``. The root lies outside the main repository, so a read-only mount of a worktree
carries nothing else of it, and both of the lane's pack directories exist before the sandbox
mounts them. Removing a worktree keeps its branch, so a rejected candidate's commit stays
reachable.

One lock in the main repository's git directory serializes the worktree, branch and tag
operations of every lane process. Reading a ref needs no lock: git updates refs atomically.

The sandbox can write anything into a worktree's pack directories, a nested repository or a
symlink included, and git on the host must never act on that: :func:`check_pack_files` checks
the packs before any git command runs in a candidate's worktree, and :meth:`Workspace.remove`
deletes a worktree's files itself before git forgets the worktree.

For T22: best/<lane> can move to a commit that does not descend from the remote's tag, so
pushing it must force the tag.
"""

import logging
import os
import shutil
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import Lane
from goaly.rsi.git import GitError, run_git, run_git_unless_no
from goaly.rsi.lanes import lane_info
from goaly.rsi.ledger import check_id
from goaly.rsi.locks import file_lock

__all__ = [
    "DEFAULT_ROOT",
    "GIT_FILES",
    "UnsafePackError",
    "Workspace",
    "WorkspaceError",
    "Worktree",
    "check_pack_files",
]

DEFAULT_ROOT = Path("~/goaly-rsi-worktrees")
_LOCK = "goaly-rsi.lock"  # In the main repository's git directory, shared by all its worktrees.
# Files that tell git how to treat a tree; a pack may hold none of them.
GIT_FILES = frozenset({".git", ".gitattributes", ".gitmodules", ".gitignore"})

_logger = logging.getLogger(__name__)


class WorkspaceError(Exception):
    """A worktree, branch or tag operation was refused; nothing was changed."""


class UnsafePackError(Exception):
    """A pack holds something git must not act on, so nothing ran on it and nothing was
    committed. The candidate is rejected.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__("unsafe pack: " + "; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Worktree:
    """An experiment's worktree."""

    path: Path
    lane: Lane.ValueType
    experiment_id: str

    @property
    def branch(self) -> str:
        """The candidate's branch, ``rsi/<lane>/<experiment_id>``."""
        return lane_info(self.lane).branch(self.experiment_id)

    @property
    def pack_dirs(self) -> tuple[Path, Path]:
        """The lane's two pack directories here: the only paths the candidate may change."""
        source, tests = lane_info(self.lane).pack_dirs
        return self.path / source, self.path / tests


class Workspace:
    """The candidate worktrees of the repository at ``repo``, under ``root``.

    ``root`` defaults to DEFAULT_ROOT and must lie outside the repository. Operations that
    change worktrees, branches or tags wait at most ``lock_timeout_s`` for the repository lock,
    then raise goaly.rsi.locks.LockTimeoutError.
    """

    def __init__(
        self,
        repo: Path,
        *,
        root: Path | None = None,
        lock_timeout_s: float = 120.0,
        git_timeout_s: float = 120.0,
    ) -> None:
        self.repo = repo.resolve()
        self.root = (DEFAULT_ROOT if root is None else root).expanduser().resolve()
        if self.root.is_relative_to(self.repo):
            raise ValueError(f"the worktree root {self.root} must lie outside {self.repo}")
        self.lock_timeout_s = lock_timeout_s
        self.git_timeout_s = git_timeout_s

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold the repository lock, which every worktree, branch and tag change takes.

        It is not reentrant: a second lock() in the same process waits for the first and times
        out.
        """
        with file_lock(self._git_common_dir / _LOCK, timeout_s=self.lock_timeout_s):
            yield

    def create(self, lane: Lane.ValueType, experiment_id: str) -> Worktree:
        """Create the experiment's worktree and branch from ``best/<lane>``, and both pack
        directories in it.

        Its HEAD, see :meth:`head`, is the best version it was built on. Raises ValueError for an
        invalid experiment ID or lane, and WorkspaceError if the lane has no best tag, the
        worktree or branch already exists, or a pack directory would lie behind a symlink.
        """
        check_id(experiment_id, "experiment_id")
        info = lane_info(lane)
        worktree = Worktree(self.root / info.name / experiment_id, lane, experiment_id)
        with self.lock():
            base = self.best_commit(lane)
            if base is None:
                raise WorkspaceError(f"the lane has no {info.best_tag} tag to build on")
            if worktree.path.exists() or worktree.path.is_symlink():
                raise WorkspaceError(f"{worktree.path} already exists")
            if self._resolve(f"refs/heads/{worktree.branch}") is not None:
                raise WorkspaceError(
                    f"the branch {worktree.branch} already exists: an experiment ID is used once"
                )
            worktree.path.parent.mkdir(parents=True, exist_ok=True)
            self._git(
                ["worktree", "add", "--quiet", "-b", worktree.branch, str(worktree.path), base]
            )
            try:
                _make_pack_dirs(worktree)
            except BaseException:
                self._discard(worktree)
                raise
        return worktree

    def head(self, worktree: Worktree) -> str:
        """Return the commit the worktree has checked out."""
        return self._git(["rev-parse", "--verify", "HEAD^{commit}"], cwd=worktree.path).strip()

    def worktrees(self, lane: Lane.ValueType) -> list[Worktree]:
        """Return the worktrees git knows of in the lane's directory, by directory, including any
        whose directory is gone, in the order git lists them.

        A worktree not on its experiment's branch is listed too, so that it can be removed, and
        logged as a warning.
        """
        info = lane_info(lane)
        listed = []
        for path, branch in self._registered():
            if path.parent != self.root / info.name:
                continue
            expected = f"refs/heads/{info.branch(path.name)}"
            if branch != expected:
                _logger.warning(
                    "the worktree %s is on %s, not on %s", path, branch or "no branch", expected
                )
            listed.append(Worktree(path, lane, path.name))
        return listed

    def check_registered(self, worktree: Worktree) -> None:
        """Raise WorkspaceError unless git knows ``worktree`` as the worktree of its experiment:
        ``<root>/<lane>/<experiment_id>``. Raises ValueError for an invalid experiment ID.
        """
        check_id(worktree.experiment_id, "experiment_id")
        expected = self.root / lane_info(worktree.lane).name / worktree.experiment_id
        registered = {path for path, _ in self._registered()}
        if worktree.path != expected or worktree.path not in registered:
            raise WorkspaceError(
                f"{worktree.path} is not a registered worktree of experiment "
                f"{worktree.experiment_id}"
            )

    def remove(self, worktree: Worktree) -> None:
        """Remove the worktree's directory, untracked files included, and keep its branch.

        Removing it again does nothing. Raises WorkspaceError, deleting nothing, for a path
        outside the root, or for a directory git does not know as a worktree.
        """
        if not worktree.path.is_relative_to(self.root):
            raise WorkspaceError(f"{worktree.path} lies outside the worktree root {self.root}")
        with self.lock():
            if worktree.path in {path for path, _ in self._registered()}:
                if worktree.path.exists():  # Deleted here, so that git never reads its files.
                    shutil.rmtree(worktree.path, onexc=_writable_and_retry)
                # Twice forced: also when the worktree is locked.
                self._git(["worktree", "remove", "--force", "--force", str(worktree.path)])
            elif worktree.path.exists() or worktree.path.is_symlink():
                raise WorkspaceError(
                    f"{worktree.path} is not a registered worktree, so it was left alone"
                )

    def best_commit(self, lane: Lane.ValueType) -> str | None:
        """Return the commit that ``best/<lane>`` marks, or None if the tag does not exist."""
        return self._resolve(f"refs/tags/{lane_info(lane).best_tag}^{{commit}}")

    def set_best(self, lane: Lane.ValueType, commit: str, *, expected: str | None) -> None:
        """Move ``best/<lane>`` to ``commit``, the candidate just accepted.

        The tag must mark ``expected`` now, the best the candidate was built on, and ``commit``
        must descend from it; with ``expected`` None, the tag must not exist yet. Otherwise
        WorkspaceError is raised and the tag is left as it was.
        """
        tag = lane_info(lane).best_tag
        with self.lock():
            new = self._commit_of(commit)
            current = self._resolve(f"refs/tags/{tag}")  # The ref itself, maybe a tag object.
            if expected is None:
                if current is not None:
                    raise WorkspaceError(f"{tag} already exists")
                old = "0" * len(new)  # update-ref's way to require that the ref is absent.
            else:
                base = self._commit_of(expected)
                if current is None or self.best_commit(lane) != base:
                    raise WorkspaceError(f"{tag} no longer marks the expected {base}")
                if not self._descends(new, base):
                    raise WorkspaceError(f"{new} does not descend from {tag}, {base}")
                old = current
            try:  # update-ref moves the tag only if it still holds old.
                self._git(["update-ref", f"refs/tags/{tag}", new, old])
            except GitError as error:
                raise WorkspaceError(f"{tag} moved while it was being set: {error}") from error

    @cached_property
    def _git_common_dir(self) -> Path:
        output = self._git(["rev-parse", "--path-format=absolute", "--git-common-dir"])
        return Path(output.strip())

    def _git(self, args: list[str], *, cwd: Path | None = None) -> str:
        return run_git(args, cwd=self.repo if cwd is None else cwd, timeout_s=self.git_timeout_s)

    def _resolve(self, revision: str) -> str | None:
        """Return the object ``revision`` names, or None if it names nothing."""
        output = self._git_unless_no(["rev-parse", "--verify", "--quiet", revision])
        return None if output is None else output.strip()

    def _commit_of(self, revision: str) -> str:
        commit = self._resolve(f"{revision}^{{commit}}")
        if commit is None:
            raise WorkspaceError(f"{revision} is not a commit")
        return commit

    def _descends(self, commit: str, ancestor: str) -> bool:
        return self._git_unless_no(["merge-base", "--is-ancestor", ancestor, commit]) is not None

    def _discard(self, worktree: Worktree) -> None:
        """Undo a create that failed: remove the new worktree and its branch, which holds no
        commit of its own yet. A failure is logged, so that the create's own error is raised.
        """
        steps = (
            ["worktree", "remove", "--force", "--force", str(worktree.path)],
            ["branch", "--delete", "--force", worktree.branch],
        )
        for step in steps:
            try:
                self._git(step)
            except GitError as error:
                _logger.warning("could not undo a failed create: %s", error)

    def _git_unless_no(self, args: list[str]) -> str | None:
        return run_git_unless_no(args, cwd=self.repo, timeout_s=self.git_timeout_s)

    def _registered(self) -> list[tuple[Path, str | None]]:
        """The worktrees git knows of, each with the branch ref it has checked out, if any."""
        found: list[tuple[Path, str | None]] = []
        path: Path | None = None
        branch: str | None = None
        for field in self._git(["worktree", "list", "--porcelain", "-z"]).split("\0"):
            if field.startswith("worktree "):
                path, branch = Path(field.removeprefix("worktree ")), None
            elif field.startswith("branch "):
                branch = field.removeprefix("branch ")
            elif not field and path is not None:  # An empty field ends each worktree's record.
                found.append((path, branch))
                path = None
        return found


def check_pack_files(worktree: Worktree) -> None:
    """Refuse a pack that holds anything git on the host could be made to act on.

    Call it before any git command runs in a candidate's worktree. Every entry under the pack
    directories is examined with lstat, and nothing is followed. Raises UnsafePackError naming
    every nested repository (a .git file or directory), each of git's .gitattributes, .gitmodules
    and .gitignore, every symlink, and anything that is neither a regular file nor a directory.
    """
    problems = []
    for pack in worktree.pack_dirs:
        if not pack.exists() and not pack.is_symlink():
            continue
        if pack.is_symlink() or not pack.is_dir():
            problems.append(f"{pack.relative_to(worktree.path)}: not a directory")
            continue
        for directory, subdirectories, files in os.walk(pack):  # Never enters a symlink.
            for name in sorted([*subdirectories, *files]):
                path = Path(directory) / name
                problem = _problem(name, path.lstat().st_mode)
                if problem:
                    problems.append(f"{path.relative_to(worktree.path)}: {problem}")
            subdirectories[:] = [name for name in subdirectories if name not in GIT_FILES]
    if problems:
        raise UnsafePackError(problems)


def _problem(name: str, mode: int) -> str | None:
    if name == ".git":
        return "a nested repository"
    if name in GIT_FILES:
        return "git's own file"
    if stat.S_ISLNK(mode):
        return "a symlink"
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        return "not a regular file"
    return None


def _writable_and_retry(function: Callable[[str], object], path: str, error: BaseException) -> None:
    """Let rmtree delete what the sandbox made read-only: open up the parent and try once more,
    which raises if that was not the problem.
    """
    Path(path).parent.chmod(0o700)
    function(path)


def _make_pack_dirs(worktree: Worktree) -> None:
    """Create the worktree's pack directories, refusing any path through a symlink.

    A symlinked pack directory would let the sandbox's writable mount, and the commit, reach
    outside the worktree.
    """
    for directory in worktree.pack_dirs:
        parts = directory.relative_to(worktree.path).parts
        for depth in range(1, len(parts) + 1):
            prefix = worktree.path.joinpath(*parts[:depth])
            if prefix.is_symlink():
                raise WorkspaceError(f"{prefix} is a symlink; a pack directory must be real")
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise WorkspaceError(
                f"cannot create the pack directory {directory}: {error}"
            ) from error
