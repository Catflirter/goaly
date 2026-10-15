"""Candidate worktrees, best tags and the repository lock (spec §6.2, §8.2 step 2, §8.7).

Each experiment builds its candidate in its own git worktree, ``<root>/<lane>/<experiment_id>``,
on the branch ``rsi/<lane>/<experiment_id>``, from the lane's best version: the tag
``best/<lane>``. The root lies outside the main repository, so a read-only mount of a worktree
carries nothing else of it, and both of the lane's pack directories exist before the sandbox
mounts them. Removing a worktree keeps its branch, so a rejected candidate's commit stays
reachable.

One lock in the main repository's git directory serializes the worktree, branch and tag
operations of every lane process. Reading a ref needs no lock: git updates refs atomically.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import Lane
from goaly.rsi.git import GitError, run_git
from goaly.rsi.lanes import lane_info
from goaly.rsi.ledger import check_id
from goaly.rsi.locks import file_lock

__all__ = ["DEFAULT_ROOT", "Workspace", "WorkspaceError", "Worktree"]

DEFAULT_ROOT = Path("~/goaly-rsi-worktrees")
_LOCK = "goaly-rsi.lock"  # In the main repository's git directory, shared by all its worktrees.


class WorkspaceError(Exception):
    """A worktree, branch or tag operation was refused; nothing was changed."""


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
        """Hold the repository lock, which every worktree, branch and tag change takes."""
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
                self._git(["worktree", "remove", "--force", "--force", str(worktree.path)])
                raise
        return worktree

    def head(self, worktree: Worktree) -> str:
        """Return the commit the worktree has checked out."""
        return self._git(["rev-parse", "--verify", "HEAD^{commit}"], cwd=worktree.path).strip()

    def worktrees(self, lane: Lane.ValueType) -> list[Worktree]:
        """Return the lane's worktrees that git knows of, including any whose directory is gone,
        in the order git lists them.
        """
        info = lane_info(lane)
        return [
            Worktree(path, lane, path.name)
            for path, branch in self._registered()
            if path.parent == self.root / info.name
            and branch == f"refs/heads/{info.branch(path.name)}"
        ]

    def remove(self, worktree: Worktree) -> None:
        """Remove the worktree's directory, untracked files included, and keep its branch.

        Removing it again does nothing. Raises WorkspaceError, deleting nothing, for a path
        outside the root, or for a directory git does not know as a worktree.
        """
        if not worktree.path.is_relative_to(self.root):
            raise WorkspaceError(f"{worktree.path} lies outside the worktree root {self.root}")
        with self.lock():
            if worktree.path in {path for path, _ in self._registered()}:
                # Twice forced: also when the worktree is locked or its directory is gone.
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
            self._git(["update-ref", f"refs/tags/{tag}", new, old])

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

    def _git_unless_no(self, args: list[str]) -> str | None:
        """Run a git command that answers no by exiting with 1: return None then, its output
        otherwise. Any other failure raises GitError.
        """
        try:
            return self._git(args)
        except GitError as error:
            if error.returncode == 1:
                return None
            raise

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
