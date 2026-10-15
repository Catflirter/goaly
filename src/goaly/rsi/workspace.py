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

A candidate is evaluated only from a snapshot of a commit, never from its live worktree: the
guards, the unit-test guard, goaly-smoke and screening all see the same immutable files. Host
git reads a worktree only once its sandbox has stopped and been removed (spec §8.8):
:meth:`Workspace.pin`, for goaly-smoke, and the commit helper both check that no container of
the experiment exists before they run git there.

For T22: best/<lane> can move to a commit that does not descend from the remote's tag, so
pushing it must force the tag.
"""

import logging
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath

from goaly.contracts.gen.experiment_pb2 import Lane
from goaly.rsi.git import GitError, run_git, run_git_unless_no
from goaly.rsi.ids import check_id
from goaly.rsi.lanes import lane_info
from goaly.rsi.locks import file_lock
from goaly.rsi.sandbox import containers

__all__ = [
    "DEFAULT_ROOT",
    "GIT_FILES",
    "REGULAR_MODES",
    "SandboxProbe",
    "UnsafePackError",
    "Workspace",
    "WorkspaceError",
    "Worktree",
    "check_modes",
    "check_pack_files",
    "mode_problem",
    "pack_modes",
]

DEFAULT_ROOT = Path("~/goaly-rsi-worktrees")
_LOCK = "goaly-rsi.lock"  # In the main repository's git directory, shared by all its worktrees.
# Files that tell git how to treat a tree; a pack may hold none of them.
GIT_FILES = frozenset({".git", ".gitattributes", ".gitmodules", ".gitignore"})
# git's modes for a file; 120000 is a symlink and 160000 a gitlink.
REGULAR_MODES = frozenset({"100644", "100755"})
_SNAPSHOT_LIMIT = 2 * 1024**3  # The most bytes a snapshot extracts: far more than the repository.

# The containers of an experiment, running or not; the default asks docker.
type SandboxProbe = Callable[[str], Sequence[str]]

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
    then raise goaly.rsi.locks.LockTimeoutError. ``sandbox_probe`` lists an experiment's
    containers, running or not; it defaults to goaly.rsi.sandbox.containers.
    """

    def __init__(
        self,
        repo: Path,
        *,
        root: Path | None = None,
        lock_timeout_s: float = 120.0,
        git_timeout_s: float = 120.0,
        sandbox_probe: SandboxProbe | None = None,
    ) -> None:
        self.repo = repo.resolve()
        self.root = (DEFAULT_ROOT if root is None else root).expanduser().resolve()
        if self.root.is_relative_to(self.repo):
            raise ValueError(f"the worktree root {self.root} must lie outside {self.repo}")
        self.lock_timeout_s = lock_timeout_s
        self.git_timeout_s = git_timeout_s
        self._sandbox_probe = sandbox_probe

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
            try:
                # git can fail, or time out, after it has made the branch.
                self._git(
                    ["worktree", "add", "--quiet", "-b", worktree.branch, str(worktree.path), base]
                )
                _make_pack_dirs(worktree)
            except BaseException:
                self._discard(worktree)
                raise
        return worktree

    def head(self, worktree: Worktree) -> str:
        """Return the commit the worktree has checked out."""
        return self._git(["rev-parse", "--verify", "HEAD^{commit}"], cwd=worktree.path).strip()

    @contextmanager
    def snapshot(self, commit: str) -> Iterator[Path]:
        """Yield a read-only copy of ``commit``'s files, in a new temporary directory that is
        removed afterwards.

        It is written from the commit's tree objects, so it holds exactly what the commit holds:
        no attribute such as export-ignore or export-subst changes it, nothing untracked or
        ignored is in it, and nothing the live worktree gains later. File modes are kept, less
        their write bits. Raises WorkspaceError if ``commit`` is not a commit, its tree holds a
        path or link that leads out of it, or it is larger than any repository should be.
        """
        sha = self._commit_of(commit)
        with tempfile.TemporaryDirectory(prefix=f"goaly-rsi-snapshot-{sha[:12]}-") as scratch:
            root = Path(scratch) / "tree"
            root.mkdir()
            self._extract(sha, root)
            _set_writable(root, writable=False)
            try:
                yield root
            finally:
                _set_writable(root, writable=True)  # So that the directory can be removed.

    def pin(self, worktree: Worktree) -> str:
        """Commit the worktree's pack directories as they stand, on top of its HEAD, and return
        the commit, which no branch holds.

        Call it only once the worktree's sandbox has stopped and been removed (spec §8.8): it
        checks that no container of the experiment exists before any git command runs in the
        worktree, and then walks the packs with lstat. Ignored files are left out, and so is any
        change outside the pack. The worktree, its index and its branch stay as they are. Raises
        WorkspaceError for a worktree that is not registered or whose sandbox still exists, and
        UnsafePackError, pinning nothing, if a pack holds anything but regular files and
        directories, as the commit helper does.
        """
        with self.lock():
            self.check_registered(worktree)
            self.check_stopped(worktree)
            check_pack_files(worktree)  # Before any git command runs in the worktree.
            head = self._git(
                ["rev-parse", "--verify", "--end-of-options", "HEAD^{commit}"], cwd=worktree.path
            ).strip()
            with tempfile.TemporaryDirectory(prefix="goaly-rsi-pin-") as scratch:
                index = {"GIT_INDEX_FILE": str(Path(scratch) / "index")}  # Never the worktree's.
                self._git(["read-tree", head], cwd=worktree.path, env=index)
                packs = [
                    str(relative)
                    for relative, directory in zip(
                        lane_info(worktree.lane).pack_dirs, worktree.pack_dirs, strict=True
                    )
                    if directory.exists()
                ]
                if packs:  # git add without a path would add the whole worktree.
                    self._git(["add", "--all", "--", *packs], cwd=worktree.path, env=index)
                staged = pack_modes(worktree, staged=True, env=index, timeout_s=self.git_timeout_s)
                check_modes(staged, "staged")
                tree = self._git(["write-tree"], cwd=worktree.path, env=index).strip()
            message = f"goaly-smoke: {worktree.experiment_id} as it stands"
            return self._git(
                ["commit-tree", tree, "-p", head, f"-m{message}"], cwd=worktree.path
            ).strip()

    def check_stopped(self, worktree: Worktree) -> None:
        """Raise WorkspaceError while any container of the worktree's experiment exists, running
        or not, since host git must never read a worktree that a sandbox could still change.
        Raises goaly.rsi.sandbox.SandboxError if docker cannot tell.
        """
        if self._sandbox_probe is None:
            found = containers(experiment_id=worktree.experiment_id)
        else:
            found = list(self._sandbox_probe(worktree.experiment_id))
        if found:
            raise WorkspaceError(
                f"the sandbox of experiment {worktree.experiment_id} is not stopped and removed: "
                f"{len(found)} container{'' if len(found) == 1 else 's'} left"
            )

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

        Whatever modes the sandbox left on the pack's files and directories, they are deleted.
        Removing it again does nothing. Raises WorkspaceError, deleting nothing, for a path
        outside the root, or for a directory git does not know as a worktree; and WorkspaceError
        if its files cannot be deleted.
        """
        if not worktree.path.is_relative_to(self.root):
            raise WorkspaceError(f"{worktree.path} lies outside the worktree root {self.root}")
        with self.lock():
            if worktree.path in {path for path, _ in self._registered()}:
                if worktree.path.exists():  # Deleted here, so that git never reads its files.
                    _remove_tree(worktree.path)
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

    def _git(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> str:
        return run_git(
            args,
            cwd=self.repo if cwd is None else cwd,
            timeout_s=self.git_timeout_s,
            env=env,
            input_text=input_text,
        )

    def _resolve(self, revision: str) -> str | None:
        """Return the object ``revision`` names, or None if it names nothing."""
        output = self._git_unless_no(
            ["rev-parse", "--verify", "--quiet", "--end-of-options", revision]
        )
        return None if output is None else output.strip()

    def _extract(self, commit: str, root: Path) -> None:
        """Write the files of ``commit``'s tree under ``root``, from git's objects."""
        listing = self._git(["ls-tree", "-r", "-z", "--full-tree", commit])
        entries: list[tuple[str, str, str, PurePosixPath]] = []  # (mode, kind, object, path)
        for entry in listing.split("\0"):
            if entry:
                meta, _, listed = entry.partition("\t")
                mode, kind, obj = meta.split(" ")
                entries.append((mode, kind, obj, _inside(listed, commit)))
        contents = self._blobs([obj for _, kind, obj, _ in entries if kind == "blob"], commit)
        links: list[tuple[Path, bytes, PurePosixPath]] = []
        for mode, kind, obj, relative in entries:
            target = root.joinpath(*relative.parts)
            if kind == "commit":  # A gitlink: git archive gives an empty directory too.
                target.mkdir(parents=True, exist_ok=True)
            elif mode == "120000":
                links.append((target, contents[obj], relative))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
                with os.fdopen(os.open(target, flags, 0o600), "wb") as file:
                    file.write(contents[obj])
                target.chmod(0o755 if mode == "100755" else 0o644)
        for target, data, relative in links:  # Last, so that no file is written through one.
            link = data.decode("utf-8", "surrogateescape")
            resolved = os.path.normpath((relative.parent / link).as_posix())
            if link.startswith("/") or resolved.split("/")[0] == "..":
                raise WorkspaceError(f"cannot snapshot {commit}: {relative} links out of the tree")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(link)

    def _blobs(self, objects: list[str], commit: str) -> dict[str, bytes]:
        """The contents of git's blob ``objects``, read in one batch."""
        if not objects:
            return {}
        listed = "\n".join(objects) + "\n"
        sizes = self._git(["cat-file", "--batch-check=%(objectsize)"], input_text=listed)
        if sum(int(size) for size in sizes.split()) > _SNAPSHOT_LIMIT:
            raise WorkspaceError(f"cannot snapshot {commit}: it is larger than any repository")
        raw = self._git(["cat-file", "--batch"], input_text=listed).encode(
            "utf-8", "surrogateescape"
        )  # git's output, byte for byte.
        contents: dict[str, bytes] = {}
        offset = 0
        for _ in objects:  # Each is "<object> blob <size>\n<content>\n".
            header_end = raw.index(b"\n", offset)
            obj, _kind, size = raw[offset:header_end].decode().split(" ")
            start = header_end + 1
            contents[obj] = raw[start : start + int(size)]
            offset = start + int(size) + 1
        return contents

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
    and .gitignore, every symlink, anything that is neither a regular file nor a directory, and
    every file or directory this user cannot read, which the walk could not examine and git could
    not add.
    """
    problems: dict[str, None] = {}  # In the order found, each once.

    def note(path: Path, problem: str) -> None:
        problems.setdefault(f"{path.relative_to(worktree.path)}: {problem}")

    def unreadable(error: OSError) -> None:
        note(Path(error.filename), _UNREADABLE)

    for pack in worktree.pack_dirs:
        if not pack.exists() and not pack.is_symlink():
            continue
        if pack.is_symlink() or not pack.is_dir():
            note(pack, "not a directory")
            continue
        # os.walk never enters a symlink, and reports a directory it cannot list.
        for directory, subdirectories, files in os.walk(pack, onerror=unreadable):
            for name in sorted([*subdirectories, *files]):
                path = Path(directory) / name
                try:
                    mode = path.lstat().st_mode
                except OSError:  # Its directory can be listed but not searched.
                    note(path, _UNREADABLE)
                    continue
                problem = _problem(path, mode)
                if problem:
                    note(path, problem)
            subdirectories[:] = [name for name in subdirectories if name not in GIT_FILES]
    if problems:
        raise UnsafePackError(list(problems))


def pack_modes(
    worktree: Worktree,
    *,
    staged: bool = False,
    env: Mapping[str, str] | None = None,
    timeout_s: float = 120.0,
) -> dict[str, str]:
    """Return the git mode of every path in the worktree's packs: as HEAD holds it, or as the
    index does with ``staged``. ``env`` can name another index. Neither reads a file of the
    worktree.
    """
    packs = [str(pack) for pack in lane_info(worktree.lane).pack_dirs]
    if staged:  # Each entry is "<mode> <object> <stage>\t<path>".
        command = ["ls-files", "--stage", "-z", "--", *packs]
    else:  # Each entry is "<mode> <type> <object>\t<path>".
        command = ["ls-tree", "-r", "-z", "HEAD", "--", *packs]
    output = run_git(command, cwd=worktree.path, env=env, timeout_s=timeout_s)
    entries = (entry.partition("\t") for entry in output.split("\0") if entry)
    return {path: meta.split(" ")[0] for meta, _, path in entries}


def check_modes(modes: Mapping[str, str], where: str) -> None:
    """Raise UnsafePackError for every path in ``modes`` that is not a regular file, or that is
    one of git's own files, saying ``where`` it was found.
    """
    problems = [
        f"{path}: {problem} {where}"
        for path, mode in sorted(modes.items())
        if (problem := mode_problem(path, mode))
    ]
    if problems:
        raise UnsafePackError(problems)


def mode_problem(path: str, mode: str) -> str | None:
    """Why a tree entry at ``path`` with git's ``mode`` may not be in a pack, or None: a
    gitlink, anything but a regular file, or one of git's own files in any component.
    """
    if mode == "160000":
        return "a gitlink"
    if mode not in REGULAR_MODES:
        return f"mode {mode}"
    if GIT_FILES.intersection(PurePosixPath(path).parts):
        return "git's own file"
    return None


def _inside(name: str, commit: str) -> PurePosixPath:
    """A tree path, refused if it could lead out of the snapshot or into git's own files."""
    if name.startswith("/") or any(part in ("", ".", "..", ".git") for part in name.split("/")):
        raise WorkspaceError(f"cannot snapshot {commit}: its tree holds the path {name!r}")
    return PurePosixPath(name)


_UNREADABLE = "cannot be read"


def _problem(path: Path, mode: int) -> str | None:
    if path.name == ".git":
        return "a nested repository"
    if path.name in GIT_FILES:
        return "git's own file"
    if stat.S_ISLNK(mode):
        return "a symlink"
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        return "not a regular file"
    access = os.R_OK | os.X_OK if stat.S_ISDIR(mode) else os.R_OK
    if not os.access(path, access, follow_symlinks=False):
        return _UNREADABLE
    return None


def _remove_tree(root: Path) -> None:
    """Delete ``root`` and everything under it, whatever modes the sandbox left: each directory
    first gets its owner's read, write and search bits, top-down and without following links.
    Raises WorkspaceError if that fails.
    """
    try:
        pending = [root]
        while pending:
            directory = pending.pop()
            mode = stat.S_IMODE(directory.lstat().st_mode)
            if mode & stat.S_IRWXU != stat.S_IRWXU:
                directory.chmod(mode | stat.S_IRWXU)
            with os.scandir(directory) as entries:
                pending += [
                    Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=False)
                ]
        shutil.rmtree(root)
    except OSError as error:
        raise WorkspaceError(f"cannot remove {root}: {error}") from error


def _set_writable(root: Path, *, writable: bool) -> None:
    """Give every file and directory under ``root`` its owner's write bit, or take every write
    bit away, keeping the other bits, such as an executable's.
    """
    for directory, _, names in os.walk(root, topdown=False):
        for path in [*(Path(directory) / name for name in names), Path(directory)]:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):  # chmod would follow it.
                continue
            bits = stat.S_IMODE(mode)
            path.chmod(bits | stat.S_IWUSR if writable else bits & ~0o222)


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
