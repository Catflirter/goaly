"""Tests for the checks that keep git on the host away from what a candidate planted in its pack.

The sandbox can write any file into the pack directories, a nested repository or a symlink
included. git on the host must never act on those, so the pack is checked before any git
command runs in the worktree, and what git stages is checked again before anything is committed.
"""

import os
import shutil
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL
from goaly.rsi import commit, workspace
from goaly.rsi.commit import commit_candidate
from goaly.rsi.git import run_git
from goaly.rsi.workspace import (
    UnsafePackError,
    Workspace,
    WorkspaceError,
    Worktree,
    check_pack_files,
)
from tests.rsi.gitrepo import git, make_repo, write

PACK = "src/goaly/harness/packs/effisql"
PACK_TESTS = "tests/harness/packs/effisql"

# Root ignores permission bits, so the tests of unreadable entries need another user.
needs_non_root = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


@pytest.fixture
def space(repo: Path, tmp_path: Path) -> Workspace:
    return Workspace(repo, root=tmp_path / "worktrees")


@pytest.fixture
def worktree(space: Workspace) -> Worktree:
    return space.create(LANE_EFFISQL, "exp-1")


@pytest.fixture
def readable_again(worktree: Worktree) -> Iterator[None]:
    """Give everything under the worktree its owner's permissions back after the test, even if
    it failed, so that pytest can delete its temporary directory.
    """
    yield
    pending = [worktree.path] if worktree.path.is_dir() else []
    while pending:
        directory = pending.pop()
        directory.chmod(0o755)  # Before listing it, which a mode of 000 would refuse.
        for entry in directory.iterdir():
            if entry.is_symlink():
                continue
            if entry.is_dir():
                pending.append(entry)
            elif entry.is_file():
                entry.chmod(0o644)


@pytest.fixture
def git_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record where each git command of goaly.rsi runs."""
    calls: list[Path] = []

    def spy(args: Sequence[str], *, cwd: Path, **options: Any) -> str:
        calls.append(cwd)
        return run_git(args, cwd=cwd, **options)

    for module in (commit, workspace):
        monkeypatch.setattr(module, "run_git", spy)
    return calls


def ran_in(calls: list[Path], worktree: Worktree) -> list[Path]:
    return [cwd for cwd in calls if cwd.is_relative_to(worktree.path)]


def commit_gitlink(repo: Path, path: str) -> None:
    """Make the best version hold a gitlink at ``path``, as a past candidate could have."""
    head = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},{path}")
    git(repo, "commit", "--quiet", "--message=a gitlink in the pack")
    git(repo, "tag", "--force", "best/effisql")


def try_commit(space: Workspace, worktree: Worktree) -> None:
    commit_candidate(space, worktree, hypothesis_id="hyp-1", summary="x")


# What the sandbox can plant in a pack directory.


def nested_repository(worktree: Worktree) -> None:
    nested = worktree.path / PACK / "sub"
    nested.mkdir()
    git(nested, "init", "--quiet")


def nested_git_file(worktree: Worktree) -> None:
    nested = worktree.path / PACK / "sub"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: /tmp/elsewhere\n", encoding="utf-8")


def symlink_to_the_env_file(worktree: Worktree) -> None:
    secrets = Path.home() / ".config" / "goaly" / ".env"
    secrets.parent.mkdir(parents=True)
    secrets.write_text("GOALY_API_KEY=not-a-real-key\n", encoding="utf-8")
    (worktree.path / PACK / "notes.md").symlink_to(secrets)


def symlink_to_core(worktree: Worktree) -> None:
    (worktree.path / PACK / "core").symlink_to("../../core")


def fifo(worktree: Worktree) -> None:
    os.mkfifo(worktree.path / PACK_TESTS / "pipe")


def unreadable_directory(mode: int) -> Callable[[Worktree], None]:
    def plant(worktree: Worktree) -> None:
        write(worktree.path, {f"{PACK}/hidden/notes.md": "x\n"})
        (worktree.path / PACK / "hidden").chmod(mode)

    return plant


def unreadable_file(worktree: Worktree) -> None:
    write(worktree.path, {f"{PACK}/notes.md": "x\n"})
    (worktree.path / PACK / "notes.md").chmod(0o000)


def git_control_file(name: str) -> Callable[[Worktree], None]:
    def plant(worktree: Worktree) -> None:
        write(worktree.path, {f"{PACK}/{name}": "* -text\n"})

    return plant


@pytest.mark.parametrize(
    ("plant", "problem"),
    [
        (nested_repository, f"{PACK}/sub/.git: a nested repository"),
        (nested_git_file, f"{PACK}/sub/.git: a nested repository"),
        (symlink_to_the_env_file, f"{PACK}/notes.md: a symlink"),
        (symlink_to_core, f"{PACK}/core: a symlink"),
        (fifo, f"{PACK_TESTS}/pipe: not a regular file"),
        (git_control_file(".gitattributes"), f"{PACK}/.gitattributes: git's own file"),
        (git_control_file(".gitmodules"), f"{PACK}/.gitmodules: git's own file"),
        (git_control_file(".gitignore"), f"{PACK}/.gitignore: git's own file"),
        pytest.param(
            unreadable_directory(0o000),
            f"{PACK}/hidden: cannot be read",
            marks=needs_non_root,
            id="unreadable-directory",
        ),
        pytest.param(
            unreadable_directory(0o111),
            f"{PACK}/hidden: cannot be read",
            marks=needs_non_root,
            id="unlistable-directory",
        ),
        pytest.param(
            unreadable_directory(0o444),
            f"{PACK}/hidden: cannot be read",
            marks=needs_non_root,
            id="unsearchable-directory",
        ),
        pytest.param(
            unreadable_file,
            f"{PACK}/notes.md: cannot be read",
            marks=needs_non_root,
            id="unreadable-file",
        ),
    ],
)
@pytest.mark.usefixtures("readable_again")
def test_an_unsafe_pack_is_refused_before_git_runs_in_the_worktree(
    space: Workspace,
    worktree: Worktree,
    git_calls: list[Path],
    plant: Callable[[Worktree], None],
    problem: str,
) -> None:
    head = space.head(worktree)
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n"})
    plant(worktree)
    git_calls.clear()

    with pytest.raises(UnsafePackError) as caught:
        try_commit(space, worktree)

    assert problem in caught.value.problems
    assert ran_in(git_calls, worktree) == []
    assert space.head(worktree) == head


def test_check_pack_files_passes_regular_files_and_directories(worktree: Worktree) -> None:
    write(worktree.path, {f"{PACK}/skills/plans.md": "# Plans\n", f"{PACK_TESTS}/test_x.py": ""})
    (worktree.path / PACK / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (worktree.path / PACK / "run.sh").chmod(0o755)

    check_pack_files(worktree)


def test_the_two_step_gitlink_attack_fails_safely(
    repo: Path, space: Workspace, git_calls: list[Path]
) -> None:
    # Step 1, as a past candidate might have committed it: a gitlink in the pack.
    commit_gitlink(repo, f"{PACK}/sub")
    later = space.create(LANE_EFFISQL, "exp-2")
    head = space.head(later)
    # Step 2: the next candidate turns the gitlink's directory into a repository of its own.
    (later.path / PACK / "sub").mkdir(exist_ok=True)
    git(later.path / PACK / "sub", "init", "--quiet")
    write(later.path, {f"{PACK}/sub/notes.md": "planted\n", f"{PACK}/prompt.md": "Be faster.\n"})
    git_calls.clear()

    with pytest.raises(UnsafePackError, match="a nested repository"):
        try_commit(space, later)

    assert ran_in(git_calls, later) == []
    assert space.head(later) == head


def test_a_gitlink_the_best_version_holds_is_refused(repo: Path, space: Workspace) -> None:
    commit_gitlink(repo, f"{PACK}/sub")
    later = space.create(LANE_EFFISQL, "exp-2")
    head = space.head(later)
    write(later.path, {f"{PACK}/prompt.md": "Be faster.\n"})

    with pytest.raises(UnsafePackError, match=f"{PACK}/sub: a gitlink"):
        try_commit(space, later)

    assert space.head(later) == head


def test_what_git_staged_is_checked_again_before_the_commit(
    space: Workspace, worktree: Worktree, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Were the first check to miss a symlink, the staged check still refuses it.
    monkeypatch.setattr(commit, "check_pack_files", lambda worktree: None)
    head = space.head(worktree)
    (worktree.path / PACK / "link.md").symlink_to("prompt.md")

    with pytest.raises(UnsafePackError, match=f"{PACK}/link.md: mode 120000"):
        try_commit(space, worktree)

    assert space.head(worktree) == head
    assert git(worktree.path, "diff", "--cached", "--name-only") == ""  # Unstaged again.


def test_a_pack_directory_replaced_by_a_symlink_is_refused(
    worktree: Worktree, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (worktree.path / PACK_TESTS).rmdir()
    (worktree.path / PACK_TESTS).symlink_to(elsewhere)

    with pytest.raises(UnsafePackError, match=f"{PACK_TESTS}: not a directory"):
        check_pack_files(worktree)


@needs_non_root
@pytest.mark.parametrize("mode", [0o000, 0o111, 0o444, 0o555])
@pytest.mark.usefixtures("readable_again")
def test_remove_deletes_what_the_sandbox_made_unreadable_or_read_only(
    space: Workspace, worktree: Worktree, repo: Path, mode: int
) -> None:
    write(worktree.path, {f"{PACK}/locked/deep/notes.md": "x\n", f"{PACK_TESTS}/test_x.py": ""})
    for directory in ("locked/deep", "locked", ""):  # The deepest first, the pack itself last.
        (worktree.path / PACK / directory).chmod(mode)
    (worktree.path / PACK_TESTS).chmod(mode)

    space.remove(worktree)

    assert not worktree.path.exists()
    assert str(worktree.path) not in git(repo, "worktree", "list")


def test_a_pack_that_cannot_be_removed_is_a_workspace_error(
    space: Workspace, worktree: Worktree, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(path: Path) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(shutil, "rmtree", fail)

    with pytest.raises(WorkspaceError, match="cannot remove"):
        space.remove(worktree)


def test_a_git_file_the_best_version_holds_is_refused(repo: Path, space: Workspace) -> None:
    write(repo, {f"{PACK}/.gitattributes": "* -text\n"})
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message=attributes in the pack")
    git(repo, "tag", "--force", "best/effisql")
    later = space.create(LANE_EFFISQL, "exp-2")
    (later.path / PACK / ".gitattributes").unlink()  # Only the best version holds it now.
    write(later.path, {f"{PACK}/prompt.md": "Be faster.\n"})

    with pytest.raises(UnsafePackError, match=r"\.gitattributes: git's own file in its best"):
        try_commit(space, later)


def test_with_neither_pack_directory_nothing_is_staged(
    repo: Path, space: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    git(repo, "rm", "--quiet", "-r", PACK)
    git(repo, "commit", "--quiet", "--message=no pack")
    git(repo, "tag", "--force", "best/effisql")
    bare = space.create(LANE_EFFISQL, "exp-2")
    for pack in bare.pack_dirs:
        pack.rmdir()
    commands: list[list[str]] = []

    def spy(args: Sequence[str], *, cwd: Path, **options: Any) -> str:
        commands.append(list(args))
        return run_git(args, cwd=cwd, **options)

    monkeypatch.setattr(commit, "run_git", spy)

    assert commit_candidate(space, bare, hypothesis_id="hyp-1", summary="x") is None
    assert not [args for args in commands if args[:1] == ["add"]]
