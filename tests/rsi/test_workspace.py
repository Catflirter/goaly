"""Tests for goaly.rsi.workspace: candidate worktrees, best tags and the repository lock."""

import logging
from pathlib import Path
from typing import Any

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, LANE_SWE, LANE_UNSPECIFIED
from goaly.rsi import workspace as workspace_module
from goaly.rsi.git import GitError
from goaly.rsi.locks import LockTimeoutError, file_lock
from goaly.rsi.workspace import Workspace, WorkspaceError, Worktree
from tests.rsi.gitrepo import git, make_repo, write


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


@pytest.fixture
def workspace(repo: Path, tmp_path: Path) -> Workspace:
    return Workspace(repo, root=tmp_path / "worktrees")


def registered(repo: Path) -> list[Path]:
    """The worktrees git knows of, the main one included."""
    listing = git(repo, "worktree", "list", "--porcelain")
    return [
        Path(line.removeprefix("worktree "))
        for line in listing.splitlines()
        if line.startswith("worktree ")
    ]


def add_commit(repo: Path, message: str) -> str:
    git(repo, "commit", "--quiet", "--allow-empty", f"--message={message}")
    return git(repo, "rev-parse", "HEAD").strip()


# Creating worktrees.


def test_create_checks_out_the_lanes_best_on_the_experiments_branch(
    workspace: Workspace, repo: Path, tmp_path: Path
) -> None:
    best = git(repo, "rev-parse", "best/effisql").strip()
    add_commit(repo, "after the best")  # main moves on; the worktree must start from the tag.

    worktree = workspace.create(LANE_EFFISQL, "exp-1")

    assert worktree == Worktree(tmp_path / "worktrees" / "effisql" / "exp-1", LANE_EFFISQL, "exp-1")
    assert worktree.branch == "rsi/effisql/exp-1"
    assert git(worktree.path, "symbolic-ref", "HEAD").strip() == "refs/heads/rsi/effisql/exp-1"
    assert workspace.head(worktree) == best
    assert worktree.path in registered(repo)
    assert (worktree.path / "src/goaly/harness/packs/effisql/prompt.md").is_file()


def test_create_makes_both_pack_directories(workspace: Workspace) -> None:
    worktree = workspace.create(LANE_EFFISQL, "exp-1")

    assert worktree.pack_dirs == (
        worktree.path / "src/goaly/harness/packs/effisql",
        worktree.path / "tests/harness/packs/effisql",
    )
    for directory in worktree.pack_dirs:
        assert directory.is_dir()
        assert not directory.is_symlink()


def test_each_lane_has_its_own_directory_and_names(
    workspace: Workspace, repo: Path, tmp_path: Path
) -> None:
    git(repo, "tag", "best/swe")

    worktree = workspace.create(LANE_SWE, "exp-1")

    assert worktree.path == tmp_path / "worktrees" / "swe" / "exp-1"
    assert worktree.branch == "rsi/swe/exp-1"
    assert (worktree.path / "src/goaly/harness/packs/swefficiency").is_dir()


def test_create_needs_the_lanes_best_tag(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="best/swe"):
        workspace.create(LANE_SWE, "exp-1")


@pytest.mark.parametrize("experiment_id", ["", "a/b", "../up", "-x", "has space"])
def test_create_refuses_an_invalid_experiment_id(workspace: Workspace, experiment_id: str) -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        workspace.create(LANE_EFFISQL, experiment_id)


def test_create_refuses_an_unknown_lane(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="lane"):
        workspace.create(LANE_UNSPECIFIED, "exp-1")


def test_an_experiment_id_is_used_once(workspace: Workspace) -> None:
    first = workspace.create(LANE_EFFISQL, "exp-1")
    with pytest.raises(WorkspaceError, match="exists"):
        workspace.create(LANE_EFFISQL, "exp-1")

    # Removing the worktree keeps its branch, so the ID stays used.
    workspace.remove(first)
    with pytest.raises(WorkspaceError, match="rsi/effisql/exp-1"):
        workspace.create(LANE_EFFISQL, "exp-1")


def test_create_refuses_a_pack_directory_reached_through_a_symlink(
    workspace: Workspace, repo: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    git(repo, "rm", "--quiet", "-r", "src/goaly/harness/packs")
    (repo / "src/goaly/harness/packs").symlink_to(elsewhere)
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message=symlinked packs")
    git(repo, "tag", "--force", "best/effisql")

    with pytest.raises(WorkspaceError, match="symlink"):
        workspace.create(LANE_EFFISQL, "exp-1")

    assert list(elsewhere.iterdir()) == []
    assert not (tmp_path / "worktrees" / "effisql" / "exp-1").exists()
    assert len(registered(repo)) == 1


def test_the_root_must_be_outside_the_repository(repo: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        Workspace(repo, root=repo / "worktrees")


def test_the_default_root_is_in_the_home_directory(repo: Path) -> None:
    assert Workspace(repo).root == Path.home() / "goaly-rsi-worktrees"


def test_git_hooks_do_not_run(workspace: Workspace, repo: Path, tmp_path: Path) -> None:
    marker = tmp_path / "hook-ran"
    hook = repo / ".git" / "hooks" / "post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)

    workspace.create(LANE_EFFISQL, "exp-1")

    assert not marker.exists()


# Listing and removing worktrees.


def test_worktrees_lists_one_lanes_worktrees(workspace: Workspace, repo: Path) -> None:
    git(repo, "tag", "best/swe")
    first = workspace.create(LANE_EFFISQL, "exp-1")
    second = workspace.create(LANE_EFFISQL, "exp-2")
    swe = workspace.create(LANE_SWE, "exp-3")

    assert workspace.worktrees(LANE_EFFISQL) == [first, second]
    assert workspace.worktrees(LANE_SWE) == [swe]


def test_remove_deletes_the_worktree_but_keeps_its_branch_and_commits(
    workspace: Workspace, repo: Path
) -> None:
    worktree = workspace.create(LANE_EFFISQL, "exp-1")
    write(worktree.path, {"src/goaly/harness/packs/effisql/prompt.md": "Be faster.\n"})
    git(worktree.path, "commit", "--quiet", "--all", "--message=candidate")
    candidate = workspace.head(worktree)
    write(worktree.path, {"untracked.txt": "left behind\n"})

    workspace.remove(worktree)

    assert not worktree.path.exists()
    assert worktree.path not in registered(repo)
    assert workspace.worktrees(LANE_EFFISQL) == []
    assert git(repo, "rev-parse", "rsi/effisql/exp-1").strip() == candidate


def test_remove_forgets_a_worktree_whose_directory_is_gone(
    workspace: Workspace, repo: Path
) -> None:
    worktree = workspace.create(LANE_EFFISQL, "exp-1")
    git(repo, "worktree", "lock", str(worktree.path))  # Even a locked one.
    for path in sorted(worktree.path.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() and not path.is_symlink() else path.unlink()
    worktree.path.rmdir()

    workspace.remove(worktree)

    assert worktree.path not in registered(repo)


def test_removing_a_worktree_twice_is_harmless(workspace: Workspace) -> None:
    worktree = workspace.create(LANE_EFFISQL, "exp-1")
    workspace.remove(worktree)

    workspace.remove(worktree)


def test_remove_never_deletes_a_directory_that_is_not_a_worktree(
    workspace: Workspace, tmp_path: Path
) -> None:
    stray = tmp_path / "worktrees" / "effisql" / "exp-9"
    stray.mkdir(parents=True)
    (stray / "keep.txt").write_text("mine\n", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="not a registered worktree"):
        workspace.remove(Worktree(stray, LANE_EFFISQL, "exp-9"))

    assert (stray / "keep.txt").is_file()


def test_remove_refuses_a_path_outside_the_root(workspace: Workspace, tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError, match="outside"):
        workspace.remove(Worktree(tmp_path / "repo", LANE_EFFISQL, "exp-1"))


# The best tag.


def test_best_commit_reads_the_tag(workspace: Workspace, repo: Path) -> None:
    assert workspace.best_commit(LANE_EFFISQL) == git(repo, "rev-parse", "best/effisql").strip()
    assert workspace.best_commit(LANE_SWE) is None


def test_set_best_moves_the_tag_forward_from_the_expected_commit(
    workspace: Workspace, repo: Path
) -> None:
    old = git(repo, "rev-parse", "best/effisql").strip()
    new = add_commit(repo, "accepted candidate")

    workspace.set_best(LANE_EFFISQL, new, expected=old)

    assert workspace.best_commit(LANE_EFFISQL) == new


def test_set_best_refuses_when_the_tag_moved_meanwhile(workspace: Workspace, repo: Path) -> None:
    old = git(repo, "rev-parse", "best/effisql").strip()
    first = add_commit(repo, "first")
    second = add_commit(repo, "second")
    workspace.set_best(LANE_EFFISQL, first, expected=old)

    with pytest.raises(WorkspaceError, match="best/effisql"):
        workspace.set_best(LANE_EFFISQL, second, expected=old)

    assert workspace.best_commit(LANE_EFFISQL) == first


def test_set_best_refuses_a_commit_that_does_not_descend_from_the_best(
    workspace: Workspace, repo: Path
) -> None:
    old = git(repo, "rev-parse", "best/effisql").strip()
    git(repo, "checkout", "--quiet", "--orphan", "unrelated")
    unrelated = add_commit(repo, "unrelated")

    with pytest.raises(WorkspaceError, match="descend"):
        workspace.set_best(LANE_EFFISQL, unrelated, expected=old)

    assert workspace.best_commit(LANE_EFFISQL) == old


def test_set_best_creates_a_missing_tag_only_when_none_is_expected(
    workspace: Workspace, repo: Path
) -> None:
    commit = git(repo, "rev-parse", "HEAD").strip()

    workspace.set_best(LANE_SWE, commit, expected=None)
    assert workspace.best_commit(LANE_SWE) == commit

    with pytest.raises(WorkspaceError, match="best/effisql"):
        workspace.set_best(LANE_EFFISQL, commit, expected=None)


def test_set_best_refuses_something_that_is_not_a_commit(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="not-a-commit"):
        workspace.set_best(LANE_SWE, "not-a-commit", expected=None)


# The repository lock.


def test_worktree_and_tag_operations_wait_for_the_repository_lock(
    repo: Path, tmp_path: Path
) -> None:
    workspace = Workspace(repo, root=tmp_path / "worktrees", lock_timeout_s=0.2)
    commit = git(repo, "rev-parse", "HEAD").strip()

    with file_lock(repo / ".git" / "goaly-rsi.lock"):
        with pytest.raises(LockTimeoutError):
            workspace.create(LANE_EFFISQL, "exp-1")
        with pytest.raises(LockTimeoutError):
            workspace.set_best(LANE_SWE, commit, expected=None)

    assert not (tmp_path / "worktrees" / "effisql" / "exp-1").exists()
    assert workspace.best_commit(LANE_SWE) is None


def test_worktrees_of_a_worktree_share_the_main_repositorys_lock(
    workspace: Workspace, repo: Path, tmp_path: Path
) -> None:
    worktree = workspace.create(LANE_EFFISQL, "exp-1")
    from_worktree = Workspace(worktree.path, root=tmp_path / "worktrees", lock_timeout_s=0.2)

    with file_lock(repo / ".git" / "goaly-rsi.lock"), pytest.raises(LockTimeoutError):
        from_worktree.create(LANE_EFFISQL, "exp-2")


def test_a_pack_directory_that_is_a_file_is_refused(
    workspace: Workspace, repo: Path, tmp_path: Path
) -> None:
    write(repo, {"tests/harness/packs/effisql": "not a directory\n"})
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message=a file where the pack tests go")
    git(repo, "tag", "--force", "best/effisql")

    with pytest.raises(WorkspaceError, match="tests/harness/packs/effisql"):
        workspace.create(LANE_EFFISQL, "exp-1")

    assert not (tmp_path / "worktrees" / "effisql" / "exp-1").exists()
    assert len(registered(repo)) == 1


def test_a_git_failure_other_than_no_is_raised(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()

    with pytest.raises(GitError, match="not a git repository"):
        Workspace(not_a_repo, root=tmp_path / "worktrees").best_commit(LANE_EFFISQL)


# A failed create leaves nothing behind.


def test_a_failed_create_removes_its_worktree_and_its_branch(
    workspace: Workspace, repo: Path, tmp_path: Path
) -> None:
    write(repo, {"tests/harness/packs/effisql": "not a directory\n"})
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message=a file where the pack tests go")
    git(repo, "tag", "--force", "best/effisql")

    with pytest.raises(WorkspaceError):
        workspace.create(LANE_EFFISQL, "exp-1")

    assert git(repo, "branch", "--list", "rsi/effisql/exp-1") == ""


def test_a_failed_cleanup_after_a_failed_create_keeps_the_first_error(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(worktree: Worktree) -> None:
        raise WorkspaceError("the first error")

    real_git = workspace._git

    def failing_git(args: list[str], **options: Any) -> str:
        if args[:2] == ["worktree", "remove"]:
            raise GitError(args, 1, "the cleanup failed too")
        return real_git(args, **options)

    monkeypatch.setattr(workspace_module, "_make_pack_dirs", refuse)
    monkeypatch.setattr(workspace, "_git", failing_git)

    with pytest.raises(WorkspaceError, match="the first error"):
        workspace.create(LANE_EFFISQL, "exp-1")


# The tag moves only from the commit the caller saw.


def test_a_tag_moved_between_the_check_and_the_update_is_left_alone(
    workspace: Workspace, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = git(repo, "rev-parse", "best/effisql").strip()
    ours = add_commit(repo, "our candidate")
    theirs = add_commit(repo, "someone else's")

    def move_meanwhile(commit: str, ancestor: str) -> bool:
        git(repo, "tag", "--force", "best/effisql", theirs)  # Outside the lock, by someone else.
        return True

    monkeypatch.setattr(workspace, "_descends", move_meanwhile)

    with pytest.raises(WorkspaceError, match="best/effisql"):
        workspace.set_best(LANE_EFFISQL, ours, expected=old)

    assert workspace.best_commit(LANE_EFFISQL) == theirs


# Listing worktrees.


def test_a_worktree_on_an_unexpected_branch_is_listed_and_reported(
    workspace: Workspace, repo: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "worktrees" / "effisql" / "exp-7"
    git(repo, "worktree", "add", "--quiet", "-b", "something-else", str(path), "best/effisql")

    with caplog.at_level(logging.WARNING):
        listed = workspace.worktrees(LANE_EFFISQL)

    assert listed == [Worktree(path, LANE_EFFISQL, "exp-7")]
    assert "refs/heads/something-else" in caplog.text
