"""Tests for Workspace.snapshot and Workspace.pin: what candidates are evaluated from."""

from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL
from goaly.rsi import workspace as workspace_module
from goaly.rsi.workspace import UnsafePackError, Workspace, WorkspaceError, Worktree
from tests.rsi.gitrepo import git, make_repo, write

PACK = "src/goaly/harness/packs/effisql"
PACK_TESTS = "tests/harness/packs/effisql"


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(make_repo(tmp_path / "repo"), root=tmp_path / "worktrees")


@pytest.fixture
def worktree(workspace: Workspace) -> Worktree:
    return workspace.create(LANE_EFFISQL, "exp-1")


def commit_all(worktree: Worktree, message: str) -> str:
    git(worktree.path, "add", "--all")
    git(worktree.path, "commit", "--quiet", f"--message={message}")
    return git(worktree.path, "rev-parse", "HEAD").strip()


def files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# Snapshots.


def test_a_snapshot_holds_the_commits_files_read_only(
    workspace: Workspace, worktree: Worktree
) -> None:
    write(worktree.path, {f"{PACK}/prompt.md": "Read the plan first.\n"})
    commit = commit_all(worktree, "candidate")

    with workspace.snapshot(commit) as root:
        assert files(root)[f"{PACK}/prompt.md"] == "Read the plan first.\n"
        assert files(root)["src/goaly/bench/runner.py"] == "def run() -> None:\n    pass\n"
        with pytest.raises(PermissionError):
            (root / PACK / "prompt.md").write_text("changed\n", encoding="utf-8")
        with pytest.raises(PermissionError):
            (root / PACK / "new.md").write_text("new\n", encoding="utf-8")

    assert not root.exists()


def test_changing_the_live_worktree_does_not_change_a_snapshot(
    workspace: Workspace, worktree: Worktree
) -> None:
    write(worktree.path, {f"{PACK}/prompt.md": "Read the plan first.\n"})
    commit = commit_all(worktree, "candidate")

    with workspace.snapshot(commit) as root:
        write(worktree.path, {f"{PACK}/prompt.md": "swapped\n", f"{PACK}/evil.py": "x = 1\n"})
        (worktree.path / PACK / "__pycache__").mkdir()
        (worktree.path / PACK / "__pycache__" / "policy.cpython-312.pyc").write_bytes(b"\0")

        assert (root / PACK / "prompt.md").read_text(encoding="utf-8") == "Read the plan first.\n"
        assert not (root / PACK / "evil.py").exists()
        assert not (root / PACK / "__pycache__").exists()


def test_a_snapshot_is_removed_when_its_block_raises(workspace: Workspace) -> None:
    with pytest.raises(RuntimeError, match="boom"), workspace.snapshot("HEAD") as root:
        raise RuntimeError("boom")

    assert not root.exists()


def test_a_snapshot_needs_a_commit(workspace: Workspace) -> None:
    with (
        pytest.raises(WorkspaceError, match="no-such-commit"),
        workspace.snapshot("no-such-commit"),
    ):
        pass


def test_a_snapshot_refuses_a_symlink_that_leaves_it(
    workspace: Workspace, worktree: Worktree, tmp_path: Path
) -> None:
    (worktree.path / PACK / "secret.md").symlink_to(tmp_path / "outside.txt")
    commit = commit_all(worktree, "a symlink out of the tree")

    with pytest.raises(WorkspaceError, match=r"secret\.md"), workspace.snapshot(commit):
        pass


# Pinning a worktree as it stands.


def test_pin_commits_the_pack_as_it_stands_without_touching_the_worktree(
    workspace: Workspace, worktree: Worktree
) -> None:
    head = workspace.head(worktree)
    write(worktree.path, {f"{PACK}/prompt.md": "Draft.\n", f"{PACK_TESTS}/test_x.py": "x = 1\n"})
    write(worktree.path, {f"{PACK}/__pycache__/policy.cpython-312.pyc": "bytecode"})
    write(worktree.path, {"src/goaly/bench/runner.py": "changed outside the pack\n"})
    status = git(worktree.path, "status", "--porcelain", "--untracked-files=all")

    pinned = workspace.pin(worktree)

    assert git(worktree.path, "rev-parse", f"{pinned}^").strip() == head
    assert workspace.head(worktree) == head
    assert git(worktree.path, "status", "--porcelain", "--untracked-files=all") == status
    with workspace.snapshot(pinned) as root:
        assert (root / PACK / "prompt.md").read_text(encoding="utf-8") == "Draft.\n"
        assert (root / PACK_TESTS / "test_x.py").is_file()
        assert not (root / PACK / "__pycache__").exists()
        # Only the pack is pinned: a change elsewhere stays out.
        runner = (root / "src/goaly/bench/runner.py").read_text(encoding="utf-8")
        assert runner == "def run() -> None:\n    pass\n"


def test_pinning_an_unchanged_worktree_gives_its_heads_tree(
    workspace: Workspace, worktree: Worktree
) -> None:
    pinned = workspace.pin(worktree)

    assert git(worktree.path, "rev-parse", f"{pinned}^{{tree}}") == git(
        worktree.path, "rev-parse", "HEAD^{tree}"
    )


def test_pin_refuses_an_unsafe_pack_before_git_runs(
    workspace: Workspace, worktree: Worktree
) -> None:
    nested = worktree.path / PACK / "sub"
    nested.mkdir()
    git(nested, "init", "--quiet")

    with pytest.raises(UnsafePackError, match="a nested repository"):
        workspace.pin(worktree)


def test_pin_checks_what_it_staged(
    workspace: Workspace, worktree: Worktree, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Were the first check to miss a symlink, the staged check still refuses it.
    monkeypatch.setattr(workspace_module, "check_pack_files", lambda worktree: None)
    (worktree.path / PACK / "link.md").symlink_to("prompt.md")

    with pytest.raises(UnsafePackError, match=r"link\.md: mode 120000 staged"):
        workspace.pin(worktree)


def test_a_symlink_that_stays_inside_the_snapshot_is_kept(
    workspace: Workspace, worktree: Worktree
) -> None:
    (worktree.path / "docs").mkdir()
    (worktree.path / "docs" / "readme.md").symlink_to("../README.md")
    commit = commit_all(worktree, "a link within the tree")

    with workspace.snapshot(commit) as root:
        assert (root / "docs" / "readme.md").is_symlink()
        assert (root / "docs" / "readme.md").read_text(encoding="utf-8") == "# goaly\n"
