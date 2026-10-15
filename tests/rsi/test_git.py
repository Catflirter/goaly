"""Tests for goaly.rsi.git: how goaly.rsi runs git."""

import time
from pathlib import Path

import pytest

from goaly.rsi.git import GitError, run_git
from tests.rsi.gitrepo import git, make_repo


def test_returns_the_commands_output(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")

    assert (
        run_git(["rev-parse", "HEAD"], cwd=repo).strip() == git(repo, "rev-parse", "HEAD").strip()
    )


def test_a_failure_raises_with_the_exit_code_and_stderr(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")

    with pytest.raises(GitError, match="no-such-ref") as caught:
        run_git(["rev-parse", "--verify", "no-such-ref"], cwd=repo)

    assert caught.value.returncode == 128


def test_input_is_passed_on_stdin(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")

    blob = run_git(["hash-object", "--stdin"], cwd=repo, input_text="hello\n").strip()

    assert blob == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_a_command_that_runs_too_long_is_stopped_with_its_children(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    marker = tmp_path / "child-survived"
    git(repo, "config", "alias.slow", f"!sleep 1; touch {marker}")

    with pytest.raises(GitError, match="timed out"):
        run_git(["slow"], cwd=repo, timeout_s=0.3)

    time.sleep(1.5)
    assert not marker.exists()


def test_variables_that_redirect_git_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path / "repo")
    other = make_repo(tmp_path / "other")
    git(other, "commit", "--quiet", "--allow-empty", "--message=other")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))

    head = run_git(["rev-parse", "HEAD"], cwd=repo).strip()

    monkeypatch.delenv("GIT_DIR")
    monkeypatch.delenv("GIT_WORK_TREE")
    assert head == git(repo, "rev-parse", "HEAD").strip()


def test_hooks_never_run(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    marker = tmp_path / "hook-ran"
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)

    run_git(["commit", "--quiet", "--allow-empty", "--message=x"], cwd=repo)

    assert not marker.exists()
