"""Tests for scripts/freeze.py, on a throwaway repository with a bare origin."""

import subprocess
from pathlib import Path

import pytest

from scripts import freeze
from scripts.freeze import FreezeError, freeze_plan
from tests.scripts.gitrepo import commit_pack, git, repo_with_origin


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The annotated tag needs a committer, which CI's git has no config for."""
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Freeze Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "freeze@example.com")


def remote_tags(origin: Path) -> str:
    return git(origin, "tag", "--list")


def test_freezes_main_when_it_holds_the_best_pack(tmp_path: Path) -> None:
    repo, origin = repo_with_origin(tmp_path)
    git(repo, "checkout", "--quiet", "-b", "rsi/effisql/e1")
    best = commit_pack(repo, "v1")
    git(repo, "tag", "best/effisql", best)
    git(repo, "checkout", "--quiet", "main")
    merged = commit_pack(repo, "v1")  # The results PR, merged: the same pack, another commit.
    git(repo, "push", "--quiet", "origin", "main")

    plan = freeze_plan(repo)

    assert plan.commit == merged
    assert f"its Effi-SQL pack is best/effisql's, {best}" in plan.message
    assert plan.steps()[1] == "git push origin refs/tags/final:refs/tags/final"
    assert remote_tags(origin) == ""  # The plan changes nothing.

    plan.apply(repo)

    assert git(repo, "rev-parse", "final^{commit}") == merged
    assert git(repo, "cat-file", "-t", "final") == "tag"  # Annotated.
    assert remote_tags(origin) == "final"
    assert git(origin, "rev-parse", "main") == merged  # No branch moved.


def test_refuses_main_without_the_best_pack(tmp_path: Path) -> None:
    repo, origin = repo_with_origin(tmp_path)
    git(repo, "tag", "best/effisql", commit_pack(repo, "v1"))  # Committed, but not pushed to main.

    with pytest.raises(FreezeError, match="merge its results PR into main first"):
        freeze_plan(repo)
    assert remote_tags(origin) == ""


def test_without_a_best_tag_the_final_harness_is_v0(tmp_path: Path) -> None:
    repo, _ = repo_with_origin(tmp_path)

    plan = freeze_plan(repo)

    assert "no round accepted a change, so the final harness is v0" in plan.message


def test_takes_main_from_the_remote_not_the_local_branch(tmp_path: Path) -> None:
    repo, _ = repo_with_origin(tmp_path)
    pushed = git(repo, "rev-parse", "HEAD")
    commit_pack(repo, "local only")

    assert freeze_plan(repo).commit == pushed


def test_an_existing_tag_on_the_commit_is_done_and_on_another_refused(tmp_path: Path) -> None:
    repo, _ = repo_with_origin(tmp_path)
    plan = freeze_plan(repo)
    plan.apply(repo)

    assert freeze_plan(repo).done
    assert freeze_plan(repo).steps() == [
        f"# final marks {plan.commit[:12]} here and on origin: nothing to do"
    ]

    commit_pack(repo, "v2")
    git(repo, "push", "--quiet", "origin", "main")
    with pytest.raises(FreezeError, match=r"already marks .* here"):
        freeze_plan(repo)


def test_pushes_again_a_tag_whose_push_failed(tmp_path: Path) -> None:
    repo, origin = repo_with_origin(tmp_path)
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)

    with pytest.raises(subprocess.CalledProcessError):
        freeze_plan(repo).apply(repo)
    assert remote_tags(origin) == ""
    hook.unlink()
    plan = freeze_plan(repo)

    assert (plan.tagged, plan.pushed) == (True, False)
    assert plan.steps()[-1] == "git push origin refs/tags/final:refs/tags/final"
    plan.apply(repo)
    assert remote_tags(origin) == "final"


def test_refuses_when_the_remote_final_marks_another_commit(tmp_path: Path) -> None:
    repo, origin = repo_with_origin(tmp_path)
    old = git(repo, "rev-parse", "HEAD")
    git(origin, "tag", "final", old)
    commit_pack(repo, "v2")
    git(repo, "push", "--quiet", "origin", "main")

    with pytest.raises(FreezeError, match=r"already marks .* origin"):
        freeze_plan(repo)


def test_refuses_a_best_tag_on_the_remote_that_differs_here(tmp_path: Path) -> None:
    repo, origin = repo_with_origin(tmp_path)
    git(origin, "tag", "best/effisql", git(repo, "rev-parse", "HEAD"))

    with pytest.raises(FreezeError, match="fetch the lane's tag first"):
        freeze_plan(repo)

    git(repo, "fetch", "--quiet", "origin", "tag", "best/effisql")
    assert "its Effi-SQL pack is best/effisql's" in freeze_plan(repo).message


def test_compares_the_pack_tests_too(tmp_path: Path) -> None:
    repo, _ = repo_with_origin(tmp_path)
    tests = repo / "tests" / "harness" / "packs" / "effisql"
    tests.mkdir(parents=True)
    (tests / "test_policy.py").write_text("# best's\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "--quiet", "--message", "best's tests")
    git(repo, "tag", "best/effisql")
    git(repo, "reset", "--quiet", "--hard", "HEAD~1")

    with pytest.raises(FreezeError, match="tests/harness/packs/effisql"):
        freeze_plan(repo)


def test_main_tags_only_with_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, origin = repo_with_origin(tmp_path)
    monkeypatch.setattr(freeze, "REPO", repo)

    assert freeze.main([]) == 0
    out = capsys.readouterr().out
    assert "git tag --annotate final" in out
    assert "# dry run: pass --yes" in out
    assert remote_tags(origin) == ""

    assert freeze.main(["--yes"]) == 0
    assert remote_tags(origin) == "final"


def test_main_reports_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, _ = repo_with_origin(tmp_path)
    git(repo, "tag", "best/effisql", commit_pack(repo, "v1"))
    monkeypatch.setattr(freeze, "REPO", repo)

    assert freeze.main(["--yes"]) == 1
    assert "this script never merges" in capsys.readouterr().err


def test_main_reports_a_failed_git_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, _ = repo_with_origin(tmp_path)
    monkeypatch.setattr(freeze, "REPO", repo)

    assert freeze.main(["--remote", "nowhere"]) == 1
    assert "git -C" in capsys.readouterr().err


def test_fails_unless_the_remote_has_the_tag_after_the_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = repo_with_origin(tmp_path)
    plan = freeze_plan(repo)
    monkeypatch.setattr(freeze, "_remote_tags", lambda repo, remote: {})

    with pytest.raises(FreezeError, match="does not have final"):
        plan.apply(repo)
