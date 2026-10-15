"""Tests for scripts/freeze.py, on a throwaway repository with a bare origin."""

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
    git(repo, "tag", "--annotate", "final", "--message", "x")

    assert freeze_plan(repo).steps() == [
        f"# final already marks {freeze_plan(repo).commit[:12]}: nothing to do"
    ]

    commit_pack(repo, "v2")
    git(repo, "push", "--quiet", "origin", "main")
    with pytest.raises(FreezeError, match="already marks"):
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
