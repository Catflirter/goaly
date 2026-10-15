"""Tests for goaly.rsi.commit: turning the implementer's work into one commit, on the host."""

from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL
from goaly.rsi.commit import OutsidePackError, commit_candidate
from goaly.rsi.workspace import Workspace, WorkspaceError, Worktree
from tests.rsi.gitrepo import git, make_repo, write

PACK = "src/goaly/harness/packs/effisql"
PACK_TESTS = "tests/harness/packs/effisql"


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(make_repo(tmp_path / "repo"), root=tmp_path / "worktrees")


@pytest.fixture
def worktree(workspace: Workspace) -> Worktree:
    return workspace.create(LANE_EFFISQL, "exp-1")


def changed_files(worktree: Worktree, commit: str) -> list[str]:
    output = git(worktree.path, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    return output.splitlines()


def trailer(worktree: Worktree, key: str) -> str:
    return git(worktree.path, "log", "-1", f"--format=%(trailers:key={key},valueonly)").strip()


def test_the_packs_changes_become_one_commit_with_trailers(
    workspace: Workspace, worktree: Worktree
) -> None:
    base = workspace.head(worktree)
    write(worktree.path, {f"{PACK}/prompt.md": "Read the plan first.\n"})
    write(worktree.path, {f"{PACK}/skills/plans.md": "# Plans\n"})
    write(worktree.path, {f"{PACK_TESTS}/test_policy.py": "def test_ok() -> None:\n    pass\n"})
    (worktree.path / PACK / "policy.py").unlink()

    commit = commit_candidate(
        workspace, worktree, hypothesis_id="hyp-3", summary="Read the plan before rewriting"
    )

    assert commit is not None
    assert commit == workspace.head(worktree)
    assert git(worktree.path, "rev-parse", f"{commit}^").strip() == base
    assert sorted(changed_files(worktree, commit)) == [
        f"A\t{PACK}/skills/plans.md",
        f"A\t{PACK_TESTS}/test_policy.py",
        f"D\t{PACK}/policy.py",
        f"M\t{PACK}/prompt.md",
    ]
    assert git(worktree.path, "log", "-1", "--format=%s").strip() == (
        "rsi(effisql): Read the plan before rewriting"
    )
    assert trailer(worktree, "Experiment-Id") == "exp-1"
    assert trailer(worktree, "Hypothesis-Id") == "hyp-3"
    assert git(worktree.path, "status", "--porcelain") == ""


def test_no_change_makes_no_commit(workspace: Workspace, worktree: Worktree) -> None:
    base = workspace.head(worktree)

    assert commit_candidate(workspace, worktree, hypothesis_id="hyp-3", summary="x") is None
    assert workspace.head(worktree) == base


def test_a_change_outside_the_pack_commits_nothing(
    workspace: Workspace, worktree: Worktree
) -> None:
    base = workspace.head(worktree)
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n", "src/goaly/bench/runner.py": ""})
    write(worktree.path, {"notes.txt": "stray\n"})

    with pytest.raises(OutsidePackError) as caught:
        commit_candidate(workspace, worktree, hypothesis_id="hyp-3", summary="x")

    assert caught.value.paths == ["notes.txt", "src/goaly/bench/runner.py"]
    assert "src/goaly/bench/runner.py" in str(caught.value)
    assert workspace.head(worktree) == base
    assert git(worktree.path, "diff", "--cached", "--name-only") == ""


def test_ignored_files_in_the_pack_are_not_committed(
    workspace: Workspace, worktree: Worktree
) -> None:
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n"})
    write(worktree.path, {f"{PACK}/__pycache__/policy.cpython-312.pyc": "bytecode"})

    commit = commit_candidate(workspace, worktree, hypothesis_id="hyp-3", summary="x")

    assert commit is not None
    assert changed_files(worktree, commit) == [f"M\t{PACK}/prompt.md"]


def test_a_missing_pack_directory_is_skipped(workspace: Workspace, worktree: Worktree) -> None:
    (worktree.path / PACK_TESTS).rmdir()
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n"})

    commit = commit_candidate(workspace, worktree, hypothesis_id="hyp-3", summary="x")

    assert commit is not None
    assert changed_files(worktree, commit) == [f"M\t{PACK}/prompt.md"]


def test_the_summary_is_cleaned_into_a_subject_and_a_body(
    workspace: Workspace, worktree: Worktree
) -> None:
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n"})
    summary = (
        "  Add a plan-reading step\x1b[31m before every rewrite, so the agent sees which join "
        "dominates\nThen check the timing.\r\n\n# Not a comment.\nExperiment-Id: forged\n"
    )

    commit_candidate(workspace, worktree, hypothesis_id="hyp-3", summary=summary)

    message = git(worktree.path, "log", "-1", "--format=%B")
    subject, _, rest = message.partition("\n\n")
    # 72 characters end inside "the", so the subject stops at the word before.
    assert subject == "rsi(effisql): Add a plan-reading step[31m before every rewrite, so..."
    assert rest.startswith("Add a plan-reading step[31m before every rewrite, so the agent sees")
    assert "# Not a comment." in rest
    assert "\r" not in message
    assert "\x1b" not in message
    assert trailer(worktree, "Experiment-Id") == "exp-1"


def test_a_blank_summary_gets_a_subject_naming_the_hypothesis(
    workspace: Workspace, worktree: Worktree
) -> None:
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n"})

    commit_candidate(workspace, worktree, hypothesis_id="hyp-3", summary=" \n ")

    assert git(worktree.path, "log", "-1", "--format=%B").startswith(
        "rsi(effisql): implement hypothesis hyp-3\n\nExperiment-Id: exp-1\nHypothesis-Id: hyp-3\n"
    )


def test_an_invalid_hypothesis_id_is_refused(workspace: Workspace, worktree: Worktree) -> None:
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n"})

    with pytest.raises(ValueError, match="hypothesis_id"):
        commit_candidate(workspace, worktree, hypothesis_id="h\nInjected: x", summary="x")


def test_commit_hooks_do_not_run(workspace: Workspace, worktree: Worktree, tmp_path: Path) -> None:
    marker = tmp_path / "hook-ran"
    for name in ("pre-commit", "commit-msg", "post-commit"):
        hook = workspace.repo / ".git" / "hooks" / name
        hook.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n", encoding="utf-8")
        hook.chmod(0o755)
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n"})

    assert commit_candidate(workspace, worktree, hypothesis_id="hyp-3", summary="x") is not None
    assert not marker.exists()


@pytest.mark.parametrize(
    ("first_line", "subject"),
    [
        # 55 characters end right before a space, so the last word stays whole.
        ("word " * 10 + "words more text", "rsi(effisql): " + "word " * 10 + "words..."),
        # One long word, such as a path, is cut where it must be.
        ("x" * 100, "rsi(effisql): " + "x" * 55 + "..."),
        ("Short and whole", "rsi(effisql): Short and whole"),
    ],
)
def test_long_subjects_are_shortened_to_72_characters(
    workspace: Workspace, worktree: Worktree, first_line: str, subject: str
) -> None:
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n"})

    commit_candidate(workspace, worktree, hypothesis_id="hyp-3", summary=first_line)

    assert git(worktree.path, "log", "-1", "--format=%s").strip() == subject
    assert len(subject) <= 72


# The pack's boundary.


@pytest.mark.parametrize(
    "path",
    [
        "src/goaly/harness/packs/swefficiency/x.py",
        "src/goaly/harness/packs/effisql2/x.py",
        "src/goaly/harness/packs/README.md",
        "tests/harness/packs/effisql-extra/x.py",
    ],
)
def test_a_path_just_outside_the_pack_is_outside_it(
    workspace: Workspace, worktree: Worktree, path: str
) -> None:
    write(worktree.path, {path: "x = 1\n", f"{PACK}/prompt.md": "Be faster.\n"})

    with pytest.raises(OutsidePackError) as caught:
        commit_candidate(workspace, worktree, hypothesis_id="hyp-3", summary="x")

    assert caught.value.paths == [path]


# The worktree must be the one it claims to be.


def test_a_directory_that_is_not_a_registered_worktree_is_refused(
    workspace: Workspace, tmp_path: Path
) -> None:
    stray = tmp_path / "worktrees" / "effisql" / "exp-9"
    stray.mkdir(parents=True)

    with pytest.raises(WorkspaceError, match="not a registered worktree"):
        commit_candidate(
            workspace, Worktree(stray, LANE_EFFISQL, "exp-9"), hypothesis_id="h", summary="x"
        )


def test_a_worktree_claiming_another_experiment_is_refused(
    workspace: Workspace, worktree: Worktree
) -> None:
    impostor = Worktree(worktree.path, LANE_EFFISQL, "exp-2")

    with pytest.raises(WorkspaceError, match="not a registered worktree"):
        commit_candidate(workspace, impostor, hypothesis_id="h", summary="x")


def test_a_worktree_on_another_branch_is_refused(workspace: Workspace, worktree: Worktree) -> None:
    git(worktree.path, "checkout", "--quiet", "-b", "elsewhere")
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n"})

    with pytest.raises(WorkspaceError, match="refs/heads/elsewhere"):
        commit_candidate(workspace, worktree, hypothesis_id="h", summary="x")


def test_an_invalid_experiment_id_is_refused(workspace: Workspace, worktree: Worktree) -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        commit_candidate(
            workspace,
            Worktree(worktree.path, LANE_EFFISQL, "bad id"),
            hypothesis_id="h",
            summary="x",
        )


# The message.


def test_tabs_become_spaces_and_a_line_of_dashes_cannot_end_the_message(
    workspace: Workspace, worktree: Worktree
) -> None:
    write(worktree.path, {f"{PACK}/prompt.md": "Be faster.\n"})

    commit_candidate(
        workspace,
        worktree,
        hypothesis_id="hyp-3",
        summary="Read\tthe plan\n---\n--- diff --git a/x b/x\nend",
    )

    message = git(worktree.path, "log", "-1", "--format=%B")
    assert message.startswith("rsi(effisql): Read the plan\n\n")
    assert "\n ---\n --- diff --git a/x b/x\nend\n" in message
    assert "\n---" not in message


def test_a_worktree_with_a_detached_head_is_refused(
    workspace: Workspace, worktree: Worktree
) -> None:
    git(worktree.path, "checkout", "--quiet", "--detach")

    with pytest.raises(WorkspaceError, match="no branch"):
        commit_candidate(workspace, worktree, hypothesis_id="h", summary="x")
