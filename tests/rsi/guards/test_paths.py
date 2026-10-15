"""Tests for goaly.rsi.guards.paths: a candidate changes only its lane's pack."""

import subprocess
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import GUARD_ALLOWED_PATHS, LANE_EFFISQL
from goaly.rsi.git import GitError
from goaly.rsi.guards.paths import PathsConfig, check_allowed_paths
from tests.rsi.gitrepo import git, make_repo, write

PACK = "src/goaly/harness/packs/effisql"
PACK_TESTS = "tests/harness/packs/effisql"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def candidate(repo: Path, files: dict[str, str], *, delete: tuple[str, ...] = ()) -> str:
    """Commit ``files`` and the deletions on top of HEAD, and return the commit."""
    write(repo, files)
    for path in delete:
        (repo / path).unlink()
    git(repo, "add", "--all", "--force")  # Even what .gitignore names, as a hand commit could.
    git(repo, "commit", "--quiet", "--message=candidate")
    return git(repo, "rev-parse", "HEAD").strip()


def check(repo: Path, commit: str) -> tuple[bool, str]:
    result = check_allowed_paths(repo, LANE_EFFISQL, parent="best/effisql", candidate=commit)
    assert result.guard == GUARD_ALLOWED_PATHS
    return result.passed, result.detail


def test_changes_inside_the_pack_pass(repo: Path) -> None:
    commit = candidate(
        repo,
        {
            f"{PACK}/prompt.md": "Read the plan first.\n",
            f"{PACK}/skills/plans.md": "# Plans\n",
            f"{PACK}/config.yaml": "steps: 30\n",
            f"{PACK_TESTS}/test_policy.py": "def test_ok() -> None:\n    pass\n",
        },
        delete=(f"{PACK}/policy.py",),
    )

    assert check(repo, commit) == (True, "5 changed paths, all in the pack")


@pytest.mark.parametrize(
    "path",
    [
        "src/goaly/bench/runner.py",
        "src/goaly/harness/core/llm.py",
        "README.md",
        "src/goaly/harness/packs/swefficiency/x.py",
        "src/goaly/harness/packs/effisql2/x.py",
        "src/goaly/harness/packs/README.md",
        "tests/harness/packs/effisql-extra/x.py",
    ],
)
def test_a_change_outside_the_pack_fails(repo: Path, path: str) -> None:
    commit = candidate(repo, {path: "x = 1\n", f"{PACK}/prompt.md": "Be faster.\n"})

    assert check(repo, commit) == (False, f"{path}: outside the lane's pack directories")


def test_a_deletion_outside_the_pack_fails(repo: Path) -> None:
    commit = candidate(repo, {}, delete=("src/goaly/bench/runner.py",))

    assert check(repo, commit) == (
        False,
        "src/goaly/bench/runner.py: outside the lane's pack directories",
    )


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("speedup.cpython-312-x86_64-linux-gnu.so", "a pack holds no .so files"),
        ("cache.pyc", "a pack holds no .pyc files"),
        ("Makefile", "a pack holds no files without a suffix"),
        (".gitattributes", "git's own file"),
    ],
)
def test_a_file_of_a_kind_a_pack_never_holds_fails(repo: Path, name: str, problem: str) -> None:
    commit = candidate(repo, {f"{PACK}/{name}": "x\n"})

    assert check(repo, commit) == (False, f"{PACK}/{name}: {problem}")


def test_a_symlink_in_the_pack_fails(repo: Path) -> None:
    (repo / PACK / "link.md").symlink_to("prompt.md")
    commit = candidate(repo, {})

    assert check(repo, commit) == (False, f"{PACK}/link.md: mode 120000, not a regular file")


def test_a_gitlink_in_the_pack_fails(repo: Path) -> None:
    head = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},{PACK}/sub")
    git(repo, "commit", "--quiet", "--message=a gitlink")
    commit = git(repo, "rev-parse", "HEAD").strip()

    assert check(repo, commit) == (False, f"{PACK}/sub: a gitlink")


def test_an_executable_file_in_the_pack_passes(repo: Path) -> None:
    write(repo, {f"{PACK}/run.py": "print('x')\n"})
    (repo / PACK / "run.py").chmod(0o755)
    commit = candidate(repo, {})

    assert check(repo, commit) == (True, "1 changed path, all in the pack")


def test_every_problem_is_listed(repo: Path) -> None:
    commit = candidate(repo, {"README.md": "changed\n", f"{PACK}/x.so": "x\n"})

    passed, detail = check(repo, commit)

    assert not passed
    assert detail.splitlines() == [
        "README.md: outside the lane's pack directories",
        f"{PACK}/x.so: a pack holds no .so files",
    ]


def test_a_git_directory_anywhere_in_a_path_fails(repo: Path) -> None:
    # git add refuses such a path, but a commit made with plumbing can hold one.
    blob = run_stdin(repo, ["hash-object", "-w", "--stdin"], "x = 1\n")
    inner = run_stdin(repo, ["mktree"], f"100644 blob {blob}\thooks.py\n")
    head_tree = git(repo, "rev-parse", "HEAD^{tree}").strip()
    tree = with_entry(repo, head_tree, PACK.split("/"), f"040000 tree {inner}\t.git")
    commit = run_stdin(repo, ["commit-tree", tree, "-p", "HEAD", "-m", "plumbing"], "")

    assert check(repo, commit) == (False, f"{PACK}/.git/hooks.py: git's own file")


def run_stdin(repo: Path, args: list[str], text: str) -> str:
    return subprocess.run(  # noqa: S603  (git with arguments built by the test)
        ["git", *args],  # noqa: S607
        cwd=repo,
        input=text,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def with_entry(repo: Path, tree: str, parts: list[str], entry: str) -> str:
    """``tree`` with ``entry``, a line of ls-tree's output, added in its subtree at ``parts``."""
    lines = git(repo, "ls-tree", tree).splitlines()
    if not parts:
        return run_stdin(repo, ["mktree"], "\n".join([*lines, entry]) + "\n")
    changed = []
    for line in lines:
        meta, name = line.split("\t")
        if name == parts[0]:
            line = f"040000 tree {with_entry(repo, meta.split(' ')[2], parts[1:], entry)}\t{name}"
        changed.append(line)
    return run_stdin(repo, ["mktree"], "\n".join(changed) + "\n")


def test_a_gitlink_shows_whatever_the_repository_config_says(repo: Path) -> None:
    git(repo, "config", "diff.ignoreSubmodules", "all")
    head = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},{PACK}/sub")
    git(repo, "commit", "--quiet", "--message=a gitlink")
    commit = git(repo, "rev-parse", "HEAD").strip()

    assert check(repo, commit) == (False, f"{PACK}/sub: a gitlink")


def test_a_revision_that_looks_like_an_option_is_never_read_as_one(
    repo: Path, tmp_path: Path
) -> None:
    planted = tmp_path / "planted"

    with pytest.raises(GitError):
        check_allowed_paths(repo, LANE_EFFISQL, parent=f"--output={planted}", candidate="HEAD")

    assert not planted.exists()


def test_a_path_that_is_not_plain_text_is_shown_quoted(repo: Path) -> None:
    commit = candidate(repo, {"src/goaly/bench/a\nPASS b.py": "x\n"})

    passed, detail = check(repo, commit)

    assert not passed
    assert detail.splitlines() == [
        '"src/goaly/bench/a\\nPASS b.py": outside the lane\'s pack directories'
    ]


def test_a_file_larger_than_the_limit_fails(repo: Path) -> None:
    commit = candidate(repo, {f"{PACK}/table.json": "x" * 2048})
    config = PathsConfig(max_file_bytes=1024)

    result = check_allowed_paths(
        repo, LANE_EFFISQL, parent="best/effisql", candidate=commit, config=config
    )

    assert (result.passed, result.detail) == (
        False,
        f"{PACK}/table.json: 2048 bytes, more than 1024",
    )


def test_files_larger_together_than_the_limit_fail(repo: Path) -> None:
    commit = candidate(repo, {f"{PACK}/a.md": "x" * 600, f"{PACK}/b.md": "y" * 600})
    config = PathsConfig(max_total_bytes=1000)

    result = check_allowed_paths(
        repo, LANE_EFFISQL, parent="best/effisql", candidate=commit, config=config
    )

    assert result.detail == "the changed files hold 1200 bytes, more than 1000"


def test_a_candidate_that_only_deletes_needs_no_size_check(repo: Path) -> None:
    commit = candidate(repo, {}, delete=(f"{PACK}/policy.py",))

    assert check(repo, commit) == (True, "1 changed path, all in the pack")
