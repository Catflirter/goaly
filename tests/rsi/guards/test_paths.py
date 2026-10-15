"""Tests for goaly.rsi.guards.paths: a candidate changes only its lane's pack."""

from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import GUARD_ALLOWED_PATHS, LANE_EFFISQL
from goaly.rsi.guards.paths import check_allowed_paths
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

    assert check(repo, commit) == (False, f"{PACK}/sub: mode 160000, not a regular file")


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
