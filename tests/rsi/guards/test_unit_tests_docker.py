"""Integration tests for goaly.rsi.guards.unit_tests in real containers.

They need Docker and run only with ``pytest --integration``.
"""

import secrets
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL
from goaly.rsi.guards.unit_tests import check_unit_tests
from goaly.rsi.sandbox import ensure_image
from goaly.rsi.workspace import Workspace
from tests.rsi.gitrepo import git, make_repo, write

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
PACK_TESTS = "tests/harness/packs/effisql"


@pytest.fixture(scope="module")
def image() -> str:
    return ensure_image(REPO_ROOT)


def guard(tmp_path: Path, image: str, test_source: str) -> tuple[bool, str]:
    repo = make_repo(tmp_path / "repo")
    write(repo, {f"{PACK_TESTS}/test_pack.py": test_source})
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message=the pack's tests")
    workspace = Workspace(repo, root=tmp_path / "worktrees")
    with workspace.snapshot("HEAD") as snapshot:
        result = check_unit_tests(
            snapshot,
            LANE_EFFISQL,
            experiment_id=f"it-{secrets.token_hex(4)}",
            image=image,
            timeout_s=120,
        )
    return result.passed, result.detail


def test_passing_pack_tests_pass_the_guard(tmp_path: Path, image: str) -> None:
    source = (
        "from goaly.harness.packs.effisql import policy\n\n\n"
        "def test_steps() -> None:\n    assert policy.STEPS == 3\n"
    )

    passed, detail = guard(tmp_path, image, source)

    assert passed, detail
    assert "1 passed" in detail


def test_failing_pack_tests_fail_the_guard(tmp_path: Path, image: str) -> None:
    passed, detail = guard(tmp_path, image, "def test_fails() -> None:\n    assert 1 == 2\n")

    assert not passed
    assert "1 failed" in detail
