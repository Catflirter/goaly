"""Tests for goaly.rsi.guards.checksums: the frozen layers stay as they were."""

import shutil
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import GUARD_CHECKSUMS
from goaly.rsi.guards.checksums import FROZEN_DIRS, check_checksums, frozen_checksums
from tests.rsi.gitrepo import make_repo, write


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A copy of a repository's files, as a snapshot holds them."""
    repo = make_repo(tmp_path / "repo")
    shutil.rmtree(repo / ".git")
    write(
        repo,
        {
            "src/goaly/harness/core/tools.py": "TOOLS = []\n",
            "src/goaly/contracts/gen/task_pb2.py": "TASK = 1\n",
        },
    )
    return repo


def test_the_frozen_layers_are_bench_core_and_contracts() -> None:
    assert FROZEN_DIRS == ("src/goaly/bench", "src/goaly/harness/core", "src/goaly/contracts")


def test_an_unchanged_tree_passes(tree: Path) -> None:
    result = check_checksums(tree, frozen_checksums(tree))

    assert result.guard == GUARD_CHECKSUMS
    assert result.passed
    assert result.detail == "src/goaly/bench, src/goaly/harness/core, src/goaly/contracts unchanged"


@pytest.mark.parametrize(
    ("change", "changed"),
    [
        ({"src/goaly/harness/core/tools.py": "TOOLS = ['evil']\n"}, "src/goaly/harness/core"),
        ({"src/goaly/contracts/extra.py": ""}, "src/goaly/contracts"),
        ({"src/goaly/contracts/gen/task_pb2.py": "TASK = 2\n"}, "src/goaly/contracts"),
        ({"src/goaly/bench/runner.py": "def run() -> None:\n    pass\n\n"}, "src/goaly/bench"),
    ],
)
def test_any_change_to_a_frozen_layer_fails(
    tree: Path, change: dict[str, str], changed: str
) -> None:
    expected = frozen_checksums(tree)
    write(tree, change)

    result = check_checksums(tree, expected)

    assert not result.passed
    assert result.detail == f"{changed} changed"


def test_a_renamed_file_changes_the_checksum(tree: Path) -> None:
    expected = frozen_checksums(tree)
    (tree / "src/goaly/harness/core/tools.py").rename(tree / "src/goaly/harness/core/tool.py")

    assert not check_checksums(tree, expected).passed


def test_a_removed_layer_fails(tree: Path) -> None:
    expected = frozen_checksums(tree)
    shutil.rmtree(tree / "src/goaly/bench")

    assert check_checksums(tree, expected).detail == "src/goaly/bench changed"


def test_a_symlinks_target_counts(tree: Path) -> None:
    (tree / "src/goaly/contracts/link.py").symlink_to("__init__.py")
    expected = frozen_checksums(tree)
    (tree / "src/goaly/contracts/link.py").unlink()
    (tree / "src/goaly/contracts/link.py").symlink_to("other.py")

    assert not check_checksums(tree, expected).passed


def test_bytecode_caches_do_not_count(tree: Path) -> None:
    expected = frozen_checksums(tree)
    write(tree, {"src/goaly/harness/core/__pycache__/tools.cpython-312.pyc": "bytecode"})

    assert check_checksums(tree, expected).passed


def test_a_missing_expected_checksum_fails(tree: Path) -> None:
    expected = frozen_checksums(tree)
    del expected["src/goaly/contracts"]

    assert check_checksums(tree, expected).detail == "src/goaly/contracts changed"
