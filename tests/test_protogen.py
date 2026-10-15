"""Tests for scripts/protogen.py, the code generator behind `make proto` and `make proto-check`."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOGEN = REPO_ROOT / "scripts" / "protogen.py"

LEAF_PROTO = """\
syntax = "proto3";

package goaly.v1;

message Leaf {
  string name = 1;
}
"""

TREE_PROTO = """\
syntax = "proto3";

package goaly.v1;

import "goaly/v1/leaf.proto";
import "google/protobuf/timestamp.proto";

message Tree {
  Leaf leaf = 1;
  google.protobuf.Timestamp planted_at = 2;
}
"""

GENERATED = {"leaf_pb2.py", "leaf_pb2.pyi", "tree_pb2.py", "tree_pb2.pyi"}

# Imports the generated fixture modules from a package root that holds goaly/contracts/gen/.
IMPORT_PROBE = """
import json
import pickle

import goaly
from goaly.contracts.gen import leaf_pb2, tree_pb2

tree = tree_pb2.Tree(leaf=leaf_pb2.Leaf(name="oak"))
print(json.dumps({
    "goaly": goaly.__file__,
    "module": tree_pb2.Tree.__module__,
    "file": tree_pb2.DESCRIPTOR.name,
    "dependencies": [file.name for file in tree_pb2.DESCRIPTOR.dependencies],
    "message": tree_pb2.Tree.DESCRIPTOR.full_name,
    "leaf_type": tree_pb2.Tree.DESCRIPTOR.fields_by_name["leaf"].message_type.full_name,
    "unpickled": pickle.loads(pickle.dumps(tree)) == tree,
}))
"""

STUB_PROBE = """\
from goaly.contracts.gen.tree_pb2 import Tree


def leaf_name(tree: Tree) -> str:
    return tree.leaf.name
"""


def run(
    *command: str | Path, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    # The commands are fixed paths and arguments built by these tests, not untrusted input.
    return subprocess.run(  # noqa: S603
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120, check=False
    )


def protogen(*args: str | Path) -> subprocess.CompletedProcess[str]:
    """Run the generator the way make does: with this interpreter, from the repository root."""
    return run(sys.executable, PROTOGEN, *args, cwd=REPO_ROOT)


def snapshot(directory: Path) -> dict[str, bytes]:
    """Return the name and content of every file directly in ``directory``."""
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


@pytest.fixture
def proto_dir(tmp_path: Path) -> Path:
    """A proto root holding goaly/v1/leaf.proto and goaly/v1/tree.proto, which imports it."""
    root = tmp_path / "proto"
    package = root / "goaly" / "v1"
    package.mkdir(parents=True)
    (package / "leaf.proto").write_text(LEAF_PROTO, encoding="utf-8")
    (package / "tree.proto").write_text(TREE_PROTO, encoding="utf-8")
    return root


@pytest.fixture
def package_root(proto_dir: Path, tmp_path: Path) -> Path:
    """A package root whose goaly/contracts/gen/ holds the code generated from ``proto_dir``."""
    root = tmp_path / "site"
    out_dir = root / "goaly" / "contracts" / "gen"
    result = protogen("--proto-dir", proto_dir, "--out-dir", out_dir)
    assert result.returncode == 0, result.stderr
    for package in (root / "goaly", root / "goaly" / "contracts", out_dir):
        (package / "__init__.py").touch()
    return root


@pytest.fixture
def stale(proto_dir: Path, tmp_path: Path) -> Path:
    """An output directory that no longer matches ``proto_dir``.

    tree.proto gained a field, leaf_pb2.pyi is missing, old_pb2.py is left over from a deleted
    proto, and two files the generator does not own sit beside them. leaf_pb2.py is current.
    """
    out_dir = tmp_path / "gen"
    assert protogen("--proto-dir", proto_dir, "--out-dir", out_dir).returncode == 0
    tree = proto_dir / "goaly" / "v1" / "tree.proto"
    grown = TREE_PROTO.replace("planted_at = 2;", "planted_at = 2;\n  string species = 3;")
    tree.write_text(grown, encoding="utf-8")
    (out_dir / "leaf_pb2.pyi").unlink()
    (out_dir / "old_pb2.py").write_text("# From a deleted proto.\n", encoding="utf-8")
    (out_dir / "__init__.py").write_text('"""Hand-written."""\n', encoding="utf-8")
    (out_dir / "NOTES.txt").write_text("Not generated.\n", encoding="utf-8")
    return out_dir


# Generating.


def test_writes_a_module_and_a_stub_for_every_proto(proto_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "gen"

    result = protogen("--proto-dir", proto_dir, "--out-dir", out_dir)

    assert result.returncode == 0, result.stderr
    assert set(snapshot(out_dir)) == GENERATED


def test_generated_modules_live_in_goaly_contracts_gen(package_root: Path, tmp_path: Path) -> None:
    result = run(
        sys.executable,
        "-c",
        IMPORT_PROBE,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(package_root)},
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        # The probe imported the generated copy, not the goaly package installed for the tests.
        "goaly": str(package_root / "goaly" / "__init__.py"),
        # The classes name the module they are imported from, so pickling works...
        "module": "goaly.contracts.gen.tree_pb2",
        "unpickled": True,
        # ...while the descriptors keep the names protoc gave them.
        "file": "goaly/v1/tree.proto",
        "dependencies": ["goaly/v1/leaf.proto", "google/protobuf/timestamp.proto"],
        "message": "goaly.v1.Tree",
        "leaf_type": "goaly.v1.Leaf",
    }


def test_generated_stubs_resolve_from_goaly_contracts_gen(
    package_root: Path, tmp_path: Path
) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(STUB_PROBE, encoding="utf-8")

    # A stub still importing goaly.v1 would leave Tree.leaf unresolved: an error under --strict.
    result = run(
        sys.executable,
        "-m",
        "mypy",
        "--strict",
        "--cache-dir",
        os.devnull,
        probe,
        cwd=tmp_path,
        env={**os.environ, "MYPYPATH": str(package_root)},
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_output_depends_only_on_the_protos(proto_dir: Path, tmp_path: Path) -> None:
    moved = shutil.copytree(proto_dir, tmp_path / "elsewhere" / "proto")
    outputs = [tmp_path / "first", tmp_path / "second", tmp_path / "moved"]

    for source, out_dir in zip([proto_dir, proto_dir, moved], outputs, strict=True):
        assert protogen("--proto-dir", source, "--out-dir", out_dir).returncode == 0

    first, second, from_moved = (snapshot(out_dir) for out_dir in outputs)
    assert first == second == from_moved


def test_write_brings_a_stale_directory_up_to_date(proto_dir: Path, stale: Path) -> None:
    result = protogen("--proto-dir", proto_dir, "--out-dir", stale)

    assert result.returncode == 0, result.stderr
    assert set(snapshot(stale)) == GENERATED | {"__init__.py", "NOTES.txt"}
    assert (stale / "__init__.py").read_text(encoding="utf-8") == '"""Hand-written."""\n'
    assert protogen("--check", "--proto-dir", proto_dir, "--out-dir", stale).returncode == 0


# Checking.


def test_check_passes_and_changes_nothing_when_up_to_date(proto_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "gen"
    assert protogen("--proto-dir", proto_dir, "--out-dir", out_dir).returncode == 0
    before = snapshot(out_dir)

    result = protogen("--check", "--proto-dir", proto_dir, "--out-dir", out_dir)

    assert result.returncode == 0, result.stderr
    assert "up to date" in result.stdout
    assert snapshot(out_dir) == before


def test_check_names_every_stale_file_and_changes_nothing(proto_dir: Path, stale: Path) -> None:
    before = snapshot(stale)

    result = protogen("--check", "--proto-dir", proto_dir, "--out-dir", stale)

    assert result.returncode == 1
    listed = (re.fullmatch(r"  (\w+): +(\S+)", line) for line in result.stderr.splitlines())
    assert {match[2]: match[1] for match in listed if match} == {
        "leaf_pb2.pyi": "missing",
        "old_pb2.py": "extra",
        "tree_pb2.py": "changed",
        "tree_pb2.pyi": "changed",
    }
    assert "make proto" in result.stderr
    assert snapshot(stale) == before


# Errors: nothing is written.


@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        ("goaly/v1/broken.proto", 'syntax = "proto3";\npackage goaly.v1;\nmessage {', "broken"),
        (
            "goaly/v1/unused.proto",
            'syntax = "proto3";\npackage goaly.v1;\nimport "goaly/v1/leaf.proto";\n',
            "unused",
        ),
        ("other/thing.proto", 'syntax = "proto3";\npackage other;\n', "other/thing.proto"),
    ],
    ids=["syntax-error", "warning", "outside-goaly-v1"],
)
def test_fails_on_a_proto_it_cannot_generate_cleanly(
    proto_dir: Path, tmp_path: Path, name: str, content: str, reason: str
) -> None:
    path = proto_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    out_dir = tmp_path / "gen"

    result = protogen("--proto-dir", proto_dir, "--out-dir", out_dir)

    assert result.returncode == 1
    assert reason in result.stderr.lower()
    assert not out_dir.exists()


def test_fails_when_there_is_no_proto(tmp_path: Path) -> None:
    out_dir = tmp_path / "gen"

    result = protogen("--proto-dir", tmp_path / "missing", "--out-dir", out_dir)

    assert result.returncode == 1
    assert "no .proto files" in result.stderr
    assert not out_dir.exists()


# The freshness check on copies of the repository, as CI's make proto-check runs it.


@pytest.fixture
def repository_copy(tmp_path: Path) -> tuple[Path, Path]:
    """Copies of proto/ and of the committed src/goaly/contracts/gen/."""
    proto_dir = shutil.copytree(REPO_ROOT / "proto", tmp_path / "proto")
    out_dir = shutil.copytree(
        REPO_ROOT / "src" / "goaly" / "contracts" / "gen",
        tmp_path / "gen",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return proto_dir, out_dir


def test_check_passes_on_the_committed_code(repository_copy: tuple[Path, Path]) -> None:
    proto_dir, out_dir = repository_copy

    result = protogen("--check", "--proto-dir", proto_dir, "--out-dir", out_dir)

    assert result.returncode == 0, result.stderr


def test_check_fails_when_a_proto_gains_a_field(repository_copy: tuple[Path, Path]) -> None:
    proto_dir, out_dir = repository_copy
    task = proto_dir / "goaly" / "v1" / "task.proto"
    original = task.read_text(encoding="utf-8")
    grown = original.replace("message TaskInput {\n", "message TaskInput {\n  string added = 99;\n")
    assert grown != original
    task.write_text(grown, encoding="utf-8")

    result = protogen("--check", "--proto-dir", proto_dir, "--out-dir", out_dir)

    assert result.returncode == 1
    assert "changed: task_pb2.py\n" in result.stderr
    assert "changed: task_pb2.pyi\n" in result.stderr
    assert "make proto" in result.stderr


def test_check_fails_when_a_generated_file_is_edited(repository_copy: tuple[Path, Path]) -> None:
    proto_dir, out_dir = repository_copy
    stub = out_dir / "llm_call_pb2.pyi"
    stub.write_text(stub.read_text(encoding="utf-8") + "# A hand edit.\n", encoding="utf-8")

    result = protogen("--check", "--proto-dir", proto_dir, "--out-dir", out_dir)

    assert result.returncode == 1
    assert "changed: llm_call_pb2.pyi\n" in result.stderr
    assert "make proto" in result.stderr
