"""Tests for goaly.harness.core.manifest: run IDs, hashes, provenance and the manifest file."""

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path, PurePath
from typing import Any

import pytest
from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.manifest_pb2 import MachineInfo, ModelSettings, RunManifest
from goaly.contracts.gen.task_pb2 import SPLIT_VAL
from goaly.contracts.interfaces import JsonValue
from goaly.contracts.jsonl import to_json_line
from goaly.harness.core import manifest
from goaly.harness.core.manifest import (
    CORE_DIR,
    CORE_IN_CHECKOUT,
    REPO_ROOT,
    GitState,
    checkout_harness_hash,
    config_hash,
    content_hash,
    core_hash,
    git_state,
    harness_hash,
    machine_info,
    new_run_id,
    provenance,
    read_manifest,
    sha256_file,
    write_manifest,
)

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git is not installed")


def make_tree(root: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    return make_tree(
        tmp_path / "pack",
        {"prompt.md": "Be fast.\n", "config.yaml": "steps: 30\n", "skills/joins.md": "Index.\n"},
    )


# Run IDs.


def test_a_run_id_is_the_utc_start_time_and_a_random_suffix() -> None:
    eastern = timezone(timedelta(hours=-7))
    started = datetime(2026, 10, 15, 2, 30, 12, tzinfo=eastern)

    run_id = new_run_id(started)

    assert re.fullmatch(r"20261015T093012Z-[0-9a-f]{8}", run_id), run_id


def test_run_ids_differ_even_when_started_together() -> None:
    now = datetime.now(UTC)

    assert len({new_run_id(now) for _ in range(100)}) == 100


def test_a_run_id_defaults_to_now() -> None:
    before = datetime.now(UTC).replace(microsecond=0)

    stamp = datetime.strptime(new_run_id().split("-")[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)

    assert before <= stamp <= datetime.now(UTC)


def test_a_run_id_needs_an_aware_time() -> None:
    with pytest.raises(ValueError, match="time zone"):
        new_run_id(datetime(2026, 10, 15, 9, 30))  # noqa: DTZ001  # Naive on purpose.


# Content hashes.


def test_a_content_hash_depends_only_on_the_files(tmp_path: Path) -> None:
    files = {"a.py": "x = 1\n", "sub/b.md": "text\n", "sub/deeper/c.txt": ""}
    first = make_tree(tmp_path / "first", files)
    second = make_tree(tmp_path / "second", dict(reversed(files.items())))

    assert content_hash({"tree": first}) == content_hash({"tree": second})
    assert re.fullmatch(r"[0-9a-f]{64}", content_hash({"tree": first}))


def test_a_content_hash_does_not_depend_on_the_listing_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_tree(tmp_path / "tree", {"a.py": "1\n", "b.py": "2\n", "c/d.py": "3\n"})
    before = content_hash({"tree": root})
    walk = Path.walk

    def reversed_walk(self: Path) -> Iterator[tuple[Path, list[str], list[str]]]:
        for directory, dirnames, filenames in walk(self):
            dirnames.reverse()
            yield directory, dirnames, filenames[::-1]

    monkeypatch.setattr(Path, "walk", reversed_walk)

    assert content_hash({"tree": root}) == before


@pytest.mark.parametrize(
    "change",
    [
        lambda root: (root / "a.py").write_text("x = 2\n", encoding="utf-8"),
        lambda root: (root / "a.py").rename(root / "renamed.py"),
        lambda root: (root / "a.py").rename(root / "sub" / "a.py"),
        lambda root: (root / "new.md").touch(),
        lambda root: (root / "a.py").unlink(),
    ],
    ids=["content", "rename", "move", "new empty file", "removal"],
)
def test_a_content_hash_changes_with_any_file(
    tmp_path: Path, change: Callable[[Path], object]
) -> None:
    root = make_tree(tmp_path / "tree", {"a.py": "x = 1\n", "sub/b.md": "text\n"})
    before = content_hash({"tree": root})

    change(root)

    assert content_hash({"tree": root}) != before


def test_a_content_hash_ignores_bytecode(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "tree", {"a.py": "x = 1\n"})
    before = content_hash({"tree": root})

    bytecode = {"__pycache__/a.cpython-312.pyc": "bytes", "sub/__pycache__/b.pyc": "", "c.pyc": ""}
    make_tree(root, {**bytecode, "__pycache__/anything.txt": "derived\n"})

    assert content_hash({"tree": root}) == before


def test_a_content_hash_tells_trees_apart_by_label(tmp_path: Path) -> None:
    one = make_tree(tmp_path / "one", {"a.py": "1\n"})
    two = make_tree(tmp_path / "two", {"a.py": "2\n"})

    assert content_hash({"core": one, "pack": two}) != content_hash({"core": two, "pack": one})
    assert content_hash({"core": one}) != content_hash({"pack": one})


# A link's target could change while the hash stays the same, and reading it would read outside
# the trees, so a tree that holds a link of any kind cannot be hashed.
@pytest.mark.parametrize(
    "link",
    [
        lambda root, outside: (root / "linked.txt").symlink_to(outside / "a.txt"),
        lambda root, outside: (root / "linked").symlink_to(outside, target_is_directory=True),
        lambda root, outside: (root / "gone").symlink_to(outside / "missing"),
        lambda root, outside: (root / "sub" / "inner").symlink_to(root / "a.py"),
    ],
    ids=["to a file", "to a directory", "dangling", "inside the tree"],
)
def test_a_content_hash_refuses_a_link(
    tmp_path: Path, link: Callable[[Path, Path], object]
) -> None:
    outside = make_tree(tmp_path / "outside", {"a.txt": "a\n"})
    root = make_tree(tmp_path / "tree", {"a.py": "x = 1\n", "sub/b.py": "y = 2\n"})
    link(root, outside)

    with pytest.raises(ValueError, match="is a symbolic link, which cannot be hashed"):
        content_hash({"tree": root})


def test_a_content_hash_needs_each_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        content_hash({"tree": tmp_path / "missing"})


def test_the_harness_hash_covers_core_and_the_pack(pack: Path) -> None:
    assert Path(manifest.__file__).resolve().parent == CORE_DIR
    assert CORE_DIR.parts[-3:] == ("goaly", "harness", "core")
    assert harness_hash(pack) == content_hash({"core": CORE_DIR, "pack": pack})
    assert core_hash() == content_hash({"core": CORE_DIR})

    before = harness_hash(pack)
    (pack / "skills" / "joins.md").write_text("Use a hash join.\n", encoding="utf-8")

    assert harness_hash(pack) != before


def test_a_config_hash_ignores_key_order_at_every_level() -> None:
    first: dict[str, JsonValue] = {"seeds": [0, 1], "llm": {"qwen": 40, "glm": 8}, "bench": "x"}
    second: dict[str, JsonValue] = {"bench": "x", "llm": {"glm": 8, "qwen": 40}, "seeds": [0, 1]}

    assert config_hash(first) == config_hash(second)
    assert re.fullmatch(r"[0-9a-f]{64}", config_hash(first))


@pytest.mark.parametrize(
    "other",
    [
        {"seeds": [1, 0], "bench": "x"},
        {"seeds": [0, 1], "bench": "y"},
        {"seeds": [0, 1], "bench": "x", "max_tasks": 16},
        {"seeds": [0, 1]},
    ],
)
def test_a_config_hash_changes_with_any_value(other: dict[str, JsonValue]) -> None:
    assert config_hash(other) != config_hash({"seeds": [0, 1], "bench": "x"})


@pytest.mark.parametrize("value", [float("nan"), {1, 2}, Path("/x")], ids=["nan", "set", "path"])
def test_a_config_hash_refuses_values_json_cannot_hold(value: object) -> None:
    with pytest.raises(ValueError, match="JSON cannot hold"):
        config_hash({"timeout_s": value})  # type: ignore[dict-item]  # Not JSON, on purpose.


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "uv.lock"
    path.write_bytes(b"version = 1\n")

    assert sha256_file(path) == hashlib.sha256(b"version = 1\n").hexdigest()


# Git.


def git(directory: Path, *args: str) -> str:
    assert GIT is not None
    # The arguments are fixed strings built by the tests, not untrusted input.
    completed = subprocess.run(  # noqa: S603
        [GIT, "-C", str(directory), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return completed.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = make_tree(tmp_path / "repo", {"src/a.py": "x = 1\n", ".gitignore": "runs/\n"})
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    return root


@needs_git
def test_git_state_names_the_commit_of_a_clean_checkout(repo: Path) -> None:
    state = git_state(repo / "src")

    assert state == GitState(commit=git(repo, "rev-parse", "HEAD"), dirty=False, root=repo)


@needs_git
@pytest.mark.parametrize(
    ("change", "dirty"),
    [
        (lambda root: (root / "src" / "a.py").write_text("x = 2\n", encoding="utf-8"), True),
        (lambda root: (root / "src" / "new.py").touch(), True),
        (lambda root: make_tree(root, {"runs/r1/results.jsonl": "{}\n"}), False),
    ],
    ids=["modified", "untracked", "ignored"],
)
def test_git_state_reports_uncommitted_changes(
    repo: Path, change: Callable[[Path], object], dirty: bool
) -> None:
    change(repo)

    state = git_state(repo)

    assert state is not None
    assert state.dirty is dirty


@needs_git
def test_git_state_ignores_git_variables_of_the_calling_process(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = make_tree(tmp_path / "other", {"b.txt": "b\n"})
    git(other, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))  # As inside a git hook.
    monkeypatch.setenv("GIT_WORK_TREE", str(other))

    state = git_state(repo)

    assert state is not None
    assert state.root == repo


@needs_git
def test_git_state_is_none_outside_a_repository_or_before_the_first_commit(
    tmp_path: Path,
) -> None:
    plain = make_tree(tmp_path / "plain", {"a.txt": "a\n"})
    assert git_state(plain) is None

    git(plain, "init", "-q")
    assert git_state(plain) is None


def test_git_state_is_none_when_git_hangs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    run = subprocess.run

    def hung(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if Path(command[0]).name != "git":
            return run(command, **kwargs)  # Such as the platform module's own uname.
        raise subprocess.TimeoutExpired(cmd=command, timeout=60)

    monkeypatch.setattr(subprocess, "run", hung)

    with caplog.at_level(logging.WARNING, logger="goaly.harness.core.manifest"):
        assert git_state(tmp_path) is None
        found = provenance(tmp_path)

    assert "git did not answer within 60 s" in caplog.text
    assert "not in a git checkout" not in caplog.text  # It may well be in one.
    assert not found.git_commit


def test_git_state_is_none_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert git_state(tmp_path) is None


# The machine.


def test_machine_info_describes_this_machine() -> None:
    info = machine_info()

    assert info.hostname == socket.gethostname()
    assert info.os == platform.platform()
    assert info.cpu_count == (os.cpu_count() or 0)
    assert info.python_version == platform.python_version()
    # What the platform reports, if anything: /proc/cpuinfo and sysconf are not everywhere.
    assert info.cpu_model == manifest.cpu_model()
    assert info.memory_bytes >= 0


@pytest.mark.parametrize("sysconf", [ValueError("unknown name"), OSError("unsupported"), -1])
def test_the_memory_is_0_where_the_system_does_not_report_it(
    monkeypatch: pytest.MonkeyPatch, sysconf: Exception | int
) -> None:
    def fake_sysconf(name: str) -> int:
        if isinstance(sysconf, Exception):
            raise sysconf
        return sysconf

    monkeypatch.setattr(os, "sysconf", fake_sysconf)

    assert machine_info().memory_bytes == 0


def test_the_cpu_model_comes_from_cpuinfo(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 9 3900X 12-Core"
        " Processor\n\nprocessor\t: 1\nmodel name\t: AMD Ryzen 9 3900X 12-Core Processor\n",
        encoding="utf-8",
    )

    assert manifest.cpu_model(cpuinfo) == "AMD Ryzen 9 3900X 12-Core Processor"


def test_the_cpu_model_falls_back_to_the_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform, "processor", lambda: "x86_64")
    no_model = tmp_path / "cpuinfo"
    no_model.write_text("processor\t: 0\n", encoding="utf-8")

    assert manifest.cpu_model(tmp_path / "missing") == "x86_64"
    assert manifest.cpu_model(no_model) == "x86_64"


# Provenance.


@needs_git
def test_provenance_records_the_checkout_and_the_machine(repo: Path) -> None:
    (repo / "uv.lock").write_bytes(b"version = 1\n")
    git(repo, "add", "uv.lock")
    git(repo, "commit", "-q", "-m", "lock")

    found = provenance(repo / "src")

    assert found == RunManifest(
        git_commit=git(repo, "rev-parse", "HEAD"),
        git_dirty=False,
        uv_lock_sha256=hashlib.sha256(b"version = 1\n").hexdigest(),
        machine=machine_info(),
    )


@needs_git
def test_provenance_leaves_out_a_missing_lock_file(repo: Path) -> None:
    found = provenance(repo)

    assert found.git_commit == git(repo, "rev-parse", "HEAD")
    assert found.uv_lock_sha256 == ""


def test_provenance_outside_a_repository_has_only_the_machine(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    found = provenance(tmp_path)

    assert found == RunManifest(machine=machine_info())
    assert "found no git state for" in caplog.text


# The manifest file.


def full_manifest() -> RunManifest:
    return RunManifest(
        run_id="20261015T093012Z-3f2a9c01",
        benchmark="effisql",
        split=SPLIT_VAL,
        git_commit="a" * 40,
        harness_hash="b" * 64,
        config_hash="c" * 64,
        dataset_revision="86cbcc9efbd31db32a46897d47cbf865b1cba3a4",
        dataset_notes=["dropped 5 tasks over 300 s"],
        models=[ModelSettings(model="qwen3.6-35b-a3b", temperature=0.6, top_k=20, thinking=True)],
        seeds=[0, 1],
        instance_ids=["3", "17"],
        command_line=["goaly", "run", "--bench", "effisql"],
        started_at=Timestamp(seconds=1_790_000_000),
        machine=MachineInfo(hostname="host", cpu_count=24),
        uv_lock_sha256="d" * 64,
    )


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    """An empty directory: tmp_path also holds the isolated HOME of conftest.py."""
    path = tmp_path / "run"
    path.mkdir()
    return path


def test_a_manifest_reads_back_as_written(directory: Path) -> None:
    path = directory / "manifest.json"

    write_manifest(path, full_manifest())

    assert read_manifest(path) == full_manifest()
    assert list(directory.iterdir()) == [path]  # Written atomically: no temporary file is left.


def test_a_manifest_file_is_indented_json_in_field_order(directory: Path) -> None:
    path = directory / "manifest.json"

    write_manifest(path, full_manifest())

    text = path.read_text(encoding="utf-8")
    expected = json.loads(to_json_line(full_manifest()))
    assert text == json.dumps(expected, ensure_ascii=False, indent=2) + "\n"
    assert list(json.loads(text)) == list(expected)  # Keys in the order the .proto declares.


def test_a_manifest_is_written_by_a_rename_in_its_own_directory(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only a rename within one directory, so one file system, replaces a file atomically.
    path = directory / "manifest.json"
    write_manifest(path, full_manifest())
    renames: list[tuple[Path, Path, str]] = []
    replace = Path.replace

    def recording_replace(source: Path, target: str | PurePath) -> Path:
        renames.append((source, Path(target), Path(target).read_text(encoding="utf-8")))
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    finished = full_manifest()
    finished.finished_at.FromDatetime(datetime(2026, 10, 15, 10, tzinfo=UTC))

    write_manifest(path, finished)

    ((source, target, before),) = renames
    assert (target, source.parent) == (path, directory)
    assert source != target
    assert json.loads(before) == json.loads(to_json_line(full_manifest()))  # Until the rename.
    assert read_manifest(path) == finished


def test_rewriting_a_manifest_replaces_it(directory: Path) -> None:
    path = directory / "manifest.json"
    write_manifest(path, full_manifest())
    finished = full_manifest()
    finished.finished_at.FromDatetime(datetime(2026, 10, 15, 10, tzinfo=UTC))

    write_manifest(path, finished)

    assert read_manifest(path) == finished


def test_reading_a_manifest_ignores_unknown_fields(directory: Path) -> None:
    path = directory / "manifest.json"
    fields = json.loads(to_json_line(full_manifest()))
    path.write_text(json.dumps({**fields, "added_later": 1}), encoding="utf-8")

    assert read_manifest(path) == full_manifest()


@pytest.mark.parametrize("content", [b"", b"[]", b'{"seeds": "x"}', b"\xff"])
def test_reading_something_else_names_the_file(directory: Path, content: bytes) -> None:
    path = directory / "manifest.json"
    path.write_bytes(content)

    with pytest.raises(ValueError, match=r"manifest\.json"):
        read_manifest(path)


@pytest.mark.parametrize("pack", [Path("/abs/pack"), Path("packs/../../outside")])
def test_a_checkouts_pack_must_be_within_it(tmp_path: Path, pack: Path) -> None:
    with pytest.raises(ValueError, match="must be within the checkout"):
        checkout_harness_hash(tmp_path, pack)


def test_this_checkouts_hash_is_the_harness_hash() -> None:
    pack = Path("src/goaly/config")

    assert checkout_harness_hash(REPO_ROOT, pack) == harness_hash(REPO_ROOT / pack)
    assert REPO_ROOT / CORE_IN_CHECKOUT == CORE_DIR
