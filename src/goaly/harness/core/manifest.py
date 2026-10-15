"""The run manifest (spec §6.1, §10): run IDs, the hashes that key and describe a run, where its
code and machine came from, and the manifest file itself.

The harness hash is a content hash of ``src/goaly/harness/core/`` plus the run's pack, as
RunManifest.harness_hash defines it. It covers every file under those directories except Python
bytecode, which the sources determine, so a pack's config.yaml, prompt and skills count. Nothing
else does: bench, contracts and the dependencies are frozen while RSI runs, and the manifest
records the commit and uv.lock's hash for them separately.
"""

import hashlib
import json
import logging
import os
import platform
import secrets
import shutil
import socket
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from google.protobuf import json_format

from goaly.contracts.gen.manifest_pb2 import MachineInfo, RunManifest
from goaly.contracts.interfaces import JsonValue
from goaly.contracts.jsonl import from_json_line, to_json_line
from goaly.harness.core.atomic import write_text_atomically

__all__ = [
    "CORE_DIR",
    "GitState",
    "config_hash",
    "content_hash",
    "core_hash",
    "cpu_model",
    "git_state",
    "harness_hash",
    "machine_info",
    "new_run_id",
    "provenance",
    "read_manifest",
    "sha256_file",
    "write_manifest",
]

_logger = logging.getLogger(__name__)

# src/goaly/harness/core/: the part of the harness RSI cannot edit.
CORE_DIR = Path(__file__).resolve().parent
_BYTECODE_DIR = "__pycache__"
_BYTECODE_SUFFIXES = (".pyc", ".pyo")
_GIT_TIMEOUT_S = 60


def new_run_id(now: datetime | None = None) -> str:
    """Return a new run ID: the start time in UTC, to the second, and 8 random hex digits.

    For example ``20261015T093012Z-3f2a9c01``: IDs sort by start time, and runs started in the
    same second still differ. ``now`` defaults to the current time and must have a time zone.
    """
    moment = datetime.now(UTC) if now is None else now
    if moment.utcoffset() is None:
        raise ValueError("a run ID needs a time with a time zone")
    return f"{moment.astimezone(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def content_hash(trees: Mapping[str, Path]) -> str:
    """Return the SHA-256, in hex, of every file under the directories in ``trees``.

    Each file counts with its contents and its path: the tree's label, then the path within the
    tree. Python bytecode (``__pycache__``, ``*.pyc``, ``*.pyo``) is left out. A symbolic link
    raises ValueError: what it links to could change while the hash stayed the same, and reading
    it could read outside the trees. It reads every file, so it blocks: run it in a worker
    thread from the event loop.
    """
    files: list[tuple[str, str, str]] = []
    for label, root in trees.items():
        if not root.is_dir():
            raise FileNotFoundError(f"no such directory: {root}")
        for directory, dirnames, filenames in root.walk():
            dirnames[:] = [name for name in dirnames if name != _BYTECODE_DIR]
            for name in filenames:  # Path.walk lists links, to directories too, among the files.
                path = directory / name
                if name.endswith(_BYTECODE_SUFFIXES):
                    continue
                if path.is_symlink():
                    raise ValueError(f"{path} is a symbolic link, which cannot be hashed")
                where = f"{label}/{path.relative_to(root).as_posix()}"
                files.append((where, "file", sha256_file(path)))
    listing = json.dumps(sorted(files), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def harness_hash(pack_dir: Path) -> str:
    """Return the harness hash: the content hash of core plus the pack in ``pack_dir``.

    It reads both trees, so it blocks: run it in a worker thread from the event loop.
    """
    return content_hash({"core": CORE_DIR, "pack": pack_dir})


def core_hash() -> str:
    """Return the content hash of core alone, for a harness without a pack, such as a test's.

    It reads core's files, so it blocks: run it in a worker thread from the event loop.
    """
    return content_hash({"core": CORE_DIR})


def config_hash(config: dict[str, JsonValue]) -> str:
    """Return the SHA-256, in hex, of ``config`` as JSON with its keys sorted at every level.

    Raise ValueError for a value JSON cannot hold, such as NaN or an object of another type.
    """
    try:
        text = json.dumps(
            config, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"the config holds a value JSON cannot hold: {error}") from error
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256, in hex, of the file at ``path``."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@dataclass(frozen=True)
class GitState:
    """Where a checkout stands."""

    # The commit checked out.
    commit: str
    # True if the working tree has uncommitted changes, untracked files included.
    dirty: bool
    # The checkout's top directory.
    root: Path


def git_state(directory: Path) -> GitState | None:
    """Return the state of the git checkout that holds ``directory``.

    Return None if git is not installed, ``directory`` is in no checkout, the checkout has no
    commit yet, or git does not answer within a minute. GIT_* variables of the calling process,
    which a git hook sets, are not passed on. git runs as a subprocess, so this blocks: run it in
    a worker thread from the event loop.
    """
    git = shutil.which("git")
    if git is None:
        return None
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}

    def output(*args: str) -> str | None:
        try:
            # The arguments are fixed strings and a path, run without a shell.
            completed = subprocess.run(  # noqa: S603
                [git, "--no-optional-locks", "-C", str(directory), *args],
                env=environment,
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:  # A hung git, say on a locked repository.
            _logger.warning(
                "git did not answer within %d s in %s: no git state", _GIT_TIMEOUT_S, directory
            )
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    root = output("rev-parse", "--show-toplevel")
    commit = output("rev-parse", "--verify", "--quiet", "HEAD")
    status = output("status", "--porcelain")
    if root is None or commit is None or status is None:
        return None
    return GitState(commit=commit, dirty=bool(status), root=Path(root))


def machine_info() -> MachineInfo:
    """Describe this machine, as RunManifest.machine records it."""
    return MachineInfo(
        hostname=socket.gethostname(),
        os=platform.platform(),
        cpu_model=cpu_model(),
        cpu_count=os.cpu_count() or 0,
        memory_bytes=_memory_bytes(),
        python_version=platform.python_version(),
    )


def cpu_model(cpuinfo: Path = Path("/proc/cpuinfo")) -> str:
    """Return the CPU's model name from Linux's ``cpuinfo``, else what the platform reports."""
    try:
        with cpuinfo.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                key, _, value = line.partition(":")
                if key.strip() == "model name":
                    return value.strip()
    except OSError:
        pass
    return platform.processor()


def _memory_bytes() -> int:
    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):  # Not a POSIX system that reports it.
        return 0
    return pages * page_size if pages > 0 and page_size > 0 else 0  # -1: not reported.


def provenance(directory: Path) -> RunManifest:
    """Return a manifest that holds only where a run's code and machine came from.

    It has the commit and dirty flag of the checkout that holds ``directory``, the SHA-256 of
    the checkout's uv.lock, and the machine. Outside a checkout it has only the machine, and a
    warning is logged. Merge it into the run's manifest with ``MergeFrom``. It runs git and
    reads files, so it blocks: run it in a worker thread from the event loop.
    """
    found = RunManifest(machine=machine_info())
    state = git_state(directory)
    if state is None:
        _logger.warning(
            "found no git state for %s (no checkout, no commit yet, or git did not answer):"
            " the manifest records no commit and no uv.lock",
            directory,
        )
        return found
    found.git_commit = state.commit
    found.git_dirty = state.dirty
    lock = state.root / "uv.lock"
    if lock.is_file():
        found.uv_lock_sha256 = sha256_file(lock)
    return found


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Write ``manifest`` to ``path`` as indented JSON, replacing the file atomically.

    The keys follow the .proto's field order and the protobuf JSON mapping, as in a JSONL line.
    It writes and syncs a file, so it blocks: run it in a worker thread from the event loop.
    """
    fields = json.loads(to_json_line(manifest))
    write_text_atomically(path, json.dumps(fields, ensure_ascii=False, indent=2) + "\n")


def read_manifest(path: Path) -> RunManifest:
    """Read the manifest :func:`write_manifest` wrote, ignoring unknown fields.

    Raise ValueError, naming ``path``, if the file does not hold one.
    """
    try:
        return from_json_line(path.read_bytes().decode("utf-8"), RunManifest)
    except (UnicodeDecodeError, json_format.ParseError) as error:
        raise ValueError(f"{path} does not hold a RunManifest: {error}") from error
