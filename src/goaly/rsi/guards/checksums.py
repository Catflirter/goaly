"""GUARD_CHECKSUMS (spec §8.5): the layers RSI cannot change are as the lane found them.

bench, harness core and contracts are frozen while RSI runs (spec §8.1). A lane takes their
checksums when it starts, from its first best version's snapshot, and checks a candidate's
snapshot against them before every evaluation.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import GUARD_CHECKSUMS, GuardResult

__all__ = ["FROZEN_DIRS", "check_checksums", "frozen_checksums"]

FROZEN_DIRS = ("src/goaly/bench", "src/goaly/harness/core", "src/goaly/contracts")
_CACHE = "__pycache__"  # Bytecode Python writes as it imports; never part of a layer.


def frozen_checksums(root: Path) -> dict[str, str]:
    """Return one SHA-256 for each frozen directory under ``root``, over its files' paths and
    contents and its symlinks' targets.
    """
    return {name: _tree_checksum(root / name) for name in FROZEN_DIRS}


def check_checksums(root: Path, expected: Mapping[str, str]) -> GuardResult:
    """Check that each frozen directory under ``root`` still has its ``expected`` checksum."""
    actual = frozen_checksums(root)
    changed = [name for name in FROZEN_DIRS if actual[name] != expected.get(name)]
    if changed:
        return GuardResult(
            guard=GUARD_CHECKSUMS, passed=False, detail=f"{', '.join(changed)} changed"
        )
    return GuardResult(
        guard=GUARD_CHECKSUMS, passed=True, detail=f"{', '.join(FROZEN_DIRS)} unchanged"
    )


def _tree_checksum(directory: Path) -> str:
    digest = hashlib.sha256()
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if _CACHE in relative.parts:
                continue
            if path.is_symlink():
                entry = f"link\0{relative.as_posix()}\0{path.readlink()}"
            elif path.is_file():
                content = hashlib.sha256(path.read_bytes()).hexdigest()
                entry = f"file\0{relative.as_posix()}\0{content}"
            else:
                continue  # A directory counts through its files.
            digest.update(entry.encode("utf-8", "surrogateescape") + b"\n")
    return digest.hexdigest()
