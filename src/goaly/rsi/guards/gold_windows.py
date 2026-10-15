"""The gold-only windows the hard-coding scan looks for (spec §8.5), and the setup that builds them.

Text is normalized before it is cut into windows: lowercased, every run of whitespace made one
space, and the ends stripped. A window is any WINDOW characters of normalized text. A gold-only
window is a window of a task's gold optimized_sql that occurs in no task's base_sql and not in
the schema DDL. Standard TPC-H joins recur across base queries and golds, so only gold-only
windows mark a candidate that copied a gold answer.

The setup reads the full dataset, gold included, which only the setup tools may do (spec §8.5),
and T09's schema snapshots. It stores the SHA-256 of each gold-only window outside the
repository: the gold text itself is never stored again, and the sandbox never sees the store.
Run it as::

    python -m goaly.rsi.guards.gold_windows --dataset effi-sql-pg.jsonl \\
        --schema schema_snapshot.json [--out ~/.cache/goaly/effisql/gold_windows.json]
"""

import argparse
import hashlib
import json
import secrets
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from goaly.rsi.git import GitError, run_git, run_git_unless_no

__all__ = [
    "DEFAULT_STORE",
    "WINDOW",
    "GoldWindows",
    "build_from_files",
    "main",
    "normalize",
    "windows",
]

WINDOW = 40
DEFAULT_STORE = Path("~/.cache/goaly/effisql/gold_windows.json")
_FORMAT = "goaly-gold-windows/1"


def normalize(text: str) -> str:
    """Lowercase ``text``, make every run of whitespace one space, and strip the ends."""
    return " ".join(text.lower().split())


def windows(text: str, width: int = WINDOW) -> Iterator[str]:
    """Yield every run of ``width`` characters of ``text``, in order."""
    for start in range(len(text) - width + 1):
        yield text[start : start + width]


def _hash(window: str) -> str:
    return hashlib.sha256(window.encode("utf-8", "surrogateescape")).hexdigest()


@dataclass(frozen=True)
class GoldWindows:
    """The SHA-256 hashes of the gold-only windows, with the dataset they came from."""

    hashes: frozenset[str]
    width: int = WINDOW
    dataset_sha256: str = ""

    @classmethod
    def build(
        cls,
        golds: Iterable[str],
        bases: Iterable[str],
        ddl: Iterable[str],
        *,
        width: int = WINDOW,
        dataset_sha256: str = "",
    ) -> "GoldWindows":
        """Hash the windows of ``golds`` that occur in none of ``bases`` and ``ddl``."""
        common: set[str] = set()
        for text in (*bases, *ddl):
            common.update(windows(normalize(text), width))
        gold: set[str] = set()
        for text in golds:
            gold.update(windows(normalize(text), width))
        return cls(frozenset(map(_hash, gold - common)), width, dataset_sha256)

    def __contains__(self, window: object) -> bool:
        """Whether ``window``, normalized text, is a gold-only window."""
        return isinstance(window, str) and _hash(window) in self.hashes

    def save(self, path: Path) -> None:
        """Write the store to ``path``, replacing any; readers see the old file or the new one.

        Raises ValueError for a path inside a git working tree that git would not ignore, so
        that the store is never committed or seen by a sandbox.
        """
        path = path.expanduser()
        if _committable(path):
            raise ValueError(f"{path}: store the gold windows outside the repository")
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "format": _FORMAT,
            "width": self.width,
            "dataset_sha256": self.dataset_sha256,
            "windows": sorted(self.hashes),
        }
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        try:
            temporary.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "GoldWindows":
        """Read the store at ``path``; raise ValueError if it is not one."""
        document = json.loads(path.expanduser().read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("format") != _FORMAT:
            raise ValueError(f"{path}: not a gold windows store in the format {_FORMAT}")
        return cls(
            frozenset(document["windows"]),
            int(document["width"]),
            str(document["dataset_sha256"]),
        )


def build_from_files(dataset: Path, schemas: Sequence[Path], *, width: int = WINDOW) -> GoldWindows:
    """Build the windows from the full dataset, a JSONL file of tasks with base_sql and
    optimized_sql, and from the DDL of T09's schema snapshots.
    """
    data = dataset.read_bytes()
    golds, bases = [], []
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            task = json.loads(line)
            bases.append(task["base_sql"])
            golds.append(task.get("optimized_sql") or "")
    ddl = [json.loads(schema.read_text(encoding="utf-8"))["ddl"] for schema in schemas]
    return GoldWindows.build(
        golds, bases, ddl, width=width, dataset_sha256=hashlib.sha256(data).hexdigest()
    )


def main(argv: Sequence[str] | None = None) -> int:
    """The setup command: build the windows and store them; 0 on success."""
    parser = argparse.ArgumentParser(
        prog="python -m goaly.rsi.guards.gold_windows",
        description="Store the hashes of the gold-only windows the hard-coding scan looks for.",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="the full Effi-SQL JSONL")
    parser.add_argument(
        "--schema", type=Path, action="append", required=True, help="a schema_snapshot.json"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_STORE, help="where to store them")
    arguments = parser.parse_args(argv)
    gold = build_from_files(arguments.dataset, arguments.schema)
    gold.save(arguments.out)
    sys.stdout.write(f"{len(gold.hashes)} gold-only windows stored in {arguments.out}\n")
    return 0


def _committable(path: Path) -> bool:
    """Whether ``path`` lies in a git working tree and git would not ignore it there."""
    existing = path.parent
    while not existing.exists():
        existing = existing.parent
    try:
        inside = run_git(["rev-parse", "--is-inside-work-tree"], cwd=existing).strip() == "true"
    except GitError:
        return False  # Not in a repository at all.
    if not inside:
        return False  # Inside a .git directory.
    # check-ignore answers "not ignored" by exiting with 1.
    check = ["check-ignore", "--quiet", "--", str(path)]
    return run_git_unless_no(check, cwd=existing) is None


if __name__ == "__main__":  # pragma: no cover  (tested through a subprocess)
    raise SystemExit(main())
