"""The gold-only windows the hard-coding scan looks for (spec §8.5), and the setup that builds them.

Text is normalized before it is cut into windows: NFKC-normalized, its invisible format characters
such as zero-width spaces dropped, lowercased, every run of whitespace made one space, and the
ends stripped. For ASCII text, as SQL is, only the case and the whitespace change. A window is
any WINDOW characters of normalized text. A gold-only window is a window of a task's gold
optimized_sql that occurs in no task's base_sql and not in the schema DDL. Standard TPC-H joins
recur across base queries and golds, so only gold-only windows mark a candidate that copied a
gold answer.

The setup reads the full dataset, gold included, which only the setup tools may do (spec §8.5),
and T09's schema snapshots. It stores the SHA-256 of each gold-only window outside any
repository, with mode 0600 in a directory of mode 0700, and the sandbox never sees the store.
The store is gold-equivalent: windows that overlap one character at a time can be rebuilt into
the gold from the base queries, so it is kept as private as the gold itself. A task without both
queries, or a store with no window or another width, is refused, since the scan would then pass
everything.
Run it as::

    python -m goaly.rsi.guards.gold_windows --dataset effi-sql-pg.jsonl \\
        --schema schema_snapshot.json [--out ~/.cache/goaly/effisql/gold_windows.json]
"""

import argparse
import hashlib
import json
import os
import re
import secrets
import sys
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from goaly.rsi.git import GitError, run_git

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
_QUERIES = ("base_sql", "optimized_sql")


def normalize(text: str) -> str:
    """NFKC-normalize ``text``, drop its format characters, lowercase it, make every run of
    whitespace one space, and strip the ends.
    """
    folded = unicodedata.normalize("NFKC", text)
    visible = "".join(char for char in folded if unicodedata.category(char) != "Cf")
    return " ".join(visible.lower().split())


def windows(text: str, width: int = WINDOW) -> Iterator[str]:
    """Yield every run of ``width`` characters of ``text``, in order."""
    for start in range(len(text) - width + 1):
        yield text[start : start + width]


def _hash(window: str) -> str:
    return hashlib.sha256(window.encode("utf-8", "surrogateescape")).hexdigest()


_HEX = re.compile(r"[0-9a-f]{64}")


class _Store(BaseModel):
    """The store's file, as it is written and read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: str
    width: int = Field(ge=WINDOW, le=WINDOW)
    dataset_sha256: str = Field(pattern=r"^([0-9a-f]{64})?$")
    windows: list[str] = Field(min_length=1)


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

        The file gets mode 0600 and a new directory 0700. Raises ValueError for a store with no
        window or another width, and for a path inside any git repository, ignored or not, its
        .git included, so that the store is never committed or seen by a sandbox.
        """
        path = path.expanduser().resolve()
        self.check()
        if _in_repository(path):
            raise ValueError(f"{path}: store the gold windows outside any repository")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        document = {
            "format": _FORMAT,
            "width": self.width,
            "dataset_sha256": self.dataset_sha256,
            "windows": sorted(self.hashes),
        }
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(json.dumps(document, indent=1) + "\n")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def check(self) -> None:
        """Raise ValueError for a store with no window or another width: the scan would then
        pass every candidate.
        """
        if not self.hashes:
            raise ValueError("no gold-only window: the scan would pass every candidate")
        if self.width != WINDOW:
            raise ValueError(f"the windows are {self.width} characters wide, not {WINDOW}")

    @classmethod
    def load(cls, path: Path) -> "GoldWindows":
        """Read the store at ``path``; raise ValueError if it is not one, or if it holds no
        window, windows of another width, or an entry that is not a SHA-256.
        """
        try:
            store = _Store.model_validate_json(path.expanduser().read_bytes())
        except ValidationError as error:
            raise ValueError(f"{path}: not a gold windows store: {error}") from None
        if store.format != _FORMAT:
            raise ValueError(f"{path}: not a gold windows store in the format {_FORMAT}")
        bad = [entry for entry in store.windows if not _HEX.fullmatch(entry)]
        if bad:
            raise ValueError(f"{path}: {len(bad)} entries are not SHA-256 hashes")
        windows = cls(frozenset(store.windows), store.width, store.dataset_sha256)
        windows.check()
        return windows


def build_from_files(dataset: Path, schemas: Sequence[Path], *, width: int = WINDOW) -> GoldWindows:
    """Build the windows from the full dataset, a JSONL file of tasks with base_sql and
    optimized_sql, and from the DDL of T09's schema snapshots.

    Raises ValueError, naming the file and line but never the text, for a line that is not a
    task with both queries: T08's gold-free runtime file, for one, has no gold to build from.
    """
    data = dataset.read_bytes()
    golds, bases = [], []
    for number, line in enumerate(data.decode("utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        task = json.loads(line)
        queries = [task.get(field) if isinstance(task, dict) else None for field in _QUERIES]
        if not all(isinstance(query, str) and query.strip() for query in queries):
            raise ValueError(f"{dataset}:{number}: not a task with base_sql and optimized_sql")
        bases.append(str(queries[0]))
        golds.append(str(queries[1]))
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


def _in_repository(path: Path) -> bool:
    """Whether ``path``, absolute, lies anywhere in a git repository: its working tree, ignored
    or not, or its git directory.
    """
    existing = path.parent
    while not existing.exists():
        existing = existing.parent
    try:
        run_git(["rev-parse", "--git-dir"], cwd=existing)
    except GitError:
        return False
    return True


if __name__ == "__main__":  # pragma: no cover  (tested through a subprocess)
    raise SystemExit(main())
