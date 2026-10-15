"""The catalog fingerprint of a grading instance (spec §8.5): proof that nobody changed it.

The grader computes it before every batch and compares it with the one taken when the instance
came up. For each database it holds the indexes (``pg_indexes``), the size in pages of every
relation in the public schema (``pg_class.relpages``), and each table's write counters
(``n_tup_ins``, ``n_tup_upd`` and ``n_tup_del`` from ``pg_stat_user_tables``). An added index,
a write, or a rewrite changes it. Reads never do, and neither does a restart after a clean
shutdown, which keeps the counters. Any role that can connect can compute it.
"""

from collections.abc import Mapping
from typing import Any

import psycopg
from pydantic import BaseModel, ConfigDict

__all__ = ["DatabaseFingerprint", "Fingerprint", "compute_fingerprint", "fingerprint_changes"]

_INDEXES_SQL = "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'"
_RELPAGES_SQL = """
SELECT c.relname, c.relpages FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'i', 'm', 't')
"""
_WRITES_SQL = """
SELECT relname, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables
WHERE schemaname = 'public'
"""


class DatabaseFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    indexes: dict[str, str]  # Index name → its definition.
    relpages: dict[str, int]  # Relation name → its size in pages.
    writes: dict[str, tuple[int, int, int]]  # Table name → rows inserted, updated, deleted.


class Fingerprint(BaseModel):
    """An instance's fingerprint, database by database."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    databases: dict[str, DatabaseFingerprint]


async def compute_fingerprint(conns: Mapping[str, psycopg.AsyncConnection[Any]]) -> Fingerprint:
    """Compute the fingerprint from one connection to each database, keyed by its name."""
    databases = {}
    for db, conn in conns.items():
        indexes = await (await conn.execute(_INDEXES_SQL)).fetchall()
        relpages = await (await conn.execute(_RELPAGES_SQL)).fetchall()
        writes = await (await conn.execute(_WRITES_SQL)).fetchall()
        databases[db] = DatabaseFingerprint(
            indexes=dict(indexes),
            relpages={name: int(pages) for name, pages in relpages},
            writes={name: (int(i), int(u), int(d)) for name, i, u, d in writes},
        )
    return Fingerprint(databases=databases)


def fingerprint_changes(before: Fingerprint, after: Fingerprint) -> list[str]:
    """Name every difference from ``before`` to ``after``; an empty list means unchanged."""
    changes: list[str] = []
    for db, old in before.databases.items():
        new = after.databases.get(db)
        if new is None:
            changes.append(f"{db}: database missing")
            continue
        changes += _differences(db, "index", old.indexes, new.indexes, _index)
        changes += _differences(db, "relation", old.relpages, new.relpages, _pages)
        changes += _differences(db, "table", old.writes, new.writes, _writes)
    return changes


def _differences[T](
    db: str, kind: str, old: Mapping[str, T], new: Mapping[str, T], describe: Any
) -> list[str]:
    lines = []
    for name in sorted(old.keys() | new.keys()):
        if old.get(name) != new.get(name):
            lines.append(f"{db}: {kind} {name} {describe(old.get(name), new.get(name))}")
    return lines


def _index(old: str | None, new: str | None) -> str:
    if new is None:
        return "was dropped"
    return f"was added: {new}" if old is None else f"was redefined: {new}"


def _pages(old: int | None, new: int | None) -> str:
    if new is None:
        return "disappeared"
    return f"appeared, {new} pages" if old is None else f"has {new} pages, not {old}"


def _writes(old: tuple[int, int, int] | None, new: tuple[int, int, int] | None) -> str:
    if new is None:
        return "lost its write counters"
    if old is None:
        return f"appeared, with writes {new}"
    return f"has writes (inserted, updated, deleted) {new}, not {old}"
