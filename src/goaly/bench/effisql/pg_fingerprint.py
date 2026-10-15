"""The catalog fingerprint of a grading instance (spec §8.5): evidence that nobody changed it.

The grader computes it before it grades and compares it with a baseline taken when the instance
came up. For each database it holds:

- the indexes (``pg_indexes``);
- each relation's size in pages (``pg_class.relpages``) and file (``relfilenode``), so a
  rewrite such as VACUUM FULL, CLUSTER or TRUNCATE shows even when the size does not;
- each table's write counters (``n_tup_ins``, ``n_tup_upd`` and ``n_tup_del``);
- when those counters were last reset (``pg_stat_database.stats_reset``).

What it can and cannot see:

- An added or dropped index, a rewrite and a write change it; reads never do.
- ``relpages`` changes on VACUUM, ANALYZE, CREATE INDEX and rewrites, never on INSERT, UPDATE
  or DELETE, so a write shows in the counters, not the pages. A backend publishes its counters
  up to about a minute after its transaction ends, so a write that happened just before the
  check can still be missing from it.
- A restart after a clean shutdown keeps the counters. An unclean one, or ``pg_stat_reset()``,
  resets them, and they can no longer show whether rows were written: that is one line per
  database, and the instance fails the check until it is recopied and a new baseline is taken.

Comparing fingerprints across instances, such as checking that all five are equivalent, is the
caller's job. Any role that can connect can compute one. Both functions take one connection
per database, keyed by its name, each in autocommit mode and idle: every query is then its own
transaction, so the counters are read fresh and no transaction is left open.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import psycopg
from psycopg.pq import TransactionStatus
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "FINGERPRINT_QUERIES",
    "DatabaseFingerprint",
    "Fingerprint",
    "compute_fingerprint",
    "compute_fingerprint_sync",
    "fingerprint_changes",
    "fingerprint_from_rows",
]

# What a database's fingerprint is read from: fingerprint_from_rows takes each query's rows.
FINGERPRINT_QUERIES: Mapping[str, str] = {
    "indexes": "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'",
    "relations": (
        "SELECT c.relname, c.relpages, c.relfilenode FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'i', 'm')"
    ),
    "writes": (
        "SELECT relname, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables "
        "WHERE schemaname = 'public'"
    ),
    "stats_reset": (
        "SELECT stats_reset::text FROM pg_stat_database WHERE datname = current_database()"
    ),
}

type _Rows = Sequence[Sequence[Any]]


class DatabaseFingerprint(BaseModel):
    """One database's part of the fingerprint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    indexes: dict[str, str]  # Index name → its definition.
    relpages: dict[str, int]  # Relation name → its size in pages.
    files: dict[str, int]  # Relation name → its relfilenode; a rewrite gives a new one.
    writes: dict[str, tuple[int, int, int]]  # Table name → rows inserted, updated, deleted.
    stats_reset: str | None = None  # When the counters were last reset, if ever.


class Fingerprint(BaseModel):
    """An instance's fingerprint, database by database."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    databases: dict[str, DatabaseFingerprint] = Field(min_length=1)


def fingerprint_from_rows(rows: Mapping[str, _Rows]) -> DatabaseFingerprint:
    """Build one database's fingerprint from the rows of each of :data:`FINGERPRINT_QUERIES`."""
    relations = rows["relations"]
    reset = rows["stats_reset"]
    return DatabaseFingerprint(
        indexes={str(name): str(definition) for name, definition in rows["indexes"]},
        relpages={str(name): int(pages) for name, pages, _ in relations},
        files={str(name): int(node) for name, _, node in relations},
        writes={str(name): (int(i), int(u), int(d)) for name, i, u, d in rows["writes"]},
        stats_reset=str(reset[0][0]) if reset and reset[0][0] is not None else None,
    )


async def compute_fingerprint(conns: Mapping[str, psycopg.AsyncConnection[Any]]) -> Fingerprint:
    """Compute the fingerprint from one autocommit connection to each database, by name."""
    _check(conns)
    databases = {}
    for db, conn in conns.items():
        rows = {}
        for key, query in FINGERPRINT_QUERIES.items():
            rows[key] = await (await conn.execute(query)).fetchall()
        databases[db] = fingerprint_from_rows(rows)
    return Fingerprint(databases=databases)


def compute_fingerprint_sync(conns: Mapping[str, psycopg.Connection[Any]]) -> Fingerprint:
    """:func:`compute_fingerprint` over sync connections, as T10's grading workers hold."""
    _check(conns)
    databases = {}
    for db, conn in conns.items():
        rows = {key: conn.execute(query).fetchall() for key, query in FINGERPRINT_QUERIES.items()}
        databases[db] = fingerprint_from_rows(rows)
    return Fingerprint(databases=databases)


def fingerprint_changes(before: Fingerprint, after: Fingerprint) -> list[str]:
    """Name every difference from ``before`` to ``after``; an empty list means unchanged."""
    changes: list[str] = []
    for db in sorted(before.databases.keys() | after.databases.keys()):
        old, new = before.databases.get(db), after.databases.get(db)
        if old is None or new is None:
            changes.append(f"{db}: database {'appeared' if old is None else 'missing'}")
            continue
        changes += _differences(db, "index", old.indexes, new.indexes, _index)
        changes += _differences(db, "relation", old.relpages, new.relpages, _pages)
        changes += [
            f"{db}: relation {name} was rewritten (file {new.files[name]}, not {old.files[name]})"
            for name in sorted(old.files.keys() & new.files.keys())
            if old.files[name] != new.files[name]
        ]
        if (reset := _reset(old, new)) is not None:
            # Every table's counters would differ, so one line says why instead.
            changes.append(f"{db}: the write counters {reset}, so they cannot show writes")
        else:
            changes += _differences(db, "table", old.writes, new.writes, _writes)
    return changes


def _check(conns: Mapping[str, psycopg.Connection[Any] | psycopg.AsyncConnection[Any]]) -> None:
    if not conns:
        raise ValueError("a fingerprint needs a connection to each database, and none was given")
    for db, conn in conns.items():
        if conn.info.dbname != db:
            raise ValueError(f"the connection given for {db} is to {conn.info.dbname}")
        # In a transaction, the server would keep returning the counters it read first.
        if not conn.autocommit or conn.info.transaction_status != TransactionStatus.IDLE:
            raise ValueError(f"the connection to {db} must be idle, in autocommit mode")


def _reset(old: DatabaseFingerprint, new: DatabaseFingerprint) -> str | None:
    """Why the write counters of ``old`` and ``new`` cannot be compared, if they cannot."""
    if old.stats_reset != new.stats_reset:
        return f"were reset at {new.stats_reset}" if new.stats_reset else "were reset"
    # A counter only grows until it is reset, and an unclean restart resets every one of them
    # without recording when.
    went_back = any(
        now < then
        for table, counters in old.writes.items()
        if table in new.writes
        for then, now in zip(counters, new.writes[table], strict=True)
    )
    return "went backwards, as after an unclean restart" if went_back else None


def _differences[T](
    db: str,
    kind: str,
    old: Mapping[str, T],
    new: Mapping[str, T],
    describe: Callable[[T | None, T | None], str],
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
