from typing import Any

import pytest
from psycopg.pq import TransactionStatus
from pydantic import ValidationError

from goaly.bench.effisql.pg_fingerprint import (
    FINGERPRINT_QUERIES,
    DatabaseFingerprint,
    Fingerprint,
    compute_fingerprint,
    compute_fingerprint_sync,
    fingerprint_changes,
    fingerprint_from_rows,
)
from tests.bench.effisql.pg_fakes import Answer, AsyncFakeConnection, FakeConnection

pytestmark = pytest.mark.anyio


def _db(**changes: object) -> DatabaseFingerprint:
    values: dict[str, object] = {
        "indexes": {"lineitem_pkey": "CREATE UNIQUE INDEX lineitem_pkey ON public.lineitem"},
        "relpages": {"lineitem": 100, "lineitem_pkey": 40},
        "files": {"lineitem": 16390, "lineitem_pkey": 16400},
        "writes": {"lineitem": (60175, 7, 3)},
    }
    values.update(changes)
    return DatabaseFingerprint.model_validate(values)


def _fingerprint(**databases: DatabaseFingerprint) -> Fingerprint:
    return Fingerprint(databases={"tpch": _db(), "tpch_alt": _db(), **databases})


def test_an_unchanged_catalog_has_no_changes() -> None:
    assert fingerprint_changes(_fingerprint(), _fingerprint()) == []


def test_an_added_index_and_its_relation_are_named() -> None:
    after = _db(
        indexes={**_db().indexes, "x_idx": "CREATE INDEX x_idx ON public.lineitem (l_tax)"},
        relpages={**_db().relpages, "x_idx": 9},
        files={**_db().files, "x_idx": 16500},
    )

    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch=after)) == [
        "tpch: index x_idx was added: CREATE INDEX x_idx ON public.lineitem (l_tax)",
        "tpch: relation x_idx appeared, 9 pages",
    ]


def test_a_dropped_or_redefined_index_is_named() -> None:
    redefined = _db(indexes={"lineitem_pkey": "CREATE INDEX lineitem_pkey ON public.lineitem"})
    dropped = _db(indexes={})

    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch=redefined)) == [
        "tpch: index lineitem_pkey was redefined: CREATE INDEX lineitem_pkey ON public.lineitem"
    ]
    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch_alt=dropped)) == [
        "tpch_alt: index lineitem_pkey was dropped"
    ]


def test_grown_relations_and_writes_are_named_with_both_values() -> None:
    after = _db(relpages={"lineitem": 101, "lineitem_pkey": 40}, writes={"lineitem": (60176, 7, 4)})

    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch=after)) == [
        "tpch: relation lineitem has 101 pages, not 100",
        "tpch: table lineitem has writes (inserted, updated, deleted) (60176, 7, 4), "
        "not (60175, 7, 3)",
    ]


def test_each_write_counter_is_compared_in_its_own_place() -> None:
    swapped = _db(writes={"lineitem": (60175, 3, 7)})  # Updated and deleted swapped.
    updated = _db(writes={"lineitem": (60175, 8, 3)})

    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch=updated)) == [
        "tpch: table lineitem has writes (inserted, updated, deleted) (60175, 8, 3), "
        "not (60175, 7, 3)"
    ]
    # 3 < 7 in the updated place: a counter went down, which only a reset does.
    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch=swapped)) == [
        "tpch: the write counters went backwards, as after an unclean restart, "
        "so they cannot show writes"
    ]


def test_a_rewritten_relation_is_named_even_when_its_size_is_the_same() -> None:
    after = _db(files={"lineitem": 16600, "lineitem_pkey": 16400})

    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch=after)) == [
        "tpch: relation lineitem was rewritten (file 16600, not 16390)"
    ]


def test_a_dropped_table_and_a_new_one_are_named() -> None:
    dropped = _db(relpages={"lineitem_pkey": 40}, files={"lineitem_pkey": 16400}, writes={})
    added = _db(
        relpages={**_db().relpages, "extra": 0},
        files={**_db().files, "extra": 16700},
        writes={**_db().writes, "extra": (0, 0, 1)},
    )

    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch=dropped)) == [
        "tpch: relation lineitem disappeared",
        "tpch: table lineitem lost its write counters",
    ]
    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch=added)) == [
        "tpch: relation extra appeared, 0 pages",
        "tpch: table extra appeared, with writes (0, 0, 1)",
    ]


def test_reset_counters_give_one_line_instead_of_one_per_table() -> None:
    reset = _db(writes={"lineitem": (0, 0, 0)}, stats_reset="2026-10-15 09:00:00+00")
    crashed = _db(writes={"lineitem": (0, 0, 0)})

    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch=reset)) == [
        "tpch: the write counters were reset at 2026-10-15 09:00:00+00, so they cannot show writes"
    ]
    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch_alt=crashed)) == [
        "tpch_alt: the write counters went backwards, as after an unclean restart, "
        "so they cannot show writes"
    ]
    # A reset time that went away is a reset too.
    before = _fingerprint(tpch=reset)
    assert fingerprint_changes(before, _fingerprint(tpch=crashed)) == [
        "tpch: the write counters were reset, so they cannot show writes"
    ]


def test_a_database_missing_from_either_side_is_named() -> None:
    one = Fingerprint(databases={"tpch": _db()})

    assert fingerprint_changes(_fingerprint(), one) == ["tpch_alt: database missing"]
    assert fingerprint_changes(one, _fingerprint()) == ["tpch_alt: database appeared"]


def test_a_fingerprint_without_databases_is_refused() -> None:
    with pytest.raises(ValidationError):
        Fingerprint(databases={})


_ROWS: dict[str, list[tuple[Any, ...]]] = {
    "indexes": [("lineitem_pkey", "CREATE UNIQUE INDEX lineitem_pkey ON x")],
    "relations": [("lineitem", 100, 16390), ("lineitem_pkey", 40, 16400)],
    "writes": [("lineitem", 60175, 7, 3)],
    "stats_reset": [(None,)],
}
_EXPECTED = DatabaseFingerprint(
    indexes={"lineitem_pkey": "CREATE UNIQUE INDEX lineitem_pkey ON x"},
    relpages={"lineitem": 100, "lineitem_pkey": 40},
    files={"lineitem": 16390, "lineitem_pkey": 16400},
    writes={"lineitem": (60175, 7, 3)},
)


def _answers() -> list[tuple[str, Answer]]:
    """Each fingerprint query, answered with its rows in _ROWS."""
    return [(FINGERPRINT_QUERIES[key], rows) for key, rows in _ROWS.items()]


def test_the_rows_of_each_query_build_a_database_fingerprint() -> None:
    assert fingerprint_from_rows(_ROWS) == _EXPECTED
    reset = fingerprint_from_rows({**_ROWS, "stats_reset": [("2026-10-15 09:00:00+00",)]})
    assert reset.stats_reset == "2026-10-15 09:00:00+00"
    assert fingerprint_from_rows({**_ROWS, "stats_reset": []}).stats_reset is None


async def test_compute_reads_every_query_of_each_database() -> None:
    tpch = AsyncFakeConnection("tpch", _answers())
    alt = AsyncFakeConnection("tpch_alt", _answers())

    fingerprint = await compute_fingerprint({"tpch": tpch, "tpch_alt": alt})  # type: ignore[dict-item]

    assert fingerprint == Fingerprint(databases={"tpch": _EXPECTED, "tpch_alt": _EXPECTED})
    assert tpch.statements == list(FINGERPRINT_QUERIES.values())


def test_the_sync_compute_reads_the_same_fingerprint() -> None:
    tpch = FakeConnection("tpch", _answers())
    alt = FakeConnection("tpch_alt", _answers())

    fingerprint = compute_fingerprint_sync({"tpch": tpch, "tpch_alt": alt})  # type: ignore[dict-item]

    assert fingerprint == Fingerprint(databases={"tpch": _EXPECTED, "tpch_alt": _EXPECTED})
    assert alt.statements == list(FINGERPRINT_QUERIES.values())


@pytest.mark.parametrize(
    ("conns", "message"),
    [
        ({}, "none was given"),
        ({"tpch": FakeConnection("tpch_alt")}, "the connection given for tpch is to tpch_alt"),
        ({"tpch": FakeConnection("tpch", autocommit=False)}, "must be idle, in autocommit mode"),
        (
            {"tpch": FakeConnection("tpch", transaction_status=TransactionStatus.INTRANS)},
            "must be idle, in autocommit mode",
        ),
    ],
)
def test_compute_refuses_connections_it_cannot_trust(
    conns: dict[str, FakeConnection], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_fingerprint_sync(conns)  # type: ignore[arg-type]
    for conn in conns.values():
        assert conn.statements == []


async def test_the_async_compute_refuses_a_connection_in_a_transaction() -> None:
    conn = AsyncFakeConnection("tpch", transaction_status=TransactionStatus.INTRANS)

    with pytest.raises(ValueError, match="must be idle, in autocommit mode"):
        await compute_fingerprint({"tpch": conn})  # type: ignore[dict-item]
    assert conn.statements == []
