import pytest

from goaly.bench.effisql.pg_fingerprint import (
    DatabaseFingerprint,
    Fingerprint,
    compute_fingerprint,
    fingerprint_changes,
)
from tests.bench.effisql.pg_fakes import Answer, AsyncFakeConnection

pytestmark = pytest.mark.anyio


def _db(**changes: object) -> DatabaseFingerprint:
    values: dict[str, object] = {
        "indexes": {"lineitem_pkey": "CREATE UNIQUE INDEX lineitem_pkey ON public.lineitem"},
        "relpages": {"lineitem": 100, "lineitem_pkey": 40},
        "writes": {"lineitem": (60175, 0, 0)},
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
    after = _db(relpages={"lineitem": 101, "lineitem_pkey": 40}, writes={"lineitem": (60176, 0, 1)})

    assert fingerprint_changes(_fingerprint(), _fingerprint(tpch=after)) == [
        "tpch: relation lineitem has 101 pages, not 100",
        "tpch: table lineitem has writes (inserted, updated, deleted) (60176, 0, 1), "
        "not (60175, 0, 0)",
    ]


def test_a_missing_database_is_named() -> None:
    after = Fingerprint(databases={"tpch": _db()})

    assert fingerprint_changes(_fingerprint(), after) == ["tpch_alt: database missing"]


async def test_compute_reads_indexes_pages_and_write_counters_of_each_database() -> None:
    answers: list[tuple[str, Answer]] = [
        ("FROM pg_indexes", [("lineitem_pkey", "CREATE UNIQUE INDEX lineitem_pkey ON x")]),
        ("relpages", [("lineitem", 100), ("lineitem_pkey", 40)]),
        ("n_tup_ins", [("lineitem", 60175, 0, 0)]),
    ]
    tpch = AsyncFakeConnection("tpch", answers)
    alt = AsyncFakeConnection("tpch_alt", answers)

    fingerprint = await compute_fingerprint({"tpch": tpch, "tpch_alt": alt})  # type: ignore[dict-item]

    expected = DatabaseFingerprint(
        indexes={"lineitem_pkey": "CREATE UNIQUE INDEX lineitem_pkey ON x"},
        relpages={"lineitem": 100, "lineitem_pkey": 40},
        writes={"lineitem": (60175, 0, 0)},
    )
    assert fingerprint == Fingerprint(databases={"tpch": expected, "tpch_alt": expected})
    assert len(tpch.statements) == 3
