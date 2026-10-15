"""The grader's session against the real grading instances.

Run with ``GOALY_EFFISQL_ITEST_REAL=1 pytest --integration``: these tests only read, so they run
on the running environment. Without the variable they skip.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from goaly.bench.effisql.grader import (
    DigestBuilder,
    PgSession,
    QueryError,
    QueryTimeoutError,
    speedup,
)
from goaly.bench.effisql.pg_settings import ALT_DB
from tests.bench.effisql.real_grading import GRADE_1_PORT, GRADE_2_PORT, real_session

pytestmark = pytest.mark.integration


@pytest.fixture
def session() -> Iterator[PgSession]:
    opened = real_session(GRADE_1_PORT)
    yield opened
    opened.close()


TYPED_ROW = """
SELECT 35098.38::numeric(15,2), 0.0700::numeric, 'NaN'::numeric, date '1997-02-09',
       1.5::float8, 0.1::float4, NULL::int, 'COLLECT COD'::char(14), 'it''s'::varchar,
       999939::bigint, true, interval '1 mon 2 days',
       timestamptz '2026-10-15 05:28:00+00',
       '6f9619ff-8b86-d011-b42d-00c04fc964ff'::uuid, row(1, 'a')
"""
EXPECTED_ROW = (
    Decimal("35098.38"),
    Decimal("0.0700"),
    Decimal("NaN"),
    date(1997, 2, 9),
    1.5,
    0.1,
    None,
    "COLLECT COD   ",
    "it's",
    999939,
    True,
    timedelta(days=32),
    datetime(2026, 10, 15, 5, 28, tzinfo=UTC),
    "6f9619ff-8b86-d011-b42d-00c04fc964ff",  # psycopg2 leaves uuid as text.
    "(1,a)",  # And an anonymous record.
)


def test_rows_load_as_psycopg2_builds_them(session: PgSession) -> None:
    row = session.connection.execute(TYPED_ROW).fetchone()
    assert row is not None
    assert str(row) == str(EXPECTED_ROW)
    builder = DigestBuilder()
    builder.add([EXPECTED_ROW])
    assert session.digest(TYPED_ROW) == builder.digest()


def test_a_tpch_result_digests_the_same_on_two_instances(session: PgSession) -> None:
    query = "SELECT * FROM lineitem WHERE l_orderkey < 2000 ORDER BY random()"
    other = real_session(GRADE_2_PORT)
    try:
        first, second = session.digest(query), other.digest(query)
    finally:
        other.close()
    assert first.row_count > 1000
    assert first.matches(second)


def test_the_alternate_database_gives_a_different_result() -> None:
    main, alt = real_session(GRADE_1_PORT), real_session(GRADE_1_PORT, ALT_DB)
    try:
        query = "SELECT count(*), sum(l_extendedprice) FROM lineitem"
        assert not main.digest(query).matches(alt.digest(query))
        same = "SELECT * FROM nation"
        assert main.digest(same).matches(alt.digest(same))
    finally:
        main.close()
        alt.close()


def test_a_row_limit_stops_the_fetch_early(session: PgSession) -> None:
    digest = session.digest("SELECT * FROM lineitem", row_limit=10)
    assert not digest.complete
    assert session.digest("SELECT 1").row_count == 1  # The session is still usable.


def test_a_slow_statement_is_cancelled_and_the_session_survives() -> None:
    slow = real_session(GRADE_1_PORT, timeout_s=0.5)
    try:
        with pytest.raises(QueryTimeoutError):
            slow.digest("SELECT pg_sleep(5)")
        with pytest.raises(QueryTimeoutError):
            slow.execution_ms("SELECT pg_sleep(5)")
        assert slow.digest("SELECT 1").row_count == 1
    finally:
        slow.close()


def test_a_query_cannot_lift_its_own_timeout() -> None:
    slow = real_session(GRADE_1_PORT, timeout_s=0.5)
    try:
        with pytest.raises(QueryTimeoutError):
            slow.digest("SELECT set_config('statement_timeout', '0', true), pg_sleep(5)")
    finally:
        slow.close()


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TEMP TABLE t AS SELECT 1",
        "WITH d AS (DELETE FROM region RETURNING *) SELECT * FROM d",
        "SELECT 1; SELECT 2",
    ],
)
def test_writes_and_second_statements_fail(session: PgSession, statement: str) -> None:
    with pytest.raises(QueryError):
        session.digest(statement)
    with pytest.raises(QueryError):
        session.execution_ms(statement)
    assert session.digest("SELECT count(*) FROM region").row_count == 1


def test_execution_time_and_plan(session: PgSession) -> None:
    query = "SELECT count(*) FROM orders WHERE o_orderdate < date '1993-01-01';"
    first, second = session.execution_ms(query), session.execution_ms(query)
    assert first > 0
    assert second > 0
    assert 0.2 < speedup(first, second) < 5
    plan = session.plan(query)
    assert "Execution Time" not in plan
    assert plan["Plan"]["Node Type"] in {"Aggregate", "Finalize Aggregate"}
