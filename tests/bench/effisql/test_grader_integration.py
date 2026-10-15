"""The grader's session against the real grading instances.

Run with ``GOALY_EFFISQL_ITEST_REAL=1 pytest --integration``: these tests only read, so they run
on the running environment. Without the variable they skip.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
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
       '6f9619ff-8b86-d011-b42d-00c04fc964ff'::uuid, row(1, 'a'),
       '1.2.3.4'::inet, '10.0.0.0/8'::cidr, time '12:34:56', '{"a": [1, 2.5]}'::json,
       ARRAY[1, 2], ARRAY['6f9619ff-8b86-d011-b42d-00c04fc964ff'::uuid], ARRAY['1.2.3.4'::inet]
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
    "1.2.3.4",
    "10.0.0.0/8",
    time(12, 34, 56),
    {"a": [1, 2.5]},
    [1, 2],
    "{6f9619ff-8b86-d011-b42d-00c04fc964ff}",  # psycopg2 leaves arrays of these as text.
    "{1.2.3.4}",
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


def test_the_servers_statement_timeout_stops_a_slow_statement() -> None:
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
    slow = real_session(GRADE_1_PORT, timeout_s=0.5, hard_cap_s=3.0)
    try:
        with pytest.raises(QueryTimeoutError):
            slow.digest("SELECT set_config('statement_timeout', '0', false), pg_sleep(10)")
        assert slow.connection.execute("SHOW statement_timeout").fetchone() == ("500ms",)
    finally:
        slow.close()


def scalar(session: PgSession, query: str) -> object:
    row = session.connection.execute(query).fetchone()
    return None if row is None else row[0]


def test_nothing_a_prediction_does_outlives_its_statement(session: PgSession) -> None:
    session.digest("SELECT pg_advisory_lock(42)")
    with pytest.raises(QueryError):  # The server runs it; it returns no rows to stream.
        session.digest("PREPARE leftover AS SELECT 1")
    with pytest.raises(QueryError):
        session.digest("LISTEN leftover")
    with pytest.raises(QueryError):
        session.digest("CREATE TEMP TABLE leftover AS SELECT 1")
    session.digest("SELECT set_config('work_mem', '777MB', false)")
    with pytest.raises(QueryError):
        session.digest("SET work_mem = '777MB'")
    advisory = (
        "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
    )
    assert scalar(session, advisory) == 0
    assert scalar(session, "SELECT count(*) FROM pg_prepared_statements") == 0
    assert scalar(session, "SELECT count(*) FROM pg_listening_channels()") == 0
    assert scalar(session, "SELECT count(*) FROM pg_class WHERE relname = 'leftover'") == 0
    assert scalar(session, "SHOW work_mem") != "777MB"
    assert scalar(session, "SHOW statement_timeout") == "5min"
    assert session.digest("SELECT 1").row_count == 1


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
    assert 0.05 < speedup(first, second) < 20  # Loose: the host may be busy.
    plan = session.plan(query)
    assert "Execution Time" not in plan
    assert plan["Plan"]["Node Type"] in {"Aggregate", "Finalize Aggregate"}
