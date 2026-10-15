"""The agent's SQL environment against real PostgreSQL instances.

Run with ``pytest --integration``. Tests that load an instance for long, or change the agent
role's defaults, run only on the throwaway test environment.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any

import pytest
from google.protobuf.struct_pb2 import Value
from pydantic import SecretStr

from goaly.bench.effisql.pg_build import connector, read_golden_record
from goaly.bench.effisql.pg_instances import apply_roles
from goaly.bench.effisql.pg_settings import ADMIN_ROLE
from goaly.bench.effisql.sql_env import (
    STATEMENT_TIMEOUT,
    PgSqlEnvironment,
    SolvingPool,
    SqlEnvOptions,
)
from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.environment_pb2 import SqlResult
from tests.bench.effisql.conftest import PgEnv

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

SLEEP = "SELECT pg_sleep(30)"


def _solving(env: PgEnv) -> list[Any]:
    return [instance for instance in env.instances if instance.kind == "solving"]


@pytest.fixture
def pool(pg_env: PgEnv) -> SolvingPool:
    options = SqlEnvOptions(cancel_grace_s=2.0)
    return SolvingPool(pg_env.config, pg_env.credentials, options, _solving(pg_env))


@pytest.fixture
async def env(pool: SolvingPool) -> AsyncIterator[PgSqlEnvironment]:
    async with pool.environment() as environment:
        yield environment


def _json(value: Value) -> object:
    kind = value.WhichOneof("kind")
    return None if kind is None or kind == "null_value" else getattr(value, kind)


def _pid(value: object) -> int:
    assert isinstance(value, float)  # A number in a SqlResult row.
    return int(value)


def _rows(result: SqlResult) -> list[list[object]]:
    return [[_json(value) for value in row.values] for row in result.rows]


def _active(pg_env: PgEnv, instance: Any, text: str) -> int:
    """How many agent statements containing ``text`` are running on ``instance``."""
    with pg_env.connect(instance, ADMIN_ROLE, "postgres", autocommit=True) as admin:
        row = admin.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE usename = 'effisql_agent' "
            "AND state = 'active' AND query LIKE %s",
            [f"%{text}%"],
        ).fetchone()
    return int(row[0]) if row else -1


async def test_every_row_is_counted_but_only_the_first_reach_the_agent(
    env: PgSqlEnvironment, pg_env: PgEnv
) -> None:
    record = read_golden_record(pg_env.config.golden_record)
    assert record is not None

    result = await env.execute("SELECT * FROM supplier", timeout_s=120)  # Light, yet > 100.

    assert result.error == ""
    assert result.row_count == record.databases["tpch"].tables["supplier"].rows
    assert len(result.rows) == 100
    assert result.truncated
    assert list(result.columns)[:2] == ["s_suppkey", "s_name"]
    assert result.elapsed_s > 0
    assert result.queue_wait_s >= 0


async def test_values_are_encoded_as_the_contract_says(env: PgSqlEnvironment) -> None:
    result = await env.execute(
        "SELECT NULL::int, 9007199254740993::bigint, 12.50::numeric, 'NaN'::float8, "
        "'-Infinity'::float8, 3.25::float8, 42, true, DATE '1996-03-13', 'NULL'",
        timeout_s=10,
    )

    assert _rows(result) == [
        [
            None,
            "9007199254740993",
            "12.50",
            "NaN",
            "-Infinity",
            3.25,
            42.0,
            True,
            "1996-03-13",
            "NULL",
        ]
    ]
    assert (result.row_count, result.truncated) == (1, False)


async def test_each_call_runs_exactly_one_statement(env: PgSqlEnvironment) -> None:
    result = await env.execute("SELECT 1; SELECT 2", timeout_s=10)

    assert "cannot insert multiple commands" in result.error
    assert _rows(await env.execute("SELECT 2;", timeout_s=10)) == [[2.0]]  # One, with a ';'.


# Each attempt, and what the next statement in the same session must then see.
ATTEMPTS = [
    ("ALTER ROLE effisql_agent PASSWORD 'stolen'", "SELECT 1", [[1.0]]),
    ("SET ROLE effisql_grader", "SELECT current_user::text", [["effisql_agent"]]),
    (
        "SET SESSION AUTHORIZATION effisql_grader",
        "SELECT session_user::text, current_user::text",
        [["effisql_agent", "effisql_agent"]],
    ),
    ("SET TRANSACTION READ WRITE", "INSERT INTO region VALUES (99, 'x', 'x')", "read-only"),
    ("BEGIN READ WRITE", "INSERT INTO region VALUES (99, 'x', 'x')", "read-only"),
    (
        "SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE",
        "SHOW default_transaction_read_only",
        [["on"]],
    ),
    ("SET work_mem = '64MB'", "SHOW work_mem", [["4MB"]]),
    (
        "CREATE TEMP TABLE itest_tmp (x int)",
        "SELECT to_regclass('pg_temp.itest_tmp')::text",
        [[None]],
    ),
    (
        "SELECT pg_advisory_lock(424242)",
        "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()",
        [[0.0]],
    ),
    ("PREPARE itest_p AS SELECT 1", "EXECUTE itest_p", "does not exist"),
    ("DECLARE itest_c CURSOR WITH HOLD FOR SELECT 1", "FETCH itest_c", "does not exist"),
    ("LISTEN itest_channel", "SELECT count(*) FROM pg_listening_channels()", [[0.0]]),
    ("COMMIT", "SELECT 1", [[1.0]]),
]


@pytest.mark.parametrize(("attempt", "check", "expected"), ATTEMPTS, ids=[a[0] for a in ATTEMPTS])
async def test_nothing_a_statement_does_reaches_the_next(
    env: PgSqlEnvironment, attempt: str, check: str, expected: object
) -> None:
    await env.execute(attempt, timeout_s=10)  # Refused, or undone once the call ends.

    result = await env.execute(check, timeout_s=10)

    if isinstance(expected, str):
        assert expected in result.error
    else:
        assert result.error == ""
        assert _rows(result) == expected


async def test_the_agent_password_survives_an_attempt_to_change_it(pool: SolvingPool) -> None:
    async with pool.environment() as env:
        await env.execute("ALTER ROLE effisql_agent PASSWORD 'stolen'", timeout_s=10)
    async with pool.environment() as env:  # A new session logs in with the real password.
        assert _rows(await env.execute("SELECT 1", timeout_s=10)) == [[1.0]]
    await pool.preflight()


async def test_a_task_cannot_read_another_tasks_statements(pool: SolvingPool) -> None:
    async with pool.environment() as watcher:
        activity = await watcher.execute("SELECT query FROM pg_stat_activity", timeout_s=10)
        backend = await watcher.execute("SELECT pg_stat_get_backend_activity(1)", timeout_s=10)
        member = await watcher.execute(
            "SELECT pg_has_role(current_user, 'pg_read_all_stats', 'MEMBER')", timeout_s=10
        )

    assert "permission denied" in activity.error
    assert "permission denied" in backend.error
    assert _rows(member) == [[False]]


async def test_preflight_passes_and_names_a_login_that_fails(pg_env: PgEnv) -> None:
    await SolvingPool(pg_env.config, pg_env.credentials, None, _solving(pg_env)).preflight()
    wrong = pg_env.credentials.model_copy(update={"agent_password": SecretStr("wrong")})

    with pytest.raises(EnvironmentFailure, match="agent role cannot log in"):
        await SolvingPool(pg_env.config, wrong, None, _solving(pg_env)).preflight()


async def test_preflight_names_an_agent_role_that_is_not_read_only(writable_env: PgEnv) -> None:
    instance = _solving(writable_env)[0]
    admin = connector(
        writable_env.config.host, instance.port, writable_env.credentials.admin_password
    )
    with admin("postgres") as conn:
        conn.execute("ALTER ROLE effisql_agent RESET default_transaction_read_only")
    try:
        with pytest.raises(EnvironmentFailure, match="not read-only by default"):
            await SolvingPool(
                writable_env.config, writable_env.credentials, None, [instance]
            ).preflight()
    finally:
        apply_roles(admin, writable_env.credentials)


async def test_a_failing_statement_is_a_result_and_the_session_lives_on(
    env: PgSqlEnvironment,
) -> None:
    failed = await env.execute("SELECT 1 / 0", timeout_s=10)
    refused = await env.execute("INSERT INTO region VALUES (99, 'x', 'x')", timeout_s=10)
    copied = await env.execute("COPY region TO STDOUT", timeout_s=10)

    assert "division by zero" in failed.error
    assert "read-only transaction" in refused.error
    assert "COPY to the client is not supported" in copied.error
    assert _rows(await env.execute("SELECT 1", timeout_s=10)) == [[1.0]]


async def test_a_long_query_is_stopped_from_the_client_even_with_statement_timeout_off(
    env: PgSqlEnvironment, pg_env: PgEnv
) -> None:
    await env.execute("SET statement_timeout = 0", timeout_s=1)  # The first bypass attempt.
    started = time.monotonic()
    result = await env.execute(SLEEP, timeout_s=1)

    assert result.error == STATEMENT_TIMEOUT
    assert 1 <= result.elapsed_s < 5
    assert time.monotonic() - started < 6
    assert _active(pg_env, env.instance, "pg_sleep(30)") == 0
    assert _rows(await env.execute("SELECT 2", timeout_s=10)) == [[2.0]]


async def test_a_cancelled_execute_stops_its_backend_statement(
    env: PgSqlEnvironment, pg_env: PgEnv
) -> None:
    task = asyncio.create_task(env.execute(SLEEP, timeout_s=120))
    await asyncio.sleep(0.5)
    assert _active(pg_env, env.instance, "pg_sleep(30)") == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _active(pg_env, env.instance, "pg_sleep(30)") == 0  # Stopped before the raise.
    assert _rows(await env.execute("SELECT 3", timeout_s=10)) == [[3.0]]


async def test_the_statement_cap_holds_and_queue_wait_is_reported_apart(
    pg_env: PgEnv,
) -> None:
    instance = _solving(pg_env)[0]
    pool = SolvingPool(pg_env.config, pg_env.credentials, SqlEnvOptions(), [instance])
    async with AsyncExitStack() as stack:
        envs = [await stack.enter_async_context(pool.environment()) for _ in range(5)]
        assert pool.load() == {instance.name: 5}
        running = asyncio.gather(*(e.execute("SELECT pg_sleep(1)", timeout_s=30) for e in envs))
        peak = 0
        while not running.done():
            peak = max(peak, _active(pg_env, instance, "pg_sleep(1)"))
            await asyncio.sleep(0.1)
        results = await running

    assert peak == 2
    assert all(r.error == "" and 0.9 <= r.elapsed_s < 1.8 for r in results)
    waits = sorted(r.queue_wait_s for r in results)
    assert waits[1] < 0.3  # Two start at once,
    assert waits[2] >= 0.8  # two wait for the first pair,
    assert waits[4] >= 1.8  # and the last waits for the second pair.


async def test_least_loaded_assignment_spreads_tasks_over_the_solving_instances(
    pool: SolvingPool,
) -> None:
    async with pool.environment() as first, pool.environment() as second:
        assert first.instance != second.instance
        assert set(pool.load().values()) == {1}
    assert set(pool.load().values()) == {0}


async def test_a_parallel_plan_stays_parallel(writable_env: PgEnv) -> None:
    query = "SELECT count(*) FROM lineitem WHERE md5(repeat(l_comment, 1000)) LIKE 'ff%'"
    instance = _solving(writable_env)[0]
    admin = connector(
        writable_env.config.host, instance.port, writable_env.credentials.admin_password
    )
    with admin("postgres") as conn:  # The tiny tables are below the parallel thresholds.
        for setting in (
            "parallel_setup_cost",
            "parallel_tuple_cost",
            "min_parallel_table_scan_size",
        ):
            conn.execute(f"ALTER DATABASE tpch SET {setting} = 0")
    try:
        pool = SolvingPool(writable_env.config, writable_env.credentials, None, [instance])
        async with pool.environment() as env:
            plan = _rows(await env.execute(f"EXPLAIN {query}", timeout_s=10))
            assert any("Gather" in str(line[0]) for line in plan)
            pid = _rows(await env.execute("SELECT pg_backend_pid()", timeout_s=10))[0][0]
            running = asyncio.create_task(env.execute(query, timeout_s=120))
            workers = 0
            with writable_env.connect(instance, ADMIN_ROLE, "postgres", autocommit=True) as watch:
                while not running.done() and workers == 0:
                    row = watch.execute(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE leader_pid = %s AND backend_type = 'parallel worker'",
                        [_pid(pid)],
                    ).fetchone()
                    workers = int(row[0]) if row else 0
                    await asyncio.sleep(0.05)
            result = await running
    finally:
        with admin("postgres") as conn:
            conn.execute("ALTER DATABASE tpch RESET ALL")

    assert result.error == ""
    assert workers >= 1  # Streaming the rows left the query parallel.


async def test_a_dead_connection_raises_environment_failure(
    env: PgSqlEnvironment, pg_env: PgEnv
) -> None:
    pid = _rows(await env.execute("SELECT pg_backend_pid()", timeout_s=10))[0][0]
    with pg_env.connect(env.instance, ADMIN_ROLE, "postgres", autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(%s)", [_pid(pid)])

    with pytest.raises(EnvironmentFailure, match="failed"):
        await env.execute("SELECT 1", timeout_s=10)


async def test_defaults_set_on_the_agent_role_never_reach_a_session(
    writable_env: PgEnv,
) -> None:
    instance = _solving(writable_env)[0]
    admin = connector(
        writable_env.config.host, instance.port, writable_env.credentials.admin_password
    )
    with admin("postgres") as conn:
        conn.execute("ALTER ROLE effisql_agent SET work_mem = '64MB'")
    pool = SolvingPool(writable_env.config, writable_env.credentials, None, [instance])
    async with pool.environment() as env:
        settings = await env.execute(
            "SELECT current_setting('work_mem'), current_setting('default_transaction_read_only')",
            timeout_s=10,
        )

    assert _rows(settings) == [["4MB", "on"]]
