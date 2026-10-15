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

from goaly.bench.effisql.pg_build import read_golden_record
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

    result = await env.execute("SELECT * FROM lineitem", timeout_s=120)

    assert result.error == ""
    assert result.row_count == record.databases["tpch"].tables["lineitem"].rows
    assert len(result.rows) == 100
    assert result.truncated
    assert list(result.columns)[:2] == ["l_orderkey", "l_partkey"]
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


async def test_the_result_is_the_last_statements(env: PgSqlEnvironment) -> None:
    result = await env.execute("SET work_mem = '8MB'; SHOW work_mem", timeout_s=10)

    assert _rows(result) == [["8MB"]]
    assert list(result.columns) == ["work_mem"]


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
    started = time.monotonic()
    result = await env.execute(f"SET statement_timeout = 0; {SLEEP}", timeout_s=1)

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
    pool = SolvingPool(writable_env.config, writable_env.credentials, None, _solving(writable_env))
    async with pool.environment() as env:
        await env.execute(
            "SET parallel_setup_cost = 0; SET parallel_tuple_cost = 0; "
            "SET min_parallel_table_scan_size = 0",
            timeout_s=10,
        )
        plan = _rows(await env.execute(f"EXPLAIN {query}", timeout_s=10))
        assert any("Gather" in str(line[0]) for line in plan)
        pid = _rows(await env.execute("SELECT pg_backend_pid()", timeout_s=10))[0][0]
        running = asyncio.create_task(env.execute(query, timeout_s=120))
        workers = 0
        with writable_env.connect(env.instance, ADMIN_ROLE, "postgres", autocommit=True) as admin:
            while not running.done() and workers == 0:
                row = admin.execute(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE leader_pid = %s AND backend_type = 'parallel worker'",
                    [int(pid)],  # type: ignore[call-overload]
                ).fetchone()
                workers = int(row[0]) if row else 0
                await asyncio.sleep(0.05)
        result = await running

    assert result.error == ""
    assert workers >= 1  # Streaming the rows left the query parallel.


async def test_a_dead_connection_raises_environment_failure(
    env: PgSqlEnvironment, pg_env: PgEnv
) -> None:
    pid = _rows(await env.execute("SELECT pg_backend_pid()", timeout_s=10))[0][0]
    with pg_env.connect(env.instance, ADMIN_ROLE, "postgres", autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(%s)", [int(pid)])  # type: ignore[call-overload]

    with pytest.raises(EnvironmentFailure, match="failed"):
        await env.execute("SELECT 1", timeout_s=10)


async def test_defaults_an_earlier_task_set_on_the_role_never_reach_the_next(
    writable_env: PgEnv,
) -> None:
    pool = SolvingPool(writable_env.config, writable_env.credentials, None, _solving(writable_env))
    async with pool.environment() as env:
        changed = await env.execute(
            "BEGIN READ WRITE; ALTER ROLE CURRENT_USER SET work_mem = '64MB'; COMMIT",
            timeout_s=10,
        )
        assert changed.error == ""

    async with pool.environment() as env:
        settings = await env.execute(
            "SELECT current_setting('work_mem'), current_setting('default_transaction_read_only')",
            timeout_s=10,
        )

    assert _rows(settings) == [["4MB", "on"]]
