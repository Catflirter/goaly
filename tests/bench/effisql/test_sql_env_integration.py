"""The agent's SQL environment against real PostgreSQL instances.

Run with ``pytest --integration``. Tests that load an instance for long, or change the agent
role's defaults or rights, run only on the throwaway test environment.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import psycopg
import pytest
from google.protobuf.struct_pb2 import Value
from psycopg import errors
from pydantic import SecretStr

from goaly.bench.effisql.pg_build import connector, read_golden_record
from goaly.bench.effisql.pg_instances import apply_roles, socket_dirs
from goaly.bench.effisql.pg_settings import ADMIN_ROLE, AGENT_ROLE, GRADER_ROLE, AgentCredentials
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


def _pool(env: PgEnv, options: SqlEnvOptions | None = None, **kwargs: Any) -> SolvingPool:
    """A pool on the environment's solving instances, holding the agent's password alone."""
    instances = kwargs.pop("instances", _solving(env))
    return SolvingPool(env.config, env.credentials.agent_only(), options, instances, **kwargs)


@pytest.fixture
def pool(pg_env: PgEnv) -> SolvingPool:
    return _pool(pg_env, SqlEnvOptions(cancel_grace_s=2.0))


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


async def _backend(env: PgSqlEnvironment) -> int:
    return _pid(_rows(await env.execute("SELECT pg_backend_pid()", timeout_s=10))[0][0])


@asynccontextmanager
async def _watching(pg_env: PgEnv, instance: Any) -> AsyncIterator[Callable[[str], Awaitable[int]]]:
    """Count the agent statements running on ``instance`` that contain a text, over one async
    admin session, so watching never blocks the event loop the statements run on."""
    async with await psycopg.AsyncConnection.connect(
        host=pg_env.config.host,
        port=instance.port,
        user=ADMIN_ROLE,
        password=pg_env.credentials.admin_password.get_secret_value(),
        dbname="postgres",
        autocommit=True,
    ) as admin:

        async def active(text: str) -> int:
            cursor = await admin.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE usename = 'effisql_agent' "
                "AND state = 'active' AND query LIKE %s",
                [f"%{text}%"],
            )
            row = await cursor.fetchone()
            return int(row[0]) if row else -1

        yield active


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


async def test_long_values_and_large_results_are_cut_but_counted(env: PgSqlEnvironment) -> None:
    long_value = await env.execute("SELECT repeat('x', 10000)", timeout_s=10)
    wide = await env.execute(
        "SELECT repeat('x', 5000), repeat('y', 5000), repeat('z', 5000) "
        "FROM generate_series(1, 500)",
        timeout_s=30,
    )

    assert _rows(long_value) == [["x" * 4096 + "[... 5904 more bytes]"]]
    # Each row keeps 3 x 4 KiB: the 1 MiB of a result holds 86 of them, of the 500 counted.
    assert (len(wide.rows), wide.row_count, wide.truncated) == (86, 500, True)


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
    ("SET temp_file_limit = -1", "SHOW temp_file_limit", [["5GB"]]),
    ("SET statement_timeout = 0", "SHOW statement_timeout", [["130s"]]),  # The backstop holds.
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
    await _pool(pg_env).preflight()
    wrong = AgentCredentials(agent_password=SecretStr("wrong"), _env_file=None)
    refused = SolvingPool(pg_env.config, wrong, None, _solving(pg_env))

    with pytest.raises(EnvironmentFailure, match="agent role cannot log in"):
        await refused.preflight()
    with pytest.raises(EnvironmentFailure, match="cannot open a session"):
        async with refused.environment():
            pass  # Only an instance that refuses a session fails the environment.


async def test_preflight_names_an_agent_role_that_is_not_read_only(writable_env: PgEnv) -> None:
    instance = _solving(writable_env)[0]
    admin = connector(
        writable_env.config.host, instance.port, writable_env.credentials.admin_password
    )
    with admin("postgres") as conn:
        conn.execute("ALTER ROLE effisql_agent RESET default_transaction_read_only")
        conn.execute("ALTER ROLE effisql_agent RESET temp_file_limit")
    with admin("tpch") as conn:
        conn.execute(
            "REVOKE EXECUTE ON FUNCTION pg_terminate_backend(integer, bigint) FROM effisql_agent"
        )
    try:
        with pytest.raises(EnvironmentFailure) as caught:
            await _pool(writable_env, instances=[instance]).preflight()
    finally:
        apply_roles(admin, writable_env.credentials, instance.kind)

    message = str(caught.value)
    assert "default_transaction_read_only is off, not on" in message
    assert "temp_file_limit is -1, not 5GB" in message
    assert "cannot end its own sessions" in message


async def test_a_failing_statement_is_a_result_and_the_session_lives_on(
    env: PgSqlEnvironment,
) -> None:
    failed = await env.execute("SELECT 1 / 0", timeout_s=10)
    refused = await env.execute("INSERT INTO region VALUES (99, 'x', 'x')", timeout_s=10)
    copied = await env.execute("COPY region TO STDOUT", timeout_s=10)
    # A COPY that fails partway, after its first rows reached the client.
    broken_copy = await env.execute(
        "COPY (SELECT 1 / (5 - g) FROM generate_series(1, 10) AS g) TO STDOUT", timeout_s=10
    )
    copied_in = await env.execute("COPY region FROM STDIN", timeout_s=10)
    unencodable = await env.execute("SELECT '\ud800'", timeout_s=10)
    do_block = await env.execute("DO $$ BEGIN PERFORM 1; END $$", timeout_s=10)

    assert "division by zero" in failed.error
    assert "read-only transaction" in refused.error
    assert "COPY to the client is not supported" in copied.error
    assert "division by zero" in broken_copy.error
    assert copied_in.error != ""
    assert "not valid UTF-8" in unencodable.error
    assert "permission denied for language plpgsql" in do_block.error
    assert _rows(await env.execute("SELECT 1", timeout_s=10)) == [[1.0]]


async def test_the_timeout_message_is_postgresqls_own(pg_env: PgEnv) -> None:
    instance = _solving(pg_env)[0]
    with pg_env.connect(instance, AGENT_ROLE, "tpch", autocommit=True) as conn:
        conn.execute("SET statement_timeout = 100")
        with pytest.raises(errors.QueryCanceled) as caught:
            conn.execute("SELECT pg_sleep(5)")

    assert caught.value.diag.message_primary == STATEMENT_TIMEOUT


async def test_a_long_query_is_stopped_from_the_client_even_with_statement_timeout_off(
    env: PgSqlEnvironment, pg_env: PgEnv
) -> None:
    started = time.monotonic()
    # The bypass attempt: the statement turns its own timeout off for the queries it runs.
    result = await env.execute(
        "SELECT set_config('statement_timeout', '0', false), pg_sleep(30)", timeout_s=1
    )

    assert result.error == STATEMENT_TIMEOUT
    assert 1 <= result.elapsed_s < 5
    assert time.monotonic() - started < 6
    async with _watching(pg_env, env.instance) as active:
        assert await active("pg_sleep(30)") == 0
    assert _rows(await env.execute("SELECT 2", timeout_s=10)) == [[2.0]]


async def test_a_statement_whose_cancel_goes_astray_is_ended_by_the_agent_role(
    pool: SolvingPool, pg_env: PgEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def lost(self: psycopg.AsyncConnection[Any], *, timeout: float = 30.0) -> None:
        return None  # The cancel request never reaches the server.

    monkeypatch.setattr(psycopg.AsyncConnection, "cancel_safe", lost)
    async with pool.environment() as env:
        first = await _backend(env)
        result = await env.execute(SLEEP, timeout_s=1)
        second = await _backend(env)
        async with _watching(pg_env, env.instance) as active:
            running = await active("pg_sleep(30)")

    assert result.error == STATEMENT_TIMEOUT
    assert second != first  # The agent role ended the old backend; a new session took over.
    assert running == 0


async def test_a_cancelled_execute_stops_its_backend_statement(
    env: PgSqlEnvironment, pg_env: PgEnv
) -> None:
    async with _watching(pg_env, env.instance) as active:
        task = asyncio.create_task(env.execute(SLEEP, timeout_s=120))
        await asyncio.sleep(0.5)
        assert await active("pg_sleep(30)") == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert await active("pg_sleep(30)") == 0  # Stopped before the raise.
    assert _rows(await env.execute("SELECT 3", timeout_s=10)) == [[3.0]]


async def test_the_statement_cap_holds_and_queue_wait_is_reported_apart(
    pg_env: PgEnv,
) -> None:
    instance = _solving(pg_env)[0]
    pool = _pool(pg_env, SqlEnvOptions(), instances=[instance])
    async with AsyncExitStack() as stack:
        envs = [await stack.enter_async_context(pool.environment()) for _ in range(5)]
        assert pool.load() == {instance.name: 5}
        active = await stack.enter_async_context(_watching(pg_env, instance))
        running = asyncio.gather(*(e.execute("SELECT pg_sleep(1)", timeout_s=30) for e in envs))
        peak = 0
        while not running.done():
            peak = max(peak, await active("pg_sleep(1)"))
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


async def test_the_pool_works_through_the_unix_socket_directories(pg_env: PgEnv) -> None:
    instances = _solving(pg_env)
    pool = _pool(pg_env, socket_dirs=socket_dirs(pg_env.config, instances))

    await pool.preflight()
    async with pool.environment() as env:
        result = await env.execute("SELECT inet_server_addr(), count(*) FROM region", timeout_s=10)

    assert _rows(result) == [[None, 5.0]]  # No IP address: the session is on the socket.


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
        pool = _pool(writable_env, instances=[instance])
        async with pool.environment() as env:
            plan = _rows(await env.execute(f"EXPLAIN {query}", timeout_s=10))
            assert any("Gather" in str(line[0]) for line in plan)
            pid = await _backend(env)
            running = asyncio.create_task(env.execute(query, timeout_s=120))
            workers = 0
            async with await psycopg.AsyncConnection.connect(
                host=writable_env.config.host,
                port=instance.port,
                user=ADMIN_ROLE,
                password=writable_env.credentials.admin_password.get_secret_value(),
                dbname="postgres",
                autocommit=True,
            ) as watch:
                while not running.done() and workers == 0:
                    cursor = await watch.execute(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE leader_pid = %s AND backend_type = 'parallel worker'",
                        [pid],
                    )
                    row = await cursor.fetchone()
                    workers = int(row[0]) if row else 0
                    await asyncio.sleep(0.05)
            result = await running
    finally:
        with admin("postgres") as conn:
            conn.execute("ALTER DATABASE tpch RESET ALL")

    assert result.error == ""
    assert workers >= 1  # Streaming the rows left the query parallel.


async def test_a_session_ended_from_outside_is_an_error_and_the_next_call_reconnects(
    env: PgSqlEnvironment, pg_env: PgEnv
) -> None:
    pid = await _backend(env)
    with pg_env.connect(env.instance, ADMIN_ROLE, "postgres", autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(%s)", [pid])

    ended = await env.execute("SELECT 1", timeout_s=10)

    assert ended.error.startswith("the session ended")
    assert _rows(await env.execute("SELECT 1", timeout_s=10)) == [[1.0]]
    assert await _backend(env) != pid


async def test_a_statement_that_ends_its_own_session_is_an_error_not_a_failure(
    env: PgSqlEnvironment,
) -> None:
    pid = await _backend(env)

    ended = await env.execute("SELECT pg_terminate_backend(pg_backend_pid())", timeout_s=10)

    assert ended.error.startswith("the session ended")
    assert await _backend(env) != pid  # The next call opened a new session.


async def test_one_agent_session_may_end_another_but_never_the_graders(pg_env: PgEnv) -> None:
    instance = _solving(pg_env)[0]
    one_instance = _pool(pg_env, instances=[instance])
    with pg_env.connect(instance, GRADER_ROLE, "tpch", autocommit=True) as grader:
        grader_pid = grader.info.backend_pid
        async with one_instance.environment() as first, one_instance.environment() as second:
            other = await _backend(second)
            ended = await first.execute(f"SELECT pg_terminate_backend({other})", timeout_s=10)
            refused = await first.execute(
                f"SELECT pg_terminate_backend({grader_pid})", timeout_s=10
            )
            assert (await second.execute("SELECT 1", timeout_s=10)).error.startswith(
                "the session ended"
            )
        assert grader.execute("SELECT 1").fetchone() == (1,)  # The grader's session lives.

    assert _rows(ended) == [[True]]  # The residual risk the module docstring names.
    assert "permission denied to terminate process" in refused.error


async def test_defaults_set_on_the_agent_role_never_reach_a_session(
    writable_env: PgEnv,
) -> None:
    instance = _solving(writable_env)[0]
    admin = connector(
        writable_env.config.host, instance.port, writable_env.credentials.admin_password
    )
    with admin("postgres") as conn:
        conn.execute("ALTER ROLE effisql_agent SET work_mem = '64MB'")
        conn.execute("ALTER ROLE effisql_agent RESET default_transaction_read_only")
        conn.execute("ALTER ROLE effisql_agent IN DATABASE template1 SET work_mem = '32MB'")
    try:
        async with _pool(writable_env, instances=[instance]).environment() as env:
            settings = await env.execute(
                "SELECT current_setting('work_mem'), "
                "current_setting('default_transaction_read_only'), "
                "current_setting('temp_file_limit')",
                timeout_s=10,
            )
        with admin("postgres") as conn:
            stored = conn.execute(
                "SELECT d.datname, s.setconfig FROM pg_db_role_setting s "
                "LEFT JOIN pg_database d ON d.oid = s.setdatabase "
                "WHERE s.setrole = 'effisql_agent'::regrole"
            ).fetchall()
    finally:
        apply_roles(admin, writable_env.credentials, instance.kind)

    assert _rows(settings) == [["4MB", "on", "5GB"]]
    assert [(db, sorted(config)) for db, config in stored] == [
        (None, ["default_transaction_read_only=on", "temp_file_limit=5GB"])
    ]
