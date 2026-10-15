import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest
from google.protobuf.struct_pb2 import Value
from psycopg.pq import ExecStatus
from pydantic import SecretStr

from goaly.bench.effisql.pg_instances import SOLVING
from goaly.bench.effisql.pg_settings import PgCredentials, PgEnvConfig
from goaly.bench.effisql.sql_env import (
    STATEMENT_TIMEOUT,
    PgSqlEnvironment,
    ResultBuilder,
    SolvingPool,
    SqlEnvOptions,
    _Slot,
    connection_lost,
    encode_value,
)
from goaly.contracts.errors import EnvironmentFailure

INT4, INT8, NUMERIC, FLOAT8, BOOL, TEXT, DATE = 23, 20, 1700, 701, 16, 25, 1082


def _json(value: Value) -> object:
    kind = value.WhichOneof("kind")
    return None if kind is None or kind == "null_value" else getattr(value, kind)


@pytest.mark.parametrize(
    ("raw", "oid", "expected"),
    [
        (None, INT4, None),
        (b"42", INT4, 42.0),
        (b"9007199254740992", INT8, 9007199254740992.0),  # 2^53 is still exact.
        (b"9007199254740993", INT8, "9007199254740993"),  # Beyond 2^53: a string.
        (b"-9007199254740993", INT8, "-9007199254740993"),
        (b"12345.67", NUMERIC, "12345.67"),  # numeric is always a string.
        (b"NaN", NUMERIC, "NaN"),
        (b"3.25", FLOAT8, 3.25),
        (b"NaN", FLOAT8, "NaN"),
        (b"Infinity", FLOAT8, "Infinity"),
        (b"-Infinity", FLOAT8, "-Infinity"),
        (b"t", BOOL, True),
        (b"f", BOOL, False),
        (b"NULL", TEXT, "NULL"),  # A string, distinct from SQL NULL.
        (b"1996-03-13", DATE, "1996-03-13"),  # Anything else: the server's text.
        (b"caf\xc3\xa9 \xff", TEXT, "café �"),
    ],
)
def test_values_are_encoded_as_the_sql_result_contract_says(
    raw: bytes | None, oid: int, expected: object
) -> None:
    assert _json(encode_value(raw, oid)) == expected


@dataclass
class FakeResult:
    """A PGresult as the builder reads it."""

    status: ExecStatus
    rows: list[list[bytes | None]] = field(default_factory=list)
    columns: list[tuple[str, int]] = field(default_factory=list)
    command_tuples: int | None = None
    error_message: bytes = b""
    sqlstate: bytes | None = None

    @property
    def ntuples(self) -> int:
        return len(self.rows)

    @property
    def nfields(self) -> int:
        return len(self.columns)

    def fname(self, column: int) -> bytes:
        return self.columns[column][0].encode()

    def ftype(self, column: int) -> int:
        return self.columns[column][1]

    def get_value(self, row: int, column: int) -> bytes | None:
        return self.rows[row][column]

    def error_field(self, code: int) -> bytes | None:
        return self.sqlstate


COLUMNS = [("id", INT4), ("name", TEXT)]


def _chunk(start: int, count: int) -> FakeResult:
    rows: list[list[bytes | None]] = [
        [str(i).encode(), f"n{i}".encode()] for i in range(start, start + count)
    ]
    return FakeResult(ExecStatus.TUPLES_CHUNK, rows, COLUMNS)


def test_rows_are_counted_in_full_but_kept_only_up_to_the_limit() -> None:
    builder = ResultBuilder(row_limit=3)
    for res in (_chunk(0, 2), _chunk(2, 2), _chunk(4, 2), FakeResult(ExecStatus.TUPLES_OK)):
        builder.add(res)

    result = builder.result()

    assert list(result.columns) == ["id", "name"]
    assert [[_json(v) for v in row.values] for row in result.rows] == [
        [0.0, "n0"],
        [1.0, "n1"],
        [2.0, "n2"],
    ]
    assert result.row_count == 6
    assert result.truncated
    assert result.error == ""


def test_a_query_with_no_rows_still_has_its_columns() -> None:
    builder = ResultBuilder(row_limit=3)
    builder.add(FakeResult(ExecStatus.TUPLES_OK, [], COLUMNS))

    result = builder.result()

    assert list(result.columns) == ["id", "name"]
    assert (result.row_count, result.truncated, len(result.rows)) == (0, False, 0)


def test_the_result_is_the_last_statements() -> None:
    builder = ResultBuilder(row_limit=10)
    for res in (
        _chunk(0, 5),
        FakeResult(ExecStatus.TUPLES_OK),
        FakeResult(ExecStatus.COMMAND_OK, command_tuples=0),  # SET work_mem = ...
        FakeResult(ExecStatus.TUPLES_CHUNK, [[b"8MB"]], [("work_mem", TEXT)]),
        FakeResult(ExecStatus.TUPLES_OK),
    ):
        builder.add(res)

    result = builder.result()

    assert list(result.columns) == ["work_mem"]
    assert result.row_count == 1
    assert _json(result.rows[0].values[0]) == "8MB"


def test_a_command_reports_the_rows_it_affected() -> None:
    builder = ResultBuilder(row_limit=10)
    builder.add(FakeResult(ExecStatus.COMMAND_OK, command_tuples=7))

    result = builder.result()

    assert (result.row_count, list(result.columns), len(result.rows)) == (7, [], 0)


def test_an_error_ends_the_result_with_the_servers_message() -> None:
    builder = ResultBuilder(row_limit=10)
    for res in (
        _chunk(0, 2),
        FakeResult(
            ExecStatus.FATAL_ERROR,
            error_message=b"ERROR:  division by zero\n",
            sqlstate=b"22012",
        ),
    ):
        builder.add(res)

    result = builder.result()

    assert result.error == "ERROR:  division by zero"
    assert (result.row_count, len(result.rows), list(result.columns)) == (0, 0, [])
    assert builder.sqlstate == "22012"


def test_an_empty_query_is_an_empty_result() -> None:
    builder = ResultBuilder(row_limit=10)
    builder.add(FakeResult(ExecStatus.EMPTY_QUERY))

    assert builder.result() == builder.result().__class__()


@pytest.mark.parametrize(
    ("sqlstate", "lost"),
    [
        ("08006", True),  # connection_failure
        ("57P01", True),  # admin_shutdown: terminating connection
        ("57P02", True),  # crash_shutdown
        ("57P03", True),  # cannot_connect_now
        ("57014", False),  # query_canceled: the statement failed, the session lives
        ("22012", False),  # division_by_zero
        ("42501", False),  # insufficient_privilege
        (None, False),
    ],
)
def test_only_errors_that_end_the_session_count_as_a_lost_connection(
    sqlstate: str | None, lost: bool
) -> None:
    assert connection_lost(sqlstate) is lost


def test_a_timeout_reads_as_postgresqls_own_message() -> None:
    assert STATEMENT_TIMEOUT == "canceling statement due to statement timeout"


# The orchestration around a statement, with a fake connection and a fake runner.


class FakeConn:
    def __init__(self, pid: int = 4242, role_defaults_changed: bool = False) -> None:
        self.info = SimpleNamespace(backend_pid=pid)
        self.pgconn = SimpleNamespace(finish=self._finish)
        self.cancelled = asyncio.Event()
        self.closed = False
        self.finished = False
        self.role_defaults_changed = role_defaults_changed
        self.executed: list[str] = []

    async def cancel_safe(self, *, timeout: float) -> None:
        self.cancelled.set()

    async def close(self) -> None:
        self.closed = True

    def _finish(self) -> None:
        self.finished = True

    async def execute(self, query: str, params: object = None) -> Any:
        self.executed.append(query)
        count = 1 if self.role_defaults_changed and "pg_db_role_setting" in query else 0
        return SimpleNamespace(fetchone=lambda: _resolved((count,)))

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


async def _resolved(value: object) -> object:
    return value


class Scenario:
    """A fake runner and the connections and terminations it saw."""

    def __init__(self, behave: str = "ok", seconds: float = 0.0) -> None:
        self.behave = behave
        self.seconds = seconds
        self.connections: list[FakeConn] = []
        self.terminated: list[int] = []
        self.statements: list[str] = []
        self.terminate_event = asyncio.Event()

    async def connect(self) -> Any:
        conn = FakeConn(pid=1000 + len(self.connections))
        self.connections.append(conn)
        return conn

    async def terminate(self, pid: int) -> None:
        self.terminated.append(pid)
        self.terminate_event.set()

    async def run(self, conn: Any, sql: str, builder: ResultBuilder) -> None:
        self.statements.append(sql)
        if self.behave == "ok":
            await asyncio.sleep(self.seconds)
            builder.add(FakeResult(ExecStatus.TUPLES_CHUNK, [[b"1"]], [("x", INT4)]))
            builder.add(FakeResult(ExecStatus.TUPLES_OK))
        elif self.behave == "cancellable":
            await conn.cancelled.wait()
            builder.add(_error(b"57014", b"ERROR:  canceling statement due to user request"))
        elif self.behave == "stubborn":
            await self.terminate_event.wait()
            raise psycopg.OperationalError("terminating connection due to administrator command")
        elif self.behave == "broken":
            raise psycopg.OperationalError("server closed the connection unexpectedly")
        elif self.behave == "shutdown":
            builder.add(_error(b"57P01", b"FATAL:  terminating connection"))


def _error(sqlstate: bytes, message: bytes) -> FakeResult:
    return FakeResult(ExecStatus.FATAL_ERROR, error_message=message, sqlstate=sqlstate)


def _environment(
    scenario: Scenario, slot: _Slot | None = None, grace: float = 0.2
) -> PgSqlEnvironment:
    slot = slot if slot is not None else _Slot(SOLVING[0], asyncio.Semaphore(2))
    return PgSqlEnvironment(
        slot,
        SqlEnvOptions(cancel_grace_s=grace),
        connect=scenario.connect,
        terminate=scenario.terminate,
        runner=scenario.run,
    )


@pytest.mark.anyio
async def test_execute_returns_the_last_statements_rows_with_its_timings() -> None:
    scenario = Scenario(seconds=0.05)
    env = _environment(scenario)
    await env.open()

    result = await env.execute("SELECT 1", timeout_s=5)

    assert [[_json(v) for v in row.values] for row in result.rows] == [[1.0]]
    assert 0.05 <= result.elapsed_s < 1
    assert 0 <= result.queue_wait_s < 0.05
    assert scenario.statements == ["SELECT 1"]


@pytest.mark.anyio
async def test_a_timeout_cancels_the_statement_and_keeps_the_session() -> None:
    scenario = Scenario("cancellable")
    env = _environment(scenario)
    await env.open()

    result = await env.execute("SELECT pg_sleep(60)", timeout_s=0.1)

    assert result.error == STATEMENT_TIMEOUT
    assert 0.1 <= result.elapsed_s < 1
    assert scenario.connections[0].cancelled.is_set()
    assert scenario.terminated == []
    scenario.behave = "ok"
    assert (await env.execute("SELECT 1", timeout_s=5)).error == ""
    assert len(scenario.connections) == 1  # The same session.


@pytest.mark.anyio
async def test_a_statement_that_ignores_its_cancel_has_its_backend_terminated() -> None:
    scenario = Scenario("stubborn")
    env = _environment(scenario)
    await env.open()

    result = await env.execute("SELECT stubborn()", timeout_s=0.1)

    assert result.error == STATEMENT_TIMEOUT
    assert scenario.terminated == [1000]
    scenario.behave = "ok"
    assert (await env.execute("SELECT 1", timeout_s=5)).error == ""
    assert len(scenario.connections) == 2  # A new session after the terminated one.


@pytest.mark.anyio
async def test_a_statement_that_survives_even_termination_is_abandoned() -> None:
    scenario = Scenario("stubborn")
    scenario.terminate = _never_terminates  # type: ignore[method-assign]
    env = _environment(scenario, grace=0.05)
    await env.open()

    result = await env.execute("SELECT stuck()", timeout_s=0.05)

    assert result.error == STATEMENT_TIMEOUT
    assert scenario.connections[0].finished  # Its socket is closed; the next call reconnects.


async def _never_terminates(pid: int) -> None:
    raise psycopg.OperationalError("the admin connection failed")


@pytest.mark.anyio
async def test_a_cancelled_execute_stops_its_statement_before_raising() -> None:
    scenario = Scenario("cancellable")
    env = _environment(scenario)
    await env.open()
    task = asyncio.create_task(env.execute("SELECT pg_sleep(60)", timeout_s=60))
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert scenario.connections[0].cancelled.is_set()
    scenario.behave = "ok"
    assert (await env.execute("SELECT 1", timeout_s=5)).error == ""


@pytest.mark.anyio
@pytest.mark.parametrize("behave", ["broken", "shutdown"])
async def test_a_lost_connection_raises_environment_failure_and_reconnects_next_time(
    behave: str,
) -> None:
    scenario = Scenario(behave)
    env = _environment(scenario)
    await env.open()

    with pytest.raises(EnvironmentFailure, match=f"connection to {SOLVING[0].name} failed"):
        await env.execute("SELECT 1", timeout_s=5)

    assert scenario.connections[0].closed
    scenario.behave = "ok"
    assert (await env.execute("SELECT 1", timeout_s=5)).error == ""
    assert len(scenario.connections) == 2


@pytest.mark.anyio
async def test_an_instance_that_refuses_the_session_is_an_environment_failure() -> None:
    async def refuse() -> Any:
        raise psycopg.OperationalError("password authentication failed")

    env = PgSqlEnvironment(
        _Slot(SOLVING[0], asyncio.Semaphore(2)), SqlEnvOptions(), refuse, Scenario().terminate
    )

    with pytest.raises(EnvironmentFailure, match="cannot open a session on solve-1"):
        await env.open()


@pytest.mark.anyio
async def test_role_defaults_an_earlier_task_set_get_a_fresh_session() -> None:
    scenario = Scenario()
    first = FakeConn(role_defaults_changed=True)
    connections = [first, FakeConn()]

    async def connect() -> Any:
        return connections.pop(0)

    env = PgSqlEnvironment(
        _Slot(SOLVING[0], asyncio.Semaphore(2)), SqlEnvOptions(), connect, scenario.terminate
    )
    await env.open()

    assert first.closed
    assert any("RESET ALL" in query for query in first.executed)
    assert connections == []  # The second, clean session is the one in use.


@pytest.mark.anyio
async def test_a_closed_environment_refuses_calls_and_closes_once() -> None:
    scenario = Scenario()
    env = _environment(scenario)
    await env.open()

    await env.close()
    await env.close()

    assert scenario.connections[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        await env.execute("SELECT 1", timeout_s=5)


@pytest.mark.anyio
async def test_the_statement_cap_queues_the_third_statement_and_reports_its_wait() -> None:
    scenario = Scenario(seconds=0.2)
    slot = _Slot(SOLVING[0], asyncio.Semaphore(2))
    envs = [_environment(scenario, slot) for _ in range(3)]
    for env in envs:
        await env.open()

    results = await asyncio.gather(*(env.execute("SELECT 1", timeout_s=5) for env in envs))

    waits = sorted(result.queue_wait_s for result in results)
    assert waits[1] < 0.1
    assert waits[2] >= 0.15  # The third waited for a slot,
    assert all(result.elapsed_s < 0.4 for result in results)  # which its elapsed_s leaves out.


@pytest.mark.anyio
async def test_the_pool_assigns_the_least_loaded_instance_and_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[dict[str, object]] = []

    async def connect(**kwargs: object) -> Any:
        opened.append(kwargs)
        return FakeConn()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", connect)
    credentials = PgCredentials(
        admin_password=SecretStr("a"),
        agent_password=SecretStr("b"),
        grader_password=SecretStr("c"),
        _env_file=None,
    )
    pool = SolvingPool(PgEnvConfig(), credentials)

    async with pool.environment() as first, pool.environment() as second:
        assert {first.instance.name, second.instance.name} == {"solve-1", "solve-2"}
        assert pool.load() == {"solve-1": 1, "solve-2": 1}
        async with pool.environment() as third:
            assert third.instance.name == "solve-1"  # Ties go to the first instance.
    assert pool.load() == {"solve-1": 0, "solve-2": 0}
    assert {o["user"] for o in opened} == {"effisql_agent"}
    assert {o["dbname"] for o in opened} == {"tpch"}
    assert all(o["autocommit"] for o in opened)


class PreflightConn:
    def __init__(self, row: tuple[object, ...]) -> None:
        self.row = row
        self.closed = False

    async def execute(self, query: str) -> Any:
        return SimpleNamespace(fetchone=lambda: _resolved(self.row))

    async def close(self) -> None:
        self.closed = True


def _preflight_pool(monkeypatch: pytest.MonkeyPatch, answers: Mapping[int, object]) -> SolvingPool:
    async def connect(**kwargs: object) -> Any:
        answer = answers[int(str(kwargs["port"]))]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", connect)
    credentials = PgCredentials(
        admin_password=SecretStr("a"),
        agent_password=SecretStr("b"),
        grader_password=SecretStr("c"),
        _env_file=None,
    )
    return SolvingPool(PgEnvConfig(), credentials)


@pytest.mark.anyio
async def test_preflight_passes_when_every_solving_instance_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = {55432: PreflightConn(("on", True)), 55433: PreflightConn(("on", True))}

    await _preflight_pool(monkeypatch, ready).preflight()

    assert all(conn.closed for conn in ready.values())


@pytest.mark.anyio
async def test_preflight_fails_fast_naming_every_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _preflight_pool(
        monkeypatch,
        {
            55432: psycopg.OperationalError("password authentication failed"),
            55433: PreflightConn(("off", False)),
        },
    )

    with pytest.raises(EnvironmentFailure) as caught:
        await pool.preflight()

    message = str(caught.value)
    assert "solve-1 (port 55432): the agent role cannot log in: password" in message
    assert "solve-2 (port 55433): the agent's transactions are not read-only by default" in message
    assert "solve-2 (port 55433): the agent role cannot read the TPC-H tables" in message
    assert "pgenv.py start" in message
