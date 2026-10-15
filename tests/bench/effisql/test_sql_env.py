import asyncio
import logging
import math
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest
from google.protobuf.struct_pb2 import Value
from psycopg.pq import ExecStatus
from psycopg.waiting import Ready, Wait
from pydantic import SecretStr

from goaly.bench.effisql import sql_env
from goaly.bench.effisql.pg_instances import SOLVING
from goaly.bench.effisql.pg_settings import AgentCredentials, PgCredentials, PgEnvConfig
from goaly.bench.effisql.sql_env import (
    STATEMENT_TIMEOUT,
    PgSqlEnvironment,
    ResultBuilder,
    SolvingPool,
    SqlEnvOptions,
    connection_lost,
    encode_value,
    refuse_copy_data,
    role_defaults_problems,
    skip_copy_data,
    statement_text_problem,
)
from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.interfaces import SqlEnvironment
from tests.bench.effisql.pg_fakes import as_text

INT4, INT8, NUMERIC, FLOAT8, BOOL, TEXT, DATE = 23, 20, 1700, 701, 16, 25, 1082
LOGGER = "goaly.bench.effisql.sql_env"


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


def test_a_long_value_is_cut_and_says_what_was_left_out() -> None:
    assert _json(encode_value(b"x" * 10, TEXT, max_bytes=4)) == "xxxx[... 6 more bytes]"
    assert _json(encode_value(b"x" * 4, TEXT, max_bytes=4)) == "xxxx"
    assert _json(encode_value(b"1" * 5000, NUMERIC)) == "1" * 4096 + "[... 904 more bytes]"


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


def _chunk(start: int, count: int, status: ExecStatus = ExecStatus.TUPLES_CHUNK) -> FakeResult:
    rows: list[list[bytes | None]] = [
        [str(i).encode(), f"n{i}".encode()] for i in range(start, start + count)
    ]
    return FakeResult(status, rows, COLUMNS)


def _rows(result: Any) -> list[list[object]]:
    return [[_json(v) for v in row.values] for row in result.rows]


def test_rows_are_counted_in_full_but_kept_only_up_to_the_limit() -> None:
    builder = ResultBuilder(row_limit=3)
    for res in (_chunk(0, 2), _chunk(2, 2), _chunk(4, 2), FakeResult(ExecStatus.TUPLES_OK)):
        builder.add(res)

    result = builder.result()

    assert list(result.columns) == ["id", "name"]
    assert _rows(result) == [[0.0, "n0"], [1.0, "n1"], [2.0, "n2"]]
    assert result.row_count == 6
    assert result.truncated
    assert result.error == ""


def test_rows_stop_being_kept_once_their_values_reach_the_byte_cap() -> None:
    builder = ResultBuilder(row_limit=100, value_bytes=8, result_bytes=20)
    big: list[list[bytes | None]] = [[b"a" * 30], [b"b" * 30], [b"c" * 30], [b"d" * 30]]
    builder.add(FakeResult(ExecStatus.TUPLES_CHUNK, big, [("v", TEXT)]))
    builder.add(FakeResult(ExecStatus.TUPLES_OK))

    result = builder.result()

    # Each value counts its first 8 bytes: three rows reach 24 of the 20 allowed.
    assert _rows(result) == [[f"{c * 8}[... 22 more bytes]"] for c in "abc"]
    assert (result.row_count, result.truncated) == (4, True)


def test_a_result_without_chunks_still_has_its_rows_decoded_and_counted() -> None:
    builder = ResultBuilder(row_limit=10)
    builder.add(_chunk(0, 3, ExecStatus.TUPLES_OK))  # As libpq gives it outside chunked mode.

    result = builder.result()

    assert _rows(result) == [[0.0, "n0"], [1.0, "n1"], [2.0, "n2"]]
    assert (result.row_count, result.truncated) == (3, False)


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
        FakeResult(ExecStatus.COMMAND_OK, command_tuples=0),
        FakeResult(ExecStatus.TUPLES_CHUNK, [[b"8MB"]], [("work_mem", TEXT)]),
        FakeResult(ExecStatus.TUPLES_OK),
    ):
        builder.add(res)

    result = builder.result()

    assert list(result.columns) == ["work_mem"]
    assert result.row_count == 1
    assert _rows(result) == [["8MB"]]


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
        ("57P04", True),  # database_dropped
        ("57P05", True),  # idle_session_timeout
        ("25P03", True),  # idle_in_transaction_session_timeout
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


@pytest.mark.parametrize(
    ("sql", "problem"),
    [
        ("SELECT 1", None),
        ("SELECT 'café'", None),
        ("SELECT 1\x00; DROP TABLE x", "contains a NUL character"),
        ("SELECT '\ud800'", "not valid UTF-8 at character 8: surrogates not allowed"),
    ],
)
def test_statement_text_postgresql_cannot_take_is_named(sql: str, problem: str | None) -> None:
    found = statement_text_problem(sql)

    if problem is None:
        assert found is None
    else:
        assert found is not None
        assert problem in found


ROLE_WIDE = [
    "client_connection_check_interval=5s",
    "default_transaction_read_only=on",
    "statement_timeout=130s",
    "temp_file_limit=5GB",
]
GOOD_DEFAULTS: list[tuple[Any, ...]] = [(None, list(reversed(ROLE_WIDE)))]  # In any order.
EXPECTED_DEFAULTS = str(ROLE_WIDE)


@pytest.mark.parametrize(
    ("rows", "problems"),
    [
        (GOOD_DEFAULTS, []),
        # The role's own RESET ALL removed the row: its sessions would start read-write.
        ([], [f"role-wide defaults [], not {EXPECTED_DEFAULTS}"]),
        (
            [(None, ROLE_WIDE[:3])],  # The temp-file cap is gone.
            [f"role-wide defaults {ROLE_WIDE[:3]}, not {EXPECTED_DEFAULTS}"],
        ),
        (
            [*GOOD_DEFAULTS, ("template1", ["work_mem=1GB"])],
            ["defaults in database template1: work_mem=1GB"],
        ),
    ],
)
def test_role_defaults_must_be_exactly_the_setups(
    rows: list[tuple[Any, ...]], problems: list[str]
) -> None:
    assert role_defaults_problems(rows) == problems


class FakeCopyConn:
    """A PGconn in a COPY: hands out ``chunks``, where None means nothing has arrived yet."""

    def __init__(self, chunks: Sequence[bytes | None] = (), *, blocked_ends: int = 0) -> None:
        self.chunks = list(chunks)
        self.consumed = 0
        self.blocked_ends = blocked_ends  # put_copy_end calls that find the buffer full.
        self.ended: list[bytes | None] = []
        self.flushes = [1, 0]  # The first flush leaves data to send.

    def get_copy_data(self, async_: int) -> tuple[int, memoryview]:
        assert async_ == 1
        if not self.chunks:
            return -1, memoryview(b"")
        chunk = self.chunks.pop(0)
        return (0, memoryview(b"")) if chunk is None else (len(chunk), memoryview(chunk))

    def consume_input(self) -> None:
        self.consumed += 1

    def put_copy_end(self, error: bytes | None) -> int:
        if self.blocked_ends:
            self.blocked_ends -= 1
            return 0
        self.ended.append(error)
        return 1

    def flush(self) -> int:
        return self.flushes.pop(0)


def _drive(gen: Any) -> list[Wait]:
    """Run a libpq generator to its end, answering every wait as ready; return the waits."""
    waits = []
    try:
        waits.append(gen.send(None))
        while True:
            waits.append(gen.send(Ready.RW))
    except StopIteration:
        return waits


def test_copy_data_is_read_and_dropped_until_the_copy_ends() -> None:
    pgconn = FakeCopyConn([b"1\tx\n", None, b"2\ty\n", None])

    waits = _drive(skip_copy_data(pgconn))  # type: ignore[arg-type]

    assert waits == [Wait.R, Wait.R]  # It waited only when nothing had arrived,
    assert pgconn.consumed == 2  # and read the input that then came.
    assert pgconn.chunks == []


def test_a_copy_from_the_client_is_refused_with_a_failure() -> None:
    pgconn = FakeCopyConn(blocked_ends=1)

    waits = _drive(refuse_copy_data(pgconn))  # type: ignore[arg-type]

    assert pgconn.ended == [b"COPY from the client is not supported here"]
    assert waits == [Wait.W, Wait.W]  # Once for the full buffer, once to finish the flush.


# The orchestration around a statement, with a fake connection and a fake runner.


class FakeCursor:
    def __init__(self, rows: Sequence[tuple[Any, ...]]) -> None:
        self._rows = list(rows)

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    async def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None


class FakeConn:
    """An agent session as the environment sees it; ``defaults`` are its role's stored ones."""

    def __init__(
        self,
        pid: int = 4242,
        defaults: Sequence[tuple[Any, ...]] = GOOD_DEFAULTS,
        *,
        close_delay: float = 0.0,
        settings_gate: asyncio.Event | None = None,
    ) -> None:
        self.info = SimpleNamespace(backend_pid=pid)
        self.defaults = list(defaults)
        self.cancel_error: BaseException | None = None
        self.close_delay = close_delay
        self.settings_gate = settings_gate
        self.cancelled = asyncio.Event()
        self.closed = False
        self.finished = False  # The socket was closed under a running statement.
        self.executed: list[str] = []
        self.params: list[object] = []
        self.pgconn = SimpleNamespace(finish=self._finish)

    def _finish(self) -> None:
        self.finished = True
        self.closed = True

    async def cancel_safe(self, *, timeout: float) -> None:
        self.cancelled.set()
        if self.cancel_error is not None:
            raise self.cancel_error

    async def close(self) -> None:
        if self.close_delay:
            await asyncio.sleep(self.close_delay)  # As a real close may wait on the network.
        self.closed = True

    async def execute(self, query: object, params: object = None) -> FakeCursor:
        text = as_text(query)
        self.executed.append(text)
        self.params.append(params)
        if "pg_db_role_setting" in text:
            if self.settings_gate is not None:
                await self.settings_gate.wait()
            return FakeCursor(self.defaults)
        return FakeCursor([])

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        self.executed.append("BEGIN")
        yield
        self.executed.append("COMMIT")

    async def __aenter__(self) -> "FakeConn":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


class Scenario:
    """A fake runner, and the sessions it opened."""

    def __init__(self, behave: str = "ok", seconds: float = 0.0) -> None:
        self.behave = behave
        self.seconds = seconds
        self.connections: list[FakeConn] = []
        self.statements: list[str] = []
        self.stopped = 0  # Statements whose runner was cancelled under them.
        self.connect_delay = 0.0
        self.hold_on_cancel_s = 0.0  # How long a cancelled runner still keeps its session.

    async def connect(self) -> Any:
        await asyncio.sleep(self.connect_delay)
        conn = FakeConn(pid=1000 + len(self.connections))
        self.connections.append(conn)
        return conn

    async def run(self, conn: Any, sql: str, builder: ResultBuilder) -> None:
        self.statements.append(sql)
        if self.behave == "ok":
            await asyncio.sleep(self.seconds)
            builder.add(FakeResult(ExecStatus.TUPLES_CHUNK, [[b"1"]], [("x", INT4)]))
            builder.add(FakeResult(ExecStatus.TUPLES_OK))
        elif self.behave == "cancellable":
            await conn.cancelled.wait()
            builder.add(_error(b"57014", b"ERROR:  canceling statement due to user request"))
        elif self.behave == "stubborn":  # It ignores its cancel, until its runner is cancelled.
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.stopped += 1
                if self.hold_on_cancel_s:  # As psycopg's own cleanup may take a while.
                    await asyncio.shield(asyncio.sleep(self.hold_on_cancel_s))
                raise
        elif self.behave == "broken":
            raise psycopg.OperationalError("server closed the connection unexpectedly")
        elif self.behave == "shutdown":
            builder.add(_error(b"57P01", b"FATAL:  terminating connection"))
        elif self.behave == "closes":  # The session went away, with no error to say so.
            conn.closed = True
        elif self.behave == "bug":
            raise ValueError("a result this code did not expect")


def _error(sqlstate: bytes, message: bytes) -> FakeResult:
    return FakeResult(ExecStatus.FATAL_ERROR, error_message=message, sqlstate=sqlstate)


def _environment(
    scenario: Scenario,
    cap: asyncio.Semaphore | None = None,
    grace: float = 0.2,
    timeout_s: float = 120.0,
) -> PgSqlEnvironment:
    return PgSqlEnvironment(
        SOLVING[0],
        cap if cap is not None else asyncio.Semaphore(2),
        SqlEnvOptions(cancel_grace_s=grace, timeout_s=timeout_s),
        connect=scenario.connect,
        runner=scenario.run,
    )


def _sql_environment(env: PgSqlEnvironment) -> SqlEnvironment:
    return env  # mypy checks PgSqlEnvironment against the contract here.


def _logged(caplog: pytest.LogCaptureFixture, start: str) -> logging.LogRecord:
    return next(r for r in caplog.records if r.getMessage().startswith(start))


@pytest.mark.anyio
async def test_execute_returns_the_last_statements_rows_with_its_timings() -> None:
    scenario = Scenario(seconds=0.05)
    env = _sql_environment(_environment(scenario))

    result = await env.execute("SELECT 1", timeout_s=5)

    assert _rows(result) == [[1.0]]
    assert 0.05 <= result.elapsed_s < 1
    assert 0 <= result.queue_wait_s < 0.05
    assert scenario.statements == ["SELECT 1"]


@pytest.mark.anyio
async def test_the_trusted_timeout_caps_what_a_call_asks_for() -> None:
    scenario = Scenario("cancellable")
    env = _environment(scenario, timeout_s=0.2)
    await env.open()

    for timeout_s in (1e9, None):
        scenario.connections[0].cancelled.clear()  # Only the cancel of this call ends it.
        result = await env.execute("SELECT pg_sleep(1e9)", timeout_s=timeout_s)

        assert result.error == STATEMENT_TIMEOUT
        assert 0.2 <= result.elapsed_s < 0.5  # The trusted 0.2 s, not the 1e9 s asked for.


@pytest.mark.anyio
@pytest.mark.parametrize("timeout_s", [0.0, -1.0, math.nan])
async def test_a_timeout_that_is_not_positive_is_refused(timeout_s: float) -> None:
    scenario = Scenario()
    env = _environment(scenario)

    with pytest.raises(ValueError, match="timeout_s must be positive"):
        await env.execute("SELECT 1", timeout_s=timeout_s)
    assert scenario.statements == []


@pytest.mark.anyio
async def test_text_postgresql_cannot_take_is_an_error_result_and_never_sent() -> None:
    scenario = Scenario()
    env = _environment(scenario)
    await env.open()

    result = await env.execute("SELECT '\ud800'", timeout_s=5)

    assert "not valid UTF-8" in result.error
    assert scenario.statements == []
    assert (await env.execute("SELECT 1", timeout_s=5)).error == ""


@pytest.mark.anyio
async def test_a_timeout_cancels_the_statement_and_keeps_the_session() -> None:
    scenario = Scenario("cancellable")
    env = _environment(scenario)
    await env.open()

    result = await env.execute("SELECT pg_sleep(60)", timeout_s=0.1)

    assert result.error == STATEMENT_TIMEOUT
    assert 0.1 <= result.elapsed_s < 1
    assert scenario.connections[0].cancelled.is_set()
    assert scenario.stopped == 0
    scenario.behave = "ok"
    assert (await env.execute("SELECT 1", timeout_s=5)).error == ""
    assert len(scenario.connections) == 1  # The same session.


@pytest.mark.anyio
async def test_a_statement_that_ignores_its_cancel_has_its_session_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scenario = Scenario("stubborn")
    env = _environment(scenario)
    await env.open()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await env.execute("SELECT stubborn()", timeout_s=0.1)

    assert result.error == STATEMENT_TIMEOUT
    assert scenario.connections[0].cancelled.is_set()  # The cancel key came first;
    assert scenario.stopped == 1  # then its runner let go of the session,
    assert scenario.connections[0].closed  # which was closed: the server ends the backend.
    closing = _logged(caplog, "closing the session of backend 1000 on solve-1")
    assert closing.levelno == logging.WARNING
    scenario.behave = "ok"
    assert (await env.execute("SELECT 1", timeout_s=5)).error == ""
    assert len(scenario.connections) == 2  # A new session after the closed one.


@pytest.mark.anyio
async def test_a_failed_cancel_is_logged_with_its_cause(caplog: pytest.LogCaptureFixture) -> None:
    scenario = Scenario("stubborn")
    env = _environment(scenario, grace=0.05)
    await env.open()
    scenario.connections[0].cancel_error = psycopg.OperationalError("cancel refused")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await env.execute("SELECT stubborn()", timeout_s=0.05)

    assert result.error == STATEMENT_TIMEOUT
    failed = _logged(caplog, "the cancel request to solve-1 failed")
    assert failed.levelno == logging.WARNING
    assert failed.exc_info is not None
    assert "cancel refused" in str(failed.exc_info[1])


@pytest.mark.anyio
async def test_a_runner_that_keeps_its_session_has_the_socket_closed_under_it(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sql_env, "_CLOSE_WAIT_S", 0.05)
    scenario = Scenario("stubborn")
    scenario.hold_on_cancel_s = 0.3  # Longer than the environment waits for it.
    env = _environment(scenario, grace=0.05)
    await env.open()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        async with asyncio.timeout(2):
            result = await env.execute("SELECT stuck()", timeout_s=0.05)

    assert result.error == STATEMENT_TIMEOUT
    assert scenario.connections[0].finished  # Its socket was closed under the statement,
    assert _logged(caplog, "the statement on solve-1 kept its session").levelno == logging.ERROR
    scenario.behave = "ok"
    assert (await env.execute("SELECT 1", timeout_s=5)).error == ""  # and the next call
    assert len(scenario.connections) == 2  # opens a new session.


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
async def test_a_second_cancel_still_waits_for_the_statement_to_stop() -> None:
    scenario = Scenario("stubborn")
    env = _environment(scenario, grace=0.2)
    await env.open()
    task = asyncio.create_task(env.execute("SELECT stubborn()", timeout_s=60))
    await asyncio.sleep(0.05)

    task.cancel()  # The stop starts: a cancel, then the session closed 0.2 s later.
    await asyncio.sleep(0.1)
    task.cancel()  # A second cancel while it runs.
    with pytest.raises(asyncio.CancelledError):
        await task

    assert scenario.stopped == 1  # The stop finished before the raise.
    assert scenario.connections[0].closed


@pytest.mark.anyio
@pytest.mark.parametrize("behave", ["broken", "shutdown", "closes"])
async def test_a_lost_session_is_an_error_result_and_the_next_call_reopens(
    behave: str,
) -> None:
    scenario = Scenario(behave)
    env = _environment(scenario)
    await env.open()

    result = await env.execute("SELECT pg_terminate_backend(pg_backend_pid())", timeout_s=5)

    assert result.error.startswith("the session ended: ")
    assert result.error.endswith("the next statement opens a new one")
    assert scenario.connections[0].closed
    scenario.behave = "ok"
    assert (await env.execute("SELECT 1", timeout_s=5)).error == ""
    assert len(scenario.connections) == 2


@pytest.mark.anyio
async def test_a_runner_that_fails_unexpectedly_gives_an_error_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scenario = Scenario("bug")
    env = _environment(scenario)
    await env.open()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await env.execute("SELECT 1", timeout_s=5)

    assert "a result this code did not expect" in result.error
    assert _logged(caplog, "a statement on solve-1 broke its session").exc_info is not None
    scenario.behave = "ok"
    assert (await env.execute("SELECT 1", timeout_s=5)).error == ""


@pytest.mark.anyio
async def test_a_session_that_closed_between_calls_is_reopened_first() -> None:
    scenario = Scenario()
    env = _environment(scenario)
    await env.open()
    scenario.connections[0].closed = True  # It broke, e.g. while a statement timed out.

    result = await env.execute("SELECT 1", timeout_s=5)

    assert result.error == ""
    assert len(scenario.connections) == 2


@pytest.mark.anyio
async def test_opening_a_session_holds_no_statement_slot_and_counts_in_no_timing() -> None:
    scenario = Scenario()
    env = _environment(scenario)
    scenario.connect_delay = 0.2

    result = await env.execute("SELECT 1", timeout_s=5)  # Its first call opens the session.

    assert result.error == ""
    assert result.queue_wait_s < 0.1
    assert result.elapsed_s < 0.1


@pytest.mark.anyio
async def test_an_instance_that_refuses_the_session_is_an_environment_failure() -> None:
    async def refuse() -> Any:
        raise psycopg.OperationalError("password authentication failed")

    env = PgSqlEnvironment(SOLVING[0], asyncio.Semaphore(2), SqlEnvOptions(), refuse)

    with pytest.raises(EnvironmentFailure, match="cannot open a session on solve-1"):
        await env.open()


@pytest.mark.anyio
async def test_a_cancel_while_the_session_opens_closes_it() -> None:
    gate = asyncio.Event()
    conn = FakeConn(settings_gate=gate)

    async def connect() -> Any:
        return conn

    env = PgSqlEnvironment(SOLVING[0], asyncio.Semaphore(2), SqlEnvOptions(), connect)
    opening = asyncio.create_task(env.open())
    await asyncio.sleep(0.05)  # Connected, and reading the role's defaults.

    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening

    assert conn.closed


@pytest.mark.anyio
async def test_role_defaults_an_earlier_session_changed_are_restored_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    changed: list[tuple[Any, ...]] = [
        (None, ["default_transaction_read_only=off", "temp_file_limit=5GB"]),
        ("template1", ["work_mem=1GB"]),
    ]  # The session lifted read-only and both timeouts, and set work_mem for template1.
    first, second = FakeConn(defaults=changed), FakeConn()
    connections = [first, second]

    async def connect() -> Any:
        return connections.pop(0)

    env = PgSqlEnvironment(SOLVING[0], asyncio.Semaphore(2), SqlEnvOptions(), connect)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await env.open()

    assert first.closed
    assert first.executed[1:] == [
        "BEGIN",
        "SET TRANSACTION READ WRITE",
        'ALTER ROLE CURRENT_USER IN DATABASE "template1" RESET ALL',
        "ALTER ROLE CURRENT_USER RESET ALL",  # It leaves the superuser-only temp_file_limit.
        "ALTER ROLE CURRENT_USER SET \"default_transaction_read_only\" = 'on'",
        "ALTER ROLE CURRENT_USER SET \"statement_timeout\" = '130s'",
        "ALTER ROLE CURRENT_USER SET \"client_connection_check_interval\" = '5s'",
        "COMMIT",
    ]
    assert connections == []  # The second, clean session is the one in use.
    assert not second.closed
    assert _logged(caplog, "restoring the agent role's defaults on solve-1").levelno == 30


@pytest.mark.anyio
async def test_defaults_the_agent_role_cannot_restore_fail_the_session() -> None:
    lost_cap: list[tuple[Any, ...]] = [(None, ROLE_WIDE[:3])]  # temp_file_limit is gone.
    first, second = FakeConn(defaults=lost_cap), FakeConn(defaults=lost_cap)
    connections = [first, second]

    async def connect() -> Any:
        return connections.pop(0)

    env = PgSqlEnvironment(SOLVING[0], asyncio.Semaphore(2), SqlEnvOptions(), connect)

    with pytest.raises(EnvironmentFailure, match=r"cannot restore them.*pgenv\.py start"):
        await env.open()
    assert first.closed
    assert second.closed


@pytest.mark.anyio
async def test_close_stops_a_statement_still_in_flight() -> None:
    scenario = Scenario("cancellable")
    env = _environment(scenario)
    await env.open()
    running = asyncio.create_task(env.execute("SELECT pg_sleep(60)", timeout_s=60))
    await asyncio.sleep(0.05)

    await env.close()

    assert scenario.connections[0].cancelled.is_set()  # Its backend does not run on,
    assert scenario.connections[0].closed
    assert "canceling statement" in (await running).error  # and its caller gets a result.


@pytest.mark.anyio
async def test_close_stops_even_a_statement_nothing_stops_within_its_bound() -> None:
    scenario = Scenario("stubborn")
    env = _environment(scenario, grace=0.05)
    await env.open()
    running = asyncio.create_task(env.execute("SELECT stuck()", timeout_s=60))  # Left behind.
    await asyncio.sleep(0.05)

    async with asyncio.timeout(1):  # The grace periods, not the statement's 60 s.
        await env.close()

    assert scenario.connections[0].cancelled.is_set()  # The cancel key came first,
    assert scenario.connections[0].closed  # and the socket closed at the end.
    assert (await running).error.startswith("the session ended")


def test_the_trusted_timeout_stays_below_the_servers_backstop() -> None:
    assert SqlEnvOptions(timeout_s=129).timeout_s == 129
    with pytest.raises(ValueError, match="less than 130"):
        SqlEnvOptions(timeout_s=130)


@pytest.mark.anyio
async def test_close_completes_even_when_its_caller_is_cancelled() -> None:
    scenario = Scenario()
    env = _environment(scenario)
    await env.open()
    scenario.connections[0].close_delay = 0.05

    closing = asyncio.create_task(env.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    await asyncio.sleep(0.1)

    assert scenario.connections[0].closed


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
    cap = asyncio.Semaphore(2)
    envs = [_environment(scenario, cap) for _ in range(3)]
    for env in envs:
        await env.open()

    results = await asyncio.gather(*(env.execute("SELECT 1", timeout_s=5) for env in envs))

    waits = sorted(result.queue_wait_s for result in results)
    assert waits[1] < 0.1
    assert waits[2] >= 0.15  # The third waited for a slot,
    assert all(result.elapsed_s < 0.4 for result in results)  # which its elapsed_s leaves out.


AGENT = AgentCredentials(agent_password=SecretStr("b"), _env_file=None)


def _fake_connect(
    monkeypatch: pytest.MonkeyPatch, answers: Mapping[int, object] | None = None
) -> list[dict[str, object]]:
    """Make psycopg's connect hand out FakeConns, or the answer for a port; record each call."""
    opened: list[dict[str, object]] = []

    async def connect(**kwargs: object) -> Any:
        opened.append(kwargs)
        answer = (answers or {}).get(int(str(kwargs["port"])))
        if isinstance(answer, BaseException):
            raise answer
        return answer if answer is not None else FakeConn()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", connect)
    return opened


@pytest.mark.anyio
async def test_the_pool_assigns_the_least_loaded_instance_and_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = _fake_connect(monkeypatch)
    pool = SolvingPool(PgEnvConfig(), AGENT)

    async with pool.environment() as first, pool.environment() as second:
        assert {first.instance.name, second.instance.name} == {"solve-1", "solve-2"}
        assert pool.load() == {"solve-1": 1, "solve-2": 1}
        async with pool.environment() as third:
            assert third.instance.name == "solve-1"  # Ties go to the first instance.
    assert pool.load() == {"solve-1": 0, "solve-2": 0}
    assert {o["user"] for o in opened} == {"effisql_agent"}
    assert {o["password"] for o in opened} == {"b"}  # The agent's, the only one it holds.
    assert {o["dbname"] for o in opened} == {"tpch"}
    assert {o["host"] for o in opened} == {"127.0.0.1"}
    assert all(o["autocommit"] and o["connect_timeout"] == 10 for o in opened)


@pytest.mark.anyio
async def test_the_pool_connects_through_the_socket_directories_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = _fake_connect(monkeypatch)
    sockets = {"solve-1": Path("/run/pg/solve-1"), "solve-2": Path("/run/pg/solve-2")}
    pool = SolvingPool(PgEnvConfig(), AGENT, socket_dirs=sockets)

    async with pool.environment(), pool.environment():
        pass

    assert sorted((str(o["host"]), str(o["port"])) for o in opened) == [
        ("/run/pg/solve-1", "55432"),
        ("/run/pg/solve-2", "55433"),
    ]


@pytest.mark.parametrize(
    "sockets",
    [{"solve-1": Path("/run/pg/solve-1")}, {"solve-1": Path("/a"), "solve-2": Path("rel")}],
)
def test_the_pool_needs_an_absolute_socket_directory_for_every_instance(
    sockets: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="absolute socket directory"):
        SolvingPool(PgEnvConfig(), AGENT, socket_dirs=sockets)


def test_the_pool_refuses_credentials_beyond_the_agents() -> None:
    everything = PgCredentials(
        admin_password=SecretStr("a"),
        agent_password=SecretStr("b"),
        grader_password=SecretStr("c"),
        _env_file=None,
    )

    with pytest.raises(TypeError, match="AgentCredentials"):
        SolvingPool(PgEnvConfig(), everything)  # type: ignore[arg-type]
    SolvingPool(PgEnvConfig(), everything.agent_only())  # Its agent part alone is fine.


@pytest.mark.anyio
async def test_an_environment_is_released_even_when_its_close_is_cut_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_connect(monkeypatch, {55432: FakeConn(close_delay=0.2)})
    pool = SolvingPool(PgEnvConfig(), AGENT, instances=SOLVING[:1])

    async def hold() -> None:
        async with pool.environment():
            await asyncio.sleep(60)

    task = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    task.cancel()  # The block exits, and close() starts its 0.2 s close.
    await asyncio.sleep(0.05)
    task.cancel()  # A second cancel, during that close.
    with pytest.raises(asyncio.CancelledError):
        await task

    assert pool.load() == {"solve-1": 0}


class PreflightConn(FakeConn):
    def __init__(self, row: tuple[object, ...]) -> None:
        super().__init__()
        self.row = row

    async def execute(self, query: object, params: object = None) -> FakeCursor:
        return FakeCursor([self.row])


READY = (True, ["on", "5GB", "130s", "5s"])


@pytest.mark.anyio
async def test_preflight_passes_when_every_solving_instance_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = {55432: PreflightConn(READY), 55433: PreflightConn(READY)}
    _fake_connect(monkeypatch, ready)

    await SolvingPool(PgEnvConfig(), AGENT).preflight()

    assert all(conn.closed for conn in ready.values())


@pytest.mark.anyio
async def test_preflight_fails_fast_naming_every_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_connect(
        monkeypatch,
        {
            55432: psycopg.OperationalError("password authentication failed"),
            55433: PreflightConn((False, ["off", "-1", "0", "5s"])),
        },
    )

    with pytest.raises(EnvironmentFailure) as caught:
        await SolvingPool(PgEnvConfig(), AGENT).preflight()

    message = str(caught.value)
    assert "solve-1 (port 55432): the agent role cannot log in: password" in message
    assert "solve-2 (port 55433): the agent's default_transaction_read_only is off, not on" in (
        message
    )
    assert "solve-2 (port 55433): the agent's temp_file_limit is -1, not 5GB" in message
    assert "solve-2 (port 55433): the agent's statement_timeout is 0, not 130s" in message
    assert "client_connection_check_interval" not in message  # That one is right.
    assert "solve-2 (port 55433): the agent role cannot read the TPC-H tables" in message
    assert "pgenv.py start" in message


class FailingConn(FakeConn):
    async def execute(self, query: object, params: object = None) -> FakeCursor:
        raise psycopg.errors.InsufficientPrivilege("permission denied for table lineitem")


@pytest.mark.anyio
async def test_preflight_names_a_session_that_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_connect(monkeypatch, {55432: PreflightConn(READY), 55433: FailingConn()})

    with pytest.raises(EnvironmentFailure, match=r"solve-2 .*: the agent's session failed"):
        await SolvingPool(PgEnvConfig(), AGENT).preflight()
