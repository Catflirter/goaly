"""The agent's SQL environment on the solving instances (spec §4.1, §6.2).

:class:`SolvingPool` gives each task a :class:`PgSqlEnvironment` on the least-loaded solving
instance. The environment holds the task's own connection, as the read-only agent role, from
provisioning until ``close()``. No other task ever uses that connection, so session settings
never leak from one task to another.

Statements stream through libpq's chunked-rows mode on the simple query protocol:

- the server runs them as it runs any client's query, so a parallel plan stays parallel; a
  DECLARE CURSOR would forbid one;
- every row is counted, so ``row_count`` is honest, but only the first ``row_limit`` rows are
  decoded and kept, so a 60M-row result costs the memory of 100 rows;
- ``elapsed_s`` therefore includes transferring the rows to the client, which EXPLAIN
  ANALYZE's "Execution Time" leaves out.

Each instance runs at most ``statement_cap`` statements at once. The wait for a slot is
``queue_wait_s``, and ``elapsed_s`` leaves it out. A session can change its own
``statement_timeout``, so a statement that runs past its timeout is cancelled from the client:

1. libpq's cancel request;
2. if it is still running after a grace period, its backend is terminated, and the next
   statement opens a new connection.

A cancelled ``execute`` stops its statement the same way before CancelledError propagates.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import psycopg
from google.protobuf.struct_pb2 import NULL_VALUE, Value
from psycopg import generators
from psycopg.pq import DiagnosticField, ExecStatus
from pydantic import BaseModel, ConfigDict, Field

from goaly.bench.effisql.pg_instances import SOLVING, Instance
from goaly.bench.effisql.pg_settings import (
    ADMIN_ROLE,
    AGENT_ROLE,
    MAIN_DB,
    PgCredentials,
    PgEnvConfig,
)
from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.environment_pb2 import SqlResult

__all__ = [
    "STATEMENT_TIMEOUT",
    "PgSqlEnvironment",
    "ResultBuilder",
    "SolvingPool",
    "SqlEnvOptions",
    "connection_lost",
    "encode_value",
    "stream_statements",
]

# PostgreSQL's own message for a statement that ran past statement_timeout.
STATEMENT_TIMEOUT = "canceling statement due to statement timeout"

_log = logging.getLogger(__name__)

_CHUNK_ROWS = 10_000
_INT32_MAX = 2**31 - 1
_MAX_EXACT_INTEGER = 2**53
_BOOL, _NUMERIC = 16, 1700
_INTEGERS = frozenset({20, 21, 23, 26})  # int8, int2, int4, oid
_FLOATS = frozenset({700, 701})  # float4, float8
_NON_FINITE = frozenset({"NaN", "Infinity", "-Infinity"})
# Errors after which the session is gone: connection exceptions and server shutdowns.
_LOST = frozenset({"57P01", "57P02", "57P03"})


class SqlEnvOptions(BaseModel):
    """How the agent's SQL environment runs statements; the defaults are the spec's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_s: float = Field(default=120.0, gt=0)  # The agent's statement timeout (T11 passes it).
    row_limit: int = Field(default=100, ge=0)  # Rows the agent sees; row_count counts them all.
    statement_cap: int = Field(default=2, ge=1)  # Statements in flight per solving instance.
    cancel_grace_s: float = Field(default=5.0, gt=0)  # How long a cancel may take to work.


def encode_value(raw: bytes | None, type_oid: int) -> Value:
    """Encode a value as the server printed it, typed as the SqlResult contract says.

    SQL NULL is null; booleans are booleans; integers within ±2^53 and finite floats are
    numbers. Larger integers, numeric values, NaN and infinities are strings, and so is every
    other type, in the server's text form.
    """
    if raw is None:
        return Value(null_value=NULL_VALUE)
    text = raw.decode("utf-8", errors="replace")
    if type_oid == _BOOL:
        return Value(bool_value=text == "t")
    if type_oid in _INTEGERS and abs(int(text)) <= _MAX_EXACT_INTEGER:
        return Value(number_value=int(text))
    if type_oid in _FLOATS and text not in _NON_FINITE:
        return Value(number_value=float(text))
    return Value(string_value=text)


def connection_lost(sqlstate: str | None) -> bool:
    """Say whether an error with ``sqlstate`` ended the session, not just the statement."""
    return sqlstate is not None and (sqlstate.startswith("08") or sqlstate in _LOST)


class _Result(Protocol):
    """The parts of a libpq result (psycopg.pq.PGresult) that the builder reads."""

    @property
    def status(self) -> int: ...
    @property
    def ntuples(self) -> int: ...
    @property
    def nfields(self) -> int: ...
    @property
    def command_tuples(self) -> int | None: ...
    @property
    def error_message(self) -> bytes: ...
    def fname(self, column_number: int) -> bytes | None: ...
    def ftype(self, column_number: int) -> int: ...
    def get_value(self, row_number: int, column_number: int) -> bytes | None: ...
    def error_field(self, fieldcode: DiagnosticField) -> bytes | None: ...


class ResultBuilder:
    """Folds the results of one statement string, as libpq returns them, into the last
    statement's SqlResult. Rows past ``row_limit`` are only counted, never decoded."""

    def __init__(self, row_limit: int) -> None:
        self._limit = row_limit
        self._finished = SqlResult()  # The last statement that finished.
        self._current: SqlResult | None = None  # The statement whose rows are arriving.
        self._count = 0
        self.sqlstate: str | None = None  # The SQLSTATE of the error, if one ended the string.

    def add(self, res: _Result) -> None:
        status = res.status
        if status in (ExecStatus.TUPLES_CHUNK, ExecStatus.SINGLE_TUPLE):
            current = self._start(res)
            for row in range(min(res.ntuples, self._limit - len(current.rows))):
                values = current.rows.add().values
                values.extend(
                    encode_value(res.get_value(row, c), res.ftype(c)) for c in range(res.nfields)
                )
            self._count += res.ntuples
        elif status == ExecStatus.TUPLES_OK:
            current = self._start(res)
            current.row_count = min(self._count, _INT32_MAX)
            current.truncated = self._count > len(current.rows)
            self._finish(current)
        elif status == ExecStatus.COMMAND_OK:
            self._finish(SqlResult(row_count=min(res.command_tuples or 0, _INT32_MAX)))
        elif status == ExecStatus.EMPTY_QUERY:
            self._finish(SqlResult())
        else:  # FATAL_ERROR, or a result this environment does not support.
            sqlstate = res.error_field(DiagnosticField.SQLSTATE)
            self.sqlstate = sqlstate.decode() if sqlstate else None
            message = res.error_message.decode("utf-8", errors="replace").strip()
            self.fail(message or f"unsupported result: {ExecStatus(status).name}")

    def fail(self, message: str) -> None:
        """End the string's result with an error, as when a statement fails."""
        self._finish(SqlResult(error=message))

    def result(self) -> SqlResult:
        result = SqlResult()
        result.CopyFrom(self._finished)
        return result

    def _start(self, res: _Result) -> SqlResult:
        if self._current is None:
            self._current = SqlResult(
                columns=[
                    (res.fname(c) or b"?").decode("utf-8", errors="replace")
                    for c in range(res.nfields)
                ]
            )
        return self._current

    def _finish(self, result: SqlResult) -> None:
        self._finished, self._current, self._count = result, None, 0


async def stream_statements(
    conn: psycopg.AsyncConnection[Any], sql: str, builder: ResultBuilder
) -> None:
    """Run ``sql``, one statement or more, and feed every result to ``builder``.

    The simple query protocol in chunked-rows mode: rows reach the client a chunk at a time,
    and every statement's results arrive in turn. This uses psycopg's generators, which
    uv.lock pins; the integration tests cover it.
    """
    pgconn = conn.pgconn
    async with conn.lock:
        pgconn.send_query(sql.encode())
        pgconn.set_chunked_rows_mode(_CHUNK_ROWS)
        await conn.wait(generators.send(pgconn))
        while (res := await conn.wait(generators.fetch(pgconn))) is not None:
            if res.status in (ExecStatus.COPY_OUT, ExecStatus.COPY_BOTH):
                while isinstance(await conn.wait(generators.copy_from(pgconn)), memoryview):
                    pass  # Read and drop the copied data until the copy ends.
                builder.fail("COPY to the client is not supported here; run a SELECT instead")
            elif res.status == ExecStatus.COPY_IN:
                await conn.wait(
                    generators.copy_end(pgconn, b"COPY from the client is not supported")
                )
            else:
                builder.add(res)


type _Runner = Callable[
    [psycopg.AsyncConnection[Any], str, ResultBuilder], Coroutine[Any, Any, None]
]


@dataclass
class _Slot:
    """A solving instance: its statement cap and how many environments it holds."""

    instance: Instance
    cap: asyncio.Semaphore
    environments: int = 0


class PgSqlEnvironment:
    """One task's read-only session on a solving instance; see the module docstring."""

    def __init__(
        self,
        slot: _Slot,
        options: SqlEnvOptions,
        connect: Callable[[], Awaitable[psycopg.AsyncConnection[Any]]],
        terminate: Callable[[int], Awaitable[None]],
        runner: _Runner = stream_statements,
    ) -> None:
        self._slot = slot
        self._options = options
        self._connect = connect
        self._terminate = terminate
        self._runner = runner
        self._conn: psycopg.AsyncConnection[Any] | None = None
        self._lock = asyncio.Lock()  # One statement at a time on the task's connection.
        self._closed = False

    @property
    def instance(self) -> Instance:
        return self._slot.instance

    async def open(self) -> None:
        """Open the task's connection; raise EnvironmentFailure if the instance refuses it."""
        self._conn = await self._open_connection()

    async def execute(self, sql: str, *, timeout_s: float) -> SqlResult:
        if self._closed:
            raise RuntimeError("the SQL environment is closed")
        async with self._lock:
            queued = time.monotonic()
            async with self._slot.cap:
                return await self._run(sql, timeout_s, time.monotonic() - queued)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        conn, self._conn = self._conn, None
        if conn is not None:
            await asyncio.shield(_close_quietly(conn))

    async def _run(self, sql: str, timeout_s: float, queue_wait_s: float) -> SqlResult:
        conn = self._conn if self._conn is not None else await self._open_connection()
        self._conn = conn
        builder = ResultBuilder(self._options.row_limit)
        started = time.monotonic()
        task = asyncio.create_task(self._runner(conn, sql, builder))
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout_s)
        except asyncio.CancelledError:
            await self._stop_fully(conn, task)
            raise
        elapsed_s = time.monotonic() - started
        if not done:
            await self._stop_fully(conn, task)
            return SqlResult(
                error=STATEMENT_TIMEOUT, elapsed_s=elapsed_s, queue_wait_s=queue_wait_s
            )
        failure = task.exception()
        if failure is not None or connection_lost(builder.sqlstate) or conn.closed:
            self._conn = None
            await _close_quietly(conn)
            reason = str(failure) if failure is not None else builder.result().error
            raise EnvironmentFailure(
                f"the connection to {self.instance.name} failed: {reason}"
            ) from failure
        result = builder.result()
        result.elapsed_s, result.queue_wait_s = elapsed_s, queue_wait_s
        return result

    async def _stop_fully(
        self, conn: psycopg.AsyncConnection[Any], task: asyncio.Task[None]
    ) -> None:
        """Stop the statement ``task`` runs, even if this coroutine is cancelled meanwhile."""
        stopping = asyncio.ensure_future(self._stop(conn, task))
        cancelled = False
        while not stopping.done():
            try:
                await asyncio.shield(stopping)
            except asyncio.CancelledError:
                cancelled = True
        stopping.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _stop(self, conn: psycopg.AsyncConnection[Any], task: asyncio.Task[None]) -> None:
        grace = self._options.cancel_grace_s
        with contextlib.suppress(psycopg.Error):
            await conn.cancel_safe(timeout=grace)
        if await _finished(task, grace):
            return
        # The cancel did not take: end the backend, and open a new connection next time.
        self._conn = None
        pid = conn.info.backend_pid
        _log.warning("terminating a backend that ignored its cancel", extra={"pid": pid})
        with contextlib.suppress(Exception):
            await self._terminate(pid)
        if not await _finished(task, grace):
            _log.error(
                "abandoning a statement that cancel and terminate did not stop", extra={"pid": pid}
            )
            conn.pgconn.finish()

    async def _open_connection(self) -> psycopg.AsyncConnection[Any]:
        try:
            conn = await self._connect()
            if await _reset_role_defaults(conn):
                # This session started with the old defaults: only a new one starts clean.
                await _close_quietly(conn)
                conn = await self._connect()
        except psycopg.Error as error:
            raise EnvironmentFailure(
                f"cannot open a session on {self.instance.name}: {error}"
            ) from error
        return conn


async def _reset_role_defaults(conn: psycopg.AsyncConnection[Any]) -> bool:
    """Undo defaults an earlier task set on the agent role; say whether there were any.

    A role may ALTER ROLE itself, which would change every later session. The only default the
    agent role should carry is ``default_transaction_read_only = on``.
    """
    cursor = await conn.execute(
        "SELECT count(*) FROM pg_db_role_setting WHERE setrole = "
        "(SELECT oid FROM pg_roles WHERE rolname = current_user) "
        "AND setconfig <> ARRAY['default_transaction_read_only=on']"
    )
    row = await cursor.fetchone()
    if row is None or row[0] == 0:
        return False
    _log.warning("resetting the agent role's defaults, which an earlier task changed")
    async with conn.transaction():
        await conn.execute("SET TRANSACTION READ WRITE")
        await conn.execute("ALTER ROLE CURRENT_USER RESET ALL")
        for db in ("tpch", "tpch_alt", "postgres"):
            await conn.execute(f"ALTER ROLE CURRENT_USER IN DATABASE {db} RESET ALL")
        await conn.execute("ALTER ROLE CURRENT_USER SET default_transaction_read_only = on")
    return True


async def _finished(task: asyncio.Task[None], timeout: float) -> bool:
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if done:
        task.exception()  # Retrieved, so asyncio does not log it as never retrieved.
    return bool(done)


async def _close_quietly(conn: psycopg.AsyncConnection[Any]) -> None:
    with contextlib.suppress(Exception):
        await conn.close()


class SolvingPool:
    """Gives each task its own SQL environment on the least-loaded solving instance."""

    def __init__(
        self,
        config: PgEnvConfig,
        credentials: PgCredentials,
        options: SqlEnvOptions | None = None,
        instances: Sequence[Instance] = SOLVING,
    ) -> None:
        self._config = config
        self._credentials = credentials
        self.options = options if options is not None else SqlEnvOptions()
        self._slots = [_Slot(i, asyncio.Semaphore(self.options.statement_cap)) for i in instances]

    def load(self) -> dict[str, int]:
        """How many environments each instance holds now."""
        return {slot.instance.name: slot.environments for slot in self._slots}

    @asynccontextmanager
    async def environment(self) -> AsyncIterator[PgSqlEnvironment]:
        """Provision a task's environment; it is closed however the block exits."""
        slot = min(self._slots, key=lambda candidate: candidate.environments)
        slot.environments += 1
        environment = PgSqlEnvironment(
            slot,
            self.options,
            connect=lambda: self._connect(slot.instance, AGENT_ROLE, MAIN_DB),
            terminate=lambda pid: self._terminate(slot.instance, pid),
        )
        try:
            await environment.open()
            yield environment
        finally:
            await environment.close()
            slot.environments -= 1

    async def _connect(
        self, instance: Instance, role: str, db: str
    ) -> psycopg.AsyncConnection[Any]:
        password = {
            AGENT_ROLE: self._credentials.agent_password,
            ADMIN_ROLE: self._credentials.admin_password,
        }[role]
        return await psycopg.AsyncConnection.connect(
            host=self._config.host,
            port=instance.port,
            user=role,
            password=password.get_secret_value(),
            dbname=db,
            autocommit=True,
            connect_timeout=10,
            application_name=f"goaly-effisql-{role}",
        )

    async def _terminate(self, instance: Instance, pid: int) -> None:
        async with await self._connect(instance, ADMIN_ROLE, "postgres") as admin:
            await admin.execute("SELECT pg_terminate_backend(%s)", [pid])
