"""The agent's SQL environment on the solving instances (spec §4.1, §6.2).

:class:`SolvingPool` gives each task a :class:`PgSqlEnvironment` on the least-loaded solving
instance. The environment holds the task's own connection, as the read-only agent role, from
provisioning until ``close()``; no other task ever uses it. The pool holds the agent role's
password and nothing else (spec §6.1, §8.5): no superuser or grader password reaches the
solving side. It connects over TCP, or through each instance's Unix-socket directory
(``socket_dirs``), as the solve container must, since it may not use IP at all (V28). Before a
run, T11 and T21 call :meth:`SolvingPool.preflight`, which fails fast if an instance is not
ready.

Nothing an agent statement does can persist. Each ``execute`` runs one statement, through the
extended query protocol, inside ``BEGIN READ ONLY`` … ``ROLLBACK``, followed by ``DISCARD ALL``.
A string with several statements fails with the server's error. This narrows the interface's
"one statement or more": a statement's effects last only for that call. Read-only itself comes
from the role's privileges; the transaction, the rollback and the discard add defense in depth.

Results:

- the server runs the statement as it runs any client's query, so a parallel plan stays
  parallel; a DECLARE CURSOR would forbid one;
- rows stream through libpq's chunked-rows mode, 500 at a time. Every row is counted, so
  ``row_count`` is honest, but only the first ``row_limit`` rows are decoded and kept, each
  value cut to 4 KiB with a note of what was left out, and at most 1 MiB of values in all;
  ``truncated`` says rows were left out. libpq still holds one chunk at a time, so a
  statement whose rows are each huge (PostgreSQL allows 1 GB per value) can take that much
  memory per row of a chunk;
- ``elapsed_s`` includes transferring the rows to the client, which EXPLAIN ANALYZE's
  "Execution Time" leaves out.

A statement that fails gives a result with ``error`` set. So does one that ends or breaks its
own session, which the agent role may do; the next call then opens a new session. Only an
instance that refuses a new session raises EnvironmentFailure.

Capacity. Each instance runs at most ``statement_cap`` statements at once from one process:
the semaphore lives in the process. Runs that share the solving instances, such as a lane and
a v0 held-out run, split that capacity through the runner's concurrency budgets, which give
each an explicit share (spec §6.2); this pool does not coordinate across processes. The wait
for a slot is ``queue_wait_s``, and ``elapsed_s`` leaves it out; both leave out opening a new
session.

Timeouts. A call's timeout is its ``timeout_s`` capped at ``SqlEnvOptions.timeout_s``, which
trusted wiring sets. A session can change its own ``statement_timeout``, so a statement that
runs past its timeout is stopped from the client:

1. libpq's cancel request, with the session's cancel key;
2. if it still runs after a grace period, the client closes the session's socket, and the next
   statement opens a new session. The server then ends the backend: a backend blocked sending
   rows fails at once, the agent role's sessions check their client every 5 s
   (client_connection_check_interval, which a statement could turn off for itself), and
   statement_timeout ends any statement at 130 s, a limit no running statement can lift.

No session ever ends another one: no role here may call pg_cancel_backend or
pg_terminate_backend, so one task cannot stop another task's, or another run's, statements.

A cancelled ``execute``, and ``close()`` with a statement in flight, stop the statement the
same way, within the grace period and psycopg's own cleanup, before they return or
CancelledError propagates. A cancel that fails is logged at WARNING with its cause, and so is
a session closed under its statement.

Other sessions. Every task shares the agent role, and PostgreSQL shows a role the query text of
its own sessions. So the instances revoke pg_stat_activity and the functions behind it from
PUBLIC, and the agent role is no member of pg_read_all_stats: a task cannot read another task's
statements. What privileges do not stop: within one statement, ``set_config`` can raise
settings such as ``work_mem`` for queries the statement itself runs later, through
``query_to_xml`` and the like. PL/pgSQL is revoked, so no DO block runs many.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import psycopg
from google.protobuf.struct_pb2 import NULL_VALUE, Value
from psycopg import generators
from psycopg.abc import PQGen
from psycopg.pq import DiagnosticField, ExecStatus, TransactionStatus
from psycopg.pq.abc import PGconn
from psycopg.sql import SQL, Identifier, Literal
from psycopg.waiting import Wait
from pydantic import BaseModel, ConfigDict, Field

from goaly.bench.effisql.pg_instances import (
    AGENT_STATEMENT_TIMEOUT_S,
    ROLE_DEFAULTS,
    SOLVING,
    Instance,
)
from goaly.bench.effisql.pg_settings import (
    AGENT_ROLE,
    MAIN_DB,
    AgentCredentials,
    PgEnvConfig,
    connection_kwargs,
)
from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.environment_pb2 import SqlResult

__all__ = [
    "STATEMENT_TIMEOUT",
    "PgSqlEnvironment",
    "ResultBuilder",
    "SolvingPool",
    "SqlEnvOptions",
    "capped_error",
    "connection_lost",
    "encode_value",
    "refuse_copy_data",
    "role_defaults_problems",
    "run_statement",
    "skip_copy_data",
    "statement_text_problem",
]

# PostgreSQL's own message for a statement that ran past statement_timeout.
STATEMENT_TIMEOUT = "canceling statement due to statement timeout"

_log = logging.getLogger(__name__)

_CHUNK_ROWS = 500  # Rows libpq gathers before handing them over: the memory one chunk takes.
_VALUE_BYTES = 4096  # The most of one value the agent sees; the rest is cut, with a note.
_RESULT_BYTES = 1 << 20  # The most bytes of values one result keeps; later rows are counted.
_INT32_MAX = 2**31 - 1
_MAX_EXACT_INTEGER = 2**53
_BOOL, _NUMERIC = 16, 1700
_INTEGERS = frozenset({20, 21, 23, 26})  # int8, int2, int4, oid
_FLOATS = frozenset({700, 701})  # float4, float8
_NON_FINITE = frozenset({"NaN", "Infinity", "-Infinity"})
# Errors after which the session is gone: connection exceptions, server shutdowns, a dropped
# database, and the idle-session and idle-in-transaction timeouts, which end the session.
_LOST = frozenset({"57P01", "57P02", "57P03", "57P04", "57P05", "25P03"})
# How long psycopg may take to let go of a session whose runner was cancelled: one more
# cancel request and a wait for the statement, up to 5 s each.
_CLOSE_WAIT_S = 15.0
_ERROR_BYTES = 8192  # The most of an error message the agent sees; the rest is cut, with a note.
_NO_COPY_OUT = "COPY to the client is not supported here; run a SELECT instead"
_NO_COPY_IN = b"COPY from the client is not supported here"

# Its parameter: the names of the role defaults to read, in order.
_PREFLIGHT_SQL = (
    "SELECT has_table_privilege('public.lineitem', 'SELECT'), "
    "ARRAY(SELECT current_setting(name) FROM unnest(%s::text[]) WITH ORDINALITY AS s(name, i) "
    "ORDER BY i)"
)
_ROLE_SETTINGS_SQL = (
    "SELECT d.datname, s.setconfig FROM pg_db_role_setting s "
    "LEFT JOIN pg_database d ON d.oid = s.setdatabase "
    "WHERE s.setrole = (SELECT oid FROM pg_roles WHERE rolname = current_user)"
)
# The role defaults only a superuser may set: the agent's RESET ALL leaves them, and only
# setup can restore them. The agent role restores every other one itself.
_SUPERUSER_ONLY = frozenset({"temp_file_limit"})


class SqlEnvOptions(BaseModel):
    """How the agent's SQL environment runs statements; the defaults are the spec's. Trusted
    wiring sets them; pack code only passes each call's ``timeout_s``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The longest a statement may run: a call's default, and the cap on what it asks for. It
    # stays below the agent role's server-side statement_timeout, the backstop behind it.
    timeout_s: float = Field(default=120.0, gt=0, lt=AGENT_STATEMENT_TIMEOUT_S, allow_inf_nan=False)
    row_limit: int = Field(default=100, ge=0)  # Rows the agent sees; row_count counts them all.
    statement_cap: int = Field(default=2, ge=1)  # Statements in flight per solving instance.
    cancel_grace_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)  # Per stop step.


def encode_value(raw: bytes | None, type_oid: int, *, max_bytes: int = _VALUE_BYTES) -> Value:
    """Encode a value as the server printed it, typed as the SqlResult contract says.

    SQL NULL is null; booleans are booleans; integers within ±2^53 and finite floats are
    numbers. Larger integers, numeric values, NaN and infinities are strings, and so is every
    other type, in the server's text form. A value longer than ``max_bytes`` is cut there and
    becomes a string that ends by saying how many bytes were left out.
    """
    if raw is None:
        return Value(null_value=NULL_VALUE)
    if len(raw) > max_bytes:
        head = raw[:max_bytes].decode("utf-8", errors="replace")
        return Value(string_value=f"{head}[... {len(raw) - max_bytes} more bytes]")
    text = raw.decode("utf-8", errors="replace")
    if type_oid == _BOOL:
        return Value(bool_value=text == "t")
    if type_oid in _INTEGERS and abs(int(text)) <= _MAX_EXACT_INTEGER:
        return Value(number_value=int(text))
    if type_oid in _FLOATS and text not in _NON_FINITE:
        return Value(number_value=float(text))
    return Value(string_value=text)


def capped_error(message: str | bytes, *, max_bytes: int = _ERROR_BYTES) -> str:
    """An error message as the agent sees it: stripped, and at most ``max_bytes`` long, a cut
    one ending with a note of how many bytes were left out. A server error can quote a value
    whole, up to its 1 GB. A message already cut comes back unchanged."""
    raw = message if isinstance(message, bytes) else message.encode("utf-8", errors="replace")
    raw = raw.strip()
    if len(raw) <= max_bytes:
        return raw.decode("utf-8", errors="replace")
    room = max(max_bytes - len(f"[truncated {len(raw)} bytes]"), 0)  # The note fits too.
    head = raw[:room].decode("utf-8", errors="ignore")  # No half character at the cut.
    return f"{head}[truncated {len(raw) - len(head.encode())} bytes]"


def connection_lost(sqlstate: str | None) -> bool:
    """Say whether an error with ``sqlstate`` ended the session, not just the statement."""
    return sqlstate is not None and (sqlstate.startswith("08") or sqlstate in _LOST)


def statement_text_problem(sql: str) -> str | None:
    """Why PostgreSQL could not take ``sql`` as statement text, or None if it can."""
    if "\x00" in sql:
        return "the statement contains a NUL character, which PostgreSQL does not accept"
    try:
        sql.encode("utf-8")
    except UnicodeEncodeError as error:
        return f"the statement is not valid UTF-8 at character {error.start}: {error.reason}"
    return None


def role_defaults_problems(rows: Sequence[Sequence[Any]]) -> list[str]:
    """What differs from the setup's ROLE_DEFAULTS in the agent role's stored defaults.

    ``rows`` are pg_db_role_setting's (database name, or None for the role-wide row; setconfig)
    pairs for the role. The setup leaves one role-wide row, exactly ROLE_DEFAULTS, and no other.
    """
    expected = sorted(f"{name}={value}" for name, value in ROLE_DEFAULTS[AGENT_ROLE].items())
    problems = []
    role_wide: list[str] | None = None
    for db, config in rows:
        entries = sorted(str(entry) for entry in config or ())
        if db is None:
            role_wide = entries
        else:
            problems.append(f"defaults in database {db}: {', '.join(entries)}")
    if role_wide != expected:
        problems.append(f"role-wide defaults {role_wide or []}, not {expected}")
    return problems


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
    """Folds the results of a statement, as libpq returns them, into its SqlResult.

    Every row is counted. Only the first ``row_limit`` rows are decoded and kept, each value cut
    to ``value_bytes``, until the kept values hold ``result_bytes``: later rows are only counted.
    Of a string of statements, the last one's result is kept.
    """

    def __init__(
        self,
        row_limit: int,
        *,
        value_bytes: int = _VALUE_BYTES,
        result_bytes: int = _RESULT_BYTES,
    ) -> None:
        self._limit = row_limit
        self._value_bytes = value_bytes
        self._result_bytes = result_bytes
        self._finished = SqlResult()  # The last statement that finished.
        self._current: SqlResult | None = None  # The statement whose rows are arriving.
        self._count = 0
        self._bytes = 0
        self.sqlstate: str | None = None  # The SQLSTATE of the error, if one ended the string.

    def add(self, res: _Result) -> None:
        status = res.status
        if status in (ExecStatus.TUPLES_CHUNK, ExecStatus.SINGLE_TUPLE, ExecStatus.TUPLES_OK):
            current = self._start(res)
            self._keep(current, res)  # A TUPLES_OK holds rows too when no chunking applied.
            self._count += res.ntuples
            if status == ExecStatus.TUPLES_OK:  # The statement's last result.
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
            message = capped_error(res.error_message)  # It can quote a huge value whole.
            self.fail(message or f"unsupported result: {ExecStatus(status).name}")

    def fail(self, message: str) -> None:
        """End the string's result with an error, as when a statement fails."""
        self._finish(SqlResult(error=capped_error(message)))

    def result(self) -> SqlResult:
        result = SqlResult()
        result.CopyFrom(self._finished)
        return result

    def _keep(self, current: SqlResult, res: _Result) -> None:
        for row in range(res.ntuples):
            if len(current.rows) >= self._limit or self._bytes >= self._result_bytes:
                return
            values = current.rows.add().values
            for column in range(res.nfields):
                raw = res.get_value(row, column)
                self._bytes += min(len(raw), self._value_bytes) if raw is not None else 0
                values.append(encode_value(raw, res.ftype(column), max_bytes=self._value_bytes))

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
        self._finished, self._current, self._count, self._bytes = result, None, 0, 0


async def run_statement(
    conn: psycopg.AsyncConnection[Any], sql: str, builder: ResultBuilder
) -> None:
    """Run the agent's one statement in a read-only transaction that is then rolled back.

    ``sql`` must be valid statement text (see :func:`statement_text_problem`). It goes through
    the extended query protocol, which takes exactly one statement; a string with more fails
    with the server's error. Its rows reach the client a chunk at a time. A COPY gives an error
    result: its data is read and dropped, or, for a COPY from the client, refused. ROLLBACK and
    DISCARD ALL then undo everything a statement can leave in a session: settings, prepared
    statements, cursors, temporary tables, advisory locks and listens. A DISCARD ALL first also
    cleans up after a call whose own cleanup a cancel cut short.

    This drives libpq through psycopg's generators, which uv.lock pins; the integration tests
    cover it.
    """
    pgconn = conn.pgconn
    async with conn.lock:
        if pgconn.transaction_status != TransactionStatus.IDLE:
            await _control(conn, b"ROLLBACK")
        await _control(conn, b"DISCARD ALL")
        await _control(conn, b"BEGIN READ ONLY")
        pgconn.send_query_params(sql.encode(), None)
        pgconn.set_chunked_rows_mode(_CHUNK_ROWS)
        await conn.wait(generators.send(pgconn))
        copied = False
        while (res := await conn.wait(generators.fetch(pgconn))) is not None:
            if res.status in (ExecStatus.COPY_OUT, ExecStatus.COPY_BOTH):
                await conn.wait(skip_copy_data(pgconn))
                copied = True  # The copy's own result comes next.
            elif res.status == ExecStatus.COPY_IN:
                await conn.wait(refuse_copy_data(pgconn))  # The server's error comes next.
            elif copied and res.status == ExecStatus.COMMAND_OK:
                builder.fail(_NO_COPY_OUT)
            else:
                builder.add(res)  # A COPY that failed keeps the server's message.
        await _control(conn, b"ROLLBACK")
        await _control(conn, b"DISCARD ALL")


def skip_copy_data(pgconn: PGconn) -> PQGen[None]:
    """Read a COPY's data and drop it until the copy ends; its result then comes from fetch.

    Unlike psycopg's copy_from, it never gathers the results that follow, nor raises for them.
    """
    while True:
        nbytes, _ = pgconn.get_copy_data(1)  # Without blocking: 0 means nothing has arrived.
        if nbytes < 0:  # The copy ended. (libpq's -2, an error, raises in psycopg.)
            return
        if nbytes == 0:
            while not (yield Wait.R):
                continue
            pgconn.consume_input()


def refuse_copy_data(pgconn: PGconn) -> PQGen[None]:
    """End a COPY from the client with a failure; the server's error result then follows."""
    while pgconn.put_copy_end(_NO_COPY_IN) == 0:
        while not (yield Wait.W):
            continue
    while pgconn.flush() != 0:
        while not (yield Wait.W):
            continue


async def _control(conn: psycopg.AsyncConnection[Any], command: bytes) -> None:
    """Run one of the environment's own commands, which must succeed."""
    pgconn = conn.pgconn
    pgconn.send_query(command)
    await conn.wait(generators.send(pgconn))
    while (res := await conn.wait(generators.fetch(pgconn))) is not None:
        if res.status == ExecStatus.FATAL_ERROR:
            message = capped_error(res.error_message)
            raise psycopg.OperationalError(f"{command.decode()} failed: {message}")


type _Runner = Callable[
    [psycopg.AsyncConnection[Any], str, ResultBuilder], Coroutine[Any, Any, None]
]
type _Connect = Callable[[], Awaitable[psycopg.AsyncConnection[Any]]]
type _Conn = psycopg.AsyncConnection[Any]


class PgSqlEnvironment:
    """One task's read-only session on a solving instance, under the instance's statement cap
    ``cap``; see the module docstring. :class:`SolvingPool` builds it."""

    def __init__(
        self,
        instance: Instance,
        cap: asyncio.Semaphore,
        options: SqlEnvOptions,
        connect: _Connect,
        runner: _Runner = run_statement,
    ) -> None:
        self._instance = instance
        self._cap = cap
        self._options = options
        self._connect = connect
        self._runner = runner
        self._conn: _Conn | None = None
        self._running: asyncio.Task[None] | None = None  # The statement in flight, if any.
        self._stopping: tuple[asyncio.Task[None], asyncio.Future[None]] | None = None
        self._lock = asyncio.Lock()  # One statement at a time on the task's session.
        self._closed = False

    @property
    def instance(self) -> Instance:
        return self._instance

    async def open(self) -> None:
        """Open the task's session; raise EnvironmentFailure if the instance refuses it."""
        async with self._lock:
            await self._session()

    async def execute(self, sql: str, *, timeout_s: float | None = None) -> SqlResult:
        """Run one statement and return its result; see SqlEnvironment.execute.

        ``timeout_s`` defaults to ``SqlEnvOptions.timeout_s`` and is capped at it; it must be
        positive. Raise EnvironmentFailure only if the instance refuses a new session.
        """
        if self._closed:
            raise RuntimeError("the SQL environment is closed")
        timeout = self._timeout(timeout_s)
        if (problem := statement_text_problem(sql)) is not None:
            return SqlResult(error=problem)
        async with self._lock:
            conn = await self._session()  # Outside the cap: not a statement's time.
            queued = time.monotonic()
            async with self._cap:
                return await self._run(conn, sql, timeout, time.monotonic() - queued)

    async def close(self) -> None:
        """Stop any statement in flight and close the session; it completes even while its
        caller is cancelled. Calling it again does nothing."""
        if self._closed:
            return
        self._closed = True
        await asyncio.shield(self._shut_down())

    def _was_closed(self) -> bool:
        return self._closed  # A call, so a type checker does not assume it unchanged by an await.

    def _timeout(self, timeout_s: float | None) -> float:
        cap = self._options.timeout_s
        if timeout_s is None:
            return cap
        if not timeout_s > 0:  # Also refuses NaN.
            raise ValueError(f"timeout_s must be positive, not {timeout_s}")
        return min(timeout_s, cap)

    async def _shut_down(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        running = self._running
        if running is not None and not running.done():
            await self._stopper(conn, running)  # Its backend must not run on, uncapped.
        await _close_quietly(conn)

    async def _session(self) -> _Conn:
        """The task's session, opened anew if it has none or its connection broke."""
        if self._closed:
            raise RuntimeError("the SQL environment is closed")
        if self._conn is not None and not self._conn.closed:
            return self._conn
        if self._conn is not None:
            await _close_quietly(self._conn)
            self._conn = None
        conn = await self._open()
        if self._was_closed():  # close() ran while the session opened: it must not outlive it.
            await asyncio.shield(_close_quietly(conn))
            raise RuntimeError("the SQL environment was closed while its session opened")
        self._conn = conn
        return conn

    async def _run(self, conn: _Conn, sql: str, timeout_s: float, queue_wait_s: float) -> SqlResult:
        builder = ResultBuilder(self._options.row_limit)
        started = time.monotonic()
        task = asyncio.create_task(self._runner(conn, sql, builder))
        self._running = task
        try:
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
        finally:
            self._running = None
        failure = _failure(task)
        if failure is not None or conn.closed or connection_lost(builder.sqlstate):
            reason = capped_error(str(failure)) if failure is not None else builder.result().error
            await self._drop(conn, failure)
            return SqlResult(
                error=capped_error(
                    f"the session ended: {reason or 'its connection closed'}; the next "
                    "statement opens a new one"
                ),
                elapsed_s=elapsed_s,
                queue_wait_s=queue_wait_s,
            )
        result = builder.result()
        result.elapsed_s, result.queue_wait_s = elapsed_s, queue_wait_s
        return result

    async def _drop(self, conn: _Conn, failure: BaseException | None) -> None:
        """Forget a session that ended or broke, so the next statement opens a new one."""
        if self._conn is conn:
            self._conn = None
        await _close_quietly(conn)
        if failure is not None and not isinstance(
            failure, psycopg.OperationalError | asyncio.CancelledError
        ):
            _log.warning(
                "a statement on %s broke its session", self._instance.name, exc_info=failure
            )

    async def _stop_fully(self, conn: _Conn, task: asyncio.Task[None]) -> None:
        """Stop the statement ``task`` runs, even if this coroutine is cancelled meanwhile."""
        stopping = self._stopper(conn, task)
        cancelled = False
        while not stopping.done():
            try:
                await asyncio.shield(stopping)
            except asyncio.CancelledError:
                cancelled = True
        stopping.result()
        if cancelled:
            raise asyncio.CancelledError

    def _stopper(self, conn: _Conn, task: asyncio.Task[None]) -> asyncio.Future[None]:
        """The one stop of ``task``'s statement, shared by everyone who asks for it."""
        if self._stopping is None or self._stopping[0] is not task:
            self._stopping = (task, asyncio.ensure_future(self._stop(conn, task)))
        return self._stopping[1]

    async def _stop(self, conn: _Conn, task: asyncio.Task[None]) -> None:
        name, grace = self._instance.name, self._options.cancel_grace_s
        try:
            await conn.cancel_safe(timeout=grace)
        except psycopg.Error:
            _log.warning("the cancel request to %s failed", name, exc_info=True)
        if await _finished(task, grace):
            return
        # The cancel did not take: close the socket, which the server notices within 5 s, and
        # open a new session next time. Cancelling the runner first lets psycopg let go of the
        # socket itself: it tries one more cancel, then closes the connection.
        if self._conn is conn:
            self._conn = None
        pid = conn.info.backend_pid
        _log.warning("closing the session of backend %s on %s, which ignored its cancel", pid, name)
        task.cancel()
        if not await _finished(task, _CLOSE_WAIT_S):
            _log.error("the statement on %s kept its session; closing its socket under it", name)
            conn.pgconn.finish()
        await _close_quietly(conn)

    async def _open(self) -> _Conn:
        """Open a session with the agent role's own defaults, restoring them first if an earlier
        session changed them. Raise EnvironmentFailure if the instance refuses a session, or
        if the defaults cannot be restored from the agent role."""
        name = self._instance.name
        conn: _Conn | None = None
        try:
            conn = await self._connect()
            rows = await _role_settings(conn)
            if problems := role_defaults_problems(rows):
                _log.warning("restoring the agent role's defaults on %s: %s", name, problems)
                await _restore_role_defaults(conn, rows)
                await _close_quietly(conn)  # Only a new session starts with them.
                conn = None
                conn = await self._connect()
                if left := role_defaults_problems(await _role_settings(conn)):
                    raise EnvironmentFailure(
                        f"the agent role's defaults on {name} differ from the setup's, and the "
                        f"agent role cannot restore them: {left}. `pgenv.py start` reapplies "
                        "the roles."
                    )
            return conn
        except BaseException as error:
            if conn is not None:
                await asyncio.shield(_close_quietly(conn))
            if isinstance(error, psycopg.Error):
                raise EnvironmentFailure(f"cannot open a session on {name}: {error}") from error
            raise


async def _role_settings(conn: psycopg.AsyncConnection[Any]) -> list[tuple[Any, ...]]:
    return await (await conn.execute(_ROLE_SETTINGS_SQL)).fetchall()


async def _restore_role_defaults(
    conn: psycopg.AsyncConnection[Any], rows: Sequence[Sequence[Any]]
) -> None:
    """Undo, as the agent role, what it may: every per-database default, and every role-wide
    one but the superuser-only ones, which its RESET ALL leaves in place."""
    async with conn.transaction():
        await conn.execute("SET TRANSACTION READ WRITE")
        for db, _ in rows:
            if db is not None:
                await conn.execute(
                    SQL("ALTER ROLE CURRENT_USER IN DATABASE {} RESET ALL").format(Identifier(db))
                )
        await conn.execute("ALTER ROLE CURRENT_USER RESET ALL")
        for setting, value in ROLE_DEFAULTS[AGENT_ROLE].items():
            if setting not in _SUPERUSER_ONLY:
                await conn.execute(
                    SQL("ALTER ROLE CURRENT_USER SET {} = {}").format(
                        Identifier(setting), Literal(value)
                    )
                )


def _failure(task: asyncio.Task[None]) -> BaseException | None:
    """What ended ``task``, which is done: its exception, a cancellation, or None."""
    if task.cancelled():
        return asyncio.CancelledError("the statement was stopped")
    return task.exception()


async def _finished(task: asyncio.Task[None], timeout: float) -> bool:
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if done and not task.cancelled():
        task.exception()  # Retrieved, so asyncio does not log it as never retrieved.
    return bool(done)


async def _close_quietly(conn: psycopg.AsyncConnection[Any]) -> None:
    with contextlib.suppress(Exception):
        await conn.close()


@dataclass
class _Slot:
    """A solving instance: its statement cap and how many environments it holds."""

    instance: Instance
    cap: asyncio.Semaphore
    environments: int = 0


class SolvingPool:
    """Gives each task its own SQL environment on the least-loaded solving instance.

    It takes the agent role's password alone. With ``socket_dirs``, a directory per instance
    name, it connects through each instance's Unix socket there, with the instance's port
    naming the socket, instead of over TCP: the solve container's way in, since it may not use
    IP (V28). ``pg_instances.socket_dirs`` gives the host's directories.
    """

    def __init__(
        self,
        config: PgEnvConfig,
        credentials: AgentCredentials,
        options: SqlEnvOptions | None = None,
        instances: Sequence[Instance] = SOLVING,
        *,
        socket_dirs: Mapping[str, Path] | None = None,
    ) -> None:
        if not isinstance(credentials, AgentCredentials):
            raise TypeError("SolvingPool takes AgentCredentials: the agent's password, no other")
        if socket_dirs is not None:
            missing = [i.name for i in instances if i.name not in socket_dirs]
            relative = [str(path) for path in socket_dirs.values() if not path.is_absolute()]
            if missing or relative:
                raise ValueError(
                    "every solving instance needs an absolute socket directory: missing for "
                    f"{missing}, relative: {relative}"
                )
        self._config = config
        self._credentials = credentials
        self._socket_dirs = dict(socket_dirs) if socket_dirs is not None else None
        self.options = options if options is not None else SqlEnvOptions()
        self._slots = [_Slot(i, asyncio.Semaphore(self.options.statement_cap)) for i in instances]

    def load(self) -> dict[str, int]:
        """How many environments each instance holds now."""
        return {slot.instance.name: slot.environments for slot in self._slots}

    async def preflight(self) -> None:
        """Check every solving instance before a run, and fail fast if one is not ready.

        The agent role must log in, start with the setup's defaults (read-only transactions,
        its temp-file cap and its timeout backstops) and read the TPC-H tables. Raise
        EnvironmentFailure naming every problem found.
        """
        defaults = ROLE_DEFAULTS[AGENT_ROLE]
        problems: list[str] = []
        for slot in self._slots:
            name = f"{slot.instance.name} (port {slot.instance.port})"
            try:
                conn = await self._connect(slot.instance)
            except psycopg.Error as error:
                problems.append(f"{name}: the agent role cannot log in: {error}")
                continue
            try:
                cursor = await conn.execute(_PREFLIGHT_SQL, [list(defaults)])
                row = await cursor.fetchone()
            except psycopg.Error as error:
                problems.append(f"{name}: the agent's session failed: {error}")
            else:
                can_read, settings = row if row is not None else (False, [])
                found = dict(zip(defaults, settings or [], strict=False))
                problems += [
                    f"{name}: the agent's {setting} is {found.get(setting)}, not {wanted}"
                    for setting, wanted in defaults.items()
                    if found.get(setting) != wanted
                ]
                if not can_read:
                    problems.append(f"{name}: the agent role cannot read the TPC-H tables")
            finally:
                await _close_quietly(conn)
        if problems:
            raise EnvironmentFailure(
                f"the solving instances are not ready: {'; '.join(problems)}. "
                "`pgenv.py start` starts them and reapplies the roles."
            )

    @asynccontextmanager
    async def environment(self) -> AsyncIterator[PgSqlEnvironment]:
        """Provision a task's environment; it is closed however the block exits."""
        slot = min(self._slots, key=lambda candidate: candidate.environments)
        slot.environments += 1
        try:
            environment = PgSqlEnvironment(
                slot.instance, slot.cap, self.options, connect=lambda: self._connect(slot.instance)
            )
            try:
                await environment.open()
                yield environment
            finally:
                await environment.close()
        finally:
            slot.environments -= 1  # Even when a second cancel cuts close() short.

    async def _connect(self, instance: Instance) -> psycopg.AsyncConnection[Any]:
        host = (
            str(self._socket_dirs[instance.name])
            if self._socket_dirs is not None
            else self._config.host
        )
        kwargs = connection_kwargs(
            host, instance.port, AGENT_ROLE, self._credentials.agent_password, MAIN_DB
        )
        return await psycopg.AsyncConnection.connect(
            **kwargs,
            application_name="goaly-effisql-agent",
            prepare_threshold=None,  # Nothing prepared for DISCARD ALL to pull away.
        )
