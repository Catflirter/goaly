"""Grader-role connections to the grading instances (spec §6.2, §8.5).

:class:`GradingConnections` keeps one small connection pool per instance and database (``tpch``
and ``tpch_alt``), as the grader role: it opens a pool on first use, looks it up after that,
and closes them all at the end. It schedules nothing. T10's ``grading_pool`` is the one
scheduler: it decides which evaluation runs on which instance, one at a time per instance, by
phase priority with SMOKE first, and T20 and T21 take that priority from T10.

The connections are in autocommit mode: open a transaction to use a server-side cursor
(``async with conn.transaction():`` and then ``conn.cursor(name=...)``). They default to
read-only transactions, as the grader role does, and each goes back to its pool with DISCARD
ALL, so no setting, prepared statement, cursor, lock or listen outlives a loan.

T10 owns comparison, timing, hashing and the catalog-fingerprint checks.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

import psycopg
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr

from goaly.bench.effisql.pg_instances import Instance
from goaly.bench.effisql.pg_settings import (
    ALT_DB,
    GRADER_ROLE,
    MAIN_DB,
    PgEnvConfig,
    connection_kwargs,
)

__all__ = ["GradingConnections"]

_log = logging.getLogger(__name__)


class _ConnectionPool(Protocol):
    """What GradingConnections needs of psycopg_pool.AsyncConnectionPool."""

    async def open(self, wait: bool = False, timeout: float = 30.0) -> None: ...
    def connection(
        self, timeout: float | None = None
    ) -> AbstractAsyncContextManager[psycopg.AsyncConnection[Any]]: ...
    async def close(self, timeout: float = 5.0) -> None: ...


type _Key = tuple[str, str]  # (instance name, database)


class GradingConnections:
    """One connection pool per grading instance and database, as the grader role.

    Each pool keeps one connection open and lends at most ``max_size`` at once: a caller that
    asks for more waits for one to come back, up to the pool's 30 s, and then gets
    psycopg_pool.PoolTimeout. Opening a pool waits up to ``open_timeout_s`` for its first
    connection; if the instance refuses it, the call raises psycopg_pool.PoolTimeout too, and
    the next call tries again. Opening one instance's pool never holds up another instance.
    """

    def __init__(
        self,
        config: PgEnvConfig,
        password: SecretStr,
        *,
        max_size: int = 2,
        open_timeout_s: float = 30.0,
        pool_factory: Callable[[Instance, str], _ConnectionPool] | None = None,
    ) -> None:
        """``password`` is the grader role's, the only one these connections need."""
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, not {max_size}")
        self._config = config
        self._password = password
        self._max_size = max_size
        self._open_timeout_s = open_timeout_s
        self._make_pool = pool_factory if pool_factory is not None else self._new_pool
        self._pools: dict[_Key, _ConnectionPool] = {}
        self._opening: dict[_Key, asyncio.Task[_ConnectionPool]] = {}
        self._closed = False

    def lookup(self, instance: Instance, db: str) -> _ConnectionPool | None:
        """The open pool for ``db`` on ``instance``, or None if it has not opened yet."""
        return self._pools.get((instance.name, _checked(db)))

    async def open(self, instance: Instance, db: str) -> _ConnectionPool:
        """The pool for ``db`` on ``instance``, opened on first use.

        Callers that ask at once share one opening. A caller cancelled meanwhile leaves the
        opening to finish: its pool is kept, and closed by :meth:`close`, never leaked.
        """
        key = (instance.name, _checked(db))
        if self._closed:
            raise RuntimeError("the grading connections are closed")
        if (pool := self._pools.get(key)) is not None:  # No lock: a lookup never waits.
            return pool
        opening = self._opening.get(key)
        if opening is None:
            opening = asyncio.create_task(self._open(key, instance, db))
            self._opening[key] = opening
            opening.add_done_callback(lambda task: self._opened(key, task))
        return await asyncio.shield(opening)

    @asynccontextmanager
    async def connection(
        self, instance: Instance, db: str
    ) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
        """Lend a grader-role connection to ``db``, tpch or tpch_alt, on ``instance``.

        It goes back to its pool however the block exits. A caller that abandons a statement
        must cancel it first (``conn.cancel_safe()``): the pool closes a connection that comes
        back mid-statement, and its backend may run on until it next writes to the socket.
        """
        pool = await self.open(instance, db)
        async with pool.connection() as conn:
            yield conn

    async def close(self) -> None:
        """Close every pool, those still opening too; later calls raise RuntimeError.

        A pool that fails to close is logged at WARNING, and the others are still closed.
        """
        self._closed = True
        openings = list(self._opening.values())
        for opening in openings:
            opening.cancel()  # Each closes the pool it was opening.
        await asyncio.gather(*openings, return_exceptions=True)
        pools, self._pools = list(self._pools.items()), {}
        for (name, db), pool in pools:
            try:
                await pool.close()
            except Exception:  # noqa: BLE001  # Logged with its cause; the rest still close.
                _log.warning("closing the %s pool on %s failed", db, name, exc_info=True)

    async def _open(self, key: _Key, instance: Instance, db: str) -> _ConnectionPool:
        pool = self._make_pool(instance, db)
        try:
            await pool.open(wait=True, timeout=self._open_timeout_s)
        except BaseException:
            await asyncio.shield(_close_quietly(pool))  # Its workers must not linger.
            raise
        if self._closed:  # close() ran meanwhile: this pool must not outlive it.
            await _close_quietly(pool)
            raise RuntimeError("the grading connections were closed while a pool opened")
        self._pools[key] = pool
        return pool

    def _opened(self, key: _Key, task: asyncio.Task[_ConnectionPool]) -> None:
        if self._opening.get(key) is task:
            del self._opening[key]  # A failed opening is tried again by the next call.
        if not task.cancelled():
            task.exception()  # Retrieved, so asyncio does not log it as never retrieved.

    def _new_pool(self, instance: Instance, db: str) -> AsyncConnectionPool:
        kwargs = connection_kwargs(
            self._config.host, instance.port, GRADER_ROLE, self._password, db
        )
        return AsyncConnectionPool(
            kwargs={**kwargs, "application_name": "goaly-effisql-grader"},
            min_size=1,
            max_size=self._max_size,
            open=False,
            name=f"grading-{instance.name}-{db}",
            check=AsyncConnectionPool.check_connection,
            reset=_reset_session,
        )


def _checked(db: str) -> str:
    if db not in (MAIN_DB, ALT_DB):
        raise ValueError(f"grading connects to {MAIN_DB} or {ALT_DB}, not {db!r}")
    return db


async def _reset_session(conn: psycopg.AsyncConnection[Any]) -> None:
    """Return a connection to its pool as a new session of the grader role would start.

    DISCARD ALL also drops prepared statements, which psycopg sees and forgets too.
    """
    await conn.execute("DISCARD ALL")


async def _close_quietly(pool: _ConnectionPool) -> None:
    with contextlib.suppress(Exception):
        await pool.close()
