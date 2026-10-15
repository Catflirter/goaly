"""The grading-instance pool, for the grader (T10; spec §6.2, §8.5).

A grading instance runs one evaluation at a time, so acquiring one is exclusive. Each request
carries a :class:`GradingPriority`. A free instance goes to the request with the lowest value,
and among requests with the same value, to the earliest. goaly-smoke's requests (SMOKE) go
ahead of every other phase, so they wait at most for the evaluations already running. After
final solving, :meth:`GradingPool.add` brings in the solving instances, which hold the
alternate database too.

A lease lends grader-role connections to ``tpch`` and ``tpch_alt``, from one small connection
pool per instance and database. The connections are in autocommit mode: open a transaction to
use a server-side cursor (``async with conn.transaction():`` and then ``conn.cursor(name=...)``).
They default to read-only transactions, as the grader role does, and each goes back to its
pool with its session settings reset.

This module only lends instances and connections. T10 owns comparison, timing, hashing and the
catalog-fingerprint checks.
"""

import asyncio
import heapq
import itertools
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from enum import IntEnum
from typing import Any, Protocol

import psycopg
from psycopg_pool import AsyncConnectionPool

from goaly.bench.effisql.pg_instances import GRADING, Instance
from goaly.bench.effisql.pg_settings import ALT_DB, GRADER_ROLE, MAIN_DB, PgCredentials, PgEnvConfig

__all__ = ["GradingLease", "GradingPool", "GradingPriority"]


class GradingPriority(IntEnum):
    """Who gets the next free grading instance: the lowest value first, then first come.

    T10's scheduler maps the phases (spec §6.2) onto these levels; SMOKE is goaly-smoke's.
    """

    SMOKE = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class _ConnectionPool(Protocol):
    """What a lease needs of psycopg_pool.AsyncConnectionPool."""

    async def open(self, wait: bool = False, timeout: float = 30.0) -> None: ...
    def connection(self) -> Any: ...  # An async context manager lending a connection.
    async def close(self) -> None: ...


class GradingLease:
    """One exclusive hold on a grading instance, until its ``acquire`` block exits."""

    def __init__(self, instance: Instance, pool: "GradingPool") -> None:
        self.instance = instance
        self._pool = pool

    @asynccontextmanager
    async def connection(self, db: str) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
        """Lend a grader-role connection to ``db``, tpch or tpch_alt, on this instance."""
        if db not in (MAIN_DB, ALT_DB):
            raise ValueError(f"grading connects to {MAIN_DB} or {ALT_DB}, not {db!r}")
        connections = await self._pool._connections(self.instance, db)
        async with connections.connection() as conn:
            yield conn


class GradingPool:
    """Lends the grading instances one evaluation at a time, by priority, then in order."""

    def __init__(
        self,
        config: PgEnvConfig,
        credentials: PgCredentials,
        instances: Sequence[Instance] = GRADING,
        *,
        pool_factory: Callable[[Instance, str], _ConnectionPool] | None = None,
    ) -> None:
        self._config = config
        self._credentials = credentials
        self._instances = list(instances)
        self._free = list(instances)
        self._waiting: list[tuple[int, int, asyncio.Future[Instance]]] = []
        self._arrivals = itertools.count()
        self._make_pool = pool_factory if pool_factory is not None else self._new_pool
        self._pools: dict[tuple[str, str], _ConnectionPool] = {}
        self._pools_lock = asyncio.Lock()

    @property
    def instances(self) -> tuple[Instance, ...]:
        return tuple(self._instances)

    def add(self, instances: Sequence[Instance]) -> None:
        """Bring more instances into the pool, such as the solving instances after solving."""
        for instance in instances:
            if instance not in self._instances:
                self._instances.append(instance)
                self._release(instance)

    @asynccontextmanager
    async def acquire(
        self, priority: GradingPriority = GradingPriority.NORMAL
    ) -> AsyncIterator[GradingLease]:
        """Hold a grading instance exclusively for the block, waiting by ``priority``."""
        instance = await self._take(priority)
        try:
            yield GradingLease(instance, self)
        finally:
            self._release(instance)

    async def close(self) -> None:
        """Close every connection pool. Leases already handed out must have ended."""
        pools, self._pools = list(self._pools.values()), {}
        for connections in pools:
            await connections.close()

    async def _take(self, priority: GradingPriority) -> Instance:
        if self._free and not self._waiting:
            return self._free.pop(0)
        waiter: asyncio.Future[Instance] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiting, (int(priority), next(self._arrivals), waiter))
        try:
            return await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                self._release(waiter.result())  # It arrived as we were cancelled: pass it on.
            raise

    def _release(self, instance: Instance) -> None:
        while self._waiting:
            _, _, waiter = heapq.heappop(self._waiting)
            if not waiter.done():  # A cancelled waiter is skipped here, lazily.
                waiter.set_result(instance)
                return
        self._free.append(instance)

    async def _connections(self, instance: Instance, db: str) -> _ConnectionPool:
        key = (instance.name, db)
        async with self._pools_lock:
            if key not in self._pools:
                connections = self._make_pool(instance, db)
                await connections.open(wait=True)
                self._pools[key] = connections
            return self._pools[key]

    def _new_pool(self, instance: Instance, db: str) -> AsyncConnectionPool:
        return AsyncConnectionPool(
            kwargs={
                "host": self._config.host,
                "port": instance.port,
                "user": GRADER_ROLE,
                "password": self._credentials.grader_password.get_secret_value(),
                "dbname": db,
                "autocommit": True,
                "connect_timeout": 10,
                "application_name": "goaly-effisql-grader",
            },
            min_size=1,
            max_size=2,
            open=False,
            name=f"grading-{instance.name}-{db}",
            check=AsyncConnectionPool.check_connection,
            reset=_reset_session,
        )


async def _reset_session(conn: psycopg.AsyncConnection[Any]) -> None:
    """Return a connection with its session settings as the grader role's defaults set them."""
    await conn.execute("RESET ALL")
