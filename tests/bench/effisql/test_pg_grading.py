import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from psycopg_pool import PoolTimeout
from pydantic import SecretStr

from goaly.bench.effisql.pg_grading import GradingConnections
from goaly.bench.effisql.pg_instances import GRADING, Instance
from goaly.bench.effisql.pg_settings import PgEnvConfig

pytestmark = pytest.mark.anyio

PASSWORD = SecretStr("grader-pw")


class FakeConnectionPool:
    """A psycopg_pool pool that lends its key; ``gate`` holds its open until set."""

    def __init__(self, instance: Instance, db: str, gate: asyncio.Event | None = None) -> None:
        self.key = (instance.name, db)
        self.gate = gate
        self.open_error: BaseException | None = None
        self.close_error: BaseException | None = None
        self.opened_with: tuple[bool, float] | None = None
        self.closed = False
        self.lent = 0
        self.returned = 0

    async def open(self, wait: bool = False, timeout: float = 30.0) -> None:
        if self.gate is not None:
            await self.gate.wait()
        if self.open_error is not None:
            raise self.open_error
        self.opened_with = (wait, timeout)

    @asynccontextmanager
    async def connection(self, timeout: float | None = None) -> AsyncIterator[Any]:
        self.lent += 1
        try:
            yield self.key
        finally:
            self.returned += 1

    async def close(self, timeout: float = 5.0) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakePools:
    def __init__(self) -> None:
        self.made: list[FakeConnectionPool] = []
        self.gates: dict[tuple[str, str], asyncio.Event] = {}
        self.open_errors: list[BaseException] = []

    def __call__(self, instance: Instance, db: str) -> Any:
        pool = FakeConnectionPool(instance, db, self.gates.get((instance.name, db)))
        if self.open_errors:
            pool.open_error = self.open_errors.pop(0)
        self.made.append(pool)
        return pool


def _connections(pools: FakePools, **kwargs: Any) -> GradingConnections:
    return GradingConnections(PgEnvConfig(), PASSWORD, pool_factory=pools, **kwargs)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_one_pool_per_instance_and_database_lends_the_connections() -> None:
    pools = FakePools()
    connections = _connections(pools, open_timeout_s=7.0)
    grade_1 = GRADING[0]
    assert connections.lookup(grade_1, "tpch") is None

    async with (
        connections.connection(grade_1, "tpch") as main,
        connections.connection(grade_1, "tpch_alt") as alt,
    ):
        lent: list[object] = [main, alt]  # The fake lends its key, not a connection.
        assert lent == [("grade-1", "tpch"), ("grade-1", "tpch_alt")]
    async with connections.connection(grade_1, "tpch"):
        pass

    assert [made.key for made in pools.made] == [("grade-1", "tpch"), ("grade-1", "tpch_alt")]
    assert all(made.opened_with == (True, 7.0) for made in pools.made)
    assert pools.made[0].lent == pools.made[0].returned == 2  # The second loan reused it.
    assert connections.lookup(grade_1, "tpch") is pools.made[0]
    with pytest.raises(ValueError, match="tpch or tpch_alt"):
        async with connections.connection(grade_1, "postgres"):
            pass


async def test_an_instance_whose_pool_is_slow_to_open_holds_up_no_other() -> None:
    pools = FakePools()
    pools.gates[("grade-1", "tpch")] = asyncio.Event()  # grade-1 never answers.
    connections = _connections(pools)
    stuck = asyncio.create_task(connections.open(GRADING[0], "tpch"))
    await _settle()

    async with asyncio.timeout(1), connections.connection(GRADING[1], "tpch") as conn:
        lent: object = conn  # The fake lends its key, not a connection.
        assert lent == ("grade-2", "tpch")

    assert not stuck.done()
    stuck.cancel()
    await connections.close()


async def test_callers_that_ask_at_once_share_one_opening() -> None:
    pools = FakePools()
    gate = pools.gates[("grade-1", "tpch")] = asyncio.Event()
    connections = _connections(pools)

    first = asyncio.create_task(connections.open(GRADING[0], "tpch"))
    second = asyncio.create_task(connections.open(GRADING[0], "tpch"))
    await _settle()
    gate.set()

    assert await first is await second
    assert len(pools.made) == 1


async def test_a_caller_cancelled_while_its_pool_opens_leaks_nothing() -> None:
    pools = FakePools()
    gate = pools.gates[("grade-1", "tpch")] = asyncio.Event()
    connections = _connections(pools)

    caller = asyncio.create_task(connections.open(GRADING[0], "tpch"))
    await _settle()
    caller.cancel()  # A time limit, say, while the instance is slow to answer.
    with pytest.raises(asyncio.CancelledError):
        await caller
    gate.set()
    await _settle()

    pool = pools.made[0]
    assert connections.lookup(GRADING[0], "tpch") is pool  # Kept for the next caller,
    await connections.close()
    assert pool.closed  # and closed with the others.


async def test_close_also_closes_a_pool_still_opening() -> None:
    pools = FakePools()
    pools.gates[("grade-1", "tpch")] = asyncio.Event()  # It never finishes opening.
    connections = _connections(pools)
    opening = asyncio.create_task(connections.open(GRADING[0], "tpch"))
    await _settle()

    async with asyncio.timeout(1):
        await connections.close()

    assert pools.made[0].closed
    assert connections.lookup(GRADING[0], "tpch") is None
    with pytest.raises(asyncio.CancelledError):
        await opening
    with pytest.raises(RuntimeError, match="closed"):
        await connections.open(GRADING[1], "tpch")


async def test_a_pool_that_fails_to_open_is_closed_and_the_next_call_tries_again() -> None:
    pools = FakePools()
    pools.open_errors = [PoolTimeout("couldn't get a connection after 30.00 sec")]
    connections = _connections(pools)

    with pytest.raises(PoolTimeout):
        await connections.open(GRADING[0], "tpch")
    again = await connections.open(GRADING[0], "tpch")

    assert pools.made[0].closed  # The failed pool's workers do not linger.
    assert again is pools.made[1]


async def test_a_connection_goes_back_however_its_block_exits() -> None:
    pools = FakePools()
    connections = _connections(pools)

    with pytest.raises(ZeroDivisionError):
        async with connections.connection(GRADING[0], "tpch"):
            raise ZeroDivisionError

    async def hold() -> None:
        async with connections.connection(GRADING[0], "tpch"):
            await asyncio.sleep(60)

    holder = asyncio.create_task(hold())
    await _settle()
    holder.cancel()
    async with asyncio.timeout(1):
        with pytest.raises(asyncio.CancelledError):
            await holder

    pool = pools.made[0]
    assert pool.lent == pool.returned == 2


async def test_close_closes_every_pool_even_when_one_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pools = FakePools()
    connections = _connections(pools)
    for instance in GRADING:
        await connections.open(instance, "tpch")
    pools.made[0].close_error = OSError("the socket is gone")

    with caplog.at_level(logging.WARNING, logger="goaly.bench.effisql.pg_grading"):
        await connections.close()

    assert all(made.closed for made in pools.made)
    failed = next(r for r in caplog.records if "closing the tpch pool on grade-1" in r.getMessage())
    assert failed.exc_info is not None
    assert all(connections.lookup(instance, "tpch") is None for instance in GRADING)


def test_the_default_pools_connect_as_the_grader_role_in_autocommit() -> None:
    connections = GradingConnections(PgEnvConfig(), PASSWORD, max_size=3)

    pool = connections._new_pool(GRADING[0], "tpch_alt")  # Built, never opened.

    assert pool.kwargs == {
        "host": "127.0.0.1",
        "port": 55434,
        "user": "effisql_grader",
        "password": "grader-pw",
        "dbname": "tpch_alt",
        "autocommit": True,
        "connect_timeout": 10,
        "application_name": "goaly-effisql-grader",
    }
    assert (pool.min_size, pool.max_size) == (1, 3)
    assert pool.name == "grading-grade-1-tpch_alt"


async def test_a_returned_connection_is_discarded_to_a_fresh_session() -> None:
    connections = GradingConnections(PgEnvConfig(), PASSWORD)
    pool = connections._new_pool(GRADING[0], "tpch")
    executed: list[str] = []

    class Conn:
        async def execute(self, query: str) -> None:
            executed.append(query)

    assert pool._reset is not None
    await pool._reset(Conn())  # type: ignore[arg-type]

    assert executed == ["DISCARD ALL"]
    assert pool._check is not None  # A connection is checked before each loan.


def test_a_pool_needs_room_for_a_connection() -> None:
    with pytest.raises(ValueError, match="max_size must be at least 1"):
        GradingConnections(PgEnvConfig(), PASSWORD, max_size=0)
