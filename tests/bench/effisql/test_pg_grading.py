import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import SecretStr

from goaly.bench.effisql.pg_grading import GradingPool, GradingPriority
from goaly.bench.effisql.pg_instances import GRADING, SOLVING, Instance
from goaly.bench.effisql.pg_settings import PgCredentials, PgEnvConfig

pytestmark = pytest.mark.anyio

CREDENTIALS = PgCredentials(
    admin_password=SecretStr("a"),
    agent_password=SecretStr("b"),
    grader_password=SecretStr("c"),
    _env_file=None,
)


class FakeConnectionPool:
    def __init__(self, instance: Instance, db: str) -> None:
        self.key = (instance.name, db)
        self.opened = False
        self.closed = False
        self.lent = 0

    async def open(self, wait: bool = False, timeout: float = 30.0) -> None:
        self.opened = True

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Any]:
        self.lent += 1
        yield self.key

    async def close(self) -> None:
        self.closed = True


class FakePools:
    def __init__(self) -> None:
        self.made: list[FakeConnectionPool] = []

    def __call__(self, instance: Instance, db: str) -> Any:
        pool = FakeConnectionPool(instance, db)
        self.made.append(pool)
        return pool


def _pool(*instances: Instance, pools: FakePools | None = None) -> GradingPool:
    factory = pools if pools is not None else FakePools()
    return GradingPool(PgEnvConfig(), CREDENTIALS, instances or GRADING[:1], pool_factory=factory)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_acquiring_an_instance_is_exclusive() -> None:
    pool = _pool()
    async with pool.acquire() as lease:
        assert lease.instance == GRADING[0]
        second = asyncio.create_task(_take(pool, GradingPriority.NORMAL, []))
        await _settle()
        assert not second.done()  # One evaluation at a time on an instance.
    await second


async def _take(
    pool: GradingPool, priority: GradingPriority, order: list[str], name: str = ""
) -> None:
    async with pool.acquire(priority):
        order.append(name or priority.name)
        await _settle()


async def test_a_smoke_request_queued_behind_normal_ones_gets_the_next_free_instance() -> None:
    pool = _pool()
    order: list[str] = []
    async with pool.acquire():
        normal = [
            asyncio.create_task(_take(pool, GradingPriority.NORMAL, order, f"normal-{i}"))
            for i in (1, 2)
        ]
        await _settle()
        smoke = asyncio.create_task(_take(pool, GradingPriority.SMOKE, order, "smoke"))
        await _settle()
    await asyncio.gather(*normal, smoke)

    assert order == ["smoke", "normal-1", "normal-2"]


async def test_levels_are_served_lowest_first_and_in_arrival_order_within_one() -> None:
    pool = _pool()
    order: list[str] = []
    async with pool.acquire():
        tasks = []
        for name, priority in [
            ("low", GradingPriority.LOW),
            ("normal-1", GradingPriority.NORMAL),
            ("high", GradingPriority.HIGH),
            ("normal-2", GradingPriority.NORMAL),
        ]:
            tasks.append(asyncio.create_task(_take(pool, priority, order, name)))
            await _settle()
    await asyncio.gather(*tasks)

    assert order == ["high", "normal-1", "normal-2", "low"]
    assert sorted(GradingPriority) == [
        GradingPriority.SMOKE,
        GradingPriority.HIGH,
        GradingPriority.NORMAL,
        GradingPriority.LOW,
    ]


async def test_a_cancelled_waiter_leaves_the_queue() -> None:
    pool = _pool()
    order: list[str] = []
    async with pool.acquire():
        gone = asyncio.create_task(_take(pool, GradingPriority.SMOKE, order, "gone"))
        kept = asyncio.create_task(_take(pool, GradingPriority.NORMAL, order, "kept"))
        await _settle()
        gone.cancel()
        await _settle()
    await kept

    assert order == ["kept"]
    assert gone.cancelled()


async def test_a_waiter_cancelled_as_its_instance_arrives_hands_it_on() -> None:
    pool = _pool()
    order: list[str] = []
    holder = pool.acquire()
    await holder.__aenter__()
    first = asyncio.create_task(_take(pool, GradingPriority.NORMAL, order, "first"))
    second = asyncio.create_task(_take(pool, GradingPriority.NORMAL, order, "second"))
    await _settle()

    await holder.__aexit__(None, None, None)  # The instance goes to "first"...
    first.cancel()  # ...which is cancelled before it runs.
    await asyncio.gather(first, second, return_exceptions=True)

    assert order == ["second"]  # The instance was not lost.


async def test_added_instances_serve_waiting_requests_at_once() -> None:
    pool = _pool()
    order: list[str] = []
    async with pool.acquire():
        waiting = asyncio.create_task(_take(pool, GradingPriority.NORMAL, order, "waiting"))
        await _settle()
        pool.add(SOLVING)  # After final solving, the solving instances grade too.
        await waiting

    assert order == ["waiting"]
    assert pool.instances == GRADING[:1] + SOLVING


async def test_a_lease_lends_grader_connections_from_one_pool_per_instance_and_database() -> None:
    pools = FakePools()
    pool = _pool(GRADING[0], GRADING[1], pools=pools)

    async with pool.acquire() as lease:
        async with lease.connection("tpch") as main, lease.connection("tpch_alt") as alt:
            lent: list[object] = [main, alt]  # The fake lends its key, not a connection.
            assert lent == [("grade-1", "tpch"), ("grade-1", "tpch_alt")]
        async with lease.connection("tpch"):
            pass

    assert [made.key for made in pools.made] == [("grade-1", "tpch"), ("grade-1", "tpch_alt")]
    assert all(made.opened for made in pools.made)
    assert pools.made[0].lent == 2  # The second use reused the pool.
    with pytest.raises(ValueError, match="tpch or tpch_alt"):
        async with pool.acquire() as lease, lease.connection("postgres"):
            pass

    await pool.close()
    assert all(made.closed for made in pools.made)


async def test_the_default_connection_pools_use_the_grader_role_in_autocommit() -> None:
    pool = GradingPool(PgEnvConfig(), CREDENTIALS, GRADING[:1])

    connections = pool._new_pool(GRADING[0], "tpch_alt")  # Built, never opened.

    kwargs = connections.kwargs
    assert isinstance(kwargs, dict)
    assert kwargs["user"] == "effisql_grader"
    assert kwargs["dbname"] == "tpch_alt"
    assert kwargs["port"] == 55434
    assert kwargs["autocommit"] is True
    assert kwargs["password"] == CREDENTIALS.grader_password.get_secret_value()
    assert connections.max_size == 2
