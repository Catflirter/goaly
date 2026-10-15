"""The grading pool against real PostgreSQL instances; read-only, so safe on the real ones too.

Run with ``pytest --integration``.
"""

import pytest
from psycopg import errors

from goaly.bench.effisql.pg_fingerprint import compute_fingerprint
from goaly.bench.effisql.pg_grading import GradingPool, GradingPriority
from goaly.bench.effisql.pg_instances import Instance
from tests.bench.effisql.conftest import PgEnv

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _grading(pg_env: PgEnv) -> list[Instance]:
    grading = [instance for instance in pg_env.instances if instance.kind == "grading"]
    if not grading:
        pytest.skip("no grading instance in GOALY_EFFISQL_ITEST_INSTANCES")
    return grading


async def test_a_lease_lends_working_grader_connections_to_both_databases(pg_env: PgEnv) -> None:
    grading = _grading(pg_env)
    pool = GradingPool(pg_env.config, pg_env.credentials, grading)
    try:
        async with pool.acquire(GradingPriority.SMOKE) as lease:
            async with lease.connection("tpch") as conn:
                user = await (await conn.execute("SELECT current_user")).fetchone()
                assert user == ("effisql_grader",)
                async with conn.transaction(), conn.cursor(name="signature") as cursor:
                    await cursor.execute("SELECT * FROM lineitem")  # A server-side cursor.
                    assert len(await cursor.fetchmany(1000)) == 1000
                await conn.execute("SET work_mem = '64MB'")
            async with lease.connection("tpch_alt") as alt:
                nations = await (await alt.execute("SELECT count(*) FROM nation")).fetchone()
                assert nations == (25,)
            async with lease.connection("tpch") as main, lease.connection("tpch_alt") as alt:
                fingerprint = await compute_fingerprint({"tpch": main, "tpch_alt": alt})
                assert set(fingerprint.databases) == {"tpch", "tpch_alt"}
            async with lease.connection("tpch") as again:
                work_mem = await (await again.execute("SHOW work_mem")).fetchone()
                assert work_mem == ("4MB",)  # Settings do not outlive a loan.
                with pytest.raises(errors.ReadOnlySqlTransaction):
                    await again.execute("DELETE FROM region")
    finally:
        await pool.close()


async def test_every_grading_instance_can_be_held_at_once(pg_env: PgEnv) -> None:
    grading = _grading(pg_env)
    if len(grading) < 3:
        pytest.skip("needs all three grading instances")
    pool = GradingPool(pg_env.config, pg_env.credentials, grading)
    try:
        async with (
            pool.acquire() as first,
            pool.acquire() as second,
            pool.acquire(GradingPriority.HIGH) as third,
        ):
            held = {first.instance, second.instance, third.instance}
            assert held == set(grading)
            for lease in (first, second, third):
                async with lease.connection("tpch") as conn:
                    port = await (await conn.execute("SHOW port")).fetchone()
                    assert port == (str(lease.instance.port),)
    finally:
        await pool.close()
