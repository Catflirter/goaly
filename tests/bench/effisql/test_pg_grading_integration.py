"""Grader connections against real PostgreSQL instances; read-only, so safe on the real ones too.

Run with ``pytest --integration``.
"""

import pytest
from psycopg import errors

from goaly.bench.effisql.pg_fingerprint import compute_fingerprint
from goaly.bench.effisql.pg_grading import GradingConnections
from goaly.bench.effisql.pg_instances import Instance
from tests.bench.effisql.conftest import PgEnv

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _grading(pg_env: PgEnv) -> list[Instance]:
    grading = [instance for instance in pg_env.instances if instance.kind == "grading"]
    if not grading:
        pytest.skip("no grading instance in GOALY_EFFISQL_ITEST_INSTANCES")
    return grading


async def test_grader_connections_reach_both_databases_and_nothing_outlives_a_loan(
    pg_env: PgEnv,
) -> None:
    instance = _grading(pg_env)[0]
    # One connection per pool, so every loan below is the same backend.
    connections = GradingConnections(pg_env.config, pg_env.credentials.grader_password, max_size=1)
    try:
        async with connections.connection(instance, "tpch") as conn:
            user = await (await conn.execute("SELECT current_user")).fetchone()
            assert user == ("effisql_grader",)
            async with conn.transaction(), conn.cursor(name="signature") as cursor:
                await cursor.execute("SELECT * FROM lineitem")  # A server-side cursor.
                assert len(await cursor.fetchmany(1000)) == 1000
            backend = conn.info.backend_pid
            await conn.execute("SET work_mem = '64MB'")
            await conn.execute("PREPARE itest_p AS SELECT 1")
        async with connections.connection(instance, "tpch_alt") as alt:
            nations = await (await alt.execute("SELECT count(*) FROM nation")).fetchone()
            assert nations == (25,)
        async with (
            connections.connection(instance, "tpch") as main,
            connections.connection(instance, "tpch_alt") as alt,
        ):
            fingerprint = await compute_fingerprint({"tpch": main, "tpch_alt": alt})
            assert set(fingerprint.databases) == {"tpch", "tpch_alt"}
        async with connections.connection(instance, "tpch") as again:
            assert again.info.backend_pid == backend  # The same session, discarded:
            work_mem = await (await again.execute("SHOW work_mem")).fetchone()
            assert work_mem == ("4MB",)  # its settings
            with pytest.raises(errors.InvalidSqlStatementName):
                await again.execute("EXECUTE itest_p")  # and its prepared statements are gone.
        async with connections.connection(instance, "tpch") as conn:
            with pytest.raises(errors.ReadOnlySqlTransaction):
                await conn.execute("DELETE FROM region")
    finally:
        await connections.close()


async def test_every_grading_instance_has_its_own_pools(pg_env: PgEnv) -> None:
    grading = _grading(pg_env)
    connections = GradingConnections(pg_env.config, pg_env.credentials.grader_password)
    try:
        for instance in grading:
            async with connections.connection(instance, "tpch") as conn:
                port = await (await conn.execute("SHOW port")).fetchone()
                assert port == (str(instance.port),)
    finally:
        await connections.close()
