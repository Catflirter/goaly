"""The catalog fingerprint and the server settings, on a real PostgreSQL instance.

Run with ``pytest --integration``. The fingerprint test adds an index and writes rows, so it
runs only on the throwaway test environment, never on the real instances.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import psycopg
import pytest

from goaly.bench.effisql.pg_fingerprint import (
    Fingerprint,
    compute_fingerprint,
    fingerprint_changes,
)
from goaly.bench.effisql.pg_instances import fetch_server_settings
from goaly.bench.effisql.pg_settings import ADMIN_ROLE, GRADER_ROLE
from tests.bench.effisql.conftest import PgEnv

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
async def grader_conns(pg_env: PgEnv) -> AsyncIterator[dict[str, psycopg.AsyncConnection[Any]]]:
    instance = pg_env.instance("grading")
    conns = {
        db: await psycopg.AsyncConnection.connect(
            host=pg_env.config.host,
            port=instance.port,
            user=GRADER_ROLE,
            password=pg_env.credentials.grader_password.get_secret_value(),
            dbname=db,
            autocommit=True,
        )
        for db in ("tpch", "tpch_alt")
    }
    yield conns
    for conn in conns.values():
        await conn.close()


async def test_the_servers_run_the_postgres_16_defaults(
    grader_conns: dict[str, psycopg.AsyncConnection[Any]],
) -> None:
    settings = await fetch_server_settings(grader_conns["tpch"])

    assert settings["server_version"].startswith("16.")
    assert {name: settings[name] for name in ("shared_buffers", "work_mem", "jit")} == {
        "shared_buffers": "128MB",
        "work_mem": "4MB",
        "jit": "on",
    }
    assert settings["max_parallel_workers_per_gather"] == "2"
    assert settings["max_parallel_workers"] == "8"
    assert settings["max_worker_processes"] == "8"


async def _changed(
    conns: dict[str, psycopg.AsyncConnection[Any]], before: Fingerprint, needle: str
) -> list[str]:
    # Backends publish their write counters a moment after their transaction ends.
    deadline = time.monotonic() + 20
    while True:
        changes = fingerprint_changes(before, await compute_fingerprint(conns))
        if any(needle in change for change in changes) or time.monotonic() > deadline:
            return changes
        time.sleep(0.2)


async def test_the_fingerprint_changes_when_an_index_is_added_or_a_row_is_written(
    writable_env: PgEnv, grader_conns: dict[str, psycopg.AsyncConnection[Any]]
) -> None:
    before = await compute_fingerprint(grader_conns)
    assert fingerprint_changes(before, await compute_fingerprint(grader_conns)) == []
    assert set(before.databases) == {"tpch", "tpch_alt"}
    assert "lineitem_pkey" in before.databases["tpch"].indexes

    grading = writable_env.instance("grading")
    with writable_env.connect(grading, ADMIN_ROLE, "tpch", autocommit=True) as admin:
        admin.execute("CREATE INDEX itest_fp_idx ON orders (o_orderdate)")
        try:
            changes = await _changed(grader_conns, before, "itest_fp_idx")
            assert any(c.startswith("tpch: index itest_fp_idx was added") for c in changes)
        finally:
            admin.execute("DROP INDEX itest_fp_idx")

        before = await compute_fingerprint(grader_conns)
        admin.execute("INSERT INTO region VALUES (99, 'ITEST', 'written by a test')")
        admin.execute("DELETE FROM region WHERE r_regionkey = 99")
        changes = await _changed(grader_conns, before, "table region")
        assert any(c.startswith("tpch: table region has writes") for c in changes)
