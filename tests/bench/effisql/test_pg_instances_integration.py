"""The instances, their pinning and their roles, on real PostgreSQL servers.

Run with ``pytest --integration``. Every write here is an attempt that must fail, and each runs
in a transaction that is always rolled back, so with ``GOALY_EFFISQL_ITEST_REAL=1`` these checks
also run against the real instances: even a privilege that regressed could commit nothing.
"""

import os
import subprocess
import sys
from typing import Any

import psycopg
import pytest
from psycopg import errors

from goaly.bench.effisql.pg_build import connector, read_golden_record
from goaly.bench.effisql.pg_containers import inspect
from goaly.bench.effisql.pg_instances import (
    PYTHON_CPUS,
    confine_to_python_cpus,
    count_problems,
    start_instances,
    stop_instances,
    table_counts,
)
from goaly.bench.effisql.pg_settings import ADMIN_ROLE, AGENT_ROLE, GRADER_ROLE
from tests.bench.effisql.conftest import PgEnv

pytestmark = pytest.mark.integration

WRITES = [
    "CREATE INDEX itest_idx ON lineitem (l_shipdate)",
    "INSERT INTO region VALUES (99, 'X', 'x')",
    "UPDATE region SET r_comment = 'x' WHERE r_regionkey = 0",
    "DELETE FROM region WHERE r_regionkey = 0",
    "WITH d AS (DELETE FROM region RETURNING *) SELECT count(*) FROM d",
    "TRUNCATE nation",
    "CREATE TABLE itest_t (x int)",
    "CREATE TEMP TABLE itest_tmp (x int)",
    "SELECT lo_create(0)",
    "SELECT pg_terminate_backend(pg_backend_pid())",
]


def _cpus(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def test_every_instance_runs_pinned_to_both_smt_siblings_of_its_cores(pg_env: PgEnv) -> None:
    for instance in pg_env.instances:
        assert inspect(instance.container, "{{.HostConfig.CpusetCpus}}") == ",".join(
            str(cpu) for cpu in instance.cpus
        )
        status = subprocess.run(  # noqa: S603
            ["docker", "exec", instance.container, "grep", "Cpus_allowed_list", "/proc/1/status"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert _cpus(status.split(":", 1)[1].strip()) == set(instance.cpus)


def test_every_instance_listens_on_loopback_only(pg_env: PgEnv) -> None:
    # The image's postgresql.conf says listen_addresses = '*'; the command line overrides it.
    for instance in pg_env.instances:
        with pg_env.connect(instance, GRADER_ROLE, "tpch") as conn:
            assert conn.execute("SHOW listen_addresses").fetchone() == ("127.0.0.1",)


def test_python_processes_and_their_children_stay_on_cpus_0_1_12_13() -> None:
    before = os.sched_getaffinity(0)
    try:
        confine_to_python_cpus()
        child = subprocess.run(
            [sys.executable, "-c", "import os; print(sorted(os.sched_getaffinity(0)))"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert os.sched_getaffinity(0) == set(PYTHON_CPUS)
        assert child.strip() == str(list(PYTHON_CPUS))
    finally:
        os.sched_setaffinity(0, before)


def test_all_instances_hold_the_same_rows_and_the_alternate_database_differs(
    pg_env: PgEnv,
) -> None:
    record = read_golden_record(pg_env.config.golden_record)
    assert record is not None
    admin = pg_env.credentials.admin_password
    counts = {
        instance.name: table_counts(connector(pg_env.config.host, instance.port, admin))
        for instance in pg_env.instances
    }

    assert count_problems(counts, record) == []
    assert len({str(by_db) for by_db in counts.values()}) == 1
    main, alt = counts[pg_env.instances[0].name].values()
    assert [table for table in main if main[table] == alt[table]] == ["region", "nation"]


# Who tries, where: the agent on its solving instances, the grader on both of its databases.
ATTEMPTERS = [
    (AGENT_ROLE, "solving", "tpch"),
    (GRADER_ROLE, "grading", "tpch"),
    (GRADER_ROLE, "grading", "tpch_alt"),
]


@pytest.mark.parametrize("statement", WRITES)
@pytest.mark.parametrize(("role", "kind", "db"), ATTEMPTERS)
def test_neither_role_can_write_or_create_even_after_lifting_read_only(
    pg_env: PgEnv, role: str, kind: str, db: str, statement: str
) -> None:
    # Each attempt runs in a transaction that is always rolled back, so even a privilege that
    # regressed could not commit a change to the data the scores come from.
    instance = pg_env.instance(kind)
    with pg_env.connect(instance, role, db, autocommit=True) as conn:
        assert conn.execute("SHOW default_transaction_read_only").fetchone() == ("on",)
        with pytest.raises(psycopg.Error):
            _attempt(conn, statement)  # Refused by the read-only default, or by privileges.
        conn.execute("SET default_transaction_read_only = off")  # The first bypass attempt.
        with pytest.raises(errors.InsufficientPrivilege):
            _attempt(conn, statement)
        conn.execute("RESET default_transaction_read_only")
        with pytest.raises(errors.InsufficientPrivilege):
            _attempt(conn, statement, writable=True)  # The second: a writable transaction.


def _attempt(conn: psycopg.Connection[Any], statement: str, *, writable: bool = False) -> None:
    """Try ``statement`` in a transaction that is always rolled back."""
    with conn.transaction(force_rollback=True):
        if writable:
            conn.execute("SET TRANSACTION READ WRITE")
            assert conn.execute("SHOW transaction_read_only").fetchone() == ("off",)
        conn.execute(statement)


def test_the_agent_reaches_tpch_on_solving_instances_and_nothing_on_grading_ones(
    pg_env: PgEnv,
) -> None:
    for instance in pg_env.instances:
        allowed = ("tpch",) if instance.kind == "solving" else ()
        for db in ("tpch", "tpch_alt", "postgres", "template1"):
            if db in allowed:
                with pg_env.connect(instance, AGENT_ROLE, db) as conn:
                    assert conn.execute("SELECT count(*) FROM region").fetchone() == (5,)
                continue
            with pytest.raises(psycopg.OperationalError, match="permission denied for database"):
                pg_env.connect(instance, AGENT_ROLE, db)


def test_agent_sessions_start_with_the_server_side_backstops(pg_env: PgEnv) -> None:
    solving = pg_env.instance("solving")
    with pg_env.connect(solving, AGENT_ROLE, "tpch", autocommit=True) as agent:
        settings = agent.execute(
            "SELECT current_setting('statement_timeout'), "
            "current_setting('client_connection_check_interval'), "
            "current_setting('temp_file_limit'), "
            "current_setting('default_transaction_read_only')"
        ).fetchone()

    assert settings == ("130s", "5s", "5GB", "on")


def test_the_grader_reads_both_databases_read_only_by_default(pg_env: PgEnv) -> None:
    grading = pg_env.instance("grading")
    for db in ("tpch", "tpch_alt"):
        with pg_env.connect(grading, GRADER_ROLE, db, autocommit=True) as conn:
            assert conn.execute("SELECT count(*) FROM nation").fetchone() == (25,)
            assert conn.execute("SHOW default_transaction_read_only").fetchone() == ("on",)
            with (
                pytest.raises(errors.ReadOnlySqlTransaction),
                conn.transaction(force_rollback=True),
            ):
                conn.execute("WITH d AS (DELETE FROM region RETURNING *) SELECT count(*) FROM d")


@pytest.mark.parametrize("role", [ADMIN_ROLE, AGENT_ROLE, GRADER_ROLE])
def test_every_role_needs_its_password_over_tcp(pg_env: PgEnv, role: str) -> None:
    instance = pg_env.instance("solving")
    with pytest.raises(psycopg.OperationalError, match="password authentication failed"):
        psycopg.connect(
            host=pg_env.config.host,
            port=instance.port,
            user=role,
            password="wrong",  # noqa: S106  # A deliberately wrong test value.
            dbname="tpch",
        )


def test_every_instance_answers_through_its_socket_directory_and_only_with_a_password(
    pg_env: PgEnv,
) -> None:
    # What the solve container does, with no IP at all (V28): libpq's host is the directory.
    for instance in pg_env.instances:
        with pg_env.connect(instance, GRADER_ROLE, "tpch", socket=True) as conn:
            # No TCP address on either end: the session came through the Unix socket.
            assert conn.execute("SELECT inet_server_addr(), inet_client_addr()").fetchone() == (
                None,
                None,
            )
            assert conn.execute("SELECT count(*) FROM region").fetchone() == (5,)
    instance = pg_env.instance("solving")
    with pytest.raises(psycopg.OperationalError, match="password authentication failed"):
        psycopg.connect(
            host=str(pg_env.config.socket_dir(instance.name)),
            port=instance.port,
            user=AGENT_ROLE,
            password="wrong",  # noqa: S106  # A deliberately wrong test value.
            dbname="tpch",
        )


def test_even_the_local_socket_inside_the_container_needs_a_password(pg_env: PgEnv) -> None:
    instance = pg_env.instance("grading")
    completed = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "docker",
            "exec",
            instance.container,
            "psql",
            "-h",
            "/var/run/postgresql",
            "-p",
            str(instance.port),
            "-U",
            ADMIN_ROLE,
            "-d",
            "postgres",
            "-w",
            "-c",
            "SELECT 1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "password" in completed.stderr


def test_stop_and_start_are_idempotent_and_keep_the_data(writable_env: PgEnv) -> None:
    env = writable_env
    assert set(stop_instances(env.instances).values()) == {"stopped"}
    assert set(stop_instances(env.instances).values()) == {"not running"}

    assert set(start_instances(env.config, env.credentials, env.instances).values()) == {"started"}
    assert set(start_instances(env.config, env.credentials, env.instances).values()) == {"running"}

    with env.connect(env.instance("grading"), GRADER_ROLE, "tpch") as conn:
        assert conn.execute("SELECT count(*) FROM lineitem").fetchone() == (120_515,)
