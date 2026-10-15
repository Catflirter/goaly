"""The environment's five PostgreSQL instances (spec §4.1, §6.2): two solve, three grade.

Each instance runs a copy of the golden data directory, which holds both ``tpch`` and
``tpch_alt``, in its own container on its own port. It is pinned with cpuset to both SMT
siblings of two physical cores; on this machine physical core N is logical CPUs N and N + 12:

- solve-1 and solve-2: CPUs {2,3,14,15} and {4,5,16,17};
- grade-1, grade-2 and grade-3: CPUs {6,7,18,19}, {8,9,20,21} and {10,11,22,23}.

Our own Python processes keep to CPUs 0, 1, 12 and 13. A goaly process calls
:func:`confine_to_python_cpus` at startup, and the threads and child processes it starts later
inherit that; any other command runs under ``taskset -c 0,1,12,13``.

:func:`start_instances`, :func:`stop_instances` and :func:`instance_status` are idempotent. A
restart reuses each instance's data directory; only a new golden build makes it copy again.
Every start reapplies the roles and privileges. Read-only comes from the privileges, and the
role defaults only add defense in depth:

- the agent role connects to ``tpch`` only, with SELECT on its tables;
- the grader role connects to both databases, with SELECT on their tables;
- both default to read-only transactions;
- PUBLIC loses CONNECT and TEMPORARY on every database and CREATE on the public schema. It
  also loses the functions that let one session stop another or write large objects.
"""

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psycopg
from psycopg import sql
from pydantic import SecretStr

from goaly.bench.effisql import pg_containers
from goaly.bench.effisql.pg_build import (
    TPCH_TABLES,
    Connect,
    GoldenRecord,
    connector,
    read_golden_record,
    wait_until_ready,
)
from goaly.bench.effisql.pg_containers import CONTAINER_PREFIX, ContainerSpec
from goaly.bench.effisql.pg_settings import (
    AGENT_ROLE,
    ALT_DB,
    GRADER_ROLE,
    MAIN_DB,
    PgCredentials,
    PgEnvConfig,
    write_private,
)

__all__ = [
    "GRADING",
    "INSTANCES",
    "PYTHON_CPUS",
    "SOLVING",
    "Instance",
    "InstanceError",
    "InstanceStatus",
    "apply_roles",
    "confine_to_python_cpus",
    "connect_statements",
    "count_problems",
    "database_statements",
    "instance_spec",
    "instance_status",
    "logical_cpus",
    "role_statements",
    "start_instances",
    "stop_instances",
    "table_counts",
]

_SMT_OFFSET = 12  # Physical core N is logical CPUs N and N + 12 (spec §6.2).
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
_STOP_TIMEOUT_S = 600


class InstanceError(Exception):
    """The instances cannot be started or checked; the message says why."""


def logical_cpus(cores: Sequence[int]) -> tuple[int, ...]:
    """Return both SMT siblings of each physical core in ``cores``, in order."""
    return tuple(sorted({cpu for core in cores for cpu in (core, core + _SMT_OFFSET)}))


@dataclass(frozen=True)
class Instance:
    name: str
    kind: Literal["solving", "grading"]
    port: int
    cpus: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"instance name {self.name!r} must match {_NAME.pattern}")

    @property
    def container(self) -> str:
        return CONTAINER_PREFIX + self.name


INSTANCES = (
    Instance("solve-1", "solving", 55432, logical_cpus((2, 3))),
    Instance("solve-2", "solving", 55433, logical_cpus((4, 5))),
    Instance("grade-1", "grading", 55434, logical_cpus((6, 7))),
    Instance("grade-2", "grading", 55435, logical_cpus((8, 9))),
    Instance("grade-3", "grading", 55436, logical_cpus((10, 11))),
)
SOLVING = tuple(instance for instance in INSTANCES if instance.kind == "solving")
GRADING = tuple(instance for instance in INSTANCES if instance.kind == "grading")
PYTHON_CPUS = logical_cpus((0, 1))


def confine_to_python_cpus() -> None:
    """Pin this process, and every thread and child it starts later, to CPUs 0, 1, 12 and 13."""
    os.sched_setaffinity(0, PYTHON_CPUS)


def instance_spec(config: PgEnvConfig, instance: Instance) -> ContainerSpec:
    """The container of ``instance``. Docker restarts it with the daemon unless we stopped it."""
    return ContainerSpec(
        name=instance.container,
        image=config.image,
        port=instance.port,
        data_dir=_instance_dir(config, instance) / "pgdata",
        user=f"{os.getuid()}:{os.getgid()}",
        cpuset=",".join(str(cpu) for cpu in instance.cpus),
        restart="unless-stopped",
    )


_READERS = {MAIN_DB: (AGENT_ROLE, GRADER_ROLE), ALT_DB: (GRADER_ROLE,)}
# Functions PUBLIC may call by default that let a session stop another one or write data.
_REVOKED_FUNCTIONS = (
    "pg_catalog.pg_cancel_backend(integer)",
    "pg_catalog.pg_terminate_backend(integer, bigint)",
    "pg_catalog.lo_creat(integer)",
    "pg_catalog.lo_create(oid)",
    "pg_catalog.lo_from_bytea(oid, bytea)",
    "pg_catalog.lo_import(text)",
    "pg_catalog.lo_import(text, oid)",
)


def role_statements(verifiers: Mapping[str, str]) -> list[sql.Composed]:
    """Statements that give each role login with its SCRAM verifier and read-only defaults.

    Any default the role set for itself (a role may ALTER ROLE itself) is removed first.
    """
    statements = []
    for role, verifier in verifiers.items():
        name = sql.Identifier(role)
        attributes = "LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        statements += [
            sql.SQL(f"ALTER ROLE {{}} WITH {attributes} PASSWORD {{}}").format(
                name, sql.Literal(verifier)
            ),
            sql.SQL("ALTER ROLE {} RESET ALL").format(name),
            *(
                sql.SQL("ALTER ROLE {} IN DATABASE {} RESET ALL").format(name, sql.Identifier(db))
                for db in (MAIN_DB, ALT_DB)
            ),
            sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(name),
        ]
    return statements


def connect_statements() -> list[sql.Composed]:
    """Statements that let only our roles connect, each to the databases it may use."""
    revoke = [
        sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(db))
        for db in (MAIN_DB, ALT_DB, "postgres", "template1")
    ]
    grant = [
        sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(db), _names(roles))
        for db, roles in _READERS.items()
    ]
    return revoke + grant


def database_statements(db: str) -> list[sql.Composed]:
    """Statements, run in ``db``, that grant SELECT on its tables and nothing that writes."""
    public = sql.Identifier("public")
    functions = sql.SQL(", ").join(sql.SQL(function) for function in _REVOKED_FUNCTIONS)
    return [
        sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(public),
        sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(public, _names(_READERS[db])),
        sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(
            public, _names(_READERS[db])
        ),
        sql.SQL("REVOKE EXECUTE ON FUNCTION {} FROM PUBLIC").format(functions),
    ]


def apply_roles(connect: Connect, credentials: PgCredentials) -> None:
    """Create the agent and grader roles if needed and (re)apply their passwords and grants.

    Passwords reach the server only as SCRAM verifiers computed here by libpq, so no
    statement, and no server log line, ever holds one.
    """
    passwords = {AGENT_ROLE: credentials.agent_password, GRADER_ROLE: credentials.grader_password}
    with connect("postgres") as conn, conn.transaction():
        rows = conn.execute(
            "SELECT rolname FROM pg_roles WHERE rolname = ANY(%s)", [list(passwords)]
        )
        existing = {row[0] for row in rows}
        for role in passwords:
            if role not in existing:
                conn.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(role)))
        verifiers = {role: _verifier(conn, role, password) for role, password in passwords.items()}
        for statement in role_statements(verifiers) + connect_statements():
            conn.execute(statement)
    for db in _READERS:
        with connect(db) as conn, conn.transaction():
            for statement in database_statements(db):
                conn.execute(statement)


def start_instances(
    config: PgEnvConfig, credentials: PgCredentials, instances: Sequence[Instance] = INSTANCES
) -> dict[str, str]:
    """Bring every instance up from the golden build, and say what each took. Idempotent."""
    record = read_golden_record(config.golden_record)
    if record is None:
        raise InstanceError("there is no golden build yet: run `pgenv.py build` first")
    with ThreadPoolExecutor(max_workers=len(instances)) as pool:  # Copying is disk-bound.
        copied = dict(
            zip(
                instances,
                pool.map(lambda i: _ensure_copy(config, i, record), instances),
                strict=True,
            )
        )
    actions: dict[str, str] = {}
    for instance in instances:
        action = pg_containers.ensure(instance_spec(config, instance))
        actions[instance.name] = f"{action}, data copied" if copied[instance] else action
    for instance in instances:
        connect = connector(config.host, instance.port, credentials.admin_password)
        wait_until_ready(connect, instance.container)
        apply_roles(connect, credentials)
    return actions


def stop_instances(instances: Sequence[Instance] = INSTANCES) -> dict[str, str]:
    """Shut every running instance down cleanly; its container and data stay for a restart."""
    results: dict[str, str] = {}
    for instance in instances:
        if pg_containers.state(instance.container) != "running":
            results[instance.name] = "not running"
            continue
        code = pg_containers.stop(instance.container, timeout_s=_STOP_TIMEOUT_S)
        results[instance.name] = "stopped" if code == 0 else f"stopped with exit code {code}"
    return results


@dataclass(frozen=True)
class InstanceStatus:
    instance: Instance
    state: str | None  # The container's state, or None if there is no container.
    cpuset: str | None  # The CPUs Docker pins it to.
    accepting: bool  # True if the server accepts connections.


def instance_status(
    config: PgEnvConfig, instances: Sequence[Instance] = INSTANCES
) -> list[InstanceStatus]:
    """Report each instance's container, pinning and readiness. It needs no credentials."""
    statuses = []
    for instance in instances:
        current = pg_containers.state(instance.container)
        cpuset = pg_containers.inspect(instance.container, "{{.HostConfig.CpusetCpus}}")
        conninfo = f"host={config.host} port={instance.port} dbname=postgres connect_timeout=3"
        statuses.append(InstanceStatus(instance, current, cpuset, _accepts(conninfo)))
    return statuses


def _accepts(conninfo: str) -> bool:
    """libpq's PQping, as pg_isready uses it: whether the server accepts connections."""
    return psycopg.pq.PGconn.ping(conninfo.encode()) == psycopg.pq.Ping.OK


def table_counts(connect: Connect) -> dict[str, dict[str, int]]:
    """Return the row count of every TPC-H table in both databases."""
    counts: dict[str, dict[str, int]] = {}
    for db in (MAIN_DB, ALT_DB):
        with connect(db) as conn:
            counts[db] = {}
            for table in TPCH_TABLES:
                row = conn.execute(
                    sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
                ).fetchone()
                counts[db][table] = int(row[0]) if row else -1
    return counts


def count_problems(
    counts: Mapping[str, Mapping[str, Mapping[str, int]]], record: GoldenRecord
) -> list[str]:
    """Compare each instance's row counts (instance → db → table) with the golden build's."""
    problems = []
    for name, by_db in counts.items():
        for db, build in record.databases.items():
            for table, entry in build.tables.items():
                got = by_db.get(db, {}).get(table)
                if got != entry.rows:
                    problems.append(f"{name}: {db}.{table} has {got} rows, not {entry.rows}")
    # The robustness check relies on the alternate database differing (spec §4.1).
    main, alt = record.databases.get(MAIN_DB), record.databases.get(ALT_DB)
    if main is not None and alt is not None:
        same = sorted(t for t, e in alt.tables.items() if main.tables[t].rows == e.rows)
        if same != ["nation", "region"]:
            problems.append(f"{ALT_DB} has the same row counts as {MAIN_DB} in {same}")
    return problems


def _instance_dir(config: PgEnvConfig, instance: Instance) -> Path:
    return config.data_root / "instances" / instance.name


def _ensure_copy(config: PgEnvConfig, instance: Instance, record: GoldenRecord) -> bool:
    """Copy the golden data directory for ``instance`` unless it holds this build's copy."""
    directory = _instance_dir(config, instance)
    marker = directory / "golden.json"
    built_at = record.built_at.isoformat()
    if (
        (directory / "pgdata").is_dir()
        and marker.is_file()
        and json.loads(marker.read_text(encoding="utf-8")).get("built_at") == built_at
    ):
        return False
    pg_containers.remove(instance.container, force=True)  # It holds an older copy.
    if (directory / "pgdata").exists():
        shutil.rmtree(directory / "pgdata")
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.golden_pgdata, directory / "pgdata", symlinks=True)
    write_private(marker, json.dumps({"built_at": built_at}) + "\n")
    return True


def _names(roles: Sequence[str]) -> sql.Composed:
    return sql.SQL(", ").join(sql.Identifier(role) for role in roles)


def _verifier(conn: psycopg.Connection[Any], role: str, password: SecretStr) -> str:
    encrypted = conn.pgconn.encrypt_password(
        password.get_secret_value().encode(), role.encode(), b"scram-sha-256"
    )
    return encrypted.decode()
