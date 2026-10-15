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
Every start reapplies the roles and privileges, revoking everything from both roles before it
grants anything, so no stray grant survives a restart. Read-only comes from the privileges,
and the role defaults only add defense in depth:

- the agent role connects only to ``tpch`` on the solving instances, with SELECT on its tables;
  on the grading instances its CONNECT is revoked outright (spec §6.2: solving and grading
  never share an instance). Its temp files are capped at 5 GB, so one runaway sort cannot fill
  the disk the grading instances share, and its connections per instance are capped. As a
  backstop behind the SQL environment's own client-side stop, its statements default to a
  130 s statement_timeout (the 120 s cap plus a margin), and a backend checks every 5 s that
  its client is still there (client_connection_check_interval), so an abandoned statement
  stops soon after its socket closes. A session can lift both for itself; they catch what the
  environment lets slip, not what a hostile client does;
- the grader role connects to both databases on every instance, with SELECT on their tables;
- both default to read-only transactions;
- PUBLIC loses CONNECT and TEMPORARY on every database and CREATE on the public schema. It
  also loses the functions that let one session stop another or write large objects.

Each server also listens on a Unix socket, in a directory bind-mounted from the host
(``~/data/effisql/sockets/<instance>/``; see :func:`socket_dirs`). A sandbox that may not use
IP at all, such as the solve container, mounts the solving instances' directories read-only and
connects with ``host=<directory>``; pg_hba.conf asks for scram there as well (spec §6.1, V28).

The servers listen on ports 55432-55436, inside Linux's ephemeral port range (32768-60999):
a client socket can briefly hold one while its server starts, and Docker's restart policy then
retries. Reserving the ports (``sysctl net.ipv4.ip_local_reserved_ports=55432-55439``) removes
that race.
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
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr

from goaly.bench.effisql import pg_containers
from goaly.bench.effisql.pg_build import (
    TPCH_TABLES,
    BuildError,
    Connect,
    GoldenRecord,
    connector,
    hba_problems,
    read_golden_record,
    wait_until_ready,
)
from goaly.bench.effisql.pg_containers import CONTAINER_PREFIX, SOCKET_DIR, ContainerSpec, Mount
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
    "AGENT_STATEMENT_TIMEOUT_S",
    "GRADING",
    "INSTANCES",
    "PYTHON_CPUS",
    "ROLE_DEFAULTS",
    "SOLVING",
    "Instance",
    "InstanceError",
    "InstanceStatus",
    "Kind",
    "apply_roles",
    "confine_to_python_cpus",
    "connect_statements",
    "count_problems",
    "database_statements",
    "instance_spec",
    "instance_status",
    "logical_cpus",
    "readers",
    "require_golden_record",
    "role_statements",
    "socket_dirs",
    "start_instances",
    "stop_instances",
    "table_counts",
]

_SMT_OFFSET = 12  # Physical core N is logical CPUs N and N + 12 (spec §6.2).
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
_STOP_TIMEOUT_S = 600
_UP = frozenset({"running", "restarting", "paused"})  # States a stop must still shut down.


class InstanceError(Exception):
    """The instances cannot be started or checked; the message says why."""


def logical_cpus(cores: Sequence[int]) -> tuple[int, ...]:
    """Return both SMT siblings of each physical core in ``cores``, in order."""
    return tuple(sorted({cpu for core in cores for cpu in (core, core + _SMT_OFFSET)}))


@dataclass(frozen=True)
class Instance:
    """One PostgreSQL server: its name, what it is for, its port and the CPUs it is pinned to."""

    name: str
    kind: Literal["solving", "grading"]
    port: int
    cpus: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"instance name {self.name!r} must match {_NAME.pattern}")
        if not 1024 <= self.port <= 65535:
            raise ValueError(f"instance {self.name} has port {self.port}, outside 1024-65535")
        if not self.cpus or any(cpu < 0 for cpu in self.cpus):
            raise ValueError(f"instance {self.name} needs at least one CPU, none negative")

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
    """The container of ``instance``. Docker restarts it with the daemon unless we stopped it.

    Its server's Unix-socket directory is a bind mount of the instance's socket directory on
    the host (:meth:`PgEnvConfig.socket_dir`), so a sandbox with no network can still reach it.
    """
    return ContainerSpec(
        name=instance.container,
        image=config.image,
        port=instance.port,
        data_dir=_instance_dir(config, instance) / "pgdata",
        user=f"{os.getuid()}:{os.getgid()}",
        cpuset=",".join(str(cpu) for cpu in instance.cpus),
        restart="unless-stopped",
        mounts=(Mount(config.socket_dir(instance.name), SOCKET_DIR),),
    )


def socket_dirs(config: PgEnvConfig, instances: Sequence[Instance] = INSTANCES) -> dict[str, Path]:
    """Each instance's Unix-socket directory on the host, by instance name.

    libpq connects through one with ``host=<directory>`` and the instance's port, which names
    the socket file. The solve container mounts only the solving instances' directories.
    """
    return {instance.name: config.socket_dir(instance.name) for instance in instances}


type Kind = Literal["solving", "grading"]

_ROLES = (AGENT_ROLE, GRADER_ROLE)
# Databases a role's per-database defaults may live in; a reapply clears them all.
_DATABASES = (MAIN_DB, ALT_DB, "postgres", "template1")
# The server-side backstop on an agent statement's run time, above the 120 s client-side cap.
AGENT_STATEMENT_TIMEOUT_S = 130
# Role-wide defaults, each written as SHOW prints it. temp_file_limit is superuser-only, so the
# agent cannot lift its cap; the others a session may change for itself.
ROLE_DEFAULTS: Mapping[str, Mapping[str, str]] = {
    AGENT_ROLE: {
        "default_transaction_read_only": "on",
        "temp_file_limit": "5GB",
        "statement_timeout": f"{AGENT_STATEMENT_TIMEOUT_S}s",
        "client_connection_check_interval": "5s",
    },
    GRADER_ROLE: {"default_transaction_read_only": "on"},
}
_CONNECTION_LIMITS = {AGENT_ROLE: 64, GRADER_ROLE: -1}  # Per instance; -1 means no limit.
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


def readers(kind: Kind, db: str) -> tuple[str, ...]:
    """The roles that may connect to ``db`` on an instance of ``kind`` and read its tables."""
    if db == MAIN_DB:
        return (AGENT_ROLE, GRADER_ROLE) if kind == "solving" else (GRADER_ROLE,)
    if db == ALT_DB:
        return (GRADER_ROLE,)
    return ()


def role_statements(verifiers: Mapping[str, str]) -> list[sql.Composed]:
    """Statements that give each role login with its SCRAM verifier, its connection limit and
    exactly :data:`ROLE_DEFAULTS`: any other default, such as one a role set itself, is removed."""
    statements = []
    for role, verifier in verifiers.items():
        name = sql.Identifier(role)
        attributes = "LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        statements += [
            # The limit is our own integer, written as is: a negative Literal gets a space.
            sql.SQL(
                f"ALTER ROLE {{}} WITH {attributes} "
                f"CONNECTION LIMIT {int(_CONNECTION_LIMITS[role])} PASSWORD {{}}"
            ).format(name, sql.Literal(verifier)),
            sql.SQL("ALTER ROLE {} RESET ALL").format(name),
            *(
                sql.SQL("ALTER ROLE {} IN DATABASE {} RESET ALL").format(name, sql.Identifier(db))
                for db in _DATABASES
            ),
            *(
                sql.SQL("ALTER ROLE {} SET {} = {}").format(
                    name, sql.Identifier(setting), sql.Literal(value)
                )
                for setting, value in ROLE_DEFAULTS[role].items()
            ),
        ]
    return statements


def connect_statements(kind: Kind) -> list[sql.Composed]:
    """Statements that let only our roles connect, each to the databases it may use here.

    Every right is revoked first, from PUBLIC and from both roles, so that a right granted
    before, such as the agent's CONNECT in a copy taken from a solving instance, never survives.
    """
    revoke = [
        sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC, {}").format(
            sql.Identifier(db), _names(_ROLES)
        )
        for db in _DATABASES
    ]
    grant = [
        sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(db), _names(roles))
        for db in (MAIN_DB, ALT_DB)
        if (roles := readers(kind, db))
    ]
    return revoke + grant


def database_statements(db: str, kind: Kind) -> list[sql.Composed]:
    """Statements, run in ``db``, that grant SELECT on its tables to its readers, nothing that
    writes, and nothing to anyone else."""
    public = sql.Identifier("public")
    functions = sql.SQL(", ").join(sql.SQL(function) for function in _REVOKED_FUNCTIONS)
    statements = [
        sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC, {}").format(public, _names(_ROLES)),
        sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM {}").format(public, _names(_ROLES)),
    ]
    if roles := readers(kind, db):
        statements += [
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(public, _names(roles)),
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(public, _names(roles)),
        ]
    statements.append(sql.SQL("REVOKE EXECUTE ON FUNCTION {} FROM PUBLIC").format(functions))
    return statements


def apply_roles(connect: Connect, credentials: PgCredentials, kind: Kind) -> None:
    """Create the agent and grader roles if needed and (re)apply their passwords and rights on
    an instance of ``kind``.

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
        for statement in role_statements(verifiers) + connect_statements(kind):
            conn.execute(statement)
    for db in (MAIN_DB, ALT_DB):
        with connect(db) as conn, conn.transaction():
            for statement in database_statements(db, kind):
                conn.execute(statement)


def require_golden_record(config: PgEnvConfig) -> GoldenRecord:
    """The golden build's record; raise InstanceError if there is no complete build yet."""
    record = read_golden_record(config.golden_record)
    if record is None:
        raise InstanceError("there is no golden build yet: run `pgenv.py build` first")
    return record


def start_instances(
    config: PgEnvConfig, credentials: PgCredentials, instances: Sequence[Instance] = INSTANCES
) -> dict[str, str]:
    """Bring every instance up from the golden build, and say what each took. Idempotent."""
    record = require_golden_record(config)
    if not instances:
        return {}
    for instance in instances:
        _check_socket_path(config, instance)
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
        hba = _instance_dir(config, instance) / "pgdata" / "pg_hba.conf"
        problems = hba_problems(hba.read_text(encoding="utf-8"))
        if problems:  # Each server shares the host network: no rule may skip scram.
            raise InstanceError(f"{hba} lets in without a password: {problems}")
        _make_socket_dir(config, instance)
        action = pg_containers.ensure(instance_spec(config, instance))
        actions[instance.name] = f"{action}, data copied" if copied[instance] else action
    for instance in instances:
        connect = connector(config.host, instance.port, credentials.admin_password)
        try:
            wait_until_ready(connect, instance.container)
        except BuildError as error:
            raise InstanceError(f"{instance.name} did not start: {error}") from None
        apply_roles(connect, credentials, instance.kind)
    return actions


# The longest path a Unix socket may have (sun_path holds 108 bytes, with its terminator).
_MAX_SOCKET_PATH = 107


def _check_socket_path(config: PgEnvConfig, instance: Instance) -> None:
    socket = config.socket_dir(instance.name) / f".s.PGSQL.{instance.port}"
    if len(os.fsencode(socket)) > _MAX_SOCKET_PATH:
        raise InstanceError(
            f"the socket path {socket} is longer than {_MAX_SOCKET_PATH} bytes: "
            "choose a shorter socket_root"
        )


def _make_socket_dir(config: PgEnvConfig, instance: Instance) -> None:
    """Create the instance's socket directory: its server writes there, anyone may look up."""
    directory = config.socket_dir(instance.name)
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o755)  # Clients only need to reach the socket; scram still guards it.


def stop_instances(instances: Sequence[Instance] = INSTANCES) -> dict[str, str]:
    """Shut every running instance down cleanly; its container and data stay for a restart."""
    results: dict[str, str] = {}
    for instance in instances:
        if pg_containers.state(instance.container) not in _UP:
            results[instance.name] = "not running"
            continue
        code = pg_containers.stop(instance.container, timeout_s=_STOP_TIMEOUT_S)
        results[instance.name] = "stopped" if code == 0 else f"stopped with exit code {code}"
    return results


@dataclass(frozen=True)
class InstanceStatus:
    """What :func:`instance_status` found of one instance."""

    instance: Instance
    state: str | None  # The container's state, or None if there is no container.
    cpuset: str | None  # The CPUs Docker pins it to.
    accepting: bool  # True if the server accepts connections over TCP.
    socket_dir: Path  # The host directory of its Unix socket.
    socket_accepting: bool  # True if the server accepts connections through that socket.

    @property
    def pinned(self) -> bool:
        """Whether Docker pins the container to exactly the instance's CPUs."""
        return self.cpuset == ",".join(str(cpu) for cpu in self.instance.cpus)

    @property
    def healthy(self) -> bool:
        return self.state == "running" and self.accepting and self.socket_accepting and self.pinned


def instance_status(
    config: PgEnvConfig, instances: Sequence[Instance] = INSTANCES
) -> list[InstanceStatus]:
    """Report each instance's container, pinning and readiness, over TCP and through its
    socket. It needs no credentials."""
    statuses = []
    for instance in instances:
        current = pg_containers.state(instance.container)
        cpuset = pg_containers.inspect(instance.container, "{{.HostConfig.CpusetCpus}}")
        sockets = config.socket_dir(instance.name)
        statuses.append(
            InstanceStatus(
                instance,
                current,
                cpuset,
                accepting=_accepts(config.host, instance.port),
                socket_dir=sockets,
                socket_accepting=_accepts(str(sockets), instance.port),
            )
        )
    return statuses


def _accepts(host: str, port: int) -> bool:
    """Whether the server accepts connections. A ``host`` that is a directory means the Unix
    socket in it."""
    return _ping(make_conninfo(host=host, port=port, dbname="postgres", connect_timeout=3))


def _ping(conninfo: str) -> bool:
    """libpq's PQping, as pg_isready uses it."""
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
                if row is None:
                    raise InstanceError(f"count(*) of {db}.{table} returned no row")
                counts[db][table] = int(row[0])
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
            problems.append(
                f"{ALT_DB} should differ from {MAIN_DB} in every table but nation and region, "
                f"but their row counts are equal in {same}"
            )
    return problems


def _instance_dir(config: PgEnvConfig, instance: Instance) -> Path:
    return config.data_root / "instances" / instance.name


def _ensure_copy(config: PgEnvConfig, instance: Instance, record: GoldenRecord) -> bool:
    """Copy the golden data directory for ``instance`` unless it holds this build's copy."""
    directory = _instance_dir(config, instance)
    marker = directory / "golden.json"
    built_at = record.built_at.isoformat()
    if (directory / "pgdata").is_dir() and _copied_at(marker) == built_at:
        return False
    pg_containers.remove(instance.container, force=True)  # It holds an older copy.
    marker.unlink(missing_ok=True)  # Gone before the data, so a crash midway copies again.
    if (directory / "pgdata").exists():
        shutil.rmtree(directory / "pgdata")
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.golden_pgdata, directory / "pgdata", symlinks=True)
    write_private(marker, json.dumps({"built_at": built_at}) + "\n")
    return True


def _copied_at(marker: Path) -> str | None:
    """The build the copy came from, or None if the marker is missing or unreadable."""
    try:
        value = json.loads(marker.read_text(encoding="utf-8")).get("built_at")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def _names(roles: Sequence[str]) -> sql.Composed:
    return sql.SQL(", ").join(sql.Identifier(role) for role in roles)


def _verifier(conn: psycopg.Connection[Any], role: str, password: SecretStr) -> str:
    encrypted = conn.pgconn.encrypt_password(
        password.get_secret_value().encode(), role.encode(), b"scram-sha-256"
    )
    return encrypted.decode()
