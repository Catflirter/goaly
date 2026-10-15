import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from goaly.bench.effisql import pg_containers, pg_instances
from goaly.bench.effisql.pg_build import (
    TPCH,
    TPCH_ALT,
    TPCH_TABLES,
    BuildError,
    DbgenBuild,
    GoldenRecord,
)
from goaly.bench.effisql.pg_containers import CONTAINER_PREFIX, ContainerSpec, Mount, run_args
from goaly.bench.effisql.pg_instances import (
    INSTANCES,
    PYTHON_CPUS,
    SERVER_SETTINGS,
    Instance,
    InstanceError,
    Kind,
    apply_roles,
    confine_to_python_cpus,
    connect_statements,
    count_problems,
    database_statements,
    environment_settings,
    fetch_server_settings,
    fetch_server_settings_sync,
    instance_spec,
    instance_status,
    logical_cpus,
    require_golden_record,
    role_statements,
    socket_dirs,
    start_instances,
    stop_instances,
    table_counts,
)
from goaly.bench.effisql.pg_settings import PgCredentials, PgEnvConfig
from goaly.contracts.gen.manifest_pb2 import RunManifest
from tests.bench.effisql.pg_fakes import AsyncFakeConnection, FakeServer, as_text


def test_a_physical_core_is_both_of_its_smt_siblings() -> None:
    assert logical_cpus((2, 3)) == (2, 3, 14, 15)
    assert logical_cpus((0, 1)) == PYTHON_CPUS == (0, 1, 12, 13)


def test_the_instances_are_pinned_as_the_spec_partitions_the_cpus() -> None:
    assert [(i.name, i.kind, i.port, i.cpus) for i in INSTANCES] == [
        ("solve-1", "solving", 55432, (2, 3, 14, 15)),
        ("solve-2", "solving", 55433, (4, 5, 16, 17)),
        ("grade-1", "grading", 55434, (6, 7, 18, 19)),
        ("grade-2", "grading", 55435, (8, 9, 20, 21)),
        ("grade-3", "grading", 55436, (10, 11, 22, 23)),
    ]
    taken = [cpu for instance in INSTANCES for cpu in instance.cpus] + list(PYTHON_CPUS)
    assert sorted(taken) == list(range(24))  # Every CPU has exactly one owner.


def test_an_instance_runs_from_its_own_copy_pinned_and_restarted_by_docker(
    tmp_path: Path,
) -> None:
    config = PgEnvConfig(data_root=tmp_path)

    spec = instance_spec(config, INSTANCES[2])

    assert spec.name == f"{CONTAINER_PREFIX}grade-1"
    assert spec.port == 55434
    assert spec.cpuset == "6,7,18,19"
    assert spec.data_dir == tmp_path / "instances" / "grade-1" / "pgdata"
    assert spec.restart == "unless-stopped"
    assert spec.user == f"{os.getuid()}:{os.getgid()}"
    assert dict(spec.env) == {}  # The data directory is initialized: no secret is needed.
    # Its server's socket directory is the instance's own directory on the host.
    assert spec.mounts == (Mount(tmp_path / "sockets" / "grade-1", "/var/run/postgresql"),)
    assert "type=bind,source=" + str(tmp_path / "sockets" / "grade-1") + (
        ",target=/var/run/postgresql"
    ) in run_args(spec)


def test_every_instance_has_its_own_socket_directory(tmp_path: Path) -> None:
    config = PgEnvConfig(data_root=tmp_path, socket_root=Path("/run/pg"))

    assert socket_dirs(config, INSTANCES[:2]) == {
        "solve-1": Path("/run/pg/solve-1"),
        "solve-2": Path("/run/pg/solve-2"),
    }
    assert (
        socket_dirs(PgEnvConfig(data_root=tmp_path))["grade-3"] == tmp_path / "sockets" / "grade-3"
    )


def test_instance_names_must_stay_in_our_container_namespace() -> None:
    with pytest.raises(ValueError, match="name"):
        Instance("Solve 1", "solving", 55432, (2,))


@pytest.mark.parametrize(
    ("port", "cpus", "complaint"),
    [(80, (2,), "port"), (70000, (2,), "port"), (55432, (), "CPU"), (55432, (-1,), "CPU")],
)
def test_an_instance_needs_a_usable_port_and_cpus(
    port: int, cpus: tuple[int, ...], complaint: str
) -> None:
    with pytest.raises(ValueError, match=complaint):
        Instance("solve-9", "solving", port, cpus)


def test_python_processes_are_confined_to_their_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: calls.append((pid, set(cpus))))

    confine_to_python_cpus()

    assert calls == [(0, {0, 1, 12, 13})]


def _texts(statements: Iterable[object]) -> list[str]:
    return [as_text(statement) for statement in statements]


def test_roles_get_login_a_limit_a_verifier_and_exactly_their_defaults() -> None:
    texts = _texts(
        role_statements({"effisql_agent": "SCRAM-SHA-256$v1", "effisql_grader": "SCRAM-SHA-256$v2"})
    )

    def expected(role: str, limit: int, verifier: str, settings: list[str]) -> list[str]:
        return [
            f'ALTER ROLE "{role}" WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE '
            f"NOREPLICATION NOBYPASSRLS CONNECTION LIMIT {limit} PASSWORD '{verifier}'",
            f'ALTER ROLE "{role}" RESET ALL',  # Drops any default the role set for itself.
            f'ALTER ROLE "{role}" IN DATABASE "tpch" RESET ALL',
            f'ALTER ROLE "{role}" IN DATABASE "tpch_alt" RESET ALL',
            f'ALTER ROLE "{role}" IN DATABASE "postgres" RESET ALL',
            f'ALTER ROLE "{role}" IN DATABASE "template1" RESET ALL',
            *(f'ALTER ROLE "{role}" SET {setting}' for setting in settings),
        ]

    assert texts == expected(
        "effisql_agent",
        64,
        "SCRAM-SHA-256$v1",
        ["\"default_transaction_read_only\" = 'on'", "\"temp_file_limit\" = '5GB'"],
    ) + expected(
        "effisql_grader", -1, "SCRAM-SHA-256$v2", ["\"default_transaction_read_only\" = 'on'"]
    )


REVOKE_CONNECT = [
    f'REVOKE ALL ON DATABASE "{db}" FROM PUBLIC, "effisql_agent", "effisql_grader"'
    for db in ("tpch", "tpch_alt", "postgres", "template1")
]


def test_on_a_solving_instance_the_agent_may_connect_to_tpch_only() -> None:
    assert _texts(connect_statements("solving")) == [
        *REVOKE_CONNECT,
        'GRANT CONNECT ON DATABASE "tpch" TO "effisql_agent", "effisql_grader"',
        'GRANT CONNECT ON DATABASE "tpch_alt" TO "effisql_grader"',
    ]


def test_on_a_grading_instance_the_agent_may_connect_to_nothing() -> None:
    assert _texts(connect_statements("grading")) == [
        *REVOKE_CONNECT,  # Explicit, so a grant copied with the data cannot survive.
        'GRANT CONNECT ON DATABASE "tpch" TO "effisql_grader"',
        'GRANT CONNECT ON DATABASE "tpch_alt" TO "effisql_grader"',
    ]


REVOKED_FUNCTIONS = (
    "REVOKE EXECUTE ON FUNCTION pg_catalog.pg_cancel_backend(integer), "  # One session may not
    "pg_catalog.pg_terminate_backend(integer, bigint), "  # stop another,
    "pg_catalog.pg_stat_get_activity(integer), "  # read the text of another's query,
    "pg_catalog.pg_stat_get_backend_activity(integer), "
    "pg_catalog.lo_creat(integer), pg_catalog.lo_create(oid), "  # or write large objects.
    "pg_catalog.lo_from_bytea(oid, bytea), pg_catalog.lo_import(text), "
    'pg_catalog.lo_import(text, oid) FROM PUBLIC, "effisql_agent", "effisql_grader"'
)
REVOKE_STAT_ACTIVITY = (
    'REVOKE ALL ON "pg_catalog"."pg_stat_activity" FROM PUBLIC, "effisql_agent", "effisql_grader"'
)


@pytest.mark.parametrize(
    ("db", "kind", "readers"),
    [
        ("tpch", "solving", '"effisql_agent", "effisql_grader"'),
        ("tpch", "grading", '"effisql_grader"'),
        ("tpch_alt", "solving", '"effisql_grader"'),
        ("tpch_alt", "grading", '"effisql_grader"'),
    ],
)
def test_each_database_grants_select_to_its_readers_and_revokes_the_rest(
    db: str, kind: Kind, readers: str
) -> None:
    assert _texts(database_statements(db, kind)) == [
        'REVOKE ALL ON SCHEMA "public" FROM PUBLIC, "effisql_agent", "effisql_grader"',
        'REVOKE ALL ON ALL TABLES IN SCHEMA "public" FROM "effisql_agent", "effisql_grader"',
        f'GRANT USAGE ON SCHEMA "public" TO {readers}',
        f'GRANT SELECT ON ALL TABLES IN SCHEMA "public" TO {readers}',
        REVOKED_FUNCTIONS,
        REVOKE_STAT_ACTIVITY,
        'REVOKE USAGE ON LANGUAGE plpgsql FROM PUBLIC, "effisql_agent", "effisql_grader"',
        # Only where the agent reads: it ends its own statement that ignored a cancel.
        *(
            [
                "GRANT EXECUTE ON FUNCTION pg_catalog.pg_terminate_backend(integer, bigint) "
                'TO "effisql_agent"'
            ]
            if (db, kind) == ("tpch", "solving")
            else []
        ),
    ]


CREDENTIALS = PgCredentials(
    admin_password=SecretStr("admin-secret"),
    agent_password=SecretStr("agent-secret"),
    grader_password=SecretStr("grader-secret"),
    _env_file=None,
)


@pytest.mark.parametrize("kind", ["solving", "grading"])
def test_apply_roles_runs_every_statement_for_the_kind_and_sends_only_verifiers(
    kind: Kind,
) -> None:
    server = FakeServer([("FROM pg_roles", [("effisql_grader",)])])

    apply_roles(server.connect, CREDENTIALS, kind)

    verifiers = {
        "effisql_agent": "SCRAM-SHA-256$fake:effisql_agent:12",
        "effisql_grader": "SCRAM-SHA-256$fake:effisql_grader:13",
    }
    assert server.statements("postgres") == [
        "BEGIN",
        "SELECT rolname FROM pg_roles WHERE rolname = ANY(%s)",
        'CREATE ROLE "effisql_agent"',  # The grader exists already.
        *_texts(role_statements(verifiers)),
        *_texts(connect_statements(kind)),
        "COMMIT",
    ]
    for db in ("tpch", "tpch_alt"):
        assert server.statements(db) == ["BEGIN", *_texts(database_statements(db, kind)), "COMMIT"]
    every = " ".join(statement for c in server.connections for statement in c.statements)
    assert "secret" not in every  # No statement holds a password.


SCRAM_HBA = "local all all scram-sha-256\nhost all all 127.0.0.1/32 scram-sha-256\n"


def _golden(config: PgEnvConfig, built_at: str, hba: str = SCRAM_HBA) -> None:
    (config.golden_pgdata / "base").mkdir(parents=True, exist_ok=True)
    (config.golden_pgdata / "PG_VERSION").write_text("16\n")
    (config.golden_pgdata / "pg_hba.conf").write_text(hba)
    record = {
        "image": config.image,
        "dbgen": {"repository": "r", "commit": "c", "version": "v", "binary_sha256": "b"},
        "databases": {},
        "built_at": built_at,
    }
    config.golden_record.write_text(json.dumps(record))


class FakeLifecycle:
    """Stands in for pg_containers and the servers, recording the order of what happened."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, data_root: Path | None = None) -> None:
        self.data_root = data_root
        self.events: list[str] = []
        self.ready: list[str] = []
        self.states: dict[str, str | None] = {}
        self.cpusets: dict[str, str] = {}
        self.exit_codes: dict[str, int] = {}
        self.containers: set[str] = set()
        monkeypatch.setattr(pg_containers, "ensure", self.ensure)
        monkeypatch.setattr(pg_containers, "remove", self.remove)
        monkeypatch.setattr(pg_containers, "state", lambda name: self.states.get(name))
        monkeypatch.setattr(pg_containers, "stop", self.stop)
        monkeypatch.setattr(pg_containers, "inspect", lambda name, template: self.cpusets.get(name))
        monkeypatch.setattr(pg_instances, "wait_until_ready", self.wait)
        monkeypatch.setattr(pg_instances, "connector", lambda *a, **k: FakeServer().connect)

    def ensure(self, spec: ContainerSpec) -> str:
        self.events.append(f"ensure {spec.name}")
        if spec.name in self.containers:
            return "running"
        self.containers.add(spec.name)
        return "created"

    def remove(self, name: str, *, force: bool = False) -> None:
        found = ""
        if self.data_root is not None:  # Which data the container still had when it went.
            version = self.data_root / "instances" / name.removeprefix(CONTAINER_PREFIX)
            version = version / "pgdata" / "PG_VERSION"
            found = f" holding {version.read_text().strip()!r}" if version.exists() else ""
        self.events.append(f"remove {name}{found}")
        self.containers.discard(name)

    def stop(self, name: str, *, timeout_s: int) -> int:
        self.events.append(f"stop {name}")
        return self.exit_codes.get(name, 0)

    def wait(self, connect: object, name: str) -> None:
        self.ready.append(name)


def test_start_copies_the_golden_build_once_and_again_only_for_a_new_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PgEnvConfig(data_root=tmp_path)
    _golden(config, "2026-10-15T05:00:00Z")
    lifecycle = FakeLifecycle(monkeypatch, tmp_path)

    first = start_instances(config, CREDENTIALS)
    second = start_instances(config, CREDENTIALS)
    copy = tmp_path / "instances" / "solve-1" / "pgdata" / "PG_VERSION"
    copy.write_text("touched\n")
    _golden(config, "2026-10-15T06:00:00Z")
    lifecycle.events.clear()
    third = start_instances(config, CREDENTIALS, INSTANCES[:1])

    assert set(first.values()) == {"created, data copied"}
    assert set(second.values()) == {"running"}
    # The running server's container went while it still held the old data, and only then
    # was its data replaced.
    assert lifecycle.events == [
        "remove goaly-effisql-solve-1 holding 'touched'",
        "ensure goaly-effisql-solve-1",
    ]
    assert third == {"solve-1": "created, data copied"}
    assert copy.read_text() == "16\n"  # The new build replaced the old copy.
    assert lifecycle.ready.count("goaly-effisql-grade-3") == 2
    for instance in INSTANCES:  # Each server's socket directory exists before it starts.
        directory = tmp_path / "sockets" / instance.name
        assert directory.is_dir()
        assert directory.stat().st_mode & 0o777 == 0o755


def test_start_refuses_a_socket_path_too_long_for_unix_sockets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PgEnvConfig(data_root=tmp_path, socket_root=tmp_path / ("d" * 100))
    _golden(config, "2026-10-15T05:00:00Z")
    lifecycle = FakeLifecycle(monkeypatch, tmp_path)

    with pytest.raises(InstanceError, match="longer than 107 bytes"):
        start_instances(config, CREDENTIALS, INSTANCES[:1])
    assert lifecycle.events == []  # Refused before any container or copy was touched.
    assert not (tmp_path / "instances").exists()


def test_a_corrupt_copy_marker_means_copy_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PgEnvConfig(data_root=tmp_path)
    _golden(config, "2026-10-15T05:00:00Z")
    lifecycle = FakeLifecycle(monkeypatch, tmp_path)
    start_instances(config, CREDENTIALS, INSTANCES[:1])
    (tmp_path / "instances" / "solve-1" / "golden.json").write_text("{not json")

    assert start_instances(config, CREDENTIALS, INSTANCES[:1]) == {
        "solve-1": "created, data copied"
    }
    assert "remove goaly-effisql-solve-1 holding '16'" in lifecycle.events


def test_a_server_that_does_not_start_is_an_instance_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PgEnvConfig(data_root=tmp_path)
    _golden(config, "2026-10-15T05:00:00Z")
    lifecycle = FakeLifecycle(monkeypatch)

    def refuse(connect: object, name: str) -> None:
        raise BuildError(f"{name} stopped while starting")

    monkeypatch.setattr(pg_instances, "wait_until_ready", refuse)

    with pytest.raises(InstanceError, match="solve-1 did not start"):
        start_instances(config, CREDENTIALS, INSTANCES[:1])
    assert lifecycle.ready == []


def test_starting_no_instance_does_nothing(tmp_path: Path) -> None:
    config = PgEnvConfig(data_root=tmp_path)
    _golden(config, "2026-10-15T05:00:00Z")

    assert start_instances(config, CREDENTIALS, ()) == {}


def test_an_instance_whose_pg_hba_skips_scram_is_never_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PgEnvConfig(data_root=tmp_path)
    _golden(config, "2026-10-15T05:00:00Z", hba=SCRAM_HBA + "host all all ::1/128 trust\n")
    lifecycle = FakeLifecycle(monkeypatch)

    with pytest.raises(InstanceError, match="lets in without a password"):
        start_instances(config, CREDENTIALS)
    assert not any(event.startswith("ensure") for event in lifecycle.events)


def test_start_needs_a_golden_build(tmp_path: Path) -> None:
    with pytest.raises(InstanceError, match="no golden build"):
        start_instances(PgEnvConfig(data_root=tmp_path), CREDENTIALS)


def test_stop_shuts_down_every_instance_that_is_up_and_reports_unclean_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = FakeLifecycle(monkeypatch)
    lifecycle.states = {
        "goaly-effisql-solve-1": "running",
        "goaly-effisql-solve-2": "exited",
        "goaly-effisql-grade-1": "restarting",  # Docker would bring it back otherwise.
        "goaly-effisql-grade-2": "paused",
    }
    lifecycle.exit_codes = {"goaly-effisql-grade-2": 137}

    assert stop_instances(INSTANCES) == {
        "solve-1": "stopped",
        "solve-2": "not running",
        "grade-1": "stopped",
        "grade-2": "stopped with exit code 137",
        "grade-3": "not running",
    }


def test_status_reports_container_cpus_and_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle = FakeLifecycle(monkeypatch)
    lifecycle.states = {"goaly-effisql-solve-1": "running", "goaly-effisql-solve-2": "running"}
    lifecycle.cpusets = {"goaly-effisql-solve-1": "2,3,14,15", "goaly-effisql-solve-2": "2,3"}
    pings: list[str] = []

    def ping(conninfo: str) -> bool:
        pings.append(conninfo)
        socket = "/" in conninfo
        return "port=55432" in conninfo or ("port=55434" in conninfo and not socket)

    monkeypatch.setattr(pg_instances, "_ping", ping)

    statuses = instance_status(PgEnvConfig(data_root=tmp_path), INSTANCES[:3])

    assert [
        (s.instance.name, s.state, s.cpuset, s.accepting, s.socket_accepting, s.healthy)
        for s in statuses
    ] == [
        ("solve-1", "running", "2,3,14,15", True, True, True),
        ("solve-2", "running", "2,3", False, False, False),  # Not accepting, pinned wrongly.
        ("grade-1", None, None, True, False, False),  # Not through its socket; no container.
    ]
    assert not statuses[1].pinned
    assert statuses[0].socket_dir == tmp_path / "sockets" / "solve-1"
    assert pings[:2] == [
        "host=127.0.0.1 port=55432 dbname=postgres connect_timeout=3",
        f"host={tmp_path}/sockets/solve-1 port=55432 dbname=postgres connect_timeout=3",
    ]
    assert "password" not in " ".join(pings)  # Readiness needs no credentials.


def test_table_counts_counts_every_table_of_both_databases() -> None:
    counts = table_counts(FakeServer([("SELECT count(*)", [(7,)])]).connect)

    assert set(counts) == {"tpch", "tpch_alt"}
    assert counts["tpch"] == dict.fromkeys(TPCH_TABLES, 7)


def test_a_count_that_returns_no_row_is_an_error() -> None:
    with pytest.raises(InstanceError, match="returned no row"):
        table_counts(FakeServer().connect)


def _record() -> GoldenRecord:
    return GoldenRecord(
        image="i",
        dbgen=DbgenBuild(repository="r", commit="c", version="v", binary_sha256="b"),
        databases={TPCH.db: TPCH.record(), TPCH_ALT.db: TPCH_ALT.record()},
        built_at=datetime(2026, 10, 15, tzinfo=UTC),
    )


def _counts(**changes: int) -> dict[str, dict[str, int]]:
    counts = {d.db: {t: e.rows for t, e in d.files.items()} for d in (TPCH, TPCH_ALT)}
    counts["tpch"].update(changes)
    return counts


def test_count_problems_finds_nothing_when_every_instance_matches_the_build() -> None:
    assert count_problems({"solve-1": _counts(), "grade-1": _counts()}, _record()) == []


def test_count_problems_names_each_table_that_differs() -> None:
    problems = count_problems({"grade-2": _counts(orders=4_499_999)}, _record())

    assert problems == ["grade-2: tpch.orders has 4499999 rows, not 4500000"]


def test_the_environment_settings_record_what_can_change_results(tmp_path: Path) -> None:
    config = PgEnvConfig(data_root=tmp_path)
    server = {"work_mem": "4MB", "jit": "on"}
    record = _record().model_copy(
        update={"server_version": "16.15", "encoding": "UTF8", "collation": "en_US.utf8"}
    )

    settings = environment_settings(config, record, server)

    checksums = {
        f"data.{dataset.db}.{table}.sha256": entry.sha256
        for dataset in (TPCH, TPCH_ALT)
        for table, entry in dataset.files.items()
    }
    assert settings == {
        "postgres.image": config.image,
        "postgres.shm_size": "64m",
        "postgres.jit": "on",
        "postgres.work_mem": "4MB",
        "golden.image": "i",
        "golden.server_version": "16.15",
        "golden.encoding": "UTF8",
        "golden.collation": "en_US.utf8",
        "golden.built_at": "2026-10-15T00:00:00+00:00",
        "dbgen.repository": "r",
        "dbgen.commit": "c",
        "dbgen.version": "v",
        "dbgen.binary_sha256": "b",
        "data.tpch.scale_factor": "3",
        "data.tpch_alt.scale_factor": "0.3",
        **checksums,
        "cpuset.solve-1": "2,3,14,15",
        "cpuset.solve-2": "4,5,16,17",
        "cpuset.grade-1": "6,7,18,19",
        "cpuset.grade-2": "8,9,20,21",
        "cpuset.grade-3": "10,11,22,23",
        "cpuset.python": "0,1,12,13",
    }
    assert len(checksums) == 2 * len(TPCH_TABLES)
    manifest = RunManifest(environment_settings=settings)
    assert (
        dict(RunManifest.FromString(manifest.SerializeToString()).environment_settings) == settings
    )


def test_an_old_golden_record_leaves_out_the_details_it_lacks(tmp_path: Path) -> None:
    settings = environment_settings(PgEnvConfig(data_root=tmp_path), _record(), {})

    assert [key for key in settings if key.startswith("golden.")] == [
        "golden.image",
        "golden.built_at",
    ]


def _settings_answer(params: object) -> list[tuple[str, str]]:
    """Answers the settings query, as a server that knows every name asked for."""
    assert params == [list(SERVER_SETTINGS)]
    return [(name, "x") for name in SERVER_SETTINGS] + [("enable_hashjoin", "on")]


@pytest.mark.anyio
async def test_server_settings_are_read_as_show_prints_them() -> None:
    conn = AsyncFakeConnection("tpch", [("current_setting", _settings_answer)])

    settings = await fetch_server_settings(conn)  # type: ignore[arg-type]

    assert settings == {**dict.fromkeys(SERVER_SETTINGS, "x"), "enable_hashjoin": "on"}
    assert "LIKE 'enable\\_%'" in conn.statements[0].replace("%%", "%")


def test_the_sync_read_of_the_server_settings_is_the_same() -> None:
    conn = FakeServer([("current_setting", _settings_answer)]).connect("tpch")

    assert fetch_server_settings_sync(conn) == {
        **dict.fromkeys(SERVER_SETTINGS, "x"),
        "enable_hashjoin": "on",
    }


def test_a_server_that_lacks_a_setting_is_refused() -> None:
    conn = FakeServer([("current_setting", [("work_mem", "4MB")])]).connect("tpch")

    with pytest.raises(InstanceError, match="does not report the settings"):
        fetch_server_settings_sync(conn)


def test_count_problems_names_an_alternate_database_that_is_too_similar() -> None:
    record = _record()
    alt = record.databases["tpch_alt"]
    same_orders = alt.model_copy(
        update={"tables": {**alt.tables, "orders": record.databases["tpch"].tables["orders"]}}
    )
    similar = record.model_copy(update={"databases": {**record.databases, "tpch_alt": same_orders}})
    counts = {d: {t: e.rows for t, e in b.tables.items()} for d, b in similar.databases.items()}

    assert count_problems({"solve-1": counts}, similar) == [
        "tpch_alt should differ from tpch in every table but nation and region, "
        "but their row counts are equal in ['nation', 'orders', 'region']"
    ]


def test_the_golden_record_is_required_by_one_helper(tmp_path: Path) -> None:
    config = PgEnvConfig(data_root=tmp_path)
    with pytest.raises(InstanceError, match="no golden build"):
        require_golden_record(config)
    _golden(config, "2026-10-15T05:00:00Z")

    assert require_golden_record(config).built_at == datetime(2026, 10, 15, 5, tzinfo=UTC)
