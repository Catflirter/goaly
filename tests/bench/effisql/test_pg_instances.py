import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from goaly.bench.effisql import pg_containers, pg_instances
from goaly.bench.effisql.pg_build import TPCH, TPCH_ALT, TPCH_TABLES, DbgenBuild, GoldenRecord
from goaly.bench.effisql.pg_containers import CONTAINER_PREFIX
from goaly.bench.effisql.pg_instances import (
    INSTANCES,
    PYTHON_CPUS,
    Instance,
    InstanceError,
    apply_roles,
    confine_to_python_cpus,
    count_problems,
    database_statements,
    instance_spec,
    instance_status,
    logical_cpus,
    role_statements,
    start_instances,
    stop_instances,
    table_counts,
)
from goaly.bench.effisql.pg_settings import PgCredentials, PgEnvConfig
from tests.bench.effisql.pg_fakes import FakeServer, as_text


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


def test_instance_names_must_stay_in_our_container_namespace() -> None:
    with pytest.raises(ValueError, match="name"):
        Instance("Solve 1", "solving", 55432, (2,))


def test_python_processes_are_confined_to_their_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: calls.append((pid, set(cpus))))

    confine_to_python_cpus()

    assert calls == [(0, {0, 1, 12, 13})]


def _texts(statements: object) -> list[str]:
    return [as_text(statement) for statement in statements]  # type: ignore[attr-defined]


def test_roles_get_login_a_password_verifier_and_read_only_defaults() -> None:
    texts = _texts(
        role_statements({"effisql_agent": "SCRAM-SHA-256$v1", "effisql_grader": "SCRAM-SHA-256$v2"})
    )

    def expected(role: str, verifier: str) -> list[str]:
        return [
            f'ALTER ROLE "{role}" WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE '
            f"NOREPLICATION NOBYPASSRLS PASSWORD '{verifier}'",
            f'ALTER ROLE "{role}" RESET ALL',  # Drops any default the role set for itself.
            f'ALTER ROLE "{role}" IN DATABASE "tpch" RESET ALL',
            f'ALTER ROLE "{role}" IN DATABASE "tpch_alt" RESET ALL',
            f'ALTER ROLE "{role}" SET default_transaction_read_only = on',
        ]

    assert texts == expected("effisql_agent", "SCRAM-SHA-256$v1") + expected(
        "effisql_grader", "SCRAM-SHA-256$v2"
    )


def test_only_the_grader_may_connect_to_the_alternate_database() -> None:
    texts = _texts(pg_instances.connect_statements())

    assert texts[:4] == [
        'REVOKE ALL ON DATABASE "tpch" FROM PUBLIC',
        'REVOKE ALL ON DATABASE "tpch_alt" FROM PUBLIC',
        'REVOKE ALL ON DATABASE "postgres" FROM PUBLIC',
        'REVOKE ALL ON DATABASE "template1" FROM PUBLIC',
    ]
    assert texts[4:] == [
        'GRANT CONNECT ON DATABASE "tpch" TO "effisql_agent", "effisql_grader"',
        'GRANT CONNECT ON DATABASE "tpch_alt" TO "effisql_grader"',
    ]


@pytest.mark.parametrize(
    ("db", "readers"),
    [("tpch", '"effisql_agent", "effisql_grader"'), ("tpch_alt", '"effisql_grader"')],
)
def test_each_database_grants_select_and_nothing_that_writes(db: str, readers: str) -> None:
    texts = _texts(database_statements(db))

    assert texts[:3] == [
        'REVOKE ALL ON SCHEMA "public" FROM PUBLIC',
        f'GRANT USAGE ON SCHEMA "public" TO {readers}',
        f'GRANT SELECT ON ALL TABLES IN SCHEMA "public" TO {readers}',
    ]
    revoked = " ".join(texts[3:])
    for function in ("pg_cancel_backend", "pg_terminate_backend", "lo_create", "lo_from_bytea"):
        assert function in revoked  # One session may not stop another or write large objects.
    assert all(text.startswith("REVOKE EXECUTE") for text in texts[3:])


CREDENTIALS = PgCredentials(
    admin_password=SecretStr("admin-secret"),
    agent_password=SecretStr("agent-secret"),
    grader_password=SecretStr("grader-secret"),
    _env_file=None,
)


def test_apply_roles_creates_missing_roles_and_sends_only_verifiers() -> None:
    server = FakeServer([("FROM pg_roles", [("effisql_grader",)])])

    apply_roles(server.connect, CREDENTIALS)

    cluster = server.statements("postgres")
    assert cluster[0] == "BEGIN"
    assert 'CREATE ROLE "effisql_agent"' in cluster
    assert 'CREATE ROLE "effisql_grader"' not in cluster  # It already exists.
    assert any("SCRAM-SHA-256$fake:effisql_agent" in statement for statement in cluster)
    every = " ".join(statement for c in server.connections for statement in c.statements)
    assert "secret" not in every  # No statement holds a password.
    assert server.statements("tpch")[1:4] == _texts(database_statements("tpch"))[:3]
    assert server.statements("tpch_alt")[1:4] == _texts(database_statements("tpch_alt"))[:3]


def _golden(config: PgEnvConfig, built_at: str) -> None:
    (config.golden_pgdata / "base").mkdir(parents=True, exist_ok=True)
    (config.golden_pgdata / "PG_VERSION").write_text("16\n")
    record = {
        "image": config.image,
        "dbgen": {"repository": "r", "commit": "c", "version": "v", "binary_sha256": "b"},
        "databases": {},
        "built_at": built_at,
    }
    config.golden_record.write_text(json.dumps(record))


class FakeLifecycle:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.ensured: list[str] = []
        self.ready: list[str] = []
        self.states: dict[str, str | None] = {}
        monkeypatch.setattr(pg_containers, "ensure", self.ensure)
        monkeypatch.setattr(pg_containers, "remove", lambda name, force=False: None)
        monkeypatch.setattr(pg_containers, "state", lambda name: self.states.get(name))
        monkeypatch.setattr(pg_containers, "stop", lambda name, timeout_s: 0)
        monkeypatch.setattr(pg_containers, "inspect", lambda name, template: "2,3,14,15")
        monkeypatch.setattr(
            pg_instances, "wait_until_ready", lambda c, name: self.ready.append(name)
        )
        monkeypatch.setattr(pg_instances, "connector", lambda *a, **k: FakeServer().connect)

    def ensure(self, spec: object) -> str:
        self.ensured.append(spec.name)  # type: ignore[attr-defined]
        return "running" if len(self.ensured) > len(INSTANCES) else "created"


def test_start_copies_the_golden_build_once_and_again_only_for_a_new_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PgEnvConfig(data_root=tmp_path)
    _golden(config, "2026-10-15T05:00:00Z")
    lifecycle = FakeLifecycle(monkeypatch)

    first = start_instances(config, CREDENTIALS)
    second = start_instances(config, CREDENTIALS)
    copy = tmp_path / "instances" / "solve-1" / "pgdata" / "PG_VERSION"
    copy.write_text("touched\n")
    _golden(config, "2026-10-15T06:00:00Z")
    third = start_instances(config, CREDENTIALS, INSTANCES[:1])

    assert set(first.values()) == {"created, data copied"}
    assert set(second.values()) == {"running"}
    assert third == {"solve-1": "running, data copied"}
    assert copy.read_text() == "16\n"  # The new build replaced the old copy.
    assert lifecycle.ready.count("goaly-effisql-grade-3") == 2


def test_start_needs_a_golden_build(tmp_path: Path) -> None:
    with pytest.raises(InstanceError, match="no golden build"):
        start_instances(PgEnvConfig(data_root=tmp_path), CREDENTIALS)


def test_stop_shuts_down_only_running_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    lifecycle = FakeLifecycle(monkeypatch)
    lifecycle.states = {"goaly-effisql-solve-1": "running", "goaly-effisql-solve-2": "exited"}

    assert stop_instances(INSTANCES[:3]) == {
        "solve-1": "stopped",
        "solve-2": "not running",
        "grade-1": "not running",
    }


def test_status_reports_container_cpus_and_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle = FakeLifecycle(monkeypatch)
    lifecycle.states = {"goaly-effisql-solve-1": "running"}
    pings: list[str] = []

    def accepts(conninfo: str) -> bool:
        pings.append(conninfo)
        return "port=55432" in conninfo

    monkeypatch.setattr(pg_instances, "_accepts", accepts)

    statuses = instance_status(PgEnvConfig(data_root=tmp_path), INSTANCES[:2])

    assert [(s.instance.name, s.state, s.cpuset, s.accepting) for s in statuses] == [
        ("solve-1", "running", "2,3,14,15", True),
        ("solve-2", None, "2,3,14,15", False),
    ]
    assert "password" not in " ".join(pings)  # Readiness needs no credentials.


def test_table_counts_counts_every_table_of_both_databases() -> None:
    counts = table_counts(FakeServer([("SELECT count(*)", [(7,)])]).connect)

    assert set(counts) == {"tpch", "tpch_alt"}
    assert counts["tpch"] == dict.fromkeys(TPCH_TABLES, 7)


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
