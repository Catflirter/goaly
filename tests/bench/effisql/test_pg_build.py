import fcntl
import hashlib
import io
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
from pydantic import SecretStr

from goaly.bench.effisql import pg_build, pg_containers
from goaly.bench.effisql.pg_build import (
    DATASETS,
    DBGEN_COMMIT,
    DBGEN_REPOSITORY,
    TPCH,
    TPCH_ALT,
    TPCH_TABLES,
    BuildError,
    Connect,
    Dataset,
    DbgenBuild,
    GoldenRecord,
    TableFile,
    build_golden,
    connector,
    dbgen_binary,
    ensure_dbgen,
    ensure_generated,
    inspect_table_file,
    read_golden_record,
    run_command,
    tpch_ddl,
    without_trailing_delimiter,
)
from goaly.bench.effisql.pg_containers import ContainerSpec
from goaly.bench.effisql.pg_settings import PgEnvConfig
from tests.bench.effisql.pg_fakes import Answer, FakeServer

# The official schema (Diff-SQL docker/tpch/postgresql/schema.sql): every column in order, with
# its type, plus each table's primary and foreign keys.
OFFICIAL_COLUMNS = {
    "region": "r_regionkey INTEGER, r_name CHAR(25), r_comment VARCHAR(152)",
    "nation": "n_nationkey INTEGER, n_name CHAR(25), n_regionkey INTEGER, n_comment VARCHAR(152)",
    "supplier": "s_suppkey INTEGER, s_name CHAR(25), s_address VARCHAR(40), s_nationkey INTEGER, "
    "s_phone CHAR(15), s_acctbal NUMERIC(15,2), s_comment VARCHAR(101)",
    "customer": "c_custkey INTEGER, c_name VARCHAR(25), c_address VARCHAR(40), "
    "c_nationkey INTEGER, c_phone CHAR(15), c_acctbal NUMERIC(15,2), c_mktsegment CHAR(10), "
    "c_comment VARCHAR(117)",
    "part": "p_partkey INTEGER, p_name VARCHAR(55), p_mfgr CHAR(25), p_brand CHAR(10), "
    "p_type VARCHAR(25), p_size INTEGER, p_container CHAR(10), p_retailprice NUMERIC(15,2), "
    "p_comment VARCHAR(23)",
    "partsupp": "ps_partkey INTEGER, ps_suppkey INTEGER, ps_availqty INTEGER, "
    "ps_supplycost NUMERIC(15,2), ps_comment VARCHAR(199)",
    "orders": "o_orderkey BIGINT, o_custkey INTEGER, o_orderstatus CHAR(1), "
    "o_totalprice NUMERIC(15,2), o_orderdate DATE, o_orderpriority CHAR(15), o_clerk CHAR(15), "
    "o_shippriority INTEGER, o_comment VARCHAR(79)",
    "lineitem": "l_orderkey BIGINT, l_partkey INTEGER, l_suppkey INTEGER, l_linenumber INTEGER, "
    "l_quantity NUMERIC(15,2), l_extendedprice NUMERIC(15,2), l_discount NUMERIC(15,2), "
    "l_tax NUMERIC(15,2), l_returnflag CHAR(1), l_linestatus CHAR(1), l_shipdate DATE, "
    "l_commitdate DATE, l_receiptdate DATE, l_shipinstruct CHAR(25), l_shipmode CHAR(10), "
    "l_comment VARCHAR(44)",
}
OFFICIAL_KEYS = {
    "region": ["PRIMARY KEY (r_regionkey)"],
    "nation": [
        "PRIMARY KEY (n_nationkey)",
        "FOREIGN KEY (n_regionkey) REFERENCES region (r_regionkey)",
    ],
    "supplier": [
        "PRIMARY KEY (s_suppkey)",
        "FOREIGN KEY (s_nationkey) REFERENCES nation (n_nationkey)",
    ],
    "customer": [
        "PRIMARY KEY (c_custkey)",
        "FOREIGN KEY (c_nationkey) REFERENCES nation (n_nationkey)",
    ],
    "part": ["PRIMARY KEY (p_partkey)"],
    "partsupp": [
        "PRIMARY KEY (ps_partkey, ps_suppkey)",
        "FOREIGN KEY (ps_partkey) REFERENCES part (p_partkey)",
        "FOREIGN KEY (ps_suppkey) REFERENCES supplier (s_suppkey)",
    ],
    "orders": [
        "PRIMARY KEY (o_orderkey)",
        "FOREIGN KEY (o_custkey) REFERENCES customer (c_custkey)",
    ],
    "lineitem": [
        "PRIMARY KEY (l_orderkey, l_linenumber)",
        "FOREIGN KEY (l_orderkey) REFERENCES orders (o_orderkey)",
        "FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp (ps_partkey, ps_suppkey)",
    ],
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_tables_are_created_and_loaded_in_the_official_order() -> None:
    created = [line.split()[2] for line in tpch_ddl().splitlines() if line.startswith("CREATE")]

    assert created == list(TPCH_TABLES)
    assert list(TPCH_TABLES) == [
        "region",
        "nation",
        "supplier",
        "customer",
        "part",
        "partsupp",
        "orders",
        "lineitem",
    ]


@pytest.mark.parametrize("name", TPCH_TABLES)
def test_each_table_matches_the_official_schema(name: str) -> None:
    ddl = tpch_ddl()
    statement = ddl.split(f"CREATE TABLE {name} (\n", 1)[1].split("\n);", 1)[0]
    lines = [line.strip().rstrip(",") for line in statement.splitlines()]
    columns = ", ".join(line for line in lines if not line.startswith(("PRIMARY", "FOREIGN")))
    keys = [line for line in lines if line.startswith(("PRIMARY", "FOREIGN"))]

    assert columns == OFFICIAL_COLUMNS[name]
    assert keys == OFFICIAL_KEYS[name]


def test_the_ddl_is_one_create_table_per_table_and_nothing_else() -> None:
    ddl = tpch_ddl()

    assert ddl.count("CREATE TABLE") == len(TPCH_TABLES)
    assert "CREATE TABLE region (\n    r_regionkey INTEGER,\n" in ddl
    assert ddl.endswith(");\n")
    assert "INDEX" not in ddl  # Primary and foreign keys only, as the official schema has.


def test_trailing_delimiters_are_trimmed_like_the_official_sed() -> None:
    lines = [b"1|a|\n", b"2|b\n", b"3|c||\n", b"4|d|"]

    assert b"".join(without_trailing_delimiter(io.BytesIO(b"".join(lines)))) == (
        b"1|a\n2|b\n3|c|\n4|d"
    )


def test_lines_are_written_in_blocks_of_bounded_size() -> None:
    source = io.BytesIO(b"x|\n" * 1000)

    blocks = list(without_trailing_delimiter(source, block_bytes=100))

    assert b"".join(blocks) == b"x\n" * 1000
    assert len(blocks) > 1
    assert all(len(block) <= 100 for block in blocks)


def test_a_table_file_is_hashed_and_its_rows_counted(tmp_path: Path) -> None:
    path = tmp_path / "region.tbl"
    path.write_bytes(b"0|A\n1|B\n")

    assert inspect_table_file(path) == TableFile(sha256=_sha256(b"0|A\n1|B\n"), rows=2)


def test_the_committed_checksums_cover_every_table_of_both_databases() -> None:
    assert [dataset.db for dataset in DATASETS] == ["tpch", "tpch_alt"]
    assert (TPCH.scale, TPCH_ALT.scale) == ("3", "0.3")
    for dataset in DATASETS:
        assert list(dataset.files) == list(TPCH_TABLES)
        assert all(len(entry.sha256) == 64 for entry in dataset.files.values())
    assert TPCH.files["lineitem"].rows == 17_996_609
    # region and nation are the same at every scale; every other table differs.
    for name, entry in TPCH.files.items():
        assert (entry == TPCH_ALT.files[name]) == (name in {"region", "nation"})


class FakeRunner:
    """Stands in for git, make and dbgen, recording each command."""

    def __init__(
        self,
        *,
        head: str = DBGEN_COMMIT,
        dirty: bool = False,
        tables: Mapping[str, bytes] | None = None,
        checked_out: bool = False,
    ) -> None:
        self.calls: list[list[str]] = []
        self.head = head
        self.dirty = dirty
        self.tables = dict(tables or {})
        self.checked_out = checked_out  # HEAD resolves only once a checkout has happened.

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        self.calls.append(list(args))
        if args[:2] == ["git", "init"]:
            (Path(args[3]) / ".git").mkdir(parents=True)
        elif "rev-parse" in args:
            if not self.checked_out:
                raise BuildError("git rev-parse failed with exit code 1: ")
            return self.head + "\n"
        elif "checkout" in args:
            self.checked_out = True
        elif "diff" in args and self.dirty:
            raise BuildError("git diff failed with exit code 1")
        elif args[0] == "make":
            Path(args[2]).mkdir(parents=True, exist_ok=True)  # The checkout's dbgen directory.
            (Path(args[2]) / "dbgen").write_bytes(b"binary")
        elif args[0].endswith("dbgen"):
            assert cwd is not None
            assert env is not None
            assert env["DSS_PATH"] == str(cwd)
            assert env["DSS_CONFIG"] == str(Path(args[0]).parent)  # Where dists.dss lives.
            for name, content in self.tables.items():
                (cwd / f"{name}.tbl").write_bytes(content)
        return ""


def test_ensure_dbgen_checks_out_and_builds_the_pinned_commit(tmp_path: Path) -> None:
    runner = FakeRunner()

    build = ensure_dbgen(tmp_path, run=runner)

    source = tmp_path / "tpch-kit"
    assert ["git", "init", "-q", str(source)] in runner.calls
    fetch = ["git", "-C", str(source), "fetch", "-q", "--depth", "1"]
    assert [*fetch, DBGEN_REPOSITORY, DBGEN_COMMIT] in runner.calls
    make = [call for call in runner.calls if call[0] == "make"]
    assert len(make) == 1
    assert "MACHINE=LINUX" in make[0]
    assert "DATABASE=POSTGRESQL" in make[0]
    assert build == DbgenBuild(
        repository=DBGEN_REPOSITORY,
        commit=DBGEN_COMMIT,
        version="2.17.3",
        binary_sha256=_sha256(b"binary"),
    )
    assert dbgen_binary(tmp_path) == source / "dbgen" / "dbgen"


def test_ensure_dbgen_reuses_an_existing_build(tmp_path: Path) -> None:
    ensure_dbgen(tmp_path, run=FakeRunner())
    runner = FakeRunner(checked_out=True)

    ensure_dbgen(tmp_path, run=runner)

    assert not any(call[0] == "make" or "fetch" in call for call in runner.calls)


def test_an_interrupted_checkout_is_started_over(tmp_path: Path) -> None:
    source = tmp_path / "tpch-kit"
    (source / ".git").mkdir(parents=True)  # git init ran; the fetch never finished.
    (source / "stale").write_text("left by the interrupted build")
    runner = FakeRunner()

    assert ensure_dbgen(tmp_path, run=runner).commit == DBGEN_COMMIT

    assert not (source / "stale").exists()
    assert any("fetch" in call for call in runner.calls)
    assert any("checkout" in call for call in runner.calls)


def test_ensure_dbgen_refuses_a_checkout_at_another_commit(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="not the pinned commit"):
        ensure_dbgen(tmp_path, run=FakeRunner(head="0" * 40))


def test_ensure_dbgen_refuses_a_checkout_with_local_changes(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="local changes"):
        ensure_dbgen(tmp_path, run=FakeRunner(dirty=True))


def _synthetic_dataset(contents: Mapping[str, bytes]) -> Dataset:
    files = {
        name: TableFile(sha256=_sha256(content), rows=content.count(b"\n"))
        for name, content in contents.items()
    }
    return Dataset(db="tpch_test", scale="0.01", files=files)


CONTENTS = {name: f"{name}|1\n".encode() for name in TPCH_TABLES}


def test_ensure_generated_runs_dbgen_and_returns_the_verified_files(tmp_path: Path) -> None:
    dataset = _synthetic_dataset(CONTENTS)
    runner = FakeRunner(tables=CONTENTS)
    out = tmp_path / "raw"

    files = ensure_generated(dataset, tmp_path / "dbgen", out, run=runner)

    assert files == dataset.files
    assert runner.calls == [[str(tmp_path / "dbgen"), "-s", "0.01", "-f", "-q"]]


def test_ensure_generated_skips_dbgen_when_the_files_already_match(tmp_path: Path) -> None:
    dataset = _synthetic_dataset(CONTENTS)
    ensure_generated(dataset, tmp_path / "dbgen", tmp_path, run=FakeRunner(tables=CONTENTS))
    runner = FakeRunner()

    assert ensure_generated(dataset, tmp_path / "dbgen", tmp_path, run=runner) == dataset.files
    assert runner.calls == []


def test_stale_files_make_dbgen_run_again(tmp_path: Path) -> None:
    dataset = _synthetic_dataset(CONTENTS)
    for name, content in CONTENTS.items():
        (tmp_path / f"{name}.tbl").write_bytes(content)
    (tmp_path / "lineitem.tbl").write_bytes(b"cut short by a crash")
    runner = FakeRunner(tables=CONTENTS)

    assert ensure_generated(dataset, tmp_path / "dbgen", tmp_path, run=runner) == dataset.files
    assert runner.calls == [[str(tmp_path / "dbgen"), "-s", "0.01", "-f", "-q"]]


def test_commands_are_named_by_program_and_subcommand() -> None:
    assert pg_build._command(["git", "-C", "/data/tpch-kit", "fetch", "-q"]) == "git fetch"
    assert pg_build._command(["/usr/bin/make", "-C", "/data", "dbgen"]) == "make"
    assert pg_build._command(["/data/dbgen/dbgen", "-s", "3"]) == "dbgen"


def test_ensure_generated_rejects_output_that_differs_from_the_pinned_build(
    tmp_path: Path,
) -> None:
    dataset = _synthetic_dataset(CONTENTS)
    changed = {**CONTENTS, "orders": b"orders|2\n"}

    with pytest.raises(BuildError, match=r"differs .* in orders"):
        ensure_generated(dataset, tmp_path / "dbgen", tmp_path, run=FakeRunner(tables=changed))


def _record() -> GoldenRecord:
    return GoldenRecord(
        image=PgEnvConfig().image,
        dbgen=DbgenBuild(
            repository=DBGEN_REPOSITORY,
            commit=DBGEN_COMMIT,
            version="2.17.3",
            binary_sha256="1" * 64,
        ),
        databases={dataset.db: dataset.record() for dataset in DATASETS},
        built_at=datetime(2026, 10, 15, tzinfo=UTC),
    )


def test_a_golden_record_round_trips_through_its_file(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(_record().model_dump_json())

    assert read_golden_record(path) == _record()
    assert read_golden_record(tmp_path / "absent.json") is None


def test_an_unreadable_golden_record_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text("{}")

    with pytest.raises(BuildError, match="golden record"):
        read_golden_record(path)


def _refuse(*args: object, **kwargs: object) -> str:
    raise AssertionError("nothing may run when the golden build is complete")


def _finished(tmp_path: Path, record: GoldenRecord) -> PgEnvConfig:
    config = PgEnvConfig(data_root=tmp_path)
    config.golden_pgdata.mkdir(parents=True)
    config.golden_record.write_text(record.model_dump_json())
    return config


def test_build_golden_returns_a_matching_record_without_building(tmp_path: Path) -> None:
    config = _finished(tmp_path, _record())

    run: Callable[..., str] = _refuse
    assert build_golden(config, SecretStr("unused"), run=run) == _record()


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"image": "postgres@sha256:" + "9" * 64}, "used the image postgres@sha256:999"),
        ({"databases": {TPCH.db: TPCH.record()}}, "lacks database tpch_alt"),
        (
            {"databases": {TPCH.db: TPCH_ALT.record(), TPCH_ALT.db: TPCH_ALT.record()}},
            "its tpch (SF 0.3) differs in customer, lineitem, orders, part, partsupp, supplier",
        ),
    ],
    ids=["image", "missing database", "other data"],
)
def test_a_record_of_another_build_is_refused_naming_the_difference(
    tmp_path: Path, change: dict[str, object], named: str
) -> None:
    config = _finished(tmp_path, _record().model_copy(update=change))

    with pytest.raises(BuildError, match="does not match this setup") as caught:
        build_golden(config, SecretStr("unused"), run=_refuse)
    assert named in str(caught.value)


def test_a_record_whose_data_directory_is_gone_is_refused(tmp_path: Path) -> None:
    config = _finished(tmp_path, _record())
    config.golden_pgdata.rmdir()

    with pytest.raises(BuildError, match=r"data directory .* is missing"):
        build_golden(config, SecretStr("unused"), run=_refuse)


def test_run_command_returns_standard_output() -> None:
    assert run_command(["sh", "-c", "echo out"]) == "out\n"


def test_run_command_reports_a_failure_with_its_standard_error() -> None:
    with pytest.raises(BuildError, match="exit code 3: err"):
        run_command(["sh", "-c", "echo err >&2; exit 3"])


def test_run_command_reports_a_missing_program() -> None:
    with pytest.raises(BuildError, match="not installed"):
        run_command(["goaly-no-such-program"])


CLUSTER: tuple[str, Answer] = ("pg_encoding_to_char", [("16.15", "UTF8", "en_US.utf8")])


class FakeContainers:
    """Stands in for pg_containers: records the build server's lifecycle."""

    def __init__(self, *, exit_code: int = 0, status: str = "running") -> None:
        self.events: list[str] = []
        self.specs: list[ContainerSpec] = []
        self.password_files: list[str] = []
        self.exit_code = exit_code
        self.status = status

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pg_containers, "start", self.start)
        monkeypatch.setattr(pg_containers, "stop", self.stop)
        monkeypatch.setattr(pg_containers, "remove", self.remove)
        monkeypatch.setattr(pg_containers, "state", lambda name: self.status)
        monkeypatch.setattr(pg_containers, "logs", lambda name, lines=30: "FATAL: boom")

    def start(self, spec: ContainerSpec) -> None:
        self.events.append("start")
        self.specs.append(spec)
        secrets_dir = spec.mounts[0].source
        password_file = secrets_dir / "admin_password"
        assert stat.S_IMODE(password_file.stat().st_mode) == 0o600
        self.password_files.append(password_file.read_text())

    def stop(self, name: str, *, timeout_s: int) -> int:
        self.events.append("stop")
        return self.exit_code

    def remove(self, name: str, *, force: bool = False) -> None:
        self.events.append("remove")


def _install_server(
    monkeypatch: pytest.MonkeyPatch, server: FakeServer, containers: FakeContainers
) -> None:
    containers.install(monkeypatch)

    def fake_connector(host: str, port: int, password: SecretStr, *, user: str = "") -> Connect:
        return server.connect

    monkeypatch.setattr(pg_build, "connector", fake_connector)


def test_build_golden_builds_loads_and_records_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PgEnvConfig(data_root=tmp_path)
    dataset = _synthetic_dataset({**CONTENTS, "lineitem": b"1|2|\n3|4|\n"})
    server = FakeServer([("SELECT count(*)", lambda params: [(1,)]), CLUSTER])
    server.responses.insert(0, ('SELECT count(*) FROM "lineitem"', [(2,)]))
    containers = FakeContainers()
    _install_server(monkeypatch, server, containers)
    config.golden_pgdata.mkdir(parents=True)
    (config.golden_pgdata / "stale").write_text("from an unfinished build")
    seen: list[str] = []

    record = build_golden(
        config,
        SecretStr("s3cret"),
        run=FakeRunner(tables={**CONTENTS, "lineitem": b"1|2|\n3|4|\n"}),
        datasets=[dataset],
        after_load=lambda connect: seen.append(connect("tpch_test").info.dbname),
    )

    assert not (config.golden_pgdata / "stale").exists()
    assert stat.S_IMODE(config.golden_pgdata.stat().st_mode) == 0o700
    assert containers.events == ["remove", "start", "stop", "remove"]
    spec = containers.specs[0]
    assert spec.name == config.build_container
    assert spec.data_dir == config.golden_pgdata
    assert spec.cpuset == config.build_cpuset
    assert spec.env["POSTGRES_INITDB_ARGS"] == "--auth=scram-sha-256"
    assert "s3cret" not in str(spec)  # Only the mounted file holds the password.
    assert containers.password_files == ["s3cret\n"]
    assert server.statements("postgres")[0] == 'CREATE DATABASE "tpch_test"'
    loaded = server.statements("tpch_test")
    assert loaded[:3] == ["BEGIN", tpch_ddl(), "COMMIT"]
    copies = [statement for statement in loaded if statement.startswith("COPY")]
    assert copies == [
        f"COPY \"{name}\" FROM STDIN (FORMAT text, DELIMITER '|')" for name in TPCH_TABLES
    ]
    loading = next(c for c in server.connections if c.info.dbname == "tpch_test")
    lineitem_copy = loading.copied[copies[-1]]
    assert lineitem_copy == b"1|2\n3|4\n"
    assert (
        loaded.index("ANALYZE")
        < loaded.index("VACUUM")
        < loaded.index('SELECT count(*) FROM "region"')
    )
    assert seen == ["tpch_test"]
    assert "CHECKPOINT" in server.statements("postgres")
    assert read_golden_record(config.golden_record) == record
    assert stat.S_IMODE(config.golden_record.stat().st_mode) == 0o600
    assert record.databases == {"tpch_test": dataset.record()}
    assert (record.server_version, record.encoding, record.collation) == (
        "16.15",
        "UTF8",
        "en_US.utf8",
    )


def test_two_builds_never_run_at_once(tmp_path: Path) -> None:
    config = PgEnvConfig(data_root=tmp_path)
    lock = config.golden_record.parent / ".lock"
    lock.parent.mkdir(parents=True)
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)

        with pytest.raises(BuildError, match="another golden build"):
            build_golden(config, SecretStr("pw"), run=_refuse)


def test_a_table_that_loads_short_fails_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _synthetic_dataset(CONTENTS)
    server = FakeServer([("SELECT count(*)", [(0,)])])
    containers = FakeContainers()
    _install_server(monkeypatch, server, containers)

    with pytest.raises(BuildError, match=r"tpch_test\.region holds 0 rows, not the 1"):
        build_golden(
            PgEnvConfig(data_root=tmp_path),
            SecretStr("pw"),
            run=FakeRunner(tables=CONTENTS),
            datasets=[dataset],
        )
    assert containers.events[-1] == "remove"  # The build server never outlives the build.
    assert read_golden_record(PgEnvConfig(data_root=tmp_path).golden_record) is None


def test_an_unclean_shutdown_fails_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    containers = FakeContainers(exit_code=137)
    _install_server(monkeypatch, FakeServer([("SELECT count(*)", [(1,)]), CLUSTER]), containers)

    with pytest.raises(BuildError, match="exited with code 137"):
        build_golden(
            PgEnvConfig(data_root=tmp_path),
            SecretStr("pw"),
            run=FakeRunner(tables=CONTENTS),
            datasets=[_synthetic_dataset(CONTENTS)],
        )


def test_a_server_that_dies_while_starting_is_reported_with_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    containers = FakeContainers(status="exited")
    containers.install(monkeypatch)

    def refusing(dbname: str) -> Any:
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(pg_build, "connector", lambda *args, **kwargs: refusing)

    with pytest.raises(BuildError, match="stopped while starting:\nFATAL: boom"):
        build_golden(
            PgEnvConfig(data_root=tmp_path),
            SecretStr("pw"),
            run=FakeRunner(tables=CONTENTS),
            datasets=[_synthetic_dataset(CONTENTS)],
        )


def test_a_server_that_docker_is_restarting_is_still_starting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    containers = FakeContainers(status="restarting")  # Its port was briefly taken, say.
    containers.install(monkeypatch)
    attempts: list[str] = []
    server = FakeServer([("SELECT count(*)", [(1,)]), CLUSTER])

    def connect_on_third_try(dbname: str) -> Any:
        attempts.append(dbname)
        if len(attempts) < 3:
            raise psycopg.OperationalError("connection refused")
        return server.connect(dbname)

    monkeypatch.setattr(pg_build, "connector", lambda *args, **kwargs: connect_on_third_try)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    build_golden(
        PgEnvConfig(data_root=tmp_path),
        SecretStr("pw"),
        run=FakeRunner(tables=CONTENTS),
        datasets=[_synthetic_dataset(CONTENTS)],
    )

    assert attempts[:3] == ["postgres", "postgres", "postgres"]


def test_a_server_that_never_accepts_connections_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeContainers().install(monkeypatch)
    monkeypatch.setattr(pg_build, "_READY_TIMEOUT_S", 0.0)

    def refusing(dbname: str) -> Any:
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(pg_build, "connector", lambda *args, **kwargs: refusing)

    with pytest.raises(BuildError, match="did not accept connections in time"):
        build_golden(
            PgEnvConfig(data_root=tmp_path),
            SecretStr("pw"),
            run=FakeRunner(tables=CONTENTS),
            datasets=[_synthetic_dataset(CONTENTS)],
        )


def test_the_connector_opens_autocommit_connections_with_the_given_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_connect(**kwargs: object) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(psycopg, "connect", fake_connect)

    connect = connector("127.0.0.1", 55439, SecretStr("pw"), user="effisql_grader")

    assert connect("tpch_alt") is sentinel
    assert calls == [
        {
            "host": "127.0.0.1",
            "port": 55439,
            "user": "effisql_grader",
            "password": "pw",
            "dbname": "tpch_alt",
            "autocommit": True,
            "connect_timeout": 10,
        }
    ]


def test_a_directory_that_is_not_a_git_checkout_is_replaced(tmp_path: Path) -> None:
    (tmp_path / "tpch-kit").mkdir()
    runner = FakeRunner()

    assert ensure_dbgen(tmp_path, run=runner).commit == DBGEN_COMMIT
    assert ["git", "init", "-q", str(tmp_path / "tpch-kit")] in runner.calls
