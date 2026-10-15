"""Build the Effi-SQL databases (spec §4.1): TPC-H from a pinned dbgen, loaded into one golden
PostgreSQL data directory that every instance copies.

The load follows the official TPC-H container: the schema with its primary and foreign keys,
then each table copied in, in key order, with each line's trailing delimiter trimmed, then
ANALYZE. A VACUUM follows, as autovacuum would run soon after a bulk load, so every copy starts
from the same settled state. Each step is idempotent, and a finished build leaves its record,
which names the dbgen build and every generated file's checksum, so a rebuild can be verified.
"""

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import psycopg
from psycopg import sql
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from goaly.bench.effisql import pg_containers
from goaly.bench.effisql.pg_containers import ContainerSpec, Mount
from goaly.bench.effisql.pg_settings import ADMIN_ROLE, ALT_DB, MAIN_DB, PgEnvConfig, write_private

__all__ = [
    "DATASETS",
    "DBGEN_COMMIT",
    "DBGEN_REPOSITORY",
    "DBGEN_VERSION",
    "TPCH",
    "TPCH_ALT",
    "TPCH_TABLES",
    "BuildError",
    "Connect",
    "DatabaseBuild",
    "Dataset",
    "DbgenBuild",
    "GoldenRecord",
    "Runner",
    "TableFile",
    "build_golden",
    "connector",
    "dbgen_binary",
    "ensure_dbgen",
    "ensure_generated",
    "hba_problems",
    "inspect_table_file",
    "load_dataset",
    "read_golden_record",
    "run_command",
    "tpch_ddl",
    "without_trailing_delimiter",
]

# tpch-kit: TPC-H dbgen 2.17.3 with fixes for modern compilers. It writes no trailing delimiter,
# so the official loader's trim leaves its lines unchanged.
DBGEN_REPOSITORY = "https://github.com/gregrahn/tpch-kit.git"
DBGEN_COMMIT = "852ad0a5ee31ebefeed884cea4188781dd9613a3"
DBGEN_VERSION = "2.17.3"
# dbgen is C89; GCC 14 and later reject parts of it under their default standard.
_DBGEN_CC = "gcc -std=gnu89 -fcommon"

_SECRETS_TARGET = "/run/goaly"
_READY_TIMEOUT_S = 300.0
_STOP_TIMEOUT_S = 600

_log = logging.getLogger(__name__)


class BuildError(Exception):
    """A build step failed; the message says which and why."""


# The official load order: each table after the tables it references.
TPCH_TABLES = ("region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem")


def tpch_ddl() -> str:
    """Return the TPC-H schema's CREATE TABLE statements (tpch_schema.sql), in load order.

    Constraints get PostgreSQL's default names, the same ones the official schema's get, such
    as ``lineitem_pkey``, so plans and index names read the same as in the dataset.
    """
    return resources.files(__package__).joinpath("tpch_schema.sql").read_text(encoding="utf-8")


class TableFile(BaseModel):
    """A generated ``.tbl`` file: its SHA-256 and how many rows (lines) it holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str
    rows: int


class DatabaseBuild(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scale_factor: str
    tables: dict[str, TableFile]


@dataclass(frozen=True)
class Dataset:
    """A database to build: its name, its dbgen scale factor, and the files dbgen must write."""

    db: str
    scale: str
    files: Mapping[str, TableFile]

    def record(self) -> DatabaseBuild:
        return DatabaseBuild(scale_factor=self.scale, tables=dict(self.files))


def _datasets() -> tuple[Dataset, ...]:
    """What the pinned dbgen writes (tpch_files.json). A rebuild must reproduce these files
    exactly: the parity check against the dataset's plans holds for them."""
    text = resources.files(__package__).joinpath("tpch_files.json").read_text(encoding="utf-8")
    return tuple(
        Dataset(db, entry["scale"], {n: TableFile(**f) for n, f in entry["files"].items()})
        for db, entry in json.loads(text).items()
    )


DATASETS = _datasets()
TPCH, TPCH_ALT = (next(d for d in DATASETS if d.db == db) for db in (MAIN_DB, ALT_DB))


class DbgenBuild(BaseModel):
    """The dbgen that generated the data: its source and its binary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str
    commit: str
    version: str
    binary_sha256: str


class GoldenRecord(BaseModel):
    """What the golden data directory holds, and what built it. Written last, so it exists
    only for a complete build."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str
    dbgen: DbgenBuild
    databases: dict[str, DatabaseBuild]
    built_at: datetime
    # What initdb and the image made of the cluster. None in a record from before they were kept.
    server_version: str | None = None
    encoding: str | None = None
    collation: str | None = None


class Runner(Protocol):
    """Runs a command and returns its standard output, raising BuildError if it fails."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str: ...


# Opens an autocommit connection to the named database; see connector().
type Connect = Callable[[str], psycopg.Connection[Any]]


def run_command(
    args: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None
) -> str:
    """Run ``args`` and return its standard output; raise BuildError if it fails."""
    executable = shutil.which(args[0], path=None if env is None else env.get("PATH"))
    if executable is None:
        raise BuildError(f"{args[0]} is not installed or not on PATH")
    # The arguments are paths, the pinned commit and constants, never untrusted input.
    completed = subprocess.run(  # noqa: S603
        [executable, *args[1:]], cwd=cwd, env=env, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-800:]
        raise BuildError(f"{_command(args)} failed with exit code {completed.returncode}: {detail}")
    return completed.stdout


def _command(args: Sequence[str]) -> str:
    """Name a command by its program and subcommand, such as ``git fetch``; paths are omitted."""
    words = [Path(args[0]).name]
    rest = list(args[1:])
    if words[0] == "git" and rest[:1] == ["-C"]:
        rest = rest[2:]  # git -C <checkout> <subcommand> ...
    words += [arg for arg in rest[:1] if not arg.startswith("-")]
    return " ".join(words)


def dbgen_binary(dbgen_dir: Path) -> Path:
    return dbgen_dir / "tpch-kit" / "dbgen" / "dbgen"


def ensure_dbgen(dbgen_dir: Path, *, run: Runner = run_command) -> DbgenBuild:
    """Check out the pinned tpch-kit commit under ``dbgen_dir`` and build dbgen, unless done."""
    source = dbgen_dir / "tpch-kit"
    git = ["git", "-C", str(source)]
    if not (source / ".git").is_dir() or _head(run, git) != DBGEN_COMMIT:
        # Missing, or left at another commit or half-fetched by an interrupted build: the
        # checkout is ours, under the data root, so start it over.
        if source.exists():
            shutil.rmtree(source)
        run(["git", "init", "-q", str(source)])
        run([*git, "fetch", "-q", "--depth", "1", DBGEN_REPOSITORY, DBGEN_COMMIT])
        run([*git, "checkout", "-q", "--detach", "FETCH_HEAD"])
    head = _head(run, git)
    if head != DBGEN_COMMIT:
        raise BuildError(f"{source} is at {head}, not the pinned commit {DBGEN_COMMIT}")
    try:
        run([*git, "diff", "--quiet", "HEAD"])
    except BuildError:
        raise BuildError(f"{source} has local changes; restore it to {DBGEN_COMMIT}") from None
    binary = dbgen_binary(dbgen_dir)
    if not binary.is_file():
        build = ["make", "-C", str(binary.parent), "MACHINE=LINUX", "DATABASE=POSTGRESQL"]
        run([*build, f"CC={_DBGEN_CC}", "dbgen"])
    return DbgenBuild(
        repository=DBGEN_REPOSITORY,
        commit=head,
        version=DBGEN_VERSION,
        binary_sha256=inspect_table_file(binary).sha256,
    )


def _head(run: Runner, git: Sequence[str]) -> str | None:
    """The commit the checkout is at, or None if it has none yet."""
    try:
        return run([*git, "rev-parse", "--verify", "--quiet", "HEAD^{commit}"]).strip() or None
    except BuildError:
        return None


def inspect_table_file(path: Path) -> TableFile:
    """Return the SHA-256 of the file at ``path`` and how many lines it holds."""
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
            rows += block.count(b"\n")
    return TableFile(sha256=digest.hexdigest(), rows=rows)


def ensure_generated(
    dataset: Dataset, binary: Path, out_dir: Path, *, run: Runner = run_command
) -> dict[str, TableFile]:
    """Make ``out_dir`` hold exactly the files ``dataset`` names, running dbgen if needed."""
    paths = {name: out_dir / f"{name}.tbl" for name in dataset.files}
    if all(path.is_file() for path in paths.values()):
        files = {name: inspect_table_file(path) for name, path in paths.items()}
        if files == dataset.files:
            return files
    out_dir.mkdir(parents=True, exist_ok=True)
    _log.info("generating TPC-H", extra={"db": dataset.db, "scale": dataset.scale})
    env = {**os.environ, "DSS_CONFIG": str(binary.parent), "DSS_PATH": str(out_dir)}
    run([str(binary), "-s", dataset.scale, "-f", "-q"], cwd=out_dir, env=env)
    files = {name: inspect_table_file(path) for name, path in paths.items()}
    different = [name for name, entry in dataset.files.items() if files[name] != entry]
    if different:
        raise BuildError(
            f"dbgen's output for {dataset.db} (SF {dataset.scale}) differs from the pinned build "
            f"in {', '.join(different)}: the data would not match the official setup's"
        )
    return files


def without_trailing_delimiter(source: BinaryIO, *, block_bytes: int = 1 << 20) -> Iterator[bytes]:
    """Yield ``source``'s lines with one trailing ``|`` removed from each, in bounded blocks.

    This is the official loader's ``sed 's/|$//'``: dbgen's classic output ends every line with
    the delimiter, which COPY would read as an extra empty column.
    """
    block: list[bytes] = []
    size = 0
    for line in source:
        if line.endswith(b"|\n"):
            line = line[:-2] + b"\n"
        elif line.endswith(b"|"):
            line = line[:-1]
        if size + len(line) > block_bytes and block:
            yield b"".join(block)
            block, size = [], 0
        block.append(line)
        size += len(line)
    if block:
        yield b"".join(block)


def load_dataset(connect: Connect, dataset: Dataset, raw_dir: Path) -> None:
    """Create ``dataset``'s database and load it the official way, then VACUUM and verify it."""
    with connect("postgres") as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dataset.db)))
    with connect(dataset.db) as conn:
        with conn.transaction():
            conn.execute(tpch_ddl())
        for table in TPCH_TABLES:
            _log.info("loading table", extra={"db": dataset.db, "table": table})
            copy_sql = sql.SQL("COPY {} FROM STDIN (FORMAT text, DELIMITER '|')")
            with (
                conn.cursor() as cursor,
                cursor.copy(copy_sql.format(sql.Identifier(table))) as copy,
                (raw_dir / f"{table}.tbl").open("rb") as source,
            ):
                for block in without_trailing_delimiter(source):
                    copy.write(block)
        conn.execute("ANALYZE")
        # Not in the official load: an approved deviation from the spec. Autovacuum would run it
        # soon after the bulk load anyway; running it here gives all 5 copies the same
        # visibility map and hint bits from the start.
        conn.execute("VACUUM")
        for table in TPCH_TABLES:
            count_sql = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
            row = conn.execute(count_sql).fetchone()
            loaded = row[0] if row else None
            if loaded != dataset.files[table].rows:
                raise BuildError(
                    f"{dataset.db}.{table} holds {loaded} rows, "
                    f"not the {dataset.files[table].rows} its file has"
                )


def read_golden_record(path: Path) -> GoldenRecord | None:
    """Return the golden build's record, or None if the build never finished."""
    if not path.exists():
        return None
    try:
        return GoldenRecord.model_validate_json(path.read_bytes())
    except ValidationError:
        raise BuildError(
            f"the golden record {path} is unreadable; delete {path.parent} to rebuild"
        ) from None


def build_golden(
    config: PgEnvConfig,
    admin_password: SecretStr,
    *,
    run: Runner = run_command,
    datasets: Sequence[Dataset] = DATASETS,
    after_load: Callable[[Connect], None] | None = None,
) -> GoldenRecord:
    """Build the golden data directory unless it is complete, and return its record.

    An unfinished build is discarded and started over. ``after_load`` runs while the build
    server still holds the loaded databases, before its clean shutdown.
    """
    with _build_lock(config.golden_record.parent / ".lock"):
        existing = read_golden_record(config.golden_record)
        if existing is not None:
            _check_record(existing, config, datasets)
            return existing
        dbgen = ensure_dbgen(config.dbgen_dir, run=run)
        binary = dbgen_binary(config.dbgen_dir)
        databases = {
            dataset.db: DatabaseBuild(
                scale_factor=dataset.scale,
                tables=ensure_generated(dataset, binary, config.raw_dir(dataset.scale), run=run),
            )
            for dataset in datasets
        }
        # A build container left over from a failed build may still hold the data directory.
        pg_containers.remove(config.build_container, force=True)
        pgdata = config.golden_pgdata
        if pgdata.exists():
            shutil.rmtree(pgdata)
        pgdata.mkdir(mode=0o700, parents=True)
        with _build_server(config, admin_password, pgdata) as connect:
            for dataset in datasets:
                load_dataset(connect, dataset, config.raw_dir(dataset.scale))
            cluster = _cluster_facts(connect, datasets[0].db)
            if after_load is not None:
                after_load(connect)
        record = GoldenRecord(
            image=config.image,
            dbgen=dbgen,
            databases=databases,
            built_at=datetime.now(UTC),
            **cluster,
        )
        write_private(config.golden_record, record.model_dump_json(indent=2) + "\n")
        _log.info("golden build complete", extra={"pgdata": str(pgdata)})
        return record


@contextmanager
def _build_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock for the whole build, so two setups never build at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BuildError(f"another golden build holds {path}; wait for it to finish") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _cluster_facts(connect: Connect, db: str) -> dict[str, str]:
    with connect(db) as conn:
        row = conn.execute(
            "SELECT current_setting('server_version'), pg_encoding_to_char(encoding), datcollate"
            " FROM pg_database WHERE datname = current_database()"
        ).fetchone()
    if row is None:
        raise BuildError(f"{db} is missing from pg_database")
    return {"server_version": row[0], "encoding": row[1], "collation": row[2]}


def _check_record(record: GoldenRecord, config: PgEnvConfig, datasets: Sequence[Dataset]) -> None:
    """Fail unless the finished build ``record`` describes is the one this setup asks for."""
    problems = []
    if record.image != config.image:
        problems.append(f"it used the image {record.image}, not {config.image}")
    if record.dbgen.commit != DBGEN_COMMIT:
        problems.append(f"it used dbgen {record.dbgen.commit}, not {DBGEN_COMMIT}")
    expected = {dataset.db: dataset.record() for dataset in datasets}
    for db in sorted(record.databases.keys() | expected.keys()):
        built, wanted = record.databases.get(db), expected.get(db)
        if built is None or wanted is None:
            problems.append(f"it {'lacks' if built is None else 'has an extra'} database {db}")
        elif built != wanted:
            tables = sorted(
                t
                for t in built.tables.keys() | wanted.tables.keys()
                if built.tables.get(t) != wanted.tables.get(t)
            )
            problems.append(
                f"its {db} (SF {built.scale_factor}) differs in {', '.join(tables) or 'scale'}"
            )
    if not config.golden_pgdata.is_dir():
        problems.append(f"its data directory {config.golden_pgdata} is missing")
    if problems:
        raise BuildError(
            f"the golden build recorded in {config.golden_record} does not match this setup: "
            f"{'; '.join(problems)}. Delete {config.golden_record.parent} to rebuild it."
        )


@contextmanager
def _build_server(config: PgEnvConfig, password: SecretStr, pgdata: Path) -> Iterator[Connect]:
    """Run a server that initializes ``pgdata``; on success, check that it shut down cleanly."""
    name = config.build_container
    with tempfile.TemporaryDirectory(prefix="goaly-pg-") as secrets_dir:
        write_private(Path(secrets_dir) / "admin_password", password.get_secret_value() + "\n")
        spec = ContainerSpec(
            name=name,
            image=config.image,
            port=config.build_port,
            data_dir=pgdata,
            user=f"{os.getuid()}:{os.getgid()}",
            cpuset=config.build_cpuset,
            env={
                "POSTGRES_USER": ADMIN_ROLE,
                "POSTGRES_DB": "postgres",
                "POSTGRES_PASSWORD_FILE": f"{_SECRETS_TARGET}/admin_password",
                # Password authentication for every connection, local ones included.
                "POSTGRES_INITDB_ARGS": "--auth=scram-sha-256",
            },
            mounts=(Mount(Path(secrets_dir), _SECRETS_TARGET, read_only=True),),
        )
        pg_containers.start(spec)
        try:
            connect = connector(config.host, config.build_port, password)
            _wait_until_ready(connect, name)
            problems = hba_problems((pgdata / "pg_hba.conf").read_text(encoding="utf-8"))
            if problems:
                raise BuildError(f"{pgdata}/pg_hba.conf lets in without a password: {problems}")
            yield connect
            with connect("postgres") as conn:
                conn.execute("CHECKPOINT")
            exit_code = pg_containers.stop(name, timeout_s=_STOP_TIMEOUT_S)
            if exit_code != 0:
                raise BuildError(f"the build server exited with code {exit_code}, not cleanly")
        finally:
            pg_containers.remove(name, force=True)


def hba_problems(text: str) -> list[str]:
    """Say what in a pg_hba.conf would let a connection in without a password.

    Every server shares the host network, so every rule, local socket and TCP alike, must be
    scram-sha-256: never trust, peer, ident, md5 or password. An include directive is refused
    too, since its rules cannot be checked here.
    """
    problems = []
    rules = 0
    for number, line in enumerate(text.splitlines(), start=1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        rules += 1
        kind = fields[0]
        if kind.startswith("include"):
            problems.append(f"line {number}: {kind} pulls in rules that are not checked")
            continue
        # local DATABASE USER METHOD; host* DATABASE USER ADDRESS [MASK] METHOD [OPTIONS].
        method_at = 3 if kind == "local" else 5 if len(fields) > 5 and "/" not in fields[3] else 4
        method = fields[method_at] if len(fields) > method_at else "?"
        if method != "scram-sha-256":
            problems.append(
                f"line {number}: {kind} rule authenticates by {method}, not scram-sha-256"
            )
    if rules == 0:
        problems.append("it has no rule")
    return problems


def connector(host: str, port: int, password: SecretStr, *, user: str = ADMIN_ROLE) -> Connect:
    """Return a Connect that opens autocommit connections as ``user`` to the server at ``port``."""

    def connect(dbname: str) -> psycopg.Connection[Any]:
        return psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=password.get_secret_value(),
            dbname=dbname,
            autocommit=True,
            connect_timeout=10,
        )

    return connect


def _wait_until_ready(connect: Connect, container: str) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while True:
        try:
            with connect("postgres"):
                return
        except psycopg.OperationalError:
            if pg_containers.state(container) != "running":
                raise BuildError(
                    f"{container} stopped while starting:\n{pg_containers.logs(container)}"
                ) from None
            if time.monotonic() > deadline:
                raise BuildError(f"{container} did not accept connections in time") from None
            time.sleep(0.5)
