"""The golden build end to end, at TPC-H SF0.01: Docker, the pinned image and dbgen.

Run with ``pytest --integration``. It reuses the pinned dbgen checkout under ~/data/effisql when
there is one (else it clones it), and builds into a temporary data root with its own container
name and port, so it never touches the real environment.
"""

from pathlib import Path

import psycopg
import pytest
from pydantic import SecretStr

from goaly.bench.effisql import pg_containers
from goaly.bench.effisql.pg_build import (
    TPCH_TABLES,
    Connect,
    Dataset,
    TableFile,
    build_golden,
    read_golden_record,
)
from goaly.bench.effisql.pg_settings import PgEnvConfig

pytestmark = pytest.mark.integration

SF001 = Dataset(
    "tpch",
    "0.01",
    {
        name: TableFile(sha256=sha, rows=rows)
        for name, (sha, rows) in {
            "region": ("5a7c2fe9718db00ff5e5bc82a9ebfa8abc492cc75260d3c0ffb411974f235ab0", 5),
            "nation": ("7d47bc9397da331054fa92b8fb92e4c074004bad72dcbb893012093218dccf6c", 25),
            "supplier": ("b199bef3350840676cfe4be096091851bc47b6a57cd8a71b3f559d42b7d9dacd", 100),
            "customer": ("22156f2770387f5adadbc72774f2282d353aedd5092bd8fc54871b3bf5e74cba", 1500),
            "part": ("604232ee10e95dca875c196f759c07babf34293dfac8c99b9f1f0c2708f0561b", 2000),
            "partsupp": ("906f58419af6ad5d62489a7e2105257654bb6d8458a7fd8fe19b76655aac50c7", 8000),
            "orders": ("a444603dfba6c47e902e24b517608a5eb3b117127e99dff16a40f4eaa47b812c", 15000),
            "lineitem": ("517b566190fbeadc638602554d109a463631e19788936ccb97196ebd407b51f1", 60175),
        }.items()
    },
)
CONTAINER = "goaly-effisql-itest-build"
PORT = 55449


def test_the_golden_build_loads_verifies_and_describes_the_database(tmp_path: Path) -> None:
    real_dbgen = PgEnvConfig().dbgen_dir
    if real_dbgen.is_dir():
        (tmp_path / "dbgen").symlink_to(real_dbgen)
    config = PgEnvConfig(
        data_root=tmp_path, build_port=PORT, build_container=CONTAINER, build_cpuset=None
    )
    password = SecretStr("integration-only")
    counts: dict[str, int] = {}
    keys: list[tuple[str, str]] = []

    def after_load(connect: Connect) -> None:
        with connect("tpch") as conn:
            # The image's postgresql.conf says '*'; the command line keeps it to loopback.
            assert conn.execute("SHOW listen_addresses").fetchone() == ("127.0.0.1",)
            for table in TPCH_TABLES:
                row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
                counts[table] = row[0] if row else -1
            keys.extend(
                conn.execute(
                    "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
                    "WHERE conrelid = 'lineitem'::regclass ORDER BY conname"
                ).fetchall()
            )
        with pytest.raises(psycopg.OperationalError, match="password authentication failed"):
            psycopg.connect(
                host="127.0.0.1",
                port=PORT,
                user="postgres",
                password="wrong",  # noqa: S106  # A deliberately wrong test value.
                dbname="tpch",
            )

    record = build_golden(config, password, datasets=[SF001], after_load=after_load)

    assert read_golden_record(config.golden_record) == record
    assert record.server_version is not None
    assert record.server_version.startswith("16.")
    assert (record.encoding, record.collation) == ("UTF8", "en_US.utf8")  # The official locale.
    assert record.databases["tpch"].tables == dict(SF001.files)
    assert (config.golden_pgdata / "PG_VERSION").read_text().strip() == "16"
    assert pg_containers.state(CONTAINER) is None

    assert counts == {name: entry.rows for name, entry in SF001.files.items()}
    assert keys == [
        ("lineitem_l_orderkey_fkey", "FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey)"),
        (
            "lineitem_l_partkey_l_suppkey_fkey",
            "FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey)",
        ),
        ("lineitem_pkey", "PRIMARY KEY (l_orderkey, l_linenumber)"),
    ]

    # Done is done: a second call returns the record and starts nothing.
    assert build_golden(config, SecretStr("other"), datasets=[SF001]) == record
