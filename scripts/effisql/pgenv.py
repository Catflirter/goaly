"""Set up the Effi-SQL PostgreSQL environment (spec §4.1). Every command is idempotent.

    uv run python scripts/effisql/pgenv.py build
        Build the pinned dbgen, generate TPC-H SF3 and SF0.3 and load the golden data
        directory, unless the golden build is already complete.

The data lives under ~/data/effisql/ and the passwords in ~/.config/goaly/pg.env (mode 0600);
the build creates that file. Progress is logged to stderr as JSON lines; no output holds a
password.
"""

import argparse
import logging
import sys
from collections.abc import Sequence

import psycopg

from goaly.bench.effisql.pg_build import BuildError, build_golden
from goaly.bench.effisql.pg_containers import ContainerError
from goaly.bench.effisql.pg_settings import (
    PgEnvConfig,
    PgSettingsError,
    ensure_credentials,
    load_credentials,
)
from goaly.harness.core.logs import configure_logging

_log = logging.getLogger("pgenv")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up the Effi-SQL PostgreSQL environment.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="build the golden data directory")
    args = parser.parse_args(argv)

    configure_logging()
    config = PgEnvConfig()
    try:
        ensure_credentials()
        record = build_golden(config, load_credentials().admin_password)
        sys.stdout.write(f"golden data directory {config.golden_pgdata}, built {record.built_at}\n")
    except (BuildError, ContainerError, PgSettingsError, psycopg.Error) as error:
        _log.error("pgenv %s failed: %s", args.command, error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
