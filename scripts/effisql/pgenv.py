"""Set up the Effi-SQL PostgreSQL environment (spec §4.1). Every command is idempotent.

    uv run python scripts/effisql/pgenv.py build
        Build the pinned dbgen, generate TPC-H SF3 and SF0.3, load the golden data directory and
        write the schema snapshots, unless the golden build is already complete.
    uv run python scripts/effisql/pgenv.py snapshot --port PORT
        Rewrite the schema snapshots from a running server.
    uv run python scripts/effisql/pgenv.py plans --tasks FILE --port PORT [--out PATH]
        Write the plan cache: our EXPLAIN (FORMAT JSON) plan for each task in FILE that has no
        plan, or a plan recorded on another TPC-H scale.

The data lives under ~/data/effisql/ and the passwords in ~/.config/goaly/pg.env (mode 0600);
the build creates that file. Progress is logged to stderr as JSON lines; no output holds a
password.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg

from goaly.bench.effisql.pg_build import DATASETS, BuildError, build_golden, connector
from goaly.bench.effisql.pg_containers import ContainerError
from goaly.bench.effisql.pg_settings import (
    MAIN_DB,
    PgEnvConfig,
    PgSettingsError,
    ensure_credentials,
    load_credentials,
)
from goaly.bench.effisql.pg_snapshot import (
    explain_plans,
    plan_requests,
    write_json,
    write_snapshots,
)
from goaly.harness.core.logs import configure_logging

_log = logging.getLogger("pgenv")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up the Effi-SQL PostgreSQL environment.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="build the golden data directory and the snapshots")
    snapshot = commands.add_parser("snapshot", help="rewrite the schema snapshots")
    snapshot.add_argument("--port", type=int, required=True, help="a running server's port")
    plans = commands.add_parser("plans", help="write the plan cache for a task file")
    plans.add_argument("--tasks", type=Path, required=True, help="the JSONL task file")
    plans.add_argument("--port", type=int, required=True, help="a running server's port")
    plans.add_argument("--out", type=Path, help="default: ~/data/effisql/plans.json")
    args = parser.parse_args(argv)

    try:
        configure_logging()
        config = PgEnvConfig()
        if args.command == "build":
            ensure_credentials(golden_pgdata=config.golden_pgdata)
            password = load_credentials().admin_password
            dbs = [dataset.db for dataset in DATASETS]
            record = build_golden(
                config, password, after_load=lambda connect: write_snapshots(connect, config, dbs)
            )
            sys.stdout.write(
                f"golden data directory {config.golden_pgdata}, built {record.built_at}\n"
            )
        elif args.command == "snapshot":
            password = load_credentials().admin_password
            dbs = [dataset.db for dataset in DATASETS]
            write_snapshots(connector(config.host, args.port, password), config, dbs)
            sys.stdout.write(f"schema snapshots written under {config.data_root / 'snapshots'}\n")
        else:
            return _plans(config, args.tasks, args.port, args.out or config.plans_path)
    except (BuildError, ContainerError, PgSettingsError, psycopg.Error) as error:
        _log.error("pgenv %s failed: %s", args.command, error)
        return 1
    return 0


def _plans(config: PgEnvConfig, tasks: Path, port: int, out: Path) -> int:
    requests = plan_requests(tasks, db=MAIN_DB)
    password = load_credentials().admin_password
    with connector(config.host, port, password)(MAIN_DB) as conn:
        plans, errors = explain_plans(conn, requests)
    write_json(out, dict(plans))
    by_reason = {
        reason: sum(1 for request in requests if request.reason == reason)
        for reason in ("no plan", "foreign scale")
    }
    sys.stdout.write(f"{len(plans)} plans written to {out}: {by_reason}\n")
    for instance_id, message in sorted(errors.items()):
        sys.stderr.write(f"task {instance_id}: EXPLAIN failed: {message}\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
