"""Set up and run the Effi-SQL PostgreSQL environment (spec §4.1, §6.2). Every command is
idempotent, and this process keeps to CPUs 0, 1, 12 and 13.

    uv run python scripts/effisql/pgenv.py setup
        build (if needed), then start (the one command `make final` needs).
    uv run python scripts/effisql/pgenv.py build
        Build the pinned dbgen, generate TPC-H SF3 and SF0.3, load the golden data directory and
        write the schema snapshots, unless the golden build is already complete.
    uv run python scripts/effisql/pgenv.py start | stop
        Start the 5 instances from the golden build, or shut them down cleanly. A restart
        reuses their data; start also reapplies the roles and privileges.
    uv run python scripts/effisql/pgenv.py status [--check]
        Show each instance's container, CPUs, socket directory and readiness, and fail if one
        is down, unreachable through its socket or pinned to the wrong CPUs. --check also
        compares every instance's row counts with the golden build's: it scans every table on
        every instance, so never run it during timed grading.
    uv run python scripts/effisql/pgenv.py snapshot [--port PORT]
        Rewrite the schema snapshots from a running server (default: solve-1).
    uv run python scripts/effisql/pgenv.py plans --tasks FILE [--port PORT] [--out PATH]
        Write the plan cache: our EXPLAIN (FORMAT JSON) plan, as the grader role on a solving
        instance, for each task in FILE that has no plan or one recorded on another scale.

The data lives under ~/data/effisql/ and the passwords in ~/.config/goaly/pg.env (mode 0600);
the build creates that file. Progress is logged to stderr as JSON lines; no output holds a
password.

Runbook notes:
- Every goaly process keeps to CPUs 0, 1, 12 and 13 by calling
  ``goaly.bench.effisql.pg_instances.confine_to_python_cpus()`` at startup (T06's CLI does);
  any other command runs under ``taskset -c 0,1,12,13``.
- Each instance's Unix socket is in ~/data/effisql/sockets/<instance>/, which start creates
  and bind-mounts into its container. The solve container, which may not use IP, mounts the
  solving instances' directories read-only and connects through them (V28).
- The instances' ports 55432-55436 lie in Linux's ephemeral range. Reserving them, with
  ``sysctl net.ipv4.ip_local_reserved_ports=55432-55439``, stops a client socket from briefly
  holding one while its server starts; without it, Docker's restart policy retries.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg

from goaly.bench.effisql.pg_build import (
    DATASETS,
    BuildError,
    build_golden,
    connector,
)
from goaly.bench.effisql.pg_containers import ContainerError
from goaly.bench.effisql.pg_instances import (
    INSTANCES,
    SOLVING,
    InstanceError,
    confine_to_python_cpus,
    count_problems,
    instance_status,
    require_golden_record,
    start_instances,
    stop_instances,
    table_counts,
)
from goaly.bench.effisql.pg_settings import (
    GRADER_ROLE,
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
_DBS = [dataset.db for dataset in DATASETS]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up the Effi-SQL PostgreSQL environment.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="build if needed, then start the instances")
    commands.add_parser("build", help="build the golden data directory and the snapshots")
    commands.add_parser("start", help="start the instances")
    commands.add_parser("stop", help="stop the instances cleanly")
    status = commands.add_parser("status", help="show the instances")
    status.add_argument(
        "--check",
        action="store_true",
        help="also compare row counts; it scans every table, so not during timed grading",
    )
    snapshot = commands.add_parser("snapshot", help="rewrite the schema snapshots")
    snapshot.add_argument("--port", type=int, default=SOLVING[0].port)
    plans = commands.add_parser("plans", help="write the plan cache for a task file")
    plans.add_argument("--tasks", type=Path, required=True, help="the JSONL task file")
    plans.add_argument("--port", type=int, default=SOLVING[0].port)
    plans.add_argument("--out", type=Path, help="default: ~/data/effisql/plans.json")
    args = parser.parse_args(argv)

    try:
        confine_to_python_cpus()
        configure_logging()
        config = PgEnvConfig()
        return _run(args, config)
    except (
        BuildError,
        ContainerError,
        InstanceError,
        PgSettingsError,
        psycopg.Error,
        OSError,
    ) as error:
        _log.error("pgenv %s failed: %s", args.command, error)
        return 1


def _run(args: argparse.Namespace, config: PgEnvConfig) -> int:
    if args.command in {"setup", "build"}:
        ensure_credentials(golden_pgdata=config.golden_pgdata)
        password = load_credentials().admin_password
        record = build_golden(
            config, password, after_load=lambda connect: write_snapshots(connect, config, _DBS)
        )
        _write(f"golden data directory {config.golden_pgdata}, built {record.built_at}")
        if args.command == "build":
            return 0
    if args.command in {"setup", "start"}:
        for name, action in start_instances(config, load_credentials()).items():
            _write(f"{name}: {action}")
        return _status(config, check=args.command == "setup")
    if args.command == "stop":
        results = stop_instances()
        for name, result in results.items():
            _write(f"{name}: {result}")
        return 0 if all(result in ("stopped", "not running") for result in results.values()) else 1
    if args.command == "status":
        return _status(config, check=args.check)
    if args.command == "snapshot":
        password = load_credentials().admin_password
        write_snapshots(connector(config.host, args.port, password), config, _DBS)
        _write(f"schema snapshots written under {config.data_root / 'snapshots'}")
        return 0
    if args.command == "plans":
        return _plans(config, args.tasks, args.port, args.out or config.plans_path)
    raise InstanceError(f"unknown command {args.command!r}")


def _status(config: PgEnvConfig, *, check: bool) -> int:
    healthy = True
    for status in instance_status(config):
        instance = status.instance
        ready = "accepting connections" if status.accepting else "NOT accepting connections"
        socket = "" if status.socket_accepting else " (NOT accepting connections there)"
        pinning = (
            "" if status.pinned else f" (WRONG: should be {','.join(map(str, instance.cpus))})"
        )
        healthy &= status.healthy
        _write(
            f"{instance.name:8} {instance.kind:8} port {instance.port}  cpus {status.cpuset}"
            f"{pinning}  {status.state or 'absent'}, {ready}  socket {status.socket_dir}{socket}"
        )
    if check:
        record = require_golden_record(config)
        password = load_credentials().admin_password
        counts = {i.name: table_counts(connector(config.host, i.port, password)) for i in INSTANCES}
        problems = count_problems(counts, record)
        for problem in problems:
            _write(f"PROBLEM: {problem}")
        _write("row counts: " + ("differ" if problems else "every instance matches the build"))
        healthy &= not problems
    return 0 if healthy else 1


def _plans(config: PgEnvConfig, tasks: Path, port: int, out: Path) -> int:
    requests = plan_requests(tasks, db=MAIN_DB)
    # The grader role: read-only by privilege and by default, like every planned query here.
    password = load_credentials().grader_password
    with connector(config.host, port, password, user=GRADER_ROLE)(MAIN_DB) as conn:
        plans, errors = explain_plans(conn, requests)
    by_reason = {
        reason: sum(1 for request in requests if request.reason == reason)
        for reason in ("no plan", "foreign scale")
    }
    if errors:
        # A partial cache would leave tasks on a plan from another scale: keep the old one.
        for request in requests:
            if request.instance_id in errors:
                _log.error(
                    "task %s (%s) has no plan: EXPLAIN failed: %s",
                    request.instance_id,
                    request.reason,
                    errors[request.instance_id],
                )
        sys.stderr.write(f"{len(errors)} of {len(requests)} EXPLAINs failed; {out} is unchanged\n")
        return 1
    write_json(out, dict(plans))
    _write(f"{len(plans)} plans written to {out}: {by_reason}")
    return 0


def _write(line: str) -> None:
    sys.stdout.write(line + "\n")


if __name__ == "__main__":
    sys.exit(main())
