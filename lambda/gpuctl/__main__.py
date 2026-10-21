"""gpuctl command line (design spec section 5.4): `PYTHONPATH=lambda python -m gpuctl <command>`.

Read-only commands: plan, status, rules, list-nodes, ip. None of them changes anything on Lambda.

Exit codes: 0 success; 1 an operation failed or the target was not found; 2 refused or a usage error
(no LAMBDA_API_KEY, invalid spec, over budget, over max_nodes); 3 drift found (Task 20).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TextIO

from gpuctl.lambda_api import FirewallRule, Instance, LambdaAPIError, LambdaClient
from gpuctl.peers import ruleset_name
from gpuctl.planner import (
    LIVE_STATUSES, PlanError, check_limits, make_plan, owned_instances, target_counts,
)
from gpuctl.spec import Spec, SpecError, load_spec
from gpuctl.status import capacity_rows, render_capacity

DEFAULT_SPEC = Path(__file__).resolve().parents[2] / "lambda" / "gpu-nodes.yaml"
API_KEY_ENV = "LAMBDA_API_KEY"
RULESET_PREFIX = ruleset_name("")  # "k3s-gpu-": the rulesets gpuctl owns, one per region

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


class _Refused(Exception):
    """Exit 2 with this message."""


class _Failed(Exception):
    """Exit 1 with this message."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse would sys.exit(2); main returns codes instead
        self.print_usage(sys.stderr)
        raise _Refused(f"{self.prog}: {message}")


def _clean(text: str) -> str:
    """Instance names, regions and rule descriptions come from the API: never let them drive the terminal."""
    return "".join(ch if ch.isprintable() else "?" for ch in text)


def _emit(out: TextIO, line: str = "") -> None:
    out.write(_clean(line) + "\n")


def _usd(value: float) -> str:
    return f"${value:.2f}/h"


def _set_counts(items: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        pool, sep, number = item.partition("=")
        if not sep or not pool or not (number.isascii() and number.isdigit()):
            raise _Refused(f"--set-count {item!r}: expected POOL=N with N an integer >= 0")
        if pool in counts:
            raise _Refused(f"--set-count: pool {pool!r} given more than once")
        counts[pool] = int(number)
    return counts


def _node_line(instance: Instance) -> str:
    return f"{instance.name} {instance.ip or '-'} {instance.status} {instance.instance_type} {instance.region}"


def _owned_sorted(instances: Iterable[Instance]) -> list[Instance]:
    return sorted(owned_instances(instances), key=lambda i: (i.name, i.id))


def _rule_line(rule: FirewallRule) -> str:
    if rule.port_range is None:
        ports = "-"
    else:
        lo, hi = rule.port_range
        ports = str(lo) if lo == hi else f"{lo}-{hi}"
    return f"  {rule.protocol:<5} {ports:<11} {rule.source_network:<18} {rule.description}".rstrip()


# --- commands -----------------------------------------------------------------------------------

def _plan(args: argparse.Namespace, spec: Spec, client, out: TextIO) -> int:
    set_counts = _set_counts(args.set_count or ())
    try:
        counts = target_counts(spec, set_counts=set_counts)
    except PlanError as e:
        raise _Refused(str(e)) from None
    instances = client.list_instances()
    types = client.instance_types()
    try:
        plan = make_plan(spec, instances, types, counts)
    except PlanError as e:
        raise _Refused(str(e)) from None

    duplicate_ids = {i.id for i in plan.duplicates}
    for name, pool in plan.create:
        _emit(out, f"create {name} (pool {pool})")
    for instance in plan.delete:
        why = "duplicate name" if instance.id in duplicate_ids else instance.status
        _emit(out, f"delete {instance.name} {instance.id} ({why})")
    for instance in plan.keep:
        _emit(out, f"keep {instance.name} {instance.ip or '-'} ({instance.status})")
    if not plan.create and not plan.delete:
        _emit(out, "No instance changes")
    nodes = len(plan.keep) + len(plan.create)
    _emit(out, f"GPU nodes after apply: {nodes} (max_nodes {spec.max_nodes})")
    _emit(out, f"Hourly cost: {_usd(plan.hourly_usd)} (budget {_usd(spec.budget_usd_per_hour)}; "
               f"new nodes priced at their pool's most expensive candidate type)")
    try:
        check_limits(plan, spec)
    except PlanError as e:  # BudgetExceeded included
        raise _Refused(f"plan refused: {e}") from None
    return EXIT_OK


def _status(args: argparse.Namespace, spec: Spec, client, out: TextIO) -> int:
    types = client.instance_types()
    instances = client.list_instances()
    _emit(out, "Capacity (in the order gpu-apply tries them):")
    for line in render_capacity(capacity_rows(spec, types)).splitlines():
        _emit(out, line)
    owned = _owned_sorted(instances)
    _emit(out)
    _emit(out, f"GPU instances (k3s-gpu-*): {len(owned)}")
    for instance in owned:
        _emit(out, _node_line(instance))
    owned_ids = {i.id for i in owned}
    others = sum(1 for i in instances if i.status in LIVE_STATUSES and i.id not in owned_ids)
    if others:
        _emit(out, f"Note: {others} other running instance(s) on this Lambda account are not managed by gpuctl")
    return EXIT_OK


def _rules(args: argparse.Namespace, client, out: TextIO) -> int:
    global_rules = client.get_global_rules()
    rulesets = sorted((r for r in client.list_rulesets() if r.name.startswith(RULESET_PREFIX)),
                      key=lambda r: (r.name, r.id))
    _emit(out, "global")
    for rule in global_rules:
        _emit(out, _rule_line(rule))
    if not global_rules:
        _emit(out, "  (no rules)")
    for ruleset in rulesets:
        _emit(out, f"ruleset {ruleset.name} {ruleset.id} {ruleset.region}")
        for rule in ruleset.rules:
            _emit(out, _rule_line(rule))
        if not ruleset.rules:
            _emit(out, "  (no rules)")
    return EXIT_OK


def _list_nodes(args: argparse.Namespace, client, out: TextIO) -> int:
    for instance in _owned_sorted(client.list_instances()):
        _emit(out, _node_line(instance))
    return EXIT_OK


def _ip(args: argparse.Namespace, client, out: TextIO) -> int:
    # Same choice as the planner: among live instances sharing a name, the smallest id is the node.
    matches = [i for i in _owned_sorted(client.list_instances()) if i.name == args.name]
    if not matches:
        raise _Failed(f"no running GPU node named {args.name!r}")
    instance = matches[0]
    if instance.ip is None:
        raise _Failed(f"{args.name} has no IP yet (status {instance.status})")
    _emit(out, instance.ip)
    return EXIT_OK


# --- entry point --------------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="gpuctl", description="Manage Lambda GPU nodes for the hybrid k3s cluster.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC,
                        help="GPU node spec (default: lambda/gpu-nodes.yaml in the repo)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    plan = sub.add_parser("plan", help="show what gpu-apply would create and delete, and the hourly cost")
    plan.add_argument("--set-count", action="append", metavar="POOL=N",
                      help="preview with this target count for POOL (repeatable; changes nothing)")
    sub.add_parser("status", help="GPU capacity per pool/type/region, and this cluster's GPU instances")
    sub.add_parser("rules", help="Lambda global firewall rules and the k3s-gpu-<region> rulesets")
    sub.add_parser("list-nodes", help="one line per GPU node: <name> <ip|-> <status> <type> <region>")
    ip = sub.add_parser("ip", help="print the public IP of GPU node NAME")
    ip.add_argument("name", metavar="NAME")
    return parser


def _load_spec(path: Path) -> Spec:
    try:
        return load_spec(path)
    except SpecError as e:
        raise _Refused(f"invalid GPU node spec: {e}") from None
    except OSError as e:
        raise _Refused(f"cannot read GPU node spec {path}: {e.strerror or e}") from None


def _client(env: Mapping[str, str], client):
    if not env.get(API_KEY_ENV, "").strip():
        raise _Refused(f"{API_KEY_ENV} is not set: put it in .env (chmod 600) and run via scripts/gpuctl")
    if client is not None:
        return client
    try:
        return LambdaClient(env[API_KEY_ENV])
    except ValueError as e:  # the client's messages never quote the key
        raise _Refused(f"{API_KEY_ENV}: {e}") from None


def main(argv: list[str] | None = None, *, env: Mapping[str, str] = os.environ, client=None, infra=None,
         out: TextIO | None = None) -> int:
    """Run one gpuctl command and return its exit code.

    `client` replaces the LambdaClient built from env[LAMBDA_API_KEY] and `infra` the Terraform/SSH adapters
    (Task 13; unused by the read-only commands). `out` defaults to the sys.stdout current at call time; errors
    go to sys.stderr.
    """
    out = sys.stdout if out is None else out
    try:
        try:
            args = _parser().parse_args(argv)
        except SystemExit as e:  # --help
            return e.code if isinstance(e.code, int) else EXIT_OK
        if args.command in ("plan", "status"):
            spec = _load_spec(args.spec)
            api = _client(env, client)
            return (_plan if args.command == "plan" else _status)(args, spec, api, out)
        handler = {"rules": _rules, "list-nodes": _list_nodes, "ip": _ip}[args.command]
        return handler(args, _client(env, client), out)
    except _Refused as e:
        print(_clean(f"gpuctl: {e}"), file=sys.stderr)
        return EXIT_REFUSED
    except _Failed as e:
        print(_clean(f"gpuctl: {e}"), file=sys.stderr)
        return EXIT_FAILED
    except LambdaAPIError as e:
        print(_clean(f"gpuctl: Lambda API error: {e}"), file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
