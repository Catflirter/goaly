"""gpuctl command line (design spec section 5.4): `PYTHONPATH=lambda python -m gpuctl <command>`.

Read-only commands: plan, status, rules, list-nodes, ip. None of them changes anything on Lambda.
sync-peers rewrites the peer allow-lists (k3s-cp's k3s_peers set, Lambda's global rules and k3s-gpu-<region> rulesets).
apply creates and deletes GPU nodes to match the spec; down deletes all of them.
tfguard checks a saved workers-stack plan (`terraform show -json` on stdin) for cpu-pool.sh --guarded; it needs no
Lambda API key and exits 1 on any violation.

Exit codes: 0 success; 1 an operation failed or the target was not found; 2 refused or a usage error
(no LAMBDA_API_KEY, invalid spec or override file, over budget, over max_nodes, not confirmed); 3 status found drift.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TextIO

from gpuctl.infra import CommandError, Infra, SubprocessShell
from gpuctl.lambda_api import FirewallRule, Instance, LambdaAPIError, LambdaClient
from gpuctl.peers import ruleset_name
from gpuctl.planner import (
    LIVE_STATUSES, PlanError, check_limits, keeper_order, make_plan, owned_instances, plan_lines, target_counts,
)
from gpuctl.reconcile import Reconciler, ReconcileError, Settings, load_overrides
from gpuctl.spec import Spec, SpecError, load_spec
from gpuctl.status import DRIFT_SOURCES, capacity_rows, is_problem, render_capacity
from gpuctl.tfguard import check_names, check_workers_plan

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "lambda" / "gpu-nodes.yaml"
OVERRIDES_PATH = REPO_ROOT / "lambda" / ".autoscale-state.json"
API_KEY_ENV = "LAMBDA_API_KEY"
SSH_KEY_ENV = "LAMBDA_SSH_KEY_NAME"
RULESET_PREFIX = ruleset_name("")  # "k3s-gpu-": the rulesets gpuctl owns, one per region

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_DRIFT = 0, 1, 2, 3


class _Refused(Exception):
    """Exit 2 with this message."""


class _Failed(Exception):
    """Exit 1 with this message."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse would sys.exit(2); main returns codes instead
        self.print_usage(sys.stderr)
        # main() adds "gpuctl: "; a subcommand parser's prog is "gpuctl <command>", so keep only the command
        command = self.prog.removeprefix("gpuctl").strip()
        raise _Refused(f"{command}: {message}" if command else message)


def _clean(text: str) -> str:
    """Instance names, regions and rule descriptions come from the API: never let them drive the terminal."""
    return "".join(ch if ch.isprintable() else "?" for ch in text)


def _emit(out: TextIO, line: str = "") -> None:
    out.write(_clean(line) + "\n")


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


def _checked_ip(instance: Instance) -> str | None:
    """The instance's IP, refused unless it is a plain dotted-quad IPv4 address.

    `ip` and `list-nodes` output feeds scripts (ssh targets, Terraform gpu_ips): an API-supplied value such as
    "1.2.3.4 -oProxyCommand=..." must never reach them.
    """
    if instance.ip is None:
        return None
    try:
        ipaddress.IPv4Address(instance.ip)
    except ValueError:
        raise _Failed(f"{instance.name} has an invalid IP from the API: {instance.ip!r}") from None
    return instance.ip


def _node_line(instance: Instance) -> str:
    ip = _checked_ip(instance) or "-"
    return f"{instance.name} {ip} {instance.status} {instance.instance_type} {instance.region}"


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
        # Ruling R20: the same override file apply reads, so the preview is what apply would do
        counts = target_counts(spec, overrides=load_overrides(OVERRIDES_PATH), set_counts=set_counts)
    except PlanError as e:  # OverridesError included
        raise _Refused(str(e)) from None
    instances = client.list_instances()
    types = client.instance_types()
    try:
        plan = make_plan(spec, instances, types, counts)
    except PlanError as e:
        raise _Refused(str(e)) from None

    for instance in plan.keep:
        _checked_ip(instance)
    for line in plan_lines(plan, spec):
        _emit(out, line)
    try:
        check_limits(plan, spec)
    except PlanError as e:  # BudgetExceeded included
        raise _Refused(f"plan refused: {e}") from None
    return EXIT_OK


def _status(args: argparse.Namespace, spec: Spec, client, infra, env: Mapping[str, str], out: TextIO) -> int:
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

    lines = _reconciler(args, client, infra, env, out).drift(spec, instances)
    _emit(out)
    _emit(out, "Drift check (Lambda instances, cluster nodes, gpu-access state, k3s-cp k3s_peers):")
    for line in lines:
        _emit(out, f"  {line}")
    problems = [line for line in lines if is_problem(line)]
    if not problems:
        skipped = len(lines)
        # A skipped source (the gpu-access state, say) is not an all-clear: never print a bare "No drift" then.
        _emit(out, f"  No drift in the {len(DRIFT_SOURCES) - skipped} sources read; {skipped} source(s) skipped "
                   "(see above)" if skipped else "  No drift")
        return EXIT_OK
    _emit(out, f"  {len(problems)} problem(s); make gpu-apply reconciles the firewall and the instance counts")
    return EXIT_DRIFT


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
    # Same choice as the planner (Ruling R22): among live instances sharing a name, the best status, then the
    # smallest id, is the node.
    matches = [i for i in owned_instances(client.list_instances()) if i.name == args.name]
    if not matches:
        raise _Failed(f"no running GPU node named {args.name!r}")
    instance = min(matches, key=keeper_order)
    ip = _checked_ip(instance)
    if ip is None:
        raise _Failed(f"{args.name} has no IP yet (status {instance.status})")
    _emit(out, ip)
    return EXIT_OK


def _confirm(out: TextIO) -> Callable[[str], bool]:
    """Ask on `out` and read one line from stdin: only y/yes confirms; end of input declines."""
    def confirm(prompt: str) -> bool:
        out.write(prompt)
        out.flush()
        return sys.stdin.readline().strip().lower() in ("y", "yes")
    return confirm


def _reconciler(args: argparse.Namespace, client, infra, env: Mapping[str, str], out: TextIO) -> Reconciler:
    settings = Settings(spec_path=args.spec, overrides_path=OVERRIDES_PATH,
                        ssh_key_name=env.get(SSH_KEY_ENV, "").strip())
    if infra is None:
        infra = Infra(SubprocessShell(), REPO_ROOT, env)
    return Reconciler(client, infra, settings, out=out, confirm=_confirm(out))


def _sync_peers(args: argparse.Namespace, spec: Spec, client, infra, env: Mapping[str, str], out: TextIO) -> int:
    try:
        _reconciler(args, client, infra, env, out).sync_peers(spec)
    except (ReconcileError, CommandError, ValueError) as e:  # ValueError: an invalid IP from an output or OPERATOR_IP
        raise _Failed(f"sync-peers: {e}") from None
    return EXIT_OK


def _apply(args: argparse.Namespace, spec: Spec, client, infra, env: Mapping[str, str], out: TextIO) -> int:
    """apply and down: the Reconciler's own exit code, or 1 when a delete, read or firewall step raises."""
    reconciler = _reconciler(args, client, infra, env, out)
    try:
        if args.command == "down":
            return reconciler.down(spec, auto_approve=args.auto_approve)
        return reconciler.apply(spec, auto_approve=args.auto_approve, keep_on_failure=args.keep_on_failure)
    except (ReconcileError, CommandError, ValueError, OSError) as e:
        raise _Failed(f"{args.command}: {e}") from None


def _tfguard(args: argparse.Namespace, out: TextIO) -> int:
    """Stretch S3: the saved workers plan on stdin may only create --add and delete --remove workers (gpuctl.tfguard)."""
    add, remove = set(args.add), set(args.remove)
    try:
        check_names(add, remove)
    except ValueError as e:
        raise _Refused(f"tfguard: {e}") from None
    try:
        plan = json.loads(sys.stdin.read())
    except (ValueError, UnicodeDecodeError) as e:  # fail closed: a plan the guard cannot read is never applied
        violations = [f"stdin is not the JSON of terraform show -json: {e}"]
    else:
        violations = check_workers_plan(plan, add=add, remove=remove)
    for line in violations:
        _emit(out, f"violation: {line}")
    if violations:
        raise _Failed(f"tfguard: {len(violations)} violation(s): the plan must not be applied")
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
    apply = sub.add_parser("apply", help="create and delete GPU nodes to match the spec (asks first)")
    apply.add_argument("--auto-approve", action="store_true", help="do not ask for confirmation")
    apply.add_argument("--keep-on-failure", action="store_true",
                       help="keep new nodes that failed to come up (they keep billing) instead of terminating them")
    down = sub.add_parser("down", help="remove the autoscaler overrides and delete every GPU node")
    down.add_argument("--auto-approve", action="store_true", help="do not ask for confirmation")
    sub.add_parser("status", help="GPU capacity, this cluster's GPU instances and a drift check (exit 3 on drift)")
    sub.add_parser("rules", help="Lambda global firewall rules and the k3s-gpu-<region> rulesets")
    sub.add_parser("list-nodes", help="one line per GPU node: <name> <ip|-> <status> <type> <region>")
    ip = sub.add_parser("ip", help="print the public IP of GPU node NAME")
    ip.add_argument("name", metavar="NAME")
    sub.add_parser("sync-peers", help="rewrite the peer allow-lists: cp k3s_peers set, Lambda global rules and rulesets")
    tfguard = sub.add_parser("tfguard", help="check a workers plan (terraform show -json on stdin): exit 1 on a violation")
    tfguard.add_argument("--add", nargs="*", action="extend", default=[], metavar="NAME",
                         help="CPU workers the plan may create (k3s-cpu-<n>)")
    tfguard.add_argument("--remove", nargs="*", action="extend", default=[], metavar="NAME",
                         help="CPU workers the plan may delete (never k3s-cpu-1)")
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
    (gpuctl.infra.Infra over `env`; used by status, sync-peers, apply and down). `out` defaults to the sys.stdout
    current at call time; errors go to sys.stderr.
    """
    out = sys.stdout if out is None else out
    try:
        try:
            args = _parser().parse_args(argv)
        except SystemExit as e:  # --help
            return e.code if isinstance(e.code, int) else EXIT_OK
        if args.command == "tfguard":  # no spec, no Lambda client: it only reads stdin
            return _tfguard(args, out)
        if args.command == "plan":
            return _plan(args, _load_spec(args.spec), _client(env, client), out)
        if args.command in ("status", "sync-peers", "apply", "down"):
            spec = _load_spec(args.spec)
            if args.command == "apply" and not env.get(SSH_KEY_ENV, "").strip():
                raise _Refused(f"{SSH_KEY_ENV} is not set: put the name of your Lambda SSH key in .env")
            handler = {"status": _status, "sync-peers": _sync_peers, "apply": _apply, "down": _apply}[args.command]
            return handler(args, spec, _client(env, client), infra, env, out)
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
