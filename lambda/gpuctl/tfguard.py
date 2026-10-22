"""Terraform plan guardrail for unattended worker scaling (stretch S3; design spec sections 3.1, 10.2, 10.5).

`scripts/cpu-pool.sh scale|remove ... --guarded` (the stretch autoscaler's only way to change the CPU pool) pipes
`terraform show -json <saved workers plan>` into `gpuctl tfguard --add NAME... --remove NAME...` and applies that
saved plan only when this module finds nothing wrong with it.

A plan passes when every resource change is one of:
  - actions ["no-op"] or ["read"] on any address (ignored: the shared rules cp_to_workers and workers_iap_ssh, the
    base stack's remote state);
  - ["create"] of one of the four per-worker resources of an --add name N, in the root module:
    google_compute_instance.worker["N"], google_compute_address.worker["N"],
    google_compute_firewall.worker_to_cp["N"], google_compute_firewall.worker_mesh["N"];
  - ["delete"] of one of those four resources of a --remove name.
and the plan holds exactly 4 x len(add) creates and 4 x len(remove) deletes. Anything else is a violation: a replace
(["delete", "create"] or ["create", "delete"]), an update, any other action, a change to any other address (k3s-cp,
another worker, a shared rule), the same address twice, and a plan that is not the JSON `terraform show -json`
writes (fail closed: a guard that cannot read the plan never lets it through). A document without resource_changes is
a plan with no changes only when it carries a plan's landmarks (a string format_version and planned_values, which
terraform always writes for a plan); `{}` or a state document (`terraform show -json` without a plan file) is not.

Names are k3s-cpu-<n> (n >= 1, no leading zero); k3s-cpu-1 is never in remove (spec section 10.2) and a name is never
in both sets. check_workers_plan raises ValueError for a bad name; the CLI reports it as a violation (exit 1), so
`gpuctl tfguard` keeps the brief's two exit codes (argparse usage errors aside, exit 2 as for every subcommand).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

WORKER_NAME_RE = re.compile(r"k3s-cpu-[1-9][0-9]*")
PROTECTED = "k3s-cpu-1"
# The workers stack's for_each resources (infra/workers/main.tf, firewall.tf), keyed by worker name.
WORKER_RESOURCES = (
    "google_compute_instance.worker",
    "google_compute_address.worker",
    "google_compute_firewall.worker_to_cp",
    "google_compute_firewall.worker_mesh",
)
_ADDRESS_RE = re.compile(r'(?P<resource>[a-z0-9_]+\.[a-z0-9_]+)\["(?P<name>[^"\\]*)"\]')
IGNORED = (["no-op"], ["read"])


def worker_addresses(name: str) -> list[str]:
    """The four per-worker resource addresses of NAME."""
    return [f'{resource}["{name}"]' for resource in WORKER_RESOURCES]


def check_names(add: Iterable[str], remove: Iterable[str]) -> None:
    """Raise ValueError unless every name is k3s-cpu-<n>, k3s-cpu-1 is not removed, and no name is in both sets."""
    add, remove = set(add), set(remove)
    for name in sorted(add | remove):
        if not isinstance(name, str) or not WORKER_NAME_RE.fullmatch(name):
            raise ValueError(f"not a CPU worker name (k3s-cpu-<n>): {name!r}")
    if PROTECTED in remove:
        raise ValueError(f"{PROTECTED} is never removed (spec section 10.2)")
    both = sorted(add & remove)
    if both:
        raise ValueError(f"in both --add and --remove: {', '.join(both)}")


def _worker_key(address: str) -> tuple[str, str] | None:
    """(resource, name) for a root-module per-worker resource address, else None."""
    match = _ADDRESS_RE.fullmatch(address)
    if match is None or match["resource"] not in WORKER_RESOURCES:
        return None
    return match["resource"], match["name"]


def check_workers_plan(plan: object, *, add: set[str], remove: set[str]) -> list[str]:
    """The violations in PLAN (parsed `terraform show -json` of a saved plan); an empty list means it may be applied.

    Raises ValueError for bad names (see check_names); a malformed PLAN is a violation, never an exception.
    """
    check_names(add, remove)
    if not isinstance(plan, dict):
        return [f"not a Terraform plan (JSON {type(plan).__name__}, expected an object)"]
    if "resource_changes" in plan:
        changes = plan["resource_changes"]
    elif isinstance(plan.get("format_version"), str) and "planned_values" in plan:
        changes = []  # terraform show -json PLAN leaves resource_changes out when nothing changes
    else:  # {} or the state document `terraform show -json` prints without a plan: never "no changes"
        return ["not a Terraform plan: no resource_changes, and no format_version + planned_values"]
    if not isinstance(changes, list):
        return ["not a Terraform plan: resource_changes is not a list"]

    violations: list[str] = []
    seen: set[str] = set()
    creates = deletes = 0
    for i, change in enumerate(changes):
        address = change.get("address") if isinstance(change, dict) else None
        actions = change.get("change", {}).get("actions") if isinstance(change, dict) and isinstance(
            change.get("change"), dict) else None
        if not isinstance(address, str) or not address:
            violations.append(f"resource change #{i} has no address: cannot tell what it changes")
            continue
        if not isinstance(actions, list) or not actions or not all(isinstance(a, str) for a in actions):
            violations.append(f"{address}: unreadable actions {actions!r}")
            continue
        if address in seen:
            violations.append(f"{address}: listed more than once in the plan")
            continue
        seen.add(address)
        if actions in IGNORED:
            continue
        if actions == ["create"]:
            creates += 1
        elif actions == ["delete"]:
            deletes += 1
        key = _worker_key(address)
        name = key[1] if key else None
        if name in add and actions == ["create"]:
            continue
        if name in remove and actions == ["delete"]:
            continue
        violations.append(f"{address}: {'+'.join(actions)} is not allowed ({_expected(name, add, remove)})")

    for name in sorted(add):
        violations += [f"{a}: expected create, not in the plan" for a in worker_addresses(name) if a not in seen]
    for name in sorted(remove):
        violations += [f"{a}: expected delete, not in the plan" for a in worker_addresses(name) if a not in seen]
    if creates != 4 * len(add):
        violations.append(f"the plan has {creates} create(s), expected {4 * len(add)} (4 per added worker)")
    if deletes != 4 * len(remove):
        violations.append(f"the plan has {deletes} delete(s), expected {4 * len(remove)} (4 per removed worker)")
    return violations


def _expected(name: str | None, add: set[str], remove: set[str]) -> str:
    if name in add:
        return f"{name} is being added: only create"
    if name in remove:
        return f"{name} is being removed: only delete"
    return "only the per-worker resources of --add (create) and --remove (delete) may change"
