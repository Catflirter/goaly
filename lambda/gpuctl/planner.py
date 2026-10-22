"""Declarative planner: compare the GPU node spec with what runs on Lambda (design spec section 5.4).

Pure functions over Spec, Instance and InstanceType; no API calls. Only instances named
`k3s-gpu-...` in a live status are ours; everything else is never touched.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from gpuctl.lambda_api import Instance, InstanceType
from gpuctl.spec import POOL_NAME_RE, Pool, Spec

PREFIX = "k3s-gpu-"
# booting (ip still null) and unhealthy instances exist and bill; terminating/terminated do not count.
LIVE_STATUSES = frozenset({"booting", "active", "unhealthy"})
# Ruling R22: among live instances sharing a name, the best status is the node (the active copy is the one that
# joined the cluster), then the smallest id.
_STATUS_RANK = {"active": 0, "booting": 1, "unhealthy": 2}

# Canonical names only (index >= 1, no leading zeros), so a name and its (pool, index) map one to one.
_NODE_NAME_RE = re.compile(rf"{re.escape(PREFIX)}({POOL_NAME_RE.pattern})-([1-9][0-9]*)")
_TYPE_PREFIX_RE = re.compile(r"gpu_[0-9]+x_")
# make_plan refuses to materialise more nodes than this (or max_nodes, if larger). check_limits rejects anything
# above max_nodes anyway; the ceiling only stops a --set-count typo (a6000=1000000000) from allocating one create
# entry per requested node before that check can run.
PLAN_NODE_CEILING = 1000


class PlanError(Exception):
    """The plan cannot be made or violates a limit."""


class BudgetExceeded(PlanError):
    """The plan's hourly cost is above budget_usd_per_hour."""


@dataclass(frozen=True)
class Plan:
    """What to change on Lambda.

    `duplicates` is the subset of `delete` that shares its name with another owned instance (the one keeper_order()
    puts first, which is in `keep` or earlier in `delete`). The name is also the K8s node name and the host-key
    entry, so an executor must terminate a duplicate by id only: never drain/delete the node or forget the host
    key for it. That is done once, for the name's own instance, if and only if that instance is deleted.
    """

    create: tuple[tuple[str, str], ...]  # (node name, pool)
    delete: tuple[Instance, ...]
    keep: tuple[Instance, ...]
    hourly_usd: float
    duplicates: tuple[Instance, ...] = ()


def node_name(pool: str, index: int) -> str:
    name = f"{PREFIX}{pool}-{index}"
    if parse_node_name(name) != (pool, index):
        raise ValueError(f"no canonical node name for pool {pool!r}, index {index!r}")
    return name


def parse_node_name(name: str) -> tuple[str, int] | None:
    match = _NODE_NAME_RE.fullmatch(name)
    return (match.group(1), int(match.group(2))) if match else None


def owned_instances(instances: Iterable[Instance]) -> list[Instance]:
    return [i for i in instances if i.name.startswith(PREFIX) and i.status in LIVE_STATUSES]


def keeper_order(instance: Instance) -> tuple[int, str]:
    """Sort key among owned instances sharing a name: the first one is the node, the rest are duplicates (R22)."""
    return _STATUS_RANK.get(instance.status, len(_STATUS_RANK)), instance.id


def _count(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlanError(f"{where}: count must be an integer >= 0, got {value!r}")
    return value


def target_counts(spec: Spec, overrides: Mapping[str, int] | None = None,
                  set_counts: Mapping[str, int] | None = None) -> dict[str, int]:
    """Target node count per pool: set_counts (unclamped) > overrides (autoscale pools, clamped) > count."""
    overrides = overrides or {}
    set_counts = set_counts or {}
    unknown = sorted(set(set_counts) - set(spec.pools))
    if unknown:
        raise PlanError(f"unknown pool(s) {', '.join(map(repr, unknown))}; known: {sorted(spec.pools)}")
    counts = {}
    for name, pool in spec.pools.items():
        if name in set_counts:
            counts[name] = _count(set_counts[name], f"pools.{name}")
        elif name in overrides and pool.autoscale is not None:
            wanted = _count(overrides[name], f"override for pools.{name}")
            counts[name] = min(max(wanted, pool.autoscale.min), pool.autoscale.max)
        else:
            counts[name] = pool.count
    return counts


def choose_placement(pool: Pool, types: Mapping[str, InstanceType],
                     exclude: frozenset[tuple[str, str]] = frozenset()) -> tuple[str, str] | None:
    """First (instance type, region) with capacity, trying every region of a type before the next type."""
    for type_name in pool.candidate_types():
        instance_type = types.get(type_name)
        if instance_type is None:
            continue
        for region in pool.regions:
            if region in instance_type.regions_with_capacity and (type_name, region) not in exclude:
                return type_name, region
    return None


def gpu_type_label(instance_type: str) -> str:
    """gpu_1x_a6000 -> a6000; gpu_8x_a100_80gb_sxm4 -> a100_80gb_sxm4. Only a leading gpu_<n>x_ is stripped."""
    match = _TYPE_PREFIX_RE.match(instance_type)
    return instance_type[match.end():] if match else instance_type


def _worst_case_cents(pool: Pool, types: Mapping[str, InstanceType]) -> int:
    prices = [types[t].price_cents_per_hour for t in pool.candidate_types() if t in types]
    if not prices:
        raise PlanError(f"pools.{pool.name}: no price for any candidate type {list(pool.candidate_types())}")
    return max(prices)


def make_plan(spec: Spec, instances: Iterable[Instance], types: Mapping[str, InstanceType],
              counts: Mapping[str, int]) -> Plan:
    """Diff the spec at `counts` (one entry per spec pool) against the owned live instances."""
    if set(counts) != set(spec.pools):
        raise PlanError(f"counts must name exactly the spec's pools {sorted(spec.pools)}, got {sorted(counts)}")
    counts = {name: _count(counts[name], f"pools.{name}") for name in spec.pools}
    total = sum(counts.values())
    if total > max(spec.max_nodes, PLAN_NODE_CEILING):  # keep + create == total, so fail before building it
        raise PlanError(f"max_nodes: counts ask for {total} GPU nodes, limit is {spec.max_nodes}")

    # One instance per name: the best status, then the smallest id, wins (Ruling R22); the rest are deleted.
    by_name: dict[str, Instance] = {}
    duplicates: list[Instance] = []
    for instance in sorted(owned_instances(instances), key=lambda i: (i.name, keeper_order(i))):
        if instance.name in by_name:
            duplicates.append(instance)
        else:
            by_name[instance.name] = instance

    in_pool: dict[str, dict[int, Instance]] = {name: {} for name in spec.pools}
    # Ours but not wanted: its pool is gone from the spec, or the name is not a node name at all.
    unwanted: list[tuple[tuple[str, int], Instance]] = []
    for name, instance in by_name.items():
        parsed = parse_node_name(name)
        if parsed is not None and parsed[0] in in_pool:
            in_pool[parsed[0]][parsed[1]] = instance
        else:
            unwanted.append((parsed or (name, 0), instance))

    create: list[tuple[str, str]] = []
    delete: list[Instance] = []
    keep: list[Instance] = []
    cents = 0
    for pool_name, pool in spec.pools.items():
        existing = in_pool[pool_name]
        indices = sorted(existing)
        target = counts[pool_name]
        keep.extend(existing[i] for i in indices[:target])
        delete.extend(existing[i] for i in reversed(indices[target:]))  # highest index first
        new: list[int] = []  # lowest free indices, so names stay 1..target
        index = 1
        while len(indices) + len(new) < target:
            if index not in existing:
                new.append(index)
            index += 1
        create.extend((node_name(pool_name, i), pool_name) for i in new)
        if new:
            cents += _worst_case_cents(pool, types) * len(new)
    unwanted.sort(key=lambda u: (u[0][0], -u[0][1]))  # by pool, highest index first
    delete.extend(instance for _, instance in unwanted)
    delete.extend(duplicates)

    for instance in keep:
        instance_type = types.get(instance.instance_type)
        if instance_type is None:
            raise PlanError(f"{instance.name}: no price for instance type {instance.instance_type!r}")
        cents += instance_type.price_cents_per_hour

    return Plan(create=tuple(create), delete=tuple(delete), keep=tuple(keep), hourly_usd=cents / 100,
                duplicates=tuple(duplicates))


def plan_lines(plan: Plan, spec: Spec) -> list[str]:
    """How plan and apply show a plan: one line per create, delete and keep, then the node count and cost."""
    duplicate_ids = {i.id for i in plan.duplicates}
    lines = [f"create {name} (pool {pool})" for name, pool in plan.create]
    for instance in plan.delete:
        why = "duplicate name" if instance.id in duplicate_ids else instance.status
        lines.append(f"delete {instance.name} {instance.id} ({why})")
    lines += [f"keep {i.name} {i.ip or '-'} ({i.status})" for i in plan.keep]
    if not plan.create and not plan.delete:
        lines.append("No instance changes")
    lines.append(f"GPU nodes after apply: {len(plan.keep) + len(plan.create)} (max_nodes {spec.max_nodes})")
    lines.append(f"Hourly cost: ${plan.hourly_usd:.2f}/h (budget ${spec.budget_usd_per_hour:.2f}/h; "
                 "new nodes priced at their pool's most expensive candidate type)")
    return lines


def check_limits(plan: Plan, spec: Spec) -> None:
    """Raise PlanError if the plan would run more than max_nodes, BudgetExceeded if it costs too much."""
    nodes = len(plan.keep) + len(plan.create)
    if nodes > spec.max_nodes:
        raise PlanError(f"max_nodes: plan runs {nodes} GPU nodes, limit is {spec.max_nodes}")
    if plan.hourly_usd > spec.budget_usd_per_hour:
        raise BudgetExceeded(
            f"budget_usd_per_hour: plan costs ${plan.hourly_usd:.2f}/h, budget is ${spec.budget_usd_per_hour:.2f}/h"
        )
