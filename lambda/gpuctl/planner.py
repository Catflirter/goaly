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

# Canonical names only (index >= 1, no leading zeros), so a name and its (pool, index) map one to one.
_NODE_NAME_RE = re.compile(rf"{re.escape(PREFIX)}({POOL_NAME_RE.pattern})-([1-9][0-9]*)")
_TYPE_PREFIX_RE = re.compile(r"gpu_[0-9]+x_")


class PlanError(Exception):
    """The plan cannot be made or violates a limit."""


class BudgetExceeded(PlanError):
    """The plan's hourly cost is above budget_usd_per_hour."""


@dataclass(frozen=True)
class Plan:
    create: tuple[tuple[str, str], ...]  # (node name, pool)
    delete: tuple[Instance, ...]
    keep: tuple[Instance, ...]
    hourly_usd: float


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
    """gpu_1x_a6000 -> a6000; gpu_8x_a100_80gb_sxm4 -> a100_80gb_sxm4."""
    return _TYPE_PREFIX_RE.sub("", instance_type, count=1)


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

    # One instance per name: the smallest id wins, the rest are deleted.
    by_name: dict[str, Instance] = {}
    duplicates: list[Instance] = []
    for instance in sorted(owned_instances(instances), key=lambda i: (i.name, i.id)):
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

    return Plan(create=tuple(create), delete=tuple(delete), keep=tuple(keep), hourly_usd=cents / 100)


def check_limits(plan: Plan, spec: Spec) -> None:
    """Raise PlanError if the plan would run more than max_nodes, BudgetExceeded if it costs too much."""
    nodes = len(plan.keep) + len(plan.create)
    if nodes > spec.max_nodes:
        raise PlanError(f"max_nodes: plan runs {nodes} GPU nodes, limit is {spec.max_nodes}")
    if plan.hourly_usd > spec.budget_usd_per_hour:
        raise BudgetExceeded(
            f"budget_usd_per_hour: plan costs ${plan.hourly_usd:.2f}/h, budget is ${spec.budget_usd_per_hour:.2f}/h"
        )
