"""Read-only views of Lambda for `gpuctl status`: GPU capacity per (pool, candidate type, region).

Task 20 adds the drift check (Lambda instances vs K8s nodes vs gpu-access state vs the cp's guard set).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from gpuctl.lambda_api import InstanceType
from gpuctl.spec import Spec

_HEADER = ("POOL", "TYPE", "REGION", "AVAILABLE", "PRICE")


@dataclass(frozen=True)
class CapacityRow:
    pool: str
    instance_type: str
    region: str
    available: bool
    price_usd: float | None  # None: Lambda does not list the type at all


def capacity_rows(spec: Spec, types: Mapping[str, InstanceType]) -> list[CapacityRow]:
    """One row per pool x candidate type (preference order) x region, the order gpu-apply tries them in."""
    rows = []
    for pool in spec.pools.values():
        for type_name in pool.candidate_types():
            instance_type = types.get(type_name)
            price = None if instance_type is None else instance_type.price_cents_per_hour / 100
            for region in pool.regions:
                available = instance_type is not None and region in instance_type.regions_with_capacity
                rows.append(CapacityRow(pool.name, type_name, region, available, price))
    return rows


def render_capacity(rows: list[CapacityRow]) -> str:
    """A header plus one aligned line per row; availability is yes/no, price $x.xx/h or - when unknown."""
    table = [_HEADER] + [
        (r.pool, r.instance_type, r.region, "yes" if r.available else "no",
         "-" if r.price_usd is None else f"${r.price_usd:.2f}/h")
        for r in rows
    ]
    widths = [max(len(line[i]) for line in table) for i in range(len(_HEADER))]
    return "".join("  ".join(cell.ljust(w) for cell, w in zip(line, widths)).rstrip() + "\n" for line in table)
