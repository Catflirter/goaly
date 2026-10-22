"""Read-only views for `gpuctl status`: GPU capacity per (pool, candidate type, region), and the drift check
(Lambda instances vs K8s nodes vs gpu-access state vs the cp's k3s_peers set, design spec section 5.4).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from gpuctl.lambda_api import Instance, InstanceType
from gpuctl.planner import owned_instances, parse_node_name
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


SKIPPED_PREFIX = "skipped: "
SECURITY_PREFIX = "SECURITY: "
# The sources find_drift compares, in the order of its "skipped: <source> unavailable" lines.
DRIFT_SOURCES = ("target counts", "Lambda instances", "cluster nodes", "gpu-access state", "cp k3s_peers set")


def _stale_and_missing(listed: Iterable[str], live_ips: set[str], what: str) -> tuple[list[str], list[str]]:
    """An allow-list against the live GPU IPs: entries nothing live owns (a security problem), and gaps."""
    listed = set(listed)
    stale = [f"{SECURITY_PREFIX}{what} for {ip} has no live instance" for ip in sorted(listed - live_ips)]
    missing = [f"missing {what} for {ip}" for ip in sorted(live_ips - listed)]
    return stale, missing


def find_drift(*, counts: Mapping[str, int] | None, instances: Iterable[Instance] | None,
               cluster_nodes: Mapping[str, str | None] | None, tf_ips: Iterable[str] | None,
               nft_elements: Iterable[str] | None) -> list[str]:
    """Compare the four sources of truth and return one line per problem, "SECURITY: " lines first.

    Any source may be None (it could not be read): that adds a "skipped: <source> unavailable" line, which is not
    a problem, and every comparison needing it is left out. `nft_elements` holds only the GPU entries of k3s_peers
    (the caller removes the cp and worker addresses). Instances that are not ours or not live are ignored.
    """
    sources = dict(zip(DRIFT_SOURCES, (counts, instances, cluster_nodes, tf_ips, nft_elements), strict=True))
    skipped = [f"{SKIPPED_PREFIX}{name} unavailable" for name, value in sources.items() if value is None]
    if instances is None:
        return skipped
    live = owned_instances(instances)
    live_names = {i.name for i in live}
    live_ips = {i.ip for i in live if i.ip is not None}

    security: list[str] = []
    problems: list[str] = []
    if cluster_nodes is not None:
        problems += [f"{name} is not in the cluster" for name in sorted(live_names - set(cluster_nodes))]
        problems += [f"stale node {name}" for name in sorted(set(cluster_nodes) - live_names)]
    for listed, what in ((tf_ips, "firewall rule"), (nft_elements, "cp k3s_peers entry")):
        if listed is not None:
            stale, missing = _stale_and_missing(listed, live_ips, what)
            security += stale
            problems += missing
    if counts is not None:
        have = Counter(parsed[0] for parsed in map(parse_node_name, (i.name for i in live)) if parsed is not None)
        for pool in [*counts, *sorted(set(have) - set(counts))]:
            wanted = counts.get(pool, 0)
            if have[pool] != wanted:
                problems.append(f"pool {pool} has {have[pool]}, wants {wanted}")
    return skipped + security + problems


def is_problem(line: str) -> bool:
    """A find_drift line that is drift, as opposed to a source that could not be read."""
    return not line.startswith(SKIPPED_PREFIX)
