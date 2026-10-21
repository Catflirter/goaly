"""Peer allow-lists: the Lambda firewall rules and the cp's `k3s_peers` nft set, both rebuilt from scratch.

Every function here derives its output only from the current PeerSet, so re-running sync with a new
operator IP replaces the old one everywhere instead of adding to it (spec §3.3, Review Focus #1).
"""
from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from dataclasses import dataclass

from gpuctl.lambda_api import FirewallRule

NFT_SET = "inet k3s_guard k3s_peers"


def _ipv4(value: object, field: str) -> ipaddress.IPv4Address:
    if not isinstance(value, str):
        raise ValueError(f"{field}: {value!r} is not an IP address string")
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        raise ValueError(f"{field}: {value!r} is not an IP address") from None
    if not isinstance(ip, ipaddress.IPv4Address):
        raise ValueError(f"{field}: {value!r} is not IPv4 (every rule source is a single IPv4 /32)")
    return ip


def _global_ipv4(value: object, field: str) -> ipaddress.IPv4Address:
    ip = _ipv4(value, field)
    if not ip.is_global:
        raise ValueError(f"{field}: {value!r} is not a global (public) address")
    return ip


def _sorted_unique(ips: Iterable[ipaddress.IPv4Address]) -> list[str]:
    return [str(ip) for ip in sorted(set(ips))]


@dataclass(frozen=True)
class PeerSet:
    cp_ip: str
    cp_internal_ip: str
    worker_ips: tuple[str, ...]
    gpu_ips: tuple[str, ...]
    operator_ip: str

    def __post_init__(self) -> None:
        _global_ipv4(self.cp_ip, "cp_ip")
        _global_ipv4(self.operator_ip, "operator_ip")
        if not _ipv4(self.cp_internal_ip, "cp_internal_ip").is_private:
            raise ValueError(f"cp_internal_ip: {self.cp_internal_ip!r} is not a private address")
        for field in ("worker_ips", "gpu_ips"):
            values = getattr(self, field)
            if not isinstance(values, tuple):
                raise ValueError(f"{field} must be a tuple, got {type(values).__name__}")
            for value in values:
                _global_ipv4(value, field)

    def _cluster_public_ips(self) -> list[ipaddress.IPv4Address]:
        return [ipaddress.IPv4Address(ip) for ip in (self.cp_ip, *self.worker_ips, *self.gpu_ips)]


def cp_guard_elements(p: PeerSet) -> list[str]:
    """Members of the cp's `k3s_peers` set: cp (public + internal), every worker, every GPU node."""
    return _sorted_unique([*p._cluster_public_ips(), ipaddress.IPv4Address(p.cp_internal_ip)])


def nft_replace_script(elements: Iterable[str]) -> str:
    """An `nft -f` script that replaces the whole `k3s_peers` set (flush + add run in one transaction)."""
    ips = [str(_ipv4(e, "element")) for e in elements]
    if not ips:
        raise ValueError("refusing to empty k3s_peers: the cp itself is always a member")
    return f"flush set {NFT_SET}\nadd element {NFT_SET} {{ {', '.join(ips)} }}\n"


def global_rules(operator_ip: str) -> list[FirewallRule]:
    op = _global_ipv4(operator_ip, "operator_ip")
    return [FirewallRule("tcp", f"{op}/32", (22, 22), "operator ssh")]


def region_ruleset_rules(p: PeerSet) -> list[FirewallRule]:
    rules = [
        *global_rules(p.operator_ip),
        FirewallRule("tcp", f"{p.cp_ip}/32", (10250, 10250), "metrics-server from cp"),
    ]
    for ip in _sorted_unique(p._cluster_public_ips()):
        rules.append(FirewallRule("udp", f"{ip}/32", (51820, 51820), f"wireguard {ip}"))
        rules.append(FirewallRule("icmp", f"{ip}/32", None, f"icmp {ip}"))
    return rules


def ruleset_name(region: str) -> str:
    return f"k3s-gpu-{region}"
