"""Reconciler: brings Lambda and the cluster's allow-lists in line with the spec (design spec sections 3.3, 5.4).

Task 13 provides the peer sync (`gpuctl sync-peers`); Task 20 adds apply, down and wait_active on the same class.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from gpuctl.infra import CommandError
from gpuctl.lambda_api import FirewallRule, Ruleset
from gpuctl.peers import (
    PeerSet, cp_guard_elements, global_rules, nft_replace_script, region_ruleset_rules, ruleset_name,
)
from gpuctl.planner import owned_instances
from gpuctl.spec import Spec

# Ruling R14: the only writer of k3s-cp's peer list. It applies an nft_replace_script() payload from stdin and
# persists it, so k3s-guard.service restores the list after a reboot or reload (node/guard/k3s-guard-peers.sh).
GUARD_PEERS_COMMAND = "sudo /usr/local/sbin/k3s-guard-peers"


class ReconcileError(Exception):
    """The reconciler cannot proceed (e.g. a required Terraform output is missing)."""


@dataclass(frozen=True)
class Settings:
    spec_path: Path
    overrides_path: Path
    ssh_key_name: str
    launch_timeout_s: float = 900.0
    poll_s: float = 10.0


def mask_ip(ip: str) -> str:
    """73.15.40.40 -> 73.15.40.x: the operator's home IP stays out of logs and screen shares."""
    head, _, _ = ip.rpartition(".")
    return f"{head}.x"


def _clean(text: str) -> str:
    return "".join(ch if ch.isprintable() else "?" for ch in text)


class Reconciler:
    def __init__(self, client, infra, settings: Settings, *, out: TextIO | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 confirm: Callable[[str], bool] | None = None) -> None:
        self.client = client
        self.infra = infra
        self.settings = settings
        self.out = out  # None: the sys.stdout current at each write (Ruling R18)
        self.clock = clock
        self.sleep = sleep
        self.confirm = confirm

    def _emit(self, line: str = "") -> None:
        out = sys.stdout if self.out is None else self.out
        out.write(_clean(line) + "\n")

    def ensure_ruleset(self, region: str, rules: Iterable[FirewallRule]) -> Ruleset:
        """Create k3s-gpu-<region> with `rules`, or replace the rules of every existing ruleset of that name."""
        rules = tuple(rules)
        name = ruleset_name(region)
        existing = sorted((r for r in self.client.list_rulesets() if r.name == name), key=lambda r: r.id)
        if not existing:
            return self.client.create_ruleset(name, region, rules)
        for ruleset in existing:
            self.client.update_ruleset(ruleset.id, rules)
        first = existing[0]
        return Ruleset(first.id, first.name, first.region, rules)

    def _worker_ips(self) -> tuple[str, ...]:
        try:
            outputs = self.infra.tf_output("workers")
        except CommandError as e:
            self._emit(f"warning: cannot read infra/workers outputs, treating the CPU worker list as empty ({e})")
            return ()
        workers = outputs.get("worker_ips") or {}
        if isinstance(workers, dict):
            workers = list(workers.values())
        if not isinstance(workers, list):
            raise ReconcileError(f"infra/workers output worker_ips has an unexpected shape: {workers!r}")
        return tuple(workers)

    def sync_peers(self, spec: Spec) -> PeerSet:
        """Rebuild every peer allow-list from the current IPs (spec section 3.3); safe to re-run.

        Order: cp's k3s_peers set (persisted by the guard helper), Lambda's global rules, then the k3s-gpu-<region>
        ruleset of every spec region and of any other k3s-gpu-* ruleset still on the account.
        """
        base = self.infra.tf_output("base")
        missing = [key for key in ("cp_ip", "cp_internal_ip") if not base.get(key)]
        if missing:
            raise ReconcileError(f"infra/base outputs lack {', '.join(missing)}: apply the base stack first")
        worker_ips = self._worker_ips()
        gpu_ips = tuple(sorted(i.ip for i in owned_instances(self.client.list_instances()) if i.ip is not None))
        operator_ip = self.infra.detect_operator_ip()
        peers = PeerSet(cp_ip=base["cp_ip"], cp_internal_ip=base["cp_internal_ip"], worker_ips=worker_ips,
                        gpu_ips=gpu_ips, operator_ip=operator_ip)

        elements = cp_guard_elements(peers)
        self.infra.cp_ssh(GUARD_PEERS_COMMAND, input=nft_replace_script(elements))
        self.client.update_global_rules(global_rules(peers.operator_ip))
        # Spec regions first; then owned rulesets of regions dropped from the spec, so an old operator IP is
        # removed from those too (Review Focus #1).
        leftover = [r.region for r in self.client.list_rulesets() if r.name == ruleset_name(r.region)]
        regions = list(dict.fromkeys([*spec.regions(), *leftover]))
        rules = region_ruleset_rules(peers)
        for region in regions:
            self.ensure_ruleset(region, rules)

        self._emit(f"sync-peers: operator {mask_ip(peers.operator_ip)}, cp {peers.cp_ip} ({peers.cp_internal_ip}), "
                   f"{len(peers.worker_ips)} CPU worker(s), {len(peers.gpu_ips)} GPU node(s)")
        self._emit(f"  k3s-cp k3s_peers: {len(elements)} addresses")
        self._emit("  Lambda global rules: operator ssh only")
        self._emit(f"  Lambda rulesets: {', '.join(ruleset_name(r) for r in regions)}")
        return peers
