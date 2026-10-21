"""Reconciler: brings Lambda and the cluster's allow-lists in line with the spec (design spec sections 3.3, 5.4).

sync_peers rewrites the peer allow-lists (`gpuctl sync-peers`); apply and down create and delete GPU nodes
(`gpuctl apply`, `gpuctl down`); drift feeds `gpuctl status`.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from gpuctl.infra import CommandError, MissingStackError
from gpuctl.lambda_api import FirewallRule, Instance, InstanceType, InsufficientCapacity, LambdaAPIError, Ruleset
from gpuctl.peers import (
    PeerSet, cp_guard_elements, global_rules, nft_replace_script, region_ruleset_rules, ruleset_name,
)
from gpuctl.planner import (
    Plan, PlanError, check_limits, choose_placement, gpu_type_label, make_plan, owned_instances, plan_lines,
    target_counts,
)
from gpuctl.spec import Pool, Spec
from gpuctl.status import find_drift

# Ruling R14: the only writer of k3s-cp's peer list. It applies an nft_replace_script() payload from stdin and
# persists it, so k3s-guard.service restores the list after a reboot or reload (node/guard/k3s-guard-peers.sh).
GUARD_PEERS_COMMAND = "sudo /usr/local/sbin/k3s-guard-peers"


class ReconcileError(Exception):
    """The reconciler cannot proceed (e.g. a required Terraform output is missing)."""


class OverridesError(PlanError):
    """The autoscaler's override file exists but cannot be read as {"pools": {"<pool>": <int >= 0>}}."""


def load_overrides(path: Path) -> dict[str, int]:
    """Per-pool targets from the stretch autoscaler's lambda/.autoscale-state.json; {} when there is no file.

    plan, apply and the status drift check all read the file through this (Ruling R20), so gpu-plan shows exactly
    what gpu-apply does. target_counts() applies an override only to a pool with autoscale bounds, clamped to them.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except OSError as e:
        raise OverridesError(f"cannot read {path}: {e.strerror or e}") from None
    except ValueError as e:  # JSONDecodeError, UnicodeDecodeError
        raise OverridesError(f"{path} is not valid JSON ({e})") from None
    pools = data.get("pools") if isinstance(data, dict) else None
    if not isinstance(pools, dict) or not all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in pools.values()):
        raise OverridesError(f'{path}: expected {{"pools": {{"<pool>": <integer >= 0>, ...}}}}')
    return dict(pools)


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


# A launched instance in these states will never become active.
_GONE_STATUSES = frozenset({"terminating", "terminated"})
CONFIRM_PROMPT = "Apply these changes? [y/N] "


@dataclass(frozen=True)
class _Launched:
    id: str
    name: str
    instance_type: str


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
        """The CPU worker IPs from infra/workers. Runs before any write, so a failure here changes nothing.

        Empty only when there are provably no workers: infra/workers is not in this checkout (warning on stderr),
        or the stack has no worker_ips (never applied: init succeeds and the outputs are {}). ensure_init comes
        first, so a checkout without infra/workers/.terraform (fresh clone, git worktree) is not "no workers".
        Any other CommandError (init, GCS, a timeout) fails the sync: treating it as empty would drop every CPU
        worker from k3s-cp's k3s_peers and from the Lambda rulesets.
        """
        try:
            self.infra.ensure_init("workers")
            outputs = self.infra.tf_output("workers")
        except MissingStackError as e:
            sys.stderr.write(_clean(f"warning: infra/workers is not in this checkout, treating the CPU worker list "
                                    f"as empty ({e})") + "\n")
            return ()
        except CommandError as e:
            raise ReconcileError(
                f"cannot read infra/workers outputs ({e}); nothing was changed. An empty CPU worker list would cut "
                "every worker off from k3s-cp, so fix the error (for example terraform init in infra/workers) and "
                "re-run sync-peers") from e
        workers = outputs.get("worker_ips") or {}
        if isinstance(workers, dict):
            workers = list(workers.values())
        if not isinstance(workers, list):
            raise ReconcileError(f"infra/workers output worker_ips has an unexpected shape: {workers!r}")
        return tuple(workers)

    def _base_outputs(self) -> dict:
        base = self.infra.tf_output("base")
        missing = [key for key in ("cp_ip", "cp_internal_ip") if not base.get(key)]
        if missing:
            raise ReconcileError(f"infra/base outputs lack {', '.join(missing)}: apply the base stack first")
        return base

    def _peer_set(self) -> PeerSet:
        """Every peer's current IP: cp (Terraform), CPU workers (Terraform), live GPU instances, the operator."""
        base = self._base_outputs()
        worker_ips = self._worker_ips()
        gpu_ips = tuple(sorted(i.ip for i in owned_instances(self.client.list_instances()) if i.ip is not None))
        operator_ip = self.infra.detect_operator_ip()
        return PeerSet(cp_ip=base["cp_ip"], cp_internal_ip=base["cp_internal_ip"], worker_ips=worker_ips,
                       gpu_ips=gpu_ips, operator_ip=operator_ip)

    def sync_peers(self, spec: Spec) -> PeerSet:
        """Rebuild every peer allow-list from the current IPs (spec section 3.3); safe to re-run.

        Order: cp's k3s_peers set (persisted by the guard helper), Lambda's global rules, then the k3s-gpu-<region>
        ruleset of every spec region and of any other k3s-gpu-* ruleset still on the account.
        """
        peers = self._peer_set()
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

    # --- apply / down ---------------------------------------------------------------------------

    def _warn(self, line: str) -> None:
        sys.stderr.write(_clean(line) + "\n")

    def wait_active(self, instance_id: str) -> Instance:
        """Poll get_instance every poll_s until the instance is active with an IP (Review Focus #2: never return a
        null IP). TimeoutError after launch_timeout_s; LambdaAPIError if it is terminating or terminated."""
        deadline = self.clock() + self.settings.launch_timeout_s
        while True:
            instance = self.client.get_instance(instance_id)
            if instance.status in _GONE_STATUSES:
                raise LambdaAPIError("gpuctl/instance-gone", 200,
                                     f"{instance.name} ({instance_id}) is {instance.status}; it never became active")
            if instance.status == "active" and instance.ip:
                return instance
            if self.clock() >= deadline:
                raise TimeoutError(f"{instance.name} ({instance_id}) has no IP after "
                                   f"{self.settings.launch_timeout_s:.0f}s (status {instance.status})")
            self.sleep(self.settings.poll_s)

    def apply(self, spec: Spec, *, auto_approve: bool, keep_on_failure: bool) -> int:
        """Make Lambda match the spec (plus the autoscaler's overrides), then the firewalls and the cluster.

        Returns 0 on success, 1 if a node failed (new nodes that failed are terminated again unless
        keep_on_failure), 2 if the plan is refused (invalid override file, over budget or max_nodes) or the operator
        declines. Exceptions before any new instance exists (a failed delete, an unreadable account) propagate.
        """
        try:
            overrides = load_overrides(self.settings.overrides_path)
        except OverridesError as e:
            self._warn(f"apply refused: {e}")
            return 2
        return self._reconcile(spec, overrides=overrides, set_counts=None, auto_approve=auto_approve,
                               keep_on_failure=keep_on_failure)

    def down(self, spec: Spec, *, auto_approve: bool) -> int:
        """Remove the autoscaler's override file, then reconcile every pool to 0 (spec section 10.4: gpu-down always
        wins over the autoscaler). A second run finds nothing to delete and only re-applies the empty firewall."""
        path = Path(self.settings.overrides_path)
        if path.exists():
            path.unlink()
            self._emit(f"Removed the autoscaler overrides {path}")
        return self._reconcile(spec, overrides=None, set_counts={name: 0 for name in spec.pools},
                               auto_approve=auto_approve, keep_on_failure=False)

    def _reconcile(self, spec: Spec, *, overrides: Mapping[str, int] | None, set_counts: Mapping[str, int] | None,
                   auto_approve: bool, keep_on_failure: bool) -> int:
        # 1. plan and limits: nothing has changed yet
        instances = self.client.list_instances()
        types = self.client.instance_types()
        try:
            counts = target_counts(spec, overrides=overrides, set_counts=set_counts)
            plan = make_plan(spec, instances, types, counts)
            check_limits(plan, spec)
        except PlanError as e:  # BudgetExceeded included
            self._warn(f"plan refused: {e}")
            return 2
        # 2. show it; with no instance changes go straight to the firewall (step 7)
        for line in plan_lines(plan, spec):
            self._emit(line)
        if plan.create or plan.delete:
            # 3. confirm
            if not auto_approve:
                if self.confirm is None:
                    self._warn("refused: nothing to confirm with; re-run with --auto-approve")
                    return 2
                if not self.confirm(CONFIRM_PROMPT):
                    self._warn("Cancelled: nothing was changed")
                    return 2
            # 4. deletes before creates
            self._delete(plan)
        return self._create_and_join(spec, plan, types, keep_on_failure=keep_on_failure)

    def _delete(self, plan: Plan) -> None:
        duplicate_ids = {i.id for i in plan.duplicates}
        for instance in plan.delete:
            if instance.id in duplicate_ids:
                # Ruling R17: the node name and host key belong to the instance that is kept, so by id only
                self.client.terminate([instance.id])
                self._emit(f"Terminated duplicate {instance.name} {instance.id}")
                continue
            self.infra.drain_and_delete_node(instance.name)
            self.client.terminate([instance.id])
            self.infra.forget_host_key(instance.name)
            self._emit(f"Deleted {instance.name} {instance.id}")

    def _launch(self, pool: Pool, name: str, types: Mapping[str, InstanceType], rules: list[FirewallRule],
                failures: list[str], unknown: list[str]) -> _Launched | None:
        """Step 5 for one node: the first (type, region) with capacity, falling back on insufficient-capacity."""
        exclude: set[tuple[str, str]] = set()
        while True:
            choice = choose_placement(pool, types, frozenset(exclude))
            if choice is None:
                failures.append(f"{name}: no capacity for any candidate type in any region of pool {pool.name}")
                return None
            instance_type, region = choice
            try:
                ruleset = self.ensure_ruleset(region, rules)
                instance_id = self.client.launch(name=name, region=region, instance_type=instance_type,
                                                 ssh_key_name=self.settings.ssh_key_name,
                                                 image_family=pool.image_family, ruleset_id=ruleset.id)
            except InsufficientCapacity:
                self._emit(f"{name}: no capacity for {instance_type} in {region}, trying the next candidate")
                exclude.add(choice)
                continue
            except LambdaAPIError as e:
                # The outcome is unknown (Lambda may have started it): the cleanup looks the name up.
                failures.append(f"{name}: launch failed: {e}")
                unknown.append(name)
                return None
            self._emit(f"Launched {name} {instance_id} ({instance_type} in {region})")
            return _Launched(instance_id, name, instance_type)

    def _reconcile_firewall(self, spec: Spec) -> None:
        """Step 7: the gpu-access stack and every peer allow-list, for exactly the live GPU instances that have IPs."""
        ips = [i.ip for i in owned_instances(self.client.list_instances()) if i.ip is not None]
        self.infra.apply_gpu_access(ips, max_nodes=spec.max_nodes)
        self.sync_peers(spec)

    def _create_and_join(self, spec: Spec, plan: Plan, types: Mapping[str, InstanceType], *,
                         keep_on_failure: bool) -> int:
        launched: list[_Launched] = []
        unknown: list[str] = []  # names whose launch may or may not have started an instance
        joined: set[str] = set()
        failures: list[str] = []
        try:
            # 5. launch, each node into its region's ruleset (rules from the peers as they are after the deletes)
            if plan.create:
                rules = region_ruleset_rules(self._peer_set())
                for name, pool_name in plan.create:
                    new = self._launch(spec.pools[pool_name], name, types, rules, failures, unknown)
                    if new is not None:
                        launched.append(new)
            # 6. wait for each IP
            ready: list[tuple[_Launched, Instance]] = []
            for new in launched:
                try:
                    ready.append((new, self.wait_active(new.id)))
                except (TimeoutError, LambdaAPIError) as e:
                    failures.append(f"{new.name}: {e}")
            # 7. firewalls for every live IP (also corrects them when nothing else changed)
            self._reconcile_firewall(spec)
            # 8. join
            for new, instance in ready:
                try:
                    self.infra.join_node(new.name, instance.ip, gpu_type_label(new.instance_type))
                    joined.add(new.id)
                except (CommandError, ValueError) as e:
                    failures.append(f"{new.name}: join failed: {e}")
        except (Exception, KeyboardInterrupt) as e:  # anything else (Ctrl-C included): clean up as for a failed node
            failures.append(f"apply failed: {str(e) or type(e).__name__}")
        if not failures:
            self._emit("GPU nodes match the spec")
            return 0
        for failure in failures:
            self._warn(f"error: {failure}")
        failed = [new for new in launched if new.id not in joined]
        if keep_on_failure:
            if failed or unknown:
                names = ", ".join([new.name for new in failed] + unknown)
                self._warn(f"--keep-on-failure: kept {names} for troubleshooting; they bill until "
                           "make gpu-down (or a gpu-apply that no longer wants them)")
            return 1
        self._clean_up(spec, failed, unknown)
        return 1

    def _clean_up(self, spec: Spec, failed: list[_Launched], unknown: list[str]) -> None:
        """Step 9: terminate this run's failed new instances, drop their nodes and host keys, redo step 7.

        Best effort: one failing step does not stop the others, and each is reported.
        """
        victims = [(new.id, new.name) for new in failed]
        if unknown:
            try:
                known = {new.id for new in failed}
                victims += [(i.id, i.name) for i in owned_instances(self.client.list_instances())
                            if i.name in unknown and i.id not in known]
            except Exception as e:
                self._warn(f"cleanup: cannot list instances to find {', '.join(unknown)}: {e}; check by hand")
        for instance_id, name in victims:
            for what, step in (("terminate", lambda: self.client.terminate([instance_id])),
                               ("delete node", lambda: self.infra.drain_and_delete_node(name)),
                               ("forget host key", lambda: self.infra.forget_host_key(name))):
                try:
                    step()
                except Exception as e:
                    self._warn(f"cleanup: {what} {name} ({instance_id}) failed: {e}")
            self._emit(f"Cleaned up {name} {instance_id}")
        try:
            self._reconcile_firewall(spec)
        except Exception as e:
            self._warn(f"cleanup: firewall reconcile failed: {e}; re-run make gpu-apply or make sync-peers")

    # --- drift ----------------------------------------------------------------------------------

    def _gpu_access_ips(self) -> list[str]:
        self.infra.ensure_init("gpu-access")
        ips = self.infra.tf_output("gpu-access").get("gpu_ips", [])  # never applied: no outputs, no rules
        if not isinstance(ips, list) or not all(isinstance(ip, str) for ip in ips):
            raise ValueError(f"infra/gpu-access output gpu_ips has an unexpected shape: {ips!r}")
        return ips

    def _gpu_peer_elements(self) -> list[str]:
        """The k3s_peers entries that are not the cp or a CPU worker, i.e. the ones that should be GPU nodes."""
        base = self._base_outputs()
        others = {base["cp_ip"], base["cp_internal_ip"], *self._worker_ips()}
        return [ip for ip in self.infra.cp_guard_elements() if ip not in others]

    def drift(self, spec: Spec, instances: Iterable[Instance]) -> list[str]:
        """find_drift over the live sources; a source that cannot be read is skipped, with the reason on stderr."""
        def read(what: str, fn: Callable[[], object]):
            try:
                return fn()
            except (CommandError, ReconcileError, PlanError, ValueError, OSError) as e:
                self._warn(f"warning: cannot read {what}: {e}")
                return None

        return find_drift(
            counts=read("the target counts", lambda: target_counts(
                spec, overrides=load_overrides(self.settings.overrides_path))),
            instances=list(instances),
            cluster_nodes=read("the cluster's GPU nodes", self.infra.cluster_gpu_nodes),
            tf_ips=read("the gpu-access state", self._gpu_access_ips),
            nft_elements=read("k3s-cp's k3s_peers set", self._gpu_peer_elements),
        )
