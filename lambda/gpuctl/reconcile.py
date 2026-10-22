"""Reconciler: brings Lambda and the cluster's allow-lists in line with the spec (design spec sections 3.3, 5.4).

sync_peers rewrites the peer allow-lists (`gpuctl sync-peers`); apply and down create and delete GPU nodes
(`gpuctl apply`, `gpuctl down`); drift feeds `gpuctl status`.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from gpuctl.infra import DEFAULT_KUBECONFIG, CommandError, MissingStackError
from gpuctl.lambda_api import FirewallRule, Instance, InstanceType, InsufficientCapacity, LambdaAPIError, Ruleset
from gpuctl.output import write
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
# The fix when the kubeconfig reaches another cluster, or may (D-2, review X0-3): ~/.kube/hybrid-gpu.yaml and the
# tunnel port are one per operator, so another project's make up or make kubeconfig may have taken them.
POINT_AT_THIS_CLUSTER = ("point the kubeconfig and the API tunnel at this project's k3s-cp (make kubeconfig; stop any "
                         "other tunnel on 127.0.0.1:6443, then make tunnel)")


def _is_transient(error: LambdaAPIError) -> bool:
    """A poll error worth retrying: no HTTP answer (network/transport, status 0), rate limiting, or a 5xx."""
    return error.status == 0 or error.status == 429 or error.status >= 500
CONFIRM_PROMPT = "Apply these changes? [y/N] "


def duration(seconds: float) -> str:
    """A step's duration for the progress lines: 42s, 5m21s."""
    whole = int(seconds)
    return f"{whole}s" if whole < 60 else f"{whole // 60}m{whole % 60:02d}s"


class _Step:
    """What a step reports when its block ends without an exception: done, unless the block marked it failed."""

    def __init__(self) -> None:
        self.ok = True

    def failed(self) -> None:
        self.ok = False


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
        self._started = clock()  # reset by each apply/down: the progress stamps count from there
        self._command = "apply"
        self._no_drain = False  # gpuctl down --no-drain
        # Once a gpu-down drain failed (Ruling R54): why the cluster is known to be gone (later nodes are terminated
        # without a drain), or why an existing cluster cannot be drained now (later nodes are refused at once).
        self._gone: str | None = None
        self._refusal: str | None = None
        # With _refusal (set together by _refuse): what is wrong, and what the operator does about it.
        self._problem: str | None = None
        self._fix: str | None = None
        self._cluster_checked = False  # _check_cluster ran in this apply or down (D-2)

    def _emit(self, line: str = "") -> None:
        # Flushed (L21-4: a piped stdout, make gpu-apply | tee, shows each line as it happens) and best effort: a
        # reader that is gone never stops the step being reported (review C0-1/C0-2/X0-1, gpuctl.output).
        write(sys.stdout if self.out is None else self.out, _clean(line) + "\n")

    def _stamp(self, now: float) -> str:
        """[+MM:SS] since this apply or down started."""
        elapsed = int(now - self._started)
        return f"[+{elapsed // 60:02d}:{elapsed % 60:02d}]"

    @contextmanager
    def _step(self, what: str) -> Iterator[_Step]:
        """Progress lines around one step of apply or down (L21-4): "[+MM:SS] WHAT ..." when it starts, then
        "WHAT: done in 12s", or "WHAT: failed after 12s" when it raises or the block calls step.failed()."""
        step = _Step()
        start = self.clock()
        self._emit(f"{self._stamp(start)} {what} ...")
        try:
            yield step
        except BaseException:
            end = self.clock()
            self._emit(f"{self._stamp(end)} {what}: failed after {duration(end - start)}")
            raise
        end = self.clock()
        result = "done in" if step.ok else "failed after"
        self._emit(f"{self._stamp(end)} {what}: {result} {duration(end - start)}")

    def _join_line(self, line: str) -> None:
        """One line of join-node.sh's output, under its step."""
        self._emit(f"    | {line}")

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
        first, so a checkout whose infra/workers was never initialised against GCS (fresh clone, git worktree, or
        only `make tf-test`) is not "no workers".
        Any other CommandError (init, GCS, a timeout) fails the sync: treating it as empty would drop every CPU
        worker from k3s-cp's k3s_peers and from the Lambda rulesets.
        """
        try:
            self.infra.ensure_init("workers")
            outputs = self.infra.tf_output("workers")
        except MissingStackError as e:
            self._warn(f"warning: infra/workers is not in this checkout, treating the CPU worker list as empty ({e})")
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

    def _read_base(self) -> dict:
        # A fresh clone or worktree has never initialised infra/base (gpu-down's final sync-peers included).
        self.infra.ensure_init("base")
        return self.infra.tf_output("base")

    @staticmethod
    def _base_destroyed(base: Mapping) -> bool:
        """infra/base has none of the cluster's outputs: make down destroyed it (or it was never applied). Ruling
        R54's "no base stack outputs"; a stack with only some of them is an error, not a gone cluster."""
        return not base.get("cp_ip") and not base.get("cp_internal_ip")

    def _base_outputs(self, base: Mapping | None = None) -> Mapping:
        base = self._read_base() if base is None else base
        missing = [key for key in ("cp_ip", "cp_internal_ip") if not base.get(key)]
        if missing:
            raise ReconcileError(f"infra/base outputs lack {', '.join(missing)}: apply the base stack first")
        return base

    def _cluster_gone(self) -> str | None:
        """Why the cluster is known to be gone (Ruling R54: no kubeconfig, or infra/base has no cluster outputs), or
        None. An API server that does not answer is not enough (review F3-1), and neither is a base stack that
        cannot be read: not knowing is not "gone"."""
        missing = self.infra.kubeconfig_missing()
        if missing is not None:
            return missing
        try:
            base = self._read_base()
        except (CommandError, ValueError, OSError) as e:
            self._warn(f"warning: cannot read infra/base outputs ({e}); treating the cluster as still there")
            return None
        if self._base_destroyed(base):
            return "infra/base has no cluster outputs (cp_ip, cp_internal_ip): the cluster was destroyed"
        return None

    def _live_gpu_ips(self, exclude_ids: Iterable[str] = ()) -> list[str]:
        """The IPs of the owned live GPU instances, minus `exclude_ids`: instances this run terminated, which Lambda
        may still list as active for a moment after the terminate call returns."""
        gone = set(exclude_ids)
        return sorted(i.ip for i in owned_instances(self.client.list_instances())
                      if i.ip is not None and i.id not in gone)

    def _peer_set(self, exclude_ids: Iterable[str] = (), base: Mapping | None = None) -> PeerSet:
        """Every peer's current IP: cp (Terraform), CPU workers (Terraform), live GPU instances, the operator.
        `base` is infra/base's outputs when the caller has just read them."""
        base = self._base_outputs(base)
        worker_ips = self._worker_ips()
        gpu_ips = tuple(self._live_gpu_ips(exclude_ids))
        operator_ip = self.infra.detect_operator_ip()
        return PeerSet(cp_ip=base["cp_ip"], cp_internal_ip=base["cp_internal_ip"], worker_ips=worker_ips,
                       gpu_ips=gpu_ips, operator_ip=operator_ip)

    def sync_peers(self, spec: Spec, *, exclude_ids: Iterable[str] = (), base: Mapping | None = None) -> PeerSet:
        """Rebuild every peer allow-list from the current IPs (spec section 3.3); safe to re-run.

        Order: cp's k3s_peers set (persisted by the guard helper), Lambda's global rules, then the k3s-gpu-<region>
        ruleset of every spec region and of any other k3s-gpu-* ruleset still on the account. GPU instances in
        `exclude_ids` (terminated by this run) are left out even if Lambda still lists them as live.
        """
        peers = self._peer_set(exclude_ids, base)
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

    def _sync_lambda_rules_cluster_gone(self) -> None:
        """gpu-down's sync-peers once the cluster is gone (review F3-3, Ruling R54): there is no k3s-cp whose peer set
        could be updated, so that is skipped; Lambda's global rules and every existing owned k3s-gpu-* ruleset keep
        only the operator's SSH (no cluster address is left in them). No ruleset is created."""
        operator = self.infra.detect_operator_ip()
        rules = global_rules(operator)
        self.client.update_global_rules(rules)
        owned = [r for r in self.client.list_rulesets() if r.name == ruleset_name(r.region)]
        for ruleset in owned:
            self.client.update_ruleset(ruleset.id, rules)
        self._emit(f"sync-peers: the cluster is gone (infra/base has no cluster outputs), operator {mask_ip(operator)}")
        self._emit("  k3s-cp k3s_peers: skipped (there is no k3s-cp)")
        self._emit("  Lambda global rules: operator ssh only")
        self._emit(f"  Lambda rulesets: {', '.join(sorted({r.name for r in owned})) or 'none'} (operator ssh only)")

    # --- apply / down ---------------------------------------------------------------------------

    def _warn(self, line: str) -> None:
        write(sys.stderr, _clean(line) + "\n")  # best effort, like _emit

    def wait_active(self, instance_id: str) -> Instance:
        """Poll get_instance every poll_s until the instance is active with an IP (Review Focus #2: never return a
        null IP). TimeoutError after launch_timeout_s; LambdaAPIError if it is terminating or terminated, or on a
        permanent API error. A transient poll error (network, 429, 5xx) only logs a warning: the launch may still
        come up, and a failure here would terminate it."""
        deadline = self.clock() + self.settings.launch_timeout_s
        while True:
            try:
                instance = self.client.get_instance(instance_id)
            except LambdaAPIError as e:
                if not _is_transient(e):
                    raise
                if self.clock() >= deadline:
                    raise TimeoutError(f"{instance_id} has no IP after {self.settings.launch_timeout_s:.0f}s "
                                       f"(last poll failed: {e})") from e
                self._warn(f"warning: polling {instance_id} failed ({e}); retrying")
                self.sleep(self.settings.poll_s)
                continue
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
        declines. A failed delete returns 1 without launching anything, after removing the IPs of the instances it
        did terminate from every allow-list. Exceptions before any instance changes (an unreadable account)
        propagate.
        """
        try:
            overrides = load_overrides(self.settings.overrides_path)
        except OverridesError as e:
            self._warn(f"apply refused: {e}")
            return 2
        return self._reconcile(spec, overrides=overrides, set_counts=None, auto_approve=auto_approve,
                               keep_on_failure=keep_on_failure, command="apply")

    def down(self, spec: Spec, *, auto_approve: bool, no_drain: bool = False) -> int:
        """Remove the autoscaler's override file, then reconcile every pool to 0 (spec section 10.4: gpu-down always
        wins over the autoscaler). A second run finds nothing to delete and only re-applies the empty firewall.
        Nothing is recorded (spec section 5.4: no local state); `gpuctl status` treats the pools below their count
        as a note, not drift (L21-1).

        An owned instance whose drain fails is still terminated, with a warning, when the cluster is known to be gone
        (Ruling R54: no kubeconfig, or infra/base has no cluster outputs, as after make down); else it would bill on
        with no way to stop it (review F2-3, G-1). `no_drain` skips the drains on purpose. A cluster that still
        exists keeps its node when the drain fails, whether its API server refused the drain or does not answer
        (`make tunnel` not running, review F3-1): that instance is not terminated, the other steps go on, exit 1.
        With the cluster gone, the final sync-peers skips k3s-cp and leaves only the operator's SSH in the Lambda
        rules (review F3-3), so gpu-down exits 0 once no owned instance remains."""
        path = Path(self.settings.overrides_path)
        if path.exists():
            path.unlink()
            self._emit(f"Removed the autoscaler overrides {path}")
        return self._reconcile(spec, overrides=None, set_counts={name: 0 for name in spec.pools},
                               auto_approve=auto_approve, keep_on_failure=False, command="down", no_drain=no_drain)

    def _reconcile(self, spec: Spec, *, overrides: Mapping[str, int] | None, set_counts: Mapping[str, int] | None,
                   auto_approve: bool, keep_on_failure: bool, command: str = "apply", no_drain: bool = False) -> int:
        self._started = self.clock()
        self._command = command
        self._no_drain = no_drain
        self._gone = self._refusal = self._problem = self._fix = None
        self._cluster_checked = False
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
        gone: set[str] = set()  # ids this run terminated
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
            gone, attempted, failures = self._delete(plan)
            if plan.create and not failures:
                failures = self._launches_refused()
            if failures:
                return self._delete_failed(spec, gone, attempted, failures)
        return self._create_and_join(spec, plan, types, keep_on_failure=keep_on_failure, gone=gone)

    def _delete(self, plan: Plan) -> tuple[set[str], set[str], list[str]]:
        """Step 4, best effort: returns (ids terminated, ids a terminate was sent for, failures). A failing delete
        does not stop the others; Ctrl-C stops the remaining ones. Either way the caller still removes the
        terminated IPs from the firewalls. A terminate that raised may still have been accepted by Lambda (Ctrl-C
        or a network error after the request went out), so its id is in the second set but not the first.
        """
        duplicate_ids = {i.id for i in plan.duplicates}
        gone: set[str] = set()
        attempted: set[str] = set()
        failures: list[str] = []
        undrained: list[str] = []
        try:
            for instance in plan.delete:
                try:
                    if instance.id in duplicate_ids:
                        # Ruling R17: the node name and host key belong to the instance that is kept, so by id only
                        with self._step(f"terminate duplicate {instance.name} {instance.id}"):
                            attempted.add(instance.id)
                            self.client.terminate([instance.id])
                            gone.add(instance.id)
                        self._emit(f"Terminated duplicate {instance.name} {instance.id}")
                        continue
                    with self._step(f"delete {instance.name}"):
                        if not self._drain(instance.name):
                            undrained.append(instance.name)
                        attempted.add(instance.id)
                        self.client.terminate([instance.id])
                        gone.add(instance.id)
                        self.infra.forget_host_key(instance.name)
                    self._emit(f"Deleted {instance.name} {instance.id}")
                except Exception as e:
                    failures.append(f"delete {instance.name} ({instance.id}): {e}")
        except KeyboardInterrupt:
            failures.append("deletes interrupted (KeyboardInterrupt); the remaining deletes were not attempted")
        for name in undrained:
            if name in {i.name for i in plan.delete if i.id in gone}:
                self._warn(f"WARNING: {name} was terminated without draining it; its node object may remain in the "
                           "cluster (make gpu-status reports it). Once the cluster answers, remove it: "
                           f"kubectl --kubeconfig ~/{DEFAULT_KUBECONFIG} delete node {name} (join-node.sh also "
                           "deletes it, NotReady, before the next node of that name joins)")
        return gone, attempted, failures

    def _how_to_go_on(self, fix: str) -> str:
        """What the operator does after a refused drain: FIX and re-run, or (gpu-down only) --no-drain."""
        if self._command != "down":
            return f"{fix} and re-run make gpu-{self._command}"
        return (f"{fix} and re-run make gpu-down, or pass --no-drain (scripts/gpuctl down --no-drain) to terminate "
                "it anyway")

    def _refuse(self, problem: str, fix: str, refusal: str) -> None:
        """Refuse every drain, node delete and launch for the rest of this run: PROBLEM is what is wrong, FIX what
        the operator does about it, REFUSAL the whole message of a refused drain."""
        self._problem, self._fix, self._refusal = problem, fix, refusal

    def _not_this_cluster(self, problem: str, fix: str = POINT_AT_THIS_CLUSTER) -> None:
        """Refuse: the cluster the kubeconfig reaches is not, or may not be, this project's (D-2). FIX is the
        remedy for this cause (review X0-3)."""
        self._refuse(problem, fix, f"{problem}; gpuctl {self._command} never drains or deletes a node in another "
                                   "cluster, and does not terminate a node it could not drain: "
                     + self._how_to_go_on(fix))

    def _unreachable(self, reason: str) -> None:
        """Refuse: the API server does not answer while the cluster may still exist (review F3-1)."""
        self._refuse(reason, "start make tunnel",
                     f"{reason}, but the cluster may still exist (the kubeconfig is there and infra/base was not "
                     f"found destroyed), so gpuctl {self._command} does not terminate a node it could not drain: "
                     + self._how_to_go_on("start make tunnel"))

    def _check_cluster(self) -> None:
        """Before the first drain, node delete or launch of this apply or down (D-2): is the cluster the kubeconfig
        reaches this project's? The kubeconfig and the tunnel port are one per operator, not one per project, so it
        may be another one. Checked as scripts/down.sh same_cluster does: Node k3s-cp's ExternalIP is infra/base's
        cp_ip.

        Sets self._gone when the cluster is known to be gone (Ruling R54: no kubeconfig, or infra/base has no
        cluster outputs; gpu-down then terminates without draining, never touching whatever cluster the kubeconfig
        still reaches), or refuses (self._refusal) when it is not, or may not be, this project's (handled like an
        API server that does not answer: no drain, no node delete, no launch and no undrained terminate in this
        run). Neither when the IPs match. A check that raises refuses the rest of the run before the error goes on
        (review C0-2): a later node is never drained unchecked."""
        self._cluster_checked = True
        try:
            self._identify_cluster()
        except BaseException as e:
            if self._gone is None and self._refusal is None:
                self._not_this_cluster(f"the check of which cluster the kubeconfig reaches failed "
                                       f"({str(e) or type(e).__name__})", "fix that error")
            raise

    def _identify_cluster(self) -> None:
        """_check_cluster's checks, in order."""
        gone = self.infra.kubeconfig_missing()
        fix = "run make kubeconfig (make up writes it)"  # review S0-4: base may be fine
        base: Mapping = {}
        if gone is None:
            try:
                base = self._read_base()
            except (CommandError, ValueError, OSError) as e:  # not knowing is not "gone"
                self._not_this_cluster(
                    f"cannot read infra/base outputs ({e}), so nothing tells whether the cluster the kubeconfig "
                    "reaches is this project's",
                    "fix the read of infra/base's outputs (terraform -chdir=infra/base init; gcloud auth "
                    "application-default login if the state bucket refuses the credentials)")
                return
            if self._base_destroyed(base):
                gone = "infra/base has no cluster outputs (cp_ip, cp_internal_ip): the cluster was destroyed"
                fix = "apply the base stack first (make up)"
        if gone is not None:
            if self._command == "down":
                self._gone = gone
                self._warn(f"WARNING: {gone}. gpu-down terminates the gpuctl-owned instances without draining them, "
                           "so they stop billing, and drains nothing through the kubeconfig (Ruling R54)")
            else:
                self._refuse(gone, fix, f"{gone}, so gpuctl {self._command} cannot drain, delete or join a node: "
                             + self._how_to_go_on(fix))
            return
        cp_ip = base.get("cp_ip")
        if not cp_ip:  # cp_internal_ip without cp_ip: a broken base stack, not a destroyed one (review S0-2)
            self._not_this_cluster("infra/base has no cp_ip output to check the cluster against",
                                   "re-apply the base stack (make up)")
            return
        try:
            node_ip = self.infra.cp_external_ip()
        except CommandError as read_error:
            try:
                reason = self.infra.api_unreachable()
            except (CommandError, ValueError, OSError) as e:
                reason, read_error = None, f"{read_error}; the cluster could not be probed either: {e}"
            if reason is not None:
                self._unreachable(reason)
            else:
                self._not_this_cluster(f"cannot read Node k3s-cp's ExternalIP ({read_error})")
            return
        if node_ip != cp_ip:
            self._not_this_cluster(
                f"the cluster the kubeconfig reaches is not this project's: Node k3s-cp has ExternalIP "
                f"{node_ip or 'none'}, infra/base's cp_ip is {cp_ip}")

    def _launches_refused(self) -> list[str]:
        """Before step 5 (review S0-1/X0-1): a new node joins through the kubeconfig, and join-node.sh deletes a
        stale NotReady Node of its name there (Ruling R55), so nothing is launched unless the cluster the kubeconfig
        reaches is this project's. Read-only, and before any launch: a refused apply starts nothing that bills.
        The failure to report, or [] when the launches may go ahead."""
        if not self._cluster_checked:
            try:
                self._check_cluster()
            except (Exception, KeyboardInterrupt):
                pass  # _check_cluster refused the run, with the error in its message
        problem = self._problem if self._refusal is not None else self._gone
        if problem is None:
            return []
        return [f"launched nothing: {problem}; gpuctl {self._command} joins a new node only to this project's "
                "cluster (the join deletes a stale NotReady Node of that name there): "
                + self._how_to_go_on(self._fix or "fix that")]

    def _drain(self, name: str) -> bool:
        """Drain and delete the node of an instance about to be terminated: True when done, False when gpu-down goes
        on without it (--no-drain, or the cluster is known to be gone: Ruling R54). Raises when the drain failed in
        gpu-apply, or in a cluster that still exists (its API server refused the drain, or does not answer, as when
        `make tunnel` is not running: review F3-1), or when the cluster the kubeconfig reaches is not this
        project's (D-2): that instance is then not terminated."""
        if self._no_drain:
            self._warn(f"WARNING: --no-drain: terminating {name} without draining it or deleting its node")
            return False
        if not self._cluster_checked:
            self._check_cluster()
        if self._gone is not None:  # found out before the first drain, or on an earlier node: no second wait
            self._warn(f"WARNING: terminating {name} without draining it: {self._gone}")
            return False
        if self._refusal is not None:  # another cluster, or an earlier node found the API server silent
            raise CommandError(["kubectl", "drain", name], 1, f"not drained: {self._refusal}")
        try:
            self.infra.drain_and_delete_node(name)
            return True
        except CommandError as drain_error:
            if self._command != "down":
                raise
            gone = self._cluster_gone()
            if gone is not None:
                self._gone = gone
                self._warn(f"WARNING: cannot drain {name} ({str(drain_error)[-300:]}) and {gone}. gpu-down "
                           "terminates this gpuctl-owned instance without draining it, so it stops billing")
                return False
            try:
                reason = self.infra.api_unreachable()
            except (CommandError, ValueError, OSError) as e:
                raise CommandError(drain_error.args, drain_error.returncode,
                                   f"{drain_error.stderr} (and the cluster could not be probed: {e})") from None
            if reason is None:
                raise CommandError(drain_error.args, drain_error.returncode,
                                   f"{drain_error.stderr.strip()}; the API server answers, so gpu-down does not "
                                   "terminate a node it could not drain: fix the drain and re-run make gpu-down, "
                                   "or terminate it anyway with scripts/gpuctl down --no-drain") from None
            # Exactly what a destroyed cluster looks like to kubectl, but nothing says this one was destroyed (the
            # kubeconfig is there, infra/base was not found empty): terminating undrained would leave the Node object
            # and k3s's node-password secret for the next instance of this name (review F3-1, Ruling R54).
            self._unreachable(reason)
            raise CommandError(drain_error.args, drain_error.returncode,
                               f"{drain_error.stderr.strip()}; {self._refusal}") from None

    def _delete_node(self, name: str) -> None:
        """The cleanup's node delete (step 9), only in this project's cluster (D-2): raises CommandError instead
        when the cluster the kubeconfig reaches is gone, or is not (or may not be) this project's. The cleanup
        terminates the instance either way, so the refusal says how to remove a Node left behind (review C0-3)."""
        if not self._cluster_checked:
            self._check_cluster()
        problem = self._problem if self._refusal is not None else self._gone
        if problem is not None:
            raise CommandError(["kubectl", "delete", "node", name], 1,
                               f"not deleted: {problem}. A Node {name} in this project's cluster stays: once the "
                               "kubeconfig and make tunnel reach this project's k3s-cp, remove it with kubectl "
                               f"--kubeconfig ~/{DEFAULT_KUBECONFIG} delete node {name} (join-node.sh also deletes "
                               "it, NotReady, before the next node of that name joins)")
        self.infra.drain_and_delete_node(name)

    def _delete_failed(self, spec: Spec, gone: set[str], attempted: set[str], failures: list[str]) -> int:
        """A delete failed, or the launches were refused (_launches_refused): launch nothing, but drop the IPs of the
        instances that were terminated from every
        allow-list (step 7), since Lambda may hand those IPs to another tenant. This runs whenever a terminate was
        sent, even one that raised: step 7 reads the live instances, so one Lambda accepted is left out anyway."""
        for failure in failures:
            self._warn(f"error: {failure}")
        if attempted:
            try:
                self._reconcile_firewall(spec, gone)
            except (Exception, KeyboardInterrupt) as e:
                self._warn(f"error: firewall reconcile after the failed delete failed: {str(e) or type(e).__name__}; "
                           "re-run make gpu-apply or make sync-peers")
        self._warn("No new GPU nodes were launched; fix the error and re-run")
        return 1

    def _launch(self, pool: Pool, name: str, types: Mapping[str, InstanceType], rules: list[FirewallRule],
                failures: list[str], unknown: list[str]) -> _Launched | None:
        """Step 5 for one node: the first (type, region) with capacity, falling back on insufficient-capacity.

        While a launch request is in flight the name is in `unknown`, so if anything interrupts it (an API error,
        Ctrl-C, a bug) the cleanup looks the name up. It stays there when this returns an instance: the caller
        removes it once the instance is recorded in `launched`.
        """
        exclude: set[tuple[str, str]] = set()
        while True:
            choice = choose_placement(pool, types, frozenset(exclude))
            if choice is None:
                failures.append(f"{name}: no capacity for any candidate type in any region of pool {pool.name}")
                return None
            instance_type, region = choice
            try:
                ruleset = self.ensure_ruleset(region, rules)
            except LambdaAPIError as e:  # nothing was launched
                failures.append(f"{name}: cannot prepare the firewall ruleset {ruleset_name(region)}: {e}")
                return None
            unknown.append(name)
            try:
                instance_id = self.client.launch(name=name, region=region, instance_type=instance_type,
                                                 ssh_key_name=self.settings.ssh_key_name,
                                                 image_family=pool.image_family, ruleset_id=ruleset.id)
            except InsufficientCapacity:
                unknown.remove(name)
                self._emit(f"{name}: no capacity for {instance_type} in {region}, trying the next candidate")
                exclude.add(choice)
                continue
            except LambdaAPIError as e:
                # The outcome is unknown (Lambda may have started it): the name stays in `unknown`.
                failures.append(f"{name}: launch failed: {e}")
                return None
            self._emit(f"Launched {name} {instance_id} ({instance_type} in {region})")
            return _Launched(instance_id, name, instance_type)

    def _reconcile_firewall(self, spec: Spec, gone: Iterable[str] = ()) -> None:
        """Step 7: the gpu-access stack and every peer allow-list, for exactly the live GPU instances that have IPs.

        `gone` holds the ids this run terminated: they are left out even if Lambda still lists them as active.
        """
        gone = set(gone)
        ips = self._live_gpu_ips(gone)
        with self._step(f"terraform apply gpu-access ({len(ips)} GPU IP{'' if len(ips) == 1 else 's'})"):
            self.infra.apply_gpu_access(ips, max_nodes=spec.max_nodes)
        with self._step("sync-peers"):
            base = self._read_base()
            if self._command == "down" and self._base_destroyed(base):
                self._sync_lambda_rules_cluster_gone()
            else:
                self.sync_peers(spec, exclude_ids=gone, base=base)

    def _create_and_join(self, spec: Spec, plan: Plan, types: Mapping[str, InstanceType], *,
                         keep_on_failure: bool, gone: set[str]) -> int:
        launched: list[_Launched] = []
        unknown: list[str] = []  # names whose launch may or may not have started an instance
        joined: set[str] = set()
        failures: list[str] = []
        try:
            # 5. launch, each node into its region's ruleset (rules from the peers as they are after the deletes)
            if plan.create:
                rules = region_ruleset_rules(self._peer_set(gone))
                for name, pool_name in plan.create:
                    with self._step(f"launch {name}") as step:
                        new = self._launch(spec.pools[pool_name], name, types, rules, failures, unknown)
                        if new is None:
                            step.failed()
                    if new is not None:
                        launched.append(new)
                        unknown.remove(name)
            # 6. wait for each IP
            ready: list[tuple[_Launched, Instance]] = []
            for new in launched:
                try:
                    with self._step(f"wait for {new.name} to be active with an IP"):
                        ready.append((new, self.wait_active(new.id)))
                except (TimeoutError, LambdaAPIError) as e:
                    failures.append(f"{new.name}: {e}")
            # 7. firewalls for every live IP (also corrects them when nothing else changed)
            self._reconcile_firewall(spec, gone)
            # 8. join
            for new, instance in ready:
                try:
                    with self._step(f"join {new.name}"):
                        self.infra.join_node(new.name, instance.ip, gpu_type_label(new.instance_type),
                                             progress=self._join_line)
                    joined.add(new.id)
                except (CommandError, ValueError) as e:
                    failures.append(f"{new.name}: join failed: {e}")
        except (Exception, KeyboardInterrupt) as e:  # anything else (Ctrl-C included): clean up as for a failed node
            failures.append(f"apply failed: {str(e) or type(e).__name__}")
        now = self.clock()
        if not failures:
            self._emit(f"{self._stamp(now)} GPU nodes match the spec ({self._command} took "
                       f"{duration(now - self._started)})")
            return 0
        failed = [new for new in launched if new.id not in joined]
        try:  # the report never decides whether the billed instances are cleaned up (review C0-2)
            self._emit(f"{self._stamp(now)} {self._command} failed after {duration(now - self._started)}")
            for failure in failures:
                self._warn(f"error: {failure}")
            if keep_on_failure and (failed or unknown):
                names = ", ".join([new.name for new in failed] + unknown)
                self._warn(f"--keep-on-failure: kept {names} for troubleshooting; they bill until "
                           "make gpu-down (or a gpu-apply that no longer wants them)")
        finally:
            if not keep_on_failure:
                self._clean_up(spec, failed, unknown, gone)
        return 1

    def _find_unknown(self, unknown: list[str], known: set[str]) -> list[tuple[str, str]]:
        """(id, name) of the live owned instances named in `unknown`, other than the ids in `known`."""
        return [(i.id, i.name) for i in owned_instances(self.client.list_instances())
                if i.name in unknown and i.id not in known]

    def _clean_up(self, spec: Spec, failed: list[_Launched], unknown: list[str], gone: set[str]) -> None:
        """Step 9: terminate this run's failed new instances, drop their nodes and host keys, redo step 7.

        Best effort: one failing step does not stop the others, and each is reported. The instances whose ids are
        known go first, so nothing that follows (the wait for a lagging listing, a second Ctrl-C) can leak them;
        then the launches whose outcome is unknown (see _clean_up_unknown). A second Ctrl-C does not stop the
        cleanup either (review F2-4): every step still runs, and the KeyboardInterrupt is raised again at the end.
        """
        interrupted: KeyboardInterrupt | None = None
        try:
            self._terminate_victims([(new.id, new.name) for new in failed], gone)
        except KeyboardInterrupt as e:
            interrupted = e
        if unknown:
            try:
                self._clean_up_unknown(unknown, {new.id for new in failed}, gone)
            except KeyboardInterrupt as e:
                interrupted = interrupted or e
                self._warn(f"cleanup: interrupted while cleaning up {', '.join(unknown)}; run make gpu-status and "
                           "make gpu-down if one of them appears")
        try:
            self._reconcile_firewall(spec, gone)
        except Exception as e:
            self._warn(f"cleanup: firewall reconcile failed: {e}; re-run make gpu-apply or make sync-peers")
        except KeyboardInterrupt as e:
            interrupted = interrupted or e
            self._warn("cleanup: firewall reconcile interrupted; re-run make gpu-apply or make sync-peers")
        if interrupted is not None:
            raise interrupted

    def _clean_up_unknown(self, unknown: list[str], known: set[str], gone: set[str]) -> None:
        """Terminate the instances of launches whose outcome is unknown, looked up by name. Lambda may list an
        accepted launch only a moment later, so the names not found are looked up once more after poll_s, and
        any still missing is reported for the operator to check. A failed or interrupted (Ctrl-C) lookup or wait
        ends only this search: what was already found is terminated and the caller still redoes step 7."""
        try:
            found = self._find_unknown(unknown, known)
        except (Exception, KeyboardInterrupt) as e:
            self._warn(f"cleanup: cannot list instances to find {', '.join(unknown)}: {str(e) or type(e).__name__}; "
                       "check by hand (make gpu-status, and make gpu-down if one appears)")
            return
        self._terminate_victims(found, gone)
        missing = [name for name in unknown if name not in {n for _, n in found}]
        if not missing:
            return
        self._emit(f"cleanup: waiting {self.settings.poll_s:g}s for {', '.join(missing)} to appear")
        why = ""
        try:
            self.sleep(self.settings.poll_s)
            late = self._find_unknown(missing, known | {i for i, _ in found})
        except (Exception, KeyboardInterrupt) as e:
            late, why = [], f" (the second look was cut short: {str(e) or type(e).__name__})"
        self._terminate_victims(late, gone)
        for name in missing:
            if name not in {n for _, n in late}:
                self._warn(f"cleanup: launch outcome of {name} unknown and no instance found yet{why}; run "
                           "make gpu-status in a minute and make gpu-down if it appears")

    def _terminate_victims(self, victims: list[tuple[str, str]], gone: set[str]) -> None:
        """Terminate each (id, name), delete its node and forget its host key; terminated ids go into `gone`.

        A Ctrl-C (KeyboardInterrupt) does not stop it, since every victim bills until terminated (review F2-4): an
        interrupted terminate is sent once more, every other victim is still terminated, and the node deletes (a
        drain waits up to minutes) are skipped from then on. Then it says what was and was not terminated and
        raises the KeyboardInterrupt again.
        """
        interrupted: KeyboardInterrupt | None = None
        for instance_id, name in victims:
            for attempt in (1, 2):
                try:
                    self.client.terminate([instance_id])
                    gone.add(instance_id)
                    break
                except KeyboardInterrupt as e:
                    interrupted = e
                    self._warn(f"cleanup: Ctrl-C during the terminate of {name} ({instance_id}); "
                               + ("sending it again: billed instances are terminated before gpuctl stops"
                                  if attempt == 1 else "moving on"))
                except Exception as e:
                    self._warn(f"cleanup: terminate {name} ({instance_id}) failed: {e}")
                    break
            steps = [("forget host key", lambda: self.infra.forget_host_key(name))]
            if interrupted is None:
                steps.insert(0, ("delete node", lambda: self._delete_node(name)))
            for what, step in steps:
                try:
                    step()
                except Exception as e:
                    self._warn(f"cleanup: {what} {name} ({instance_id}) failed: {e}")
                except KeyboardInterrupt as e:
                    interrupted = e
                    self._warn(f"cleanup: Ctrl-C during the {what} of {name}; skipping the node deletes")
            self._emit(f"Cleaned up {name} {instance_id}")
        if interrupted is not None:
            done = [f"{name} {instance_id}" for instance_id, name in victims if instance_id in gone]
            left = [f"{name} {instance_id}" for instance_id, name in victims if instance_id not in gone]
            self._warn(f"cleanup interrupted (Ctrl-C): terminated {', '.join(done) or 'none'}"
                       + (f"; NOT terminated, still billing: {', '.join(left)}: run make gpu-down" if left else "")
                       + "; node objects may remain (make gpu-status)")
            raise interrupted

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
