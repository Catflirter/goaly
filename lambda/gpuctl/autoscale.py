"""Autoscaler decision engine (stretch S2; design spec sections 10.2, 10.4, 10.5; Ruling SR6).

Pure functions only: the caller (scripts/autoscaler.py, every 20 s) passes in kubectl JSON, the stored State and
`now`, and executes the one Action that comes back. Nothing here reads a clock, a file or the network.

decide() checks the brief's rules in priority order and returns only the first that fires:
  0. refresh idle_since: a Ready node that is not busy keeps its clock (or starts one at now); others are dropped;
  1. cooldown (now - last_action_at < COOLDOWN_S, which includes a clock that is behind last_action_at) -> nothing;
  2. a GPU Pod Pending for capacity and no GPU node launching -> +1 on the first autoscaled pool (by name) below max;
  3. a CPU/memory Pod Pending for capacity and cpu_count < cpu_max -> +1 CPU worker;
  4. a CPU worker other than k3s-cpu-1 idle for CPU_IDLE_S -> remove it (highest index first);
  5. an autoscaled pool above min whose highest-index node is idle for idle_minutes -> -1 on that pool;
  6. an action sets last_action_at to now.

"Launching" (rule 2, Review Focus #4) is: the caller's `gpu_launching` (a Lambda instance booting, or running without a
registered node), or any k3s-gpu-* node that is NotReady, has no admin label yet, or is joining. Joining is the one
addition Ruling SR6 makes to the brief: a Ready, labeled GPU node with no allocatable nvidia.com/gpu (its device plugin
has not registered yet) whose Ready condition is younger than JOIN_GRACE_S. A joining node is also not idle (rule 0
gives it no clock), so rule 5 cannot scale it away while its plugin starts. Past JOIN_GRACE_S, or with no known Ready
time, the window is over: the node is broken (plugin or GPU failed), does not hold rule 2, gets an idle clock like any
other node, and the State's `held` names it whenever GPU Pods are Pending.

Known limitation (Ruling SR6): rules 2 and 3 do not ask whether a new node would fit the Pending Pod. A Pod that no
new node can satisfy (it asks for more than one node has) drives scale-up to autoscale.max / cpu_max, one step per
COOLDOWN_S, and rules 4/5 later remove the idle nodes again. Every decision is logged with its reason (S4), so the
operator sees it in autoscaler.log and stops it with Ctrl-C, make gpu-down or make cpu-scale N=1.

When rules 2 or 3 are withheld, the returned State's `held` says why (for autoscaler.log; it is not saved).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from gpuctl.planner import node_name
from gpuctl.spec import Pool

PENDING_GRACE_S = 30
CPU_IDLE_S = 300
COOLDOWN_S = 120
# Ruling SR6: how long after turning Ready a labeled GPU node may advertise no nvidia.com/gpu and still count as
# launching (its device plugin starts only after the admin label is applied).
JOIN_GRACE_S = 300

CPU_PREFIX = "k3s-cpu-"
GPU_PREFIX = "k3s-gpu-"
# Never removed (spec section 10.2): the pool always keeps its first worker.
PROTECTED_CPU_NODE = "k3s-cpu-1"
GPU_RESOURCE = "nvidia.com/gpu"
# Applied by join-node.sh step 7 once a GPU node is ready for GPU work; a GPU node without it is still joining.
GPU_ADMIN_LABEL = ("node-restriction.kubernetes.io/accelerator", "nvidia")

_CPU_NODE_RE = re.compile(rf"{re.escape(CPU_PREFIX)}([1-9][0-9]*)")
# One FailedScheduling message can mix capacity and non-capacity reasons ("1 Insufficient nvidia.com/gpu, 1 node(s)
# didn't match Pod's node affinity/selector, ..."); any "Insufficient <resource>" makes it capacity-related.
# Checked in this order: a node that adds GPUs also adds CPU and memory, never the other way round.
_INSUFFICIENT = (
    ("gpu", re.compile(r"\bInsufficient nvidia\.com/gpu(?![\w/-]|\.\w)")),
    ("cpu", re.compile(r"\bInsufficient cpu(?![\w/-]|\.\w)")),
    ("memory", re.compile(r"\bInsufficient memory(?![\w/-]|\.\w)")),
)
_FINISHED_PHASES = frozenset({"Succeeded", "Failed"})


@dataclass(frozen=True)
class PendingPod:
    namespace: str
    name: str
    resource: str  # "cpu", "memory" or "gpu"
    since: float  # epoch seconds the Pod became unschedulable

    @property
    def key(self) -> str:
        return f"{self.namespace}/{self.name}"


@dataclass(frozen=True)
class NodeObs:
    name: str
    kind: str  # "cpu" or "gpu"
    ready: bool
    labeled: bool  # GPU node: carries the admin label; always True for a CPU node
    busy: bool
    gpu_allocatable: int = 0  # status.allocatable["nvidia.com/gpu"]; 0 when absent
    ready_since: float | None = None  # the Ready condition's lastTransitionTime while Ready; None otherwise


@dataclass
class State:
    last_action_at: float | None = None
    idle_since: dict[str, float] = field(default_factory=dict)
    # Not saved: why this tick withheld a scale-up that Pending Pods asked for, for autoscaler.log; None if nothing was.
    held: str | None = field(default=None, compare=False)

    def to_json(self) -> str:
        return json.dumps({"last_action_at": self.last_action_at, "idle_since": dict(sorted(self.idle_since.items()))})

    @classmethod
    def from_json(cls, text: str | bytes) -> State:
        """Parse a saved state. Missing, null or empty fields default and unknown keys are ignored; a malformed value
        raises ValueError, because silently starting fresh would drop the cooldown (Review Focus #1)."""
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        if not text.strip():
            return cls()
        data = json.loads(text)  # JSONDecodeError is a ValueError
        if not isinstance(data, dict):
            raise ValueError(f"autoscaler state: expected a JSON object, got {type(data).__name__}")
        last = data.get("last_action_at")
        idle = data.get("idle_since")
        idle = {} if idle is None else idle
        if not isinstance(idle, dict):
            raise ValueError(f"autoscaler state: idle_since must be an object, got {idle!r}")
        return cls(
            last_action_at=None if last is None else _seconds(last, "last_action_at"),
            idle_since={str(k): _seconds(v, f"idle_since[{k!r}]") for k, v in idle.items()},
        )


@dataclass(frozen=True)
class Action:
    kind: str  # "cpu_scale", "cpu_remove" or "gpu_scale"
    target: int | None = None
    node: str | None = None
    pool: str | None = None
    reason: str = ""


def _seconds(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"autoscaler state: {where} must be a finite number, got {value!r}")
    try:
        seconds = float(value)
    except OverflowError:
        raise ValueError(f"autoscaler state: {where} is out of range") from None
    if not math.isfinite(seconds):
        raise ValueError(f"autoscaler state: {where} must be a finite number, got {value!r}")
    return seconds


def _items(doc: Mapping | None) -> list[Mapping]:
    return [i for i in (doc or {}).get("items") or [] if isinstance(i, Mapping)]


def _listing(doc: Any, what: str) -> list[Mapping]:
    """The items of a kubectl list, or ValueError: an error or partial reply must never read as an empty cluster."""
    entries = doc.get("items") if isinstance(doc, Mapping) else None
    if not isinstance(entries, list) or not all(isinstance(i, Mapping) for i in entries):
        raise ValueError(f"{what}: not a kubectl list with an items array ({str(doc)[:120]!r})")
    return entries


def _epoch(stamp: Any) -> float | None:
    """RFC 3339 (as the API server writes lastTransitionTime) to epoch seconds; None if absent or unparseable."""
    if not isinstance(stamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _capacity_resource(message: str) -> str | None:
    for resource, pattern in _INSUFFICIENT:
        if pattern.search(message):
            return resource
    return None


def pending_pods(pods_json: Mapping, now: float) -> list[PendingPod]:
    """Pods unschedulable for lack of cpu, memory or GPUs for at least PENDING_GRACE_S.

    Anything else that keeps a Pod Pending (affinity, taints alone, an unbound PVC, an image pull) never counts.
    A missing listing reads as nothing Pending, which only ever withholds a scale-up.
    """
    found = []
    for pod in _items(pods_json):
        status = pod.get("status") or {}
        if status.get("phase") != "Pending":
            continue
        for cond in status.get("conditions") or []:
            if not (cond.get("type") == "PodScheduled" and cond.get("status") == "False"
                    and cond.get("reason") == "Unschedulable"):
                continue
            resource = _capacity_resource(str(cond.get("message") or ""))
            since = _epoch(cond.get("lastTransitionTime"))
            if resource is None or since is None or now - since < PENDING_GRACE_S:
                continue
            meta = pod.get("metadata") or {}
            found.append(PendingPod(namespace=str(meta.get("namespace", "")), name=str(meta.get("name", "")),
                                    resource=resource, since=since))
            break
    return found


def _owned_by_daemonset(pod: Mapping) -> bool:
    return any(ref.get("kind") == "DaemonSet" for ref in (pod.get("metadata") or {}).get("ownerReferences") or [])


def _requests_gpu(pod: Mapping) -> bool:
    spec = pod.get("spec") or {}
    for container in [*(spec.get("containers") or []), *(spec.get("initContainers") or [])]:
        resources = container.get("resources") or {}
        for section in ("requests", "limits"):
            quantity = str((resources.get(section) or {}).get(GPU_RESOURCE, "0")).strip()
            if quantity not in ("", "0"):
                return True
    return False


def _allocatable_gpus(node: Mapping) -> int:
    """status.allocatable["nvidia.com/gpu"] (an integer quantity); 0 when absent or unreadable."""
    quantity = ((node.get("status") or {}).get("allocatable") or {}).get(GPU_RESOURCE)
    try:
        return max(int(str(quantity).strip()), 0)
    except ValueError:
        return 0


def node_observations(nodes_json: Mapping, pods_json: Mapping) -> list[NodeObs]:
    """Our worker nodes (k3s-cpu-*, k3s-gpu-*) in the order kubectl listed them.

    Busy ignores DaemonSet Pods (they run on every node) and finished Pods. A CPU node is busy with any Pod left; a
    GPU node only with a Pod that requests nvidia.com/gpu. gpu_allocatable reads status.allocatable["nvidia.com/gpu"]
    and ready_since the Ready condition's lastTransitionTime (while Ready). Raises ValueError unless both arguments
    are kubectl lists (`items` an array, possibly empty): busyness guards every scale-down, so the caller must skip
    the tick rather than let a failed read look like a cluster where nothing runs.
    """
    node_items, pod_items = _listing(nodes_json, "nodes"), _listing(pods_json, "pods")
    workloads: dict[str, list[Mapping]] = {}
    for pod in pod_items:
        host = (pod.get("spec") or {}).get("nodeName")
        if not host or _owned_by_daemonset(pod) or (pod.get("status") or {}).get("phase") in _FINISHED_PHASES:
            continue
        workloads.setdefault(host, []).append(pod)

    observed = []
    for item in node_items:
        meta = item.get("metadata") or {}
        name = str(meta.get("name", ""))
        if name.startswith(CPU_PREFIX):
            kind = "cpu"
        elif name.startswith(GPU_PREFIX):
            kind = "gpu"
        else:
            continue
        ready_cond = next((c for c in (item.get("status") or {}).get("conditions") or []
                           if isinstance(c, Mapping) and c.get("type") == "Ready"), {})
        ready = ready_cond.get("status") == "True"
        pods = workloads.get(name, [])
        if kind == "cpu":
            labeled, busy = True, bool(pods)
        else:
            key, value = GPU_ADMIN_LABEL
            labeled = (meta.get("labels") or {}).get(key) == value
            busy = any(_requests_gpu(p) for p in pods)
        observed.append(NodeObs(name=name, kind=kind, ready=ready, labeled=labeled, busy=busy,
                                gpu_allocatable=_allocatable_gpus(item),
                                ready_since=_epoch(ready_cond.get("lastTransitionTime")) if ready else None))
    return observed


def _cpu_index(name: str) -> int | None:
    match = _CPU_NODE_RE.fullmatch(name)
    return int(match.group(1)) if match else None


def _without_gpus(obs: NodeObs) -> bool:
    """A Ready, labeled GPU node that advertises no nvidia.com/gpu: joining, or broken past the window."""
    return obs.kind == "gpu" and obs.ready and obs.labeled and obs.gpu_allocatable == 0


def _joining(obs: NodeObs, now: float) -> bool:
    """Ruling SR6's joining window: without GPUs, and Ready for less than JOIN_GRACE_S."""
    return _without_gpus(obs) and obs.ready_since is not None and now - obs.ready_since < JOIN_GRACE_S


def _gpu_launching(now: float, nodes: Iterable[NodeObs]) -> str | None:
    """Why a GPU node is still coming up (Review Focus #4), or None."""
    for obs in nodes:
        if obs.kind != "gpu":
            continue
        if not obs.ready:
            return f"{obs.name} is NotReady"
        if not obs.labeled:
            return f"{obs.name} has no {GPU_ADMIN_LABEL[0]} label yet"
        if _joining(obs, now):
            return (f"{obs.name} is joining: Ready {now - obs.ready_since:.0f}s, no allocatable {GPU_RESOURCE} yet "
                    f"(window {JOIN_GRACE_S}s)")
    return None


def _broken(now: float, nodes: Iterable[NodeObs]) -> list[str]:
    """Ready, labeled GPU nodes without GPUs past the joining window: the device plugin or the GPU has failed."""
    found = []
    for obs in nodes:
        if _without_gpus(obs) and not _joining(obs, now):
            age = "an unknown time" if obs.ready_since is None else f"{now - obs.ready_since:.0f}s"
            found.append(f"{obs.name} is broken, not launching: no allocatable {GPU_RESOURCE} {age} after Ready "
                         "(device plugin or GPU failed?)")
    return found


def _describe(pods: list[PendingPod]) -> str:
    shown = ", ".join(p.key for p in pods[:3])
    return shown + (f" (+{len(pods) - 3} more)" if len(pods) > 3 else "")


def decide(*, now: float, pending: list[PendingPod], nodes: list[NodeObs], state: State, cpu_count: int,
           cpu_max: int, gpu_counts: dict[str, int], pools: dict[str, Pool],
           gpu_launching: bool) -> tuple[Action | None, State]:
    """At most one scaling action for this tick, and the state to save. The input state is not modified."""
    # Rule 0. A joining node is not idle (Ruling SR6): its clock starts once it has GPUs or its window is over.
    idle_since = {obs.name: state.idle_since.get(obs.name, now) for obs in nodes
                  if obs.ready and not obs.busy and not _joining(obs, now)}
    new_state = State(last_action_at=state.last_action_at, idle_since=idle_since)
    action, held = _choose(now=now, pending=pending, nodes=nodes, state=new_state, cpu_count=cpu_count,
                           cpu_max=cpu_max, gpu_counts=gpu_counts, pools=pools, gpu_launching=gpu_launching)
    new_state.held = "; ".join(held) or None
    if action is None:
        return None, new_state
    return action, replace(new_state, last_action_at=now)  # rule 6


def _choose(*, now: float, pending: list[PendingPod], nodes: list[NodeObs], state: State, cpu_count: int,
            cpu_max: int, gpu_counts: Mapping[str, int], pools: Mapping[str, Pool],
            gpu_launching: bool) -> tuple[Action | None, list[str]]:
    """Rules 1-5: the first that fires, and why scale-ups that Pending Pods asked for were withheld."""
    held = []
    gpu_pending = [p for p in pending if p.resource == "gpu"]
    cpu_pending = [p for p in pending if p.resource != "gpu"]
    if gpu_pending:
        held.extend(_broken(now, nodes))

    last = state.last_action_at
    if last is not None and now - last < COOLDOWN_S:  # rule 1; also a clock that is behind last_action_at
        if pending:
            wait = (f"cooldown: {COOLDOWN_S - (now - last):.0f}s left" if now >= last else
                    f"cooldown: the clock is {last - now:.0f}s behind the last action; "
                    f"{COOLDOWN_S + last - now:.0f}s left")
            held.append(f"{_describe(gpu_pending + cpu_pending)} waiting: {wait}")
        return None, held

    def count(name: str) -> int:
        return gpu_counts.get(name, pools[name].count)

    def idle_for(name: str) -> float:
        return now - state.idle_since[name] if name in state.idle_since else -math.inf

    autoscaled = [name for name in sorted(pools) if pools[name].autoscale is not None]

    if gpu_pending:  # rule 2
        why = "a GPU instance is still booting" if gpu_launching else _gpu_launching(now, nodes)
        if why is None:
            for name in autoscaled:
                if count(name) < pools[name].autoscale.max:
                    return Action("gpu_scale", target=count(name) + 1, pool=name,
                                  reason=f"GPU capacity: {_describe(gpu_pending)} unschedulable"), held
            why = "every autoscaled GPU pool is at its max" if autoscaled else "no GPU pool has autoscale"
        held.append(f"{_describe(gpu_pending)} waiting: {why}")

    if cpu_pending:  # rule 3
        if cpu_count < cpu_max:
            return Action("cpu_scale", target=cpu_count + 1,
                          reason=f"CPU/memory capacity: {_describe(cpu_pending)} unschedulable"), held
        held.append(f"{_describe(cpu_pending)} waiting: CPU pool at cpu_max {cpu_max}")

    removable = [(index, obs.name) for obs in nodes  # rule 4
                 if obs.kind == "cpu" and obs.name != PROTECTED_CPU_NODE
                 and (index := _cpu_index(obs.name)) is not None and idle_for(obs.name) >= CPU_IDLE_S]
    if removable:
        _, victim = max(removable)
        return Action("cpu_remove", node=victim, reason=f"{victim} idle for {idle_for(victim):.0f}s"), held

    for name in autoscaled:  # rule 5
        bounds, current = pools[name].autoscale, count(name)
        if current <= bounds.min:
            continue
        highest = node_name(name, current)
        if idle_for(highest) >= bounds.idle_minutes * 60:
            return Action("gpu_scale", target=current - 1, pool=name,
                          reason=f"{highest} idle for {idle_for(highest):.0f}s"), held
    return None, held
