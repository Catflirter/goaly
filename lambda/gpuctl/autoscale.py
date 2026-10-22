"""Autoscaler decision engine (stretch S2; design spec sections 10.2, 10.4, 10.5).

Pure functions only: the caller (scripts/autoscaler.py, every 20 s) passes in kubectl JSON, the stored State and
`now`, and executes the one Action that comes back. Nothing here reads a clock, a file or the network.

decide() checks its rules in priority order and returns only the first that fires:
  1. cooldown (COOLDOWN_S after the last action) -> nothing;
  2. a GPU Pod Pending for capacity, no GPU node launching -> +1 on the first autoscaled pool (by name) below max;
  3. a CPU/memory Pod Pending for capacity, cpu_count < cpu_max -> +1 CPU worker;
  4. a CPU worker other than k3s-cpu-1 idle for CPU_IDLE_S -> remove it (highest index first);
  5. an autoscaled pool above min whose highest-index node is idle for idle_minutes -> -1 on that pool.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from gpuctl.planner import node_name, parse_node_name
from gpuctl.spec import Pool

PENDING_GRACE_S = 30
CPU_IDLE_S = 300
COOLDOWN_S = 120

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


@dataclass(frozen=True)
class NodeObs:
    name: str
    kind: str  # "cpu" or "gpu"
    ready: bool
    labeled: bool  # GPU node: carries the admin label; always True for a CPU node
    busy: bool


@dataclass
class State:
    last_action_at: float | None = None
    idle_since: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"last_action_at": self.last_action_at, "idle_since": dict(sorted(self.idle_since.items()))})

    @classmethod
    def from_json(cls, text: str | bytes) -> State:
        """Parse a saved state. Missing or empty fields default; anything malformed raises ValueError.

        A corrupt file is an error, not a fresh state: silently resetting would drop the cooldown.
        """
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
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"autoscaler state: {where} must be a finite number, got {value!r}")
    return float(value)


def _items(doc: Mapping | None) -> list[Mapping]:
    return [i for i in (doc or {}).get("items") or [] if isinstance(i, Mapping)]


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


def node_observations(nodes_json: Mapping, pods_json: Mapping) -> list[NodeObs]:
    """Our worker nodes (k3s-cpu-*, k3s-gpu-*) in the order kubectl listed them.

    Busy ignores DaemonSet Pods (they run on every node) and finished Pods. A CPU node is busy with any Pod left; a
    GPU node only with a Pod that requests nvidia.com/gpu.
    """
    workloads: dict[str, list[Mapping]] = {}
    for pod in _items(pods_json):
        host = (pod.get("spec") or {}).get("nodeName")
        if not host or _owned_by_daemonset(pod) or (pod.get("status") or {}).get("phase") in _FINISHED_PHASES:
            continue
        workloads.setdefault(host, []).append(pod)

    observed = []
    for item in _items(nodes_json):
        meta = item.get("metadata") or {}
        name = str(meta.get("name", ""))
        if name.startswith(CPU_PREFIX):
            kind = "cpu"
        elif name.startswith(GPU_PREFIX):
            kind = "gpu"
        else:
            continue
        ready = any(c.get("type") == "Ready" and c.get("status") == "True"
                    for c in (item.get("status") or {}).get("conditions") or [])
        pods = workloads.get(name, [])
        if kind == "cpu":
            labeled, busy = True, bool(pods)
        else:
            key, value = GPU_ADMIN_LABEL
            labeled = (meta.get("labels") or {}).get(key) == value
            busy = any(_requests_gpu(p) for p in pods)
        observed.append(NodeObs(name=name, kind=kind, ready=ready, labeled=labeled, busy=busy))
    return observed


def _cpu_index(name: str) -> int | None:
    match = _CPU_NODE_RE.fullmatch(name)
    return int(match.group(1)) if match else None


def _gpu_launching(nodes: Iterable[NodeObs], counts: Mapping[str, int], pools: Mapping[str, Pool]) -> str | None:
    """Why a GPU node is still coming up (review focus #4), or None.

    Joined but NotReady, Ready without the admin label, or a pool with fewer registered nodes than its count (the
    instance runs but has not joined yet). The caller reports instances still booting via `gpu_launching`.
    """
    registered: dict[str, int] = {}
    for obs in nodes:
        if obs.kind != "gpu":
            continue
        if not obs.ready:
            return f"{obs.name} is NotReady"
        if not obs.labeled:
            return f"{obs.name} has no {GPU_ADMIN_LABEL[0]} label yet"
        parsed = parse_node_name(obs.name)
        if parsed:
            registered[parsed[0]] = registered.get(parsed[0], 0) + 1
    for name in sorted(pools):
        count = counts.get(name, pools[name].count)
        if registered.get(name, 0) < count:
            return f"pool {name} has {registered.get(name, 0)} of {count} nodes registered"
    return None


def _describe(pods: list[PendingPod]) -> str:
    shown = ", ".join(f"{p.namespace}/{p.name}" for p in pods[:3])
    return shown + (f" (+{len(pods) - 3} more)" if len(pods) > 3 else "")


def decide(*, now: float, pending: list[PendingPod], nodes: list[NodeObs], state: State, cpu_count: int,
           cpu_max: int, gpu_counts: dict[str, int], pools: dict[str, Pool],
           gpu_launching: bool) -> tuple[Action | None, State]:
    """At most one scaling action for this tick, and the state to save. The input state is not modified."""
    idle_since = {obs.name: state.idle_since.get(obs.name, now)
                  for obs in nodes
                  if obs.ready and not obs.busy and (obs.kind == "cpu" or obs.labeled)}
    new_state = State(last_action_at=state.last_action_at, idle_since=idle_since)

    action = _choose(now=now, pending=pending, nodes=nodes, state=new_state, cpu_count=cpu_count, cpu_max=cpu_max,
                     gpu_counts=gpu_counts, pools=pools, gpu_launching=gpu_launching)
    if action is None:
        return None, new_state
    return action, replace(new_state, last_action_at=now)


def _choose(*, now: float, pending: list[PendingPod], nodes: list[NodeObs], state: State, cpu_count: int,
            cpu_max: int, gpu_counts: Mapping[str, int], pools: Mapping[str, Pool],
            gpu_launching: bool) -> Action | None:
    if state.last_action_at is not None and now - state.last_action_at < COOLDOWN_S:
        return None

    def count(name: str) -> int:
        return gpu_counts.get(name, pools[name].count)

    def idle_for(name: str) -> float:
        return now - state.idle_since[name] if name in state.idle_since else -math.inf

    autoscaled = [name for name in sorted(pools) if pools[name].autoscale is not None]

    gpu_pending = [p for p in pending if p.resource == "gpu"]
    if gpu_pending and not gpu_launching and _gpu_launching(nodes, gpu_counts, pools) is None:
        for name in autoscaled:
            if count(name) < pools[name].autoscale.max:
                return Action("gpu_scale", target=count(name) + 1, pool=name,
                              reason=f"GPU capacity: {_describe(gpu_pending)} unschedulable")

    cpu_pending = [p for p in pending if p.resource in ("cpu", "memory")]
    if cpu_pending and cpu_count < cpu_max:
        return Action("cpu_scale", target=cpu_count + 1,
                      reason=f"CPU/memory capacity: {_describe(cpu_pending)} unschedulable")

    removable = [(index, obs.name) for obs in nodes
                 if obs.kind == "cpu" and obs.name != PROTECTED_CPU_NODE
                 and (index := _cpu_index(obs.name)) is not None and idle_for(obs.name) >= CPU_IDLE_S]
    if removable:
        _, victim = max(removable)
        return Action("cpu_remove", node=victim, reason=f"{victim} idle for {idle_for(victim):.0f}s")

    for name in autoscaled:
        bounds, current = pools[name].autoscale, count(name)
        if current <= bounds.min:
            continue
        highest = node_name(name, current)
        if idle_for(highest) >= bounds.idle_minutes * 60:
            return Action("gpu_scale", target=current - 1, pool=name,
                          reason=f"{highest} idle for {idle_for(highest):.0f}s")
    return None
