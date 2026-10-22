"""Autoscaler decision engine (stretch S2; design spec sections 10.2, 10.4, 10.5).

Pure functions only: the caller (scripts/autoscaler.py, every 20 s) passes in kubectl JSON, the stored State and
`now`, and executes the one Action that comes back. Nothing here reads a clock, a file or the network.

decide() first refreshes the State (idle clocks, stuck Pods, GPU launches still joining), then checks its rules in
priority order and returns only the first that fires:
  1. cooldown (COOLDOWN_S after the last action) -> nothing;
  2. a GPU Pod Pending for capacity, no GPU node launching -> +1 on the first autoscaled pool (by name) below max;
  3. a CPU/memory Pod Pending for capacity, cpu_count < cpu_max -> +1 CPU worker;
  4. a CPU worker other than k3s-cpu-1 idle for CPU_IDLE_S -> remove it (highest index first);
  5. an autoscaled pool above min whose highest-index node is idle for idle_minutes -> -1 on that pool;
  6. an autoscaled pool whose last scale-up node is not usable within GPU_JOIN_DEADLINE_S -> -1 on that pool.
Rules 2 and 3 ignore a "stuck" Pod for STUCK_TTL_S: one still Pending while a node that could take it, of the kind
a scale-up would add, sat idle beside it for STUCK_IDLE_S (it does not fit such a node, so another one would not help),
or one whose GPU launch never joined. Without that, such a Pod would add and remove a node, a paid GPU instance
included, every few minutes for as long as it stays Pending; with it, at most once per STUCK_TTL_S.
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

from gpuctl.planner import node_name, parse_node_name
from gpuctl.spec import Pool

PENDING_GRACE_S = 30
CPU_IDLE_S = 300
COOLDOWN_S = 120
# A capacity-Pending Pod next to a usable node of its kind idle this long does not fit that node. Longer than a
# scheduler retry cycle, so a node that has just become usable gets its Pods first.
STUCK_IDLE_S = 60
# A stuck mark lapses after this long, so a Pod gets another chance (say, Lambda has stock again).
STUCK_TTL_S = 3600
# A GPU scale-up whose node cannot take GPU Pods after this long is rolled back (no stock, a failed join).
GPU_JOIN_DEADLINE_S = 1800
# A saved timestamp further than this ahead of `now` means the clock moved back or the state file is wrong.
CLOCK_SKEW_S = 60

CPU_PREFIX = "k3s-cpu-"
GPU_PREFIX = "k3s-gpu-"
# Never removed (spec section 10.2): the pool always keeps its first worker.
PROTECTED_CPU_NODE = "k3s-cpu-1"
GPU_RESOURCE = "nvidia.com/gpu"
# Applied by join-node.sh step 7 once a GPU node is ready for GPU work; a GPU node without it is still joining.
GPU_ADMIN_LABEL = ("node-restriction.kubernetes.io/accelerator", "nvidia")
# node-bootstrap.sh taints every GPU node nvidia.com/gpu=present:NoSchedule; GPU Pods tolerate it (RuntimeClass).
GPU_TAINT_KEY = GPU_RESOURCE
_BLOCKING_TAINT_EFFECTS = frozenset({"NoSchedule", "NoExecute"})

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

    @property
    def kind(self) -> str:
        """The node kind that would serve it: "gpu" or "cpu" (cpu and memory)."""
        return "gpu" if self.resource == "gpu" else "cpu"


@dataclass(frozen=True)
class NodeObs:
    name: str
    kind: str  # "cpu" or "gpu"
    ready: bool
    labeled: bool  # GPU node: carries the admin label; always True for a CPU node
    busy: bool
    # Not cordoned and no NoSchedule/NoExecute taint, other than the GPU taint that GPU Pods tolerate.
    schedulable: bool = True
    # GPU node: advertises nvidia.com/gpu (its device plugin runs); always True for a CPU node.
    gpu_allocatable: bool = True

    @property
    def usable(self) -> bool:
        """Can take a new Pod of its kind now."""
        return self.ready and self.labeled and self.schedulable and self.gpu_allocatable


@dataclass(frozen=True)
class Mark:
    """Why a Pending Pod is stuck, and when it was marked (the mark lapses STUCK_TTL_S later)."""
    why: str
    at: float


@dataclass(frozen=True)
class Launch:
    """A GPU scale-up whose node cannot take GPU Pods yet."""
    target: int  # the pool count it set
    at: float  # epoch seconds of the action
    pods: tuple[str, ...] = ()  # the Pending Pods ("namespace/name") it was for


@dataclass
class State:
    last_action_at: float | None = None
    idle_since: dict[str, float] = field(default_factory=dict)
    # Pending Pods ("namespace/name") a new node would not help. No scale-up for them while they stay Pending, for at
    # most STUCK_TTL_S; the mark also goes when the Pod is no longer Pending.
    stuck: dict[str, Mark] = field(default_factory=dict)
    # GPU scale-ups whose node is not usable yet, by pool.
    launches: dict[str, Launch] = field(default_factory=dict)
    # Like idle_since, for nodes that are also usable (NodeObs.usable): the evidence clock for stuck Pods.
    usable_idle_since: dict[str, float] = field(default_factory=dict)
    # Not saved: why this tick withheld a scale-up that Pending Pods asked for, for autoscaler.log; None if nothing was.
    held: str | None = field(default=None, compare=False)

    def to_json(self) -> str:
        return json.dumps({
            "last_action_at": self.last_action_at,
            "idle_since": dict(sorted(self.idle_since.items())),
            "stuck": {key: {"why": m.why, "at": m.at} for key, m in sorted(self.stuck.items())},
            "launches": {pool: {"target": launch.target, "at": launch.at, "pods": list(launch.pods)}
                         for pool, launch in sorted(self.launches.items())},
            "usable_idle_since": dict(sorted(self.usable_idle_since.items())),
        })

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
        idle, stuck, launches, usable = (_object(data, key)
                                         for key in ("idle_since", "stuck", "launches", "usable_idle_since"))
        return cls(
            last_action_at=None if last is None else _seconds(last, "last_action_at"),
            idle_since={str(k): _seconds(v, f"idle_since[{k!r}]") for k, v in idle.items()},
            stuck={str(k): _mark(v, f"stuck[{k!r}]") for k, v in stuck.items()},
            launches={str(k): _launch(v, f"launches[{k!r}]") for k, v in launches.items()},
            usable_idle_since={str(k): _seconds(v, f"usable_idle_since[{k!r}]") for k, v in usable.items()},
        )


@dataclass(frozen=True)
class Action:
    kind: str  # "cpu_scale", "cpu_remove" or "gpu_scale"
    target: int | None = None
    node: str | None = None
    pool: str | None = None
    reason: str = ""


def _object(data: Mapping, key: str) -> dict:
    value = data.get(key)
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise ValueError(f"autoscaler state: {key} must be an object, got {value!r}")
    return value


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


def _mark(value: Any, where: str) -> Mark:
    why = value.get("why") if isinstance(value, dict) else None
    if not isinstance(why, str):
        raise ValueError(f"autoscaler state: {where} must be {{\"why\": <text>, \"at\": <seconds>}}, got {value!r}")
    return Mark(why, _seconds(value.get("at"), f"{where}.at"))


def _launch(value: Any, where: str) -> Launch:
    target = value.get("target") if isinstance(value, dict) else None
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError(f"autoscaler state: {where} must be {{\"target\": <count>, \"at\": <seconds>}}, got {value!r}")
    pods = value.get("pods")
    pods = [] if pods is None else pods
    if not isinstance(pods, list) or not all(isinstance(p, str) for p in pods):
        raise ValueError(f"autoscaler state: {where}.pods must be a list of Pod keys, got {pods!r}")
    return Launch(target, _seconds(value.get("at"), f"{where}.at"), tuple(pods))


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


def _schedulable(node: Mapping, kind: str) -> bool:
    spec = node.get("spec") or {}
    if spec.get("unschedulable"):
        return False
    for taint in spec.get("taints") or []:
        if not isinstance(taint, Mapping) or taint.get("effect") not in _BLOCKING_TAINT_EFFECTS:
            continue
        if not (kind == "gpu" and taint.get("key") == GPU_TAINT_KEY):
            return False
    return True


def _advertises_gpus(node: Mapping) -> bool:
    quantity = str(((node.get("status") or {}).get("allocatable") or {}).get(GPU_RESOURCE, "0")).strip()
    return quantity not in ("", "0")


def node_observations(nodes_json: Mapping, pods_json: Mapping) -> list[NodeObs]:
    """Our worker nodes (k3s-cpu-*, k3s-gpu-*) in the order kubectl listed them.

    Busy ignores DaemonSet Pods (they run on every node) and finished Pods. A CPU node is busy with any Pod left; a
    GPU node only with a Pod that requests nvidia.com/gpu. Schedulable reads spec.unschedulable (cordon, drain) and
    NoSchedule/NoExecute taints (the GPU taint excepted on a GPU node); gpu_allocatable reads
    status.allocatable["nvidia.com/gpu"]. Raises ValueError unless both arguments are kubectl lists (`items` an
    array, possibly empty): busyness guards every scale-down, so a failed read must stop the tick.
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
        ready = any(c.get("type") == "Ready" and c.get("status") == "True"
                    for c in (item.get("status") or {}).get("conditions") or [])
        pods = workloads.get(name, [])
        if kind == "cpu":
            labeled, busy, gpus = True, bool(pods), True
        else:
            key, value = GPU_ADMIN_LABEL
            labeled = (meta.get("labels") or {}).get(key) == value
            busy = any(_requests_gpu(p) for p in pods)
            gpus = _advertises_gpus(item)
        observed.append(NodeObs(name=name, kind=kind, ready=ready, labeled=labeled, busy=busy,
                                schedulable=_schedulable(item, kind), gpu_allocatable=gpus))
    return observed


def _cpu_index(name: str) -> int | None:
    match = _CPU_NODE_RE.fullmatch(name)
    return int(match.group(1)) if match else None


def _gpu_launching(now: float, nodes: Iterable[NodeObs], counts: Mapping[str, int], pools: Mapping[str, Pool],
                   usable_idle_since: Mapping[str, float]) -> str | None:
    """Why a GPU node is still coming up (review focus #4), or None.

    Joined but NotReady, Ready without the admin label, labeled but no GPUs advertised yet (the device plugin starts
    only after labeling), usable and idle for less than STUCK_IDLE_S (the scheduler places its Pods first), or an
    autoscaled pool with fewer registered nodes than its count (the instance runs but has not joined yet). Pools
    without autoscale are managed by hand and never block. The caller reports booting instances via `gpu_launching`.
    """
    registered: dict[str, int] = {}
    for obs in nodes:
        if obs.kind != "gpu":
            continue
        if not obs.ready:
            return f"{obs.name} is NotReady"
        if not obs.labeled:
            return f"{obs.name} has no {GPU_ADMIN_LABEL[0]} label yet"
        if not obs.gpu_allocatable:
            return f"{obs.name} has no allocatable {GPU_RESOURCE} yet"
        if obs.name in usable_idle_since and now - usable_idle_since[obs.name] < STUCK_IDLE_S:
            return (f"{obs.name} can take GPU Pods since {now - usable_idle_since[obs.name]:.0f}s; "
                    "waiting for the scheduler")
        parsed = parse_node_name(obs.name)
        if parsed:
            registered[parsed[0]] = registered.get(parsed[0], 0) + 1
    for name in sorted(pools):
        if pools[name].autoscale is None:
            continue
        count = counts.get(name, pools[name].count)
        if registered.get(name, 0) < count:
            return f"pool {name} has {registered.get(name, 0)} of {count} nodes registered"
    return None


def _describe(pods: list[PendingPod]) -> str:
    shown = ", ".join(p.key for p in pods[:3])
    return shown + (f" (+{len(pods) - 3} more)" if len(pods) > 3 else "")


def _check_clock(state: State, now: float) -> None:
    stamps = [("last_action_at", state.last_action_at),
              *((f"idle_since[{k!r}]", v) for k, v in state.idle_since.items()),
              *((f"usable_idle_since[{k!r}]", v) for k, v in state.usable_idle_since.items()),
              *((f"stuck[{k!r}]", m.at) for k, m in state.stuck.items()),
              *((f"launches[{k!r}]", launch.at) for k, launch in state.launches.items())]
    for where, stamp in stamps:
        if stamp is not None and stamp > now + CLOCK_SKEW_S:
            raise ValueError(f"autoscaler state: {where} is {stamp - now:.0f}s in the future "
                             "(the clock moved back, or the state file is wrong)")


def _autoscaled(pools: Mapping[str, Pool]) -> list[str]:
    return [name for name in sorted(pools) if pools[name].autoscale is not None]


def _gpu_evidence_pools(gpu_counts: Mapping[str, int], pools: Mapping[str, Pool]) -> frozenset[str]:
    """The pools whose idle node shows a GPU Pod would not fit a new node: the one rule 2 adds to (the first
    autoscaled pool with room), or, with every autoscaled pool at max, all of them (the next scale-up goes to
    whichever shrinks first). A fixed pool, or one rule 2 does not reach, says nothing about the node it would add."""
    autoscaled = _autoscaled(pools)
    for name in autoscaled:
        if gpu_counts.get(name, pools[name].count) < pools[name].autoscale.max:
            return frozenset({name})
    return frozenset(autoscaled)


def _stuck(now: float, pending: list[PendingPod], nodes: list[NodeObs], usable_idle_since: Mapping[str, float],
           previous: Mapping[str, Mark], gpu_pools: frozenset[str]) -> dict[str, Mark]:
    """Carry the unexpired marks of Pods still Pending; mark a Pending Pod that has been Pending for STUCK_IDLE_S
    alongside a usable idle node of the kind a scale-up would add (a GPU node only from `gpu_pools`)."""
    keys = {p.key for p in pending}
    stuck = {key: mark for key, mark in previous.items() if key in keys and now - mark.at < STUCK_TTL_S}
    evidence = sorted(obs.name for obs in nodes if obs.name in usable_idle_since
                      and (obs.kind == "cpu" or (parse_node_name(obs.name) or ("",))[0] in gpu_pools))
    kinds = {obs.name: obs.kind for obs in nodes}
    for p in pending:
        if p.key in stuck:
            continue
        for name in evidence:
            if kinds[name] == p.kind and now - max(usable_idle_since[name], p.since) >= STUCK_IDLE_S:
                stuck[p.key] = Mark(f"{name} was idle", now)
                break
    return stuck


def decide(*, now: float, pending: list[PendingPod], nodes: list[NodeObs], state: State, cpu_count: int,
           cpu_max: int, gpu_counts: dict[str, int], pools: dict[str, Pool],
           gpu_launching: bool) -> tuple[Action | None, State]:
    """At most one scaling action for this tick, and the state to save. The input state is not modified.

    Raises ValueError if a saved timestamp lies more than CLOCK_SKEW_S after `now`.
    """
    _check_clock(state, now)
    idle_since = {obs.name: state.idle_since.get(obs.name, now) for obs in nodes if obs.ready and not obs.busy}
    usable_idle_since = {obs.name: state.usable_idle_since.get(obs.name, now) for obs in nodes
                         if obs.usable and not obs.busy}
    joined = {obs.name for obs in nodes if obs.kind == "gpu" and obs.usable}
    launches = {pool: launch for pool, launch in state.launches.items()
                if pool in pools and gpu_counts.get(pool, pools[pool].count) == launch.target
                and node_name(pool, launch.target) not in joined}
    stuck = _stuck(now, pending, nodes, usable_idle_since, state.stuck, _gpu_evidence_pools(gpu_counts, pools))
    new_state = State(last_action_at=state.last_action_at, idle_since=idle_since, stuck=stuck, launches=launches,
                      usable_idle_since=usable_idle_since)

    action, held = _choose(now=now, pending=pending, nodes=nodes, state=new_state, cpu_count=cpu_count,
                           cpu_max=cpu_max, gpu_counts=gpu_counts, pools=pools, gpu_launching=gpu_launching)
    new_state.held = "; ".join(held) or None
    if action is None:
        return None, new_state
    if action.kind == "gpu_scale" and action.pool is not None and action.target is not None:
        launched = launches.pop(action.pool, None)
        if launched and action.target < launched.target:
            # Scaling away a node that never became usable: the Pods it was for must not launch another (for now).
            failed = node_name(action.pool, launched.target)
            for key in sorted(set(launched.pods) & {p.key for p in pending}):
                stuck.setdefault(key, Mark(f"{failed} was launched for it and never joined", now))
        elif action.target > gpu_counts.get(action.pool, pools[action.pool].count):
            served = tuple(sorted(p.key for p in pending if p.kind == "gpu" and p.key not in stuck))
            launches[action.pool] = Launch(action.target, now, served)
    return action, replace(new_state, last_action_at=now)


def _choose(*, now: float, pending: list[PendingPod], nodes: list[NodeObs], state: State, cpu_count: int,
            cpu_max: int, gpu_counts: Mapping[str, int], pools: Mapping[str, Pool],
            gpu_launching: bool) -> tuple[Action | None, list[str]]:
    """The first rule that fires, and why scale-ups that Pending Pods asked for were withheld."""
    held = []
    stuck = [p for p in pending if p.key in state.stuck]
    if stuck:
        held.append("no new node for " + ", ".join(f"{p.key} ({state.stuck[p.key].why})" for p in stuck[:3])
                    + (f" (+{len(stuck) - 3} more)" if len(stuck) > 3 else ""))
    gpu_pending = [p for p in pending if p.kind == "gpu" and p.key not in state.stuck]
    cpu_pending = [p for p in pending if p.kind == "cpu" and p.key not in state.stuck]

    if state.last_action_at is not None and now - state.last_action_at < COOLDOWN_S:
        if gpu_pending or cpu_pending:
            held.append(f"{_describe(gpu_pending + cpu_pending)} waiting: "
                        f"cooldown: {COOLDOWN_S - (now - state.last_action_at):.0f}s left")
        return None, held

    def count(name: str) -> int:
        return gpu_counts.get(name, pools[name].count)

    def idle_for(name: str) -> float:
        return now - state.idle_since[name] if name in state.idle_since else -math.inf

    autoscaled = _autoscaled(pools)

    if gpu_pending:
        why = ("a GPU instance is still booting" if gpu_launching
               else _gpu_launching(now, nodes, gpu_counts, pools, state.usable_idle_since))
        if why is None:
            for name in autoscaled:
                if count(name) < pools[name].autoscale.max:
                    return Action("gpu_scale", target=count(name) + 1, pool=name,
                                  reason=f"GPU capacity: {_describe(gpu_pending)} unschedulable"), held
            why = "every autoscaled GPU pool is at its max" if autoscaled else "no GPU pool has autoscale"
        held.append(f"{_describe(gpu_pending)} waiting: {why}")

    if cpu_pending:
        if cpu_count < cpu_max:
            return Action("cpu_scale", target=cpu_count + 1,
                          reason=f"CPU/memory capacity: {_describe(cpu_pending)} unschedulable"), held
        held.append(f"{_describe(cpu_pending)} waiting: CPU pool at cpu_max {cpu_max}")

    removable = [(index, obs.name) for obs in nodes
                 if obs.kind == "cpu" and obs.name != PROTECTED_CPU_NODE
                 and (index := _cpu_index(obs.name)) is not None and idle_for(obs.name) >= CPU_IDLE_S]
    if removable:
        _, victim = max(removable)
        return Action("cpu_remove", node=victim, reason=f"{victim} idle for {idle_for(victim):.0f}s"), held

    for name in autoscaled:
        bounds, current = pools[name].autoscale, count(name)
        if current <= bounds.min:
            continue
        highest = node_name(name, current)
        if idle_for(highest) >= bounds.idle_minutes * 60:
            return Action("gpu_scale", target=current - 1, pool=name,
                          reason=f"{highest} idle for {idle_for(highest):.0f}s"), held

    for name in autoscaled:
        launch = state.launches.get(name)
        if launch and now - launch.at >= GPU_JOIN_DEADLINE_S and launch.target - 1 >= pools[name].autoscale.min:
            return Action("gpu_scale", target=launch.target - 1, pool=name,
                          reason=f"{node_name(name, launch.target)} did not join within {GPU_JOIN_DEADLINE_S}s"), held
    return None, held
