"""Autoscaler decision engine (stretch S2; design spec sections 10.2, 10.4, 10.5).

The fixtures build kubectl-JSON-shaped dicts (`kubectl get pods -A -o json`, `kubectl get nodes -o json`); every
decision is a pure function of them, the stored State and `now`.
"""
import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpuctl import autoscale
from gpuctl.autoscale import (
    CLOCK_SKEW_S, COOLDOWN_S, CPU_IDLE_S, GPU_JOIN_DEADLINE_S, PENDING_GRACE_S, STUCK_IDLE_S, STUCK_TTL_S, Action,
    Launch, Mark, NodeObs, PendingPod, State, decide, node_observations, pending_pods,
)
from gpuctl.spec import Autoscale, Pool

NOW = 1_800_000_000.0
GPU_LABEL = "node-restriction.kubernetes.io/accelerator"
# The overflow Pod's FailedScheduling message measured live on 2026-10-21: capacity and non-capacity reasons in one.
MIXED_GPU_MESSAGE = ("0/3 nodes are available: 1 Insufficient nvidia.com/gpu, 1 node(s) didn't match Pod's node "
                     "affinity/selector, 1 node(s) had untolerated taint(s). preemption: 0/3 nodes are available: "
                     "1 No preemption victims found for incoming pod, 2 Preemption is not helpful for scheduling.")


def ts(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pod(name="p", namespace="default", node=None, phase="Pending", unschedulable=None, since=NOW - 60,
        owner=None, gpu=0, container_statuses=None, scheduled=None):
    """A Pod. `unschedulable` is a FailedScheduling message (PodScheduled=False, reason Unschedulable)."""
    resources = {"limits": {"nvidia.com/gpu": str(gpu)}} if gpu else {"requests": {"cpu": "100m"}}
    status = {"phase": phase}
    if unschedulable is not None:
        status["conditions"] = [{"type": "PodScheduled", "status": "False", "reason": "Unschedulable",
                                 "message": unschedulable, "lastTransitionTime": ts(since)}]
    elif scheduled:
        status["conditions"] = [{"type": "PodScheduled", "status": "True", "lastTransitionTime": ts(since)}]
    if container_statuses is not None:
        status["containerStatuses"] = container_statuses
    meta = {"name": name, "namespace": namespace}
    if owner:
        meta["ownerReferences"] = [{"kind": owner, "name": f"{name}-owner", "controller": True}]
    spec = {"containers": [{"name": "c", "image": "x", "resources": resources}]}
    if node:
        spec["nodeName"] = node
    return {"metadata": meta, "spec": spec, "status": status}


def node(name, ready=True, gpu_label=None, gpus=..., unschedulable=None, taints=None):
    """A node. A GPU node advertises one nvidia.com/gpu unless `gpus` says otherwise (None: the key is absent)."""
    labels = {"kubernetes.io/hostname": name}
    if gpu_label is None:
        gpu_label = name.startswith("k3s-gpu-")
    if gpu_label:
        labels[GPU_LABEL] = "nvidia"
    status = {"conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
              "allocatable": {"cpu": "2", "memory": "4Gi"}}
    if gpus is ...:
        gpus = "1" if name.startswith("k3s-gpu-") else None
    if gpus is not None:
        status["allocatable"]["nvidia.com/gpu"] = gpus
    spec = {}
    if unschedulable is not None:
        spec["unschedulable"] = unschedulable
    if taints is not None:
        spec["taints"] = taints
    return {"metadata": {"name": name, "labels": labels}, "spec": spec, "status": status}


def items(*objs):
    return {"apiVersion": "v1", "kind": "List", "items": list(objs)}


def cpu_node(name, busy=False, ready=True, schedulable=True):
    return NodeObs(name=name, kind="cpu", ready=ready, labeled=True, busy=busy, schedulable=schedulable)


def gpu_node(name, busy=False, ready=True, labeled=True, allocatable=True, schedulable=True):
    return NodeObs(name=name, kind="gpu", ready=ready, labeled=labeled, busy=busy, schedulable=schedulable,
                   gpu_allocatable=allocatable)


def pool(name="a6000", count=1, lo=1, hi=2, idle_minutes=5, autoscaled=True):
    return Pool(name=name, count=count, instance_type="gpu_1x_a6000", fallback_types=(), regions=("us-west-2",),
                image_family="lambda-stack-22-04",
                autoscale=Autoscale(min=lo, max=hi, idle_minutes=idle_minutes) if autoscaled else None)


POOLS = {"a6000": pool()}
CLUSTER = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1", busy=True)]


def run(*, pending=(), nodes=CLUSTER, state=None, cpu_count=1, cpu_max=3, gpu_counts=None, pools=POOLS,
        gpu_launching=False, now=NOW):
    return decide(now=now, pending=list(pending), nodes=list(nodes), state=state or State(), cpu_count=cpu_count,
                  cpu_max=cpu_max, gpu_counts={"a6000": 1} if gpu_counts is None else gpu_counts, pools=pools,
                  gpu_launching=gpu_launching)


def cpu_pending(since=NOW - 60):
    return PendingPod(namespace="scale-demo", name="php-apache-x", resource="cpu", since=since)


def gpu_pending(since=NOW - 60):
    return PendingPod(namespace="gpu-test", name="gpu-burn-x", resource="gpu", since=since)


def test_constants_match_spec():
    assert (PENDING_GRACE_S, CPU_IDLE_S, COOLDOWN_S) == (30, 300, 120)


# Everything autoscale.py may import: no os, pathlib, socket, subprocess, time, requests.
ALLOWED_IMPORTS = {"__future__", "json", "math", "re", "collections.abc", "dataclasses", "datetime", "typing",
                   "gpuctl.planner", "gpuctl.spec"}
# Calls that read a clock, a file or the terminal, or run something.
FORBIDDEN_CALLS = {"open", "now", "utcnow", "today", "time", "time_ns", "monotonic", "perf_counter", "print", "input",
                   "exec", "eval", "__import__", "read_text", "system", "run", "Popen"}


def _imports_and_calls(source):
    imported, called = set(), set()
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import):
            imported |= {alias.name for alias in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add(n.module)
        elif isinstance(n, ast.Call):
            called.add(n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None))
    return imported, called


def test_module_does_no_io():
    # Scope: pure functions, no I/O, no clock reads; `now` is always a parameter.
    imported, called = _imports_and_calls(Path(autoscale.__file__).read_text(encoding="utf-8"))
    assert imported <= ALLOWED_IMPORTS
    assert not called & FORBIDDEN_CALLS


@pytest.mark.parametrize("snippet", ["import os", "from time import monotonic", "import socket", "import pathlib",
                                     "x = datetime.now(timezone.utc)", "x = datetime.utcnow()", "y = open('f')"])
def test_io_check_catches_a_regression(snippet):
    imported, called = _imports_and_calls(snippet)
    assert not imported <= ALLOWED_IMPORTS or called & FORBIDDEN_CALLS


# --- pending_pods -------------------------------------------------------------------------------------------------

def test_pending_cpu_after_grace_scales_cpu():
    pods = items(pod("php-apache-x", "scale-demo", unschedulable="0/3 nodes are available: 3 Insufficient cpu.",
                     since=NOW - PENDING_GRACE_S))
    pending = pending_pods(pods, NOW)
    assert pending == [PendingPod("scale-demo", "php-apache-x", "cpu", NOW - PENDING_GRACE_S)]
    action, state = run(pending=pending, cpu_count=1)
    assert action == Action("cpu_scale", target=2, reason=action.reason)
    assert "scale-demo/php-apache-x" in action.reason
    assert state.last_action_at == NOW


def test_pending_younger_than_grace_is_ignored():
    pods = items(pod(unschedulable="0/3 nodes are available: 3 Insufficient cpu.", since=NOW - PENDING_GRACE_S + 1))
    assert pending_pods(pods, NOW) == []
    action, state = run(pending=pending_pods(pods, NOW))
    assert action is None and state.last_action_at is None


@pytest.mark.parametrize("message, resource", [
    ("0/3 nodes are available: 3 Insufficient cpu.", "cpu"),
    ("0/3 nodes are available: 2 Insufficient memory, 1 node(s) had untolerated taint(s).", "memory"),
    ("0/3 nodes are available: 1 Insufficient cpu, 1 Insufficient memory.", "cpu"),
    (MIXED_GPU_MESSAGE, "gpu"),
    ("0/3 nodes are available: 1 Insufficient cpu, 1 Insufficient nvidia.com/gpu.", "gpu"),
])
def test_capacity_reason_maps_to_resource(message, resource):
    assert [p.resource for p in pending_pods(items(pod(unschedulable=message)), NOW)] == [resource]


def test_mixed_failedscheduling_message_counts_as_gpu_capacity():
    # Live message: one node lacks the GPU, the others fail affinity/taints. Adding a GPU node fixes it.
    pending = pending_pods(items(pod("gpu-burn-x", "gpu-test", unschedulable=MIXED_GPU_MESSAGE)), NOW)
    assert pending == [PendingPod("gpu-test", "gpu-burn-x", "gpu", NOW - 60)]
    action, _ = run(pending=pending)
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a6000", 2)


def test_non_capacity_pending_is_ignored():
    # Review Focus #2: affinity mismatch, an image that cannot be pulled, an unbound PVC.
    pods = items(
        pod("affinity", unschedulable="0/3 nodes are available: 3 node(s) didn't match Pod's node affinity"),
        pod("image", node="k3s-cpu-2", scheduled=True, container_statuses=[
            {"name": "c", "ready": False, "state": {"waiting": {"reason": "ImagePullBackOff"}}}]),
        pod("pvc", unschedulable="0/3 nodes are available: pod has unbound immediate PersistentVolumeClaims. "
                                 "preemption: 0/3 nodes are available: 3 Preemption is not helpful for scheduling."),
    )
    assert pending_pods(pods, NOW) == []
    action, state = run(pending=pending_pods(pods, NOW))
    assert action is None and state.last_action_at is None


@pytest.mark.parametrize("mutate", [
    lambda p: p["status"].update(phase="Running"),
    lambda p: p["status"]["conditions"][0].update(status="True"),
    lambda p: p["status"]["conditions"][0].update(reason="SchedulerError"),
    lambda p: p["status"]["conditions"][0].update(type="Ready"),
    lambda p: p["status"]["conditions"][0].update(message="0/3 nodes are available: 3 Insufficient cpus."),
    lambda p: p["status"]["conditions"][0].update(message="0/3 nodes are available: 3 Insufficient example.com/foo."),
    lambda p: p["status"]["conditions"][0].pop("lastTransitionTime"),
    lambda p: p["status"]["conditions"][0].update(lastTransitionTime="yesterday"),
    lambda p: p["status"].pop("conditions"),
])
def test_only_unschedulable_capacity_pods_with_a_known_age_count(mutate):
    p = pod(unschedulable="0/3 nodes are available: 3 Insufficient cpu.")
    mutate(p)
    assert pending_pods(items(p), NOW) == []


def test_pending_pods_tolerates_an_empty_list():
    assert pending_pods({"items": []}, NOW) == [] and pending_pods({}, NOW) == []


# --- node_observations ----------------------------------------------------------------------------------------------

def test_daemonset_pods_do_not_make_node_busy():
    # Review Focus #3: nettest echo/probe, the device plugin and the exporters run on every node.
    nodes = items(node("k3s-cpu-2"), node("k3s-gpu-a6000-1"))
    pods = items(*(pod(n, "kube-system", node=host, phase="Running", owner="DaemonSet")
                   for host in ("k3s-cpu-2", "k3s-gpu-a6000-1")
                   for n in ("nettest-echo", "nettest-probe", "nvdp", "node-exporter", "dcgm-exporter")))
    assert node_observations(nodes, pods) == [
        NodeObs("k3s-cpu-2", "cpu", ready=True, labeled=True, busy=False),
        NodeObs("k3s-gpu-a6000-1", "gpu", ready=True, labeled=True, busy=False),
    ]
    obs = node_observations(nodes, pods)
    _, state = run(nodes=obs, gpu_counts={"a6000": 1})
    assert state.idle_since == {"k3s-cpu-2": NOW, "k3s-gpu-a6000-1": NOW}


def test_cpu_node_busy_with_any_non_daemonset_pod():
    nodes = items(node("k3s-cpu-2"), node("k3s-cpu-3"))
    pods = items(pod("web", node="k3s-cpu-2", phase="Running", owner="ReplicaSet"),
                 pod("done", node="k3s-cpu-3", phase="Succeeded", owner="Job"),
                 pod("crashed", node="k3s-cpu-3", phase="Failed"))
    assert [(o.name, o.busy) for o in node_observations(nodes, pods)] == [("k3s-cpu-2", True), ("k3s-cpu-3", False)]


def test_gpu_node_busy_only_with_a_gpu_request():
    nodes = items(node("k3s-gpu-a6000-1"), node("k3s-gpu-a6000-2"), node("k3s-gpu-a6000-3"))
    init_gpu = pod("init-gpu", node="k3s-gpu-a6000-3", phase="Pending")
    init_gpu["spec"]["initContainers"] = [{"name": "i", "resources": {"requests": {"nvidia.com/gpu": "1"}}}]
    pods = items(pod("cpu-only", node="k3s-gpu-a6000-1", phase="Running"),
                 pod("burn", node="k3s-gpu-a6000-2", phase="Running", gpu=1),
                 init_gpu)
    assert [(o.name, o.busy) for o in node_observations(nodes, pods)] == [
        ("k3s-gpu-a6000-1", False), ("k3s-gpu-a6000-2", True), ("k3s-gpu-a6000-3", True)]


def test_finished_gpu_pod_does_not_keep_node_busy():
    nodes = items(node("k3s-gpu-a6000-2"))
    pods = items(pod("burn-1", node="k3s-gpu-a6000-2", phase="Succeeded", gpu=1))
    assert node_observations(nodes, pods)[0].busy is False


def test_node_observations_only_own_nodes_and_read_ready_and_label():
    nodes = items(node("k3s-cp"), node("other-node"), node("k3s-cpu-1"),
                  node("k3s-gpu-a6000-2", ready=False, gpu_label=False), node("k3s-gpu-a6000-1", gpu_label=False))
    nodes["items"].append({"metadata": {"name": "k3s-cpu-4"}, "status": {}})  # no conditions yet
    assert node_observations(nodes, items()) == [
        NodeObs("k3s-cpu-1", "cpu", ready=True, labeled=True, busy=False),
        NodeObs("k3s-gpu-a6000-2", "gpu", ready=False, labeled=False, busy=False),
        NodeObs("k3s-gpu-a6000-1", "gpu", ready=True, labeled=False, busy=False),
        NodeObs("k3s-cpu-4", "cpu", ready=False, labeled=True, busy=False),
    ]


@pytest.mark.parametrize("bad", [None, {}, {"kind": "Status", "message": "Unauthorized"}, {"items": None},
                                 {"items": {}}, {"items": ["not-a-pod"]}, []])
def test_node_observations_rejects_a_malformed_listing(bad):
    # Busyness guards every scale-down: a broken listing must never read as "nothing runs anywhere".
    with pytest.raises(ValueError):
        node_observations(items(node("k3s-cpu-2")), bad)
    with pytest.raises(ValueError):
        node_observations(bad, items())


def test_node_observations_accepts_an_empty_listing():
    assert node_observations(items(node("k3s-cpu-2")), items()) == [NodeObs("k3s-cpu-2", "cpu", True, True, False)]
    assert node_observations(items(), items()) == []


def test_node_observations_read_schedulability():
    # Review S1-2 / X1-1: a cordoned or tainted node cannot take new Pods, so its idleness says nothing about them.
    gpu_taint = {"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"}  # node-bootstrap.sh; tolerated
    nodes = items(
        node("k3s-cpu-2", unschedulable=True),                                                      # cordoned (drain)
        node("k3s-cpu-3", taints=[{"key": "maintenance", "effect": "NoSchedule"}]),
        node("k3s-cpu-4", taints=[{"key": "node.kubernetes.io/unreachable", "effect": "NoExecute"}]),
        node("k3s-cpu-5", taints=[{"key": "spot", "effect": "PreferNoSchedule"}]),                  # only a preference
        node("k3s-cpu-6", unschedulable=False, taints=[]),
        node("k3s-gpu-a6000-1", taints=[gpu_taint]),
        node("k3s-gpu-a6000-2", taints=[gpu_taint, {"key": "maintenance", "effect": "NoSchedule"}]),
        node("k3s-cpu-7", taints=[{"key": "nvidia.com/gpu", "effect": "NoSchedule"}]),              # not a GPU node
    )
    assert [(o.name, o.schedulable) for o in node_observations(nodes, items())] == [
        ("k3s-cpu-2", False), ("k3s-cpu-3", False), ("k3s-cpu-4", False), ("k3s-cpu-5", True), ("k3s-cpu-6", True),
        ("k3s-gpu-a6000-1", True), ("k3s-gpu-a6000-2", False), ("k3s-cpu-7", False)]


@pytest.mark.parametrize("gpus, expected", [("1", True), ("2", True), ("0", False), (None, False)])
def test_node_observations_read_advertised_gpus(gpus, expected):
    # Review C1-1: a labeled GPU node advertises no nvidia.com/gpu until the device plugin runs on it.
    obs = node_observations(items(node("k3s-gpu-a6000-2", gpus=gpus), node("k3s-cpu-2")), items())
    assert [(o.name, o.gpu_allocatable) for o in obs] == [("k3s-gpu-a6000-2", expected), ("k3s-cpu-2", True)]


def test_node_obs_defaults_keep_the_brief_signature():
    obs = NodeObs("k3s-cpu-2", "cpu", True, True, False)
    assert obs.schedulable and obs.gpu_allocatable and obs.usable
    assert not NodeObs("k3s-gpu-a6000-2", "gpu", True, True, False, gpu_allocatable=False).usable
    assert not NodeObs("k3s-cpu-2", "cpu", True, True, False, schedulable=False).usable
    assert not NodeObs("k3s-gpu-a6000-2", "gpu", True, False, False).usable
    assert not NodeObs("k3s-cpu-2", "cpu", False, True, False).usable


# --- State ----------------------------------------------------------------------------------------------------------

def test_state_roundtrip_preserves_cooldown():
    # Review Focus #1: the cooldown survives a restart through .run/autoscaler-state.json.
    s = State(last_action_at=1000.0, idle_since={"k3s-cpu-2": 900.0})
    assert State.from_json(s.to_json()) == s
    assert json.loads(s.to_json()) == {"last_action_at": 1000.0, "idle_since": {"k3s-cpu-2": 900.0}, "stuck": {},
                                       "launches": {}, "usable_idle_since": {}}


@pytest.mark.parametrize("text, expected", [
    ('{"last_action_at": 1000.0}', State(last_action_at=1000.0)),
    ('{"last_action_at": 1000, "idle_since": {}}', State(last_action_at=1000.0)),
    ('{"last_action_at": null, "idle_since": null}', State()),
    ('{}', State()),
    ('', State()),
])
def test_state_from_json_tolerates_missing_or_empty_fields(text, expected):
    assert State.from_json(text) == expected


@pytest.mark.parametrize("text", [
    "[]", "not json", '{"last_action_at": "soon"}', '{"last_action_at": true}', '{"idle_since": []}',
    '{"idle_since": {"k3s-cpu-2": "x"}}', '{"last_action_at": NaN}', '{"last_action_at": 1' + "0" * 400 + '}',
    '{"stuck": []}', '{"stuck": {"ns/p": 3}}', '{"launches": {"a6000": 2}}', '{"launches": {"a6000": {"at": 1.0}}}',
    '{"launches": {"a6000": {"target": "2", "at": 1.0}}}', '{"launches": {"a6000": {"target": 2, "at": "x"}}}',
    '{"launches": {"a6000": {"target": true, "at": 1.0}}}',
    '{"stuck": {"ns/p": "k3s-cpu-2 was idle"}}', '{"stuck": {"ns/p": {"why": "x"}}}',
    '{"stuck": {"ns/p": {"why": 3, "at": 1.0}}}', '{"stuck": {"ns/p": {"why": "x", "at": "y"}}}',
    '{"launches": {"a6000": {"target": 2, "at": 1.0, "pods": "ns/p"}}}',
    '{"launches": {"a6000": {"target": 2, "at": 1.0, "pods": [3]}}}',
    '{"usable_idle_since": []}', '{"usable_idle_since": {"k3s-cpu-2": "x"}}',
])
def test_state_from_json_rejects_garbage(text):
    # A corrupt file must not silently reset the cooldown: the caller sees the error.
    with pytest.raises(ValueError):
        State.from_json(text)


def test_state_roundtrip_preserves_stuck_pods_and_launches():
    s = State(last_action_at=1000.0, idle_since={"k3s-gpu-a6000-2": 950.0},
              stuck={"gpu-test/burn-2": Mark("k3s-gpu-a6000-2 was idle", 980.0)},
              launches={"a6000": Launch(2, 990.0, ("gpu-test/burn-1", "gpu-test/burn-2")), "b": Launch(3, 991.0)},
              usable_idle_since={"k3s-gpu-a6000-2": 960.0})
    assert State.from_json(s.to_json()) == s
    # A launch saved without its Pods still loads (nothing to mark on rollback).
    assert State.from_json('{"launches": {"a6000": {"target": 2, "at": 1.0}}}').launches == {"a6000": Launch(2, 1.0)}


def test_held_reason_is_not_saved():
    s = State(last_action_at=1000.0, held="cooldown: 10s left")
    assert "held" not in json.loads(s.to_json())
    assert State.from_json(s.to_json()).held is None


def test_state_timestamp_in_the_future_is_rejected():
    # A last_action_at ahead of `now` would hold every action silently (clock moved back, or a bad file).
    future = NOW + CLOCK_SKEW_S + 1
    for state in (State(last_action_at=future), State(idle_since={"k3s-cpu-2": future}),
                  State(launches={"a6000": Launch(2, future)}), State(stuck={"ns/p": Mark("x", future)}),
                  State(usable_idle_since={"k3s-cpu-2": future})):
        with pytest.raises(ValueError, match="future"):
            run(pending=[cpu_pending()], state=state)
    action, _ = run(pending=[cpu_pending()], state=State(last_action_at=NOW + CLOCK_SKEW_S))
    assert action is None  # within the allowed skew: just a cooldown


def test_default_states_do_not_share_idle_since():
    a, b = State(), State()
    a.idle_since["x"] = 1.0
    assert b.idle_since == {}


def test_restart_during_cooldown_does_not_scale_again():
    # Review Focus #1: scale, Ctrl-C, restart 10 s later with the saved state: no second scale-up.
    action, state = run(pending=[cpu_pending()], cpu_count=1)
    assert action.kind == "cpu_scale"
    restored = State.from_json(state.to_json())
    again, _ = run(pending=[cpu_pending()], cpu_count=1, state=restored, now=NOW + 10)
    assert again is None


# --- decide -------------------------------------------------------------------------------------------------------

def test_cpu_max_caps_scale_up():
    action, _ = run(pending=[cpu_pending()], cpu_count=3, cpu_max=3)
    assert action is None
    action, _ = run(pending=[cpu_pending()], cpu_count=2, cpu_max=3)
    assert (action.kind, action.target) == ("cpu_scale", 3)


def test_cooldown_blocks_actions():
    state = State(last_action_at=NOW - COOLDOWN_S + 1)
    action, new_state = run(pending=[cpu_pending()], state=state)
    assert action is None and new_state.last_action_at == NOW - COOLDOWN_S + 1
    action, new_state = run(pending=[cpu_pending()], state=State(last_action_at=NOW - COOLDOWN_S))
    assert action.kind == "cpu_scale" and new_state.last_action_at == NOW


def test_idle_since_tracked_during_cooldown():
    # The idle clock keeps running through a cooldown; otherwise every action would restart it.
    nodes = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2"), cpu_node("k3s-cpu-3", busy=True),
             cpu_node("k3s-cpu-4", ready=False)]
    state = State(last_action_at=NOW - 1, idle_since={"k3s-cpu-2": NOW - 500, "k3s-cpu-3": NOW - 500,
                                                       "k3s-cpu-4": NOW - 500, "k3s-cpu-9": NOW - 500})
    action, new_state = run(nodes=nodes, state=state, cpu_count=4)
    assert action is None
    assert new_state.idle_since == {"k3s-cpu-2": NOW - 500}
    assert state.idle_since["k3s-cpu-3"] == NOW - 500  # the input state is not mutated


def test_gpu_pending_scales_gpu_pool_within_max():
    action, state = run(pending=[gpu_pending()], gpu_counts={"a6000": 1})
    assert action == Action("gpu_scale", target=2, pool="a6000", reason=action.reason)
    assert state.last_action_at == NOW
    nodes = [*CLUSTER, gpu_node("k3s-gpu-a6000-2", busy=True)]
    action, state = run(pending=[gpu_pending()], nodes=nodes, gpu_counts={"a6000": 2})
    assert action is None and state.last_action_at is None


def test_gpu_scale_up_picks_first_autoscaled_pool_with_room_by_name():
    pools = {"zeta": pool("zeta", hi=3), "b": pool("b", hi=1), "a": pool("a", autoscaled=False), "c": pool("c")}
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-b-1", busy=True), gpu_node("k3s-gpu-c-1", busy=True),
             gpu_node("k3s-gpu-zeta-1", busy=True)]
    action, _ = run(pending=[gpu_pending()], nodes=nodes, pools=pools, gpu_counts={"a": 0, "b": 1, "c": 1, "zeta": 1})
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "c", 2)


def test_no_second_gpu_launch_while_launching():
    # Review Focus #4: an instance still booting (reported by the caller) blocks another launch.
    action, state = run(pending=[gpu_pending()], gpu_launching=True)
    assert action is None and state.last_action_at is None


@pytest.mark.parametrize("new_node", [
    gpu_node("k3s-gpu-a6000-2", ready=False, labeled=False),  # joined, not Ready yet
    gpu_node("k3s-gpu-a6000-2", ready=False, labeled=True),   # labeled node gone NotReady (reboot, kubelet restart)
    gpu_node("k3s-gpu-a6000-2", ready=True, labeled=False),   # Ready, admin label not applied yet
    None,                                                     # instance active, node not registered yet
])
def test_gpu_node_still_joining_counts_as_launching(new_node):
    # Review Focus #4: NotReady, unlabeled, or fewer GPU nodes than the pool's count -> still launching.
    nodes = [*CLUSTER, new_node] if new_node else CLUSTER
    action, _ = run(pending=[gpu_pending()], nodes=nodes, gpu_counts={"a6000": 2},
                    pools={"a6000": pool(hi=3)})
    assert action is None


def test_gpu_launching_does_not_block_cpu_scale_up():
    action, _ = run(pending=[gpu_pending(), cpu_pending()], gpu_launching=True)
    assert (action.kind, action.target) == ("cpu_scale", 2)


def test_idle_cpu_node_removed_but_never_cpu_1():
    # Review Focus #5.
    nodes = [cpu_node("k3s-cpu-1"), *CLUSTER[1:]]
    action, _ = run(nodes=nodes, state=State(idle_since={"k3s-cpu-1": NOW - 10 * CPU_IDLE_S}))
    assert action is None
    nodes = [cpu_node("k3s-cpu-1"), cpu_node("k3s-cpu-2"), *CLUSTER[1:]]
    idle = {"k3s-cpu-1": NOW - 10 * CPU_IDLE_S, "k3s-cpu-2": NOW - CPU_IDLE_S}
    action, state = run(nodes=nodes, cpu_count=2, state=State(idle_since=idle))
    assert action == Action("cpu_remove", node="k3s-cpu-2", reason=action.reason)
    assert state.last_action_at == NOW


def test_cpu_node_idle_less_than_threshold_is_kept():
    nodes = [*CLUSTER, cpu_node("k3s-cpu-2")]
    action, _ = run(nodes=nodes, cpu_count=2, state=State(idle_since={"k3s-cpu-2": NOW - CPU_IDLE_S + 1}))
    assert action is None
    action, state = run(nodes=nodes, cpu_count=2)  # first sighting starts the clock
    assert action is None and state.idle_since["k3s-cpu-2"] == NOW


def test_busy_highest_cpu_node_not_removed_idle_lower_one_is():
    nodes = [*CLUSTER, cpu_node("k3s-cpu-2"), cpu_node("k3s-cpu-3", busy=True), cpu_node("k3s-cpu-10"),
             cpu_node("k3s-cpu-11", busy=True)]
    idle = {name: NOW - 2 * CPU_IDLE_S for name in ("k3s-cpu-2", "k3s-cpu-3", "k3s-cpu-10", "k3s-cpu-11")}
    action, _ = run(nodes=nodes, cpu_count=5, state=State(idle_since=idle))
    assert (action.kind, action.node) == ("cpu_remove", "k3s-cpu-10")  # numeric order, not string order


def test_gpu_scale_down_only_when_highest_index_idle():
    # Review Focus #5: gpuctl deletes the highest index, so only its idleness counts.
    pools = {"a6000": pool(lo=1, hi=3, idle_minutes=5)}
    long_ago = NOW - 5 * 60
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1"), gpu_node("k3s-gpu-a6000-2", busy=True)]
    idle = {"k3s-gpu-a6000-1": long_ago, "k3s-gpu-a6000-2": long_ago}
    action, _ = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2}, state=State(idle_since=idle))
    assert action is None
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1", busy=True), gpu_node("k3s-gpu-a6000-2")]
    action, state = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2}, state=State(idle_since=idle))
    assert action == Action("gpu_scale", target=1, pool="a6000", reason=action.reason)
    assert state.last_action_at == NOW
    # One second short of idle_minutes: kept.
    action, _ = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2},
                    state=State(idle_since={"k3s-gpu-a6000-2": long_ago + 1}))
    assert action is None


def test_gpu_never_below_min():
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1")]
    idle = State(idle_since={"k3s-gpu-a6000-1": NOW - 3600})
    action, _ = run(nodes=nodes, pools={"a6000": pool(lo=1)}, gpu_counts={"a6000": 1}, state=idle)
    assert action is None
    action, _ = run(nodes=nodes, pools={"a6000": pool(lo=0)}, gpu_counts={"a6000": 1}, state=idle)
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a6000", 0)


def test_pool_without_autoscale_is_never_scaled():
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1")]
    pools = {"a6000": pool(lo=0, autoscaled=False)}
    idle = State(idle_since={"k3s-gpu-a6000-1": NOW - 3600})
    assert run(nodes=nodes, pools=pools, state=idle)[0] is None
    assert run(pending=[gpu_pending()], nodes=nodes, pools=pools)[0] is None


def test_priority_scale_up_before_scale_down():
    # Each scale-up competes with a scale-down of the other kind: a Pending Pod next to an idle node of its own kind
    # is stuck (see the stuck-Pod tests below), so it would not ask for a scale-up at all.
    pools = {"a6000": pool(lo=0, hi=3)}
    long_idle = {n: NOW - 3600 for n in ("k3s-cpu-2", "k3s-gpu-a6000-2")}
    common = dict(pools=pools, gpu_counts={"a6000": 2}, cpu_count=2, state=State(idle_since=long_idle))
    cpu_idle = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2"), gpu_node("k3s-gpu-a6000-1", busy=True),
                gpu_node("k3s-gpu-a6000-2", busy=True)]
    action, _ = run(pending=[gpu_pending()], nodes=cpu_idle, **common)
    assert (action.kind, action.target) == ("gpu_scale", 3)  # rule 2 before rule 4
    gpu_idle = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2", busy=True),
                gpu_node("k3s-gpu-a6000-1", busy=True), gpu_node("k3s-gpu-a6000-2")]
    action, _ = run(pending=[cpu_pending()], nodes=gpu_idle, **common)
    assert (action.kind, action.target) == ("cpu_scale", 3)  # rule 3 before rule 5
    both_idle = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2"), gpu_node("k3s-gpu-a6000-1", busy=True),
                 gpu_node("k3s-gpu-a6000-2")]
    action, _ = run(nodes=both_idle, **common)
    assert action == Action("cpu_remove", node="k3s-cpu-2", reason=action.reason)  # rule 4 before rule 5
    action, _ = run(nodes=gpu_idle, **common)
    assert (action.kind, action.target) == ("gpu_scale", 1)


def test_every_action_carries_a_reason():
    nodes = [*CLUSTER, cpu_node("k3s-cpu-2")]
    actions = [
        run(pending=[cpu_pending()])[0],
        run(pending=[gpu_pending()])[0],
        run(nodes=nodes, cpu_count=2, state=State(idle_since={"k3s-cpu-2": NOW - CPU_IDLE_S}))[0],
    ]
    assert all(a is not None and a.reason for a in actions)


# --- rule 0 per the brief: Ready and not busy, labeled or not ------------------------------------------------------

def test_unlabeled_idle_gpu_node_gets_an_idle_clock_and_is_scaled_down():
    # Brief rule 0 has no label condition: a GPU node whose labeling step failed must not run (and bill) forever.
    pools = {"a6000": pool(lo=1, hi=2, idle_minutes=5)}
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1", busy=True),
             gpu_node("k3s-gpu-a6000-2", labeled=False)]
    _, state = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2},
                   state=State(idle_since={"k3s-gpu-a6000-2": NOW - 100}))
    assert state.idle_since["k3s-gpu-a6000-2"] == NOW - 100
    action, _ = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2},
                    state=State(idle_since={"k3s-gpu-a6000-2": NOW - 5 * 60}))
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a6000", 1)


def test_fixed_pool_with_fewer_nodes_does_not_block_an_autoscaled_pool():
    # The "registered < count" launching check covers autoscaled pools only; a fixed pool is managed by hand.
    pools = {"a": pool("a", count=1, hi=2), "b": pool("b", count=1, autoscaled=False)}
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a-1", busy=True)]
    action, _ = run(pending=[gpu_pending()], nodes=nodes, pools=pools, gpu_counts={"a": 1})
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a", 2)


# --- held: why a scale-up that Pending Pods asked for was withheld ---------------------------------------------------

@pytest.mark.parametrize("kwargs, words", [
    (dict(pending=[gpu_pending()], gpu_launching=True), "booting"),
    (dict(pending=[gpu_pending()], nodes=[*CLUSTER, gpu_node("k3s-gpu-a6000-2", ready=False)],
          gpu_counts={"a6000": 2}, pools={"a6000": pool(hi=3)}), "k3s-gpu-a6000-2 is NotReady"),
    (dict(pending=[gpu_pending()], nodes=[*CLUSTER, gpu_node("k3s-gpu-a6000-2", labeled=False)],
          gpu_counts={"a6000": 2}, pools={"a6000": pool(hi=3)}), "k3s-gpu-a6000-2 has no"),
    (dict(pending=[gpu_pending()], gpu_counts={"a6000": 2}, pools={"a6000": pool(hi=3)}), "1 of 2"),
    (dict(pending=[gpu_pending()], nodes=[*CLUSTER, gpu_node("k3s-gpu-a6000-2", busy=True)],
          gpu_counts={"a6000": 2}), "max"),
    (dict(pending=[cpu_pending()], cpu_count=3), "cpu_max 3"),
    (dict(pending=[cpu_pending()], state=State(last_action_at=NOW - 35)), "cooldown: 85s left"),
])
def test_withheld_scale_up_reports_why(kwargs, words):
    action, state = run(**kwargs)
    assert action is None
    assert words in state.held
    assert "/" in state.held  # names the waiting Pod


def test_nothing_pending_means_nothing_held():
    assert run()[1].held is None
    assert run(state=State(last_action_at=NOW - 1))[1].held is None


# --- Pods a new node cannot help (review C0-1, X0-2; round 2: S1-2, C1-1, X1-1, X1-2) -----------------------------

def big(resource="cpu", name="too-big", since=NOW - 60):
    return PendingPod(namespace="scale-demo", name=name, resource=resource, since=since)


def test_pending_pod_next_to_an_idle_node_is_stuck_and_does_not_scale_up():
    # The scheduler retries a capacity-Pending Pod on every node; one still Pending while a same-kind node has been
    # usable and idle for STUCK_IDLE_S does not fit that node, so another node of the same shape would not help.
    nodes = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2")]
    action, state = run(pending=[big()], nodes=nodes, cpu_count=2,
                        state=State(usable_idle_since={"k3s-cpu-2": NOW - STUCK_IDLE_S}))
    assert action is None
    assert state.stuck == {"scale-demo/too-big": Mark("k3s-cpu-2 was idle", NOW)}
    assert "scale-demo/too-big" in state.held and "k3s-cpu-2" in state.held
    # Usable and idle for less than STUCK_IDLE_S (the scheduler may not have caught up yet): scale up as usual.
    action, state = run(pending=[big()], nodes=nodes, cpu_count=2,
                        state=State(usable_idle_since={"k3s-cpu-2": NOW - STUCK_IDLE_S + 1}))
    assert (action.kind, action.target) == ("cpu_scale", 3) and state.stuck == {}


def test_evidence_needs_pod_and_idle_node_together_for_stuck_idle_s():
    # The node has been idle long, but the Pod has been Pending for only 40 s of it: not enough to prove anything.
    nodes = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2")]
    idle = State(usable_idle_since={"k3s-cpu-2": NOW - 600})
    action, state = run(pending=[big(since=NOW - 40)], nodes=nodes, cpu_count=2, state=idle)
    assert action.kind == "cpu_scale" and state.stuck == {}
    _, state = run(pending=[big(since=NOW - STUCK_IDLE_S)], nodes=nodes, cpu_count=2, state=idle)
    assert set(state.stuck) == {"scale-demo/too-big"}


def test_usable_idle_clock_covers_only_nodes_that_can_take_a_pod():
    # Rule 0's idle_since stays as the brief has it (Ready and not busy); the evidence clock also needs the node to
    # be schedulable and, for a GPU node, labeled and advertising GPUs.
    nodes = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2"), cpu_node("k3s-cpu-3", schedulable=False),
             cpu_node("k3s-cpu-4", ready=False), gpu_node("k3s-gpu-a6000-1", busy=True),
             gpu_node("k3s-gpu-a6000-2", allocatable=False), gpu_node("k3s-gpu-a6000-3", labeled=False),
             gpu_node("k3s-gpu-a6000-4")]
    before = State(usable_idle_since={n: NOW - 500 for n in ("k3s-cpu-2", "k3s-cpu-3", "k3s-gpu-a6000-2")})
    _, state = run(nodes=nodes, state=before, cpu_count=4, gpu_counts={"a6000": 4}, pools={"a6000": pool(hi=4)})
    assert state.usable_idle_since == {"k3s-cpu-2": NOW - 500, "k3s-gpu-a6000-4": NOW}
    assert set(state.idle_since) == {"k3s-cpu-2", "k3s-cpu-3", "k3s-gpu-a6000-2", "k3s-gpu-a6000-3",
                                     "k3s-gpu-a6000-4"}


def test_cordoned_idle_node_is_no_evidence_and_the_pod_scales_once_it_is_gone():
    # Review S1-2: S4 drains k3s-cpu-3 (cordoned, still Ready, idle 330 s) while a php-apache replica is Pending.
    busy = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2", busy=True)]
    replica = big(name="php-apache-z", since=NOW - 100)
    draining = State(last_action_at=NOW - 10, idle_since={"k3s-cpu-3": NOW - 330},
                     usable_idle_since={"k3s-cpu-3": NOW - 330})
    action, state = run(pending=[replica], nodes=[*busy, cpu_node("k3s-cpu-3", schedulable=False)], cpu_count=3,
                        state=draining)
    assert action is None and state.stuck == {}
    # k3s-cpu-3 is gone: cpu_count 2 < cpu_max 3, so the replica gets its node once the cooldown is over.
    action, state = run(pending=[replica], nodes=busy, cpu_count=2, state=state, now=NOW + COOLDOWN_S)
    assert (action.kind, action.target) == ("cpu_scale", 3)


def test_gpu_node_without_advertised_gpus_is_still_joining_and_no_evidence():
    # Review C1-1: k3s-gpu-a6000-2 is Ready and labeled, but the device plugin has not advertised its GPU yet.
    pools = {"a6000": pool(hi=3)}
    base = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1", busy=True)]
    waiting = [gpu_pending(since=NOW - 900)]
    common = dict(pending=waiting, pools=pools, gpu_counts={"a6000": 2})
    state = State(idle_since={"k3s-gpu-a6000-2": NOW - 200}, launches={"a6000": Launch(2, NOW - 700)})
    action, state = run(nodes=[*base, gpu_node("k3s-gpu-a6000-2", allocatable=False)], state=state, **common)
    assert action is None and state.stuck == {}
    assert "k3s-gpu-a6000-2 has no allocatable nvidia.com/gpu yet" in state.held  # no third launch either
    assert state.launches == {"a6000": Launch(2, NOW - 700)}  # not joined until it can run GPU Pods
    # GPUs advertised: the node gets its Pods before anything else happens, and the evidence clock starts now.
    usable = [*base, gpu_node("k3s-gpu-a6000-2")]
    action, state = run(nodes=usable, state=state, now=NOW + 20, **common)
    assert action is None and state.stuck == {} and state.launches == {}
    assert state.usable_idle_since == {"k3s-gpu-a6000-2": NOW + 20}
    assert "waiting for the scheduler" in state.held
    # Still Pending STUCK_IDLE_S later, next to the usable idle node: it does not fit, so it is marked.
    action, state = run(nodes=usable, state=state, now=NOW + 20 + STUCK_IDLE_S, **common)
    assert action is None and state.stuck == {"gpu-test/gpu-burn-x": Mark("k3s-gpu-a6000-2 was idle",
                                                                          NOW + 20 + STUCK_IDLE_S)}


def test_gpu_pods_waiting_for_the_device_plugin_still_get_their_nodes():
    # Review C1-1's probe: max 4, four 1-GPU Pods. A new node is Ready and labeled 300 s after its launch and
    # advertises its GPU 120 s after that; the scheduler then places one waiting Pod on it.
    pools = {"a6000": pool(lo=1, hi=4)}
    waiting = [PendingPod("gpu-test", f"p{i}", "gpu", NOW - 60) for i in (2, 3, 4)]
    joins, running, count, state, actions = {}, {"k3s-gpu-a6000-1"}, 1, State(), []
    for t in range(0, 3600, 20):
        now = NOW + t
        obs = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1", busy=True)]
        for name, (labeled_at, gpus_at) in joins.items():
            if now < labeled_at:
                continue  # booting, not registered yet
            if now >= gpus_at and name not in running and waiting:
                running.add(name)
                waiting.pop(0)
            obs.append(gpu_node(name, busy=name in running, allocatable=now >= gpus_at))
        action, state = decide(now=now, pending=list(waiting), nodes=obs, state=State.from_json(state.to_json()),
                               cpu_count=1, cpu_max=3, gpu_counts={"a6000": count}, pools=pools,
                               gpu_launching=any(now < labeled_at for labeled_at, _ in joins.values()))
        if action is not None:
            actions.append((action.kind, action.target))
            count = action.target
            joins[f"k3s-gpu-a6000-{count}"] = (now + 300, now + 420)
    assert actions == [("gpu_scale", 2), ("gpu_scale", 3), ("gpu_scale", 4)]
    assert waiting == [] and state.stuck == {}


def test_gpu_evidence_comes_only_from_the_pool_rule_2_would_scale():
    # Review X1-1: pool b is fixed and its node idle (normal for a pool without jobs); a Pod pinned to the
    # autoscaled pool a is Pending for GPUs. b's idle node says nothing about a new a node.
    waiting = [gpu_pending(since=NOW - 3600)]
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a-1", busy=True), gpu_node("k3s-gpu-b-1")]
    long_idle = State(idle_since={"k3s-gpu-b-1": NOW - 3600}, usable_idle_since={"k3s-gpu-b-1": NOW - 3600})
    pools = {"a": pool("a", count=1, hi=2), "b": pool("b", count=1, autoscaled=False)}
    action, state = run(pending=waiting, nodes=nodes, pools=pools, gpu_counts={"a": 1}, state=long_idle)
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a", 2) and state.stuck == {}
    # Nor does an idle node of an autoscaled pool that rule 2 does not reach while "a" has room.
    pools = {"a": pool("a", count=1, hi=2), "b": pool("b", count=1, lo=1, hi=2)}
    action, state = run(pending=waiting, nodes=nodes, pools=pools, gpu_counts={"a": 1, "b": 1}, state=long_idle)
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a", 2) and state.stuck == {}
    # Every autoscaled pool at max: the next scale-up goes to whichever pool shrinks first, so any of them counts.
    nodes = [*nodes, gpu_node("k3s-gpu-a-2", busy=True), gpu_node("k3s-gpu-b-2", busy=True)]
    pools = {"a": pool("a", count=2, hi=2), "b": pool("b", count=2, lo=1, hi=2)}
    _, state = run(pending=waiting, nodes=nodes, pools=pools, gpu_counts={"a": 2, "b": 2}, state=long_idle)
    assert state.stuck == {"gpu-test/gpu-burn-x": Mark("k3s-gpu-b-1 was idle", NOW)}


def test_idle_gpu_node_marks_gpu_pods_only():
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1")]
    action, state = run(pending=[big("gpu", "burn"), big("cpu", "web")], nodes=nodes,
                        state=State(usable_idle_since={"k3s-gpu-a6000-1": NOW - 600}))
    assert state.stuck == {"scale-demo/burn": Mark("k3s-gpu-a6000-1 was idle", NOW)}
    assert (action.kind, action.target) == ("cpu_scale", 2)


def test_stuck_mark_outlives_the_idle_node_and_expires_after_its_ttl():
    # Review X1-1: a mark is not forever. After STUCK_TTL_S the Pod may ask once more; if the evidence still holds
    # it is marked again, so a Pod that fits nowhere costs at most one up/down cycle per STUCK_TTL_S.
    def marked(at):
        return State(stuck={"scale-demo/too-big": Mark("k3s-cpu-2 was idle", at)})

    action, state = run(pending=[big()], state=marked(NOW - STUCK_TTL_S + 1))  # k3s-cpu-2 already removed
    assert action is None and state.stuck == marked(NOW - STUCK_TTL_S + 1).stuck
    action, state = run(pending=[big()], state=marked(NOW - STUCK_TTL_S))
    assert action.kind == "cpu_scale" and state.stuck == {}
    action, state = run(pending=[big("cpu", "other")], state=marked(NOW))  # the stuck Pod is gone, so is its mark
    assert action.kind == "cpu_scale" and state.stuck == {}


def simulate(resource, *, seconds=3600, step=20, join_delay=120):
    """Ticks decide() against a toy cluster in which one Pod fits no node; the state goes through JSON every tick."""
    pools = {"a6000": pool(lo=1, hi=2, idle_minutes=5)}
    busy = {"k3s-cpu-1": True, "k3s-gpu-a6000-1": True}
    joining, cpu_count, gpus, state, actions = {}, 1, 1, State(), []
    for t in range(0, seconds, step):
        now = NOW + t
        for name in [n for n, at in joining.items() if now >= at]:
            busy[name] = False
            del joining[name]
        nodes = [cpu_node(n, busy=b) if n.startswith("k3s-cpu-") else gpu_node(n, busy=b) for n, b in busy.items()]
        action, state = decide(now=now, pending=[big(resource)], nodes=nodes, state=State.from_json(state.to_json()),
                               cpu_count=cpu_count, cpu_max=3, gpu_counts={"a6000": gpus}, pools=pools,
                               gpu_launching=any(n.startswith("k3s-gpu-") for n in joining))
        if action is None:
            continue
        actions.append((t, action.kind, action.target or action.node))
        if action.kind == "cpu_scale":
            cpu_count = action.target
            joining[f"k3s-cpu-{cpu_count}"] = now + join_delay
        elif action.kind == "cpu_remove":
            del busy[action.node]
            cpu_count -= 1
        elif action.target > gpus:
            gpus = action.target
            joining[f"k3s-gpu-a6000-{gpus}"] = now + join_delay
        else:
            del busy[f"k3s-gpu-a6000-{gpus}"]
            gpus = action.target
    return actions, state


def test_unfixable_cpu_pod_gives_one_bounded_burst_not_an_endless_loop():
    actions, state = simulate("cpu")
    assert [kind for _, kind, _ in actions] == ["cpu_scale", "cpu_scale", "cpu_remove", "cpu_remove"]
    assert actions[-1][0] < 1200  # then quiet for the rest of the hour
    assert "scale-demo/too-big" in state.stuck and "scale-demo/too-big" in state.held


def test_unfixable_pod_gets_at_most_one_burst_per_stuck_ttl():
    for resource, burst in (("cpu", 4), ("gpu", 2)):
        actions, _ = simulate(resource, seconds=3 * STUCK_TTL_S)
        starts = [t for t, kind, target in actions if kind in ("cpu_scale", "gpu_scale") and target == 2]
        assert len(actions) <= 3 * burst
        assert all(b - a >= STUCK_TTL_S for a, b in zip(starts, starts[1:]))


def test_unfixable_gpu_pod_launches_at_most_one_instance():
    actions, state = simulate("gpu")
    assert [(kind, target) for _, kind, target in actions] == [("gpu_scale", 2), ("gpu_scale", 1)]
    assert "scale-demo/too-big" in state.stuck


# --- a GPU launch whose node never joins (review X0-5, X1-2) ----------------------------------------------------------

def test_gpu_scale_up_is_remembered_until_its_node_joins():
    _, state = run(pending=[gpu_pending()])
    assert state.launches == {"a6000": Launch(2, NOW, ("gpu-test/gpu-burn-x",))}
    joined = [*CLUSTER, gpu_node("k3s-gpu-a6000-2", busy=True)]
    _, after = run(nodes=joined, gpu_counts={"a6000": 2}, state=state, now=NOW + 300)
    assert after.launches == {}
    _, after = run(gpu_counts={"a6000": 1}, state=state, now=NOW + 300)  # count changed (gpu-down, a hand edit)
    assert after.launches == {}


def test_gpu_launch_that_never_joins_is_rolled_back_and_retried_only_after_the_ttl():
    launched = State(last_action_at=NOW, launches={"a6000": Launch(2, NOW, ("gpu-test/gpu-burn-x",))})
    common = dict(pending=[gpu_pending()], gpu_counts={"a6000": 2})
    action, state = run(state=launched, now=NOW + GPU_JOIN_DEADLINE_S - 1, **common)
    assert action is None and "1 of 2" in state.held
    rolled_back_at = NOW + GPU_JOIN_DEADLINE_S
    action, state = run(state=launched, now=rolled_back_at, **common)
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a6000", 1)
    assert "k3s-gpu-a6000-2" in action.reason
    assert state.launches == {}
    assert state.stuck == {"gpu-test/gpu-burn-x": Mark("k3s-gpu-a6000-2 was launched for it and never joined",
                                                       rolled_back_at)}
    # The same Pod launches no second instance once the pool is back at 1 ...
    again, state = run(pending=[gpu_pending()], state=state, now=rolled_back_at + COOLDOWN_S)
    assert again is None
    # ... until the mark expires: Lambda may have stock again by then.
    later, _ = run(pending=[gpu_pending()], state=state, now=rolled_back_at + STUCK_TTL_S)
    assert (later.kind, later.target) == ("gpu_scale", 2)


def test_rollback_marks_only_the_pods_the_launch_was_for():
    # Review X1-2: a Pod that arrived after the launch (or wants another pool) was never served by it.
    launched = State(last_action_at=NOW, launches={"a6000": Launch(2, NOW, ("gpu-test/gpu-burn-x",))})
    late = PendingPod("gpu-test", "late", "gpu", NOW + 100)
    action, state = run(pending=[gpu_pending(), late], state=launched, gpu_counts={"a6000": 2},
                        now=NOW + GPU_JOIN_DEADLINE_S)
    assert (action.kind, action.target) == ("gpu_scale", 1)
    assert set(state.stuck) == {"gpu-test/gpu-burn-x"}


def test_gpu_launch_rollback_never_goes_below_min():
    state = State(launches={"a6000": Launch(1, NOW - GPU_JOIN_DEADLINE_S)})
    action, _ = run(nodes=[cpu_node("k3s-cpu-1", busy=True)], gpu_counts={"a6000": 1}, state=state,
                    pools={"a6000": pool(lo=1)})
    assert action is None


# --- a launched GPU node is not scaled down before it can take GPU Pods (review C2-1) ---------------------------------

def simulate_slow_device_plugin(plugin_delay, idle_minutes, *, seconds=7200, step=20):
    """Review C2-1's probe: pool a6000 min 1 max 2, k3s-gpu-a6000-1 busy, one 1-GPU Pod Pending. The new instance
    boots for 300 s (registered, Ready and labeled then), its device plugin advertises the GPU `plugin_delay` s
    later, and the scheduler binds the Pod on the tick after that (the kubectl snapshot lags by one tick)."""
    pools = {"a6000": pool(lo=1, hi=2, idle_minutes=idle_minutes)}
    waiting = [PendingPod("gpu-test", "train", "gpu", NOW - 60)]
    launched_at, gpus = None, 1
    state, actions, scheduled_at = State(), [], None
    for t in range(0, seconds, step):
        now = NOW + t
        nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1", busy=True)]
        booting = False
        if gpus == 2:
            labeled_at, gpus_at = launched_at + 300, launched_at + 300 + plugin_delay
            booting = now < labeled_at
            if not booting:
                if waiting and scheduled_at is None and now > gpus_at:
                    scheduled_at, waiting = now, []
                nodes.append(gpu_node("k3s-gpu-a6000-2", busy=scheduled_at is not None,
                                      allocatable=now >= gpus_at))
        action, state = decide(now=now, pending=list(waiting), nodes=nodes, state=State.from_json(state.to_json()),
                               cpu_count=1, cpu_max=3, gpu_counts={"a6000": gpus}, pools=pools,
                               gpu_launching=booting)
        if action is None:
            continue
        actions.append((t, action.kind, action.target))
        if action.target > gpus:
            launched_at = now
        gpus = action.target
    return actions, state, scheduled_at


@pytest.mark.parametrize("idle_minutes", [1, 5])
@pytest.mark.parametrize("extra", [-180, -10, 20, 300])
def test_slow_device_plugin_does_not_get_its_node_scaled_away(idle_minutes, extra):
    # Plugin delays around idle_minutes after Ready: before round 3, a delay just past idle_minutes removed the node
    # at idle_minutes (rule 5, "idle"), marked the Pod stuck as a failed join, and relaunched an hour later.
    actions, state, scheduled_at = simulate_slow_device_plugin(max(idle_minutes * 60 + extra, 20), idle_minutes)
    assert [(kind, target) for _, kind, target in actions] == [("gpu_scale", 2)]
    assert scheduled_at is not None and state.stuck == {}


def test_launched_gpu_node_is_left_to_the_join_deadline_not_to_its_idle_clock():
    # Ready and labeled but no GPUs advertised, idle well past idle_minutes: rule 5 waits, rule 6 decides at the
    # deadline (with its own reason), so a node that registered gets the same GPU_JOIN_DEADLINE_S as one that did not.
    pools = {"a6000": pool(lo=1, hi=2, idle_minutes=5)}
    nodes = [*CLUSTER, gpu_node("k3s-gpu-a6000-2", allocatable=False)]
    launched = State(last_action_at=NOW - 1000, idle_since={"k3s-gpu-a6000-2": NOW - 700},
                     launches={"a6000": Launch(2, NOW - 1000, ("gpu-test/gpu-burn-x",))})
    action, state = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2}, state=launched)
    assert action is None and state.launches == launched.launches
    deadline = NOW - 1000 + GPU_JOIN_DEADLINE_S
    action, state = run(pending=[gpu_pending()], nodes=nodes, pools=pools, gpu_counts={"a6000": 2}, state=launched,
                        now=deadline)
    assert (action.kind, action.target) == ("gpu_scale", 1) and "did not join" in action.reason
    assert set(state.stuck) == {"gpu-test/gpu-burn-x"}


def test_launched_gpu_node_idle_clock_starts_when_it_can_take_gpu_pods():
    pools = {"a6000": pool(lo=1, hi=2, idle_minutes=5)}
    launched = State(last_action_at=NOW - 900, idle_since={"k3s-gpu-a6000-2": NOW - 600},
                     launches={"a6000": Launch(2, NOW - 900)})
    nodes = [*CLUSTER, gpu_node("k3s-gpu-a6000-2")]
    action, state = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2}, state=launched)
    assert action is None and state.launches == {}
    assert state.idle_since["k3s-gpu-a6000-2"] == NOW
    # From then on rule 5 as usual: idle_minutes after it could first take a GPU Pod.
    action, _ = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2}, state=state, now=NOW + 5 * 60)
    assert (action.kind, action.target) == ("gpu_scale", 1)


# --- a GPU node without GPUs that is not joining does not block other launches (review X2-1) -------------------------

def test_min_pool_node_without_allocatable_gpus_does_not_block_a_scale_up():
    # No launch record: the plugin or the GPU failed for good. A second node would serve the Pod.
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1", allocatable=False)]
    action, state = run(pending=[gpu_pending()], nodes=nodes,
                        state=State(idle_since={"k3s-gpu-a6000-1": NOW - 120}))
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a6000", 2)
    assert "k3s-gpu-a6000-1 has no allocatable nvidia.com/gpu (device plugin down?)" in state.held
    assert "yet" not in state.held


def test_fixed_pool_node_without_allocatable_gpus_does_not_block_an_autoscaled_pool():
    pools = {"a6000": pool(count=1, hi=2), "h100": pool("h100", count=1, autoscaled=False)}
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1", busy=True),
             gpu_node("k3s-gpu-h100-1", allocatable=False)]
    action, state = run(pending=[gpu_pending()], nodes=nodes, pools=pools, gpu_counts={"a6000": 1})
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a6000", 2)
    assert "k3s-gpu-h100-1 has no allocatable nvidia.com/gpu (device plugin down?)" in state.held


def test_only_the_joining_node_without_gpus_counts_as_launching():
    # The node a recorded launch is waiting for still blocks (bounded by rule 6); the same node without a launch
    # record does not.
    nodes = [*CLUSTER, gpu_node("k3s-gpu-a6000-2", allocatable=False)]
    common = dict(pending=[gpu_pending()], nodes=nodes, gpu_counts={"a6000": 2}, pools={"a6000": pool(hi=3)})
    action, state = run(state=State(launches={"a6000": Launch(2, NOW - 400)}), **common)
    assert action is None and "k3s-gpu-a6000-2 has no allocatable nvidia.com/gpu yet" in state.held
    action, state = run(**common)
    assert (action.kind, action.target) == ("gpu_scale", 3)
