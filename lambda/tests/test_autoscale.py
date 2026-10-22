"""Autoscaler decision engine (stretch S2; design spec sections 10.2, 10.4, 10.5).

The fixtures build kubectl-JSON-shaped dicts (`kubectl get pods -A -o json`, `kubectl get nodes -o json`); every
decision is a pure function of them, the stored State and `now`.
"""
import json
from datetime import datetime, timezone

import pytest

from gpuctl import autoscale
from gpuctl.autoscale import (
    COOLDOWN_S, CPU_IDLE_S, PENDING_GRACE_S, Action, NodeObs, PendingPod, State, decide, node_observations,
    pending_pods,
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


def node(name, ready=True, gpu_label=None):
    labels = {"kubernetes.io/hostname": name}
    if gpu_label is None:
        gpu_label = name.startswith("k3s-gpu-")
    if gpu_label:
        labels[GPU_LABEL] = "nvidia"
    return {"metadata": {"name": name, "labels": labels},
            "status": {"conditions": [{"type": "Ready", "status": "True" if ready else "False"}]}}


def items(*objs):
    return {"apiVersion": "v1", "kind": "List", "items": list(objs)}


def cpu_node(name, busy=False, ready=True):
    return NodeObs(name=name, kind="cpu", ready=ready, labeled=True, busy=busy)


def gpu_node(name, busy=False, ready=True, labeled=True):
    return NodeObs(name=name, kind="gpu", ready=ready, labeled=labeled, busy=busy)


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


def test_module_does_no_io():
    source = open(autoscale.__file__, encoding="utf-8").read()
    for forbidden in ("import subprocess", "import requests", "import time", "time.time", "open("):
        assert forbidden not in source


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


# --- State ----------------------------------------------------------------------------------------------------------

def test_state_roundtrip_preserves_cooldown():
    # Review Focus #1: the cooldown survives a restart through .run/autoscaler-state.json.
    s = State(last_action_at=1000.0, idle_since={"k3s-cpu-2": 900.0})
    assert State.from_json(s.to_json()) == s
    assert json.loads(s.to_json()) == {"last_action_at": 1000.0, "idle_since": {"k3s-cpu-2": 900.0}}


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
    '{"idle_since": {"k3s-cpu-2": "x"}}', '{"last_action_at": NaN}',
])
def test_state_from_json_rejects_garbage(text):
    # A corrupt file must not silently reset the cooldown: the caller sees the error.
    with pytest.raises(ValueError):
        State.from_json(text)


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
    nodes = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2"), gpu_node("k3s-gpu-a6000-1"),
             gpu_node("k3s-gpu-a6000-2")]
    idle = State(idle_since={n.name: NOW - 3600 for n in nodes})
    pools = {"a6000": pool(lo=0, hi=3)}
    common = dict(nodes=nodes, pools=pools, gpu_counts={"a6000": 2}, cpu_count=2, state=idle)
    assert run(pending=[gpu_pending(), cpu_pending()], **common)[0].kind == "gpu_scale"
    assert run(pending=[gpu_pending(), cpu_pending()], **common)[0].target == 3
    assert run(pending=[cpu_pending()], **common)[0].kind == "cpu_scale"
    assert run(**common)[0] == Action("cpu_remove", node="k3s-cpu-2", reason=run(**common)[0].reason)
    nodes_no_idle_cpu = [n for n in nodes if n.name != "k3s-cpu-2"]
    action, _ = run(**{**common, "nodes": nodes_no_idle_cpu, "cpu_count": 1})
    assert (action.kind, action.target) == ("gpu_scale", 1)


def test_every_action_carries_a_reason():
    nodes = [*CLUSTER, cpu_node("k3s-cpu-2")]
    actions = [
        run(pending=[cpu_pending()])[0],
        run(pending=[gpu_pending()])[0],
        run(nodes=nodes, cpu_count=2, state=State(idle_since={"k3s-cpu-2": NOW - CPU_IDLE_S}))[0],
    ]
    assert all(a is not None and a.reason for a in actions)
