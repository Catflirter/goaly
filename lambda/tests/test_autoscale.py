"""Autoscaler decision engine (stretch S2; design spec sections 10.2, 10.4, 10.5; Ruling SR6).

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
    COOLDOWN_S, CPU_IDLE_S, JOIN_GRACE_S, PENDING_GRACE_S, Action, NodeObs, PendingPod, State, decide,
    node_observations, pending_pods,
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


def node(name, ready=True, gpu_label=None, gpus=..., ready_at=NOW - 3600):
    """A node. A GPU node advertises one nvidia.com/gpu unless `gpus` says otherwise (None: the key is absent).
    `ready_at` is the Ready condition's lastTransitionTime (None: the field is absent)."""
    labels = {"kubernetes.io/hostname": name}
    if gpu_label is None:
        gpu_label = name.startswith("k3s-gpu-")
    if gpu_label:
        labels[GPU_LABEL] = "nvidia"
    ready_cond = {"type": "Ready", "status": "True" if ready else "False"}
    if ready_at is not None:
        ready_cond["lastTransitionTime"] = ts(ready_at)
    status = {"conditions": [ready_cond], "allocatable": {"cpu": "2", "memory": "4Gi"}}
    if gpus is ...:
        gpus = "1" if name.startswith("k3s-gpu-") else None
    if gpus is not None:
        status["allocatable"]["nvidia.com/gpu"] = gpus
    return {"metadata": {"name": name, "labels": labels}, "spec": {}, "status": status}


def items(*objs):
    return {"apiVersion": "v1", "kind": "List", "items": list(objs)}


def cpu_node(name, busy=False, ready=True):
    return NodeObs(name=name, kind="cpu", ready=ready, labeled=True, busy=busy)


def gpu_node(name, busy=False, ready=True, labeled=True, gpus=1, ready_since=NOW - 3600):
    return NodeObs(name=name, kind="gpu", ready=ready, labeled=labeled, busy=busy, gpu_allocatable=gpus,
                   ready_since=ready_since if ready else None)


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


def cpu_pending(since=NOW - 60, name="php-apache-x"):
    return PendingPod(namespace="scale-demo", name=name, resource="cpu", since=since)


def gpu_pending(since=NOW - 60, name="gpu-burn-x"):
    return PendingPod(namespace="gpu-test", name=name, resource="gpu", since=since)


def test_constants_match_spec_and_ruling_sr6():
    assert (PENDING_GRACE_S, CPU_IDLE_S, COOLDOWN_S, JOIN_GRACE_S) == (30, 300, 120, 300)


def test_rounds_1_to_3_machinery_is_gone():
    # Ruling SR6: no stuck-Pod gate, no launch rollback, no extra clocks or state.
    for name in ("STUCK_IDLE_S", "STUCK_TTL_S", "GPU_JOIN_DEADLINE_S", "CLOCK_SKEW_S", "Mark", "Launch"):
        assert not hasattr(autoscale, name)
    assert set(json.loads(State().to_json())) == {"last_action_at", "idle_since"}


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


def test_module_docstring_documents_the_known_limitation():
    # Ruling SR6: an unfittable Pending Pod drives scale-up to the cap, bounded by the cooldown.
    doc = autoscale.__doc__
    assert "Known limitation" in doc and "cooldown" in doc and "JOIN_GRACE_S" in doc


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
        NodeObs("k3s-cpu-2", "cpu", ready=True, labeled=True, busy=False, ready_since=NOW - 3600),
        NodeObs("k3s-gpu-a6000-1", "gpu", ready=True, labeled=True, busy=False, gpu_allocatable=1,
                ready_since=NOW - 3600),
    ]
    _, state = run(nodes=node_observations(nodes, pods), gpu_counts={"a6000": 1})
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
    assert [(o.name, o.kind, o.ready, o.labeled, o.busy) for o in node_observations(nodes, items())] == [
        ("k3s-cpu-1", "cpu", True, True, False),
        ("k3s-gpu-a6000-2", "gpu", False, False, False),
        ("k3s-gpu-a6000-1", "gpu", True, False, False),
        ("k3s-cpu-4", "cpu", False, True, False),
    ]


@pytest.mark.parametrize("gpus, expected", [("1", 1), ("2", 2), ("0", 0), (None, 0), ("", 0), ("lots", 0), (4, 4)])
def test_node_observations_read_allocatable_gpus(gpus, expected):
    # Ruling SR6: status.allocatable["nvidia.com/gpu"], 0 when absent (the device plugin has not registered yet).
    obs = node_observations(items(node("k3s-gpu-a6000-2", gpus=gpus), node("k3s-cpu-2")), items())
    assert [(o.name, o.gpu_allocatable) for o in obs] == [("k3s-gpu-a6000-2", expected), ("k3s-cpu-2", 0)]


def test_node_observations_read_ready_since():
    # Ruling SR6: the Ready condition's lastTransitionTime while Ready; None when NotReady or the time is missing.
    nodes = items(node("k3s-gpu-a6000-1", ready_at=NOW - 120), node("k3s-gpu-a6000-2", ready=False, ready_at=NOW - 5),
                  node("k3s-gpu-a6000-3", ready_at=None), node("k3s-cpu-2", ready_at=NOW - 30))
    assert [(o.name, o.ready_since) for o in node_observations(nodes, items())] == [
        ("k3s-gpu-a6000-1", NOW - 120), ("k3s-gpu-a6000-2", None), ("k3s-gpu-a6000-3", None), ("k3s-cpu-2", NOW - 30)]


@pytest.mark.parametrize("bad", [None, {}, {"kind": "Status", "message": "Unauthorized"}, {"items": None},
                                 {"items": {}}, {"items": ["not-a-pod"]}, []])
def test_node_observations_rejects_a_malformed_listing(bad):
    # Busyness guards every scale-down (Review Focus #5): a failed read must never look like "nothing runs anywhere".
    with pytest.raises(ValueError):
        node_observations(items(node("k3s-cpu-2")), bad)
    with pytest.raises(ValueError):
        node_observations(bad, items())


def test_node_observations_accepts_an_empty_listing():
    assert [o.name for o in node_observations(items(node("k3s-cpu-2")), items())] == ["k3s-cpu-2"]
    assert node_observations(items(), items()) == []


def test_node_obs_keeps_the_brief_signature():
    obs = NodeObs("k3s-cpu-2", "cpu", True, True, False)
    assert (obs.gpu_allocatable, obs.ready_since) == (0, None)


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
    # A file written by an earlier revision of this PR: the extra keys are ignored, the cooldown is kept.
    ('{"last_action_at": 1000.0, "stuck": {}, "launches": {"a6000": {"target": 2, "at": 1.0}}, '
     '"usable_idle_since": {}}', State(last_action_at=1000.0)),
])
def test_state_from_json_tolerates_missing_empty_or_unknown_fields(text, expected):
    assert State.from_json(text) == expected


@pytest.mark.parametrize("text", [
    "[]", "not json", '{"last_action_at": "soon"}', '{"last_action_at": true}', '{"idle_since": []}',
    '{"idle_since": {"k3s-cpu-2": "x"}}', '{"last_action_at": NaN}', '{"last_action_at": 1' + "0" * 400 + '}',
])
def test_state_from_json_rejects_garbage(text):
    # A corrupt file must not silently reset the cooldown (Review Focus #1): the caller sees the error.
    with pytest.raises(ValueError):
        State.from_json(text)


def test_held_reason_is_not_saved():
    s = State(last_action_at=1000.0, held="cooldown: 10s left")
    assert "held" not in json.loads(s.to_json())
    assert State.from_json(s.to_json()).held is None


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


# --- the clock (Ruling SR6: now < last_action_at is a cooldown, never an exception) ----------------------------------

def test_clock_behind_last_action_is_a_cooldown_not_an_error():
    state = State(last_action_at=NOW + 3600, idle_since={"k3s-cpu-2": NOW + 3600})
    nodes = [*CLUSTER, cpu_node("k3s-cpu-2")]
    action, new_state = run(pending=[cpu_pending()], nodes=nodes, cpu_count=2, state=state)
    assert action is None
    assert new_state.last_action_at == NOW + 3600
    assert "clock" in new_state.held and "scale-demo/php-apache-x" in new_state.held


def test_idle_clock_in_the_future_never_counts_as_idle():
    # Clock moved back, no cooldown pending: a node whose idle clock lies ahead of `now` is not idle yet.
    nodes = [*CLUSTER, cpu_node("k3s-cpu-2")]
    action, state = run(nodes=nodes, cpu_count=2, state=State(idle_since={"k3s-cpu-2": NOW + 600}))
    assert action is None and state.idle_since == {"k3s-cpu-2": NOW + 600}


# --- decide: the brief's rules 0-6 ----------------------------------------------------------------------------------

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
    # Rule 0 runs before rule 1: the idle clock keeps running through a cooldown.
    nodes = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2"), cpu_node("k3s-cpu-3", busy=True),
             cpu_node("k3s-cpu-4", ready=False), cpu_node("k3s-cpu-5")]
    state = State(last_action_at=NOW - 1, idle_since={"k3s-cpu-2": NOW - 500, "k3s-cpu-3": NOW - 500,
                                                       "k3s-cpu-4": NOW - 500, "k3s-cpu-9": NOW - 500})
    action, new_state = run(nodes=nodes, state=state, cpu_count=5)
    assert action is None
    assert new_state.idle_since == {"k3s-cpu-2": NOW - 500, "k3s-cpu-5": NOW}
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


def test_gpu_count_defaults_to_the_manifest_count():
    action, _ = run(pending=[gpu_pending()], gpu_counts={}, pools={"a6000": pool(count=1, hi=3)})
    assert (action.pool, action.target) == ("a6000", 2)


def test_no_second_gpu_launch_while_launching():
    # Review Focus #4: an instance still booting (reported by the caller) blocks another launch.
    action, state = run(pending=[gpu_pending()], gpu_launching=True)
    assert action is None and state.last_action_at is None
    assert "booting" in state.held


@pytest.mark.parametrize("new_node, words", [
    (gpu_node("k3s-gpu-a6000-2", ready=False, labeled=False, gpus=0), "NotReady"),  # joined, not Ready yet
    (gpu_node("k3s-gpu-a6000-2", ready=False), "NotReady"),           # labeled node gone NotReady (reboot)
    (gpu_node("k3s-gpu-a6000-2", labeled=False, gpus=0), "label"),     # Ready, admin label not applied yet
])
def test_gpu_node_not_ready_or_unlabeled_counts_as_launching(new_node, words):
    # Review Focus #4: NotReady or no admin label yet -> still launching.
    action, state = run(pending=[gpu_pending()], nodes=[*CLUSTER, new_node], gpu_counts={"a6000": 2},
                        pools={"a6000": pool(hi=3)})
    assert action is None
    assert "k3s-gpu-a6000-2" in state.held and words in state.held


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
    action, state = run(pending=[gpu_pending()], nodes=nodes, pools=pools)
    assert action is None and "no GPU pool has autoscale" in state.held


def test_priority_scale_up_before_scale_down():
    # Brief: the rules are checked in order and only the first match is returned.
    pools = {"a6000": pool(lo=0, hi=3)}
    long_idle = {n: NOW - 3600 for n in ("k3s-cpu-2", "k3s-gpu-a6000-2")}
    common = dict(pools=pools, gpu_counts={"a6000": 2}, cpu_count=2, state=State(idle_since=long_idle))
    both_idle = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2"), gpu_node("k3s-gpu-a6000-1", busy=True),
                 gpu_node("k3s-gpu-a6000-2")]
    action, _ = run(pending=[gpu_pending(), cpu_pending()], nodes=both_idle, **common)
    assert (action.kind, action.target) == ("gpu_scale", 3)  # rule 2 before rules 3, 4, 5
    action, _ = run(pending=[cpu_pending()], nodes=both_idle, **common)
    assert (action.kind, action.target) == ("cpu_scale", 3)  # rule 3 before rules 4, 5
    action, _ = run(nodes=both_idle, **common)
    assert action == Action("cpu_remove", node="k3s-cpu-2", reason=action.reason)  # rule 4 before rule 5
    gpu_idle = [cpu_node("k3s-cpu-1", busy=True), cpu_node("k3s-cpu-2", busy=True),
                gpu_node("k3s-gpu-a6000-1", busy=True), gpu_node("k3s-gpu-a6000-2")]
    action, _ = run(nodes=gpu_idle, **common)
    assert (action.kind, action.target) == ("gpu_scale", 1)


def test_every_action_carries_a_reason():
    nodes = [*CLUSTER, cpu_node("k3s-cpu-2")]
    gpu_idle = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1"), gpu_node("k3s-gpu-a6000-2")]
    actions = [
        run(pending=[cpu_pending()])[0],
        run(pending=[gpu_pending()])[0],
        run(nodes=nodes, cpu_count=2, state=State(idle_since={"k3s-cpu-2": NOW - CPU_IDLE_S}))[0],
        run(nodes=gpu_idle, gpu_counts={"a6000": 2}, state=State(idle_since={"k3s-gpu-a6000-2": NOW - 3600}))[0],
    ]
    assert all(a is not None and a.reason for a in actions)


def test_unlabeled_idle_gpu_node_gets_an_idle_clock_and_is_scaled_down():
    # Brief rule 0 has no label condition: a GPU node whose labeling step failed must not run (and bill) forever.
    pools = {"a6000": pool(lo=1, hi=2, idle_minutes=5)}
    nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1", busy=True),
             gpu_node("k3s-gpu-a6000-2", labeled=False, gpus=0)]
    _, state = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2},
                   state=State(idle_since={"k3s-gpu-a6000-2": NOW - 100}))
    assert state.idle_since["k3s-gpu-a6000-2"] == NOW - 100
    action, _ = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2},
                    state=State(idle_since={"k3s-gpu-a6000-2": NOW - 5 * 60}))
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a6000", 1)


def test_unfittable_pod_scales_to_the_cap_spaced_by_the_cooldown():
    # Known limitation (Ruling SR6): a Pod no new node can satisfy drives scale-up to cpu_max, one step per cooldown.
    # (Rule 4 removes the new idle workers CPU_IDLE_S later; the window below ends before that.)
    cpu_count, state, actions = 1, State(), []
    for t in range(0, CPU_IDLE_S, 20):
        nodes = [cpu_node(f"k3s-cpu-{i}", busy=i == 1) for i in range(1, cpu_count + 1)]
        action, state = run(pending=[cpu_pending()], nodes=nodes, cpu_count=cpu_count,
                            state=State.from_json(state.to_json()), now=NOW + t)
        if action is not None:
            actions.append((t, action.kind, action.target))
            cpu_count = action.target
    assert actions == [(0, "cpu_scale", 2), (COOLDOWN_S, "cpu_scale", 3)]
    assert "cpu_max 3" in state.held


# --- held: why a scale-up that Pending Pods asked for was withheld ---------------------------------------------------

@pytest.mark.parametrize("kwargs, words", [
    (dict(pending=[gpu_pending()], gpu_launching=True), "booting"),
    (dict(pending=[gpu_pending()], nodes=[*CLUSTER, gpu_node("k3s-gpu-a6000-2", ready=False)],
          gpu_counts={"a6000": 2}, pools={"a6000": pool(hi=3)}), "k3s-gpu-a6000-2 is NotReady"),
    (dict(pending=[gpu_pending()], nodes=[*CLUSTER, gpu_node("k3s-gpu-a6000-2", labeled=False)],
          gpu_counts={"a6000": 2}, pools={"a6000": pool(hi=3)}), "k3s-gpu-a6000-2 has no"),
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


# --- the joining window (Ruling SR6, Review Focus #4) ----------------------------------------------------------------

def joining_node(age, busy=False, name="k3s-gpu-a6000-2", gpus=0):
    """Ready and labeled `age` seconds ago, no nvidia.com/gpu advertised (the device plugin has not registered)."""
    return gpu_node(name, busy=busy, gpus=gpus, ready_since=NOW - age)


def test_joining_node_inside_the_window_counts_as_launching():
    pools = {"a6000": pool(hi=3)}
    nodes = [*CLUSTER, joining_node(JOIN_GRACE_S - 1)]
    action, state = run(pending=[gpu_pending()], nodes=nodes, gpu_counts={"a6000": 2}, pools=pools)
    assert action is None and state.last_action_at is None
    assert "k3s-gpu-a6000-2" in state.held and "joining" in state.held and "broken" not in state.held


def test_joining_node_is_never_idle_inside_the_window():
    # Rule 5 must not scale away a node whose device plugin is still starting, whatever idle_minutes says.
    pools = {"a6000": pool(lo=1, hi=2, idle_minutes=1)}
    nodes = [*CLUSTER, joining_node(JOIN_GRACE_S - 1)]
    old_clock = State(idle_since={"k3s-gpu-a6000-2": NOW - 3600})
    action, state = run(nodes=nodes, pools=pools, gpu_counts={"a6000": 2}, state=old_clock)
    assert action is None and "k3s-gpu-a6000-2" not in state.idle_since


def test_joining_node_past_the_window_is_broken_not_launching():
    pools = {"a6000": pool(hi=3)}
    nodes = [*CLUSTER, joining_node(JOIN_GRACE_S)]
    action, state = run(pending=[gpu_pending()], nodes=nodes, gpu_counts={"a6000": 2}, pools=pools)
    assert (action.kind, action.pool, action.target) == ("gpu_scale", "a6000", 3)
    assert "k3s-gpu-a6000-2" in state.held and "broken" in state.held and "joining" not in state.held


def test_broken_node_gets_an_idle_clock_after_the_window_and_is_scaled_down():
    # Past the window the node is an ordinary Ready, idle node to rules 0 and 5: it does not bill forever.
    pools = {"a6000": pool(lo=1, hi=2, idle_minutes=5)}
    common = dict(pools=pools, gpu_counts={"a6000": 2})
    _, state = run(nodes=[*CLUSTER, joining_node(JOIN_GRACE_S)], **common)
    assert state.idle_since == {"k3s-gpu-a6000-2": NOW}
    later = NOW + 5 * 60
    action, _ = run(nodes=[*CLUSTER, joining_node(JOIN_GRACE_S)], state=state, now=later, **common)
    assert (action.kind, action.target) == ("gpu_scale", 1) and "k3s-gpu-a6000-2" in action.reason


def test_ready_node_with_unknown_ready_time_and_no_gpus_is_broken_not_launching():
    # No lastTransitionTime: the wait could not be bounded, so it does not hold rule 2.
    nodes = [*CLUSTER, gpu_node("k3s-gpu-a6000-2", gpus=0, ready_since=None)]
    action, state = run(pending=[gpu_pending()], nodes=nodes, gpu_counts={"a6000": 2}, pools={"a6000": pool(hi=3)})
    assert (action.kind, action.target) == ("gpu_scale", 3) and "broken" in state.held


def test_node_with_gpus_is_not_joining_however_young():
    nodes = [*CLUSTER, joining_node(10, gpus=1)]
    action, state = run(pending=[gpu_pending()], nodes=nodes, gpu_counts={"a6000": 2}, pools={"a6000": pool(hi=3)})
    assert (action.kind, action.target) == ("gpu_scale", 3) and "joining" not in (state.held or "")
    _, state = run(nodes=nodes, gpu_counts={"a6000": 2}, pools={"a6000": pool(hi=3)})
    assert state.idle_since["k3s-gpu-a6000-2"] == NOW


@pytest.mark.parametrize("pools, counts, name", [
    # A node of a fixed pool (make up / gpu-up), and one added by hand to an autoscaled pool (gpuctl scale).
    ({"a6000": pool(hi=3), "h100": pool("h100", autoscaled=False)}, {"a6000": 1, "h100": 1}, "k3s-gpu-h100-1"),
    ({"a6000": pool(hi=3)}, {"a6000": 2}, "k3s-gpu-a6000-2"),
])
def test_joining_window_covers_nodes_the_autoscaler_did_not_launch(pools, counts, name):
    nodes = [*CLUSTER, joining_node(60, name=name)]
    action, state = run(pending=[gpu_pending()], nodes=nodes, gpu_counts=counts, pools=pools)
    assert action is None and name in state.held and "joining" in state.held


def test_min_pool_node_restarting_holds_rule_2_until_its_gpus_are_back():
    # k3s-gpu-a6000-1 (the pool's min node) reboots: NotReady, then Ready with its device plugin starting.
    pools = {"a6000": pool(lo=1, hi=2)}
    common = dict(pending=[gpu_pending()], pools=pools, gpu_counts={"a6000": 1})
    cpu = cpu_node("k3s-cpu-1", busy=True)
    action, state = run(nodes=[cpu, gpu_node("k3s-gpu-a6000-1", ready=False)], **common)
    assert action is None and "NotReady" in state.held
    action, state = run(nodes=[cpu, joining_node(40, name="k3s-gpu-a6000-1")], **common)
    assert action is None and "joining" in state.held
    # GPUs advertised again: the Pending Pod lands on it, no second instance (the next snapshot shows it busy).
    action, _ = run(nodes=[cpu, joining_node(60, name="k3s-gpu-a6000-1", gpus=1, busy=True)],
                    pools=pools, gpu_counts={"a6000": 1})
    assert action is None


def simulate_slow_device_plugin(plugin_delay, idle_minutes, *, seconds=3600, step=20):
    """Pool a6000 min 1 max 2, k3s-gpu-a6000-1 busy, one 1-GPU Pod Pending. The new instance boots for 300 s (the
    caller reports it as gpu_launching), registers Ready and labeled, its device plugin advertises the GPU
    `plugin_delay` s later, and the scheduler binds the Pod on the tick after that."""
    pools = {"a6000": pool(lo=1, hi=2, idle_minutes=idle_minutes)}
    waiting = [PendingPod("gpu-test", "train", "gpu", NOW - 60)]
    launched_at, gpus, state, actions, scheduled_at = None, 1, State(), [], None
    for t in range(0, seconds, step):
        now = NOW + t
        nodes = [cpu_node("k3s-cpu-1", busy=True), gpu_node("k3s-gpu-a6000-1", busy=True)]
        booting = False
        if gpus == 2:
            ready_at, gpus_at = launched_at + 300, launched_at + 300 + plugin_delay
            booting = now < ready_at
            if not booting:
                if waiting and scheduled_at is None and now > gpus_at:
                    scheduled_at, waiting = now, []
                nodes.append(NodeObs("k3s-gpu-a6000-2", "gpu", True, True, busy=scheduled_at is not None,
                                     gpu_allocatable=int(now >= gpus_at), ready_since=ready_at))
        action, state = decide(now=now, pending=list(waiting), nodes=nodes, state=State.from_json(state.to_json()),
                               cpu_count=1, cpu_max=3, gpu_counts={"a6000": gpus}, pools=pools,
                               gpu_launching=booting)
        if action is not None:
            actions.append((t, action.kind, action.target))
            if action.target > gpus:
                launched_at = now
            gpus = action.target
    return actions, scheduled_at


@pytest.mark.parametrize("idle_minutes", [1, 5])
@pytest.mark.parametrize("plugin_delay", [20, 100, 200, JOIN_GRACE_S - 20])
def test_slow_device_plugin_inside_the_window_keeps_its_node_and_gets_the_pod(idle_minutes, plugin_delay):
    actions, scheduled_at = simulate_slow_device_plugin(plugin_delay, idle_minutes)
    assert [(kind, target) for _, kind, target in actions] == [("gpu_scale", 2)]
    assert scheduled_at is not None
