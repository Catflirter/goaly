"""Offline tests for stretch S4: the operator-side autoscaler loop scripts/autoscaler.py (design spec sections 10.2,
10.4, 10.5; Rulings SR6, SR7 and the ledger's 'Notes for S4').

Most tests run autoscaler.main() in-process with an injected clock and a fake command runner (no kubectl, gpuctl or
cpu-pool.sh ever runs); a few run the script itself against stub tools (fake_autoscaler_tools.py) under a temporary
AUTOSCALER_ROOT: the brief's --once --dry-run fixture tests, the real subprocess wiring and Ctrl-C.
"""
import fcntl
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "autoscaler.py"
FAKE = Path(__file__).resolve().parent / "fake_autoscaler_tools.py"
NOW = 1_800_000_000.0
GPU_LABEL = "node-restriction.kubernetes.io/accelerator"
CPU_MSG = "0/2 nodes are available: 2 Insufficient cpu. preemption: 0/2 nodes are available: 2 No preemption victims"
GPU_MSG = "0/3 nodes are available: 1 Insufficient nvidia.com/gpu, 2 node(s) didn't match Pod's node affinity/selector."


def load_module():
    spec = importlib.util.spec_from_file_location("autoscaler_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def autoscaler():
    return load_module()


# --- kubectl-shaped fixtures -------------------------------------------------------------------

def ts(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pod(name, namespace="scale-demo", node=None, phase="Running", unschedulable=None, since=NOW - 60, gpu=0,
        owner=None):
    resources = {"limits": {"nvidia.com/gpu": str(gpu)}} if gpu else {"requests": {"cpu": "600m"}}
    status = {"phase": phase}
    if unschedulable is not None:
        status = {"phase": "Pending", "conditions": [
            {"type": "PodScheduled", "status": "False", "reason": "Unschedulable", "message": unschedulable,
             "lastTransitionTime": ts(since)}]}
    meta = {"name": name, "namespace": namespace}
    if owner:
        meta["ownerReferences"] = [{"kind": owner, "name": "o", "controller": True}]
    spec = {"containers": [{"name": "c", "image": "x", "resources": resources}]}
    if node:
        spec["nodeName"] = node
    return {"metadata": meta, "spec": spec, "status": status}


def node(name, ready=True, labeled=None, gpus=..., ready_at=NOW - 3600):
    labels = {"kubernetes.io/hostname": name}
    if (name.startswith("k3s-gpu-") if labeled is None else labeled):
        labels[GPU_LABEL] = "nvidia"
    cond = {"type": "Ready", "status": "True" if ready else "False", "lastTransitionTime": ts(ready_at)}
    status = {"conditions": [cond], "allocatable": {"cpu": "2", "memory": "2Gi"}}
    if gpus is ...:
        gpus = "1" if name.startswith("k3s-gpu-") else None
    if gpus is not None:
        status["allocatable"]["nvidia.com/gpu"] = gpus
    return {"metadata": {"name": name, "labels": labels}, "spec": {}, "status": status}


def klist(*objs):
    return {"apiVersion": "v1", "kind": "List", "items": list(objs)}


CP = node("k3s-cp")
CPU1 = node("k3s-cpu-1")
CPU_PENDING = pod("cpu-hog-1", unschedulable=CPU_MSG)
GPU_PENDING = pod("gpu-burn-2", unschedulable=GPU_MSG, gpu=1)
GPU1_BUSY = pod("gpu-burn-1", node="k3s-gpu-a6000-1", gpu=1)
INSTANCE_1 = "k3s-gpu-a6000-1 129.80.1.2 active gpu_1x_a6000 us-west-1"


# --- in-process harness ------------------------------------------------------------------------

def make_root(tmp_path, workers=("k3s-cpu-1",), overrides=None):
    root = tmp_path / "root"
    (root / "lambda").mkdir(parents=True)
    shutil.copy(REPO / "lambda" / "gpu-nodes.yaml", root / "lambda" / "gpu-nodes.yaml")
    (root / "infra" / "workers").mkdir(parents=True)
    inventory = {"cpu_workers": {w: {"zone": "us-central1-a", "machine_type": "e2-small"} for w in workers}}
    (root / "infra" / "workers" / "workers.auto.tfvars.json").write_text(json.dumps(inventory))
    if overrides is not None:
        (root / "lambda" / ".autoscale-state.json").write_text(json.dumps({"pools": overrides}))
    return root


def lock_is_free(path):
    """Whether another process could take gpuctl's write lock now (a separate open file description)."""
    if not path.is_file():
        return True
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def hold_lock(path):
    """make gpu-down (or a make gpu-apply) running in another terminal: it holds gpuctl's write lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


class Harness:
    """autoscaler.main() with a fixed clock, a fake runner and a fake wait; records every command."""

    def __init__(self, module, tmp_path, *, pods=(), nodes=(CP, CPU1), instances=(), state=None, rc=None,
                 duration=5.0, **root_kw):
        self.module = module
        self.root = make_root(tmp_path, **root_kw)
        self.state_path = self.root / ".run" / "autoscaler-state.json"
        if state is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(state if isinstance(state, str) else json.dumps(state))
        self.pods, self.nodes = klist(*pods), klist(*nodes)
        self.instances = list(instances)
        self.rc = dict(rc or {})  # tool -> rc or [rc, ...] (successive calls; the last repeats)
        self.duration = duration
        self.now = NOW
        self.calls = []
        self.out = io.StringIO()
        self.log_dir = tmp_path / "log"
        self.hooks = {}  # tool -> fn(harness), run when that tool is called (it may raise KeyboardInterrupt)
        self.on_wait = None  # fn(harness, tick) between the ticks of a --loop run
        self.stopper = None  # a real Stopper whose handler a test calls (a Ctrl-C without a signal)

    @property
    def overrides_path(self):
        return self.root / "lambda" / ".autoscale-state.json"

    @property
    def lock_path(self):
        return self.root / ".run" / "gpuctl.lock"

    # the fake runner -------------------------------------------------------------------------
    def tool(self, argv):
        name = Path(argv[0]).name
        if name == "gpuctl":
            return f"gpuctl-{argv[1]}" if argv[1] in ("plan", "apply", "list-nodes") else "gpuctl"
        return {"cpu-pool.sh": "cpu-pool", "kubectl": "kubectl"}.get(name, name)

    def next_rc(self, tool):
        value = self.rc.get(tool, 0)
        if isinstance(value, list):
            return value.pop(0) if len(value) > 1 else value[0]
        return value

    def run(self, argv, *, env=None, stream=False, pass_fds=()):
        tool = self.tool(list(argv))
        overrides = self.root / "lambda" / ".autoscale-state.json"
        self.calls.append({"tool": tool, "argv": list(argv), "env": dict(env or {}), "stream": stream,
                           "pass_fds": tuple(pass_fds), "lock_free": lock_is_free(self.lock_path),
                           "state": self.state_path.read_text() if self.state_path.exists() else None,
                           "overrides": overrides.read_text() if overrides.exists() else None})
        result = self.module.Result
        if tool in self.hooks:
            self.hooks[tool](self)
        if tool == "kubectl":
            rc = self.next_rc("kubectl")
            if rc:
                return result(rc, "", "error: You must be logged in to the server (Unauthorized)\n")
            doc = self.pods if "pods" in argv else self.nodes
            return result(0, doc if isinstance(doc, str) else json.dumps(doc), "")
        if tool == "gpuctl-list-nodes":
            rc = self.next_rc(tool)
            if rc:
                return result(rc, "", "gpuctl: Lambda API error: 503\n")
            return result(0, "".join(line + "\n" for line in self.instances), "")
        rc = self.next_rc(tool)
        if tool in ("cpu-pool", "gpuctl-apply"):
            self.now += self.duration
        err = {"gpuctl-plan": "gpuctl: plan refused: budget_usd_per_hour: plan costs $3.20/h, budget is $3.00/h\n"}
        return result(rc, f"{tool} output\n", err.get(tool, "") if rc else "")

    def of(self, tool):
        return [c for c in self.calls if c["tool"] == tool]

    def acted(self):
        return [c for c in self.calls if c["tool"] in ("cpu-pool", "gpuctl-apply")]

    # running ----------------------------------------------------------------------------------
    def main(self, *args, ticks=1, interval=20.0, env=None):
        waits = iter([False] * (ticks - 1) + [True])
        done = iter(range(1, ticks + 1))

        def wait(seconds):
            self.now += seconds
            if self.on_wait is not None:
                self.on_wait(self, next(done))
            return next(waits)

        deps = self.module.Deps(clock=lambda: self.now, run=self.run, wait=wait, out=self.out, stopper=self.stopper)
        full_env = {"AUTOSCALER_ROOT": str(self.root), **(env or {})}
        argv = list(args) + ["--log", str(self.log_dir)]
        if ticks == 1 and "--loop" not in args:
            argv.insert(0, "--once")
        else:
            argv = [a for a in argv if a != "--loop"] + ["--interval", str(interval)]
        return self.module.main(argv, env=full_env, deps=deps)

    def log(self):
        path = self.log_dir / "autoscaler.log"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def state(self):
        return json.loads(self.state_path.read_text())

    def overrides(self):
        path = self.root / "lambda" / ".autoscale-state.json"
        return json.loads(path.read_text()) if path.exists() else None


@pytest.fixture
def harness(autoscaler, tmp_path):
    def make(**kw):
        return Harness(autoscaler, tmp_path, **kw)
    return make


# --- the brief's --once --dry-run fixture tests (subprocess) ------------------------------------

def stub_root(tmp_path, scenario, **root_kw):
    """A root for the real script: stub scripts/gpuctl, scripts/cpu-pool.sh and kubectl on PATH."""
    root = make_root(tmp_path, **root_kw)
    (root / "scripts").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for path, tool in ((root / "scripts" / "gpuctl", "gpuctl"), (root / "scripts" / "cpu-pool.sh", "cpu-pool"),
                       (bin_dir / "kubectl", "kubectl")):
        path.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{FAKE}" {tool} "$@"\n')
        path.chmod(0o755)
    (tmp_path / "scenario.json").write_text(json.dumps(scenario))
    return root, bin_dir


def script_env(tmp_path, root, bin_dir, now=NOW):
    env = {k: v for k, v in os.environ.items() if k not in ("KUBECONFIG", "LAMBDA_API_KEY")}
    env.update({"PATH": f"{bin_dir}{os.pathsep}{env['PATH']}", "HOME": str(tmp_path / "home"),
                "AUTOSCALER_ROOT": str(root), "AUTOSCALER_STATE": str(tmp_path / "state.json"),
                "FAKE_SCENARIO": str(tmp_path / "scenario.json"), "FAKE_LOG": str(tmp_path / "calls.jsonl")})
    if now is not None:
        env["AUTOSCALER_NOW"] = str(now)
    return env


def run_autoscaler(tmp_path, *args, scenario=None, now=NOW, **root_kw):
    scenario = scenario or {"pods": klist(), "nodes": klist(CP, CPU1)}
    root, bin_dir = stub_root(tmp_path, scenario, **root_kw)
    return subprocess.run([sys.executable, str(SCRIPT), *args], env=script_env(tmp_path, root, bin_dir, now),
                          capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)


def fake_calls(tmp_path):
    path = tmp_path / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def fixture_files(tmp_path, pods, nodes):
    (tmp_path / "pods.json").write_text(json.dumps(pods))
    (tmp_path / "nodes.json").write_text(json.dumps(nodes))
    return str(tmp_path / "pods.json"), str(tmp_path / "nodes.json")


def test_dry_run_prints_cpu_scale(tmp_path):
    pods, nodes = fixture_files(tmp_path, klist(CPU_PENDING), klist(CP, CPU1))
    r = run_autoscaler(tmp_path, "--once", "--dry-run", "--pods-file", pods, "--nodes-file", nodes)
    assert r.returncode == 0 and "cpu_scale target=2" in r.stdout, r.stdout + r.stderr
    tools = [c["tool"] for c in fake_calls(tmp_path)]
    assert "kubectl" not in tools and "cpu-pool" not in tools  # files instead of kubectl; dry-run executes nothing
    assert not (tmp_path / "state.json").exists()  # a dry run saves no state (it would start a real cooldown)


def test_dry_run_respects_persisted_cooldown(tmp_path):  # Review Focus #1
    (tmp_path / "state.json").write_text(json.dumps({"last_action_at": NOW - 10, "idle_since": {}}))
    pods, nodes = fixture_files(tmp_path, klist(CPU_PENDING), klist(CP, CPU1))
    r = run_autoscaler(tmp_path, "--once", "--dry-run", "--pods-file", pods, "--nodes-file", nodes)
    assert r.returncode == 0 and "no action (cooldown)" in r.stdout, r.stdout + r.stderr
    assert "cpu_scale" not in r.stdout


def test_once_runs_guarded_cpu_pool_with_auto_approve(tmp_path):
    scenario = {"pods": klist(CPU_PENDING), "nodes": klist(CP, CPU1)}
    r = run_autoscaler(tmp_path, "--once", "--log", str(tmp_path / "log"), scenario=scenario)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = fake_calls(tmp_path)
    assert [c["tool"] for c in calls] == ["kubectl", "kubectl", "gpuctl", "cpu-pool"]
    assert calls[0]["argv"][-4:] == ["get", "pods", "-A", "-o", "json"][-4:] and "pods" in calls[0]["argv"]
    assert "--kubeconfig" in calls[0]["argv"]  # never an ambient KUBECONFIG (kc in scripts/lib/common.sh)
    assert calls[2]["argv"] == ["list-nodes"]
    assert calls[3]["argv"] == ["scale", "2", "--guarded"] and calls[3]["auto_approve"] == "1"
    # Review Focus #1: the cooldown is on disk before the action starts
    assert json.loads(calls[3]["state"])["last_action_at"] == NOW
    (line,) = [json.loads(x) for x in (tmp_path / "log" / "autoscaler.log").read_text().splitlines()
               if json.loads(x)["action"] == "cpu_scale"]
    assert {"ts", "action", "reason", "result", "duration_s"} <= set(line) and line["result"] == "ok"
    assert "fake cpu-pool scale 2 --guarded" in r.stdout  # the action's output reaches the operator


def test_gpu_apply_subprocess_inherits_the_lock(tmp_path):  # review r2-1: the real Popen pass_fds wiring
    scenario = {"pods": klist(GPU_PENDING, GPU1_BUSY), "nodes": klist(CP, CPU1, node("k3s-gpu-a6000-1")),
                "instances": [INSTANCE_1]}
    r = run_autoscaler(tmp_path, "--once", "--log", str(tmp_path / "log"), scenario=scenario)
    assert r.returncode == 0, r.stdout + r.stderr
    (apply,) = [c for c in fake_calls(tmp_path) if c["tool"] == "gpuctl-apply"]
    assert apply["lock_inherited"] is True


def test_dry_run_with_kubectl_reads_the_cluster_and_changes_nothing(tmp_path):
    scenario = {"pods": klist(GPU_PENDING, GPU1_BUSY), "nodes": klist(CP, CPU1, node("k3s-gpu-a6000-1")),
                "instances": [INSTANCE_1]}
    r = run_autoscaler(tmp_path, "--once", "--dry-run", scenario=scenario)
    assert r.returncode == 0 and "gpu_scale pool=a6000 target=2" in r.stdout, r.stdout + r.stderr
    tools = [c["tool"] for c in fake_calls(tmp_path)]
    assert tools == ["kubectl", "kubectl", "gpuctl", "gpuctl-plan"]  # the budget check is read-only
    assert not (tmp_path / "root" / "lambda" / ".autoscale-state.json").exists()


def test_pods_file_and_nodes_file_go_together(tmp_path):
    pods, _ = fixture_files(tmp_path, klist(), klist())
    r = run_autoscaler(tmp_path, "--once", "--dry-run", "--pods-file", pods)
    assert r.returncode == 2 and "--nodes-file" in r.stderr


# --- decisions to actions (in-process) ----------------------------------------------------------

def test_cpu_scale_then_restart_inside_cooldown_does_not_act(harness):  # Review Focus #1
    h = harness(pods=[CPU_PENDING])
    assert h.main() == 0
    (call,) = h.acted()
    assert call["argv"][1:] == ["scale", "2", "--guarded"] and call["env"] == {"AUTO_APPROVE": "1"}
    assert call["stream"] is True
    assert h.state()["last_action_at"] == NOW + 5  # cooldown counts from the end of the action
    h.now = NOW + 60  # restarted a minute later: the Pod is still Pending
    assert h.main() == 0
    assert len(h.acted()) == 1
    assert "no action (cooldown)" in h.out.getvalue()


def test_cpu_remove_runs_guarded_remove_for_the_idle_worker(harness):
    h = harness(nodes=[CP, CPU1, node("k3s-cpu-2")], workers=("k3s-cpu-1", "k3s-cpu-2"),
                state={"last_action_at": None, "idle_since": {"k3s-cpu-1": NOW - 900, "k3s-cpu-2": NOW - 400}})
    assert h.main() == 0
    (call,) = h.acted()
    assert call["argv"][1:] == ["remove", "k3s-cpu-2", "--guarded"] and call["env"] == {"AUTO_APPROVE": "1"}
    (line,) = [x for x in h.log() if x["action"] == "cpu_remove"]
    assert line["node"] == "k3s-cpu-2" and "idle" in line["reason"] and line["result"] == "ok"


def test_cpu_count_comes_from_the_workers_inventory(harness):
    h = harness(pods=[CPU_PENDING], workers=("k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-3"),
                nodes=[CP, CPU1, node("k3s-cpu-2"), node("k3s-cpu-3")])
    assert h.main() == 0
    assert h.acted() == []  # cpu_max 3 reached
    assert "cpu_max 3" in h.out.getvalue()
    assert h.main("--cpu-max", "4") == 0
    assert h.acted()[0]["argv"][1:] == ["scale", "4", "--guarded"]


def test_cpu_pool_exit_1_is_a_failed_action_and_starts_the_cooldown(harness):
    h = harness(pods=[CPU_PENDING], rc={"cpu-pool": 1})
    assert h.main() == 1
    (line,) = [x for x in h.log() if x["action"] == "cpu_scale"]
    assert line["result"].startswith("failed (exit 1)")
    assert h.state()["last_action_at"] == NOW + 5
    h.now += 30
    h.main()
    assert len(h.acted()) == 1  # repeated only after the cooldown
    h.now = NOW + 5 + 120
    h.main()
    assert len(h.acted()) == 2


def test_cpu_pool_exit_2_retries_next_tick_without_cooldown(harness):
    h = harness(pods=[CPU_PENDING], rc={"cpu-pool": [2, 0]})
    assert h.main("--loop", ticks=2) == 0
    assert len(h.acted()) == 2  # refused (lock held) -> retried at the next tick
    first, second = [x for x in h.log() if x["action"] == "cpu_scale"]
    assert first["result"].startswith("refused (exit 2)") and "retry next tick" in first["result"]
    assert second["result"] == "ok"


def test_repeated_exit_2_falls_back_to_the_cooldown(harness, autoscaler):
    h = harness(pods=[CPU_PENDING], rc={"cpu-pool": [2]})
    h.main("--loop", ticks=autoscaler.REFUSED_RETRIES + 2)
    # REFUSED_RETRIES refusals in a row, then the cooldown: never a loop on errors without it
    assert len(h.acted()) == autoscaler.REFUSED_RETRIES
    results = [x["result"] for x in h.log() if x["action"] == "cpu_scale"]
    assert "cooldown" in results[-1]
    assert h.state()["last_action_at"] is not None


def test_refused_action_keeps_the_previous_cooldown_time(harness):
    h = harness(pods=[CPU_PENDING], rc={"cpu-pool": 2}, state={"last_action_at": NOW - 500, "idle_since": {}})
    h.main()
    assert h.state()["last_action_at"] == NOW - 500
    assert json.loads(h.acted()[0]["state"])["last_action_at"] == NOW  # still saved before the action ran


# --- GPU actions ---------------------------------------------------------------------------------

GPU_CLUSTER = dict(pods=[GPU_PENDING, GPU1_BUSY], nodes=[CP, CPU1, node("k3s-gpu-a6000-1")], instances=[INSTANCE_1])


def test_gpu_scale_checks_budget_then_writes_override_then_applies(harness):
    h = harness(**GPU_CLUSTER, overrides={"other": 0})
    assert h.main() == 0
    tools = [c["tool"] for c in h.calls]
    # the second list-nodes re-checks the instances under gpuctl's lock, right before the write (review r2-1)
    assert tools == ["kubectl", "kubectl", "gpuctl-list-nodes", "gpuctl-plan", "gpuctl-list-nodes", "gpuctl-apply"]
    plan, apply = h.of("gpuctl-plan")[0], h.of("gpuctl-apply")[0]
    assert plan["argv"][1:] == ["plan", "--set-count", "a6000=2"]
    assert json.loads(plan["overrides"]) == {"pools": {"other": 0}}  # nothing written before the check passed
    assert apply["argv"][1:] == ["apply", "--auto-approve"]
    assert json.loads(apply["overrides"]) == {"pools": {"a6000": 2, "other": 0}}  # merged
    assert h.overrides() == {"pools": {"a6000": 2, "other": 0}}
    (line,) = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert line["pool"] == "a6000" and line["target"] == 2 and line["result"] == "ok"


def test_gpu_scale_over_budget_is_refused_and_cpu_rules_still_run(harness):
    h = harness(**{**GPU_CLUSTER, "pods": [GPU_PENDING, GPU1_BUSY, CPU_PENDING]}, rc={"gpuctl-plan": 2})
    h.main()
    assert h.of("gpuctl-apply") == [] and h.overrides() is None
    (gpu,) = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert gpu["result"].startswith("refused") and "budget_usd_per_hour" in gpu["result"]
    (cpu,) = [x for x in h.log() if x["action"] == "cpu_scale"]
    assert cpu["result"] == "ok"


def test_failed_gpu_apply_restores_the_override_file(harness):
    h = harness(**GPU_CLUSTER, overrides={"a6000": 1}, rc={"gpuctl-apply": 1})
    assert h.main() == 1
    assert json.loads(h.of("gpuctl-apply")[0]["overrides"]) == {"pools": {"a6000": 2}}
    assert h.overrides() == {"pools": {"a6000": 1}}
    (line,) = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert line["result"].startswith("failed (exit 1)")
    assert h.state()["last_action_at"] == NOW + 5


def test_failed_gpu_apply_removes_an_override_file_it_created(harness):
    h = harness(**GPU_CLUSTER, rc={"gpuctl-apply": 2})
    h.main()
    assert h.overrides() is None
    assert not list((h.root / "lambda").glob(".autoscale-state.json*"))  # no temp file left behind


def test_gpu_counts_prefer_the_override_and_scale_down_the_idle_highest_node(harness):
    nodes = [CP, CPU1, node("k3s-gpu-a6000-1"), node("k3s-gpu-a6000-2")]
    h = harness(nodes=nodes, instances=[INSTANCE_1, INSTANCE_1.replace("-1 ", "-2 ")], overrides={"a6000": 2},
                pods=[GPU1_BUSY], state={"last_action_at": None, "idle_since": {"k3s-gpu-a6000-2": NOW - 301}})
    h.main()
    assert h.of("gpuctl-plan")[0]["argv"][1:] == ["plan", "--set-count", "a6000=1"]
    assert json.loads(h.of("gpuctl-apply")[0]["overrides"]) == {"pools": {"a6000": 1}}


def gpu_down(h, tick=None):
    """What make gpu-down does from another terminal: delete the override file, terminate every GPU instance."""
    h.overrides_path.unlink(missing_ok=True)
    h.instances = []


def test_gpu_down_while_running_turns_gpu_scaling_off_until_restart(harness):  # spec section 10.4
    h = harness(**GPU_CLUSTER)
    h.on_wait = lambda h, tick: gpu_down(h) if tick == 1 else h.instances.append(INSTANCE_1)
    h.main("--loop", ticks=3, interval=200)
    assert len(h.of("gpuctl-apply")) == 1  # tick 1 scaled up; gpu-down then deleted the file it wrote
    assert len(h.of("gpuctl-plan")) == 1  # ... and no GPU decision after that, even with an instance back
    assert h.overrides() is None
    notices = [x["reason"] for x in h.log() if x["action"] == "notice"]
    assert sum("deleted outside the autoscaler" in n and "make gpu-down" in n for n in notices) == 2
    nones = [x["reason"] for x in h.log() if x["action"] == "none"]
    assert len(nones) == 2 and all("deleted outside the autoscaler" in n for n in nones)
    assert not any("no GPU pool has autoscale" in n for n in nones)


def test_pool_below_its_target_holds_gpu_scaling(harness):  # gpu-down ran, or gpu-apply has not yet
    h = harness(**{**GPU_CLUSTER, "pods": [GPU_PENDING, CPU_PENDING], "nodes": [CP, CPU1], "instances": []})
    h.main()
    assert h.of("gpuctl-plan") == [] and h.of("gpuctl-apply") == [] and h.overrides() is None
    assert [c["argv"][1:] for c in h.acted()] == [["scale", "2", "--guarded"]]  # CPU rules unaffected
    notices = [x["reason"] for x in h.log() if x["action"] == "notice"]
    assert any("a6000" in n and "0 of 1" in n and "make gpu-apply" in n for n in notices)
    h.now += 200
    h.pods = klist(GPU_PENDING)
    h.main()
    (none,) = [x for x in h.log() if x["action"] == "none"]
    assert "0 of 1" in none["reason"] and "no GPU pool has autoscale" not in none["reason"]


def test_pool_below_its_target_still_scales_down_an_idle_node(harness):
    nodes = [CP, CPU1, node("k3s-gpu-a6000-1", ready=False), node("k3s-gpu-a6000-2")]
    h = harness(nodes=nodes, instances=[INSTANCE_1.replace("-1 ", "-2 ")], overrides={"a6000": 2},
                pods=[GPU_PENDING], state={"last_action_at": None, "idle_since": {"k3s-gpu-a6000-2": NOW - 301}})
    h.main()
    assert h.of("gpuctl-plan")[0]["argv"][1:] == ["plan", "--set-count", "a6000=1"]  # never more nodes, only fewer
    assert h.overrides() == {"pools": {"a6000": 1}}


def test_override_file_changed_after_the_read_aborts_the_gpu_scale(harness):
    h = harness(**GPU_CLUSTER, overrides={"a6000": 1}, state={"last_action_at": NOW - 500, "idle_since": {}})
    h.hooks["gpuctl-plan"] = gpu_down  # gpu-down lands between this tick's read and its write
    assert h.main("--loop", ticks=2, interval=200) == 0
    assert h.of("gpuctl-apply") == [] and h.overrides() is None
    assert len(h.of("gpuctl-plan")) == 1  # GPU scaling stays off
    (line,) = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert line["result"].startswith("not started") and "changed" in line["result"]
    assert h.state()["last_action_at"] == NOW - 500  # nothing ran: no cooldown


def test_gpu_down_during_a_failed_apply_is_not_undone(harness):
    h = harness(**GPU_CLUSTER, overrides={"a6000": 1}, rc={"gpuctl-apply": 1})
    h.hooks["gpuctl-apply"] = gpu_down
    assert h.main() == 1
    assert h.overrides() is None  # the restore does not bring back what gpu-down deleted
    (line,) = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert "left as it is" in line["result"]


def test_second_ctrl_c_during_gpu_apply_restores_the_override(harness, capsys):
    h = harness(**GPU_CLUSTER, overrides={"a6000": 1})

    def interrupt(h):
        raise KeyboardInterrupt

    h.hooks["gpuctl-apply"] = interrupt
    assert h.main() == 130
    assert h.overrides() == {"pools": {"a6000": 1}}  # gpuctl's cleanup terminates the new node: no target 2 left
    err = capsys.readouterr().err
    assert ".autoscale-state.json" in err and "make gpu-status" in err


def test_failed_gpu_scale_down_keeps_the_lower_target(harness):
    nodes = [CP, CPU1, node("k3s-gpu-a6000-1"), node("k3s-gpu-a6000-2")]
    h = harness(nodes=nodes, instances=[INSTANCE_1, INSTANCE_1.replace("-1 ", "-2 ")], overrides={"a6000": 2},
                pods=[GPU1_BUSY], state={"last_action_at": None, "idle_since": {"k3s-gpu-a6000-2": NOW - 301}},
                rc={"gpuctl-apply": 1})
    assert h.main() == 1
    assert h.overrides() == {"pools": {"a6000": 1}}  # never a target above what may still run: no relaunch


# --- gpu-down vs a running tick: gpuctl's write lock (review r2-1, spec section 10.4) ------------------

def test_gpu_apply_runs_under_the_lock_and_inherits_it(harness):
    h = harness(**GPU_CLUSTER)
    assert h.main() == 0
    recheck, apply = h.of("gpuctl-list-nodes")[1], h.of("gpuctl-apply")[0]
    assert not recheck["lock_free"] and not apply["lock_free"]  # held from the re-check through the apply
    (fd,) = apply["pass_fds"]
    assert apply["env"] == {"GPUCTL_LOCK_FD": str(fd)}  # its own apply does not wait for the autoscaler's lock
    assert all(c["pass_fds"] == () for c in h.calls if c is not apply)
    assert lock_is_free(h.lock_path)  # released after the action


def test_gpu_down_between_the_read_and_the_write_without_an_override_launches_nothing(harness):
    """The finding's case: no override file yet (the first scale-up), and gpu-down terminates the instances while
    gpuctl plan runs. Nothing the autoscaler owns changed, but the re-check under the lock sees the instances go."""
    h = harness(**GPU_CLUSTER, state={"last_action_at": NOW - 500, "idle_since": {}})
    h.hooks["gpuctl-plan"] = gpu_down
    assert h.main("--loop", ticks=2, interval=200) == 0
    assert h.of("gpuctl-apply") == [] and h.overrides() is None  # no {a6000: 2} written, nothing launched
    (line,) = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert line["result"].startswith("not started") and "k3s-gpu-a6000-1" in line["result"]
    assert "make gpu-down" in line["result"]
    assert h.state()["last_action_at"] == NOW - 500  # nothing ran: no cooldown
    assert len(h.of("gpuctl-plan")) == 1  # GPU scaling is off until a restart
    notices = [x["reason"] for x in h.log() if x["action"] == "notice"]
    assert any("GPU scaling is off" in n for n in notices)


def test_a_running_gpuctl_write_holds_the_lock_and_the_gpu_scale_waits_a_tick(harness):
    h = harness(**GPU_CLUSTER, state={"last_action_at": NOW - 500, "idle_since": {}})
    held, plans = [], []

    def other_terminal(h):  # a make gpu-apply in another terminal holds the lock during the first plan
        plans.append(1)
        if len(plans) == 1:
            held.append(hold_lock(h.lock_path))

    h.hooks["gpuctl-plan"] = other_terminal
    h.on_wait = lambda h, tick: os.close(held.pop()) if held else None  # ... and ends; nothing changed here
    try:
        assert h.main("--loop", ticks=2, interval=10) == 0
    finally:
        for fd in held:
            os.close(fd)
    first, second = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert first["result"].startswith("not started") and "gpuctl.lock" in first["result"]
    assert second["result"] == "ok"  # not turned off, and no cooldown: the next tick scales up
    assert len(h.of("gpuctl-apply")) == 1 and h.overrides() == {"pools": {"a6000": 2}}


def test_the_lock_is_released_when_the_apply_fails_or_is_interrupted(harness, tmp_path):
    h = harness(**GPU_CLUSTER, rc={"gpuctl-apply": 1})
    h.main()
    assert h.of("gpuctl-apply") and lock_is_free(h.lock_path)

    def interrupt(h):
        raise KeyboardInterrupt

    h.hooks["gpuctl-apply"] = interrupt
    h.rc = {}
    h.now += 500
    assert h.main() == 130
    assert len(h.of("gpuctl-apply")) == 2 and lock_is_free(h.lock_path)


def test_second_ctrl_c_right_after_the_override_write_restores_it(harness, autoscaler, monkeypatch):  # r2-3
    h = harness(**GPU_CLUSTER, overrides={"a6000": 1})
    real = autoscaler.atomic_write

    def write_then_interrupt(path, text, **kw):
        real(path, text, **kw)
        if path == h.overrides_path:
            raise KeyboardInterrupt  # lands after the rename, before gpuctl apply starts

    monkeypatch.setattr(autoscaler, "atomic_write", write_then_interrupt)
    assert h.main() == 130
    assert h.of("gpuctl-apply") == []
    assert h.overrides() == {"pools": {"a6000": 1}}  # no raised GPU target left behind
    assert lock_is_free(h.lock_path)


def test_a_third_ctrl_c_cannot_stop_the_override_restore(harness, autoscaler, monkeypatch):  # r2-3
    h = harness(**GPU_CLUSTER, overrides={"a6000": 1})
    h.stopper = autoscaler.Stopper()

    def second_ctrl_c(h):  # the second Ctrl-C interrupts gpuctl apply
        h.stopper.request()
        raise KeyboardInterrupt

    h.hooks["gpuctl-apply"] = second_ctrl_c
    real = autoscaler.atomic_write

    def third_ctrl_c(path, text, **kw):
        if path == h.overrides_path and json.loads(text) == {"pools": {"a6000": 1}}:
            h.stopper._handle(signal.SIGINT, None)  # lands as the file is put back
        real(path, text, **kw)

    monkeypatch.setattr(autoscaler, "atomic_write", third_ctrl_c)
    assert h.main() == 130
    assert h.overrides() == {"pools": {"a6000": 1}}  # put back all the same
    assert lock_is_free(h.lock_path)


def test_run_command_returns_minus_one_for_a_sighup_killed_child(autoscaler):  # r2-5: the real Popen value
    r = autoscaler.run_command(["bash", "-c", "kill -HUP $$"], stream=True)
    assert r.rc == -signal.SIGHUP == -1
    assert autoscaler._exit_text(r.rc) == "killed by SIGHUP"
    assert autoscaler._exit_text(2) == "exit 2" and autoscaler._exit_text(-200) == "killed by signal 200"


def test_a_child_killed_by_a_signal_is_a_failure_not_a_refusal(harness):  # r2-5: SIGHUP -> returncode -1
    h = harness(**GPU_CLUSTER, overrides={"a6000": 1}, rc={"gpuctl-apply": -1})
    assert h.main() == 1
    (line,) = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert line["result"].startswith("failed (killed by SIGHUP)") and "1/2 failures" in line["result"]
    assert h.state()["last_action_at"] == NOW + 5  # the cooldown counts, from the end of the run
    assert h.overrides() == {"pools": {"a6000": 1}}


def test_a_cpu_pool_killed_by_a_signal_is_a_failure(harness):
    h = harness(pods=[CPU_PENDING], rc={"cpu-pool": -1})
    assert h.main() == 1
    (line,) = [x for x in h.log() if x["action"] == "cpu_scale"]
    assert line["result"].startswith("failed (killed by SIGHUP)")


# --- a failed GPU scale-down is finished (review r2-2) ---------------------------------------------

def scaled_down_cluster(harness, rc):
    """Two GPU nodes, target 2, k3s-gpu-a6000-2 idle past idle_minutes: the first tick scales a6000 down to 1."""
    nodes = [CP, CPU1, node("k3s-gpu-a6000-1"), node("k3s-gpu-a6000-2")]
    return harness(nodes=nodes, instances=[INSTANCE_1, INSTANCE_1.replace("-1 ", "-2 ")], overrides={"a6000": 2},
                   pods=[GPU1_BUSY], state={"last_action_at": None, "idle_since": {"k3s-gpu-a6000-2": NOW - 301}},
                   rc={"gpuctl-apply": rc})


def terminated_2(h):
    h.instances = [INSTANCE_1]
    h.nodes = klist(CP, CPU1, node("k3s-gpu-a6000-1"))


def test_failed_gpu_scale_down_is_retried_after_the_cooldown(harness):
    h = scaled_down_cluster(harness, [1, 0])
    applies = []

    def apply_hook(h):
        applies.append(1)
        if len(applies) == 2:  # the retry terminates the idle node
            terminated_2(h)

    h.hooks["gpuctl-apply"] = apply_hook
    h.main("--loop", ticks=4, interval=70)  # tick 2 is inside the cooldown, tick 3 past it
    assert len(h.of("gpuctl-apply")) == 2 and h.overrides() == {"pools": {"a6000": 1}}
    first, retry = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert first["target"] == 1 and "retried after the cooldown" in first["result"]
    assert retry["target"] == 1 and retry["result"] == "ok" and "2 GPU instances live, target 1" in retry["reason"]
    notices = [x["reason"] for x in h.log() if x["action"] == "notice"]
    assert sum("pool a6000: 2 GPU instances live, target 1" in n for n in notices) == 2  # ticks 2 and 3, not 4
    assert h.log()[-1]["action"] == "none"


def test_scale_down_that_terminated_before_failing_needs_no_retry(harness):
    h = scaled_down_cluster(harness, 1)
    h.hooks["gpuctl-apply"] = terminated_2  # the terminate went through, a later step failed
    h.main("--loop", ticks=3, interval=200)
    assert len(h.of("gpuctl-apply")) == 1
    assert not any("GPU instances live" in x["reason"] for x in h.log() if x["action"] == "notice")


def test_live_above_target_is_reported_every_tick_until_it_converges(harness):
    h = scaled_down_cluster(harness, 1)
    h.main("--loop", ticks=5, interval=200)
    assert len(h.of("gpuctl-apply")) == 2  # the scale-down and one retry: then GPU scale-down is off
    notices = [x["reason"] for x in h.log() if x["action"] == "notice"]
    live = [n for n in notices if "pool a6000: 2 GPU instances live, target 1" in n]
    assert len(live) == 4 and "make gpu-apply" in live[-1] and "make gpu-down" in live[-1]


# --- a failure limit turns off only its own kind (review r2-4) ---------------------------------------

def test_gpu_scale_down_off_still_scales_up(harness):
    h = scaled_down_cluster(harness, [1, 1, 0])

    def gpu_pod_arrives(h, tick):
        if tick == 2:  # GPU scale-down is off now; a GPU Pod goes Pending
            h.pods = klist(GPU1_BUSY, GPU_PENDING)

    h.on_wait = gpu_pod_arrives
    h.main("--loop", ticks=3, interval=200)
    results = [(x["target"], x["result"]) for x in h.log() if x["action"] == "gpu_scale"]
    assert "GPU scale-down disabled after 2 failures" in results[1][1]
    assert results[2] == (2, "ok")  # GPU scale-up still works
    assert not any("waiting: GPU scale-down disabled" in x["reason"] for x in h.log())


def test_repeated_gpu_apply_failures_disable_gpu_scale_up(harness):
    h = harness(**GPU_CLUSTER, rc={"gpuctl-apply": 1})
    h.main("--loop", ticks=6, interval=200)  # every tick is past the cooldown
    assert len(h.of("gpuctl-apply")) == 2  # never a loop of paid launches
    results = [x["result"] for x in h.log() if x["action"] == "gpu_scale"]
    assert "disabled after 2 failures" in results[-1]
    notices = [x["reason"] for x in h.log() if x["action"] == "notice"]
    assert sum("GPU scale-up disabled after 2 failures" in n and "make autoscale" in n for n in notices) == 4


def test_repeated_cpu_pool_failures_disable_cpu_scale_up(harness):
    h = harness(pods=[CPU_PENDING], rc={"cpu-pool": 1})
    h.main("--loop", ticks=6, interval=200)
    assert len(h.acted()) == 3
    assert any("CPU scale-up disabled after 3 failures" in x["reason"] for x in h.log() if x["action"] == "none")


def test_gpu_plan_refusal_is_remembered_for_the_cooldown(harness):
    h = harness(**GPU_CLUSTER, rc={"gpuctl-plan": 2})
    h.main("--loop", ticks=3, interval=20)
    assert len(h.of("gpuctl-plan")) == 1  # no Lambda API call every tick
    nones = [x["reason"] for x in h.log() if x["action"] == "none"]
    assert len(nones) == 3 and all("refused by gpuctl plan" in n for n in nones)
    assert "checked again in" in nones[-1]
    h.now += 200
    h.main()
    assert len(h.of("gpuctl-plan")) == 2


def test_unwritable_override_file_is_a_logged_failure(harness, autoscaler, monkeypatch):
    h = harness(**GPU_CLUSTER)
    real = autoscaler.atomic_write

    def atomic_write(path, text, **kw):
        if path.name == ".autoscale-state.json":
            raise OSError(28, "No space left on device")
        return real(path, text, **kw)

    monkeypatch.setattr(autoscaler, "atomic_write", atomic_write)
    assert h.main() == 1
    assert h.of("gpuctl-apply") == []
    (line,) = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert line["result"].startswith("failed") and "No space left on device" in line["result"]
    assert "left as it is" not in line["result"]  # the file nobody changed is not taken for a gpu-down
    assert not any("GPU scaling is off" in x["reason"] for x in h.log())


def test_unwritable_state_file_starts_no_action(harness, autoscaler, monkeypatch):
    h = harness(pods=[CPU_PENDING])

    def atomic_write(path, text, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(autoscaler, "atomic_write", atomic_write)
    assert h.main() == 1
    assert h.acted() == []  # its cooldown cannot be saved first (Review Focus #1)
    (line,) = [x for x in h.log() if x["action"] == "cpu_scale"]
    assert "cannot write" in line["result"] and "Permission denied" in line["result"]


@pytest.mark.parametrize("instance, why", [
    ("k3s-gpu-a6000-2 - booting gpu_1x_a6000 us-west-1", "booting"),
    ("k3s-gpu-a6000-2 129.80.1.3 active gpu_1x_a6000 us-west-1", "no registered Node"),
])
def test_launching_instance_holds_gpu_scale_up(harness, instance, why):  # Review Focus #4
    h = harness(**{**GPU_CLUSTER, "instances": [INSTANCE_1, instance]}, rc={})
    h.main()
    assert h.of("gpuctl-plan") == [] and h.acted() == []
    log = h.log()
    assert any(x["action"] == "notice" and why in x["reason"] and "k3s-gpu-a6000-2" in x["reason"] for x in log)
    (none,) = [x for x in log if x["action"] == "none"]
    assert why in none["reason"] and "k3s-gpu-a6000-2" in none["reason"]  # the real reason, not a generic one
    if why != "booting":
        assert "booting" not in none["reason"]


def test_list_nodes_failure_holds_gpu_scale_up(harness):
    h = harness(**GPU_CLUSTER, rc={"gpuctl-list-nodes": 2})
    h.main()
    assert h.acted() == []
    assert any(x["action"] == "notice" and "list-nodes" in x["reason"] for x in h.log())
    (none,) = [x for x in h.log() if x["action"] == "none"]
    assert "list-nodes failed" in none["reason"] and "booting" not in none["reason"]


def test_not_ready_gpu_node_is_logged_every_tick_with_nothing_pending(harness):
    nodes = [CP, CPU1, node("k3s-gpu-a6000-1", ready=False), node("k3s-gpu-a6000-2", labeled=False)]
    h = harness(nodes=nodes, instances=[INSTANCE_1, INSTANCE_1.replace("-1 ", "-2 ")])
    h.main("--loop", ticks=3)
    notices = [x["reason"] for x in h.log() if x["action"] == "notice"]
    assert sum("k3s-gpu-a6000-1 is NotReady" in n for n in notices) == 3
    assert sum("k3s-gpu-a6000-2" in n and "label" in n for n in notices) == 3


def test_clock_skew_against_node_ready_time_is_logged(harness):
    h = harness(nodes=[CP, CPU1, node("k3s-gpu-a6000-1", ready_at=NOW + 900)], instances=[INSTANCE_1])
    h.main()
    assert any(x["action"] == "notice" and "clock skew" in x["reason"] and "k3s-gpu-a6000-1" in x["reason"]
               for x in h.log())


# --- skipped ticks (Ruling SR7(a)) ---------------------------------------------------------------

@pytest.mark.parametrize("case", ["state", "nodes", "kubectl", "overrides", "tfvars"])
def test_unreadable_input_skips_the_tick(harness, case):
    kw = {"pods": [CPU_PENDING]}
    if case == "state":
        kw["state"] = '{"last_action_at": NaN}'
    h = harness(**kw, rc={"kubectl": 1} if case == "kubectl" else {})
    if case == "nodes":
        h.nodes = {"kind": "Status", "status": "Failure"}
    if case == "overrides":
        (h.root / "lambda" / ".autoscale-state.json").write_text('{"pools": {"a6000": -1}}')
    if case == "tfvars":
        (h.root / "infra" / "workers" / "workers.auto.tfvars.json").write_text("{")
    before = h.state_path.read_text() if h.state_path.exists() else None
    assert h.main() == 1
    assert h.acted() == []
    (line,) = h.log()
    assert line["action"] == "skip_tick" and line["result"] == "skipped" and line["reason"]
    assert (h.state_path.read_text() if h.state_path.exists() else None) == before  # nothing saved


def test_a_skipped_tick_does_not_stop_the_loop(harness):
    h = harness(pods=[CPU_PENDING], rc={"kubectl": [1, 0]})
    h.main("--loop", ticks=2)
    assert [x["action"] for x in h.log()][0] == "skip_tick"
    assert len(h.acted()) == 1


# --- state and log files -------------------------------------------------------------------------

def test_every_tick_is_logged_and_state_is_saved_atomically(harness):
    h = harness(nodes=[CP, CPU1, node("k3s-cpu-2")], workers=("k3s-cpu-1", "k3s-cpu-2"))
    h.main("--loop", ticks=3)
    log = h.log()
    assert [x["action"] for x in log] == ["none", "none", "none"]
    assert all({"ts", "action", "reason", "result", "duration_s"} <= set(x) for x in log)
    assert "k3s-cpu-2 idle" in log[-1]["reason"]
    assert h.state()["idle_since"] == {"k3s-cpu-1": NOW, "k3s-cpu-2": NOW}
    assert [p.name for p in h.state_path.parent.iterdir()] == ["autoscaler-state.json"]  # no temp files


def test_ctrl_c_while_reading_starts_no_action(autoscaler, harness):
    h = harness(pods=[CPU_PENDING])
    stopper = autoscaler.Stopper()
    fake_run = h.run

    def run(argv, **kw):
        if "nodes" in argv:
            stopper.request()  # Ctrl-C lands while the tick reads the cluster
        return fake_run(argv, **kw)

    deps = autoscaler.Deps(clock=lambda: h.now, run=run, wait=lambda s: False, out=h.out, stopper=stopper)
    assert autoscaler.main(["--interval", "20", "--log", str(h.log_dir)], env={"AUTOSCALER_ROOT": str(h.root)},
                           deps=deps) == 0
    assert h.acted() == []
    (line,) = [x for x in h.log() if x["action"] == "cpu_scale"]
    assert line["result"] == "not started (Ctrl-C)"
    assert "autoscaler: stopped" in h.out.getvalue()


def test_bad_interval_is_a_usage_error(autoscaler):
    with pytest.raises(SystemExit) as e:
        autoscaler.main(["--interval", "0"], env={})
    assert e.value.code == 2


def test_dry_run_loop_simulates_the_cooldown_in_memory(harness):
    h = harness(pods=[CPU_PENDING])
    h.main("--dry-run", "--loop", ticks=2)
    out = h.out.getvalue()
    assert out.count("cpu_scale target=2") == 1 and "no action (cooldown)" in out
    assert h.acted() == [] and not h.state_path.exists()


def test_default_log_dir_is_verification_autoscale_ts(autoscaler, tmp_path):
    h = Harness(autoscaler, tmp_path)
    deps = autoscaler.Deps(clock=lambda: NOW, run=h.run, wait=lambda s: True, out=h.out)
    assert autoscaler.main(["--once"], env={"AUTOSCALER_ROOT": str(h.root)}, deps=deps) == 0
    (log,) = (h.root / "verification").glob("autoscale-*/autoscaler.log")
    assert json.loads(log.read_text().splitlines()[0])["action"] == "none"


def test_startup_says_to_stop_it_before_gpu_down(harness):  # spec section 10.4
    h = harness()
    h.main()
    assert "Ctrl-C" in h.out.getvalue() and "before make gpu-down" in h.out.getvalue()


def test_makefile_autoscale_target():
    text = (REPO / "Makefile").read_text()
    assert "\nautoscale: ## " in text and "scripts/autoscaler.py" in text
    assert ".PHONY: autoscale" in text


# --- Ctrl-C ---------------------------------------------------------------------------------------

def test_stopper_takes_no_lock_in_the_signal_handler(autoscaler, capfd):
    stopper = autoscaler.Stopper()
    stopper._handle(signal.SIGINT, None)
    assert stopper.requested and stopper.wait(30) is True  # returns at once once a stop was asked for
    assert "stopping" in capfd.readouterr().err
    assert not hasattr(stopper, "event")  # threading.Event.set() in a handler can deadlock against Event.wait()


def test_second_ctrl_c_while_reading_stops_at_once(autoscaler, capfd):
    stopper = autoscaler.Stopper()
    with stopper.reading():
        stopper._handle(signal.SIGINT, None)
        assert "Ctrl-C again stops at once" in capfd.readouterr().err
        with pytest.raises(KeyboardInterrupt):
            stopper._handle(signal.SIGINT, None)
    stopper._handle(signal.SIGINT, None)  # outside a read or an action (saving state): never raised there


def test_ctrl_c_during_the_read_phase_exits_without_the_action_hint(harness, autoscaler, capsys):
    h = harness(pods=[CPU_PENDING])

    def interrupt(h):
        raise KeyboardInterrupt

    h.hooks["gpuctl-list-nodes"] = interrupt
    assert h.main() == 130
    err = capsys.readouterr().err
    assert "nothing was changing" in err and "an action was interrupted" not in err
    assert h.acted() == [] and not h.state_path.exists()


# --- Ctrl-C (subprocess) -------------------------------------------------------------------------

def start_loop(tmp_path, scenario):
    """The autoscaler as the leader of its own process group, like a foreground job in a terminal; its stderr goes
    to tmp_path/stderr.txt so a test can wait for what the Ctrl-C handler writes."""
    root, bin_dir = stub_root(tmp_path, scenario)
    env = script_env(tmp_path, root, bin_dir, now=None)
    stderr = open(tmp_path / "stderr.txt", "w")
    try:
        return subprocess.Popen([sys.executable, str(SCRIPT), "--interval", "30", "--log", str(tmp_path / "log")],
                                env=env, stdout=subprocess.PIPE, stderr=stderr, text=True,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    finally:
        stderr.close()


def stderr_of(tmp_path):
    return (tmp_path / "stderr.txt").read_text()


def ctrl_c(proc):
    """What the terminal does on Ctrl-C: SIGINT to every process of the foreground process group."""
    os.killpg(proc.pid, signal.SIGINT)


def wait_for(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def pending_now():
    return pod("cpu-hog-1", unschedulable=CPU_MSG, since=time.time() - 60)


def test_ctrl_c_finishes_the_current_action_then_exits(tmp_path):
    proc = start_loop(tmp_path, {"pods": klist(pending_now()), "nodes": klist(CP, CPU1), "sleep": {"cpu-pool": 2}})
    try:
        assert wait_for(lambda: any(c["tool"] == "cpu-pool" for c in fake_calls(tmp_path)))
        ctrl_c(proc)
        out, _ = proc.communicate(timeout=30)
    finally:
        proc.kill()
    assert proc.returncode == 0, out + stderr_of(tmp_path)
    assert (tmp_path / "calls.jsonl.done").exists()  # the action was not interrupted
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["last_action_at"] is not None
    (line,) = [json.loads(x) for x in (tmp_path / "log" / "autoscaler.log").read_text().splitlines()
               if json.loads(x)["action"] == "cpu_scale"]
    assert line["result"] == "ok"


def test_second_ctrl_c_interrupts_the_action_and_says_how_to_clean_up(tmp_path):
    proc = start_loop(tmp_path, {"pods": klist(pending_now()), "nodes": klist(CP, CPU1), "sleep": {"cpu-pool": 20}})
    try:
        assert wait_for(lambda: any(c["tool"] == "cpu-pool" for c in fake_calls(tmp_path)))
        ctrl_c(proc)
        # the first handler has run (it wrote this) before the second Ctrl-C: two signals never merge into one
        assert wait_for(lambda: "stopping after the current action" in stderr_of(tmp_path))
        ctrl_c(proc)
        out, _ = proc.communicate(timeout=30)
    finally:
        proc.kill()
    err = stderr_of(tmp_path)
    assert proc.returncode == 130, out + err
    assert not (tmp_path / "calls.jsonl.done").exists()
    assert "make cpu-scale N=1" in err and "make gpu-status" in err
    (line,) = [json.loads(x) for x in (tmp_path / "log" / "autoscaler.log").read_text().splitlines()
               if json.loads(x)["action"] == "cpu_scale"]
    assert line["result"].startswith("interrupted")
    assert json.loads((tmp_path / "state.json").read_text())["last_action_at"] is not None


def test_ctrl_c_between_ticks_exits_at_once(tmp_path):
    proc = start_loop(tmp_path, {"pods": klist(), "nodes": klist(CP, CPU1)})
    try:
        assert wait_for(lambda: (tmp_path / "log" / "autoscaler.log").exists())
        ctrl_c(proc)
        out, _ = proc.communicate(timeout=10)  # well inside the 30 s interval: the wait ended at once
    finally:
        proc.kill()
    assert proc.returncode == 0, out + stderr_of(tmp_path)
    assert "Traceback" not in stderr_of(tmp_path)
    assert [c["tool"] for c in fake_calls(tmp_path)] == ["kubectl", "kubectl", "gpuctl"]  # no second tick


def test_a_lock_file_that_cannot_be_opened_is_a_failed_gpu_scale(harness):
    h = harness(**GPU_CLUSTER)
    h.lock_path.parent.mkdir(parents=True, exist_ok=True)
    h.lock_path.mkdir()  # not a file: it cannot be opened
    assert h.main() == 1
    assert h.of("gpuctl-apply") == [] and h.overrides() is None
    (line,) = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert line["result"].startswith("failed") and "gpuctl.lock" in line["result"]



# --- review round 3 ----------------------------------------------------------------------------------

def test_a_retried_scale_down_that_finishes_before_the_recheck_is_not_a_gpu_down(harness):  # review S4-r3-2
    """The first apply's terminate goes through while the retry runs gpuctl plan: the re-check under the lock sees
    k3s-gpu-a6000-2 gone. That is this autoscaler's own scale-down finishing, not a make gpu-down."""
    h = scaled_down_cluster(harness, [1, 0])
    plans = []

    def plan_hook(h):
        plans.append(1)
        if len(plans) == 2:  # the retry's plan (tick 3)
            terminated_2(h)

    def gpu_pod_arrives(h, tick):
        if tick == 3:
            h.pods = klist(GPU1_BUSY, GPU_PENDING)

    h.hooks["gpuctl-plan"] = plan_hook
    h.on_wait = gpu_pod_arrives
    assert h.main("--loop", ticks=5, interval=70) == 0
    first, retry, up = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert first["target"] == 1 and first["result"].startswith("failed")
    assert retry["target"] == 1 and retry["result"].startswith("not started")
    assert "scale-down finished" in retry["result"] and "make gpu-down" not in retry["result"]
    assert up["target"] == 2 and up["result"] == "ok"  # GPU scaling is still on
    assert len(h.of("gpuctl-apply")) == 2 and h.overrides() == {"pools": {"a6000": 2}}
    assert not any("GPU scaling is off" in x["reason"] for x in h.log())


def test_instances_gone_below_the_target_during_a_retry_still_turn_gpu_scaling_off(harness):  # review S4-r3-2
    """Not the scale-down finishing: every instance went (terminated outside the autoscaler) and the pool is now
    below its target."""
    h = scaled_down_cluster(harness, [1, 0])
    plans = []

    def plan_hook(h):
        plans.append(1)
        if len(plans) == 2:
            h.instances = []

    h.hooks["gpuctl-plan"] = plan_hook
    h.main("--loop", ticks=4, interval=70)
    first, retry = [x for x in h.log() if x["action"] == "gpu_scale"]
    assert retry["result"].startswith("not started") and "make gpu-down" in retry["result"]
    assert len(h.of("gpuctl-apply")) == 1
    assert any("GPU scaling is off" in x["reason"] for x in h.log() if x["action"] == "notice")


def test_an_unwritable_log_never_stops_the_loop(harness, capsys):  # review S4-r3-6
    h = harness(**GPU_CLUSTER, overrides={"a6000": 1}, rc={"gpuctl-apply": 1})
    h.hooks["gpuctl-apply"] = lambda h: shutil.rmtree(h.log_dir)  # the log directory goes during the apply
    assert h.main("--loop", ticks=2, interval=20) == 0
    assert h.overrides() == {"pools": {"a6000": 1}}  # the failed scale-up was still put back
    assert h.state()["last_action_at"] == NOW + 5  # and its cooldown saved
    assert len(h.of("kubectl")) == 4  # the second tick ran
    out = h.out.getvalue()
    assert "gpu_scale pool=a6000 target=2 -> failed" in out and "no action" in out
    err = capsys.readouterr().err
    assert err.count("cannot write the log") == 1 and str(h.log_dir / "autoscaler.log") in err
