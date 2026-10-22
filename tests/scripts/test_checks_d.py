"""Offline tests for the GPU checks D1-D6 (Task 21) in scripts/lib/checks.sh.

Same harness as test_checks_abc.py: a repo copy whose kubectl, ssh and scripts/gpuctl are tests/scripts/fake_tools.py,
and a scripted healthy cluster. Here it has a GPU node, k3s-gpu-a6000-1, as the early live session measured it
(2026-10-21, progress.md "Task 21 early live session DONE"): the A6000 pool fell back to an A10, so the node is
labelled gpu-type=a10 and nvidia-smi prints "NVIDIA A10". Each test breaks one thing and runs one check through
run_check, as make verify does. Nothing here reaches Lambda, a cluster or the network.
"""
import os
import re
import shlex
import signal
import subprocess
import time

import pytest

from test_checks_abc import CP, CPU1, EXPECTED, GPU1, NODES, NOT_REACHABLE, Cluster, kc, on, rule
from test_ops_scripts import free_port

GPU2 = "k3s-gpu-a6000-2"
GPU2_IP = "203.0.113.13"
SSH_IP = {GPU1: NODES[GPU1]["ext"], GPU2: GPU2_IP}
LABEL = "node-restriction.kubernetes.io/accelerator=nvidia"
# join-node.sh's check of the nvidia runtime in k3s's containerd config (fake_tools.py's CONTAINERD_GREP)
CONTAINERD_GREP = "sudo grep -A3 nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml"
FUNCS = ["check_d1_gpu_stack", "check_d2_gpu_capacity", "check_d3_device_plugin", "check_d4_gpu_smoke",
         "check_d5_vectoradd", "check_d6_overflow"]
AUDIT = ("===== gpu-audit before\n$ nvidia-smi --query-gpu=name,driver_version --format=csv\n"
         "name, driver_version\nNVIDIA A10, 580.105.08\n===== gpu-audit after\n$ cat /sys/class/net/eno1/mtu\n1460\n")
CONTAINERD = "[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia']\n  runtime_type = \"io.containerd.runc.v2\"\n"
SMI = ("+-----------------------------------------------------------------------------------------+\n"
       "| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |\n"
       "| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n"
       "|   0  NVIDIA A10                     On  |   00000000:06:00.0 Off |                    0 |\n")
VECTORADD = "[Vector addition of 50000 elements]\nCopy input data from the host memory to the CUDA device\nTest PASSED\nDone\n"
REGISTERED = ("I1022 05:24:41.123456       1 server.go:165] Starting GRPC server for 'nvidia.com/gpu'\n"
              "I1022 05:24:41.223456       1 server.go:117] Registered device plugin for 'nvidia.com/gpu' with Kubelet\n")
OVERFLOW_UID = "0f5e3c1a-7b2d-4e8f-9a6b-1c2d3e4f5a6b"
EVENTS = ("LAST SEEN   TYPE      REASON             OBJECT             MESSAGE\n"
          "31s         Warning   FailedScheduling   pod/gpu-overflow   0/3 nodes are available: 1 Insufficient "
          "nvidia.com/gpu, 2 node(s) didn't match Pod's node affinity/selector.\n")


def describe(node, capacity="1", allocatable="1"):
    """kubectl describe node, cut down to what D2 reads; None leaves nvidia.com/gpu out of that section."""
    cap = f"  nvidia.com/gpu:     {capacity}\n" if capacity is not None else ""
    alloc = f"  nvidia.com/gpu:     {allocatable}\n" if allocatable is not None else ""
    return (f"Name:               {node}\nLabels:             gpu-type=a10\n"
            f"Capacity:\n  cpu:                30\n  memory:             226Gi\n{cap}  pods:               110\n"
            f"Allocatable:\n  cpu:                30\n  memory:             225Gi\n{alloc}  pods:               110\n"
            "System Info:\n  Kernel Version:     6.8.0-1046-nvidia\n")


def gpu_answers(gpus=(GPU1,), allocatable=None, gpu_type="a10", smi=SMI):
    """Scripted answers for a healthy GPU side: every check D1-D6 passes."""
    allocatable = allocatable or {g: "1" for g in gpus}
    r = [
        kc("get nodes", f"-l {LABEL}", "allocatable", stdout="".join(f"{g} {allocatable[g]}\n" for g in gpus)),
        kc("get nodes", f"-l {LABEL}", stdout="".join(f"{g}\n" for g in gpus)),
        kc("get runtimeclass nvidia-gpu", stdout="NAME         HANDLER   AGE\nnvidia-gpu   nvidia    2d\n"),
        kc("-n nvidia-device-plugin logs", stdout=REGISTERED),
        kc("-n nvidia-device-plugin get pods -o wide",
           stdout="NAME                 READY   STATUS    NODE\nnvdp-nvidia-device-plugin-x   1/1   Running   "
                  f"{gpus[0]}\n"),
        kc("-n gpu-test delete"),
        kc("apply -f", "gpu-smoke.yaml", stdout="pod/gpu-smoke created\n"),
        kc("apply -f", "vectoradd.yaml", stdout="pod/vectoradd created\n"),
        kc("apply -f", "gpu-overflow.yaml", stdout="pod/gpu-overflow created\n"),
        kc("-n gpu-test get pod gpu-smoke", "phase", stdout="Succeeded"),
        kc("-n gpu-test get pod vectoradd", "phase", stdout="Succeeded"),
        kc("-n gpu-test get pod gpu-smoke", "nodeName", stdout=gpus[0]),
        kc("-n gpu-test get pod vectoradd", "nodeName", stdout=gpus[0]),
        kc("-n gpu-test logs gpu-smoke", stdout=smi),
        kc("-n gpu-test logs vectoradd", stdout=VECTORADD),
        kc("-n gpu-test create -f", "gpu-hold.yaml", stdout="pod/gpu-hold-x1y2z created\n"),
        kc("-n gpu-test wait pod -l app=gpu-hold", stdout="pod/gpu-hold-x1y2z condition met\n"),
        kc("-n gpu-test get pods -l app=gpu-hold", "status.phase=Running",
           stdout="".join(f"gpu-hold-{i}\n" for i in range(sum(int(v or 0) for v in allocatable.values())))),
        kc("-n gpu-test get pod gpu-overflow", "phase", stdout="Pending"),
        kc("-n gpu-test get pod gpu-overflow", "{.metadata.uid}", stdout=OVERFLOW_UID),
        kc("-n gpu-test get events", "involvedObject.name=gpu-overflow", f"involvedObject.uid={OVERFLOW_UID}",
           stdout=EVENTS),
        kc("-n gpu-test get events", stdout="No resources found in gpu-test namespace.\n"),
        kc("-n gpu-test describe pod"),
    ]
    for g in gpus:
        ssh = f"ubuntu@{SSH_IP[g]}"
        r += [kc(f"get node {g} ", "gpu-type", stdout=gpu_type),
              kc(f"describe node {g} ", stdout=describe(g)),
              rule("ssh", (ssh, "sudo cat /var/log/gpu-audit.txt"), stdout=AUDIT),
              rule("ssh", (ssh, CONTAINERD_GREP), stdout=CONTAINERD)]
    return r


class GpuCluster(Cluster):
    """k3s-cp, k3s-cpu-1 and k3s-gpu-a6000-1 healthy for A-C, plus the GPU nodes `gpus` healthy for D."""

    def __init__(self, tmp, gpus=(GPU1,), **kw):
        super().__init__(tmp, nodes=(CP, CPU1, GPU1))
        self.first(*gpu_answers(gpus, **kw))


@pytest.fixture
def gpu(tmp_path):
    return GpuCluster(tmp_path)


@pytest.fixture
def no_gpu(tmp_path):
    return Cluster(tmp_path)


def two_gpus(tmp_path, **kw):
    """GPU1 and GPU2; scripts/gpuctl ip tells their SSH IPs apart (the fake answers gpu_ip for any other name)."""
    env = GpuCluster(tmp_path, gpus=(GPU1, GPU2), **kw)
    env.first(rule("gpuctl", ["ip", GPU2], stdout=GPU2_IP + "\n"))
    return env


def remote(env, ip):
    return [a[-1] for a in env.argvs("ssh") if a[-2] == f"ubuntu@{ip}"]


def gpu_test_calls(env):
    return [a for a in env.kubectl() if "gpu-test" in " ".join(a)]


# ---- every D check: SKIP without a GPU node, FAIL when the node list cannot be read ---------------

@pytest.mark.parametrize("func", FUNCS)
def test_d_check_skips_without_a_gpu_node(no_gpu, func):
    result, text = no_gpu.check(func)
    assert result == "SKIP", text
    assert f"get nodes -l {LABEL}" in text and "no GPU node" in text
    # nothing else ran: no SSH, no .env needed, no Pod created
    assert no_gpu.argvs("ssh") == [] and len(no_gpu.kubectl()) == 1


@pytest.mark.parametrize("func", FUNCS)
def test_d_check_skips_without_reading_env(no_gpu, func):
    (no_gpu.repo / ".env").unlink()
    assert no_gpu.check(func)[0] == "SKIP"


@pytest.mark.parametrize("func", FUNCS)
def test_d_check_fails_when_the_gpu_node_list_cannot_be_read(gpu, func):
    gpu.first(kc("get nodes", f"-l {LABEL}", rc=1, stderr="Unable to connect to the server\n"))
    assert gpu.check(func)[0] == "FAIL"
    assert gpu_test_calls(gpu) == []


@pytest.mark.parametrize("func", FUNCS)
def test_d_check_refuses_an_unexpected_gpu_node_name(gpu, func):
    gpu.first(kc("get nodes", f"-l {LABEL}", stdout="k3s-cpu-1\n"))
    result, text = gpu.check(func)
    assert result == "FAIL" and "unexpected GPU node name" in text
    assert gpu.argvs("ssh") == [] and gpu_test_calls(gpu) == []


# ---- D1 GPU stack -----------------------------------------------------------------------------

def test_d1_reads_the_audit_and_containerd_config_and_the_runtimeclass(gpu):
    result, text = gpu.check("check_d1_gpu_stack")
    assert result == "PASS", text
    assert gpu.remote(GPU1) == ["sudo cat /var/log/gpu-audit.txt", CONTAINERD_GREP]
    assert ["get", "runtimeclass", "nvidia-gpu", "--request-timeout=60s"] in gpu.kubectl()
    assert "NVIDIA A10, 580.105.08" in text and "runtimes.'nvidia'" in text


def test_d1_checks_every_gpu_node(tmp_path):
    env = two_gpus(tmp_path)
    assert env.check("check_d1_gpu_stack")[0] == "PASS"
    assert remote(env, NODES[GPU1]["ext"]) == ["sudo cat /var/log/gpu-audit.txt", CONTAINERD_GREP]
    assert remote(env, GPU2_IP) == ["sudo cat /var/log/gpu-audit.txt", CONTAINERD_GREP]


@pytest.mark.parametrize("answer", [
    on(GPU1, "gpu-audit.txt", stdout=""),                                              # empty
    on(GPU1, "gpu-audit.txt", stdout="  \n"),                                          # blank
    on(GPU1, "gpu-audit.txt", rc=1, stderr="cat: /var/log/gpu-audit.txt: No such file or directory\n"),
    on(GPU1, CONTAINERD_GREP, rc=1),                                                   # no nvidia section
    on(GPU1, CONTAINERD_GREP, rc=255, stderr="ssh: connect to host 203.0.113.12 port 22: Connection timed out\n"),
    kc("get runtimeclass nvidia-gpu", rc=1,
       stderr='Error from server (NotFound): runtimeclasses.node.k8s.io "nvidia-gpu" not found\n'),
])
def test_d1_fails_on_a_missing_part(gpu, answer):
    gpu.first(answer)
    assert gpu.check("check_d1_gpu_stack")[0] == "FAIL"


def test_d1_fails_when_one_of_two_gpu_nodes_lacks_the_audit(tmp_path):
    env = two_gpus(tmp_path)
    env.first(rule("ssh", (f"ubuntu@{GPU2_IP}", "gpu-audit.txt"), stdout=""))
    result, text = env.check("check_d1_gpu_stack")
    assert result == "FAIL" and GPU2 in text


# ---- D2 capacity and allocatable --------------------------------------------------------------

def test_d2_reads_capacity_and_allocatable_from_describe_node(gpu):
    result, text = gpu.check("check_d2_gpu_capacity")
    assert result == "PASS", text
    assert ["describe", "node", GPU1, "--request-timeout=60s"] in gpu.kubectl()
    assert f"D2: {GPU1}: nvidia.com/gpu capacity 1, allocatable 1" in text
    assert "Capacity:" in text  # describe's output is in the evidence


def test_d2_waits_for_the_resource_to_appear(gpu):
    # L21-2: nvidia.com/gpu shows up ~20 s after the device plugin registers
    gpu.first(kc(f"describe node {GPU1} ", seq=[{"stdout": describe(GPU1, None, None)},
                                               {"stdout": describe(GPU1, "1", None)},
                                               {"stdout": describe(GPU1)}]))
    result, text = gpu.check("check_d2_gpu_capacity")
    assert result == "PASS", text
    assert len([a for a in gpu.kubectl() if a[:2] == ["describe", "node"]]) == 3


@pytest.mark.parametrize("cap, alloc", [(None, None), ("1", None), (None, "1"), ("0", "0"), ("1", "0"), ("x", "1")])
def test_d2_gives_up_after_90_seconds(gpu, cap, alloc):
    gpu.first(kc(f"describe node {GPU1} ", stdout=describe(GPU1, cap, alloc)))
    result, text = gpu.check("check_d2_gpu_capacity")
    assert result == "FAIL", text
    assert len([a for a in gpu.kubectl() if a[:2] == ["describe", "node"]]) == 10  # every 10 s for 90 s


def test_d2_checks_every_gpu_node(tmp_path):
    env = two_gpus(tmp_path)
    env.first(kc(f"describe node {GPU2} ", stdout=describe(GPU2, None, None)))
    result, text = env.check("check_d2_gpu_capacity")
    assert result == "FAIL"
    assert f"D2: {GPU1}: nvidia.com/gpu capacity 1, allocatable 1" in text and GPU2 in text


# ---- D3 device plugin -------------------------------------------------------------------------

def test_d3_finds_the_registration_in_the_plugin_logs(gpu):
    result, text = gpu.check("check_d3_device_plugin")
    assert result == "PASS", text
    assert ["-n", "nvidia-device-plugin", "logs", "-l", "app.kubernetes.io/name=nvidia-device-plugin",
            "--tail=100", "--request-timeout=60s"] in gpu.kubectl()
    assert "-o" in [a for a in gpu.kubectl() if "get" in a and "nvidia-device-plugin" in a][0]


@pytest.mark.parametrize("answer", [
    kc("-n nvidia-device-plugin logs", stdout="I1022 server.go:165] Starting GRPC server for 'nvidia.com/gpu'\n"),
    kc("-n nvidia-device-plugin logs", rc=1, stderr="error: no pods found\n", stdout=REGISTERED),
])
def test_d3_fails_without_the_registration(gpu, answer):
    gpu.first(answer)
    assert gpu.check("check_d3_device_plugin")[0] == "FAIL"


# ---- D4 gpu-smoke / D5 vectoradd --------------------------------------------------------------

def pod_calls(env, pod):
    return [a for a in gpu_test_calls(env) if pod in " ".join(a)]


def delete(pod):
    """Bounded twice: the wait for the deletion and, through a stalled tunnel, the request (review X0-3)."""
    return ["-n", "gpu-test", "delete", "pod", pod, "--ignore-not-found", "--timeout=120s", "--request-timeout=60s"]


def phase_polls(env, pod):
    return [a for a in pod_calls(env, pod) if a[-1] == "jsonpath={.status.phase}"]


@pytest.mark.parametrize("func, pod", [("check_d4_gpu_smoke", "gpu-smoke"), ("check_d5_vectoradd", "vectoradd")])
def test_d4_d5_delete_apply_wait_logs_then_delete(gpu, func, pod):
    result, text = gpu.check(func)
    assert result == "PASS", text
    calls = pod_calls(gpu, pod)
    assert calls[0] == delete(pod) and calls[-1] == delete(pod)
    assert calls[1][:2] == ["apply", "-f"] and calls[1][2].endswith(f"/k8s/gpu-test/{pod}.yaml")
    assert calls[2] == ["-n", "gpu-test", "get", "pod", pod, "--request-timeout=60s", "-o", "jsonpath={.status.phase}"]
    assert ["-n", "gpu-test", "logs", pod, "--request-timeout=60s"] in calls


@pytest.mark.parametrize("func, pod", [("check_d4_gpu_smoke", "gpu-smoke"), ("check_d5_vectoradd", "vectoradd")])
def test_d4_d5_wait_up_to_300_seconds_for_succeeded(gpu, func, pod):
    gpu.first(kc(f"-n gpu-test get pod {pod}", "phase", seq=[{"stdout": "Pending"}, {"stdout": "Running"},
                                                              {"stdout": "Succeeded"}]))
    result, text = gpu.check(func)
    assert result == "PASS", text
    assert len(phase_polls(gpu, pod)) == 3 and text.count("(wait 5s skipped)") == 2


@pytest.mark.parametrize("func, pod", [("check_d4_gpu_smoke", "gpu-smoke"), ("check_d5_vectoradd", "vectoradd")])
def test_d4_d5_give_up_after_300_seconds(gpu, func, pod):
    gpu.first(kc(f"-n gpu-test get pod {pod}", "phase", stdout="Pending"))
    result, text = gpu.check(func)
    assert result == "FAIL", text
    assert len(phase_polls(gpu, pod)) == 61  # every 5 s for 300 s
    assert f"{pod} did not succeed within 300 s (phase 'Pending')" in text
    assert pod_calls(gpu, pod)[-1] == delete(pod)


@pytest.mark.parametrize("func, pod", [("check_d4_gpu_smoke", "gpu-smoke"), ("check_d5_vectoradd", "vectoradd")])
def test_d4_d5_stop_waiting_when_the_pod_failed(gpu, func, pod):
    """A Pod that Failed (non-root nvidia-smi denied, spec section 12 #9) never succeeds: no 5 more billed minutes."""
    gpu.first(kc(f"-n gpu-test get pod {pod}", "phase", seq=[{"stdout": "Pending"}, {"stdout": "Failed"}]))
    result, text = gpu.check(func)
    assert result == "FAIL", text
    assert len(phase_polls(gpu, pod)) == 2 and f"{pod} Failed" in text
    assert ["-n", "gpu-test", "describe", "pod", pod, "--request-timeout=60s"] in gpu.kubectl()
    assert pod_calls(gpu, pod)[-1] == delete(pod)


def test_d4_needs_nvidia_smi_and_the_gpu_type_of_the_node(gpu):
    result, text = gpu.check("check_d4_gpu_smoke")
    assert result == "PASS", text
    assert ["get", "node", GPU1, "--request-timeout=60s", "-o", "jsonpath={.metadata.labels.gpu-type}"] in gpu.kubectl()
    assert "NVIDIA-SMI" in text and "'A10'" in text


@pytest.mark.parametrize("change", [
    dict(smi=SMI.replace("NVIDIA-SMI", "NVIDIA SMI")),                 # no nvidia-smi table
    dict(gpu_type="a6000"),                                            # the label says A6000, the GPU is an A10
    dict(smi=SMI.replace("NVIDIA A10 ", "NVIDIA A100-SXM4-40GB")),     # A10 is only part of A100
    dict(gpu_type=""),                                                 # no label: nothing to compare
])
def test_d4_fails_on_a_wrong_output_and_still_deletes(tmp_path, change):
    env = GpuCluster(tmp_path, **change)
    result, text = env.check("check_d4_gpu_smoke")
    assert result == "FAIL", text
    assert pod_calls(env, "gpu-smoke")[-1] == delete("gpu-smoke")


def test_d4_matches_the_first_part_of_a_long_gpu_type_label(tmp_path):
    # gpu_type_label("gpu_1x_a100_sxm4") is a100_sxm4; nvidia-smi says "NVIDIA A100-SXM4-40GB"
    env = GpuCluster(tmp_path, gpu_type="a100_sxm4", smi=SMI.replace("NVIDIA A10 ", "NVIDIA A100-SXM4-40GB"))
    assert env.check("check_d4_gpu_smoke")[0] == "PASS"


@pytest.mark.parametrize("func, pod", [("check_d4_gpu_smoke", "gpu-smoke"), ("check_d5_vectoradd", "vectoradd")])
@pytest.mark.parametrize("answer", ["phase", "apply", "logs"])
def test_d4_d5_fail_when_a_step_fails_and_still_delete(gpu, func, pod, answer):
    has = {"phase": (f"-n gpu-test get pod {pod}", "phase"), "apply": ("apply -f", f"{pod}.yaml"),
           "logs": ("-n gpu-test logs", pod)}[answer]
    gpu.first(kc(*has, rc=1, stderr="error: the server is currently unable to handle the request\n"))
    result, text = gpu.check(func)
    assert result == "FAIL", text
    assert pod_calls(gpu, pod)[-1] == delete(pod)


def test_d4_shows_the_pod_when_it_does_not_succeed(gpu):
    gpu.first(kc("-n gpu-test get pod gpu-smoke", "phase", stdout="Pending"))
    assert gpu.check("check_d4_gpu_smoke")[0] == "FAIL"
    assert ["-n", "gpu-test", "describe", "pod", "gpu-smoke", "--request-timeout=60s"] in gpu.kubectl()


def test_d5_needs_test_passed(gpu):
    gpu.first(kc("-n gpu-test logs vectoradd", stdout="Test FAILED\n"))
    assert gpu.check("check_d5_vectoradd")[0] == "FAIL"


# ---- D6 overflow ------------------------------------------------------------------------------

HOLD_DELETE = ["-n", "gpu-test", "delete", "pod", "-l", "app=gpu-hold", "--ignore-not-found", "--timeout=120s",
               "--request-timeout=60s"]
OVERFLOW_DELETE = delete("gpu-overflow")


def creates(env):
    return [a for a in gpu_test_calls(env) if "create" in a]


def test_d6_holds_every_gpu_then_overflow_stays_pending(gpu):
    result, text = gpu.check("check_d6_overflow")
    assert result == "PASS", text
    calls = gpu_test_calls(gpu)
    assert calls[:2] == [HOLD_DELETE, OVERFLOW_DELETE] and calls[-2:] == [HOLD_DELETE, OVERFLOW_DELETE]
    assert len(creates(gpu)) == 1 and creates(gpu)[0][4].endswith("/k8s/gpu-test/gpu-hold.yaml")
    assert ["-n", "gpu-test", "wait", "pod", "-l", "app=gpu-hold", "--for=condition=Ready", "--timeout=300s"] in calls
    order = [" ".join(a) for a in calls]
    wait = next(i for i, c in enumerate(order) if "wait pod -l app=gpu-hold" in c)
    overflow = next(i for i, c in enumerate(order) if "gpu-overflow.yaml" in c)
    assert next(i for i, c in enumerate(order) if "create" in c) < wait < overflow
    assert "Insufficient nvidia.com/gpu" in text
    # the events of this gpu-overflow only, not those an earlier run's left behind (review S0-2)
    assert ["-n", "gpu-test", "get", "events", "--field-selector=involvedObject.kind=Pod,"
            f"involvedObject.name=gpu-overflow,involvedObject.uid={OVERFLOW_UID}", "--request-timeout=60s"] in calls
    # brief: gpu-overflow is looked at 30 s after it was applied (review S0-3)
    lines = text.splitlines()
    applied = next(i for i, line in enumerate(lines) if line.startswith("$ kc apply -f") and "gpu-overflow" in line)
    waited = lines.index("(wait 30s skipped)")
    phase = next(i for i, line in enumerate(lines) if "get pod gpu-overflow" in line and "status.phase" in line)
    events = next(i for i, line in enumerate(lines) if "get events" in line)
    assert applied < waited < phase < events, lines


def test_d6_creates_one_holder_per_allocatable_gpu_on_every_node(tmp_path):
    env = two_gpus(tmp_path, allocatable={GPU1: "1", GPU2: "2"})
    result, text = env.check("check_d6_overflow")
    assert result == "PASS", text
    assert len(creates(env)) == 3


@pytest.mark.parametrize("answer", [
    kc("-n gpu-test get pod gpu-overflow", "phase", stdout="Running"),
    kc("-n gpu-test get events", stdout="No resources found in gpu-test namespace.\n"),
    # the Insufficient event belongs to an earlier gpu-overflow: this one has another uid
    kc("-n gpu-test get pod gpu-overflow", "{.metadata.uid}", stdout="7a1b2c3d-0000-4000-8000-000000000001"),
    kc("-n gpu-test get pod gpu-overflow", "{.metadata.uid}", stdout=""),
    kc("-n gpu-test wait pod -l app=gpu-hold", rc=1, stderr="error: timed out waiting for the condition\n"),
    kc("-n gpu-test get pods -l app=gpu-hold", "status.phase=Running", stdout=""),   # not all holders run
    kc("-n gpu-test create -f", rc=1, stderr="Error from server (Forbidden): pods is forbidden\n"),
    kc("apply -f", "gpu-overflow.yaml", rc=1, stderr="error: the server is currently unable to handle the request\n"),
])
def test_d6_fails_and_still_deletes_holders_and_overflow(gpu, answer):
    gpu.first(answer)
    result, text = gpu.check("check_d6_overflow")
    assert result == "FAIL", text
    assert gpu_test_calls(gpu)[-2:] == [HOLD_DELETE, OVERFLOW_DELETE]


@pytest.mark.parametrize("alloc", ["0", ""])
def test_d6_fails_without_an_allocatable_gpu_and_creates_nothing(tmp_path, alloc):
    env = GpuCluster(tmp_path, allocatable={GPU1: alloc})
    result, text = env.check("check_d6_overflow")
    assert result == "FAIL", text
    assert creates(env) == []


def test_d6_failed_pre_clean_is_a_fail(gpu):
    gpu.first(kc("-n gpu-test delete pod -l app=gpu-hold", rc=1, stderr="error: timed out\n"))
    result, text = gpu.check("check_d6_overflow")
    assert result == "FAIL" and "cannot delete the gpu-hold and gpu-overflow Pods left from an earlier run" in text
    assert creates(gpu) == []


def test_d6_passing_body_with_a_failed_final_cleanup_is_a_fail(gpu):  # review C0-6
    gpu.first(kc("-n gpu-test delete pod -l app=gpu-hold", seq=[{"rc": 0}, {"rc": 1, "stderr": "error: timed out\n"}]))
    result, text = gpu.check("check_d6_overflow")
    assert result == "FAIL", text
    assert "D6: gpu-overflow is Pending with 'Insufficient nvidia.com/gpu'" in text  # the body passed
    assert "cleanup failed" in text
    assert gpu_test_calls(gpu)[-2:] == [HOLD_DELETE, OVERFLOW_DELETE]


# ---- _vr_guarded: the cleanup also runs when a check is cut short --------------------------------

def guarded(env, body):
    code = ("source scripts/lib/checks.sh; cleanup() { echo CLEANUP; }; "
            f"body() {{ {body}; }}; _vr_guarded cleanup body; echo rc=$?")
    return env.run(["bash", "-c", code])


@pytest.mark.parametrize("body, rc", [("echo ok", 0), ("return 1", 1), ("exit 2", 2), ("set -e; false; echo x", 1),
                                      ("kill -TERM $BASHPID", 143)])
def test_guarded_runs_the_cleanup_however_the_body_ends(no_gpu, body, rc):
    proc = guarded(no_gpu, body)
    assert proc.stdout.splitlines()[-2:] == ["CLEANUP", f"rc={rc}"], proc.stdout + proc.stderr


def test_guarded_cleanup_failure_fails_a_passing_body(no_gpu):
    code = ("source scripts/lib/checks.sh; cleanup() { return 3; }; body() { echo ok; }; rc=0; "
            "_vr_guarded cleanup body || rc=$?; echo rc=$rc")
    assert no_gpu.run(["bash", "-c", code]).stdout.splitlines()[-1] == "rc=1"


def test_guarded_keeps_the_callers_exit_trap(no_gpu):
    code = ("source scripts/lib/checks.sh; trap 'echo OUTER' EXIT; cleanup() { echo CLEANUP; }; "
            "body() { exit 2; }; _vr_guarded cleanup body || true; echo after")
    assert no_gpu.run(["bash", "-c", code]).stdout.splitlines() == ["CLEANUP", "after", "OUTER"]


def test_d6_check_killed_while_waiting_still_deletes_the_holders(gpu):
    """A check stopped by a signal (make verify interrupted) while D6 waits for its holders still deletes them: a
    leftover gpu-hold keeps a billed GPU allocated (Task 16 ledger note). Here kc signals its own shell at the wait."""
    run = gpu.repo / "verification" / "t"
    code = (f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; "
            "kc() { if [[ \"$*\" == *'wait pod -l app=gpu-hold'* ]]; then kill -TERM $BASHPID; fi; "
            "kubectl --kubeconfig \"$KUBECONFIG_FILE\" \"$@\"; }; run_check T title check_d6_overflow")
    proc = gpu.run(["bash", "-c", code], VR_NO_WAIT="1")
    assert proc.returncode == 0, proc.stderr
    assert "| 1 | T | title | FAIL |" in (run / "SUMMARY.md").read_text()
    calls = gpu_test_calls(gpu)
    assert not any("wait" in a for a in calls)  # killed before kubectl wait ran
    assert calls[-2:] == [HOLD_DELETE, OVERFLOW_DELETE], calls


def test_guarded_cleanup_runs_after_a_signal_even_when_its_output_pipe_is_gone(no_gpu):  # review C0-3 / X0-2
    """vr_cmd runs nothing it cannot record; after a signal a dead output pipe must not stop the cleanup."""
    mark = no_gpu.repo / "cleaned"
    code = ("source scripts/lib/checks.sh; "
            f"cleanup() {{ vr_cmd touch {mark}; }}; body() {{ sleep 0.5; kill -INT $BASHPID; }}; "
            "_vr_guarded cleanup body")
    proc = no_gpu.run(["bash", "-c", f"bash -c {shlex.quote(code)} | true"])
    assert mark.exists(), proc.stderr


def test_ctrl_c_of_the_whole_run_still_deletes_the_holders_and_stops_the_run(gpu):  # review C0-3 / X0-2
    """A terminal Ctrl-C sends SIGINT to the whole foreground process group: the check, kubectl, and run_check's
    evidence filter (redact). The filter must outlive the check's cleanup, or the cleanup's logging fails and the
    gpu-hold Pods (restartPolicy Always) keep every GPU allocated. The run then stops: no later check starts."""
    run = gpu.repo / "verification" / "t"
    waiting = gpu.repo / "waiting"
    code = (f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; "
            "kc() { if [[ \"$*\" == *'wait pod -l app=gpu-hold'* ]]; then "
            f"touch {waiting}; sleep 30; fi; "
            "kubectl --kubeconfig \"$KUBECONFIG_FILE\" \"$@\"; }; "
            "run_check T title check_d6_overflow; run_check U later check_d1_gpu_stack")
    proc = subprocess.Popen(["bash", "-c", code], cwd=gpu.repo, env=gpu.environ({"VR_NO_WAIT": "1"}),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        for _ in range(200):
            if waiting.exists():
                break
            time.sleep(0.05)
        assert waiting.exists(), "D6 never reached the holder wait"
        time.sleep(0.2)
        os.killpg(proc.pid, signal.SIGINT)
        proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    calls = gpu_test_calls(gpu)
    assert calls[-2:] == [HOLD_DELETE, OVERFLOW_DELETE], calls
    evidence = (run / "01-T.txt").read_text()
    # the evidence filter survived: the cleanup is on record
    assert "interrupted (SIGINT); cleaning up" in evidence and "redact failed" not in evidence, evidence
    assert "delete pod -l app=gpu-hold" in evidence.split("interrupted (SIGINT); cleaning up")[1], evidence
    assert not (run / "02-U.txt").exists()  # the run stopped
    assert "| 1 | T | title | FAIL |" in (run / "SUMMARY.md").read_text()  # after recording the check
    assert proc.returncode == -signal.SIGINT  # by the signal, as without the traps: make stops too


# ---- make verify with a GPU node --------------------------------------------------------------

def test_make_verify_passes_a_to_d_with_a_gpu_node(gpu):
    port = free_port()
    proc = gpu.run(["bash", str(gpu.repo / "scripts/verify.sh")], VR_NO_WAIT="1", VR_PF_PORT=str(port),
                   **{"BASH_FUNC_timeout%%": NOT_REACHABLE, "TZ": "America/Los_Angeles"})
    [run] = list((gpu.repo / "verification").iterdir())
    summary = (run / "SUMMARY.md").read_text()
    assert proc.returncode == 0, summary + proc.stdout + proc.stderr
    results = re.findall(r"^\| \d+ \| (\w+) \| [^|]+ \| (\w+) \|$", summary, re.M)
    assert results == [(i, "PASS") for _, i, _ in EXPECTED]
    assert gpu_test_calls(gpu)[-2:] == [HOLD_DELETE, OVERFLOW_DELETE]
