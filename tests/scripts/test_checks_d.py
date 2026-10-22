"""Offline tests for the GPU checks D1-D6 (Task 21) in scripts/lib/checks.sh.

Same harness as test_checks_abc.py: a repo copy whose kubectl, ssh and scripts/gpuctl are tests/scripts/fake_tools.py,
and a scripted healthy cluster. Here it has a GPU node, k3s-gpu-a6000-1, as the early live session measured it
(2026-10-21, progress.md "Task 21 early live session DONE"): the A6000 pool fell back to an A10, so the node is
labelled gpu-type=a10 and nvidia-smi prints "NVIDIA A10". Each test breaks one thing and runs one check through
run_check, as make verify does. Nothing here reaches Lambda, a cluster or the network.
"""
import re

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
        kc("-n gpu-test wait pod/", stdout="pod/x condition met\n"),
        kc("-n gpu-test get pod gpu-smoke", "nodeName", stdout=gpus[0]),
        kc("-n gpu-test get pod vectoradd", "nodeName", stdout=gpus[0]),
        kc("-n gpu-test logs gpu-smoke", stdout=smi),
        kc("-n gpu-test logs vectoradd", stdout=VECTORADD),
        kc("-n gpu-test create -f", "gpu-hold.yaml", stdout="pod/gpu-hold-x1y2z created\n"),
        kc("-n gpu-test wait pod -l app=gpu-hold", stdout="pod/gpu-hold-x1y2z condition met\n"),
        kc("-n gpu-test get pods -l app=gpu-hold", "status.phase=Running",
           stdout="".join(f"gpu-hold-{i}\n" for i in range(sum(int(v or 0) for v in allocatable.values())))),
        kc("-n gpu-test get pod gpu-overflow", "phase", stdout="Pending"),
        kc("-n gpu-test get events", "involvedObject.name=gpu-overflow", stdout=EVENTS),
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


@pytest.mark.parametrize("func, pod", [("check_d4_gpu_smoke", "gpu-smoke"), ("check_d5_vectoradd", "vectoradd")])
def test_d4_d5_delete_apply_wait_logs_then_delete(gpu, func, pod):
    result, text = gpu.check(func)
    assert result == "PASS", text
    calls = pod_calls(gpu, pod)
    delete = ["-n", "gpu-test", "delete", "pod", pod, "--ignore-not-found", "--timeout=120s"]
    assert calls[0] == delete and calls[-1] == delete
    assert calls[1][:2] == ["apply", "-f"] and calls[1][2].endswith(f"/k8s/gpu-test/{pod}.yaml")
    assert ["-n", "gpu-test", "wait", f"pod/{pod}", "--for=jsonpath={.status.phase}=Succeeded",
            "--timeout=300s"] in calls
    assert ["-n", "gpu-test", "logs", pod, "--request-timeout=60s"] in calls


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
    assert pod_calls(env, "gpu-smoke")[-1] == ["-n", "gpu-test", "delete", "pod", "gpu-smoke", "--ignore-not-found",
                                              "--timeout=120s"]


def test_d4_matches_the_first_part_of_a_long_gpu_type_label(tmp_path):
    # gpu_type_label("gpu_1x_a100_sxm4") is a100_sxm4; nvidia-smi says "NVIDIA A100-SXM4-40GB"
    env = GpuCluster(tmp_path, gpu_type="a100_sxm4", smi=SMI.replace("NVIDIA A10 ", "NVIDIA A100-SXM4-40GB"))
    assert env.check("check_d4_gpu_smoke")[0] == "PASS"


@pytest.mark.parametrize("func, pod", [("check_d4_gpu_smoke", "gpu-smoke"), ("check_d5_vectoradd", "vectoradd")])
@pytest.mark.parametrize("answer", ["wait", "apply", "logs"])
def test_d4_d5_fail_when_a_step_fails_and_still_delete(gpu, func, pod, answer):
    has = {"wait": ("-n gpu-test wait", f"pod/{pod}"), "apply": ("apply -f", f"{pod}.yaml"),
           "logs": ("-n gpu-test logs", pod)}[answer]
    gpu.first(kc(*has, rc=1, stderr="error: timed out waiting for the condition\n"))
    result, text = gpu.check(func)
    assert result == "FAIL", text
    assert pod_calls(gpu, pod)[-1] == ["-n", "gpu-test", "delete", "pod", pod, "--ignore-not-found", "--timeout=120s"]


def test_d4_shows_the_pod_when_it_does_not_succeed(gpu):
    gpu.first(kc("-n gpu-test wait", "pod/gpu-smoke", rc=1, stderr="error: timed out waiting for the condition\n"))
    assert gpu.check("check_d4_gpu_smoke")[0] == "FAIL"
    assert ["-n", "gpu-test", "describe", "pod", "gpu-smoke", "--request-timeout=60s"] in gpu.kubectl()


def test_d5_needs_test_passed(gpu):
    gpu.first(kc("-n gpu-test logs vectoradd", stdout="Test FAILED\n"))
    assert gpu.check("check_d5_vectoradd")[0] == "FAIL"


# ---- D6 overflow ------------------------------------------------------------------------------

HOLD_DELETE = ["-n", "gpu-test", "delete", "pod", "-l", "app=gpu-hold", "--ignore-not-found", "--timeout=120s"]
OVERFLOW_DELETE = ["-n", "gpu-test", "delete", "pod", "gpu-overflow", "--ignore-not-found", "--timeout=120s"]


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


def test_d6_creates_one_holder_per_allocatable_gpu_on_every_node(tmp_path):
    env = two_gpus(tmp_path, allocatable={GPU1: "1", GPU2: "2"})
    result, text = env.check("check_d6_overflow")
    assert result == "PASS", text
    assert len(creates(env)) == 3


@pytest.mark.parametrize("answer", [
    kc("-n gpu-test get pod gpu-overflow", "phase", stdout="Running"),
    kc("-n gpu-test get events", stdout="No resources found in gpu-test namespace.\n"),
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


def test_d6_failed_cleanup_is_a_fail(gpu):
    gpu.first(kc("-n gpu-test delete pod -l app=gpu-hold", rc=1, stderr="error: timed out\n"))
    result, text = gpu.check("check_d6_overflow")
    assert result == "FAIL" and "gpu-hold" in text


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
