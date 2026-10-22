"""Offline tests for the resilience checks E1-E3 (Task 22b) in scripts/lib/checks.sh and their registration in
scripts/verify.sh (make verify RESILIENCE=1).

Same harness as test_checks_abc.py and test_checks_d.py: a repo copy whose kubectl, gcloud, ssh, terraform and
scripts/gpuctl are tests/scripts/fake_tools.py, and a scripted healthy cluster with one GPU node (k3s-gpu-a6000-1).
Each test scripts what the reboot, the GPU node replacement or the firewall drift looks like, breaks one thing and
runs one check through run_check, as make verify does. Nothing here reboots, launches, deletes or applies anything
real: gpuctl and terraform are the fakes, and `timeout` is the real one (the checks bound gpuctl's reads with it).
"""
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time

import shutil

import pytest

from test_checks_abc import CP, CPU1, GPU1, NODES, Cluster, kc, on, rule
from test_checks_d import GPU2, GpuCluster
from test_ops_scripts import FAKES, PROJECT, ROOT, free_port

REAL_TIMEOUT = '() { command timeout "$@"; }'  # B5's tests fake `timeout`; E uses the real one
READY = 'Ready")].status'
TRANSITION = 'Ready")].lastTransitionTime'
STARTED = "running.startedAt"


# k3s-cpu-1's clock: its node lease's last renewal just before the reboot command (a MicroTime, as kubectl prints
# it). Fixed, and far from the operator's clock, so no test depends on when it runs or on the clocks agreeing: E1
# judges the node's timestamps by the node's own clock (Task 22b review X1-2, C1-3).
NODE_NOW = datetime.datetime(2025, 3, 4, 5, 6, 7, 123456, tzinfo=datetime.timezone.utc)


def iso(offset_s, now=NODE_NOW):
    """A Kubernetes timestamp OFFSET_S seconds after NOW (UTC, whole seconds, Z), as the API server prints it."""
    return (now + datetime.timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%SZ")


def micro(now=NODE_NOW):
    """NOW as a MicroTime (a lease's renewTime)."""
    return now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


RENEW = micro()
# T0: Ready's lastTransitionTime before the reboot (a day before); T1: after it, within E1's 120 s window.
T0, T1 = iso(-86400), iso(70)
BOOT1, BOOT2 = "6b1f3c1e-0d2a-4e5b-9c8d-7e6f5a4b3c21", "f0e1d2c3-b4a5-4968-8776-655443322110"
UID1, UID2 = "11111111-2222-4333-8444-555555555555", "99999999-8888-4777-8666-555555555555"
GPU_IP, NEW_IP = NODES[GPU1]["ext"], "203.0.113.44"
EXTRA = "198.51.100.99"
LIST_BEFORE = f"{GPU1} {GPU_IP} active gpu_1x_a10 us-west-1\n"
LIST_AFTER = f"{GPU1} {NEW_IP} active gpu_1x_a10 us-east-1\n"
CAPACITY = ("Capacity (in the order gpu-apply tries them):\nPOOL   TYPE          REGION     AVAILABLE  PRICE\n"
            "a6000  gpu_1x_a6000  us-west-1  no         $0.80/h\n"
            "a6000  gpu_1x_a10    us-east-1  yes        $0.75/h\n")
STATUS_HEAD = (CAPACITY + "\nGPU instances (k3s-gpu-*): 1\n" + LIST_BEFORE +
               "\nDrift check (Lambda instances, cluster nodes, gpu-access state, k3s-cp k3s_peers):\n")
STATUS_OK = STATUS_HEAD + "  No drift\n"
STATUS_DRIFT = (STATUS_HEAD + f"  SECURITY: firewall rule for {EXTRA} has no live instance\n"
                "  1 problem(s); make gpu-apply reconciles them to the spec; make gpu-down removes every GPU node "
                "and its allow-list entries\n")
ORIG = json.dumps([GPU_IP])
WITH_EXTRA = json.dumps([GPU_IP, EXTRA])
INJECT_VAR = f'-var=gpu_ips=["{GPU_IP}","{EXTRA}"]'
RESTORE_VAR = f'-var=gpu_ips=["{GPU_IP}"]'
E_EXPECTED = [("E", "E1", "check_e1_worker_reboot"), ("E", "E2", "check_e2_gpu_replace"),
              ("E", "E3", "check_e3_firewall_drift")]


def seq(*stdouts, rc=0):
    return [{"stdout": s, "rc": rc} for s in stdouts]


def gpuctl(*has, **kw):
    return rule("gpuctl", has, **kw)


def check(env, func, **extra):
    return env.check(func, timeout_fn=REAL_TIMEOUT, **extra)


def gpuctl_calls(env):
    """Each scripts/gpuctl call as its subcommand and options, e.g. "apply --auto-approve"; not `ip NAME`, the
    read-only lookup node_ssh makes for D1's SSH to a GPU node."""
    return [" ".join(a) for a in env.argvs("gpuctl") if a[:1] != ["ip"]]


def tf_applies(env):
    return [a for a in env.argvs("terraform") if a[1:2] == ["apply"]]


def polls(env, node):
    return [a for a in env.kubectl() if a[:3] == ["get", "node", node] and any(READY in x for x in a)]


def with_spec(env, count=1, spec=True):
    """The repo copy gets lambda/ (gpuctl's spec loader and gpu-nodes.yaml, its pool count set to COUNT; none when
    not SPEC) and .venv/bin/python (this interpreter, which has PyYAML): E2 reads the spec's GPU pools through them."""
    shutil.copytree(ROOT / "lambda/gpuctl", env.repo / "lambda/gpuctl",
                    ignore=shutil.ignore_patterns("__pycache__"))
    if spec:
        text = (ROOT / "lambda/gpu-nodes.yaml").read_text()
        text, n = re.subn(r"^(    count: )\d+", rf"\g<1>{count}", text, flags=re.M)
        assert n == 1
        (env.repo / "lambda/gpu-nodes.yaml").write_text(text)
    (env.repo / ".venv/bin").mkdir(parents=True)
    (env.repo / ".venv/bin/python").symlink_to(sys.executable)
    return env


def wait_for_file(path, seconds=120):
    """Wait for PATH (a fake's start marker); generous, since it only bounds a liveness wait (review S1-2)."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


# ---- E1: a CPU worker reboots ----------------------------------------------------------------

def e1(env, ready=("True", "True", "Unknown", "Unknown", "True"), boots=(BOOT1, BOOT2), reboot_rc=255,
       transitions=(T0, T1), started=None, renew=RENEW):
    """k3s-cpu-1 is Ready, goes NotReady after the reboot command and comes back Ready with a new boot ID; its
    nettest probe and echo containers (re)started after the reboot command (STARTED: their startedAt answers).
    RENEW: its node lease's renewTime just before the reboot (the node's clock)."""
    started = started or (iso(90),)
    env.first(kc(f"-n kube-node-lease get lease {CPU1} ", "renewTime", stdout=renew),
              kc(f"get node {CPU1} ", READY, seq=seq(*ready)),
              kc(f"get node {CPU1} ", TRANSITION, seq=seq(*transitions)),
              kc(f"get node {CPU1} ", "bootID", seq=seq(*boots)),
              kc("-n nettest get pods", f"spec.nodeName={CPU1},", STARTED, seq=seq(*(f"true {s}\n" for s in started))),
              on(CPU1, "sudo systemctl reboot", rc=reboot_rc,
                 stderr="Connection to compute.1234567890 closed by remote host.\n" if reboot_rc == 255 else ""),
              kc("-n nettest rollout status", stdout='daemon set "probe" successfully rolled out\n'))
    return env


def started_polls(env):
    return [a for a in env.kubectl() if a[:3] == ["-n", "nettest", "get"] and any(STARTED in x for x in a)]


@pytest.fixture
def env(tmp_path):
    return Cluster(tmp_path)


@pytest.fixture
def gpu(tmp_path):
    return with_spec(GpuCluster(tmp_path))


@pytest.fixture
def no_gpu(tmp_path):
    return Cluster(tmp_path)


def test_e1_reboots_the_worker_sees_notready_then_ready_and_runs_b1(env):
    result, text = check(e1(env), "check_e1_worker_reboot")
    assert result == "PASS", text
    assert env.remote(CPU1)[0] == "sudo systemctl reboot"
    lines = text.splitlines()
    reboot = next(i for i, line in enumerate(lines) if "sudo systemctl reboot" in line)
    notready = next(i for i, line in enumerate(lines) if "NotReady" in line and line.startswith("E1"))
    ready = next(i for i, line in enumerate(lines) if "Ready again" in line)
    rollout = next(i for i, line in enumerate(lines) if "rollout status ds/probe" in line)
    restarted = next(i for i, line in enumerate(lines) if "restarted after the reboot" in line)
    b1 = next(i for i, line in enumerate(lines) if line.startswith("B1: N x N matrix"))
    assert reboot < notready < ready < rollout < restarted < b1, lines
    assert ["-n", "nettest", "rollout", "status", "ds/echo", "--timeout=300s"] in env.kubectl()
    assert len(polls(env, CPU1)) == 5                   # before, 2 until NotReady, 2 until Ready
    assert len(started_polls(env)) == 2                 # the probe and the echo Pod on k3s-cpu-1
    assert BOOT1 in text and BOOT2 in text
    # every kubectl request of the waits is bounded (KC_REQUEST_TIMEOUT, 60 s), and never beyond the time left
    for argv in polls(env, CPU1) + started_polls(env):
        assert any(re.fullmatch(r"--request-timeout=([1-9]|[1-5][0-9]|60)s", a) for a in argv), argv


def test_e1_accepts_a_reboot_command_that_returns_0(env):
    assert check(e1(env, reboot_rc=0), "check_e1_worker_reboot")[0] == "PASS"


def test_e1_reboot_command_that_fails_but_reboots_the_node_passes(env):
    """An IAP SSH can exit with neither 0 nor 255 when the tunnel drops under the reboot: the boot ID decides."""
    result, text = check(e1(env, reboot_rc=1), "check_e1_worker_reboot")
    assert result == "PASS", text
    assert "exit 1" in text and BOOT2 in text


def test_e1_reboot_command_that_fails_still_waits_then_fails_when_the_node_did_not_reboot(env):
    result, text = check(e1(env, reboot_rc=1, ready=("True",), transitions=(T0,), boots=(BOOT1, BOOT1)),
                         "check_e1_worker_reboot")
    assert result == "FAIL" and "the reboot command failed (exit 1)" in text
    assert len(polls(env, CPU1)) == 1 + 60              # the whole 120 s window, every 2 s: E2 must not start early
    assert "B1: N x N matrix" not in text


@pytest.mark.parametrize("state", ["False", "Unknown", ""])
def test_e1_refuses_to_reboot_a_worker_that_is_not_ready(env, state):
    result, text = check(e1(env, ready=(state,)), "check_e1_worker_reboot")
    assert result == "FAIL" and "not Ready before the reboot" in text
    assert "sudo systemctl reboot" not in env.remote(CPU1)


def test_e1_fails_when_the_worker_never_goes_notready_within_120_s(env):
    result, text = check(e1(env, ready=("True",), transitions=(T0,)), "check_e1_worker_reboot")
    assert result == "FAIL" and "never NotReady within 120 s" in text
    # bounded: every 2 s for 120 s after the reboot command (VR_NO_WAIT: the clock is the skipped sleeps)
    assert len(polls(env, CPU1)) == 1 + 60
    # the evidence says whether the node rebooted anyway (faster than the node-monitor grace period)
    assert "boot ID changed" in text
    assert "B1: N x N matrix" not in text


def test_e1_says_the_worker_did_not_reboot_when_it_never_goes_notready_and_keeps_its_boot_id(env):
    result, text = check(e1(env, ready=("True",), transitions=(T0,), boots=(BOOT1, BOOT1)), "check_e1_worker_reboot")
    assert result == "FAIL" and "never NotReady within 120 s" in text
    assert "did not reboot" in text and "boot ID changed" not in text


def test_e1_counts_a_notready_between_two_polls_from_the_ready_condition_transition(env):
    """Ready=True at every poll, but the Ready condition's lastTransitionTime moved: it went NotReady and back."""
    result, text = check(e1(env, ready=("True",)), "check_e1_worker_reboot")
    assert result == "PASS", text
    assert f"lastTransitionTime {T0} -> {T1}" in text
    assert "B1: N x N matrix" in text


@pytest.mark.parametrize("offset, result", [(130, "PASS"), (131, "FAIL")])
def test_e1_bounds_a_ready_transition_by_the_node_lease_plus_its_renew_interval(env, offset, result):
    """The lease was renewed at most 10 s before the reboot command: 120 s after the command is the lease + 130 s."""
    got, text = check(e1(env, ready=("True",), transitions=(T0, iso(offset))), "check_e1_worker_reboot")
    assert got == result, text


@pytest.mark.parametrize("years", [-30, 70], ids=["node-clock-behind", "node-clock-ahead"])
def test_e1_judges_the_node_timestamps_by_the_node_clock_not_the_operator_clock(env, years):
    """A skewed operator clock (a laptop after sleep) must not fail a healthy reboot: the reference time is the node
    lease's renewTime, written by the kubelet on the same host clock as the containers' startedAt (review X1-2)."""
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365 * years)
    result, text = check(e1(env, ready=("True",), transitions=(iso(-86400, now), iso(70, now)), renew=micro(now),
                            started=(iso(-3600, now), iso(40, now))), "check_e1_worker_reboot")
    assert result == "PASS", text
    assert micro(now) in text                           # the evidence shows the node's clock


@pytest.mark.parametrize("answer", [{"rc": 1, "stderr": 'Error from server (NotFound): leases "k3s-cpu-1"\n'},
                                    {"stdout": ""}, {"stdout": "yesterday"}], ids=["not-found", "empty", "garbage"])
def test_e1_does_not_reboot_when_the_node_lease_cannot_be_read(env, answer):
    e1(env).first(kc(f"-n kube-node-lease get lease {CPU1} ", **answer))
    result, text = check(env, "check_e1_worker_reboot")
    assert result == "FAIL" and "lease" in text
    assert "sudo systemctl reboot" not in env.remote(CPU1)


def test_e1_does_not_count_a_ready_transition_later_than_120_s_after_the_reboot(env):
    """The brief wants NotReady seen within 120 s: a Ready transition after that window does not show it."""
    late = iso(3600)
    result, text = check(e1(env, ready=("True",), transitions=(T0, late)), "check_e1_worker_reboot")
    assert result == "FAIL", text
    assert late in text and "more than 120 s after the reboot command" in text
    assert "B1: N x N matrix" not in text


def test_e1_fails_when_the_worker_does_not_come_back_within_600_s(env):
    result, text = check(e1(env, ready=("True", "Unknown")), "check_e1_worker_reboot")
    assert result == "FAIL" and "not Ready again within 600 s" in text
    # bounded: every 10 s until 600 s after the reboot command (VR_NO_WAIT: the clock is the skipped sleeps)
    assert len(polls(env, CPU1)) == 1 + 1 + 60
    assert "B1: N x N matrix" not in text
    # the evidence says how to bring the worker back
    assert f"gcloud compute instances reset {CPU1}" in text and "get-serial-port-output" in text


def test_e1_waits_for_the_nettest_pods_to_restart_before_b1(env):
    """A fast reboot can leave the Pods' Ready status stale: B1 waits for containers started after the reboot."""
    result, text = check(e1(env, started=(iso(-3600), iso(-3600), iso(90))), "check_e1_worker_reboot")
    assert result == "PASS", text
    assert len(started_polls(env)) == 2 + 2              # probe stale twice, then probe and echo restarted


def test_e1_fails_when_the_nettest_pods_never_restart(env):
    result, text = check(e1(env, started=(iso(-3600),)), "check_e1_worker_reboot")
    assert result == "FAIL" and "did not restart after the reboot within 300 s" in text
    assert len(started_polls(env)) == 60                # the probe Pod every 5 s for 300 s
    assert "B1: N x N matrix" not in text


def test_e1_fails_when_the_boot_id_did_not_change(env):
    result, text = check(e1(env, boots=(BOOT1, BOOT1)), "check_e1_worker_reboot")
    assert result == "FAIL" and "boot ID did not change" in text
    assert "B1: N x N matrix" not in text


def test_e1_fails_when_the_nettest_pods_do_not_come_back(env):
    e1(env).first(kc("-n nettest rollout status ds/echo", rc=1, stderr="error: timed out waiting for the condition\n"))
    result, text = check(env, "check_e1_worker_reboot")
    assert result == "FAIL" and "B1: N x N matrix" not in text
    assert "is Ready again" in text and "the nettest echo DaemonSet is not rolled out within 300 s" in text


def test_e1_fails_when_b1_fails_after_the_reboot(env):
    e1(env).first(kc("exec", "-- curl -fsS -m 5 http://10.42.", rc=7, stderr="curl: (7) Failed to connect\n"))
    result, text = check(env, "check_e1_worker_reboot")
    assert result == "FAIL" and "B1: N x N matrix" in text


# ---- E2: the GPU node is replaced ---------------------------------------------------------------

def e2(env, status=STATUS_OK, status_rc=0, lists=(LIST_BEFORE, "", LIST_AFTER), down_rc=0, apply_rc=0,
       uids=(UID1, UID2), ready="True"):
    """gpuctl down removes k3s-gpu-a6000-1, gpuctl apply brings it back on a new instance (new IP, new Node)."""
    env.first(gpuctl("status", stdout=status, rc=status_rc),
              gpuctl("list-nodes", seq=seq(*lists)),
              gpuctl("down", "--auto-approve", **({"seq": down_rc} if isinstance(down_rc, list) else {"rc": down_rc})),
              gpuctl("apply", "--auto-approve", rc=apply_rc),
              kc(f"get node {GPU1} ", "metadata.uid", seq=seq(*uids)),
              kc(f"get node {GPU1} ", READY, stdout=ready),
              kc("-n nettest rollout status", stdout='daemon set "probe" successfully rolled out\n'))
    return env


def test_e2_replaces_the_gpu_node_then_runs_every_d_check_and_b1(gpu):
    """Spec section 9 E2: the node rejoins, and the matrix and all of D pass on it."""
    result, text = check(e2(gpu), "check_e2_gpu_replace")
    assert result == "PASS", text
    assert gpuctl_calls(gpu) == ["status", "list-nodes", "down --auto-approve", "list-nodes",
                                 "apply --auto-approve", "list-nodes"]
    lines = text.splitlines()
    apply = next(i for i, line in enumerate(lines) if "gpuctl apply --auto-approve" in line)
    d1 = next(i for i, line in enumerate(lines) if line.startswith("D1: ") and "gpu-audit.txt has content" in line)
    d2 = next(i for i, line in enumerate(lines) if line.startswith("D2: "))
    d3 = next(i for i, line in enumerate(lines) if line.startswith("D3: the device plugin registered"))
    d4 = next(i for i, line in enumerate(lines) if line.startswith("D4: gpu-smoke ran"))
    d5 = next(i for i, line in enumerate(lines) if line.startswith("D5: vectoradd: Test PASSED"))
    d6 = next(i for i, line in enumerate(lines) if line.startswith("D6: gpu-overflow is Pending"))
    b1 = next(i for i, line in enumerate(lines) if line.startswith("B1: N x N matrix"))
    assert apply < d1 < d2 < d3 < d4 < d5 < d6 < b1, lines
    assert f"{GPU1}: {GPU_IP} -> {NEW_IP}" in text
    # the read-only gpuctl calls are bounded; down and apply bound their own steps
    assert re.search(r"^\$ timeout --foreground \d+ \S*scripts/gpuctl status$", text, re.M), text


@pytest.mark.parametrize("check_name, broken", [
    ("D1", kc("get runtimeclass nvidia-gpu", rc=1, stderr='Error from server (NotFound): runtimeclasses "nvidia-gpu"\n')),
    ("D3", kc("-n nvidia-device-plugin logs", stdout="I1022 starting\n")),
    ("D5", kc("-n gpu-test logs vectoradd", stdout="Test FAILED\n")),
    ("D6", kc("-n gpu-test get pod gpu-overflow", "phase", stdout="Running")),
])
def test_e2_fails_when_any_d_check_fails_on_the_new_node_and_still_runs_b1(gpu, check_name, broken):
    e2(gpu).first(broken)
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "FAIL", text
    assert "B1: N x N matrix" in text
    assert gpuctl_calls(gpu).count("down --auto-approve") == 1   # the replacement finished: nothing to sweep


def test_e2_waits_for_lambda_to_stop_listing_the_terminated_instance(gpu):
    """Lambda can list a terminated instance as active for a moment after the terminate call returns."""
    result, text = check(e2(gpu, lists=(LIST_BEFORE, LIST_BEFORE, LIST_BEFORE, "", LIST_AFTER)),
                         "check_e2_gpu_replace")
    assert result == "PASS", text
    assert gpuctl_calls(gpu) == ["status", "list-nodes", "down --auto-approve", "list-nodes", "list-nodes",
                                 "list-nodes", "apply --auto-approve", "list-nodes"]


def test_e2_reads_only_the_stdout_of_list_nodes(gpu):
    """A warning on stderr (urllib3, the wrapper's terraform init) goes into the evidence, never into the node list."""
    warning = "/usr/lib/python3/dist-packages/urllib3/connectionpool.py:1061: InsecureRequestWarning: x\n"
    e2(gpu)
    gpu.first(gpuctl("list-nodes", seq=[{"stdout": s, "stderr": warning} for s in (LIST_BEFORE, "", LIST_AFTER)]))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "PASS", text
    assert "InsecureRequestWarning" in text and "unexpected line" not in text


def test_e2_accepts_the_replacement_under_the_lowest_free_name(gpu):
    """gpuctl apply names a new node by the lowest free index: k3s-gpu-a6000-2 comes back as k3s-gpu-a6000-1."""
    before = f"{GPU2} 203.0.113.13 active gpu_1x_a10 us-west-1\n"
    e2(gpu, lists=(before, "", LIST_AFTER), uids=(UID2,))
    gpu.first(kc(f"get node {GPU2} ", "metadata.uid", stdout=UID1))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "PASS", text
    assert "more GPU instances" not in text
    assert f"{GPU1}" in text and NEW_IP in text


@pytest.mark.parametrize("rows", [
    "a6000  gpu_1x_a6000  us-west-1  no         $0.80/h\n",
    "a6000  gpu_1x_a6000  us-west-1  no         $0.80/h\na6000  gpu_1x_a10    us-east-1  no         $0.75/h\n",
], ids=["one-type", "every-fallback"])
def test_e2_skips_and_keeps_the_gpu_node_when_lambda_has_no_capacity_to_relaunch_it(gpu, rows):
    status = STATUS_OK.replace(CAPACITY, "Capacity (in the order gpu-apply tries them):\n"
                                         "POOL   TYPE          REGION     AVAILABLE  PRICE\n" + rows)
    result, text = check(e2(gpu, status=status), "check_e2_gpu_replace")
    assert result == "SKIP", text
    assert "no capacity" in text
    assert "down --auto-approve" not in gpuctl_calls(gpu)
    # the SKIP is not a pass: the evidence says so, and how to get E2 verified (review S1-1)
    assert "E2 is NOT verified" in text and "make gpu-down" in text and "make gpu-apply" in text


def test_e2_replaces_the_node_when_the_only_capacity_is_the_unit_it_holds(gpu):
    """Lambda reports no free capacity for the pool, but the node under test runs one of its candidate type/region
    pairs: terminating it frees that unit, so E2 replaces it (with a warning) instead of skipping (review C1-1)."""
    e2(gpu).first(gpuctl("status", seq=seq(held_status("no"), held_status("yes"))))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "PASS", text
    assert f"{GPU1} holds a gpu_1x_a10 in us-west-1" in text
    assert gpuctl_calls(gpu).count("down --auto-approve") == 1 and "apply --auto-approve" in gpuctl_calls(gpu)


def held_status(a10_west):
    """gpuctl status for the held-unit path: no free capacity for pool a6000, except (A10_WEST) the gpu_1x_a10 in
    us-west-1 that k3s-gpu-a6000-1 holds (LIST_BEFORE) once gpuctl down has freed it."""
    return STATUS_OK.replace(CAPACITY, "Capacity (in the order gpu-apply tries them):\n"
                                       "POOL   TYPE          REGION     AVAILABLE  PRICE\n"
                                       "a6000  gpu_1x_a6000  us-west-1  no         $0.80/h\n"
                                       f"a6000  gpu_1x_a10    us-west-1  {a10_west:<3}        $0.75/h\n")


NO_CAPACITY = ("error: k3s-gpu-a6000-1: no capacity for any candidate type in any region of pool a6000\n"
               "No new GPU nodes were launched; fix the error and re-run\n")


def test_e2_held_unit_waits_for_the_freed_unit_to_show_as_capacity_before_the_apply(gpu):
    """F-CAP (Ruling R61): Lambda can show the unit gpuctl down freed as capacity only a while after it stops listing
    the instance; an apply before that fails at once. E2 polls gpuctl status (every 15 s) until it shows it."""
    e2(gpu)
    gpu.first(gpuctl("status", seq=seq(held_status("no"), held_status("no"), held_status("no"), held_status("yes"))))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "PASS", text
    assert gpuctl_calls(gpu) == ["status", "list-nodes", "down --auto-approve", "list-nodes",
                                 "status", "status", "status", "apply --auto-approve", "list-nodes"]
    assert "waiting up to 300 s" in text and "(VR_NO_WAIT: skipped sleep 15)" in text


def test_e2_held_unit_fails_when_the_freed_unit_never_shows_within_300_s(gpu):
    """F-CAP: bounded (300 s on the clock, every 15 s); no apply; the evidence says why and that nothing is left
    billing (the cleanup's listing), and how to bring the node back."""
    e2(gpu, lists=(LIST_BEFORE, ""))
    gpu.first(gpuctl("status", stdout=held_status("no")))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "FAIL", text
    calls = gpuctl_calls(gpu)
    assert "apply --auto-approve" not in calls
    assert calls[:4] == ["status", "list-nodes", "down --auto-approve", "list-nodes"]
    assert calls[4:-1] == ["status"] * 20 and calls[-1] == "list-nodes"     # every 15 s for 300 s, then the sweep
    assert "shows no capacity for GPU pool a6000 within 300 s" in text
    assert "no GPU instance is left running; make gpu-apply brings the GPU node back" in text


def test_e2_held_unit_retries_an_apply_that_finds_no_capacity(gpu):
    """F-CAP: status can show the unit before the launch call gets it: an apply that fails with gpuctl's 'no
    capacity for any candidate type' is tried again (after 15 s, then 30 s, …) within the same 300 s."""
    e2(gpu)
    gpu.first(gpuctl("status", seq=seq(held_status("no"), held_status("yes"))),
              gpuctl("apply", "--auto-approve", seq=[{"rc": 1, "stderr": NO_CAPACITY}] * 2 + [{"rc": 0}]))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "PASS", text
    assert gpuctl_calls(gpu).count("apply --auto-approve") == 3
    assert "(VR_NO_WAIT: skipped sleep 15)" in text and "(VR_NO_WAIT: skipped sleep 30)" in text


def test_e2_held_unit_gives_up_retrying_the_apply_at_the_bound(gpu):
    e2(gpu, lists=(LIST_BEFORE, "", ""))
    gpu.first(gpuctl("status", seq=seq(held_status("no"), held_status("yes"))),
              gpuctl("apply", "--auto-approve", rc=1, stderr=NO_CAPACITY))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "FAIL", text
    # 0, 15, 45, 105, 225 s on the clock; the next try (465 s) is past 300 s
    assert gpuctl_calls(gpu).count("apply --auto-approve") == 5
    assert "no capacity until 300 s after gpuctl down" in text
    assert "no GPU instance is left running" in text


@pytest.mark.parametrize("held", [False, True], ids=["capacity", "held-unit"])
def test_e2_does_not_retry_an_apply_that_failed_for_another_reason(gpu, held):
    e2(gpu, lists=(LIST_BEFORE, "", ""), apply_rc=1)
    if held:
        gpu.first(gpuctl("status", seq=seq(held_status("no"), held_status("yes"))))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "FAIL", text
    assert gpuctl_calls(gpu).count("apply --auto-approve") == 1


def test_e2_without_a_held_unit_does_not_wait_for_capacity_or_retry(gpu):
    """The pool showed capacity before: no capacity poll after gpuctl down, and no retry of an apply that finds
    none (it is not the freed unit that is missing)."""
    e2(gpu, lists=(LIST_BEFORE, "", ""))
    gpu.first(gpuctl("apply", "--auto-approve", rc=1, stderr=NO_CAPACITY))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "FAIL", text
    assert gpuctl_calls(gpu) == ["status", "list-nodes", "down --auto-approve", "list-nodes", "apply --auto-approve",
                                 "list-nodes"]


def test_e2_skips_without_a_gpu_node_and_without_reading_env(no_gpu):
    (no_gpu.repo / ".env").unlink()
    result, text = check(no_gpu, "check_e2_gpu_replace")
    assert result == "SKIP", text
    assert gpuctl_calls(no_gpu) == []


@pytest.mark.parametrize("count, spec", [(0, True), (1, False)], ids=["no-gpu-pool", "no-spec"])
def test_e2_skips_when_the_spec_wants_no_gpu_node(tmp_path, count, spec):
    """gpuctl apply would not bring the node back: nothing to replace."""
    env = e2(with_spec(GpuCluster(tmp_path), count=count, spec=spec))
    result, text = check(env, "check_e2_gpu_replace")
    assert result == "SKIP", text
    assert gpuctl_calls(env) == []


def test_e2_changes_nothing_while_autoscaler_overrides_exist(gpu):
    """gpuctl down removes the override file, so the apply after it would launch the spec's counts instead."""
    (gpu.repo / "lambda/.autoscale-state.json").write_text('{"a6000": 2}\n')
    result, text = check(e2(gpu), "check_e2_gpu_replace")
    assert result == "FAIL" and ".autoscale-state.json" in text
    assert "down --auto-approve" not in gpuctl_calls(gpu)


def test_e2_fails_loudly_when_apply_brings_back_more_gpu_nodes_than_it_replaced_and_removes_them(gpu):
    """No extra GPU instance is left billing: the cleanup sweeps them all (make gpu-apply restores the spec)."""
    extra = "k3s-gpu-a6000-2 203.0.113.45 active gpu_1x_a10 us-east-1\n"
    result, text = check(e2(gpu, lists=(LIST_BEFORE, "", LIST_AFTER + extra, LIST_AFTER + extra, "")),
                         "check_e2_gpu_replace")
    assert result == "FAIL"
    assert "more GPU instances than the check replaced" in text and "k3s-gpu-a6000-2" in text
    assert gpuctl_calls(gpu)[-3:] == ["list-nodes", "down --auto-approve", "list-nodes"]
    assert "no GPU instance is left running" in text and "make gpu-apply" in text


@pytest.mark.parametrize("status, rc", [
    (STATUS_DRIFT, 3),
    (STATUS_HEAD + "  No drift\n  note: pool a6000: 1 of 2 running (make gpu-apply starts the rest)\n", 0),
    (STATUS_HEAD + "  skipped: gpu-access state unavailable\n  No drift in the 4 sources read; 1 source(s) skipped "
     "(see above)\n", 0),
], ids=["drift", "pool-below-count", "source-skipped"])
def test_e2_does_not_touch_the_gpu_node_when_status_is_not_clean(gpu, status, rc):
    result, text = check(e2(gpu, status=status, status_rc=rc), "check_e2_gpu_replace")
    assert result == "FAIL", text
    assert gpuctl_calls(gpu) == ["status"]


def test_e2_fails_when_gpu_down_fails_and_makes_sure_no_instance_is_left(gpu):
    e2(gpu, lists=(LIST_BEFORE, LIST_BEFORE, ""), down_rc=[{"rc": 1, "stderr": "gpuctl: down: drain failed\n"},
                                                         {"rc": 0}])
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "FAIL" and "gpuctl down failed" in text
    assert gpuctl_calls(gpu) == ["status", "list-nodes", "down --auto-approve",
                                 "list-nodes", "down --auto-approve", "list-nodes"]
    assert "no GPU instance is left running" in text


def test_e2_fails_when_the_node_is_still_listed_120_s_after_gpu_down(gpu):
    # 1 before, 24 while waiting (every 5 s for 120 s), 1 for the sweep; after the sweep's gpuctl down: none
    e2(gpu, lists=(LIST_BEFORE,) * 26 + ("",))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "FAIL" and f"{GPU1} is still listed 120 s after gpuctl down" in text
    assert "apply --auto-approve" not in gpuctl_calls(gpu)
    assert gpuctl_calls(gpu) == (["status", "list-nodes", "down --auto-approve"] + ["list-nodes"] * 24
                                 + ["list-nodes", "down --auto-approve", "list-nodes"])
    assert "no GPU instance is left running" in text


def test_e2_failed_apply_that_cleaned_up_after_itself_needs_no_second_down(gpu):
    result, text = check(e2(gpu, lists=(LIST_BEFORE, "", ""), apply_rc=1), "check_e2_gpu_replace")
    assert result == "FAIL" and "gpuctl apply failed" in text
    assert gpuctl_calls(gpu) == ["status", "list-nodes", "down --auto-approve", "list-nodes",
                                 "apply --auto-approve", "list-nodes"]
    assert "no GPU instance is left running" in text


def test_e2_failed_apply_that_left_an_instance_is_cleaned_up_with_gpu_down(gpu):  # no orphan
    result, text = check(e2(gpu, lists=(LIST_BEFORE, "", LIST_AFTER, ""), apply_rc=1), "check_e2_gpu_replace")
    assert result == "FAIL", text
    assert gpuctl_calls(gpu) == ["status", "list-nodes", "down --auto-approve", "list-nodes",
                                 "apply --auto-approve", "list-nodes", "down --auto-approve", "list-nodes"]
    assert "no GPU instance is left running" in text


def test_e2_says_loudly_when_an_instance_may_still_be_billing(gpu):
    result, text = check(e2(gpu, lists=(LIST_BEFORE, "", LIST_AFTER, LIST_AFTER), apply_rc=1), "check_e2_gpu_replace")
    assert result == "FAIL"
    assert "may still be running" in text and "make gpu-down" in text and GPU1 in text
    # R54: gpuctl down refuses an undrained terminate while the API is unreachable: the tunnel comes first
    assert "make tunnel" in text
    # the sweep waited for Lambda to stop listing it (every 5 s for 120 s) before it said so
    assert gpuctl_calls(gpu)[-26:] == ["list-nodes", "down --auto-approve"] + ["list-nodes"] * 24


def test_e2_sweep_waits_for_lambda_to_stop_listing_before_it_warns_of_billing(gpu):
    """The sweep's gpuctl down terminated it; Lambda still lists it as active for a moment: no false alarm."""
    result, text = check(e2(gpu, lists=(LIST_BEFORE, "", LIST_AFTER, LIST_AFTER, LIST_AFTER, ""), apply_rc=1),
                         "check_e2_gpu_replace")
    assert result == "FAIL" and "gpuctl apply failed" in text
    assert "no GPU instance is left running" in text and "may still be running" not in text
    assert gpuctl_calls(gpu).count("down --auto-approve") == 2


def test_e2_sweep_downs_when_list_nodes_fails(gpu):
    e2(gpu, lists=(LIST_BEFORE, ""), apply_rc=1)
    gpu.first(gpuctl("list-nodes", seq=[{"stdout": LIST_BEFORE}, {"stdout": ""},
                                        {"rc": 1, "stderr": "gpuctl: Lambda API error: 503\n"}, {"stdout": ""}]))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "FAIL"
    assert gpuctl_calls(gpu)[-3:] == ["list-nodes", "down --auto-approve", "list-nodes"]
    assert "no GPU instance is left running" in text


def test_e2_fails_when_the_node_object_was_not_replaced(gpu):
    result, text = check(e2(gpu, uids=(UID1, UID1)), "check_e2_gpu_replace")
    assert result == "FAIL" and "same Node object" in text
    assert gpuctl_calls(gpu).count("down --auto-approve") == 1   # apply succeeded: nothing to clean up


def test_e2_fails_when_the_node_is_not_listed_after_apply(gpu):
    result, text = check(e2(gpu, lists=(LIST_BEFORE, "", "")), "check_e2_gpu_replace")
    assert result == "FAIL" and "did not bring back every replaced GPU node" in text and "a6000" in text


def test_e2_waits_a_bounded_time_for_the_new_node_to_be_ready(gpu):
    result, text = check(e2(gpu, ready="False"), "check_e2_gpu_replace")
    assert result == "FAIL" and "not Ready within 300 s" in text
    assert len(polls(gpu, GPU1)) == 30                  # every 10 s for 300 s (VR_NO_WAIT: the skipped sleeps)
    assert "B1: N x N matrix" not in text


def test_e2_fails_when_d2_fails_on_the_new_node_and_still_runs_d4_and_b1(gpu):
    e2(gpu).first(kc(f"describe node {GPU1} ", stdout="Name: x\nCapacity:\n  cpu: 30\nAllocatable:\n  cpu: 30\n"))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "FAIL"
    assert "D4: gpu-smoke ran" in text and "B1: N x N matrix" in text
    assert gpuctl_calls(gpu).count("down --auto-approve") == 1


def test_e2_masks_the_project_and_the_operator_ip(gpu):
    operator = "192.0.2.77"
    env_file = gpu.repo / ".env"
    gpu.write_env(env_file.read_text() + f"OPERATOR_IP={operator}\n")
    e2(gpu, status=STATUS_OK + f"  note-free line naming {operator} and {operator} in gs://{PROJECT}-k3s-tfstate\n")
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "PASS", text
    assert operator not in text and PROJECT not in text
    assert "<operator_ip> and <operator_ip>" in text and "gs://<GCP_PROJECT>-k3s-tfstate" in text


# ---- E3: drift in the gpu-access firewall stack is found and repaired ----------------------------

def addresses(ips_json):
    """`terraform state list` of the gpu-access stack holding a rule pair for each IP of IPS_JSON."""
    return "".join(f'google_compute_firewall.gpu_to_{to}["{ip}"]\n' for ip in json.loads(ips_json)
                   for to in ("cp", "workers"))


def e3(env, statuses=(STATUS_OK, STATUS_DRIFT, STATUS_OK), status_rcs=(0, 3, 0), states=(ORIG,), apply_rc=0,
       lists=None):
    """STATES: the gpu_ips output, read before the injection and then by the cleanup (before and after a restore);
    LISTS: the cleanup's `terraform state list` answers (default: the rules for the cleanup's STATES)."""
    lists = lists or [addresses(s) for s in (states[1:] or states)]
    env.first(gpuctl("status", seq=[{"stdout": s, "rc": r} for s, r in zip(statuses, status_rcs, strict=True)]),
              gpuctl("apply", "--auto-approve", rc=apply_rc),
              rule("terraform", ["output", "-json", "gpu_ips"], seq=seq(*states)),
              rule("terraform", ["state", "list"], seq=seq(*lists)))
    return env


def test_e3_adds_a_rule_status_finds_it_apply_repairs_it(gpu):
    result, text = check(e3(gpu), "check_e3_firewall_drift")
    assert result == "PASS", text
    [inject] = tf_applies(gpu)
    assert inject[0] == f"-chdir={gpu.repo.resolve()}/infra/gpu-access"
    assert inject[1:5] == ["apply", "-auto-approve", "-input=false", "-lock-timeout=120s"]
    assert inject[-1] == INJECT_VAR and "-var=max_nodes=2" in inject
    assert gpuctl_calls(gpu) == ["status", "status", "apply --auto-approve", "status"]
    for step in ("E3 ①", "E3 ②", "E3 ③", "E3 ④"):
        assert step in text, text
    # the cleanup still looked: the gpu-access state holds no rule for the extra IP
    assert "no rule for 198.51.100.99" in text
    # Terraform got the project through TF_VAR_*, never on its command line
    tf = [c for c in gpu.calls("terraform") if c["argv"][1:2] == ["apply"]]
    assert tf[0]["env"]["TF_VAR_project_id"] == PROJECT and not any(PROJECT in a for a in inject)


def test_e3_skips_without_a_gpu_node_and_changes_nothing(no_gpu):
    (no_gpu.repo / ".env").unlink()
    result, text = check(no_gpu, "check_e3_firewall_drift")
    assert result == "SKIP", text
    assert tf_applies(no_gpu) == [] and gpuctl_calls(no_gpu) == []


def test_e3_does_not_inject_when_status_shows_drift_already(gpu):
    result, text = check(e3(gpu, statuses=(STATUS_DRIFT,), status_rcs=(3,)), "check_e3_firewall_drift")
    assert result == "FAIL"
    assert tf_applies(gpu) == [] and gpuctl_calls(gpu) == ["status"]


@pytest.mark.parametrize("answer", [
    {"rc": 1, "stderr": f"Error: Failed to load state: gs://{PROJECT}-k3s-tfstate/gpu-access/default.tfstate\n"},
    {"stdout": '{"not": "a list"}'},
    {"stdout": '["10.0.0.5 -oProxyCommand=x"]'},
])
def test_e3_does_not_inject_when_the_state_cannot_be_read(gpu, answer):
    e3(gpu)
    gpu.first(rule("terraform", ["output", "-json", "gpu_ips"], **answer))
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "FAIL"
    assert tf_applies(gpu) == [] and gpuctl_calls(gpu) == ["status"]
    assert re.search(r"^E3: (cannot read gpu_ips|unexpected gpu_ips output|gpu_ips holds )", text, re.M), text
    assert PROJECT not in text   # Terraform's messages name the state bucket: masked


def test_e3_failed_injection_is_still_cleaned_up(gpu):
    e3(gpu, states=(ORIG, WITH_EXTRA, ORIG))
    gpu.first(rule("terraform", ["apply", EXTRA], rc=1, stderr="Error: creating firewall: googleapi: Error 503\n"))
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "FAIL" and "E3 ①: terraform apply failed" in text
    assert [a[-1] for a in tf_applies(gpu)] == [INJECT_VAR, RESTORE_VAR]
    assert gpuctl_calls(gpu) == ["status"]


def test_e3_repairs_when_status_misses_the_drift(gpu):
    result, text = check(e3(gpu, statuses=(STATUS_OK, STATUS_OK, STATUS_OK), status_rcs=(0, 0, 0)),
                         "check_e3_firewall_drift")
    assert result == "FAIL" and "E3 ②: gpuctl status exits 0; want 3" in text
    assert gpuctl_calls(gpu) == ["status", "status", "apply --auto-approve", "status"]


def test_e3_fails_when_status_exits_3_without_naming_the_extra_ip(gpu):
    other = STATUS_HEAD + "  missing cp k3s_peers entry for 203.0.113.12\n  1 problem(s)\n"
    result, text = check(e3(gpu, statuses=(STATUS_OK, other, STATUS_OK)), "check_e3_firewall_drift")
    assert result == "FAIL" and f"does not name {EXTRA}" in text


def test_e3_restores_the_state_itself_when_gpuctl_apply_fails(gpu):
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT, STATUS_DRIFT), status_rcs=(0, 3, 3), states=(ORIG, WITH_EXTRA, ORIG),
       apply_rc=1)
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "FAIL" and "gpuctl apply failed" in text
    assert [a[-1] for a in tf_applies(gpu)] == [INJECT_VAR, RESTORE_VAR]
    assert "restored" in text


def test_e3_fails_when_status_still_shows_drift_after_the_repair(gpu):
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT, STATUS_DRIFT), status_rcs=(0, 3, 3), states=(ORIG, ORIG))
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "FAIL" and "E3 ④: after the repair gpuctl status exits 3; want 0" in text


def test_e3_restore_that_fails_never_says_restored_even_when_the_output_looks_clean(gpu):
    """gpu_ips echoes the input variable: a failed apply can leave it clean while the rule is still in the state."""
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT, STATUS_DRIFT), status_rcs=(0, 3, 3), states=(ORIG, WITH_EXTRA, ORIG),
       apply_rc=1, lists=(addresses(WITH_EXTRA),))
    gpu.first(rule("terraform", ["apply", RESTORE_VAR], rc=1,
                   stderr="Error: deleting fw-gpu-198-51-100-99-to-cp: googleapi: Error 503\n"))
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "FAIL"
    assert f"may STILL allow {EXTRA}" in text and "restored" not in text


def test_e3_cleanup_restores_a_rule_left_in_the_state_behind_a_clean_output(gpu):
    """gpuctl apply (③) failed half way: gpu_ips is back to the real IPs, the rule for the extra IP is not gone."""
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT, STATUS_DRIFT), status_rcs=(0, 3, 3), states=(ORIG,), apply_rc=1,
       lists=(addresses(WITH_EXTRA), addresses(ORIG)))
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "FAIL"
    assert [a[-1] for a in tf_applies(gpu)] == [INJECT_VAR, RESTORE_VAR]
    assert "restored" in text and "may STILL allow" not in text


def test_e3_says_loudly_when_the_extra_rule_may_remain(gpu):
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT, STATUS_DRIFT), status_rcs=(0, 3, 3),
       states=(ORIG, WITH_EXTRA, WITH_EXTRA), apply_rc=1)
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "FAIL"
    assert f"may STILL allow {EXTRA}" in text and "make gpu-apply" in text


def test_e3_restores_when_the_state_cannot_be_read_in_the_cleanup(gpu):
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT, STATUS_DRIFT), status_rcs=(0, 3, 3), apply_rc=1)
    gpu.first(rule("terraform", ["output", "-json", "gpu_ips"],
                   seq=[{"stdout": ORIG}, {"rc": 1, "stderr": "Error: state lock\n"}, {"stdout": ORIG}]))
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "FAIL"
    assert [a[-1] for a in tf_applies(gpu)] == [INJECT_VAR, RESTORE_VAR]


def test_e3_masks_the_project_in_terraform_output(gpu):
    e3(gpu)
    gpu.first(rule("terraform", ["apply", EXTRA],
                   stdout=f"google_compute_firewall.gpu[\"{EXTRA}\"]: Creation complete "
                          f"[id=projects/{PROJECT}/global/firewalls/fw-gpu-x]\n"))
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "PASS", text
    assert PROJECT not in text and "projects/<GCP_PROJECT>/global" in text


# ---- interrupted: the cleanup runs, and a second Ctrl-C does not stop it ----------------------------

# The cleanup runs for the SIGHUP itself (the harness's _vr_signalled, armed by _vr_e_guarded): run_check's subshell
# no longer dies at once and sends the check its _vr_reap SIGTERM (and, after 5 s, SIGKILL) while the check still
# runs (Ruling R62).
HUNG_UP = re.compile(r"^interrupted \(SIGHUP\); cleaning up \(at most 1800 s\): _vr_e[23]_cleanup$", re.M)


def hup_default():
    """preexec_fn: SIGHUP at its default in the child, even when the test runner was started with it ignored (nohup):
    the hangup tests and the SigIgn control need it."""
    signal.signal(signal.SIGHUP, signal.SIG_DFL)


def run_hung_up(env, func, first, outcome, code=None):
    """Run FUNC through run_check in its own session and hang up (SIGHUP to the whole process group, as a dropped
    SSH session does) once FIRST exists. run_check's shell dies at once; its redact filter and the cleanup go on.
    Waits for OUTCOME (the cleanup's last line) in the evidence and for the cleanup to end; asserts that OUTCOME is
    there (the cleanup ran to its end, review X1-3); returns the evidence."""
    run = env.repo / "verification" / "t"
    code = code or (f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; run_check T title {func}; "
                    "run_check U later check_a1_nodes")
    proc = subprocess.Popen(["bash", "-c", code], cwd=env.repo,
                            env=env.environ({"VR_NO_WAIT": "1", "BASH_FUNC_timeout%%": REAL_TIMEOUT}),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True, preexec_fn=hup_default)
    try:
        assert wait_for_file(first), f"{first.name} never appeared"
        time.sleep(0.2)
        os.killpg(proc.pid, signal.SIGHUP)
        proc.communicate(timeout=120)
        log = run / "01-T.txt"
        deadline = time.time() + 120
        while time.time() < deadline and outcome not in (log.read_text() if log.exists() else ""):
            time.sleep(0.1)
        # the cleanup is over once its subshell has ended by the signal it was given (no process left in the group)
        while time.time() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    assert proc.returncode == -signal.SIGHUP
    assert not (run / "02-U.txt").exists()             # the run stopped
    text = log.read_text()
    assert outcome in text, text
    return text


def test_the_e_checks_use_the_harness_clock_and_operator_channel():
    """After the sync with main (SYNC-2): E waits on _vr_clock/_vr_wait/_vr_poll and reports through _vr_tell and
    _vr_on_signal, the same helpers as the D checks; 22b's own copies are gone."""
    text = (ROOT / "scripts/lib/checks.sh").read_text()
    for gone in ("_vr_e_now", "_vr_e_wait", "_vr_e_until", "_vr_e_rt", "_VR_E_SLEPT", "VR_E_DEADLINE",
                 "_vr_cleanup_say", "cleanup.log", "_vr_guard_signal", "_vr_guarded "):
        assert gone not in text, gone


def test_on_signal_hup_is_armed_only_when_asked_and_disarms_to_run_checks_trap(env):
    code = ("source scripts/lib/checks.sh; trap 'exit 129' HUP; _vr_on_signal true; trap -p HUP; "
            "_vr_on_signal; _vr_on_signal --hup --within 5 --left x --hint y true; trap -p HUP; "
            "echo \"$_VR_ON_SIGNAL_WITHIN|$_VR_ON_SIGNAL_LEFT|$_VR_ON_SIGNAL_HINT|${_VR_ON_SIGNAL[*]}\"; "
            "_vr_on_signal; trap -p HUP INT")
    proc = env.run(["bash", "-c", code])
    assert proc.stdout.splitlines() == ["trap -- 'exit 129' SIGHUP", "trap -- '_vr_signalled HUP' SIGHUP",
                                        "5|x|y|true", "trap -- 'exit 129' SIGHUP"], proc.stdout + proc.stderr


def test_run_check_ends_a_hung_up_check_only_after_its_foreground_command(env):
    """run_check's `trap 'exit 129' HUP` (review C1-2): on a hangup the check's subshell waits for the command it
    runs, then exits 129 without running the rest of the check. Without the trap it died at once, and its EXIT trap
    (_vr_reap) sent SIGTERM to that command: an E apply that ignores SIGHUP to finish and release its state lock."""
    marks = env.tmpdir / "marks"
    marks.mkdir()
    run = env.repo / "verification" / "t"
    code = (f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; "
            f"t() {{ ( trap '' HUP; touch {marks}/started; sleep 2; echo 'command finished' ); echo 'check went on'; }}; "
            "run_check T title t; run_check U later check_a1_nodes")
    log = run_hung_up(env, "t", marks / "started", "command finished", code=code)
    assert "check went on" not in log, log


def test_a_hangup_during_an_e_cleanup_is_not_killed_after_5_s(gpu):
    """Ruling R62 with H-1's _vr_reap (TERM, then KILL after 5 s): a hangup during E3's step ③ runs the cleanup, whose
    restore takes longer than 5 s; the check's subshell waits for it (`trap 'exit 129' HUP`), so its _vr_reap never
    KILLs the cleanup's shell, and the cleanup's outcome is reported to the end (SYNC-2)."""
    marks = gpu.tmpdir / "marks"
    marks.mkdir()
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT), status_rcs=(0, 3), states=(ORIG, WITH_EXTRA, ORIG))
    script(gpu.repo / "scripts/gpuctl", f'if [ "$1" = apply ]; then touch {marks}/applying; sleep 2; fi\n'
                                        f'exec "{sys.executable}" "{FAKES}" gpuctl "$@"\n')
    script(gpu.bin / "terraform", f'if [ "$2" = apply ]; then case "$*" in *{EXTRA}*) ;; *) sleep 7; '
                                  f'touch {marks}/restored ;; esac; fi\n'
                                  f'exec "{sys.executable}" "{FAKES}" terraform "$@"\n')
    log = run_hung_up(gpu, "check_e3_firewall_drift", marks / "applying", "cleanup done")
    assert (marks / "restored").exists(), log
    assert HUNG_UP.search(log) and "restored: the gpu-access state has no rule" in log, log
    assert "cleanup FAILED" not in log and "may STILL allow" not in log, log


def test_e3_hangup_during_the_injection_lets_terraform_finish_then_removes_the_extra_rule(gpu):
    """A SIGHUP must not kill a terraform apply half way: it would orphan the GCS state lock, which never expires,
    and the restore (and make gpu-apply) would fail on it (review X1-1)."""
    marks = gpu.tmpdir / "marks"
    marks.mkdir()
    e3(gpu, statuses=(STATUS_OK,), status_rcs=(0,), states=(ORIG, WITH_EXTRA, ORIG))
    script(gpu.bin / "terraform", f'case "$*" in *apply*{EXTRA}*) touch {marks}/injecting; sleep 3; '
                                  f'touch {marks}/injected ;; esac\n'
                                  f'exec "{sys.executable}" "{FAKES}" terraform "$@"\n')
    log = run_hung_up(gpu, "check_e3_firewall_drift", marks / "injecting", "restored: the gpu-access state has no rule")
    assert (marks / "injected").exists(), log
    assert [a[-1] for a in tf_applies(gpu)] == [INJECT_VAR, RESTORE_VAR]
    assert gpuctl_calls(gpu) == ["status"]              # the run stopped after the injection: no step ②
    assert HUNG_UP.search(log) and "may STILL allow" not in log, log


def test_e3_hangup_during_the_repair_lets_gpuctl_apply_finish_then_removes_the_extra_rule(gpu):
    marks = gpu.tmpdir / "marks"
    marks.mkdir()
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT), status_rcs=(0, 3), states=(ORIG, WITH_EXTRA, ORIG))
    script(gpu.repo / "scripts/gpuctl", f'if [ "$1" = apply ]; then touch {marks}/applying; sleep 3; '
                                        f'touch {marks}/applied; fi\n'
                                        f'exec "{sys.executable}" "{FAKES}" gpuctl "$@"\n')
    log = run_hung_up(gpu, "check_e3_firewall_drift", marks / "applying", "restored: the gpu-access state has no rule")
    assert (marks / "applied").exists(), log           # gpuctl apply was not killed half way (review X1-1)
    assert [a[-1] for a in tf_applies(gpu)] == [INJECT_VAR, RESTORE_VAR]
    assert HUNG_UP.search(log) and "may STILL allow" not in log, log


def test_e2_hangup_during_apply_lets_it_finish_then_sweeps(gpu):  # no orphan
    marks = gpu.tmpdir / "marks"
    marks.mkdir()
    e2(gpu, lists=(LIST_BEFORE, "", LIST_AFTER, ""))
    script(gpu.repo / "scripts/gpuctl",
           f'case "$1" in\n  apply) touch {marks}/applying; sleep 3; touch {marks}/applied ;;\nesac\n'
           f'exec "{sys.executable}" "{FAKES}" gpuctl "$@"\n')
    log = run_hung_up(gpu, "check_e2_gpu_replace", marks / "applying", "no GPU instance is left running")
    assert (marks / "applied").exists(), log           # gpuctl apply was not killed half way (review X1-1)
    assert gpuctl_calls(gpu)[-3:] == ["list-nodes", "down --auto-approve", "list-nodes"]
    assert HUNG_UP.search(log) and "may still be running" not in log, log


def sigign_hup(env, tool, label):
    """A fake TOOL that records, per call, LABEL (a shell word) and whether it runs with SIGHUP ignored (SigIgn bit 0
    in /proc/self/status, which `grep` inherits from it), then answers as usual."""
    path = env.repo / "scripts/gpuctl" if tool == "gpuctl" else env.bin / tool
    script(path, f'echo "{label} $(grep "^SigIgn:" /proc/self/status | cut -f2)" >> "{env.tmpdir}/sigign"\n'
                 f'exec "{sys.executable}" "{FAKES}" {tool} "$@"\n')


def hup_ignored(env):
    """{label: [SIGHUP ignored?, …]} for each recorded call."""
    seen = {}
    for line in (env.tmpdir / "sigign").read_text().splitlines():
        label, mask = line.rsplit(" ", 1)
        seen.setdefault(label, []).append(bool(int(mask, 16) & 1))
    return seen


def check_hup_default(env, func):
    """env.check, with SIGHUP at its default for the run (hup_default)."""
    run = env.repo / "verification" / "t"
    code = f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; run_check T title {func}"
    proc = subprocess.run(["bash", "-c", code], cwd=env.repo,
                          env=env.environ({"VR_NO_WAIT": "1", "BASH_FUNC_timeout%%": REAL_TIMEOUT}),
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
                          preexec_fn=hup_default)
    assert proc.returncode == 0, proc.stderr
    return (run / "01-T.txt").read_text()


def test_e3_mutating_commands_run_with_sighup_ignored(gpu):
    """The inject terraform apply and gpuctl apply ignore SIGHUP (review X1-1); the reads keep its default."""
    e3(gpu)
    sigign_hup(gpu, "gpuctl", "$1")
    sigign_hup(gpu, "terraform", f'$(case "$*" in *apply*{EXTRA}*) echo inject ;; *) echo "$2" ;; esac)')
    text = check_hup_default(gpu, "check_e3_firewall_drift")
    seen = hup_ignored(gpu)
    assert seen["inject"] == [True] and seen["apply"] == [True], (seen, text)
    assert seen["status"] == [False, False, False], seen   # control: the test run itself does not ignore SIGHUP


def test_e2_mutating_commands_run_with_sighup_ignored(gpu):
    e2(gpu)
    sigign_hup(gpu, "gpuctl", "$1")
    text = check_hup_default(gpu, "check_e2_gpu_replace")
    seen = hup_ignored(gpu)
    assert seen["down"] == [True] and seen["apply"] == [True], (seen, text)
    assert set(seen["status"] + seen["list-nodes"]) == {False}, seen


LOCK_ID = "1729588812345678"
LOCK_ERROR = ("Error: Error acquiring the state lock\n\nError message: writing\n"
              f'"gs://{PROJECT}-k3s-tfstate/gpu-access/default.tflock" failed: googleapi: Error 412: Precondition '
              "Failed, conditionNotMet\nLock Info:\n"
              f"  ID:        {LOCK_ID}\n  Path:      gs://{PROJECT}-k3s-tfstate/gpu-access/default.tflock\n"
              "  Operation: OperationTypeApply\n  Who:       op@laptop\n\nTerraform acquires a state lock to protect "
              "the state from being written\nby multiple users at the same time.\n")


def test_e3_restore_blocked_by_a_state_lock_names_the_force_unlock(gpu):
    """An apply killed half way (kill -9, a crash) leaves the GCS lock, which never expires: make gpu-apply alone
    would fail on it too, so the evidence names the lock and how to release it (review X1-1)."""
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT, STATUS_DRIFT), status_rcs=(0, 3, 3),
       states=(ORIG, WITH_EXTRA, WITH_EXTRA), apply_rc=1)
    gpu.first(rule("terraform", ["apply", RESTORE_VAR], rc=1, stderr=LOCK_ERROR))
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "FAIL"
    hint = f"terraform -chdir=infra/gpu-access force-unlock {LOCK_ID}"
    assert hint in text and f"may STILL allow {EXTRA}" in text
    assert text.index(hint) < text.rindex("make gpu-apply"), text
    assert PROJECT not in text


def test_e3_restore_failure_without_a_lock_names_no_force_unlock(gpu):
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT, STATUS_DRIFT), status_rcs=(0, 3, 3),
       states=(ORIG, WITH_EXTRA, WITH_EXTRA), apply_rc=1)
    gpu.first(rule("terraform", ["apply", RESTORE_VAR], rc=1, stderr="Error: googleapi: Error 503\n"))
    result, text = check(gpu, "check_e3_firewall_drift")
    assert result == "FAIL" and "may STILL allow" in text and "force-unlock" not in text


def test_e2_sweep_blocked_by_a_state_lock_names_the_force_unlock(gpu):
    e2(gpu, lists=(LIST_BEFORE, "", LIST_AFTER, LIST_AFTER), apply_rc=1)
    gpu.first(gpuctl("down", "--auto-approve", seq=[{"rc": 0}, {"rc": 1, "stderr": LOCK_ERROR}]))
    result, text = check(gpu, "check_e2_gpu_replace")
    assert result == "FAIL" and "may still be running" in text
    assert f"terraform -chdir=infra/gpu-access force-unlock {LOCK_ID}" in text


def run_interrupted(env, func, first, second=None, last=None):
    """Run FUNC through run_check in its own session; Ctrl-C (SIGINT to the process group) once FIRST exists, again
    once SECOND exists; wait for the end. Returns (returncode, evidence, the operator's stderr)."""
    run = env.repo / "verification" / "t"
    code = f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; run_check T title {func}; " \
           "run_check U later check_a1_nodes"
    proc = subprocess.Popen(["bash", "-c", code], cwd=env.repo,
                            env=env.environ({"VR_NO_WAIT": "1", "BASH_FUNC_timeout%%": REAL_TIMEOUT}),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        assert wait_for_file(first), f"{first.name} never appeared"
        time.sleep(0.2)
        os.killpg(proc.pid, signal.SIGINT)
        if second is not None:
            assert wait_for_file(second), f"{second.name} never appeared"
            os.killpg(proc.pid, signal.SIGINT)
        _, err = proc.communicate(timeout=60)
        # Ctrl-C ends the outer shell once the check's shell has cleaned up; wait for the whole group
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    assert not (run / "02-U.txt").exists()             # the run stopped after the interrupted check
    return proc.returncode, (run / "01-T.txt").read_text(), err


def script(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def test_e3_ctrl_c_during_the_repair_still_removes_the_extra_rule_despite_a_second_ctrl_c(gpu):
    marks = gpu.tmpdir / "marks"
    marks.mkdir()
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT), status_rcs=(0, 3), states=(ORIG, WITH_EXTRA, ORIG))
    # gpuctl apply (step ③) blocks until the Ctrl-C; the restore takes a while, so the second Ctrl-C lands in it
    script(gpu.repo / "scripts/gpuctl", f'if [ "$1" = apply ]; then touch {marks}/applying; sleep 30; fi\n'
                                        f'exec "{sys.executable}" "{FAKES}" gpuctl "$@"\n')
    script(gpu.bin / "terraform", f'if [ "$2" = apply ]; then case "$*" in *{EXTRA}*) ;; *) touch {marks}/restoring; '
                                  f'sleep 2; touch {marks}/restored ;; esac; fi\n'
                                  f'exec "{sys.executable}" "{FAKES}" terraform "$@"\n')
    rc, text, err = run_interrupted(gpu, "check_e3_firewall_drift", marks / "applying", marks / "restoring")
    assert (marks / "restored").exists(), text
    assert [a[-1] for a in tf_applies(gpu)] == [INJECT_VAR, RESTORE_VAR]
    # the operator sees the cleanup (_vr_tell on run_check's stderr), and the evidence records it
    for out in (text, err):
        assert "interrupted (SIGINT); cleaning up (at most 1800 s): _vr_e3_cleanup" in out, out
        assert "restored" in out.split("cleaning up")[1] and "cleanup done" in out, out
    assert "T: cleanup done" in err
    assert rc == -signal.SIGINT


def test_e2_ctrl_c_during_apply_still_sweeps_despite_a_second_ctrl_c(gpu):  # no orphan
    marks = gpu.tmpdir / "marks"
    marks.mkdir()
    e2(gpu, lists=(LIST_BEFORE, "", LIST_AFTER, ""))
    script(gpu.repo / "scripts/gpuctl",
           f'case "$1" in\n  apply) touch {marks}/applying; sleep 30 ;;\n'
           f'  down) if [ -e {marks}/down1 ]; then touch {marks}/sweeping; sleep 2; touch {marks}/swept; '
           f'else touch {marks}/down1; fi ;;\nesac\nexec "{sys.executable}" "{FAKES}" gpuctl "$@"\n')
    rc, text, err = run_interrupted(gpu, "check_e2_gpu_replace", marks / "applying", marks / "sweeping")
    assert (marks / "swept").exists(), text
    assert gpuctl_calls(gpu)[-3:] == ["list-nodes", "down --auto-approve", "list-nodes"]
    for out in (text, err):
        assert "interrupted (SIGINT); cleaning up (at most 1800 s): _vr_e2_cleanup" in out, out
        assert "no GPU instance is left running" in out and "cleanup done" in out, out
    assert rc == -signal.SIGINT


def test_e2_failed_sweep_after_ctrl_c_tells_the_operator_what_may_be_billing(gpu):
    """The interrupt cleanup could not confirm that no GPU instance runs: the operator is told what may remain and
    what to run (the tunnel, make gpu-down, make gpu-status), not a removal command that would do nothing."""
    marks = gpu.tmpdir / "marks"
    marks.mkdir()
    e2(gpu, lists=(LIST_BEFORE, "", LIST_AFTER))
    script(gpu.repo / "scripts/gpuctl",
           f'case "$1" in\n  apply) touch {marks}/applying; sleep 30 ;;\nesac\n'
           f'exec "{sys.executable}" "{FAKES}" gpuctl "$@"\n')
    rc, text, err = run_interrupted(gpu, "check_e2_gpu_replace", marks / "applying")
    for out in (text, err):
        assert "cleanup FAILED (exit 1); may remain: GPU instance(s)" in out, out
        assert "make tunnel" in out and "make gpu-down" in out and "make gpu-status" in out, out
    assert "bash -c 'source scripts/lib/checks.sh" not in err
    assert rc == -signal.SIGINT


def test_a_ctrl_c_during_the_final_e3_cleanup_does_not_cut_it_short(gpu):
    """Not interrupted until the cleanup: step ③ failed, the restore runs, then a Ctrl-C. The restore goes on (its
    apply is out of the Ctrl-C's reach), its outcome reaches the evidence and the operator (the evidence filter died
    with the Ctrl-C), and the run then stops by the signal."""
    marks = gpu.tmpdir / "marks"
    marks.mkdir()
    e3(gpu, statuses=(STATUS_OK, STATUS_DRIFT, STATUS_DRIFT), status_rcs=(0, 3, 3), states=(ORIG, WITH_EXTRA, ORIG),
       apply_rc=1)
    script(gpu.bin / "terraform", f'if [ "$2" = apply ]; then case "$*" in *{EXTRA}*) ;; *) touch {marks}/restoring; '
                                  f'sleep 2; touch {marks}/restored ;; esac; fi\n'
                                  f'exec "{sys.executable}" "{FAKES}" terraform "$@"\n')
    rc, text, err = run_interrupted(gpu, "check_e3_firewall_drift", marks / "restoring")
    assert (marks / "restored").exists(), text
    for out in (text, err):
        assert "restored: the gpu-access state has no rule" in out, out
    assert "may STILL allow" not in text
    assert rc == -signal.SIGINT


def test_e2_ctrl_c_after_the_replacement_succeeded_downs_nothing(gpu):
    marks = gpu.tmpdir / "marks"
    marks.mkdir()
    e2(gpu)
    run = gpu.repo / "verification" / "t"
    code = (f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; "
            "kc() { if [[ \"$*\" == *'rollout status'* ]]; then "
            f"touch {marks}/rollout; sleep 30; fi; kubectl --kubeconfig \"$KUBECONFIG_FILE\" \"$@\"; }}; "
            "run_check T title check_e2_gpu_replace")
    proc = subprocess.Popen(["bash", "-c", code], cwd=gpu.repo,
                            env=gpu.environ({"VR_NO_WAIT": "1", "BASH_FUNC_timeout%%": REAL_TIMEOUT}),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        assert wait_for_file(marks / "rollout")
        time.sleep(0.2)
        os.killpg(proc.pid, signal.SIGINT)
        proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert gpuctl_calls(gpu).count("down --auto-approve") == 1


# ---- helpers ---------------------------------------------------------------------------------

def test_mask_hides_the_project_and_every_whole_operator_ip(env):
    env.write_env(f"GCP_PROJECT={PROJECT}\nLAMBDA_API_KEY=x\nOPERATOR_IP=198.51.100.2\n")
    text = (f"gs://{PROJECT}-k3s-tfstate 198.51.100.2 198.51.100.2,198.51.100.2.\n"
            "198.51.100.20 1198.51.100.2 x.198.51.100.2\n")
    proc = env.run(["bash", "-c", "source scripts/lib/checks.sh; vr_env; _vr_e_mask"], stdin=text)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ("gs://<GCP_PROJECT>-k3s-tfstate <operator_ip> <operator_ip>,<operator_ip>.\n"
                           "198.51.100.20 1198.51.100.2 x.198.51.100.2\n")


def test_mask_without_an_operator_ip_passes_text_through(env):
    proc = env.run(["bash", "-c", "source scripts/lib/checks.sh; vr_env; _vr_e_mask"], stdin="203.0.113.9 ok\n")
    assert proc.stdout == "203.0.113.9 ok\n"


def test_shielded_command_is_out_of_reach_of_a_ctrl_c_to_the_process_group(env):
    """_vr_shielded runs its command in a process group of its own: SIGINT to the caller's group does not stop it."""
    mark = env.tmpdir / "done"
    started = env.tmpdir / "started"
    # like terraform, the command takes SIGINT back even when it starts with it ignored
    child = (f"import signal, time; signal.signal(signal.SIGINT, signal.default_int_handler); "
             f"open('{started}', 'w').close(); time.sleep(1); open('{mark}', 'w').close()")
    code = f"source scripts/lib/checks.sh; trap '' INT TERM; _vr_shielded {sys.executable} -c \"{child}\"; echo rc=$?"
    proc = subprocess.Popen(["bash", "-c", code], cwd=env.repo, env=env.environ({}), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert wait_for_file(started)
        os.killpg(proc.pid, signal.SIGINT)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert mark.exists() and out.splitlines() == ["rc=0"]


# ---- verify.sh: group E under RESILIENCE=1 ----------------------------------------------------

def test_verify_registers_e1_to_e3_in_group_e():
    text = (ROOT / "scripts/verify.sh").read_text()
    regs = re.findall(r"^register_check (\w) (\w+) (?:'[^']*'|\"[^\"]*\") (\w+)$", text, re.M)
    assert [r for r in regs if r[0] == "E"] == E_EXPECTED


def test_every_e_function_is_defined_in_checks_sh():
    names = " ".join(f for _, _, f in E_EXPECTED)
    proc = subprocess.run(["bash", "-c", f"source {ROOT}/scripts/lib/checks.sh; for f in {names}; do "
                           'declare -F "$f" >/dev/null || echo "missing $f"; done'], capture_output=True, text=True)
    assert proc.stdout == "" and proc.returncode == 0


def all_e(env):
    """Answers for E1, E2 and E3 in one run (gpuctl status: E2's precondition, then E3's three)."""
    e1(env)
    e2(env)
    e3(env)
    env.first(gpuctl("status", seq=[{"stdout": STATUS_OK}, {"stdout": STATUS_OK},
                                    {"stdout": STATUS_DRIFT, "rc": 3}, {"stdout": STATUS_OK}]))
    return env


def run_verify(env, *groups, **extra):
    """scripts/verify.sh GROUPS…: a whole run is some 200 fake calls, so it gets more than Env.run's 60 s."""
    return subprocess.run(["bash", str(env.repo / "scripts/verify.sh"), *groups], cwd=env.repo,
                          env=env.environ({"VR_NO_WAIT": "1", "TZ": "America/Los_Angeles", **extra}),
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)


def test_make_verify_resilience_runs_e1_to_e3_after_a_to_d(gpu):
    proc = run_verify(all_e(gpu), VR_PF_PORT=str(free_port()), RESILIENCE="1",
                      **{"BASH_FUNC_timeout%%": '() { if [[ "$1" == 3 ]]; then return 124; fi; command timeout "$@"; }'})
    [run] = list((gpu.repo / "verification").iterdir())
    summary = (run / "SUMMARY.md").read_text()
    assert proc.returncode == 0, summary + proc.stdout + proc.stderr
    results = re.findall(r"^\| \d+ \| (\w+) \| [^|]+ \| (\w+) \|$", summary, re.M)
    assert [i for i, _ in results][-3:] == ["E1", "E2", "E3"]
    assert all(r == "PASS" for _, r in results), results


def test_verify_runs_group_e_when_asked_for_it(gpu):
    proc = run_verify(all_e(gpu), "E", **{"BASH_FUNC_timeout%%": REAL_TIMEOUT})
    [run] = list((gpu.repo / "verification").iterdir())
    summary = (run / "SUMMARY.md").read_text()
    assert proc.returncode == 0, summary + proc.stdout + proc.stderr
    assert re.findall(r"^\| \d+ \| (\w+) \| [^|]+ \| (\w+) \|$", summary, re.M) == [
        ("E1", "PASS"), ("E2", "PASS"), ("E3", "PASS")]
    assert "sudo systemctl reboot" in (run / "01-E1.txt").read_text()
