"""Offline tests for the verification checks A1-A4, B1-B6, C1-C8 (Task 18) in scripts/lib/checks.sh, their
registration in scripts/verify.sh, and the node helpers they use in scripts/lib/common.sh.

Same harness as test_ops_scripts.py: a repo copy whose kubectl, gcloud, ssh, terraform and scripts/gpuctl are
tests/scripts/fake_tools.py. Each test scripts a small healthy cluster (k3s-cp + k3s-cpu-1, optionally a GPU
node) as fake "answers", breaks one thing, and runs one check through run_check, as make verify does. Nothing
here reaches GCP, Lambda, a cluster or the network: k3s-cp's ExternalIP is a TEST-NET address and the local
`timeout` (check B5's /dev/tcp probe) is replaced by a shell function.
"""
import base64
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest

from test_ops_scripts import FAKES, ROOT, Env, free_port

CP, CPU1, GPU1 = "k3s-cp", "k3s-cpu-1", "k3s-gpu-a6000-1"
NODES = {
    CP: dict(ext="192.0.2.10", probe="probe-cp000", probe_ip="10.42.0.10", echo="echo-cp000", echo_ip="10.42.0.11"),
    CPU1: dict(ext="198.51.100.11", probe="probe-cu100", probe_ip="10.42.1.10", echo="echo-cu100",
               echo_ip="10.42.1.11"),
    GPU1: dict(ext="203.0.113.12", probe="probe-gp100", probe_ip="10.42.2.10", echo="echo-gp100", echo_ip="10.42.2.11"),
}
HELLO_CID = "0123456789abcdef" * 4
BURN_CID = "fedcba9876543210" * 4
RT = "--request-timeout=60s"
WHOAMI = "Hostname: echo\nIP: 10.42.1.11\nGET / HTTP/1.1\n"
NOT_REACHABLE = "() { printf 'fake timeout %s\\n' \"$*\"; return 124; }"  # B5: nothing answers
REACHABLE = "() { printf 'fake timeout %s\\n' \"$*\"; return 0; }"


def cgroup(cid):
    return f"/sys/fs/cgroup/kubepods.slice/kubepods-pod1234.slice/cri-containerd-{cid}.scope"


def wg_key(seed):
    """A WireGuard-shaped public key, built at run time so this file holds no key literal."""
    return base64.b64encode(bytes((seed * 37 + i * 11) % 256 for i in range(32))).decode()


def rule(tool, has, stdout="", rc=0, stderr="", **kw):
    return {"tool": tool, "has": list(has), "stdout": stdout, "stderr": stderr, "rc": rc, **kw}


def kc(*has, **kw):
    return rule("kubectl", has, **kw)


def on(node, *has, **kw):
    """A command run on NODE: IAP SSH (gcloud) for k3s-cp and CPU workers, plain ssh for a GPU node."""
    if node == GPU1:
        return rule("ssh", (f"ubuntu@{NODES[GPU1]['ext']}", *has), **kw)
    return rule("gcloud", (f"compute ssh {node} ", *has), **kw)


def wg_show(node, nodes, ago="1 minute, 5 seconds ago"):
    peers = "".join(f"\npeer: {wg_key(i + 2)}\n  endpoint: {NODES[n]['ext']}:51820\n  allowed ips: 10.42.{i}.0/24\n"
                    f"  latest handshake: {ago}\n  transfer: 1.20 KiB received, 3.40 KiB sent\n"
                    "  persistent keepalive: every 25 seconds\n"
                    for i, n in enumerate(nodes) if n != node)
    return f"interface: flannel-wg\n  public key: {wg_key(1)}\n  private key: (hidden)\n  listening port: 51820\n{peers}"


def nodes_json(nodes, public=None):
    items = []
    for n in nodes:
        ann = {"flannel.alpha.coreos.com/backend-type": "extension",
               "flannel.alpha.coreos.com/public-ip": (public or {}).get(n, NODES[n]["ext"]),
               "flannel.alpha.coreos.com/backend-data": json.dumps({"PublicKey": wg_key(9)})}
        items.append({"metadata": {"name": n, "annotations": ann},
                      "status": {"addresses": [{"type": "InternalIP", "address": "10.10.0.2"},
                                               {"type": "ExternalIP", "address": NODES[n]["ext"]},
                                               {"type": "Hostname", "address": n}]}})
    return json.dumps({"items": items})


def wg_packets(peer, n):
    """N udp/51820 packets between k3s-cp's NIC and PEER, alternating direction (tcpdump -nn -q -t lines)."""
    out = (f"IP 10.10.0.2.51820 > {peer}.51820: UDP, length 148\n", f"IP {peer}.51820 > 10.10.0.2.51820: UDP, length 92\n")
    return "".join(out[i % 2] for i in range(n))


# 20 marked requests and their answers cross as at least 40 WireGuard packets; keepalives alone give 1-2.
PCAP_OK = (wg_packets(NODES[CPU1]["ext"], 50) +
           f"IP {NODES[CPU1]['ext']}.40312 > 10.10.0.2.6443: tcp 0\n"
           f"IP 10.10.0.2.6443 > {NODES[CPU1]['ext']}.40312: tcp 517\n"
           f"IP 10.10.0.2.53210 > {NODES[CPU1]['ext']}.10250: tcp 0\n"
           "IP 10.10.0.2.41000 > 169.254.169.254.53: UDP, length 40\n"
           "IP 10.10.0.2.41000 > 35.235.240.7.443: tcp 120\n")

FIREWALL = [
    {"name": "fw-cp-allow-iap-ssh", "network": "https://www.googleapis.com/compute/v1/projects/p/global/networks/k3s-core",
     "direction": "INGRESS", "sourceRanges": ["35.235.240.0/20"], "allowed": [{"IPProtocol": "tcp", "ports": ["22"]}],
     "targetServiceAccounts": ["k3s-cp@p.iam.gserviceaccount.com"]},
    {"name": "fw-k3s-cpu-1-to-cp", "network": "https://www.googleapis.com/compute/v1/projects/p/global/networks/k3s-core",
     "direction": "INGRESS", "sourceRanges": [NODES[CPU1]["ext"] + "/32"],
     "allowed": [{"IPProtocol": "tcp", "ports": ["6443"]}, {"IPProtocol": "udp", "ports": ["51820"]},
                 {"IPProtocol": "icmp"}],
     "targetServiceAccounts": ["k3s-cp@p.iam.gserviceaccount.com"]},
]
LISTENING = "tcpdump: listening on ens4, link-type EN10MB (Ethernet), snapshot length 262144 bytes\n"
# The operator's public IP: the fake checkip.amazonaws.com answers it (OPERATOR_IP is empty in the test .env).
OPERATOR = "198.51.100.200"
GPUCTL_RULES = (f"global\n  tcp   22          {OPERATOR}/32  operator ssh\n"
                "ruleset k3s-gpu-us-west-1 rs-1 us-west-1\n  udp   51820       192.0.2.10/32      k3s-cp wireguard\n"
                "ruleset k3s-gpu-us-east-1 rs-2 us-east-1\n  (no rules)\n")


def healthy(nodes=(CP, CPU1), mtu=1460):
    """Scripted answers for a cluster where every check A1-C8 passes."""
    def lines(app):  # _vr_pods: "NODE POD IP" for the Running Pods of app=APP
        return "".join(f"{n} {NODES[n][app]} {NODES[n][app + '_ip']}\n" for n in nodes)

    r = [
        kc("get nodes -o wide", stdout="".join(f"{n} Ready <none> 1h v1.36.5+k3s1\n" for n in nodes)),
        kc("get nodes", 'Ready")].status', stdout="".join(f"{n} True\n" for n in nodes)),
        kc("get nodes", '{.metadata.name}{"\\n"}', stdout="".join(f"{n}\n" for n in nodes)),
        kc("get nodes", "-o json", stdout=nodes_json(nodes)),
        kc("-n kube-system get pods -o wide", stdout="coredns-1 1/1 Running k3s-cp\n"),
        kc("-n kube-system get pods", "k8s-app in",
           stdout="metrics-server metrics-server-1 k3s-cp True\nkube-dns coredns-1 k3s-cp True\n"),
        kc("app=probe", "range .items", stdout=lines("probe")),
        kc("app=echo", "range .items", stdout=lines("echo")),
    ]
    for n in nodes:
        d = NODES[n]
        r += [kc(f"get node {n} ", "ExternalIP", stdout=d["ext"]),
              kc(f"describe node {n} ", stdout=f"Name: {n}\n"),
              kc("app=probe", f"spec.nodeName={n},", stdout=d["probe"]),
              kc("app=echo", f"spec.nodeName={n},", stdout=d["echo_ip"]),
              kc(f"get pod {d['probe']} ", "podIP", stdout=d["probe_ip"]),
              on(n, "sudo wg show flannel-wg", stdout=wg_show(n, nodes)),
              on(n, "sudo nft list table inet k3s_guard",
                 stdout="table inet k3s_guard {\n  counter packets 3 bytes 180 drop\n}\n")]
    r += [
        kc("-n np-denied exec client", "http://echo.nettest.svc.cluster.local:8080/", stdout="200"),
        kc("-n np-denied exec client", "http://hello.hello.svc.cluster.local", rc=7,
           stderr="curl: (7) Failed to connect to hello.hello.svc.cluster.local port 80 after 357 ms\n"),
        kc("exec", "-- curl -fsS -m 5 http://10.42.", stdout=WHOAMI),
        kc("exec", "ping -M do", f"-s {mtu - 107} ", rc=1, stderr="ping: sendmsg: Message too long\n"),
        kc("exec", "ping -M do", stdout="2 packets transmitted, 2 received, 0% packet loss\n"),
        kc("exec", "http://echo.nettest.svc.cluster.local:8080/", stdout=WHOAMI),
        kc("exec", "X-Marker", stdout="20/20 requests answered\n"),
        kc("exec", "iperf3 -s -1", stdout="Server listening on 5201 (test #1)\n"),
        kc("exec", "iperf3 -c", stdout=json.dumps({"end": {"sum_sent": {"bits_per_second": 4.2e8},
                                                           "sum_received": {"bits_per_second": 4.1e8}}})),
        kc("exec", "Metadata-Flavor", rc=28, stderr="curl: (28) Connection timed out after 3002 milliseconds\n"),
        kc("exec", "nslookup kubernetes.default.svc.cluster.local", stdout="Address: 10.43.0.1\n"),
        kc("exec", "http://hello.hello.svc.cluster.local", stdout="hello world\n"),
        kc("-n hello rollout status deploy/hello --timeout=120s", stdout='deployment "hello" successfully rolled out\n'),
        kc("-n hello get pods -o wide", stdout="hello-a 1/1 Running k3s-cpu-1\nhello-b 1/1 Running k3s-cpu-1\n"),
        kc("-n hello get pods", "qosClass", stdout="hello-a Guaranteed\nhello-b Guaranteed\n"),
        kc("-n hello get pods", "containerID",
           stdout=f"hello-a k3s-cpu-1 containerd://{HELLO_CID}\nhello-b k3s-cpu-1 containerd://{BURN_CID[::-1]}\n"),
        kc("-n hello get pods", '{.spec.nodeName}{"\\n"}', stdout="k3s-cpu-1\nk3s-cpu-1\n"),
        kc("-n hello top pod", stdout="NAME CPU(cores) MEMORY(bytes)\nhello-a 1m 3Mi\n"),
        kc("-n hello port-forward svc/hello", serve="hello world\n"),
        kc("-n limits-demo delete pod oom cpu-burn"),
        kc("apply -f", "limits-demo.yaml", stdout="pod/oom created\npod/cpu-burn created\n"),
        kc("-n limits-demo get pod oom", stdout="OOMKilled 137"),
        kc("-n limits-demo wait pod/cpu-burn --for=condition=Ready", stdout="pod/cpu-burn condition met\n"),
        kc("-n limits-demo top pod cpu-burn", stdout="NAME CPU(cores) MEMORY(bytes)\ncpu-burn 100m 1Mi\n"),
        kc("-n limits-demo get pod cpu-burn", stdout=f"k3s-cpu-1 containerd://{BURN_CID}"),
        kc("delete namespace np-denied"),
        kc("apply -f", "np-denied.yaml", stdout="namespace/np-denied created\npod/client created\n"),
        kc("-n np-denied wait pod/client --for=condition=Ready --timeout=120s", stdout="pod/client condition met\n"),
        on(CP, "ip -o route show default", stdout="default via 10.10.0.1 dev ens4 proto dhcp src 10.10.0.2 metric 100\n"),
        on(CP, "tcpdump -ni ens4 -U -w /tmp/b2.pcap", rc=124, stderr=LISTENING, sleep=2),   # still capturing
        on(CP, "sudo tcpdump -A -nn -r /tmp/b2.pcap", stdout="0\n"),
        on(CP, "sudo tcpdump -nn -q -t -r /tmp/b2.pcap", stdout=PCAP_OK),
        on(CP, "sudo rm -f /tmp/b2.pcap"),
        on(CPU1, "sudo find /sys/fs/cgroup -type d", HELLO_CID, stdout=cgroup(HELLO_CID) + "\n"),
        on(CPU1, "sudo find /sys/fs/cgroup -type d", BURN_CID, stdout=cgroup(BURN_CID) + "\n"),
        on(CPU1, f"cat {cgroup(HELLO_CID)}/memory.max {cgroup(HELLO_CID)}/cpu.max", stdout="67108864\n10000 100000\n"),
        on(CPU1, f"cat {cgroup(BURN_CID)}/cpu.stat",
           seq=[{"stdout": "usage_usec 9\nnr_periods 50\nnr_throttled 40\n"},
                {"stdout": "usage_usec 19\nnr_periods 150\nnr_throttled 140\n"}]),
        rule("gcloud", ["compute firewall-rules list"], stdout=json.dumps(FIREWALL)),
        rule("gpuctl", ["rules"], stdout=GPUCTL_RULES),
        rule("curl", ["checkip.amazonaws.com"], stdout=OPERATOR + "\n"),
    ]
    return r


class Cluster(Env):
    """Env plus checks.sh/verify.sh, a healthy scripted cluster, and a way to run one check."""

    def __init__(self, tmp, nodes=(CP, CPU1)):
        super().__init__(tmp)
        for rel in ("scripts/lib/checks.sh", "scripts/verify.sh", ".gitleaks.toml"):
            shutil.copy2(ROOT / rel, self.repo / rel)
        self.scenario["answers"] = healthy(nodes)
        self.scenario["gpu_ip"] = NODES[GPU1]["ext"]
        # curl to checkip.amazonaws.com (vr_init's operator IP lookup) is the fake; C3's local curl is real.
        (self.bin / "curl").write_text(
            '#!/bin/sh\ncase "$*" in *checkip.amazonaws.com*) exec "%s" "%s" curl "$@" ;; esac\nexec "%s" "$@"\n'
            % (sys.executable, FAKES, shutil.which("curl")))
        (self.bin / "curl").chmod(0o755)
        self.interrupted = ("", "")

    def first(self, *rules):
        """Answers that win over the healthy ones (the first matching rule answers)."""
        self.scenario["answers"][:0] = list(rules)

    def check(self, func, timeout_fn=NOT_REACHABLE, **extra):
        """Run FUNC through run_check under set -euo pipefail (as verify.sh does): (result, evidence text)."""
        run = self.repo / "verification" / "t"
        shutil.rmtree(run, ignore_errors=True)
        code = f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; run_check T title {func}"
        proc = self.run(["bash", "-c", code], VR_NO_WAIT="1", **{"BASH_FUNC_timeout%%": timeout_fn, **extra})
        assert proc.returncode == 0, proc.stderr
        result = re.search(r"^\| 1 \| T \| title \| (\w+) \|$", (run / "SUMMARY.md").read_text(), re.M).group(1)
        return result, (run / "01-T.txt").read_text()

    def interrupt(self, func, when, sig, again=None, **extra):
        """Run FUNC through run_check (as verify.sh does), send SIG to the whole process group (Ctrl-C
        reaches every process in the terminal's group) once a fake call holding WHEN was made; the
        exit status of the run. self.interrupted is then (the run's stderr, the check's evidence).
        AGAIN: send SIG a second time once a further fake call holding AGAIN (the cleanup's) was made.
        EXTRA: more environment for the run."""
        run = self.repo / "verification" / "s"
        code = f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; run_check T title {func}"
        proc = subprocess.Popen(["bash", "-c", code], cwd=self.repo, start_new_session=True,
                                env=self.environ({"VR_NO_WAIT": "1", "BASH_FUNC_timeout%%": NOT_REACHABLE,
                                                  **extra}),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 30
        while when not in (self.log.read_text() if self.log.exists() else ""):
            assert proc.poll() is None, proc.communicate()
            assert time.monotonic() < deadline, "the check never got there"
            time.sleep(0.05)
        time.sleep(0.3)                                    # the call is running: the check waits on it
        before = self.log.read_text().count(again) if again is not None else 0
        os.killpg(proc.pid, sig)
        if again is not None:
            while self.log.read_text().count(again) <= before:
                assert time.monotonic() < deadline, "the cleanup never got there"
                time.sleep(0.05)
            time.sleep(0.3)                                # the cleanup's call is running
            os.killpg(proc.pid, sig)
        _, err = proc.communicate(timeout=60)
        # SIGTERM ends the outer shell at once; the check's own shell cleans up after it: wait for the group
        while time.monotonic() < deadline + 60:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("processes of the interrupted check are still running")
        self.interrupted = (err.decode(errors="replace"), (run / "01-T.txt").read_text())
        return proc.returncode

    def kubectl(self):
        """The kubectl argv of every call, without --kubeconfig FILE."""
        return [a[2:] for a in self.argvs("kubectl")]

    def remote(self, node):
        """The commands run on NODE, in order."""
        if node == GPU1:
            return [a[-1] for a in self.argvs("ssh") if a[-2] == f"ubuntu@{NODES[GPU1]['ext']}"]
        return [a[a.index("--command") + 1] for a in self.argvs("gcloud")
                if a[:3] == ["compute", "ssh", node] and "--command" in a]


@pytest.fixture
def env(tmp_path):
    return Cluster(tmp_path)


@pytest.fixture
def env3(tmp_path):
    return Cluster(tmp_path, nodes=(CP, CPU1, GPU1))


def cmd_lines(text):
    return [line for line in text.splitlines() if line.startswith("$ ")]


# ---- helpers: vr_cap, vr_retry, node_ssh, node_external_ip, probe_on, echo_on ------------------

def test_vr_cap_records_the_command_and_output_and_returns_only_the_output(env):
    proc = env.run(["bash", "-c", "source scripts/lib/checks.sh; x=$(vr_cap echo hi 2>/dev/null); "
                    "echo \"got=$x\"; vr_cap echo shown >/dev/null; vr_cap --no-echo echo hidden >/dev/null"])
    assert proc.stdout.splitlines() == ["got=hi"]
    assert proc.stderr.splitlines() == ["$ echo shown", "shown", "$ echo hidden"]


@pytest.mark.parametrize("compat", [None, "42"])
def test_a_recorded_command_with_single_quotes_runs_as_shown_in_any_bash(env, compat):
    # F4-3: bash <= 4.2 (BASH_COMPAT=42, like macOS's /bin/bash 3.2) does no quote removal on the replacement
    # of ${x//\'/...} inside double quotes
    arg = "it's a 'quoted' word"
    extra = {"BASH_COMPAT": compat} if compat else {}
    proc = env.run(["bash", "-c", 'source scripts/lib/checks.sh; vr_cap printf "%s\\n" "$1" 2>&1 >/dev/null',
                    "sh", arg], **extra)
    shown = proc.stdout.splitlines()[0]
    assert shown == "$ printf '%s\\n' 'it'\\''s a '\\''quoted'\\'' word'"
    assert env.run(["bash", "-c", shown[2:]]).stdout == arg + "\n"


def test_vr_cap_returns_the_status_and_redacts_the_command(env):
    proc = env.run(["bash", "-c", "source scripts/lib/checks.sh; rc=0; vr_cap sh -c 'exit 3' 'token: abc' || rc=$?; "
                    "echo rc=$rc"])
    assert proc.stdout.splitlines() == ["rc=3"]
    assert "<redacted>" in proc.stderr and "abc" not in proc.stderr


def test_vr_retry_stops_at_the_first_success_and_gives_up_after_the_timeout(env):
    proc = env.run(["bash", "-c", "source scripts/lib/checks.sh; n=0; f(){ n=$((n+1)); ((n >= 3)); }; "
                    "vr_retry 90 10 f; echo n=$n; n=-100; rc=0; vr_retry 90 10 f || rc=$?; echo n=$n rc=$rc"],
                   VR_NO_WAIT="1")
    assert proc.stdout.splitlines() == ["n=3", "n=-90 rc=1"]           # 10 attempts: at 0, 10, ..., 90 s


def test_node_ssh_runs_cp_ssh_for_k3s_cp(env):
    proc = env.lib("load_env; node_ssh k3s-cp 'sudo wg show flannel-wg'")
    assert proc.returncode == 0, proc.stderr
    assert env.remote(CP) == ["sudo wg show flannel-wg"]


def test_node_external_ip_probe_on_and_echo_on(env):
    proc = env.lib("node_external_ip k3s-cpu-1; probe_on k3s-cp; echo_on k3s-cpu-1")
    assert proc.stdout.splitlines() == [NODES[CPU1]["ext"], NODES[CP]["probe"], NODES[CPU1]["echo_ip"]]
    assert env.kubectl()[1] == ["-n", "nettest", "get", "pods", "-l", "app=probe",
                                "--field-selector=spec.nodeName=k3s-cp,status.phase=Running", RT,
                                "-o", "jsonpath={.items[*].metadata.name}"]


def test_vr_pick_keeps_the_callers_line_variable(env):
    proc = env.run(["bash", "-c", "source scripts/lib/checks.sh; line=mine; _vr_pick b 'a p 1' 'b q 2'; echo \"$line\""])
    assert proc.stdout.splitlines() == ["q 2", "mine"]


def test_kc_within_bounds_kubectl_and_bounded_skips_a_timeout_function(env):
    # The offline tests replace `timeout` with a shell function (B5); bounded must run the real one.
    env.first(kc("exec", "stalled", sleep=5))
    proc = env.run(["bash", "-c", "source scripts/lib/checks.sh; kc_within 5 -n nettest exec p -- true; echo rc=$?; "
                    "kc_within 1 -n nettest exec stalled -- true; echo rc=$?; bounded 1 sleep 3; echo rc=$?"],
                   **{"BASH_FUNC_timeout%%": NOT_REACHABLE})
    assert proc.stdout.splitlines() == ["rc=0", "rc=124", "rc=124"]
    assert env.kubectl() == [["-n", "nettest", "exec", "p", "--", "true"], ["-n", "nettest", "exec", "stalled", "--", "true"]]


@pytest.mark.parametrize("call, bad", [("node_external_ip k3s-cp", "10.0.0.1 192.0.2.10"),
                                       ("probe_on k3s-cp", ""), ("echo_on k3s-cp", "none")])
def test_node_helpers_fail_on_a_missing_or_bad_answer(env, call, bad):
    env.first(kc("get node k3s-cp ", stdout=bad), kc("app=probe", stdout=bad), kc("app=echo", stdout=bad))
    assert env.lib(call).returncode != 0


# ---- A ------------------------------------------------------------------------------------------

def test_a1_passes_with_two_ready_nodes_and_records_the_commands(env):
    result, text = env.check("check_a1_nodes")
    assert result == "PASS", text
    assert f"$ kc get nodes -o wide {RT}" in cmd_lines(text)
    assert "2 of 2 nodes Ready" in text


@pytest.mark.parametrize("answer", ["k3s-cp True\nk3s-cpu-1 False\n", "k3s-cp True\nk3s-cpu-1 Unknown\n",
                                    "k3s-cp True\n", ""])
def test_a1_fails_unless_all_of_at_least_two_nodes_are_ready(env, answer):
    env.first(kc("get nodes", 'Ready")].status', stdout=answer))
    assert env.check("check_a1_nodes")[0] == "FAIL"


def test_a2_describes_every_node(env):
    result, text = env.check("check_a2_describe_nodes")
    assert result == "PASS", text
    assert [a for a in env.kubectl() if a[0] == "describe"] == [["describe", "node", CP, RT], ["describe", "node", CPU1, RT]]


def test_a2_fails_when_a_describe_fails(env):
    env.first(kc(f"describe node {CPU1} ", rc=1, stderr="error: the server doesn't have a resource type\n"))
    result, text = env.check("check_a2_describe_nodes")
    assert result == "FAIL"
    assert [a for a in env.kubectl() if a[0] == "describe"][-1][2] == CPU1   # the other nodes still described


def test_a3_passes_with_metrics_server_and_coredns_ready_on_the_cp(env):
    result, text = env.check("check_a3_system_pods")
    assert result == "PASS", text
    assert f"$ kc -n kube-system get pods -o wide {RT}" in cmd_lines(text)


@pytest.mark.parametrize("answer", [
    "metrics-server m-1 k3s-cp True\nkube-dns coredns-1 k3s-cpu-1 True\n",     # CoreDNS on a worker
    "metrics-server m-1 k3s-cp False\nkube-dns coredns-1 k3s-cp True\n",       # not Ready
    "kube-dns coredns-1 k3s-cp True\n",                                         # no metrics-server
    "metrics-server m-1 k3s-cp True\n",                                         # no CoreDNS
])
def test_a3_fails_when_a_system_pod_is_elsewhere_missing_or_not_ready(env, answer):
    env.first(kc("-n kube-system get pods", "k8s-app in", stdout=answer))
    assert env.check("check_a3_system_pods")[0] == "FAIL"


def test_a4_prints_the_flannel_annotations_and_passes_when_public_ip_is_the_external_ip(env):
    result, text = env.check("check_a4_flannel_annotations")
    assert result == "PASS", text
    assert f"flannel.alpha.coreos.com/public-ip={NODES[CPU1]['ext']}" in text
    assert "flannel.alpha.coreos.com/backend-type=extension" in text


def test_a4_fails_when_public_ip_differs_from_the_external_ip(env):
    env.first(kc("get nodes", "-o json", stdout=nodes_json((CP, CPU1), public={CPU1: "10.20.0.2"})))
    result, text = env.check("check_a4_flannel_annotations")
    assert result == "FAIL"
    assert "10.20.0.2" in text


def test_a4_uses_public_ip_overwrite_when_it_is_set(env):
    doc = json.loads(nodes_json((CP, CPU1), public={CPU1: "10.20.0.2"}))
    doc["items"][1]["metadata"]["annotations"]["flannel.alpha.coreos.com/public-ip-overwrite"] = NODES[CPU1]["ext"]
    env.first(kc("get nodes", "-o json", stdout=json.dumps(doc)))
    assert env.check("check_a4_flannel_annotations")[0] == "PASS"
    doc["items"][1]["metadata"]["annotations"]["flannel.alpha.coreos.com/public-ip-overwrite"] = "203.0.113.99"
    doc["items"][1]["metadata"]["annotations"]["flannel.alpha.coreos.com/public-ip"] = NODES[CPU1]["ext"]
    env.first(kc("get nodes", "-o json", stdout=json.dumps(doc)))
    assert env.check("check_a4_flannel_annotations")[0] == "FAIL"


def test_a4_output_passes_the_secret_scan(env):
    # backend-data holds the node's WireGuard public key; it must not fail vr_finish's gitleaks scan.
    if not shutil.which("gitleaks"):
        pytest.skip("gitleaks not installed")
    run = env.repo / "verification/g"
    proc = env.run(["bash", "-c", f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; "
                    "run_check A4 t check_a4_flannel_annotations; vr_finish"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---- B1 matrix ----------------------------------------------------------------------------------

def test_b1_runs_the_full_matrix_and_prints_an_n_by_n_table(env):
    result, text = env.check("check_b1_matrix")
    assert result == "PASS", text
    execs = [a for a in env.kubectl() if "exec" in a]
    for probe in (NODES[CP]["probe"], NODES[CPU1]["probe"]):
        for ip in (NODES[CP]["echo_ip"], NODES[CPU1]["echo_ip"]):
            assert ["-n", "nettest", "exec", probe, "--", "curl", "-fsS", "-m", "5", f"http://{ip}:8080/"] in execs
            for size in ("1352", "1353"):
                assert ["-n", "nettest", "exec", probe, "--", "ping", "-M", "do", "-c", "2", "-W", "2", "-s", size,
                        ip] in execs
        assert ["-n", "nettest", "exec", probe, "--", "curl", "-fsS", "-m", "5",
                "http://echo.nettest.svc.cluster.local:8080/"] in execs
    table = text[text.index("probe \\ echo"):].splitlines()
    assert table[0].split() == ["probe", "\\", "echo", CP, CPU1, "svc-dns"]
    assert table[1].split() == [CP, "ok", "ok", "ok"] and table[2].split() == [CPU1, "ok", "ok", "ok"]
    # every exec carries an outer bound (kubectl's own request timeout does not bound exec streams)
    execs = [line for line in cmd_lines(text) if " exec " in line]
    assert execs and all(line.startswith("$ kc_within 60 -n nettest exec ") for line in execs)


def test_b1_ping_sizes_follow_underlay_mtu(env):
    env.write_env(env.repo.joinpath(".env").read_text().replace("UNDERLAY_MTU=1460", "UNDERLAY_MTU=1400"))
    env.scenario["answers"] = healthy(mtu=1400)
    result, text = env.check("check_b1_matrix")
    assert result == "PASS", text
    sizes = {a[a.index("-s") + 1] for a in env.kubectl() if "ping" in a}
    assert sizes == {"1292", "1293"}


def test_b1_fails_when_the_oversized_ping_gets_through(env):
    env.first(kc("exec", NODES[CPU1]["probe"], "ping -M do", "-s 1353 ", f" {NODES[CP]['echo_ip']}"))
    result, text = env.check("check_b1_matrix")
    assert result == "FAIL"
    rows = {line.split()[0]: line.split()[1:] for line in text[text.index("probe \\ echo"):].splitlines()[1:3]}
    assert rows == {CP: ["ok", "ok", "ok"], CPU1: ["ping-1353-passed", "ok", "ok"]}


@pytest.mark.parametrize("answer, ok", [
    (dict(rc=1, stderr="ping: local error: message too long, mtu=1380\n"), True),
    (dict(rc=1, stdout="2 packets transmitted, 0 received, +2 errors, 100% packet loss, time 1001ms\n"), True),
    (dict(rc=1, stderr="error: unable to upgrade connection: container not found (\"netshoot\")\n"), False),
    (dict(rc=124), False),                                                  # the exec's bound ran out
])
def test_b1_oversized_ping_must_fail_as_an_mtu_rejection(env, answer, ok):
    env.first(kc("exec", NODES[CPU1]["probe"], "ping -M do", "-s 1353 ", f" {NODES[CP]['echo_ip']}", **answer))
    result, text = env.check("check_b1_matrix")
    assert result == ("PASS" if ok else "FAIL"), text
    rows = {line.split()[0]: line.split()[1:] for line in text[text.index("probe \\ echo"):].splitlines()[1:3]}
    assert rows[CPU1][0] == ("ok" if ok else "ping-1353-error")


def test_b1_fails_on_one_broken_cell_and_one_broken_dns_lookup(env):
    env.first(kc("exec", NODES[CP]["probe"], f"http://{NODES[CPU1]['echo_ip']}:8080/", rc=28),
              kc("exec", NODES[CPU1]["probe"], "echo.nettest.svc", rc=6))
    result, text = env.check("check_b1_matrix")
    assert result == "FAIL"
    rows = {line.split()[0]: line.split()[1:] for line in text[text.index("probe \\ echo"):].splitlines()[1:3]}
    assert rows == {CP: ["ok", "curl", "ok"], CPU1: ["ok", "ok", "FAIL"]}


def test_b1_fails_when_a_node_has_no_probe(env):
    env.first(kc("app=probe", "range .items", stdout=f"{CP} {NODES[CP]['probe']} {NODES[CP]['probe_ip']}\n"))
    result, text = env.check("check_b1_matrix")
    assert result == "FAIL"
    assert f"{CPU1}: no Running probe Pod" in text


# ---- B2 wire capture ----------------------------------------------------------------------------

def test_b2_captures_on_the_default_interface_while_other_nodes_curl_the_cp_echo(env):
    result, text = env.check("check_b2_wire")
    assert result == "PASS", text
    cp = env.remote(CP)
    assert cp[0] == "ip -o route show default"
    # umask 077: sudo keeps the stricter umask, so the pcap is never world-readable while it exists
    assert cp[1] == "umask 077 && sudo timeout 25 tcpdump -ni ens4 -U -w /tmp/b2.pcap 'not port 22'"
    # pipefail, and grep's 'no match' masked: a non-zero status means tcpdump could not read the file
    marker = re.fullmatch(r"bash -o pipefail -c 'sudo tcpdump -A -nn -r /tmp/b2\.pcap "
                          r"\| \{ grep -c -F -- (b2-marker-[0-9a-f]{16}) \|\| true; \}'", cp[2])
    assert marker, cp[2]
    assert cp[3:] == ["sudo tcpdump -nn -q -t -r /tmp/b2.pcap", "sudo rm -f /tmp/b2.pcap"]
    curl = next(a for a in env.kubectl() if "X-Marker" in " ".join(a))
    assert curl[:5] == ["-n", "nettest", "exec", NODES[CPU1]["probe"], "--"]
    assert f'-H "X-Marker: {marker.group(1)}" http://{NODES[CP]["echo_ip"]}:8080/' in curl[-1]
    assert "seq 20" in curl[-1]
    assert "51820/udp" in text and "6443/tcp" in text and "10250/tcp" in text
    assert f"50 udp/51820 packets with {NODES[CPU1]['ext']}" in text
    assert "$ cp_resolve" in cmd_lines(text)                       # the SSH zone lookup is recorded too


def test_b2_marker_is_new_each_run(env):
    env.check("check_b2_wire")
    env.check("check_b2_wire")
    greps = [c for c in env.remote(CP) if "grep -c" in c]
    assert len(greps) == 2 and greps[0] != greps[1]


@pytest.mark.parametrize("name, answers, why", [
    ("plaintext", [on(CP, "tcpdump -A", stdout="3\n")], "MARKER found in the capture"),
    ("no-wireguard", [on(CP, "tcpdump -nn -q", stdout=f"IP {NODES[CPU1]['ext']}.40312 > 10.10.0.2.6443: tcp 0\n")],
     "no udp/51820"),
    ("other-port", [on(CP, "tcpdump -nn -q", stdout=PCAP_OK + f"IP 10.10.0.2.43000 > {NODES[CPU1]['ext']}.8080: tcp 64\n")],
     "UNEXPECTED tcp 43000 > 8080"),
    ("curl-failed", [kc("exec", "X-Marker", rc=1, stdout="12/20 requests answered\n")],
     f"B2: not all 20 requests from {NODES[CPU1]['probe']}"),
    ("capture-failed", [on(CP, "tcpdump -ni ens4", rc=1, stderr="tcpdump: ens4: No such device exists\n")],
     "the capture on k3s-cp did not start"),
    # the requests finished after tcpdump's 25 s window: the capture may not hold them
    ("capture-ended", [on(CP, "tcpdump -ni ens4", rc=124, stderr=LISTENING),
                       kc("exec", "X-Marker", stdout="20/20 requests answered\n", sleep=1)],
     "the capture on k3s-cp ended before the marked requests finished"),
    # keepalives only: the marked requests did not cross as WireGuard packets with that peer
    ("too-few-wireguard", [on(CP, "tcpdump -nn -q", stdout=wg_packets(NODES[CPU1]["ext"], 39))],
     f"39 udp/51820 packets with {NODES[CPU1]['ext']} (want at least 40"),
    # the peer's traffic is on the wire under another address: nothing of it would be examined
    ("peer-not-on-wire", [on(CP, "tcpdump -nn -q", stdout=wg_packets("203.0.113.77", 60))],
     f"0 udp/51820 packets with {NODES[CPU1]['ext']}"),
])
def test_b2_fails_and_still_removes_the_pcap(env, name, answers, why):
    env.first(*answers)
    result, text = env.check("check_b2_wire")
    assert result == "FAIL", text
    assert why in text
    assert env.remote(CP)[-1] == "sudo rm -f /tmp/b2.pcap"


def test_b2_does_not_call_an_unreadable_capture_marker_free(env):
    # tcpdump -r fails: grep's count of 0 must not read as "the marker is not in the capture"
    env.first(on(CP, "sudo tcpdump -A -nn -r", rc=1, stdout="0\n",
                 stderr="tcpdump: /tmp/b2.pcap: truncated dump file; tried to read 16 header bytes\n"))
    result, text = env.check("check_b2_wire")
    assert result == "FAIL"
    assert "not in the capture" not in text and "B2 ①: cannot search the capture (exit 1)" in text


def assert_b2_interrupt_cleanup(env):
    """The last command on k3s-cp stops the capture (pkill), waits until it is gone, then removes the file;
    its pattern matches the capture's command line but not the cleanup's own."""
    cmds = env.remote(CP)
    last = cmds[-1]
    assert "sudo pkill -f" in last and "pgrep -f" in last and last.endswith("sudo rm -f /tmp/b2.pcap")
    assert last.index("pkill") < last.index("pgrep") < last.index("rm -f")
    pattern = re.search(r"p='([^']+)'", last).group(1)
    capture = next(c for c in cmds if "tcpdump -ni ens4 -U -w /tmp/b2.pcap" in c)
    assert re.search(pattern, capture) and re.search(pattern, f"bash -c {capture}")
    assert not re.search(pattern, last)
    assert not any("tcpdump -nn -q" in c for c in cmds)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_b2_removes_the_pcap_when_interrupted_mid_capture(env, sig):
    env.first(on(CP, "tcpdump -ni ens4", rc=124, stderr=LISTENING, sleep=30))
    rc = env.interrupt("check_b2_wire", "X-Marker", sig)
    assert rc == -sig                                              # the run stops: verify.sh does not go on
    assert_b2_interrupt_cleanup(env)


def test_b2_interrupted_before_the_capture_said_it_listens_stops_it_before_removing_the_file(env):
    # Ctrl-C during the start wait: tcpdump may open /tmp/b2.pcap only after a plain rm
    env.first(on(CP, "tcpdump -ni ens4", sleep=30))
    assert env.interrupt("check_b2_wire", "tcpdump -ni ens4", signal.SIGINT) == -signal.SIGINT
    assert_b2_interrupt_cleanup(env)
    assert not any("X-Marker" in " ".join(a) for a in env.kubectl())


def test_b2_waits_out_a_late_capture_before_removing_it_when_the_start_was_not_seen(env):
    env.first(on(CP, "tcpdump -ni ens4", rc=255, stderr="ERROR: (gcloud.compute.ssh) [/usr/bin/ssh] exited\n"))
    result, text = env.check("check_b2_wire")
    assert result == "FAIL"
    wait = "waiting 30 s for a capture that may have started late, then removing its file"
    assert wait in text and "(VR_NO_WAIT: skipped sleep 30)" in text.split(wait, 1)[1]
    assert env.remote(CP)[-1] == "sudo rm -f /tmp/b2.pcap"


def test_b2_fails_when_the_pcap_cannot_be_removed(env):
    env.first(on(CP, "sudo rm -f /tmp/b2.pcap", rc=1))
    assert env.check("check_b2_wire")[0] == "FAIL"


def test_b2_fails_without_a_default_interface(env):
    env.first(on(CP, "ip -o route show default", stdout="default via 10.10.0.1 proto dhcp\n"))
    result, text = env.check("check_b2_wire")
    assert result == "FAIL"
    assert not any("tcpdump" in c for c in env.remote(CP))


# ---- B3 wireguard -------------------------------------------------------------------------------

def test_b3_reads_wg_show_on_every_node_without_dump(env):
    result, text = env.check("check_b3_wg")
    assert result == "PASS", text
    assert env.remote(CP) == ["sudo wg show flannel-wg"] and env.remote(CPU1) == ["sudo wg show flannel-wg"]
    assert "dump" not in text and "private key: <redacted>" in text
    # the SSH zone lookups (Terraform outputs) are recorded like every other command
    assert {"$ cp_resolve", f"$ _worker_zone {CPU1}"} <= set(cmd_lines(text))


def test_b3_covers_a_gpu_node_over_plain_ssh(env3):
    assert env3.check("check_b3_wg")[0] == "PASS"
    assert env3.remote(GPU1) == ["sudo wg show flannel-wg"]


@pytest.mark.parametrize("ago, ok", [("Now", True), ("59 seconds ago", True), ("2 minutes, 59 seconds ago", True),
                                     ("3 minutes ago", True), ("3 minutes, 1 second ago", False),
                                     ("1 hour, 2 seconds ago", False), ("1 day ago", False)])
def test_b3_handshake_must_be_at_most_three_minutes_old(env, ago, ok):
    env.first(on(CPU1, "wg show", stdout=wg_show(CPU1, (CP, CPU1), ago=ago)))
    assert env.check("check_b3_wg")[0] == ("PASS" if ok else "FAIL")


def test_b3_fails_on_a_missing_peer_or_handshake(env):
    env.first(on(CP, "wg show", stdout=wg_show(CP, (CP,))))
    assert env.check("check_b3_wg")[0] == "FAIL"
    no_handshake = "\n".join(line for line in wg_show(CP, (CP, CPU1)).splitlines() if "handshake" not in line)
    env.first(on(CP, "wg show", stdout=no_handshake))
    assert env.check("check_b3_wg")[0] == "FAIL"


# ---- B4 iperf3 ----------------------------------------------------------------------------------

# Stops the iperf3 server in a probe; pkill's 1 (no process matched: nothing left) counts as done, and its
# own errors (2, 3) keep their code.
B4_STOP = "pkill -x iperf3; rc=$?; [ $rc -le 1 ] || exit $rc"
PKILL_CP = ["-n", "nettest", "exec", NODES[CP]["probe"], "--", "sh", "-c", B4_STOP]


def test_b4_measures_cpu1_to_cp_and_records_bits_per_second(env):
    result, text = env.check("check_b4_iperf")
    assert result == "PASS", text
    execs = [a for a in env.kubectl() if "exec" in a]
    # --forceflush: without a tty iperf3 buffers "Server listening" until it exits
    assert ["-n", "nettest", "exec", NODES[CP]["probe"], "--", "timeout", "150", "iperf3", "-s", "-1",
            "--forceflush"] in execs
    assert ["-n", "nettest", "exec", NODES[CPU1]["probe"], "--", "timeout", "120", "iperf3", "-c",
            NODES[CP]["probe_ip"], "-n", "50M", "-J"] in execs
    assert "bits_per_second=410000000" in text
    # a server left by an interrupted run is stopped first; the one-off server then ends by itself
    assert [a for a in execs if "pkill" in " ".join(a)] == [PKILL_CP]
    assert execs.index(PKILL_CP) < next(i for i, a in enumerate(execs) if "-s" in a and "iperf3" in a)


def test_b4_also_measures_cpu1_to_each_gpu_node(env3):
    assert env3.check("check_b4_iperf")[0] == "PASS"
    clients = [a[a.index("-c") + 1] for a in env3.kubectl() if "iperf3" in a and "-c" in a]
    assert clients == [NODES[CP]["probe_ip"], NODES[GPU1]["probe_ip"]]


def test_b4_fails_when_the_client_fails_and_stops_the_server(env):
    env.first(kc("exec", "iperf3 -c", rc=1, stderr="iperf3: error - unable to connect to server\n"))
    assert env.check("check_b4_iperf")[0] == "FAIL"
    assert env.kubectl()[-1] == PKILL_CP                     # so a re-run can listen on 5201 again


def test_b4_goes_on_when_no_server_was_left_over(env):
    env.first(kc("exec", B4_STOP, run_with={"pkill": 1}))         # pkill: no process matched
    result, text = env.check("check_b4_iperf")
    assert result == "PASS", text


@pytest.mark.parametrize("pkill_rc, rc", [(0, 0), (1, 0), (2, 2), (3, 3)])
def test_b4_server_stop_counts_no_iperf3_left_as_done_but_not_pkills_own_errors(tmp_path, pkill_rc, rc):
    (tmp_path / "pkill").write_text(f"#!/bin/sh\nexit {pkill_rc}\n")
    (tmp_path / "pkill").chmod(0o755)
    assert subprocess.run(["sh", "-c", B4_STOP], env={"PATH": f"{tmp_path}:/usr/bin:/bin"}).returncode == rc


@pytest.mark.parametrize("slow, mark", [
    ("iperf3 -s -1", "forceflush"),             # Ctrl-C before the server said it listens
    ("iperf3 -c", "50M"),                        # ... while the client runs, after the one-off server ended
])
def test_b4_interrupt_with_no_iperf3_running_is_not_a_false_alarm(env, slow, mark):
    # F3-2: the cleanup's pkill matches nothing, which is a clean result, not "cleanup FAILED (exit 1)"
    env.first(kc("exec", B4_STOP, run_with={"pkill": 1}), kc("exec", slow, sleep=30))
    assert env.interrupt("check_b4_iperf", mark, signal.SIGINT) == -signal.SIGINT
    assert env.kubectl()[-1] == PKILL_CP
    err, text = env.interrupted
    for out in (err, text):
        assert "cleanup done" in out and "FAILED" not in out and "may remain" not in out


def test_b4_interrupt_cleanup_still_reports_a_failed_pkill(env):
    env.first(kc("exec", B4_STOP, run_with={"pkill": 3}), kc("exec", "iperf3 -c", sleep=30))
    assert env.interrupt("check_b4_iperf", "50M", signal.SIGINT) == -signal.SIGINT
    err, _ = env.interrupted
    # F4-1: pkill's own code (3: fatal), not 1, which would read as its harmless "no process matched"
    assert "cleanup FAILED (exit 3); may remain: an iperf3 server on port 5201" in err


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_b4_stops_the_iperf3_server_when_interrupted(env, sig):
    # else it listens on 5201 for up to 150 s and an immediate re-run fails with "address in use"
    env.first(kc("exec", "iperf3 -c", sleep=30))
    assert env.interrupt("check_b4_iperf", "50M", sig) == -sig        # the client is running
    assert env.kubectl()[-1] == PKILL_CP
    err, text = env.interrupted
    assert "may remain" not in err and "cleanup done" in err


def test_b4_fails_when_the_server_never_says_it_listens_and_stops_it(env):
    env.first(kc("exec", "iperf3 -s -1", stdout=""))
    result, text = env.check("check_b4_iperf")
    assert result == "FAIL"
    assert "the iperf3 server on k3s-cp did not start" in text
    assert not any("iperf3 -c" in " ".join(a) for a in env.kubectl())
    assert env.kubectl()[-1] == PKILL_CP


# ---- B5 negative tests --------------------------------------------------------------------------

def test_b5_passes_when_everything_that_must_fail_fails(env):
    result, text = env.check("check_b5_negative")
    assert result == "PASS", text
    cp_ip = NODES[CP]["ext"]
    assert f"fake timeout 3 bash -c </dev/tcp/{cp_ip}/6443" in text
    assert f"fake timeout 3 bash -c </dev/tcp/{cp_ip}/10250" in text
    meta = ["curl", "-m", "3", "-H", "Metadata-Flavor: Google", "http://169.254.169.254/computeMetadata/v1/"]
    execs = [a for a in env.kubectl() if "exec" in a]
    assert [a[3] for a in execs if a[5:] == meta] == [NODES[CP]["probe"], NODES[CPU1]["probe"]]
    assert [a[3] for a in execs if "nslookup" in a] == [NODES[CP]["probe"], NODES[CPU1]["probe"]]
    assert env.remote(CP) == ["sudo nft list table inet k3s_guard"]
    assert env.remote(CPU1) == ["sudo nft list table inet k3s_guard"]
    assert "counter packets 3" in text


def test_b5_fails_when_the_api_port_answers_from_here(env):
    result, text = env.check("check_b5_negative", timeout_fn=REACHABLE)
    assert result == "FAIL"
    assert "reachable from this machine" in text


@pytest.mark.parametrize("rc", [0, 1, 126, 124])
def test_b5_metadata_must_fail_inside_curl(env, rc):
    # 0: the metadata server answered; 1/126: kubectl or the probe failed, so curl's verdict is unknown;
    # 124: the exec's outer bound ran out, which is not curl's own timeout (28).
    env.first(kc("exec", NODES[CPU1]["probe"], "Metadata-Flavor", rc=rc))
    assert env.check("check_b5_negative")[0] == "FAIL"


def test_b5_does_not_probe_metadata_on_a_gpu_node_but_reads_its_guard(env3):
    assert env3.check("check_b5_negative")[0] == "PASS"
    meta = [a[3] for a in env3.kubectl() if "Metadata-Flavor: Google" in a]
    assert NODES[GPU1]["probe"] not in meta
    assert env3.remote(GPU1) == ["sudo nft list table inet k3s_guard"]


def test_b5_fails_when_dns_fails_or_the_guard_table_is_missing(env):
    env.first(kc("exec", NODES[CPU1]["probe"], "nslookup", rc=1))
    assert env.check("check_b5_negative")[0] == "FAIL"
    env.scenario["answers"] = healthy()
    env.first(on(CPU1, "nft list table", rc=1, stderr="Error: No such file or directory\n"))
    assert env.check("check_b5_negative")[0] == "FAIL"


# ---- B6 firewall inventory ----------------------------------------------------------------------

def test_b6_lists_the_gcp_rules_and_the_lambda_rules(env):
    result, text = env.check("check_b6_firewall_inventory")
    assert result == "PASS", text
    assert env.argvs("gcloud") == [["compute", "firewall-rules", "list", "--project", "k3s-demo-123456",
                                    "--filter=name~^fw-", "--format=json"]]
    assert env.argvs("gpuctl") == [["rules"]]
    row = next(line for line in text.splitlines() if line.startswith("fw-k3s-cpu-1-to-cp"))
    assert row.split() == ["fw-k3s-cpu-1-to-cp", "k3s-core", "INGRESS", NODES[CPU1]["ext"] + "/32",
                           "tcp:6443,udp:51820,icmp", "k3s-cp@p.iam.gserviceaccount.com"]
    assert "ruleset k3s-gpu-us-west-1" in text
    assert "$ bounded 120 gcloud compute firewall-rules list --project <GCP_PROJECT> " in text
    # .env has no OPERATOR_IP: the address checkip.amazonaws.com reports (gpuctl's) is masked
    assert OPERATOR not in text and "<operator_ip>/32  operator ssh" in text


def test_b6_evidence_masks_the_project_id_and_the_operator_ip(env):
    # verification/final is committed: no real project id or operator address in it (Ruling R39)
    env.write_env(env.repo.joinpath(".env").read_text() + "OPERATOR_IP=198.51.100.200\n")
    rules = json.loads(json.dumps(FIREWALL).replace("@p.iam", "@k3s-demo-123456.iam"))
    rules[0]["sourceRanges"].append("198.51.100.200/32")
    env.first(rule("gcloud", ["compute firewall-rules list"], stdout=json.dumps(rules)),
              rule("gpuctl", ["rules"], stdout=GPUCTL_RULES + "  tcp   22          198.51.100.20/32   other\n"))
    result, text = env.check("check_b6_firewall_inventory")
    assert result == "PASS", text
    assert "k3s-demo-123456" not in text and "198.51.100.200" not in text
    assert "k3s-cp@<GCP_PROJECT>.iam.gserviceaccount.com" in text
    assert "<operator_ip>/32  operator ssh" in text and "35.235.240.0/20,<operator_ip>/32" in text
    assert "198.51.100.20/32" in text                                # only that exact address is masked


@pytest.mark.parametrize("env_line, curl", [
    ("", rule("curl", ["checkip.amazonaws.com"], rc=6, stderr="curl: (6) Could not resolve host\n")),
    ("", rule("curl", ["checkip.amazonaws.com"], stdout="<html>captive portal</html>\n")),
    ("OPERATOR_IP=not-an-ip\n", None),                             # gpuctl refuses it; nothing to look up
])
def test_b6_fails_with_the_operator_rule_masked_when_the_operator_ip_is_unknown(env, env_line, curl):
    env.write_env(env.repo.joinpath(".env").read_text() + env_line)
    if curl:
        env.first(curl)
    result, text = env.check("check_b6_firewall_inventory")
    assert result == "FAIL", text
    assert OPERATOR not in text and "<operator_ip>/32  operator ssh" in text
    assert "B6: cannot determine the operator's public IP" in text
    assert "Operator IP: not determined" in (env.repo / "verification/t/SUMMARY.md").read_text()
    assert len(env.argvs("curl")) == (1 if curl else 0)


def test_b6_passes_with_an_unknown_operator_ip_when_no_rule_holds_it(env):
    no_operator_rule = "global\n  (no rules)\n" + GPUCTL_RULES.split("\n", 2)[2]
    env.first(rule("curl", ["checkip.amazonaws.com"], rc=28), rule("gpuctl", ["rules"], stdout=no_operator_rule))
    result, text = env.check("check_b6_firewall_inventory")
    assert result == "PASS", text


# ---- the operator's IP is masked in all evidence (spec §9; Ruling R39) --------------------------

def run_two_checks(env, body):
    """vr_init, then two checks running BODY; (the run directory, the shell's stdout + stderr)."""
    run = env.repo / "verification" / "m"
    code = (f"set -euo pipefail; source scripts/lib/checks.sh; vr_init {run}; f(){{ {body}; }}; "
            "run_check T one f; run_check U two f; vr_finish")
    proc = env.run(["bash", "-c", code])
    return run, proc.stdout + proc.stderr


def test_operator_ip_from_checkip_is_looked_up_once_and_masked_in_every_check(env):
    # .env.example leaves OPERATOR_IP empty: gpuctl then uses what checkip.amazonaws.com reports
    run, out = run_two_checks(env, f"echo 'ssh from {OPERATOR}:22'; vr_cmd echo {OPERATOR} >&2")
    assert env.argvs("curl") == [["-fsS", "--max-time", "5", "https://checkip.amazonaws.com"]]
    for name in ("01-T.txt", "02-U.txt"):
        text = (run / name).read_text()
        assert OPERATOR not in text
        assert "ssh from <operator_ip>:22" in text and "$ echo <operator_ip>" in text
    summary = (run / "SUMMARY.md").read_text()
    assert "Operator IP: masked as <operator_ip> (checkip.amazonaws.com)" in summary
    assert OPERATOR not in summary + out


def test_operator_ip_from_env_is_trimmed_and_not_looked_up(env):
    env.write_env(env.repo.joinpath(".env").read_text() + f'OPERATOR_IP=" {OPERATOR} "\n')
    run, out = run_two_checks(env, f"echo 'from {OPERATOR}'")
    assert env.argvs("curl") == []
    assert "from <operator_ip>" in (run / "02-U.txt").read_text()
    assert "Operator IP: masked as <operator_ip> (OPERATOR_IP in .env)" in (run / "SUMMARY.md").read_text()


def test_unknown_operator_ip_is_reported_in_the_summary_and_to_the_operator(env):
    env.first(rule("curl", ["checkip.amazonaws.com"], rc=6))
    run, out = run_two_checks(env, "echo hi")
    assert "Operator IP: not determined" in (run / "SUMMARY.md").read_text()
    assert "cannot determine the operator's public IP" in out


@pytest.mark.parametrize("line, want", [
    ("{ip},{ip}", "<operator_ip>,<operator_ip>"),                    # one separator between two occurrences
    ("{ip} {ip}", "<operator_ip> <operator_ip>"),
    ("{ip},{ip},{ip},{ip}", ",".join(["<operator_ip>"] * 4)),
    ("{ip}/32 x{ip}.51820", "<operator_ip>/32 x<operator_ip>.51820"),
    ("{ip}0 1{ip} 10.{ip} 198.51.100.20 198x51x100x200", "{ip}0 1{ip} 10.{ip} 198.51.100.20 198x51x100x200"),
])
def test_operator_ip_mask_catches_adjacent_occurrences_and_only_whole_addresses(env, line, want):
    env.write_env(env.repo.joinpath(".env").read_text() + f"OPERATOR_IP={OPERATOR}\n")
    proc = env.run(["bash", "-c", "source scripts/lib/checks.sh; vr_operator_ip; redact"],
                   stdin=line.format(ip=OPERATOR) + "\n")
    assert proc.stdout == want.format(ip=OPERATOR) + "\n", proc.stderr


@pytest.mark.parametrize("change", [
    {"sourceRanges": ["10.20.0.0/24"]},
    {"sourceRanges": ["198.51.100.11/32", "198.51.100.0/24"]},
    {"sourceRanges": []},
    {"sourceRanges": None},
    {"sourceTags": ["k3s"], "sourceRanges": None},
])
def test_b6_fails_on_any_gcp_source_other_than_a_single_ip_or_iap(env, change):
    rules = json.loads(json.dumps(FIREWALL))
    rules[1].update(change)
    rules[1] = {k: v for k, v in rules[1].items() if v is not None}
    env.first(rule("gcloud", ["compute firewall-rules list"], stdout=json.dumps(rules)))
    result, text = env.check("check_b6_firewall_inventory")
    assert result == "FAIL", text
    assert "fw-k3s-cpu-1-to-cp" in text.split("VIOLATION", 1)[1]


def test_b6_fails_on_a_lambda_rule_wider_than_one_ip(env):
    env.first(rule("gpuctl", ["rules"], stdout=GPUCTL_RULES + "  udp   51820       192.0.2.0/24      bad\n"))
    result, text = env.check("check_b6_firewall_inventory")
    assert result == "FAIL"
    assert "192.0.2.0/24" in text.split("VIOLATION", 1)[1]


def test_b6_fails_when_there_are_no_rules_or_a_command_fails(env):
    env.first(rule("gcloud", ["compute firewall-rules list"], stdout="[]"))
    assert env.check("check_b6_firewall_inventory")[0] == "FAIL"
    env.scenario["answers"] = healthy()
    env.first(rule("gpuctl", ["rules"], rc=2, stderr="gpuctl: LAMBDA_API_KEY is not set\n"))
    assert env.check("check_b6_firewall_inventory")[0] == "FAIL"


# ---- C ------------------------------------------------------------------------------------------

def test_c1_rollout_then_pods(env):
    result, text = env.check("check_c1_rollout")
    assert result == "PASS", text
    assert env.kubectl() == [["-n", "hello", "rollout", "status", "deploy/hello", "--timeout=120s"],
                             ["-n", "hello", "get", "pods", "-o", "wide", RT]]
    env.first(kc("rollout status", rc=1, stderr="error: timed out waiting for the condition\n"))
    assert env.check("check_c1_rollout")[0] == "FAIL"


def test_c2_curls_hello_from_a_probe_on_a_node_without_a_hello_replica(env):
    result, text = env.check("check_c2_cross_node_curl")
    assert result == "PASS", text
    curl = [a for a in env.kubectl() if "http://hello.hello.svc.cluster.local" in a]
    assert curl == [["-n", "nettest", "exec", NODES[CP]["probe"], "--", "curl", "-fsS", "-m", "5",
                     "http://hello.hello.svc.cluster.local"]]


def test_c2_falls_back_to_a_probe_that_differs_from_some_replica(env3):
    env3.first(kc("-n hello get pods", '{.spec.nodeName}{"\\n"}', stdout=f"{CP}\n{CPU1}\n{GPU1}\n"))
    assert env3.check("check_c2_cross_node_curl")[0] == "PASS"
    curl = [a for a in env3.kubectl() if "http://hello.hello.svc.cluster.local" in a]
    assert curl[0][3] == NODES[CP]["probe"]


@pytest.mark.parametrize("answer", [dict(stdout="hello world!\n"), dict(stdout="Hello World\n"),
                                    dict(stdout="", rc=28), dict(stdout="hello world\nhello world\n")])
def test_c2_needs_exactly_hello_world(env, answer):
    env.first(kc("exec", "http://hello.hello.svc.cluster.local", **answer))
    assert env.check("check_c2_cross_node_curl")[0] == "FAIL"


def test_c3_port_forwards_curls_and_stops_the_forward(env):
    port = free_port()
    result, text = env.check("check_c3_port_forward", VR_PF_PORT=str(port))
    assert result == "PASS", text
    assert f"$ kc -n hello port-forward svc/hello {port}:80 &" in cmd_lines(text)
    assert f"$ curl -fsS -m 5 localhost:{port}" in cmd_lines(text)
    with pytest.raises(OSError):                                   # the forward is gone
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


def test_c3_fails_on_a_wrong_answer_and_still_stops_the_forward(env):
    port = free_port()
    env.first(kc("-n hello port-forward", serve="goodbye\n"))
    assert env.check("check_c3_port_forward", VR_PF_PORT=str(port))[0] == "FAIL"
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


def test_c3_fails_when_the_forward_never_starts(env):
    env.first(kc("-n hello port-forward", rc=1, stderr="error: unable to forward port\n"))
    assert env.check("check_c3_port_forward", VR_PF_PORT=str(free_port()))[0] == "FAIL"


def test_c4_qos(env):
    assert env.check("check_c4_qos")[0] == "PASS"
    env.first(kc("qosClass", stdout="hello-a Guaranteed\nhello-b Burstable\n"))
    assert env.check("check_c4_qos")[0] == "FAIL"
    env.first(kc("qosClass", stdout=""))
    assert env.check("check_c4_qos")[0] == "FAIL"


def test_c5_reads_the_container_cgroup_on_its_node(env):
    result, text = env.check("check_c5_cgroup")
    assert result == "PASS", text
    assert env.remote(CPU1) == [f'sudo find /sys/fs/cgroup -type d -name "*{HELLO_CID}*"',
                                f"cat {cgroup(HELLO_CID)}/memory.max {cgroup(HELLO_CID)}/cpu.max"]


@pytest.mark.parametrize("values", ["67108865\n10000 100000\n", "max\n10000 100000\n", "67108864\n20000 100000\n",
                                    "67108864\nmax 100000\n", "67108864\n"])
def test_c5_compares_memory_max_and_cpu_max_exactly(env, values):
    env.first(on(CPU1, "memory.max", stdout=values))
    assert env.check("check_c5_cgroup")[0] == "FAIL"


def test_c5_fails_without_a_cgroup_directory_or_container_id(env):
    env.first(on(CPU1, "sudo find /sys/fs/cgroup", stdout=""))
    assert env.check("check_c5_cgroup")[0] == "FAIL"
    env.scenario["answers"] = healthy()
    env.log.unlink()
    env.first(kc("-n hello get pods", "containerID", stdout="hello-a k3s-cpu-1 \n"))
    result, _ = env.check("check_c5_cgroup")
    assert result == "FAIL" and env.remote(CPU1) == []


def test_c6_retries_until_top_returns_a_data_row(env):
    env.first(kc("-n hello top pod", seq=[dict(rc=1, stderr="error: metrics not available yet\n"),
                                          dict(stdout="NAME CPU(cores) MEMORY(bytes)\n"),
                                          dict(stdout="NAME CPU(cores) MEMORY(bytes)\nhello-a 1m 3Mi\n")]))
    result, text = env.check("check_c6_top")
    assert result == "PASS", text
    assert len([a for a in env.kubectl() if "top" in a]) == 3


def test_c6_gives_up_after_90_seconds(env):
    env.first(kc("-n hello top pod", rc=1, stderr="error: metrics not available yet\n"))
    assert env.check("check_c6_top")[0] == "FAIL"
    assert len([a for a in env.kubectl() if "top" in a]) == 10


def limits_calls(env):
    return [a for a in env.kubectl() if "limits-demo" in " ".join(a)]


def test_c7_oom_and_throttling_then_cleanup(env):
    result, text = env.check("check_c7_limits_demo")
    assert result == "PASS", text
    calls = limits_calls(env)
    delete = ["-n", "limits-demo", "delete", "pod", "oom", "cpu-burn", "--ignore-not-found", "--timeout=90s"]
    assert calls[0] == delete and calls[-1] == delete
    assert calls[1][:2] == ["apply", "-f"] and calls[1][2].endswith("/k8s/limits-demo/limits-demo.yaml")
    assert env.remote(CPU1) == [f'sudo find /sys/fs/cgroup -type d -name "*{BURN_CID}*"',
                                f"cat {cgroup(BURN_CID)}/cpu.stat", f"cat {cgroup(BURN_CID)}/cpu.stat"]
    assert "nr_throttled 40 -> 140" in text


@pytest.mark.parametrize("answers", [
    [kc("-n limits-demo get pod oom", stdout="Error 1")],
    [kc("-n limits-demo get pod oom", stdout="OOMKilled 1")],
    [kc("-n limits-demo get pod oom", stdout=" ")],                               # never terminated
    [kc("top pod cpu-burn", stdout="NAME CPU(cores) MEMORY(bytes)\ncpu-burn 150m 1Mi\n")],
    [kc("top pod cpu-burn", stdout="NAME CPU(cores) MEMORY(bytes)\ncpu-burn 1 1Mi\n")],
    [kc("top pod cpu-burn", rc=1, stderr="error: metrics not available yet\n")],
    [on(CPU1, "cpu.stat", stdout="nr_throttled 40\n")],                         # not increasing
    [kc("wait pod/cpu-burn", rc=1, stderr="error: timed out waiting for the condition\n")],
])
def test_c7_fails_and_still_deletes_both_pods(env, answers):
    env.first(*answers)
    result, text = env.check("check_c7_limits_demo")
    assert result == "FAIL", text
    assert limits_calls(env)[-1] == ["-n", "limits-demo", "delete", "pod", "oom", "cpu-burn", "--ignore-not-found",
                                     "--timeout=90s"]


def test_c7_still_deletes_both_pods_when_the_ssh_zone_lookup_dies(env):
    # _worker_zone exits the shell (die) when Terraform's output cannot be read; that must not skip the delete
    env.first(rule("terraform", ["output", "worker_zones"], rc=1, stderr="Error: Failed to load state\n"))
    result, text = env.check("check_c7_limits_demo")
    assert result == "FAIL", text
    assert "cannot read worker_zones" in text
    assert limits_calls(env)[-1] == ["-n", "limits-demo", "delete", "pod", "oom", "cpu-burn", "--ignore-not-found",
                                     "--timeout=90s"]


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_c7_deletes_both_pods_when_interrupted(env, sig):
    env.first(kc("-n limits-demo wait pod/cpu-burn", sleep=30))
    assert env.interrupt("check_c7_limits_demo", "pod/cpu-burn", sig) == -sig
    assert limits_calls(env)[-1] == ["-n", "limits-demo", "delete", "pod", "oom", "cpu-burn", "--ignore-not-found",
                                     "--wait=false", RT]


def test_c7_accepts_a_reading_of_exactly_110m(env):
    env.first(kc("top pod cpu-burn", stdout="NAME CPU(cores) MEMORY(bytes)\ncpu-burn 110m 1Mi\n"))
    assert env.check("check_c7_limits_demo")[0] == "PASS"


def np_calls(env):
    return [a for a in env.kubectl() if "np-denied" in " ".join(a)]


NP_DELETE = ["delete", "namespace", "np-denied", "--ignore-not-found", "--timeout=120s"]
NP_CONTROL = ["-n", "np-denied", "exec", "client", "--", "curl", "-sS", "-m", "5", "-o", "/dev/null", "-w",
              "%{http_code}", "http://echo.nettest.svc.cluster.local:8080/"]
NP_HELLO = ["-n", "np-denied", "exec", "client", "--", "curl", "-sS", "-m", "5", "http://hello.hello.svc.cluster.local"]


def test_c8_checks_c2_then_the_control_then_is_denied_from_another_namespace_and_deletes_it(env):
    result, text = env.check("check_c8_netpol")
    assert result == "PASS", text
    calls = env.kubectl()
    allowed = calls.index(["-n", "nettest", "exec", NODES[CP]["probe"], "--", "curl", "-fsS", "-m", "5",
                           "http://hello.hello.svc.cluster.local"])
    np = np_calls(env)
    assert calls.index(np[0]) > allowed
    assert np[0] == NP_DELETE and np[-1] == NP_DELETE
    assert np[1][:2] == ["apply", "-f"] and np[1][2].endswith("/k8s/verify/np-denied.yaml")
    assert np[2] == ["-n", "np-denied", "wait", "pod/client", "--for=condition=Ready", "--timeout=120s"]
    # Ruling R45: the same client first reaches echo.nettest (HTTP 200), then hello is refused (exit 7)
    assert np[3:5] == [NP_CONTROL, NP_HELLO]
    assert "positive control: the np-denied client gets HTTP 200 from echo.nettest" in text
    assert "(curl exit 7)" in text
    assert "$ kc_within 60 -n np-denied exec client -- curl -sS -m 5 http://hello.hello.svc.cluster.local" in text


@pytest.mark.parametrize("rc, ok", [(7, True), (28, True), (0, False), (1, False), (6, False), (124, False)])
def test_c8_needs_exit_code_7_or_28_and_still_deletes_the_namespace(env, rc, ok):
    # 7: kube-router rejects (measured live), 28: a policy that drops; 0 got through, 6 DNS failed, 1 kubectl
    # failed, 124 the exec's outer bound ran out: none of those shows the NetworkPolicy at work
    env.first(kc("-n np-denied exec client", "hello.hello", rc=rc))
    result, text = env.check("check_c8_netpol")
    assert result == ("PASS" if ok else "FAIL"), text
    assert np_calls(env)[-1] == NP_DELETE


@pytest.mark.parametrize("control", [dict(rc=7, stdout="000"), dict(rc=6, stdout="000"), dict(stdout="403"),
                                     dict(rc=1, stdout="")])
def test_c8_fails_when_the_positive_control_fails_even_if_hello_times_out(env, control):
    # a client with no working network or DNS must not count as "denied"
    env.first(kc("-n np-denied exec client", "echo.nettest", **control),
              kc("-n np-denied exec client", "hello.hello", rc=28))
    result, text = env.check("check_c8_netpol")
    assert result == "FAIL", text
    assert "positive control failed" in text
    assert np_calls(env)[-1] == NP_DELETE


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_c8_deletes_the_namespace_when_interrupted(env, sig):
    env.first(kc("-n np-denied wait pod/client", sleep=30))
    assert env.interrupt("check_c8_netpol", "pod/client", sig) == -sig
    assert np_calls(env)[-1] == ["delete", "namespace", "np-denied", "--ignore-not-found", "--wait=false", RT]


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_cleanup_is_shown_to_the_operator_and_recorded(env, sig):
    # Ctrl-C also ends the evidence filter: the cleanup's progress goes to the operator's terminal as well
    env.first(kc("-n np-denied wait pod/client", sleep=30))
    assert env.interrupt("check_c8_netpol", "pod/client", sig) == -sig
    err, text = env.interrupted
    for out in (err, text):
        assert f"interrupted ({sig.name}); cleaning up (at most 120 s): kc delete namespace np-denied" in out
        assert "cleanup done" in out and "may remain" not in out
    assert "T: cleanup done" in err


def test_failed_interrupt_cleanup_says_what_may_remain_and_how_to_remove_it(env):
    env.first(on(CP, "tcpdump -ni ens4", rc=124, stderr=LISTENING, sleep=30),
              on(CP, "sudo rm -f /tmp/b2.pcap", rc=255, stderr="ERROR: (gcloud.compute.ssh) IAP tunnel failed\n"))
    assert env.interrupt("check_b2_wire", "X-Marker", signal.SIGINT) == -signal.SIGINT
    err, text = env.interrupted
    for out in (err, text):
        assert "IAP tunnel failed" in out                           # the cleanup's own output is shown
        assert "cleanup FAILED (exit 255)" in out
        assert "may remain: /tmp/b2.pcap (the packet capture) on k3s-cp" in out
        assert "bash -c 'source scripts/lib/checks.sh && vr_env && cp_ssh " in out
    assert "cleanup done" not in err


def test_c7_interrupt_names_the_pods_that_may_remain(env):
    env.first(kc("-n limits-demo wait pod/cpu-burn", sleep=30),
              kc("-n limits-demo delete pod oom cpu-burn", "--wait=false", rc=1, stderr="Unable to connect\n"))
    assert env.interrupt("check_c7_limits_demo", "pod/cpu-burn", signal.SIGTERM) == -signal.SIGTERM
    err, _ = env.interrupted
    assert "may remain: the Pods oom and cpu-burn in namespace limits-demo" in err
    assert "source scripts/lib/checks.sh && vr_env && kc -n limits-demo delete pod oom cpu-burn" in err


def removal_command(err):
    """The command a failed interrupt cleanup tells the operator to run."""
    line = next(line for line in err.splitlines() if "to remove it" in line)
    return line.split("to remove it, run from the repository root: ", 1)[1]


@pytest.mark.parametrize("compat", [None, "42"])
def test_the_removal_command_runs_the_cleanup_in_a_child_shell(env, compat):
    # F3-3: pasted into the operator's interactive bash, a die (exit) must end only a child shell, and .env's
    # variables and the harness's functions must not stay in the operator's shell.
    # F4-3: B2's cleanup holds single quotes and is quoted twice; bash <= 4.2 (BASH_COMPAT=42, like macOS's
    # /bin/bash 3.2) must print the same runnable command
    env.first(on(CP, "tcpdump -ni ens4", rc=124, stderr=LISTENING, sleep=30),
              on(CP, "sudo rm -f /tmp/b2.pcap", rc=255, stderr="ERROR: (gcloud.compute.ssh) IAP tunnel failed\n"))
    extra = {"BASH_COMPAT": compat} if compat else {}
    assert env.interrupt("check_b2_wire", "X-Marker", signal.SIGINT, **extra) == -signal.SIGINT
    err, text = env.interrupted
    cmd = removal_command(err)
    assert cmd.startswith("bash -c 'source scripts/lib/checks.sh && vr_env && cp_ssh ")
    assert f"to remove it, run from the repository root: {cmd}" in text
    cleanup = env.remote(CP)[-1]
    env.scenario["answers"] = healthy()                      # IAP works again
    report = 'echo "rc=$? GCP_PROJECT=${GCP_PROJECT:-unset} vr_env=$(type -t vr_env || echo none)"'
    r = env.run(["bash", "-c", f"{cmd}; {report}"])
    assert r.stdout.splitlines()[-1] == "rc=0 GCP_PROJECT=unset vr_env=none", r.stderr
    assert env.remote(CP)[-1] == cleanup                     # the same cleanup as the interrupted check's
    env.write_env(env.repo.joinpath(".env").read_text(), mode=0o644)    # vr_env dies: exit 2
    r = env.run(["bash", "-c", f"{cmd}; {report}"])
    assert r.stdout.splitlines()[-1] == "rc=2 GCP_PROJECT=unset vr_env=none"
    assert "must be 600" in r.stderr


def test_b4_removal_command_keeps_the_exec_script_quoted(env):
    env.first(kc("exec", B4_STOP, run_with={"pkill": 3}), kc("exec", "iperf3 -c", sleep=30))
    assert env.interrupt("check_b4_iperf", "50M", signal.SIGINT) == -signal.SIGINT
    cmd = removal_command(env.interrupted[0])
    env.scenario["answers"] = healthy()
    env.first(kc("exec", B4_STOP, run_with={"pkill": 1}))
    r = env.run(["bash", "-c", cmd])
    assert r.returncode == 0, r.stderr
    assert env.kubectl()[-1] == PKILL_CP


def test_a_second_ctrl_c_does_not_stop_the_cleanup_or_hide_its_outcome(env):
    # F3-4: the check's shell, its report and the cleanup's commands ignore a further Ctrl-C
    env.first(kc("-n np-denied wait pod/client", sleep=30),
              kc("delete namespace np-denied", "--wait=false", sleep=2))
    assert env.interrupt("check_c8_netpol", "pod/client", signal.SIGINT, again="--wait=false") == -signal.SIGINT
    err, text = env.interrupted
    for out in (err, text):
        assert "cleanup done" in out and "may remain" not in out
    assert "T: cleanup done" in err


def test_a_second_ctrl_c_does_not_stop_a_bounded_cleanup_either(env):
    # F4-2: B4's cleanup runs under kc_within, whose timeout --foreground would pass a further Ctrl-C on to kubectl
    env.first(kc("exec", B4_STOP, seq=[{}, {"sleep": 2}]), kc("exec", "iperf3 -c", sleep=30))
    assert env.interrupt("check_b4_iperf", "50M", signal.SIGINT, again=B4_STOP) == -signal.SIGINT
    assert env.kubectl()[-1] == PKILL_CP
    err, text = env.interrupted
    for out in (err, text):
        assert "cleanup done" in out and "FAILED" not in out and "may remain" not in out


def test_a_sigterm_still_stops_a_bounded_cleanup_and_reports_it(env):
    env.first(kc("exec", B4_STOP, seq=[{}, {"sleep": 30}]), kc("exec", "iperf3 -c", sleep=30))
    assert env.interrupt("check_b4_iperf", "50M", signal.SIGTERM, again=B4_STOP) == -signal.SIGTERM
    err, text = env.interrupted
    for out in (err, text):
        assert "cleanup FAILED (exit 143); may remain: an iperf3 server on port 5201" in out


def test_a_sigterm_during_the_cleanup_still_reports_its_outcome(env):
    # a TERM (it is also how the 120 s bound stops the cleanup) stops the cleanup, but the operator is told
    env.first(kc("-n np-denied wait pod/client", sleep=30),
              kc("delete namespace np-denied", "--wait=false", sleep=30))
    assert env.interrupt("check_c8_netpol", "pod/client", signal.SIGTERM, again="--wait=false") == -signal.SIGTERM
    err, text = env.interrupted
    for out in (err, text):
        assert "cleanup FAILED (exit 143); may remain: the namespace np-denied (its client Pod)" in out
        assert "to remove it, run from the repository root: bash -c " in out


def test_c8_fails_without_touching_np_denied_when_c2_fails(env):
    env.first(kc("exec", "http://hello.hello.svc.cluster.local", rc=28, **{"not": ["np-denied"]}))
    result, text = env.check("check_c8_netpol")
    assert result == "FAIL"
    assert np_calls(env) == []


# ---- D placeholders and verify.sh registration --------------------------------------------------

EXPECTED = [("A", "A1", "check_a1_nodes"), ("A", "A2", "check_a2_describe_nodes"), ("A", "A3", "check_a3_system_pods"),
            ("A", "A4", "check_a4_flannel_annotations"), ("B", "B1", "check_b1_matrix"), ("B", "B2", "check_b2_wire"),
            ("B", "B3", "check_b3_wg"), ("B", "B4", "check_b4_iperf"), ("B", "B5", "check_b5_negative"),
            ("B", "B6", "check_b6_firewall_inventory"), ("C", "C1", "check_c1_rollout"),
            ("C", "C2", "check_c2_cross_node_curl"), ("C", "C3", "check_c3_port_forward"), ("C", "C4", "check_c4_qos"),
            ("C", "C5", "check_c5_cgroup"), ("C", "C6", "check_c6_top"), ("C", "C7", "check_c7_limits_demo"),
            ("C", "C8", "check_c8_netpol")] + [("D", f"D{i}", "check_d_pending") for i in range(1, 7)]


def test_verify_registers_a_b_c_and_d_placeholders_in_order():
    text = (ROOT / "scripts/verify.sh").read_text()
    regs = re.findall(r"^register_check (\w) (\w+) (?:'[^']*'|\"[^\"]*\") (\w+)$", text, re.M)
    assert regs == EXPECTED


def test_every_registered_function_is_defined_in_checks_sh():
    names = " ".join(sorted({f for _, _, f in EXPECTED}))
    proc = subprocess.run(["bash", "-c", f"source {ROOT}/scripts/lib/checks.sh; for f in {names}; do "
                           'declare -F "$f" >/dev/null || echo "missing $f"; done'], capture_output=True, text=True)
    assert proc.stdout == "" and proc.returncode == 0


def test_d_placeholder_skips(env):
    result, text = env.check("check_d_pending")
    assert result == "SKIP"
    assert "Task 21" in text


def test_make_verify_passes_a_b_c_and_skips_d_on_a_healthy_cluster(env):
    port = free_port()
    proc = env.run(["bash", str(env.repo / "scripts/verify.sh")], VR_NO_WAIT="1", VR_PF_PORT=str(port),
                   **{"BASH_FUNC_timeout%%": NOT_REACHABLE, "TZ": "America/Los_Angeles"})
    runs = list((env.repo / "verification").iterdir())
    summary = (runs[0] / "SUMMARY.md").read_text()
    assert proc.returncode == 0, summary + proc.stdout + proc.stderr
    results = re.findall(r"^\| \d+ \| (\w+) \| [^|]+ \| (\w+) \|$", summary, re.M)
    assert results == [(i, "SKIP" if i.startswith("D") else "PASS") for _, i, _ in EXPECTED]
