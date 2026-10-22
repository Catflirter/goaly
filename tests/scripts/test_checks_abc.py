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
import re
import shutil
import socket
import subprocess

import pytest

from test_ops_scripts import ROOT, Env, free_port

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


PCAP_OK = (f"IP 10.10.0.2.51820 > {NODES[CPU1]['ext']}.51820: UDP, length 148\n"
           f"IP {NODES[CPU1]['ext']}.51820 > 10.10.0.2.51820: UDP, length 92\n"
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
GPUCTL_RULES = ("global\n  tcp   22          198.51.100.200/32  operator ssh\n"
                "ruleset k3s-gpu-us-west-1 rs-1 us-west-1\n  udp   51820       192.0.2.10/32      k3s-cp wireguard\n"
                "ruleset k3s-gpu-us-east-1 rs-2 us-east-1\n  (no rules)\n")


def healthy(nodes=(CP, CPU1), mtu=1460):
    """Scripted answers for a cluster where every check A1-C8 passes."""
    def lines(app):  # _vr_pods: "NODE POD IP" for the Running Pods of app=APP
        return "".join(f"{n} {NODES[n][app]} {NODES[n][app + '_ip']}\n" for n in nodes)

    gpus = [n for n in nodes if n.startswith("k3s-gpu-")]
    r = [
        # the D checks' GPU node list (labelled by join-node.sh); first, so the plain node list below cannot answer it
        kc("get nodes", "-l node-restriction.kubernetes.io/accelerator=nvidia", "allocatable",
           stdout="".join(f"{n} 1\n" for n in gpus)),
        kc("get nodes", "-l node-restriction.kubernetes.io/accelerator=nvidia", stdout="".join(f"{n}\n" for n in gpus)),
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
        kc("-n np-denied exec client", rc=28, stderr="curl: (28) Connection timed out after 5001 milliseconds\n"),
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
        on(CP, "tcpdump -ni ens4 -U -w /tmp/b2.pcap", rc=124,
           stderr="tcpdump: listening on ens4, link-type EN10MB (Ethernet), snapshot length 262144 bytes\n"),
        on(CP, "sudo tcpdump -A -nn -r /tmp/b2.pcap", stdout="0\n", rc=1),
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
    assert cp[1] == "sudo timeout 25 tcpdump -ni ens4 -U -w /tmp/b2.pcap 'not port 22'"
    marker = re.fullmatch(r"sudo tcpdump -A -nn -r /tmp/b2\.pcap \| grep -c -F -- '(b2-marker-[0-9a-f]{16})'", cp[2])
    assert marker, cp[2]
    assert cp[3:] == ["sudo tcpdump -nn -q -t -r /tmp/b2.pcap", "sudo rm -f /tmp/b2.pcap"]
    curl = next(a for a in env.kubectl() if "X-Marker" in " ".join(a))
    assert curl[:5] == ["-n", "nettest", "exec", NODES[CPU1]["probe"], "--"]
    assert f'-H "X-Marker: {marker.group(1)}" http://{NODES[CP]["echo_ip"]}:8080/' in curl[-1]
    assert "seq 20" in curl[-1]
    assert "51820/udp" in text and "6443/tcp" in text and "10250/tcp" in text


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
    ("curl-failed", [kc("exec", "X-Marker", rc=1, stdout="12/20 requests answered\n")], "requests"),
    ("capture-failed", [on(CP, "tcpdump -ni ens4", rc=1, stderr="tcpdump: ens4: No such device exists\n")],
     "capture"),
])
def test_b2_fails_and_still_removes_the_pcap(env, name, answers, why):
    env.first(*answers)
    result, text = env.check("check_b2_wire")
    assert result == "FAIL", text
    assert why in text
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

def test_b4_measures_cpu1_to_cp_and_records_bits_per_second(env):
    result, text = env.check("check_b4_iperf")
    assert result == "PASS", text
    execs = [a for a in env.kubectl() if "exec" in a]
    assert ["-n", "nettest", "exec", NODES[CP]["probe"], "--", "timeout", "150", "iperf3", "-s", "-1"] in execs
    assert ["-n", "nettest", "exec", NODES[CPU1]["probe"], "--", "timeout", "120", "iperf3", "-c",
            NODES[CP]["probe_ip"], "-n", "50M", "-J"] in execs
    assert "bits_per_second=410000000" in text


def test_b4_also_measures_cpu1_to_each_gpu_node(env3):
    assert env3.check("check_b4_iperf")[0] == "PASS"
    clients = [a[a.index("-c") + 1] for a in env3.kubectl() if "iperf3" in a and "-c" in a]
    assert clients == [NODES[CP]["probe_ip"], NODES[GPU1]["probe_ip"]]


def test_b4_fails_when_the_client_fails(env):
    env.first(kc("exec", "iperf3 -c", rc=1, stderr="iperf3: error - unable to connect to server\n"))
    assert env.check("check_b4_iperf")[0] == "FAIL"


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


@pytest.mark.parametrize("rc", [0, 1, 126])
def test_b5_metadata_must_fail_inside_curl(env, rc):
    # 0: the metadata server answered; 1/126: kubectl or the probe failed, so curl's verdict is unknown.
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


def test_c7_accepts_a_reading_of_exactly_110m(env):
    env.first(kc("top pod cpu-burn", stdout="NAME CPU(cores) MEMORY(bytes)\ncpu-burn 110m 1Mi\n"))
    assert env.check("check_c7_limits_demo")[0] == "PASS"


def np_calls(env):
    return [a for a in env.kubectl() if "np-denied" in " ".join(a)]


def test_c8_checks_c2_then_times_out_from_another_namespace_and_deletes_it(env):
    result, text = env.check("check_c8_netpol")
    assert result == "PASS", text
    calls = env.kubectl()
    allowed = calls.index(["-n", "nettest", "exec", NODES[CP]["probe"], "--", "curl", "-fsS", "-m", "5",
                           "http://hello.hello.svc.cluster.local"])
    np = np_calls(env)
    assert calls.index(np[0]) > allowed
    delete = ["delete", "namespace", "np-denied", "--ignore-not-found", "--timeout=120s"]
    assert np[0] == delete and np[-1] == delete
    assert np[1][:2] == ["apply", "-f"] and np[1][2].endswith("/k8s/verify/np-denied.yaml")
    assert ["-n", "np-denied", "wait", "pod/client", "--for=condition=Ready", "--timeout=120s"] in np
    assert ["-n", "np-denied", "exec", "client", "--", "curl", "-sS", "-m", "5",
            "http://hello.hello.svc.cluster.local"] in np


@pytest.mark.parametrize("rc", [0, 7, 1, 6])
def test_c8_needs_exit_code_28_and_still_deletes_the_namespace(env, rc):
    env.first(kc("-n np-denied exec client", rc=rc))
    assert env.check("check_c8_netpol")[0] == "FAIL"
    assert np_calls(env)[-1] == ["delete", "namespace", "np-denied", "--ignore-not-found", "--timeout=120s"]


def test_c8_fails_without_touching_np_denied_when_c2_fails(env):
    env.first(kc("exec", "http://hello.hello.svc.cluster.local", rc=28, **{"not": ["np-denied"]}))
    result, text = env.check("check_c8_netpol")
    assert result == "FAIL"
    assert np_calls(env) == []


# ---- verify.sh registration (D1-D6: Task 21, tests/scripts/test_checks_d.py) ---------------------

EXPECTED = [("A", "A1", "check_a1_nodes"), ("A", "A2", "check_a2_describe_nodes"), ("A", "A3", "check_a3_system_pods"),
            ("A", "A4", "check_a4_flannel_annotations"), ("B", "B1", "check_b1_matrix"), ("B", "B2", "check_b2_wire"),
            ("B", "B3", "check_b3_wg"), ("B", "B4", "check_b4_iperf"), ("B", "B5", "check_b5_negative"),
            ("B", "B6", "check_b6_firewall_inventory"), ("C", "C1", "check_c1_rollout"),
            ("C", "C2", "check_c2_cross_node_curl"), ("C", "C3", "check_c3_port_forward"), ("C", "C4", "check_c4_qos"),
            ("C", "C5", "check_c5_cgroup"), ("C", "C6", "check_c6_top"), ("C", "C7", "check_c7_limits_demo"),
            ("C", "C8", "check_c8_netpol"), ("D", "D1", "check_d1_gpu_stack"),
            ("D", "D2", "check_d2_gpu_capacity"), ("D", "D3", "check_d3_device_plugin"), ("D", "D4", "check_d4_gpu_smoke"),
            ("D", "D5", "check_d5_vectoradd"), ("D", "D6", "check_d6_overflow")]


def test_verify_registers_a_b_c_and_d_in_order():
    text = (ROOT / "scripts/verify.sh").read_text()
    regs = re.findall(r"^register_check (\w) (\w+) (?:'[^']*'|\"[^\"]*\") (\w+)$", text, re.M)
    assert regs == EXPECTED


def test_every_registered_function_is_defined_in_checks_sh():
    names = " ".join(sorted({f for _, _, f in EXPECTED}))
    proc = subprocess.run(["bash", "-c", f"source {ROOT}/scripts/lib/checks.sh; for f in {names}; do "
                           'declare -F "$f" >/dev/null || echo "missing $f"; done'], capture_output=True, text=True)
    assert proc.stdout == "" and proc.returncode == 0


def test_make_verify_passes_a_b_c_and_skips_d_on_a_healthy_cluster(env):
    port = free_port()
    proc = env.run(["bash", str(env.repo / "scripts/verify.sh")], VR_NO_WAIT="1", VR_PF_PORT=str(port),
                   **{"BASH_FUNC_timeout%%": NOT_REACHABLE, "TZ": "America/Los_Angeles"})
    runs = list((env.repo / "verification").iterdir())
    summary = (runs[0] / "SUMMARY.md").read_text()
    assert proc.returncode == 0, summary + proc.stdout + proc.stderr
    results = re.findall(r"^\| \d+ \| (\w+) \| [^|]+ \| (\w+) \|$", summary, re.M)
    assert results == [(i, "SKIP" if i.startswith("D") else "PASS") for _, i, _ in EXPECTED]
