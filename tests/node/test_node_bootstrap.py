"""node/node-bootstrap.sh (every k3s agent: GCP CPU workers and Lambda GPU nodes) and gpu_audit.

Everything here runs on the operator workstation: node-bootstrap runs in --render-only mode or, for
the host sequence, with K3S_BOOTSTRAP_TEST_ROOT pointing at a scratch tree and stub binaries first
on PATH (the Task 9 seams in node/common.sh). No test runs a real sysctl, apt-get, systemctl, ip,
netplan, nft, curl, k3s, iptables or nvidia-smi.
"""
import ipaddress, os, re, shutil, signal, subprocess, time
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "node/node-bootstrap.sh"
COMMON = ROOT / "node/common.sh"
ANY_IPV4 = str(ipaddress.IPv4Network((0, 0)))
SYSTEM_TOOLS = ["sysctl", "apt-get", "systemctl", "ip", "netplan", "nft", "curl", "k3s",
                "iptables", "dpkg", "lsb_release", "uname"]
CURRENT = "v1.36.5+k3s1"

BASE = {"server": "https://34.82.10.10:6443", "token-file": "/etc/rancher/k3s/token",
        "node-name": "k3s-cpu-1", "node-external-ip": "34.29.20.20", "protect-kernel-defaults": True}
GPU = {**BASE, "node-name": "k3s-gpu-a6000-1", "node-external-ip": "129.80.30.30",
       "node-label": ["nvidia.com/gpu.present=true", "gpu-type=a6000"],
       "node-taint": ["nvidia.com/gpu=present:NoSchedule"]}

AGENT_DROPIN = "etc/systemd/system/k3s-agent.service.d/10-guard.conf"
RENDERED = {"etc/rancher/k3s/config.yaml", "etc/sysctl.d/90-kubelet.conf", "etc/k3s-guard/k3s-guard.nft",
            "etc/systemd/system/k3s-guard.service", AGENT_DROPIN}

AGENT_GUARD = """\
table inet k3s_guard
delete table inet k3s_guard
table inet k3s_guard {
  chain input {
    type filter hook input priority -5; policy accept;
    tcp dport 10250 iifname { "lo", "cni0" } accept
    tcp dport 10250 ip saddr 34.82.10.10 counter accept
    tcp dport 10250 counter drop
  }
  chain forward {
    type filter hook forward priority -5; policy accept;
    ip saddr 10.42.0.0/16 ip daddr 169.254.169.254 meta l4proto { tcp, udp } th dport 53 accept
    ip saddr 10.42.0.0/16 ip daddr 169.254.169.254 counter drop
  }
}
"""


def fake_sysfs(tmp, iface, mtu):
    d = tmp / "sysfs" / iface; d.mkdir(parents=True, exist_ok=True); (d / "mtu").write_text(f"{mtu}\n"); return tmp / "sysfs"


def make_stubs(tmp, bodies=None, names=SYSTEM_TOOLS):
    """Stub binaries that append "<name> <args>" to calls.log, then run an optional bash body."""
    bodies = bodies or {}
    bindir, log = tmp / "bin", tmp / "calls.log"
    bindir.mkdir(exist_ok=True)
    log.touch()
    for name in set(names) | set(bodies):
        stub = bindir / name
        stub.write_text(f'#!/bin/bash\nprintf "%s\\n" "{name} $*" >> "{log}"\n{bodies.get(name, "")}\n')
        stub.chmod(0o755)
    return bindir, log


def calls(log):
    return log.read_text().splitlines()


def run_agent(tmp, *flags, node="k3s-cpu-1", ext="34.29.20.20", mtu=1460, path=None, args=None):
    env = {**os.environ, "SYSFS_NET": str(fake_sysfs(tmp, "ens4", mtu))}
    if path:
        env["PATH"] = f"{path}:{env['PATH']}"
    if args is None:
        args = ["--server-ip", "34.82.10.10", "--node-name", node, "--external-ip", ext, "--iface", "ens4"]
    return subprocess.run(["bash", str(SCRIPT), "--render-only", str(tmp / "root"), *args, *flags],
                          capture_output=True, text=True, env=env)


def rendered(tmp, rel):
    return (tmp / "root" / rel).read_text()


def load_cfg(tmp):
    return yaml.safe_load(rendered(tmp, "etc/rancher/k3s/config.yaml"))


def guard_text(tmp):
    return rendered(tmp, "etc/k3s-guard/k3s-guard.nft")


# --- rendered files ------------------------------------------------------------------------

def test_cpu_agent_config_exact(tmp_path):
    r = run_agent(tmp_path, "--no-gpu")
    assert r.returncode == 0, r.stderr
    assert load_cfg(tmp_path) == BASE


def test_gpu_agent_config_exact(tmp_path):
    r = run_agent(tmp_path, "--gpu", "--gpu-type", "a6000", node="k3s-gpu-a6000-1", ext="129.80.30.30")
    assert r.returncode == 0, r.stderr
    assert load_cfg(tmp_path) == GPU


def test_no_flannel_keys_in_agent_config(tmp_path):
    # flannel settings on an agent make the node NotReady (k3s#12942); they live on the server only.
    for flags in [("--no-gpu",), ("--gpu", "--gpu-type", "a6000")]:
        assert run_agent(tmp_path, *flags).returncode == 0
        assert not [k for k in load_cfg(tmp_path) if k.startswith("flannel")]


def test_agent_guard_allows_only_cp(tmp_path):
    assert run_agent(tmp_path, "--no-gpu").returncode == 0
    text = guard_text(tmp_path)
    assert "tcp dport 10250 ip saddr 34.82.10.10 counter accept" in text and "6443" not in text


def test_agent_guard_is_the_fixed_template(tmp_path):
    assert run_agent(tmp_path, "--no-gpu").returncode == 0
    assert guard_text(tmp_path) == AGENT_GUARD
    assert ANY_IPV4 not in guard_text(tmp_path)


def test_agent_dropin_orders_k3s_agent_after_the_guard(tmp_path):
    assert run_agent(tmp_path, "--no-gpu").returncode == 0
    dropin = rendered(tmp_path, AGENT_DROPIN).splitlines()
    unit_section = dropin[dropin.index("[Unit]") + 1:]
    assert "Requires=k3s-guard.service" in unit_section and "After=k3s-guard.service" in unit_section
    unit = rendered(tmp_path, "etc/systemd/system/k3s-guard.service").splitlines()
    assert "Before=k3s.service k3s-agent.service" in unit


def test_sysctl_values(tmp_path):
    assert run_agent(tmp_path, "--no-gpu").returncode == 0
    lines = {l.strip() for l in rendered(tmp_path, "etc/sysctl.d/90-kubelet.conf").splitlines()
             if l.strip() and not l.lstrip().startswith("#")}
    assert lines == {"vm.panic_on_oom=0", "vm.overcommit_memory=1", "kernel.panic=10", "kernel.panic_on_oops=1",
                     "net.ipv4.ip_forward=1", "net.ipv4.conf.all.forwarding=1"}


def test_renders_exactly_the_expected_files(tmp_path):
    assert run_agent(tmp_path, "--gpu", "--gpu-type", "a6000").returncode == 0
    files = {str(p.relative_to(tmp_path / "root")) for p in (tmp_path / "root").rglob("*") if p.is_file()}
    assert files == RENDERED


def test_render_twice_is_identical(tmp_path):
    assert run_agent(tmp_path, "--no-gpu").returncode == 0
    first = {p: p.read_bytes() for p in (tmp_path / "root").rglob("*") if p.is_file()}
    assert run_agent(tmp_path, "--no-gpu").returncode == 0
    assert {p: p.read_bytes() for p in (tmp_path / "root").rglob("*") if p.is_file()} == first


def test_render_only_touches_nothing(tmp_path):
    bindir, log = make_stubs(tmp_path)
    r = run_agent(tmp_path, "--gpu", "--gpu-type", "a6000", path=bindir)
    assert r.returncode == 0, r.stderr
    assert calls(log) == []


def test_never_uses_k3s_token_env():
    assert "K3S_TOKEN" not in SCRIPT.read_text()


# --- GPU detection and argument handling ---------------------------------------------------

def test_gpu_requires_type(tmp_path):
    r = run_agent(tmp_path, "--gpu")
    assert r.returncode == 2 and "--gpu-type" in r.stderr
    assert not (tmp_path / "root").exists()


def test_gpu_autodetected_from_nvidia_smi_needs_a_type(tmp_path):
    bindir, log = make_stubs(tmp_path, names=["nvidia-smi"])
    r = run_agent(tmp_path, path=bindir)
    assert r.returncode == 2 and "nvidia-smi" in r.stderr and "--gpu-type" in r.stderr
    assert calls(log) == []


def test_gpu_autodetected_with_type(tmp_path):
    bindir, _ = make_stubs(tmp_path, names=["nvidia-smi"])
    r = run_agent(tmp_path, "--gpu-type", "a6000", path=bindir, node="k3s-gpu-a6000-1", ext="129.80.30.30")
    assert r.returncode == 0, r.stderr
    assert load_cfg(tmp_path) == GPU


def test_no_nvidia_smi_means_cpu(tmp_path):
    # PATH without nvidia-smi: only the system directories.
    env = {"PATH": "/usr/bin:/bin", "SYSFS_NET": str(fake_sysfs(tmp_path, "ens4", 1460))}
    r = subprocess.run(["bash", str(SCRIPT), "--render-only", str(tmp_path / "root"), "--server-ip", "34.82.10.10",
                        "--node-name", "k3s-cpu-1", "--external-ip", "34.29.20.20", "--iface", "ens4"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert load_cfg(tmp_path) == BASE


def test_gpu_type_alone_without_nvidia_smi_is_usage_error(tmp_path):
    # Detection is nvidia-smi only (brief): --gpu-type without --gpu does not make a GPU node.
    env = {"PATH": "/usr/bin:/bin", "SYSFS_NET": str(fake_sysfs(tmp_path, "ens4", 1460))}
    r = subprocess.run(["bash", str(SCRIPT), "--render-only", str(tmp_path / "root"), "--server-ip", "34.82.10.10",
                        "--node-name", "k3s-gpu-a6000-1", "--external-ip", "129.80.30.30", "--iface", "ens4",
                        "--gpu-type", "a6000"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "nvidia-smi" in r.stderr and "--gpu" in r.stderr
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize("name", ["null", "true", "yes", "123", "0x1f"])
def test_node_name_that_looks_like_a_yaml_scalar_stays_a_string(tmp_path, name):
    # Valid RFC 1123 labels that bare YAML would load as null, a bool or a number.
    r = run_agent(tmp_path, "--no-gpu", node=name)
    assert r.returncode == 0, r.stderr
    assert load_cfg(tmp_path) == {**BASE, "node-name": name}


def test_default_iface_is_used_when_iface_omitted(tmp_path):
    bindir, _ = make_stubs(tmp_path, {"ip": "echo '1.1.1.1 via 10.10.0.1 dev ens4 src 10.10.0.2 uid 0'"},
                           names=["ip"])
    r = run_agent(tmp_path, "--no-gpu", path=bindir,
                  args=["--server-ip", "34.82.10.10", "--node-name", "k3s-cpu-1", "--external-ip", "34.29.20.20"])
    assert r.returncode == 0, r.stderr
    assert load_cfg(tmp_path) == BASE


def test_unsafe_default_iface_name_is_refused(tmp_path):
    # Linux allows "$", "(" and ")" in interface names; the auto-detected name gets the --iface check
    # before it reaches mtu_preflight, set_mtu or gpu_audit.
    bindir, _ = make_stubs(tmp_path, {"ip": "echo '1.1.1.1 via 10.10.0.1 dev e$(id) src 10.10.0.2'"}, names=["ip"])
    r = run_agent(tmp_path, "--no-gpu", path=bindir,
                  args=["--server-ip", "34.82.10.10", "--node-name", "k3s-cpu-1", "--external-ip", "34.29.20.20"])
    assert r.returncode == 1 and "unexpected default-route interface name: e$(id)" in r.stderr
    assert not (tmp_path / "root").exists()


def test_no_gpu_overrides_detection(tmp_path):
    bindir, _ = make_stubs(tmp_path, names=["nvidia-smi"])
    assert run_agent(tmp_path, "--no-gpu", path=bindir).returncode == 0
    assert load_cfg(tmp_path) == BASE


@pytest.mark.parametrize("flags", [("--gpu", "--gpu-type", "a6000", "--no-gpu"),
                                   ("--no-gpu", "--gpu"),
                                   ("--no-gpu", "--gpu-type", "a6000")])
def test_contradictory_gpu_flags_are_usage_errors(tmp_path, flags):
    r = run_agent(tmp_path, *flags)
    assert r.returncode == 2 and not (tmp_path / "root").exists()


@pytest.mark.parametrize("flag, bad", [
    ("--server-ip", "34.82.10.300"), ("--server-ip", "034.82.10.10"), ("--server-ip", "0.0.0.0"),
    ("--external-ip", "127.0.0.1"), ("--external-ip", "34.29.20"),
    ("--node-name", "K3s-CPU-1"), ("--node-name", "k3s_cpu_1"), ("--node-name", "-k3s"), ("--node-name", "a" * 64),
    ("--node-name", "k3s-cpu-1\nnode-label: x"),
    ("--gpu-type", "a6000\nnode-taint: []"), ("--gpu-type", "a 6000"), ("--gpu-type", "-a6000"),
    ("--gpu-type", "a" * 64), ("--underlay-mtu", "15OO"), ("--k3s-version", "latest"), ("--iface", "ens4;id")])
def test_invalid_values_are_usage_errors(tmp_path, flag, bad):
    args = {"--server-ip": "34.82.10.10", "--node-name": "k3s-gpu-a6000-1", "--external-ip": "129.80.30.30",
            "--iface": "ens4", "--gpu-type": "a6000", flag: bad}
    r = run_agent(tmp_path, "--gpu", args=[x for kv in args.items() for x in kv])
    assert r.returncode == 2, r.stderr
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize("missing", ["--server-ip", "--node-name", "--external-ip"])
def test_missing_required_flag_is_usage_error(tmp_path, missing):
    args = {"--server-ip": "34.82.10.10", "--node-name": "k3s-cpu-1", "--external-ip": "34.29.20.20",
            "--iface": "ens4"}
    del args[missing]
    r = run_agent(tmp_path, "--no-gpu", args=[x for kv in args.items() for x in kv])
    assert r.returncode == 2 and missing in r.stderr


def test_unknown_flag_and_missing_value_are_usage_errors(tmp_path):
    assert run_agent(tmp_path, "--bogus").returncode == 2
    assert run_agent(tmp_path, "--no-gpu", "--gpu-type").returncode == 2


def test_agent_mtu_below_target_fails(tmp_path):
    r = run_agent(tmp_path, "--no-gpu", mtu=1400)
    assert r.returncode == 3 and "MTU 1400 < UNDERLAY_MTU 1460" in r.stderr
    assert not (tmp_path / "root").exists()


def test_agent_underlay_mtu_flag_is_used(tmp_path):
    r = run_agent(tmp_path, "--no-gpu", "--underlay-mtu", "1500", mtu=1460)
    assert r.returncode == 3 and "MTU 1460 < UNDERLAY_MTU 1500" in r.stderr
    assert run_agent(tmp_path, "--no-gpu", mtu=1500).returncode == 0


# --- gpu_audit (common.sh) -----------------------------------------------------------------

AUDIT_PROBES = [
    "date -Is", "nvidia-smi", "nvidia-smi --query-gpu=name,driver_version --format=csv", "lsb_release -a",
    "uname -r", "command -v nvidia-container-runtime", "grep -i mode {root}/etc/nvidia-container-runtime/config.toml",
    "dpkg -l | grep -E 'nvidia-container|libnvidia-container'", "iptables -V", "iptables -S FORWARD | head -3",
    "ip -4 addr show eno1", "cat {sysfs}/eno1/mtu", "systemctl is-enabled lambda-jupyter docker"]


def audit_stubs(tmp):
    return make_stubs(tmp, {
        "nvidia-smi": 'if [[ $1 == --query-gpu* ]]; then echo "name, driver_version"; echo "NVIDIA RTX A6000, 570.1"; '
                      'else echo "NVIDIA-SMI 570.1"; fi',
        "dpkg": 'printf "ii  libnvidia-container1 1.17\\nii  nvidia-container-toolkit 1.17\\nii  vim 9\\n"',
        "iptables": 'if [[ $1 == -S ]]; then printf -- "-P FORWARD DROP\\n-A FORWARD -j DOCKER-USER\\n'
                    '-A FORWARD -j DOCKER-ISOLATION\\n-A FORWARD -j ACCEPT\\n"; else echo "iptables v1.8.10"; fi',
        "systemctl": 'echo enabled; echo enabled',
    }, names=["nvidia-smi", "lsb_release", "uname", "dpkg", "iptables", "ip", "systemctl"])


def run_audit(tmp, bindir, label, iface="eno1", extra_env=None):
    sysroot = tmp / "sys"
    (sysroot / "etc/nvidia-container-runtime").mkdir(parents=True, exist_ok=True)
    (sysroot / "etc/nvidia-container-runtime/config.toml").write_text('#debug = "x"\nmode = "auto"\n')
    env = {**os.environ, "K3S_BOOTSTRAP_TEST_ROOT": str(sysroot), "SYSFS_NET": str(fake_sysfs(tmp, "eno1", 1500)),
           "PATH": f"{bindir}:{os.environ['PATH']}", **(extra_env or {})}
    return subprocess.run(["bash", "-c", f'set -euo pipefail; source {COMMON}; '
                                         f'gpu_audit {sysroot}/var/log/gpu-audit.txt "$1" {label}', "audit", iface],
                          capture_output=True, text=True, env=env)


def test_gpu_audit_records_every_probe_in_order(tmp_path):
    bindir, log = audit_stubs(tmp_path)
    r = run_audit(tmp_path, bindir, "before")
    assert r.returncode == 0, r.stderr
    text = (tmp_path / "sys/var/log/gpu-audit.txt").read_text()
    probes = [l[2:] for l in text.splitlines() if l.startswith("$ ")]
    assert probes == [p.format(root=tmp_path / "sys", sysfs=tmp_path / "sysfs") for p in AUDIT_PROBES]
    assert text.splitlines()[0] == "===== gpu-audit before"
    assert re.search(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$", text, re.M)
    for evidence in ["NVIDIA RTX A6000, 570.1", 'mode = "auto"', "nvidia-container-toolkit 1.17",
                     "libnvidia-container1 1.17", "-P FORWARD DROP", "-A FORWARD -j DOCKER-ISOLATION", "1500"]:
        assert evidence in text, evidence
    assert "vim 9" not in text and "-A FORWARD -j ACCEPT" not in text   # the grep and head -3 applied
    # Read-only probes only: the FORWARD policy is recorded, never changed.
    assert calls(log) == ["nvidia-smi ", "nvidia-smi --query-gpu=name,driver_version --format=csv",
                          "lsb_release -a", "uname -r", "dpkg -l", "iptables -V", "iptables -S FORWARD",
                          "ip -4 addr show eno1", "systemctl is-enabled lambda-jupyter docker"]


def test_gpu_audit_appends_and_records_failures(tmp_path):
    # Missing tools/files are evidence too: the probe's exit status is written and the audit goes on.
    bindir, _ = make_stubs(tmp_path, {"nvidia-smi": "exit 9"}, names=["nvidia-smi", "lsb_release", "uname", "dpkg",
                                                                       "iptables", "ip", "systemctl"])
    assert run_audit(tmp_path, bindir, "before").returncode == 0
    r = run_audit(tmp_path, bindir, "after")
    assert r.returncode == 0, r.stderr
    text = (tmp_path / "sys/var/log/gpu-audit.txt").read_text()
    headers = [l for l in text.splitlines() if l.startswith("===== ")]
    assert headers == ["===== gpu-audit before", "===== gpu-audit after"]
    assert text.count("[exit 9]") == 4    # two nvidia-smi probes per audit
    assert "[exit 1]" in text             # command -v nvidia-container-runtime: not on PATH


def test_gpu_audit_quotes_the_interface_name(tmp_path):
    # gpu_audit is safe whatever its caller passes: the name is a literal argument, never code.
    marker = tmp_path / "PWNED"
    bindir, log = audit_stubs(tmp_path)
    r = run_audit(tmp_path, bindir, "before", iface=f"e$(touch {marker})")
    assert r.returncode == 0, r.stderr
    assert not marker.exists()
    assert f"ip -4 addr show e$(touch {marker})" in calls(log)


def test_gpu_audit_bounds_a_hung_probe(tmp_path):
    # nvidia-smi can hang on a GPU in a bad state: the probe is cut off, recorded, and the audit goes on.
    bindir, _ = make_stubs(tmp_path, {"nvidia-smi": "sleep 30"},
                           names=["nvidia-smi", "lsb_release", "uname", "dpkg", "iptables", "ip", "systemctl"])
    start = time.monotonic()
    r = run_audit(tmp_path, bindir, "before", extra_env={"GPU_AUDIT_TIMEOUT": "1"})
    assert r.returncode == 0, r.stderr
    assert time.monotonic() - start < 15
    text = (tmp_path / "sys/var/log/gpu-audit.txt").read_text()
    assert text.count("[exit 124]") == 2   # both nvidia-smi probes
    assert "$ systemctl is-enabled lambda-jupyter docker" in text


def test_gpu_audit_abandons_a_probe_that_cannot_be_reaped(tmp_path):
    # An nvidia-smi in uninterruptible sleep survives SIGKILL, so timeout would wait for it forever.
    # Simulated by a timeout that never exits on TERM: the audit gives up on it at its own deadline
    # (limit + 2 * kill-after), records that, and goes on.
    real_timeout = shutil.which("timeout")
    bindir, _ = make_stubs(tmp_path, {
        "timeout": f'if [[ "$*" == *nvidia-smi* ]]; then trap "" TERM INT; exec sleep 300; fi\n'
                   f'exec {real_timeout} "$@"'},
        names=["nvidia-smi", "lsb_release", "uname", "dpkg", "iptables", "ip", "systemctl"])
    start = time.monotonic()
    r = run_audit(tmp_path, bindir, "before", extra_env={"GPU_AUDIT_TIMEOUT": "1"})
    assert r.returncode == 0, r.stderr
    assert time.monotonic() - start < 15
    text = (tmp_path / "sys/var/log/gpu-audit.txt").read_text()
    assert text.count("[exit 124 abandoned: still running after 3 s]") == 2   # both nvidia-smi probes
    assert "$ systemctl is-enabled lambda-jupyter docker" in text


def test_gpu_audit_stops_on_ctrl_c(tmp_path):
    # Ctrl-C reaches the whole foreground process group: the probe and the bootstrap stop, nothing
    # after the audit runs (a probe in its own process group would let the root bootstrap carry on).
    probe_pid = tmp_path / "probe.pid"
    bindir, _ = make_stubs(tmp_path, {"nvidia-smi": f'echo $$ > "{probe_pid}"; exec sleep 30'},
                           names=["nvidia-smi", "lsb_release", "uname", "dpkg", "iptables", "ip", "systemctl"])
    sysroot = tmp_path / "sys"
    env = {**os.environ, "K3S_BOOTSTRAP_TEST_ROOT": str(sysroot), "SYSFS_NET": str(fake_sysfs(tmp_path, "eno1", 1500)),
           "PATH": f"{bindir}:{os.environ['PATH']}", "GPU_AUDIT_TIMEOUT": "20"}
    p = subprocess.Popen(["bash", "-c", f'set -euo pipefail; source {COMMON}; '
                                        f'gpu_audit {sysroot}/var/log/gpu-audit.txt eno1 before; echo AFTER-AUDIT'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not probe_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        os.killpg(p.pid, signal.SIGINT)
        out, _ = p.communicate(timeout=5)
        assert p.returncode != 0
        assert "AFTER-AUDIT" not in out
        time.sleep(0.5)
        with pytest.raises(ProcessLookupError):   # the probe stopped too
            os.kill(int(probe_pid.read_text()), 0)
        with pytest.raises(ProcessLookupError):   # nothing of the run is left
            os.killpg(p.pid, 0)
    finally:
        if probe_pid.exists():
            try:
                os.kill(int(probe_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()


# --- host sequence (stubbed binaries, scratch K3S_BOOTSTRAP_TEST_ROOT) ----------------------

def host_stubs(tmp, gpu=False, units=("lambda-jupyter", "docker.socket", "docker.service")):
    """Stubs for a whole host run.

    systemctl: k3s-guard is active once enabled; k3s-agent is active while agent-active exists;
    `systemctl cat UNIT` succeeds for the units in `units` (they exist on the host).
    k3s: reports the version in k3s-version (absent = not installed).
    curl: its output is install.sh, piped to sh; it records the install env and installs CURRENT.
    """
    state = tmp / "guard-active"
    unit_cases = " ".join(f'"cat {u}") ;;' for u in units)
    bodies = {
        "systemctl": f'case "$*" in "enable --now k3s-guard") touch "{state}";; '
                     f'"is-active --quiet k3s-guard") [[ -e "{state}" ]];; '
                     f'"is-active --quiet k3s-agent") [[ -e "{tmp}/agent-active" ]];; '
                     f'{unit_cases} "cat "*) exit 1;; esac',
        "k3s": f'if [[ $1 == --version && -e "{tmp}/k3s-version" ]]; then '
               f'echo "k3s version $(cat "{tmp}/k3s-version") (abcdef12)"; fi',
        "curl": f'echo "echo {CURRENT} > {tmp}/k3s-version; '
                f'echo install EXEC=\\$INSTALL_K3S_EXEC VERSION=\\$INSTALL_K3S_VERSION '
                f'SKIP_START=\\$INSTALL_K3S_SKIP_START TOKEN=\\${{K3S_TOKEN-unset}} >> {tmp}/calls.log"',
    }
    names = SYSTEM_TOOLS + (["nvidia-smi"] if gpu else [])
    return make_stubs(tmp, bodies, names=names)


def run_host(tmp, bindir, *flags, node="k3s-cpu-1", ext="34.29.20.20", extra_env=None):
    sysroot = tmp / "sys"
    nd = sysroot / "run/systemd/network"; nd.mkdir(parents=True, exist_ok=True)
    (nd / "10-netplan-ens4.network").write_text("[Match]\nName=ens4\n")
    env = {**os.environ, "SYSFS_NET": str(fake_sysfs(tmp, "ens4", 1500)),
           "K3S_BOOTSTRAP_TEST_ROOT": str(sysroot), "PATH": f"{bindir}:/usr/bin:/bin"}
    env.pop("K3S_TOKEN", None)
    env.update(extra_env or {})
    return subprocess.run(["bash", str(SCRIPT), "--server-ip", "34.82.10.10", "--node-name", node,
                           "--external-ip", ext, "--iface", "ens4", *flags],
                          capture_output=True, text=True, env=env)


def host_calls(log):
    # The staging dir is a mktemp path; normalise it. The GPU audit's probes are checked separately.
    probes = ("nvidia-smi", "lsb_release", "uname", "dpkg", "iptables", "ip -4 addr", "systemctl is-enabled")
    return [re.sub(r"-f \S+/etc/k3s-guard/", "-f STAGE/etc/k3s-guard/", c) for c in calls(log)
            if not c.startswith(probes)]


PRE_GUARD = ["sysctl --system", "ip link set dev ens4 mtu 1460", "netplan set ethernets.ens4.mtu=1460",
             "apt-get update -q", "apt-get install -y --no-install-recommends nftables wireguard-tools"]
GUARD_FIRST = ["nft -c -f STAGE/etc/k3s-guard/k3s-guard.nft", "systemctl mask nftables.service",
               "systemctl daemon-reload", "systemctl is-active --quiet k3s-guard", "systemctl enable --now k3s-guard"]
INSTALL = ["k3s --version", "curl -fsSL https://raw.githubusercontent.com/k3s-io/k3s/v1.36.5%2Bk3s1/install.sh",
           f"install EXEC=agent VERSION={CURRENT} SKIP_START=true TOKEN=unset"]


def test_host_cpu_first_run_order(tmp_path):
    bindir, log = host_stubs(tmp_path)
    r = run_host(tmp_path, bindir, "--no-gpu")
    assert r.returncode == 0, r.stderr
    assert host_calls(log) == [*PRE_GUARD, *GUARD_FIRST, *INSTALL, "systemctl is-active --quiet k3s-agent"]
    sysroot = tmp_path / "sys"
    assert yaml.safe_load((sysroot / "etc/rancher/k3s/config.yaml").read_text()) == BASE
    for rel in RENDERED:
        assert (sysroot / rel).is_file(), rel
    assert (sysroot / "etc/k3s-guard/k3s-guard.nft").read_text() == AGENT_GUARD
    assert not (sysroot / "var/log/gpu-audit.txt").exists()
    assert not (sysroot / "var/lib/rancher/k3s/bootstrap-restart-pending").exists()


def test_host_gpu_run_order(tmp_path):
    bindir, log = host_stubs(tmp_path, gpu=True)
    r = run_host(tmp_path, bindir, "--gpu", "--gpu-type", "a6000", node="k3s-gpu-a6000-1", ext="129.80.30.30")
    assert r.returncode == 0, r.stderr
    all_calls = calls(log)
    # The "before" audit runs first, so it records the host as Lambda delivered it (the NIC MTU
    # before set_mtu included); the "after" audit follows the jupyter/docker shutdown.
    assert all_calls[0] == "nvidia-smi "
    assert all_calls.index("systemctl disable --now docker.socket docker.service") \
        < max(i for i, c in enumerate(all_calls) if c == "systemctl is-enabled lambda-jupyter docker")
    assert host_calls(log) == [
        *PRE_GUARD, *GUARD_FIRST,
        "systemctl cat lambda-jupyter", "systemctl disable --now lambda-jupyter",
        "systemctl cat docker.socket", "systemctl cat docker.service",
        "systemctl disable --now docker.socket docker.service",
        *INSTALL, "systemctl is-active --quiet k3s-agent"]
    assert not [c for c in all_calls if c.startswith("iptables -P") or c.startswith("iptables -A")]
    text = (tmp_path / "sys/var/log/gpu-audit.txt").read_text()
    assert [l for l in text.splitlines() if l.startswith("===== ")] == \
        ["===== gpu-audit before", "===== gpu-audit after"]
    assert yaml.safe_load((tmp_path / "sys/etc/rancher/k3s/config.yaml").read_text()) == GPU


def test_host_gpu_missing_units_only_warn(tmp_path):
    bindir, log = host_stubs(tmp_path, gpu=True, units=("docker.service",))
    r = run_host(tmp_path, bindir, "--gpu", "--gpu-type", "a6000")
    assert r.returncode == 0, r.stderr
    assert "WARNING: lambda-jupyter not found" in r.stderr and "WARNING: docker.socket not found" in r.stderr
    disables = [c for c in calls(log) if c.startswith("systemctl disable")]
    assert disables == ["systemctl disable --now docker.service"]


def test_host_gpu_without_nvidia_smi_fails_before_changing_anything(tmp_path):
    bindir, log = host_stubs(tmp_path)
    r = run_host(tmp_path, bindir, "--gpu", "--gpu-type", "a6000")
    assert r.returncode == 1 and "nvidia-smi" in r.stderr
    assert calls(log) == []


def test_host_rerun_unchanged_does_nothing_disruptive(tmp_path):
    bindir, log = host_stubs(tmp_path)
    assert run_host(tmp_path, bindir, "--no-gpu").returncode == 0
    (tmp_path / "agent-active").touch()          # join-node.sh started k3s-agent meanwhile
    log.write_text("")
    r = run_host(tmp_path, bindir, "--no-gpu")
    assert r.returncode == 0, r.stderr
    assert host_calls(log) == [*PRE_GUARD, "systemctl mask nftables.service", "systemctl daemon-reload",
                               "systemctl is-active --quiet k3s-guard", "k3s --version", "k3s --version"]


def test_host_config_change_restarts_a_running_agent(tmp_path):
    bindir, log = host_stubs(tmp_path)
    assert run_host(tmp_path, bindir, "--no-gpu").returncode == 0
    (tmp_path / "agent-active").touch()
    log.write_text("")
    r = run_host(tmp_path, bindir, "--no-gpu", ext="34.29.20.21")
    assert r.returncode == 0, r.stderr
    assert host_calls(log)[-3:] == ["k3s --version", "systemctl is-active --quiet k3s-agent",
                                    "systemctl restart k3s-agent"]
    assert not (tmp_path / "sys/var/lib/rancher/k3s/bootstrap-restart-pending").exists()


def test_host_failed_run_still_restarts_the_agent_on_rerun(tmp_path):
    # The run that installs the new config dies before the restart; the re-run finds the config
    # identical but must still restart k3s-agent, or it keeps running with the old config.
    bindir, log = host_stubs(tmp_path)
    assert run_host(tmp_path, bindir, "--no-gpu").returncode == 0
    (tmp_path / "agent-active").touch()
    (bindir / "curl").write_text("#!/bin/bash\nexit 22\n")
    (tmp_path / "k3s-version").write_text("v1.35.0+k3s1")        # forces install.sh, which now fails
    assert run_host(tmp_path, bindir, "--no-gpu", ext="34.29.20.21").returncode != 0
    assert "systemctl restart k3s-agent" not in calls(log)
    (tmp_path / "k3s-version").write_text(CURRENT)
    log.write_text("")
    assert run_host(tmp_path, bindir, "--no-gpu", ext="34.29.20.21").returncode == 0
    assert "systemctl restart k3s-agent" in calls(log)


def test_host_version_only_upgrade_restarts_a_running_agent(tmp_path):
    # Same flags (config unchanged), older k3s installed: install.sh replaces the binary, and the
    # running k3s-agent must be restarted to run it.
    bindir, log = host_stubs(tmp_path)
    assert run_host(tmp_path, bindir, "--no-gpu").returncode == 0
    (tmp_path / "agent-active").touch()
    (tmp_path / "k3s-version").write_text("v1.35.0+k3s1")
    log.write_text("")
    r = run_host(tmp_path, bindir, "--no-gpu")
    assert r.returncode == 0, r.stderr
    assert host_calls(log)[-6:] == ["k3s --version", *INSTALL, "systemctl is-active --quiet k3s-agent",
                                    "systemctl restart k3s-agent"]
    assert not (tmp_path / "sys/var/lib/rancher/k3s/bootstrap-restart-pending").exists()


def test_host_inherited_join_credentials_never_reach_install_sh(tmp_path):
    # install.sh copies every K3S_* variable into k3s-agent.service.env (spec §5.1: the token only
    # reaches token-file), and K3S_URL/K3S_TOKEN would change its mode.
    bindir, log = host_stubs(tmp_path)
    seen = tmp_path / "install-env"
    (bindir / "curl").write_text(f'#!/bin/bash\nprintf "%s\\n" "curl $*" >> "{log}"\n'
                                 f'echo "env | grep ^K3S_ | sort > {seen}; echo {CURRENT} > {tmp_path}/k3s-version"\n')
    leak = {"K3S_TOKEN": "leak", "K3S_AGENT_TOKEN": "leak", "K3S_TOKEN_FILE": "/leak", "K3S_URL": "https://leak:6443"}
    r = run_host(tmp_path, bindir, "--no-gpu", extra_env=leak)
    assert r.returncode == 0, r.stderr
    assert seen.read_text().splitlines() == ["K3S_BOOTSTRAP_TEST_ROOT=" + str(tmp_path / "sys")]


def test_host_writes_no_token(tmp_path):
    # The join token arrives later, piped by join-node.sh into token-file.
    bindir, _ = host_stubs(tmp_path)
    assert run_host(tmp_path, bindir, "--no-gpu").returncode == 0
    assert not (tmp_path / "sys/etc/rancher/k3s/token").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="the root check only bites for non-root users")
def test_host_mode_requires_root_without_test_root(tmp_path):
    bindir, log = make_stubs(tmp_path)
    env = {**os.environ, "SYSFS_NET": str(fake_sysfs(tmp_path, "ens4", 1460)),
           "PATH": f"{bindir}:{os.environ['PATH']}"}
    env.pop("K3S_BOOTSTRAP_TEST_ROOT", None)
    r = subprocess.run(["bash", str(SCRIPT), "--server-ip", "34.82.10.10", "--node-name", "k3s-cpu-1",
                        "--external-ip", "34.29.20.20", "--iface", "ens4", "--no-gpu"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "run as root" in r.stderr
    assert calls(log) == []
