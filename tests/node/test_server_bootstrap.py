"""node/server-bootstrap.sh and node/common.sh.

Everything here runs on the operator workstation. server-bootstrap runs either in --render-only
mode or, for the host sequence, with K3S_BOOTSTRAP_TEST_ROOT pointing at a scratch tree; the
common.sh helpers that touch the system run only against stub binaries on PATH plus a fake
K3S_BOOTSTRAP_TEST_ROOT / SYSFS_NET. No test runs a real sysctl, apt-get, systemctl, ip, netplan,
nft, curl or k3s.
"""
import os, re, subprocess
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "node/common.sh"
SYSTEM_TOOLS = ["sysctl", "apt-get", "dpkg-query", "systemctl", "ip", "netplan", "nft", "curl", "k3s"]


def fake_sysfs(tmp, iface, mtu):
    d = tmp / "sysfs" / iface; d.mkdir(parents=True, exist_ok=True); (d / "mtu").write_text(f"{mtu}\n"); return tmp / "sysfs"


def run_server(tmp, *extra, mtu=1460, path=None):
    env = {**os.environ, "SYSFS_NET": str(fake_sysfs(tmp, "ens4", mtu))}
    if path:
        env["PATH"] = f"{path}:{env['PATH']}"
    return subprocess.run(["bash", str(ROOT / "node/server-bootstrap.sh"), "--render-only", str(tmp / "root"),
                           "--cp-ip", "34.82.10.10", "--cp-internal-ip", "10.10.0.2", "--iface", "ens4", *extra],
                          capture_output=True, text=True, env=env)


def rendered(tmp, rel):
    return (tmp / "root" / rel).read_text()


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


def common(script, bindir=None, **env):
    full_env = {**os.environ, **env}
    if bindir:
        full_env["PATH"] = f"{bindir}:{full_env['PATH']}"
    return subprocess.run(["bash", "-c", f"set -euo pipefail; source {COMMON}; {script}"],
                          capture_output=True, text=True, env=full_env)


# --- rendered files ------------------------------------------------------------------------

def test_server_config_matches_spec(tmp_path):
    r = run_server(tmp_path)
    assert r.returncode == 0, r.stderr
    cfg = yaml.safe_load(rendered(tmp_path, "etc/rancher/k3s/config.yaml"))
    assert cfg == {"node-external-ip": "34.82.10.10", "flannel-backend": "wireguard-native",
        "flannel-external-ip": True, "secrets-encryption": True, "write-kubeconfig-mode": "0600",
        "protect-kernel-defaults": True, "node-taint": ["CriticalAddonsOnly=true:NoExecute"],
        "disable": ["traefik", "servicelb", "metrics-server"],
        "kube-apiserver-arg": ["admission-control-config-file=/etc/rancher/k3s/psa.yaml",
            "audit-policy-file=/etc/rancher/k3s/audit-policy.yaml",
            "audit-log-path=/var/lib/rancher/k3s/server/logs/audit.log",
            "audit-log-maxage=30", "audit-log-maxbackup=10", "audit-log-maxsize=100"]}


def test_psa_defaults_restricted(tmp_path):
    assert run_server(tmp_path).returncode == 0
    doc = yaml.safe_load(rendered(tmp_path, "etc/rancher/k3s/psa.yaml"))
    assert (doc["apiVersion"], doc["kind"]) == ("apiserver.config.k8s.io/v1", "AdmissionConfiguration")
    [plugin] = [p for p in doc["plugins"] if p["name"] == "PodSecurity"]
    conf = plugin["configuration"]
    assert (conf["apiVersion"], conf["kind"]) == ("pod-security.admission.config.k8s.io/v1",
                                                  "PodSecurityConfiguration")
    assert conf["defaults"] == {"enforce": "restricted", "enforce-version": "latest",
                                "audit": "restricted", "audit-version": "latest",
                                "warn": "restricted", "warn-version": "latest"}
    assert conf["exemptions"]["namespaces"] == ["kube-system"]
    assert not conf["exemptions"].get("usernames") and not conf["exemptions"].get("runtimeClasses")


def test_audit_policy_metadata_only(tmp_path):
    assert run_server(tmp_path).returncode == 0
    doc = yaml.safe_load(rendered(tmp_path, "etc/rancher/k3s/audit-policy.yaml"))
    assert (doc["apiVersion"], doc["kind"]) == ("audit.k8s.io/v1", "Policy")
    assert doc["rules"] == [{"level": "Metadata"}]
    assert doc["omitStages"] == ["RequestReceived"]


def test_sysctl_values(tmp_path):
    assert run_server(tmp_path).returncode == 0
    lines = {l.strip() for l in rendered(tmp_path, "etc/sysctl.d/90-kubelet.conf").splitlines() if l.strip()}
    assert lines == {"vm.panic_on_oom=0", "vm.overcommit_memory=1", "kernel.panic=10", "kernel.panic_on_oops=1"}


def test_server_guard_rules(tmp_path):
    assert run_server(tmp_path).returncode == 0
    text = rendered(tmp_path, "etc/k3s-guard/k3s-guard.nft")
    assert "elements = { 10.10.0.2, 34.82.10.10 }" in text
    assert 'tcp dport { 6443, 10250 } iifname { "lo", "cni0" } accept' in text
    assert 'tcp dport { 6443, 10250 } ip saddr @k3s_peers counter accept' in text
    assert "tcp dport { 6443, 10250 } counter drop" in text and "@" not in text.replace("@k3s_peers", "")
    assert "type filter hook input priority -5; policy accept;" in text
    assert "type filter hook forward priority -5; policy accept;" in text
    assert ("ip saddr 10.42.0.0/16 ip daddr 169.254.169.254 meta l4proto { tcp, udp } th dport 53 accept"
            in text)
    assert "ip saddr 10.42.0.0/16 ip daddr 169.254.169.254 counter drop" in text
    # The file replaces the whole table atomically on every load.
    assert text.splitlines()[:3] == ["table inet k3s_guard", "delete table inet k3s_guard",
                                     "table inet k3s_guard {"]
    assert "0.0.0.0/0" not in text


def test_guard_peers_sorted_numerically(tmp_path):
    # 9.x sorts after 10.x as text but before it numerically.
    env = {**os.environ, "SYSFS_NET": str(fake_sysfs(tmp_path, "ens4", 1460))}
    r = subprocess.run(["bash", str(ROOT / "node/server-bootstrap.sh"), "--render-only", str(tmp_path / "root"),
                        "--cp-ip", "9.9.9.9", "--cp-internal-ip", "10.10.0.2", "--iface", "ens4"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "elements = { 9.9.9.9, 10.10.0.2 }" in rendered(tmp_path, "etc/k3s-guard/k3s-guard.nft")


def test_guard_unit_ordering(tmp_path):
    assert run_server(tmp_path).returncode == 0
    unit = rendered(tmp_path, "etc/systemd/system/k3s-guard.service").splitlines()
    for line in ["Before=k3s.service k3s-agent.service", "Type=oneshot", "RemainAfterExit=yes",
                 "ExecStart=/usr/sbin/nft -f /etc/k3s-guard/k3s-guard.nft",
                 "ExecStop=/usr/sbin/nft delete table inet k3s_guard", "WantedBy=multi-user.target"]:
        assert line in unit
    dropin = rendered(tmp_path, "etc/systemd/system/k3s.service.d/10-guard.conf").splitlines()
    unit_section = dropin[dropin.index("[Unit]") + 1:]
    assert "Requires=k3s-guard.service" in unit_section and "After=k3s-guard.service" in unit_section


def test_renders_exactly_the_expected_files(tmp_path):
    assert run_server(tmp_path).returncode == 0
    files = {str(p.relative_to(tmp_path / "root")) for p in (tmp_path / "root").rglob("*") if p.is_file()}
    assert files == {"etc/rancher/k3s/config.yaml", "etc/rancher/k3s/psa.yaml",
                     "etc/rancher/k3s/audit-policy.yaml", "etc/sysctl.d/90-kubelet.conf",
                     "etc/k3s-guard/k3s-guard.nft", "etc/systemd/system/k3s-guard.service",
                     "etc/systemd/system/k3s.service.d/10-guard.conf"}


# --- argument handling and preflight -------------------------------------------------------

def test_mtu_below_target_fails(tmp_path):
    r = run_server(tmp_path, mtu=1400)
    assert r.returncode == 3 and "MTU 1400 < UNDERLAY_MTU 1460" in r.stderr
    assert not (tmp_path / "root").exists()


def test_mtu_above_target_passes_preflight(tmp_path):
    assert run_server(tmp_path, mtu=1500).returncode == 0


def test_underlay_mtu_flag_is_used(tmp_path):
    r = run_server(tmp_path, "--underlay-mtu", "1500", mtu=1460)
    assert r.returncode == 3 and "MTU 1460 < UNDERLAY_MTU 1500" in r.stderr


def test_missing_cp_ip_is_usage_error(tmp_path):
    env = {**os.environ, "SYSFS_NET": str(fake_sysfs(tmp_path, "ens4", 1460))}
    r = subprocess.run(["bash", str(ROOT / "node/server-bootstrap.sh"), "--render-only", str(tmp_path / "root"),
                        "--cp-internal-ip", "10.10.0.2", "--iface", "ens4"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "--cp-ip" in r.stderr


def test_unknown_flag_is_usage_error(tmp_path):
    assert run_server(tmp_path, "--bogus").returncode == 2


def test_flag_without_value_is_usage_error(tmp_path):
    assert run_server(tmp_path, "--k3s-version").returncode == 2


@pytest.mark.parametrize("bad_ip", [
    "34.82.10.300", "034.82.10.10", "34.82.010.10", "0.0.0.0", "0.1.2.3", "127.0.0.1",
    "224.0.0.1", "255.255.255.255", "34.82.10", "34.82.10.10.1", " 34.82.10.10"])
@pytest.mark.parametrize("flag", ["--cp-ip", "--cp-internal-ip"])
def test_invalid_ip_is_usage_error(tmp_path, flag, bad_ip):
    # "034" would reach nft as octal 28 and be rejected by k3s's net.ParseIP at startup.
    args = {"--cp-ip": "34.82.10.10", "--cp-internal-ip": "10.10.0.2", flag: bad_ip}
    env = {**os.environ, "SYSFS_NET": str(fake_sysfs(tmp_path, "ens4", 1460))}
    r = subprocess.run(["bash", str(ROOT / "node/server-bootstrap.sh"), "--render-only", str(tmp_path / "root"),
                        *[x for kv in args.items() for x in kv], "--iface", "ens4"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and not (tmp_path / "root").exists()


@pytest.mark.parametrize("ip", ["1.0.0.0", "10.0.0.0", "100.64.0.1", "223.255.255.254"])
def test_valid_ipv4_accepts_unicast(ip):
    assert common(f"valid_ipv4 {ip}").returncode == 0


def test_render_twice_is_identical(tmp_path):
    assert run_server(tmp_path).returncode == 0
    first = {p: p.read_bytes() for p in (tmp_path / "root").rglob("*") if p.is_file()}
    second_run = run_server(tmp_path)
    assert second_run.returncode == 0
    assert {p: p.read_bytes() for p in (tmp_path / "root").rglob("*") if p.is_file()} == first


def test_render_only_touches_nothing(tmp_path):
    bindir, log = make_stubs(tmp_path)
    r = run_server(tmp_path, path=bindir)
    assert r.returncode == 0, r.stderr
    assert calls(log) == []


# --- common.sh -----------------------------------------------------------------------------

def test_install_url_is_tag_pinned():
    out = subprocess.run(["bash", "-c", f"source {ROOT}/node/common.sh; k3s_install_url v1.36.5+k3s1"],
                         capture_output=True, text=True).stdout.strip()
    assert out == "https://raw.githubusercontent.com/k3s-io/k3s/v1.36.5%2Bk3s1/install.sh"


def test_render_template_substitutes_verbatim(tmp_path):
    (tmp_path / "t").write_text("a=@FOO@ b=@BAR@ @FOO@\n\n")
    r = common(f'FOO="x&y\\\\1" BAR=2; render_template {tmp_path}/t {tmp_path}/out/o FOO BAR')
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out/o").read_text() == "a=x&y\\1 b=2 x&y\\1\n\n"


def test_render_template_unresolved_placeholder_exits_4(tmp_path):
    (tmp_path / "t").write_text("a=@FOO@ b=@MISSING_VAR@\n")
    r = common(f"FOO=1; render_template {tmp_path}/t {tmp_path}/o FOO")
    assert r.returncode == 4 and "MISSING_VAR" in r.stderr


def test_default_iface_parses_route(tmp_path):
    bindir, log = make_stubs(tmp_path, {
        "ip": "echo '1.1.1.1 via 10.10.0.1 dev ens4 src 10.10.0.2 uid 0 \\    cache '"})
    r = common("default_iface", bindir)
    assert (r.returncode, r.stdout.strip()) == (0, "ens4")
    assert calls(log) == ["ip -o route get 1.1.1.1"]


def networkd(tmp, name, body):
    d = tmp / "sys/run/systemd/network"; d.mkdir(parents=True, exist_ok=True); (d / name).write_text(body)


def test_set_mtu_persists_with_netplan_set_not_apply(tmp_path):
    networkd(tmp_path, "10-netplan-ens5.network", "[Match]\nName=ens5\n")
    networkd(tmp_path, "10-netplan-ens4.network", "[Match]\nName=ens4\n\n[Network]\nDHCP=ipv4\n")
    bindir, log = make_stubs(tmp_path)
    r = common("set_mtu ens4 1460", bindir, K3S_BOOTSTRAP_TEST_ROOT=str(tmp_path / "sys"))
    assert r.returncode == 0, r.stderr
    assert calls(log) == ["ip link set dev ens4 mtu 1460", "netplan set ethernets.ens4.mtu=1460"]


def test_set_mtu_matches_netplan_id_by_mac(tmp_path):
    sysfs = fake_sysfs(tmp_path, "eno1", 1500)
    (sysfs / "eno1/address").write_text("02:00:00:aa:bb:cc\n")
    networkd(tmp_path, "10-netplan-lan0.network", "[Match]\nPermanentMACAddress=02:00:00:aa:bb:cc\n")
    bindir, log = make_stubs(tmp_path)
    r = common("set_mtu eno1 1460", bindir, K3S_BOOTSTRAP_TEST_ROOT=str(tmp_path / "sys"), SYSFS_NET=str(sysfs))
    assert r.returncode == 0, r.stderr
    assert calls(log) == ["ip link set dev eno1 mtu 1460", "netplan set ethernets.lan0.mtu=1460"]


def test_set_mtu_without_netplan_id_fails(tmp_path):
    networkd(tmp_path, "10-netplan-ens5.network", "[Match]\nName=ens5\n")
    bindir, log = make_stubs(tmp_path)
    r = common("set_mtu ens4 1460", bindir, K3S_BOOTSTRAP_TEST_ROOT=str(tmp_path / "sys"))
    assert r.returncode != 0 and "netplan" in r.stderr
    assert not [c for c in calls(log) if c.startswith("netplan")]


def test_install_packages_installs_without_upgrading(tmp_path):
    bindir, log = make_stubs(tmp_path, {"apt-get": f'echo "  DEBIAN_FRONTEND=$DEBIAN_FRONTEND" >> "{tmp_path}/calls.log"'})
    r = common("install_packages", bindir)
    assert r.returncode == 0, r.stderr
    assert calls(log) == ["apt-get update -q", "  DEBIAN_FRONTEND=noninteractive",
                          "apt-get install -y --no-install-recommends nftables wireguard-tools",
                          "  DEBIAN_FRONTEND=noninteractive"]


GUARD_FILES = ["etc/k3s-guard/k3s-guard.nft", "etc/systemd/system/k3s-guard.service",
               "etc/systemd/system/k3s.service.d/10-guard.conf"]


def guard_stage(tmp, nft_text):
    stage = tmp / "stage"
    for rel in GUARD_FILES:
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        (stage / rel).write_text(nft_text if rel.endswith(".nft") else f"# {rel}\n")
    return stage


def systemctl_stub(tmp):
    # is-active succeeds once the stub has seen "enable --now" (state file), like the real unit.
    # "reload k3s-guard" fails while the file fail-reload exists; "restart k3s" while fail-restart does.
    state = tmp / "guard-active"
    return {"systemctl": f'case "$*" in "enable --now k3s-guard") touch "{state}";; '
                         f'"is-active --quiet k3s-guard") [[ -e "{state}" ]];; '
                         f'"reload k3s-guard") [[ ! -e "{tmp}/fail-reload" ]];; '
                         f'"restart k3s") [[ ! -e "{tmp}/fail-restart" ]];; esac'}


def test_install_guard_first_run_enables(tmp_path):
    bindir, log = make_stubs(tmp_path, systemctl_stub(tmp_path))
    stage = guard_stage(tmp_path, "table inet k3s_guard\n")
    r = common(f"install_guard {stage}", bindir, K3S_BOOTSTRAP_TEST_ROOT=str(tmp_path / "sys"))
    assert r.returncode == 0, r.stderr
    assert calls(log) == [f"nft -c -f {stage}/etc/k3s-guard/k3s-guard.nft",
                          "systemctl mask nftables.service", "systemctl daemon-reload",
                          "systemctl is-active --quiet k3s-guard", "systemctl enable --now k3s-guard"]
    for rel in GUARD_FILES:
        assert (tmp_path / "sys" / rel).read_text() == (stage / rel).read_text()
    assert not (tmp_path / "sys/run/k3s-guard.reload-pending").exists()


def test_install_guard_inactive_unit_is_started_even_if_unchanged(tmp_path):
    # An inactive unit has no table (ExecStop deleted it), so starting it loses no k3s_peers.
    bindir, log = make_stubs(tmp_path, systemctl_stub(tmp_path))
    stage = guard_stage(tmp_path, "table inet k3s_guard\n")
    env = {"K3S_BOOTSTRAP_TEST_ROOT": str(tmp_path / "sys")}
    assert common(f"install_guard {stage}", bindir, **env).returncode == 0
    (tmp_path / "guard-active").unlink()
    log.write_text("")
    r = common(f"install_guard {stage}", bindir, **env)
    assert r.returncode == 0, r.stderr
    assert calls(log)[-1] == "systemctl enable --now k3s-guard"
    assert not [c for c in calls(log) if c.startswith("nft")]


def test_install_guard_rejects_bad_rules_before_installing(tmp_path):
    bindir, log = make_stubs(tmp_path, {**systemctl_stub(tmp_path),
                                        "nft": '[[ "$(cat "$3")" != bad ]]'})
    env = {"K3S_BOOTSTRAP_TEST_ROOT": str(tmp_path / "sys")}
    assert common(f"install_guard {guard_stage(tmp_path, 'good')}", bindir, **env).returncode == 0
    log.write_text("")
    r = common(f"install_guard {guard_stage(tmp_path, 'bad')}", bindir, **env)
    assert r.returncode != 0 and "nft -c" in r.stderr
    assert (tmp_path / "sys/etc/k3s-guard/k3s-guard.nft").read_text() == "good"
    assert calls(log) == [f"nft -c -f {tmp_path}/stage/etc/k3s-guard/k3s-guard.nft"]


def test_install_guard_refuses_destination_as_stage(tmp_path):
    # Rendering straight into the destination would make the "rules changed" check always false.
    sysroot = tmp_path / "sys"
    guard_stage(tmp_path, "x")
    (tmp_path / "stage").rename(sysroot)
    bindir, log = make_stubs(tmp_path, systemctl_stub(tmp_path))
    r = common(f"install_guard {sysroot}/.", bindir, K3S_BOOTSTRAP_TEST_ROOT=str(sysroot))
    assert r.returncode != 0 and "install destination" in r.stderr
    assert calls(log) == []


def test_install_guard_failed_reload_is_retried_on_rerun(tmp_path):
    bindir, log = make_stubs(tmp_path, systemctl_stub(tmp_path))
    env = {"K3S_BOOTSTRAP_TEST_ROOT": str(tmp_path / "sys")}
    assert common(f"install_guard {guard_stage(tmp_path, 'v1')}", bindir, **env).returncode == 0
    (tmp_path / "fail-reload").touch()
    assert common(f"install_guard {guard_stage(tmp_path, 'v2')}", bindir, **env).returncode != 0
    assert (tmp_path / "sys/etc/k3s-guard/k3s-guard.nft").read_text() == "v2"
    (tmp_path / "fail-reload").unlink()
    log.write_text("")
    r = common(f"install_guard {guard_stage(tmp_path, 'v2')}", bindir, **env)
    assert r.returncode == 0, r.stderr
    assert calls(log)[-1] == "systemctl reload k3s-guard"
    assert not (tmp_path / "sys/run/k3s-guard.reload-pending").exists()


def test_install_guard_unchanged_does_not_reload(tmp_path):
    # Reloading would reset k3s_peers to the template's two CP addresses and lock out joined nodes.
    bindir, log = make_stubs(tmp_path, systemctl_stub(tmp_path))
    stage = guard_stage(tmp_path, "table inet k3s_guard\n")
    env = {"K3S_BOOTSTRAP_TEST_ROOT": str(tmp_path / "sys")}
    assert common(f"install_guard {stage}", bindir, **env).returncode == 0
    log.write_text("")
    r = common(f"install_guard {stage}", bindir, **env)
    assert r.returncode == 0, r.stderr
    assert calls(log) == ["systemctl mask nftables.service", "systemctl daemon-reload",
                          "systemctl is-active --quiet k3s-guard"]


def test_install_guard_changed_rules_reload(tmp_path):
    bindir, log = make_stubs(tmp_path, systemctl_stub(tmp_path))
    env = {"K3S_BOOTSTRAP_TEST_ROOT": str(tmp_path / "sys")}
    stage = guard_stage(tmp_path, "v1\n")
    assert common(f"install_guard {stage}", bindir, **env).returncode == 0
    log.write_text("")
    stage = guard_stage(tmp_path, "v2\n")
    r = common(f"install_guard {stage}", bindir, **env)
    assert r.returncode == 0, r.stderr
    assert calls(log)[-1] == "systemctl reload k3s-guard"
    assert (tmp_path / "sys/etc/k3s-guard/k3s-guard.nft").read_text() == "v2\n"


K3S_INSTALL_ENV = ('echo "echo install VERSION=\\$INSTALL_K3S_VERSION EXEC=\\$INSTALL_K3S_EXEC '
                   'SKIP=\\${INSTALL_K3S_SKIP_START-unset} >> {log}"')


def test_install_k3s_skips_matching_version(tmp_path):
    bindir, log = make_stubs(tmp_path, {"k3s": 'echo "k3s version v1.36.5+k3s1 (abcdef12)"; echo "go version go1.25"'})
    r = common("install_k3s server v1.36.5+k3s1 false", bindir)
    assert r.returncode == 0, r.stderr
    assert calls(log) == ["k3s --version"]


def test_install_k3s_runs_pinned_installer(tmp_path):
    curl = K3S_INSTALL_ENV.replace("{log}", str(tmp_path / "calls.log"))
    bindir, log = make_stubs(tmp_path, {"k3s": 'echo "k3s version v1.35.0+k3s1 (abcdef12)"', "curl": curl})
    r = common("install_k3s agent v1.36.5+k3s1 true", bindir)
    assert r.returncode == 0, r.stderr
    assert calls(log)[1:] == [
        "curl -fsSL https://raw.githubusercontent.com/k3s-io/k3s/v1.36.5%2Bk3s1/install.sh",
        "install VERSION=v1.36.5+k3s1 EXEC=agent SKIP=true"]
    r = common("install_k3s server v1.36.5+k3s1 false", bindir)
    assert r.returncode == 0, r.stderr
    assert calls(log)[-1] == "install VERSION=v1.36.5+k3s1 EXEC=server SKIP=unset"


def test_render_template_read_failure_dies(tmp_path):
    # A directory passes the -r check but cannot be read; that must not render an empty file.
    r = common(f"render_template {tmp_path} {tmp_path}/out/o")
    assert r.returncode == 1 and "cannot read" in r.stderr
    assert not (tmp_path / "out/o").exists()


def test_guard_unit_restores_persisted_peers(tmp_path):
    # A reboot or reload reloads the table seeded with the CP addresses only; the unit re-adds the
    # list sync-peers persisted in peers.nft, ignoring a missing/broken file (k3s still starts).
    assert run_server(tmp_path).returncode == 0
    unit = rendered(tmp_path, "etc/systemd/system/k3s-guard.service").splitlines()
    assert ("ExecStartPost=-/bin/sh -c 'if [ -s /etc/k3s-guard/peers.nft ]; then "
            "exec /usr/sbin/nft -f /etc/k3s-guard/peers.nft; fi'") in unit
    assert ("ExecReload=/bin/sh -c 'cat /etc/k3s-guard/k3s-guard.nft /etc/k3s-guard/peers.nft 2>/dev/null "
            "| /usr/sbin/nft -f - || exec /usr/sbin/nft -f /etc/k3s-guard/k3s-guard.nft'") in unit


# --- host sequence (stubbed binaries, scratch K3S_BOOTSTRAP_TEST_ROOT) ----------------------

CURRENT = "v1.36.5+k3s1"


def host_stubs(tmp):
    """Stubs for a whole host run: k3s reports the version in k3s-version (absent = not installed),
    install.sh (curl's output, piped to sh) records itself and installs CURRENT, apt-get fails while
    fail-apt exists."""
    return make_stubs(tmp, {
        **systemctl_stub(tmp),
        "k3s": f'if [[ $1 == --version && -e "{tmp}/k3s-version" ]]; then '
               f'echo "k3s version $(cat "{tmp}/k3s-version") (abcdef12)"; fi',
        "curl": f'echo "echo {CURRENT} > {tmp}/k3s-version; echo install EXEC=\\$INSTALL_K3S_EXEC >> {tmp}/calls.log"',
        "apt-get": f'[[ ! -e "{tmp}/fail-apt" ]]',
    })


def run_host(tmp, bindir):
    sysroot = tmp / "sys"
    networkd(tmp, "10-netplan-ens4.network", "[Match]\nName=ens4\n")
    env = {**os.environ, "SYSFS_NET": str(fake_sysfs(tmp, "ens4", 1460)),
           "K3S_BOOTSTRAP_TEST_ROOT": str(sysroot), "PATH": f"{bindir}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(ROOT / "node/server-bootstrap.sh"), "--cp-ip", "34.82.10.10",
                           "--cp-internal-ip", "10.10.0.2", "--iface", "ens4"],
                          capture_output=True, text=True, env=env)


def host_calls(log):
    # The staging dir is a mktemp path; normalise it.
    return [re.sub(r"-f \S+/etc/k3s-guard/", "-f STAGE/etc/k3s-guard/", c) for c in calls(log)]


PRE_GUARD = ["sysctl --system", "ip link set dev ens4 mtu 1460", "netplan set ethernets.ens4.mtu=1460",
             "apt-get update -q", "apt-get install -y --no-install-recommends nftables wireguard-tools"]
READYZ = "k3s kubectl --request-timeout=5s get --raw /readyz"


def test_host_first_run_order(tmp_path):
    bindir, log = host_stubs(tmp_path)
    r = run_host(tmp_path, bindir)
    assert r.returncode == 0, r.stderr
    assert host_calls(log) == [
        "k3s --version", *PRE_GUARD,
        "nft -c -f STAGE/etc/k3s-guard/k3s-guard.nft", "systemctl mask nftables.service",
        "systemctl daemon-reload", "systemctl is-active --quiet k3s-guard", "systemctl enable --now k3s-guard",
        "k3s --version", "curl -fsSL https://raw.githubusercontent.com/k3s-io/k3s/v1.36.5%2Bk3s1/install.sh",
        "install EXEC=server", READYZ]
    sysroot = tmp_path / "sys"
    for rel in ["etc/rancher/k3s/config.yaml", "etc/rancher/k3s/psa.yaml", "etc/rancher/k3s/audit-policy.yaml",
                "etc/sysctl.d/90-kubelet.conf", "etc/k3s-guard/k3s-guard.nft"]:
        assert (sysroot / rel).is_file(), rel
    assert (sysroot / "var/lib/rancher/k3s/server/logs").is_dir()
    assert not (sysroot / "var/lib/rancher/k3s/bootstrap-restart-pending").exists()


def test_host_rerun_unchanged_does_nothing_disruptive(tmp_path):
    bindir, log = host_stubs(tmp_path)
    assert run_host(tmp_path, bindir).returncode == 0
    log.write_text("")
    r = run_host(tmp_path, bindir)
    assert r.returncode == 0, r.stderr
    assert host_calls(log) == [
        "k3s --version", *PRE_GUARD, "systemctl mask nftables.service", "systemctl daemon-reload",
        "systemctl is-active --quiet k3s-guard", "k3s --version", READYZ]


def test_host_config_change_restarts_current_k3s(tmp_path):
    bindir, log = host_stubs(tmp_path)
    assert run_host(tmp_path, bindir).returncode == 0
    (tmp_path / "sys/etc/rancher/k3s/psa.yaml").write_text("# operator edit, to be replaced\n")
    log.write_text("")
    r = run_host(tmp_path, bindir)
    assert r.returncode == 0, r.stderr
    assert host_calls(log)[-3:] == ["k3s --version", "systemctl restart k3s", READYZ]


def test_host_failed_run_still_restarts_k3s_on_rerun(tmp_path):
    # The run that installs the new config dies in apt-get; the re-run finds the files identical
    # but must still restart k3s, or the apiserver keeps the old PSA/audit config.
    bindir, log = host_stubs(tmp_path)
    assert run_host(tmp_path, bindir).returncode == 0
    (tmp_path / "sys/etc/rancher/k3s/audit-policy.yaml").write_text("# old policy\n")
    (tmp_path / "fail-apt").touch()
    assert run_host(tmp_path, bindir).returncode != 0
    assert "systemctl restart k3s" not in calls(log)
    (tmp_path / "fail-apt").unlink()
    log.write_text("")
    r = run_host(tmp_path, bindir)
    assert r.returncode == 0, r.stderr
    assert "systemctl restart k3s" in host_calls(log)
    log.write_text("")
    assert run_host(tmp_path, bindir).returncode == 0
    assert "systemctl restart k3s" not in host_calls(log)


def test_host_failed_restart_is_retried(tmp_path):
    bindir, log = host_stubs(tmp_path)
    assert run_host(tmp_path, bindir).returncode == 0
    (tmp_path / "sys/etc/rancher/k3s/config.yaml").write_text("# old config\n")
    (tmp_path / "fail-restart").touch()
    assert run_host(tmp_path, bindir).returncode != 0
    (tmp_path / "fail-restart").unlink()
    log.write_text("")
    assert run_host(tmp_path, bindir).returncode == 0
    assert "systemctl restart k3s" in host_calls(log)


def test_host_new_install_does_not_restart_twice(tmp_path):
    # install.sh (re)starts k3s itself when it installs or upgrades.
    bindir, log = host_stubs(tmp_path)
    (tmp_path / "k3s-version").write_text("v1.35.0+k3s1")
    assert run_host(tmp_path, bindir).returncode == 0
    assert "install EXEC=server" in calls(log) and "systemctl restart k3s" not in calls(log)
    assert not (tmp_path / "sys/var/lib/rancher/k3s/bootstrap-restart-pending").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="the root check only bites for non-root users")
def test_host_mode_requires_root_without_test_root(tmp_path):
    bindir, log = make_stubs(tmp_path)
    env = {**os.environ, "SYSFS_NET": str(fake_sysfs(tmp_path, "ens4", 1460)),
           "PATH": f"{bindir}:{os.environ['PATH']}"}
    env.pop("K3S_BOOTSTRAP_TEST_ROOT", None)
    r = subprocess.run(["bash", str(ROOT / "node/server-bootstrap.sh"), "--cp-ip", "34.82.10.10",
                        "--cp-internal-ip", "10.10.0.2", "--iface", "ens4"], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "run as root" in r.stderr
    assert calls(log) == []
