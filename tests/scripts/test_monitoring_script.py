"""Offline tests of scripts/monitoring.sh (stretch S0a: make monitoring, monitoring-down, grafana, prometheus).

kubectl is a stand-in (FAKE_KUBECTL below) that records every call with its stdin and keeps a little
cluster state in files: whether the Secret monitoring/grafana-admin exists and whether Grafana is
deployed. Its port-forward prints kubectl's "Forwarding from" line and sleeps until it is killed. openssl
and base64 are the real ones: the tests look at what the script generates.
"""
import base64
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "monitoring.sh"
MANIFESTS = ["prometheus.yaml", "kube-state-metrics.yaml", "node-exporter.yaml", "dcgm-exporter.yaml",
             "grafana.yaml", "network-policies.yaml"]

FAKE_KUBECTL = r'''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
state = Path(os.environ["FAKE_STATE"])
stdin = sys.stdin.read() if "-" in args and args[args.index("-") - 1] == "-f" else ""
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"argv": args, "stdin": stdin}) + "\n")
words = [a for a in args if not a.startswith("-")]
if "port-forward" in args:
    port = next(a for a in args if ":" in a and a.split(":")[0].isdigit()).split(":")[0]
    if (state / "port-forward-fails").exists():
        print("error: unable to forward port because pod is not running", file=sys.stderr)
        sys.exit(1)
    print(f"Forwarding from 127.0.0.1:{port} -> 3000", flush=True)
    while True:
        time.sleep(1)
if "get" in words and "secret" in words:
    if (state / "secret").exists():
        print("secret/grafana-admin"); sys.exit(0)
    print('Error from server (NotFound): secrets "grafana-admin" not found', file=sys.stderr); sys.exit(1)
if "get" in words and "deployment" in words:
    if (state / "grafana").exists():
        print("deployment.apps/grafana"); sys.exit(0)
    print('Error from server (NotFound): deployments.apps "grafana" not found', file=sys.stderr); sys.exit(1)
if "create" in words and "configmap" in words:
    print("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: grafana-dashboards\n"); sys.exit(0)
if "create" in words:
    if (state / "secret").exists():
        print('Error from server (AlreadyExists): secrets "grafana-admin" already exists', file=sys.stderr)
        sys.exit(1)
    (state / "secret").write_text(stdin); sys.exit(0)
if "apply" in words and any(a.endswith("grafana.yaml") for a in args):
    (state / "grafana").write_text("")
if "rollout" in words and "status" in words and (state / "rollout-fails").exists():
    print("error: timed out waiting for the condition", file=sys.stderr); sys.exit(1)
sys.exit(0)
'''


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Env:
    def __init__(self, tmp):
        self.tmp, self.home, self.bin, self.state = tmp, tmp / "home", tmp / "bin", tmp / "state"
        self.repo = tmp / "repo"
        for d in (self.home, self.bin, self.state, self.repo / "scripts/lib"):
            d.mkdir(parents=True)
        for rel in ("scripts/monitoring.sh", "scripts/lib/common.sh"):
            (self.repo / rel).write_bytes((ROOT / rel).read_bytes())
            (self.repo / rel).chmod(0o755)
        (self.repo / "k8s").mkdir()
        os.symlink(ROOT / "k8s/monitoring", self.repo / "k8s/monitoring")
        fake = tmp / "fake_kubectl.py"
        fake.write_text(FAKE_KUBECTL)
        (self.bin / "kubectl").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
        (self.bin / "kubectl").chmod(0o755)
        self.kubeconfig = self.home / ".kube/hybrid-gpu.yaml"
        self.kubeconfig.parent.mkdir()
        self.kubeconfig.write_text("apiVersion: v1\nkind: Config\n")
        self.log = tmp / "calls.jsonl"
        self.tty = tmp / "tty"  # what the script shows "on the terminal"
        self.grafana_port, self.prometheus_port = free_port(), free_port()

    def run(self, *args, stdin="", **extra):
        env = {k: v for k, v in os.environ.items() if k not in ("AUTO_APPROVE", "KUBECONFIG")}
        env.update(PATH=f"{self.bin}:{os.environ['PATH']}", HOME=str(self.home), FAKE_LOG=str(self.log),
                   FAKE_STATE=str(self.state), MONITORING_TTY=str(self.tty),
                   GRAFANA_PORT=str(self.grafana_port), PROMETHEUS_PORT=str(self.prometheus_port),
                   FORWARD_TIMEOUT_S="10", TMPDIR=str(self.tmp))
        env.update(extra)
        return subprocess.run([str(self.repo / "scripts/monitoring.sh"), *args], cwd=self.repo, env=env,
                              input=stdin, capture_output=True, text=True, timeout=60)

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def argvs(self):
        return [c["argv"] for c in self.calls()]

    def pid(self, name):
        path = self.repo / f".run/port-forward-{name}.pid"
        return int(path.read_text()) if path.exists() else None


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    for name in ("grafana", "prometheus"):
        pid = e.pid(name)
        if pid:
            for kill in (lambda: os.killpg(pid, signal.SIGKILL), lambda: os.kill(pid, signal.SIGKILL)):
                try:
                    kill()
                except (ProcessLookupError, PermissionError):
                    pass


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    proc = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return proc.returncode == 0 and not proc.stdout.strip().startswith("Z")


def secret_doc(env):
    return yaml.safe_load((env.state / "secret").read_text())


def decoded(env, key):
    return base64.b64decode(secret_doc(env)["data"][key]).decode()


def all_output(env, proc):
    """Everything the script left anywhere a log could catch: its stdout/stderr and every kubectl argv."""
    return proc.stdout + proc.stderr + json.dumps(env.argvs())


# ---- install ------------------------------------------------------------------------------------

def test_script_is_executable_bash():
    assert os.access(SCRIPT, os.X_OK)
    assert SCRIPT.read_text().startswith("#!/usr/bin/env bash\n")


def test_install_generates_the_admin_secret_and_shows_the_password_once_on_the_terminal(env):
    proc = env.run("install")
    assert proc.returncode == 0, proc.stderr
    doc = secret_doc(env)
    assert doc["kind"] == "Secret" and doc["metadata"]["name"] == "grafana-admin"
    assert doc["metadata"]["namespace"] == "monitoring"
    assert set(doc["data"]) == {"admin-user", "admin-password", "secret-key"}
    password = decoded(env, "admin-password")
    assert decoded(env, "admin-user") == "admin"
    assert re.fullmatch(r"[A-Za-z0-9]{32}", password)
    assert re.fullmatch(r"[0-9a-f]{64}", decoded(env, "secret-key"))
    # Shown on the terminal, never in the script's output (which make or a CI log may capture) or argv.
    assert env.tty.read_text().count(password) == 1
    assert password not in all_output(env, proc)
    # Created from stdin with `kubectl create`: apply would keep a copy in last-applied-configuration.
    creates = [c["argv"] for c in env.calls() if "create" in c["argv"] and "configmap" not in c["argv"]]
    assert len(creates) == 1
    i = creates[0].index("create")
    assert creates[0][i:i + 3] == ["create", "-f", "-"]
    assert not [c for c in env.calls() if "apply" in c["argv"] and "Secret" in c["stdin"]]


def test_install_never_reuses_a_password(env, tmp_path):
    assert env.run("install").returncode == 0
    first = decoded(env, "admin-password")
    (env.state / "secret").unlink()
    assert env.run("install").returncode == 0
    assert decoded(env, "admin-password") != first


def test_install_without_a_terminal_prints_how_to_read_the_password_instead(env):
    proc = env.run("install", MONITORING_TTY=str(env.tmp / "no-such-dir/tty"))
    assert proc.returncode == 0, proc.stderr
    password = decoded(env, "admin-password")
    assert password not in all_output(env, proc)
    assert "jsonpath='{.data.admin-password}'" in proc.stderr


def test_install_keeps_an_existing_secret(env):
    (env.state / "secret").write_text("existing")
    proc = env.run("install")
    assert proc.returncode == 0, proc.stderr
    assert (env.state / "secret").read_text() == "existing"
    assert not [a for a in env.argvs() if "create" in a and "configmap" not in a]
    assert not env.tty.exists()
    assert "unchanged" in proc.stderr


def test_install_applies_everything_in_order_and_waits_for_the_rollout(env):
    proc = env.run("install")
    assert proc.returncode == 0, proc.stderr
    argvs = env.argvs()
    for a in argvs:
        assert a[:2] == ["--kubeconfig", str(env.kubeconfig)]
    mon = str(env.repo / "k8s/monitoring")
    applies = [a for a in argvs if "apply" in a]
    assert applies[0][applies[0].index("-f") + 1] == f"{mon}/namespaces.yaml"
    files = [a[i + 1] for a in applies[1:] for i, w in enumerate(a) if w == "-f"]
    assert files == ["-", *(f"{mon}/{m}" for m in MANIFESTS)]  # the dashboard ConfigMap, then the manifests
    # The namespace exists before the Secret is created; the Secret and dashboard before Grafana.
    index = {json.dumps(a): i for i, a in enumerate(argvs)}
    secret_create = next(i for i, a in enumerate(argvs) if "create" in a and "configmap" not in a)
    assert index[json.dumps(applies[0])] < secret_create < index[json.dumps(applies[-1])]
    dash = next(a for a in argvs if "configmap" in a)
    assert dash[dash.index("create"):] == [
        "create", "configmap", "grafana-dashboards", f"--from-file=hybrid-gpu.json={mon}/grafana-dashboard.json",
        "--dry-run=client", "-o", "yaml"]
    rollouts = [a[a.index("rollout"):] for a in argvs if "rollout" in a]
    assert rollouts == [
        ["rollout", "status", "deployment/prometheus", "--timeout=5m"],
        ["rollout", "status", "deployment/kube-state-metrics", "--timeout=5m"],
        ["rollout", "status", "deployment/grafana", "--timeout=5m"],
        ["rollout", "status", "daemonset/node-exporter", "--timeout=5m"],
        ["rollout", "status", "daemonset/dcgm-exporter", "--timeout=5m"],
    ]


def test_install_restarts_a_running_grafana_when_the_password_is_new(env):
    (env.state / "grafana").write_text("")  # deployed before, Secret deleted since
    proc = env.run("install")
    assert proc.returncode == 0, proc.stderr
    restarts = [a for a in env.argvs() if "restart" in a]
    assert len(restarts) == 1 and restarts[0][-3:] == ["rollout", "restart", "deployment/grafana"]


def test_install_does_not_restart_grafana_on_an_ordinary_upgrade(env):
    (env.state / "grafana").write_text("")
    (env.state / "secret").write_text("existing")
    assert env.run("install").returncode == 0
    assert not [a for a in env.argvs() if "restart" in a]


def test_install_fails_when_a_rollout_does_not_finish(env):
    (env.state / "rollout-fails").write_text("")
    proc = env.run("install")
    assert proc.returncode != 0


def test_install_needs_the_kubeconfig(env):
    env.kubeconfig.unlink()
    proc = env.run("install")
    assert proc.returncode != 0 and "make kubeconfig" in proc.stderr
    assert env.calls() == []


# ---- uninstall ----------------------------------------------------------------------------------

def test_uninstall_asks_first(env):
    proc = env.run("uninstall", stdin="n\n")
    assert proc.returncode == 2
    assert env.calls() == []


def test_uninstall_deletes_the_manifests_then_the_namespaces(env):
    proc = env.run("uninstall", stdin="yes\n")
    assert proc.returncode == 0, proc.stderr
    deletes = [a for a in env.argvs() if "delete" in a]
    assert len(deletes) == 1
    mon = str(env.repo / "k8s/monitoring")
    files = [deletes[0][i + 1] for i, w in enumerate(deletes[0]) if w == "-f"]
    assert files == [*(f"{mon}/{m}" for m in MANIFESTS), f"{mon}/namespaces.yaml"]
    assert "--ignore-not-found" in deletes[0]


def test_uninstall_under_auto_approve_does_not_ask(env):
    proc = env.run("uninstall", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert [a for a in env.argvs() if "delete" in a]


# ---- port-forwards ------------------------------------------------------------------------------

@pytest.mark.parametrize("name,remote", [("grafana", 3000), ("prometheus", 9090)])
def test_forward_binds_to_loopback_in_the_background(env, name, remote):
    port = env.grafana_port if name == "grafana" else env.prometheus_port
    proc = env.run("forward", name)
    assert proc.returncode == 0, proc.stderr
    pid = env.pid(name)
    assert pid and alive(pid)
    forward = [a for a in env.argvs() if "port-forward" in a]
    assert forward == [["--kubeconfig", str(env.kubeconfig), "port-forward", "--address", "127.0.0.1",
                        "-n", "monitoring", f"svc/{name}", f"{port}:{remote}"]]
    assert f"http://127.0.0.1:{port}" in proc.stderr


def test_forward_twice_keeps_the_running_one(env):
    assert env.run("forward", "grafana").returncode == 0
    pid = env.pid("grafana")
    proc = env.run("forward", "grafana")
    assert proc.returncode == 0, proc.stderr
    assert env.pid("grafana") == pid and alive(pid)
    assert len([a for a in env.argvs() if "port-forward" in a]) == 1


def test_stop_ends_the_forward(env):
    assert env.run("forward", "grafana").returncode == 0
    pid = env.pid("grafana")
    proc = env.run("stop", "grafana")
    assert proc.returncode == 0, proc.stderr
    deadline = time.time() + 5
    while alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not alive(pid)
    assert env.pid("grafana") is None
    assert env.run("stop", "grafana").returncode == 0  # idempotent


def test_forward_refuses_a_port_in_use(env):
    with socket.socket() as s:
        s.bind(("127.0.0.1", env.grafana_port))
        s.listen()
        proc = env.run("forward", "grafana")
    assert proc.returncode != 0 and "in use" in proc.stderr
    assert not [a for a in env.argvs() if "port-forward" in a]


def test_forward_reports_a_failed_kubectl(env):
    (env.state / "port-forward-fails").write_text("")
    proc = env.run("forward", "grafana")
    assert proc.returncode != 0
    assert "pod is not running" in proc.stderr
    assert env.pid("grafana") is None


def test_forward_rejects_a_bad_port(env):
    proc = env.run("forward", "grafana", GRAFANA_PORT="0.0.0.0:3000")
    assert proc.returncode != 0
    assert env.calls() == []


@pytest.mark.parametrize("args", [[], ["forward"], ["forward", "alertmanager"], ["bogus"]])
def test_usage(env, args):
    assert env.run(*args).returncode == 2


# ---- no secrets in the repo ---------------------------------------------------------------------

def test_no_credential_is_written_into_the_script_or_the_manifests():
    text = SCRIPT.read_text() + "".join(p.read_text() for p in (ROOT / "k8s/monitoring").iterdir())
    assert not re.search(r"GF_SECURITY_ADMIN_PASSWORD\s+value:", text)
    # No Secret data anywhere but the script's printf template, which takes the generated value.
    assert re.findall(r"admin-password: *[^\s\\]+", text) == ["admin-password: %s"]
    assert "--from-literal" not in text  # would put the password on kubectl's command line


# ---- Makefile -----------------------------------------------------------------------------------

@pytest.mark.parametrize("target,command", [
    ("monitoring", "scripts/monitoring.sh install"),
    ("monitoring-down", "scripts/monitoring.sh uninstall"),
    ("grafana", "scripts/monitoring.sh forward grafana"),
    ("grafana-stop", "scripts/monitoring.sh stop grafana"),
    ("prometheus", "scripts/monitoring.sh forward prometheus"),
    ("prometheus-stop", "scripts/monitoring.sh stop prometheus")])
def test_make_targets(target, command):
    proc = subprocess.run(["make", "--no-print-directory", "-n", "-f", str(ROOT / "Makefile"), target],
                          capture_output=True, text=True, cwd="/")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{ROOT}/{command}"
