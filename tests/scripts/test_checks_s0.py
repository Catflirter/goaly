"""Offline tests for stretch S0b: check S0 (check_s0_metrics in scripts/lib/checks.sh), its registration in
scripts/verify.sh under STRETCH=1 only (group S), and the up.sh monitoring stage under STRETCH=1 only.

Same harness as test_checks_abc.py (Task 18): a repo copy whose kubectl is tests/scripts/fake_tools.py with
scripted answers. The fake `kubectl port-forward` serves Prometheus' /api/v1/query and Grafana's /api/health
on 127.0.0.1 from the scenario; curl is the real one, against that fake. Nothing reaches a cluster.
"""
import json
import os
import re
import signal
import socket
import subprocess

import pytest

from test_checks_abc import CP, CPU1, EXPECTED, GPU1, NOT_REACHABLE, Cluster, cmd_lines, kc
from test_ops_scripts import ROOT, Env, free_port

GPU_LABEL = "node-restriction.kubernetes.io/accelerator=nvidia"
UP_QUERY = 'up{job=~"node-exporter|kube-state-metrics"}'
DCGM_QUERY = "DCGM_FI_DEV_GPU_UTIL"
HEALTH_OK = json.dumps({"commit": "0123abc", "database": "ok", "version": "13.2.4"})


def vector(result):
    return json.dumps({"status": "success", "data": {"resultType": "vector", "result": result}})


def up(*series):
    """An `up` query answer: SERIES are (job, node, value); node None for kube-state-metrics (no node label)."""
    out = []
    for job, node, value in series:
        metric = {"__name__": "up", "job": job, "instance": node or "10.42.1.20:8080"}
        if node:
            metric["node"] = node
        out.append({"metric": metric, "value": [1792000000.123, value]})
    return vector(out)


def all_up(nodes):
    return up(*[("node-exporter", n, "1") for n in nodes], ("kube-state-metrics", None, "1"))


def dcgm(*nodes):
    return vector([{"metric": {"__name__": DCGM_QUERY, "job": "dcgm-exporter", "node": n, "instance": n, "gpu": "0",
                               "modelName": "NVIDIA RTX A6000"}, "value": [1792000000.123, "37"]} for n in nodes])


class S0(Cluster):
    """Cluster plus the monitoring stack's answers and the two local ports check S0 forwards to."""

    def __init__(self, tmp, nodes=(CP, CPU1), gpus=()):
        super().__init__(tmp, nodes=nodes)
        self.nodes, self.gpus = nodes, gpus
        self.pport, self.gport = free_port(), free_port()
        self.prom = {UP_QUERY: all_up(nodes), DCGM_QUERY: dcgm(*gpus)}
        self.grafana = {"/api/health": HEALTH_OK}
        self.installed = True
        self.overrides = []
        self.base = self.scenario["answers"]

    def first(self, *rules):
        """Answers that win over the monitoring stack's and the healthy cluster's."""
        self.overrides[:0] = list(rules)

    def environ(self, extra):
        self.scenario["answers"] = self.overrides + self.s0_answers() + self.base
        return super().environ({"VR_S0_PROM_PORT": str(self.pport), "VR_S0_GRAFANA_PORT": str(self.gport), **extra})

    def s0_answers(self):
        return [
            kc("get namespace monitoring", stdout="namespace/monitoring\n" if self.installed else ""),
            kc("get nodes", GPU_LABEL, stdout="".join(f"{g}\n" for g in self.gpus)),
            kc("-n monitoring get pods -o wide", stdout="prometheus-1 1/1 Running k3s-cpu-1\n"),
            kc("-n monitoring-host get pods -o wide", stdout="".join(f"node-exporter-{i} 1/1 Running {n}\n"
                                                                      for i, n in enumerate(self.nodes))),
            kc("port-forward", "svc/prometheus", serve=self.prom),
            kc("port-forward", "svc/grafana", serve=self.grafana),
        ]

    def s0(self, **extra):
        return self.check("check_s0_metrics", **extra)

    def closed(self):
        for port in (self.pport, self.gport):
            with pytest.raises(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
        assert list(self.tmpdir.iterdir()) == [], "a port-forward log was left behind"
        return True


@pytest.fixture
def env(tmp_path):
    return S0(tmp_path)


@pytest.fixture
def env3(tmp_path):
    return S0(tmp_path, nodes=(CP, CPU1, GPU1), gpus=(GPU1,))


def forwards(env):
    return [a for a in env.kubectl() if "port-forward" in a and "monitoring" in a]


# ---- check S0 ---------------------------------------------------------------------------------------

def test_s0_skips_with_a_reason_when_the_monitoring_stack_is_not_installed(env):
    env.installed = False
    result, text = env.s0()
    assert result == "SKIP", text
    assert "not installed" in text and "make monitoring" in text
    assert forwards(env) == []
    assert env.closed()


def test_s0_fails_rather_than_skips_when_the_namespace_cannot_be_read(env):
    env.first(kc("get namespace monitoring", rc=1,
                 stderr="Unable to connect to the server: dial tcp 127.0.0.1:6443: connect: connection refused\n"))
    result, text = env.s0()
    assert result == "FAIL", text
    assert forwards(env) == []


def test_s0_passes_without_a_gpu_node_and_skips_only_the_dcgm_part(env):
    result, text = env.s0()
    assert result == "PASS", text
    lines = cmd_lines(text)
    assert f"$ kc -n monitoring port-forward --address 127.0.0.1 svc/prometheus {env.pport}:9090 &" in lines
    assert f"$ kc -n monitoring port-forward --address 127.0.0.1 svc/grafana {env.gport}:3000 &" in lines
    assert any(UP_QUERY in line and f"http://127.0.0.1:{env.pport}/api/v1/query" in line for line in lines), lines
    assert f"$ curl -fsS -m 5 http://127.0.0.1:{env.gport}/api/health" in lines
    assert not any(DCGM_QUERY in line for line in lines)
    assert "no GPU node" in text and "skipped" in text
    assert "node-exporter k3s-cp" in text and "node-exporter k3s-cpu-1" in text  # the targets are evidence
    assert env.closed()


def test_s0_queries_only_through_a_port_forward_on_127_0_0_1(env):
    env.s0()
    for argv in env.kubectl():
        assert not any("proxy" in a for a in argv), argv  # never the API server's service proxy (S0a C0-5)
    for argv in forwards(env):
        assert argv[argv.index("--address") + 1] == "127.0.0.1"


def test_s0_with_a_gpu_node_needs_dcgm_data_from_it(env3):
    result, text = env3.s0()
    assert result == "PASS", text
    assert any(DCGM_QUERY in line for line in cmd_lines(text))
    assert f"{GPU1}" in text and "DCGM" in text
    assert env3.closed()


def test_s0_fails_when_a_gpu_node_has_no_dcgm_data(env3):
    env3.prom[DCGM_QUERY] = dcgm()
    result, text = env3.s0()
    assert result == "FAIL"
    assert f"no {DCGM_QUERY} from {GPU1}" in text
    assert env3.closed()


@pytest.mark.parametrize("series,why", [
    ((("node-exporter", CP, "1"), ("node-exporter", CPU1, "0"), ("kube-state-metrics", None, "1")),
     f"node-exporter target {CPU1}"),
    ((("node-exporter", CP, "1"), ("kube-state-metrics", None, "1")), f"no node-exporter target up on node {CPU1}"),
    ((("node-exporter", CP, "1"), ("node-exporter", CPU1, "1")), "no kube-state-metrics target is up"),
    ((("node-exporter", CP, "1"), ("node-exporter", CPU1, "1"), ("kube-state-metrics", None, "0")),
     "kube-state-metrics target"),
])
def test_s0_fails_unless_every_node_exporter_and_kube_state_metrics_target_is_up(env, series, why):
    env.prom[UP_QUERY] = up(*series)
    result, text = env.s0()
    assert result == "FAIL"
    assert why in text
    assert "gave up after" in text  # it waited (bounded) for the targets before failing
    assert env.closed()


def test_s0_waits_for_targets_that_come_up(env):
    env.prom[UP_QUERY] = [up(("node-exporter", CP, "1"), ("kube-state-metrics", None, "1")), all_up(env.nodes)]
    result, text = env.s0()
    assert result == "PASS", text
    assert f"no node-exporter target up on node {CPU1}" in text and "skipped sleep" in text


@pytest.mark.parametrize("answer", [json.dumps({"status": "error", "errorType": "bad_data", "error": "parse error"}),
                                    "not json", {"body": "", "status": 503}])
def test_s0_fails_on_a_prometheus_error(env, answer):
    env.prom[UP_QUERY] = answer
    assert env.s0()[0] == "FAIL"
    assert env.closed()


@pytest.mark.parametrize("answer", [json.dumps({"database": "failing", "version": "13.2.4"}),
                                    {"body": HEALTH_OK, "status": 503}, "<html>"])
def test_s0_fails_unless_grafana_health_says_ok(env, answer):
    env.grafana["/api/health"] = answer
    result, text = env.s0()
    assert result == "FAIL"
    assert "Grafana" in text
    assert env.closed()


@pytest.mark.parametrize("svc", ["prometheus", "grafana"])
def test_s0_fails_when_a_port_forward_never_starts_and_still_checks_the_other(env, svc):
    env.first(kc("port-forward", f"svc/{svc}", rc=1, stderr="error: unable to forward port: pod is not running\n"))
    result, text = env.s0()
    assert result == "FAIL"
    assert f"the {svc} port-forward did not start" in text
    assert len(forwards(env)) == 2  # one failure does not hide the other's evidence
    assert env.closed()


@pytest.mark.parametrize("bad", ["0", "70000", "x"])
def test_s0_refuses_a_bad_local_port(env, bad):
    result, text = env.s0(VR_S0_GRAFANA_PORT=bad)
    assert result == "FAIL" and "not a port" in text
    assert env.kubectl() == []


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_s0_interrupted_stops_the_forward_and_removes_its_log(env, sig):
    env.grafana["/api/health"] = {"body": HEALTH_OK, "sleep": 20}  # the check is waiting on Grafana
    rc = env.interrupt("check_s0_metrics", "svc/grafana", sig)
    err, text = env.interrupted
    assert rc != 0
    assert "interrupted" in err and "cleanup done" in err
    assert "cleanup done" in text
    assert env.closed()


# ---- verify.sh: group S only under STRETCH=1 -------------------------------------------------------

def verify(env, *groups, **extra):
    proc = env.run(["bash", str(env.repo / "scripts/verify.sh"), *groups], VR_NO_WAIT="1", VR_PF_PORT=str(free_port()),
                   **{"BASH_FUNC_timeout%%": NOT_REACHABLE, "TZ": "America/Los_Angeles", **extra})
    runs = list((env.repo / "verification").iterdir()) if (env.repo / "verification").exists() else []
    summary = (runs[0] / "SUMMARY.md").read_text() if runs else ""
    return proc, re.findall(r"^\| \d+ \| (\w+) \| [^|]+ \| (\w+) \|$", summary, re.M)


@pytest.mark.parametrize("stretch", [None, "0", ""])
def test_make_verify_without_stretch_is_unchanged(env, stretch):
    proc, results = verify(env, **({} if stretch is None else {"STRETCH": stretch}))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert results == [(i, "SKIP" if i.startswith("D") else "PASS") for _, i, _ in EXPECTED]
    assert "S0" not in proc.stdout and "group S" not in proc.stderr
    assert forwards(env) == []


def test_group_s_needs_stretch(env):
    proc, results = verify(env, "S")
    assert proc.returncode == 2
    assert "STRETCH=1" in proc.stderr
    assert results == [] and env.kubectl() == []


def test_make_verify_stretch_runs_group_s_after_d(env):
    proc, results = verify(env, STRETCH="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert results == [*[(i, "SKIP" if i.startswith("D") else "PASS") for _, i, _ in EXPECTED], ("S0", "PASS")]


def test_verify_stretch_group_s_alone(env):
    proc, results = verify(env, "S", STRETCH="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert results == [("S0", "PASS")]


def test_verify_stretch_s0_skip_is_not_a_failure(env):
    env.installed = False
    proc, results = verify(env, "S", STRETCH="1")
    assert proc.returncode == 0
    assert results == [("S0", "SKIP")]


def test_verify_stretch_s0_fail_fails_the_run(env):
    env.grafana["/api/health"] = {"body": "", "status": 503}
    proc, results = verify(env, "S", STRETCH="1")
    assert proc.returncode == 1
    assert results == [("S0", "FAIL")]


# ---- up.sh: the monitoring stage only under STRETCH=1 ----------------------------------------------

@pytest.fixture
def ops_env(tmp_path):
    e = Env(tmp_path)
    yield e
    if e.pidfile.exists():  # never leave a background fake tunnel behind (as test_ops_scripts' env fixture)
        try:
            os.killpg(int(e.pidfile.read_text()), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def stub_monitoring(env, rc=0):
    """scripts/monitoring.sh in the repo copy: logs {"tool": "monitoring.sh", "argv"} and exits RC."""
    stub = env.repo / "scripts/monitoring.sh"
    stub.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                    "open(os.environ['FAKE_LOG'], 'a').write(json.dumps({'tool': 'monitoring.sh', "
                    f"'argv': sys.argv[1:], 'env': {{}}}}) + '\\n')\nsys.exit({rc})\n")
    stub.chmod(0o755)


def tools(env):
    return [c["tool"] for c in env.calls()]


@pytest.mark.parametrize("stretch", [None, "0"])
def test_make_up_without_stretch_does_not_install_monitoring(ops_env, stretch):
    stub_monitoring(ops_env)
    proc = ops_env.script("up.sh", "all", AUTO_APPROVE="1", **({} if stretch is None else {"STRETCH": stretch}))
    assert proc.returncode == 0, proc.stderr
    assert "monitoring.sh" not in tools(ops_env)
    assert "monitoring" not in proc.stderr


def test_make_up_stretch_installs_monitoring_after_the_workloads(ops_env):
    stub_monitoring(ops_env)
    proc = ops_env.script("up.sh", "all", AUTO_APPROVE="1", STRETCH="1")
    assert proc.returncode == 0, proc.stderr
    calls = ops_env.calls()
    assert [c["argv"] for c in calls if c["tool"] == "monitoring.sh"] == [["install"]]
    assert calls[-1]["tool"] == "monitoring.sh"
    last_rollout = max(i for i, c in enumerate(calls) if c["tool"] == "kubectl" and "deployment/hello" in c["argv"])
    assert last_rollout < len(calls) - 1


def test_make_up_stretch_fails_when_the_monitoring_install_fails(ops_env):
    stub_monitoring(ops_env, rc=1)
    proc = ops_env.script("up.sh", "all", AUTO_APPROVE="1", STRETCH="1")
    assert proc.returncode != 0


def test_up_monitoring_stage_runs_the_install(ops_env):
    stub_monitoring(ops_env)
    proc = ops_env.script("up.sh", "monitoring")
    assert proc.returncode == 0, proc.stderr
    assert [c["argv"] for c in ops_env.calls()] == [["install"]]


@pytest.mark.parametrize("target,needle", [("up", "STRETCH=1"), ("verify", "STRETCH=1")])
def test_make_help_mentions_stretch(target, needle):
    line = next(ln for ln in (ROOT / "Makefile").read_text().splitlines() if ln.startswith(f"{target}:"))
    assert needle in line


def test_make_verify_passes_stretch_through_the_environment():
    proc = subprocess.run(["make", "--no-print-directory", "-n", "-f", str(ROOT / "Makefile"), "verify", "STRETCH=1"],
                          capture_output=True, text=True, cwd="/")
    assert proc.returncode == 0 and proc.stdout.strip() == f"{ROOT}/scripts/verify.sh"
