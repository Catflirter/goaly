"""Offline tests for scripts/lib/common.sh (Task 10 helpers), scripts/up.sh, kubeconfig.sh, tunnel.sh.

The scripts run from a copy of the repo layout with terraform, gcloud and kubectl replaced by
tests/scripts/fake_tools.py on PATH. The fakes record every call, so the tests assert the exact
commands and flags the scripts would run. Nothing here reaches GCP, a cluster or the network.
"""
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FAKES = Path(__file__).with_name("fake_tools.py")
PROJECT = "k3s-demo-123456"
BUCKET = f"{PROJECT}-k3s-tfstate"
SCRIPTS = ["up.sh", "kubeconfig.sh", "tunnel.sh", "lib/common.sh"]
OUTPUTS = {"base": {"cp_zone": "us-west1-b", "cp_ip": "34.82.10.10", "cp_internal_ip": "10.10.0.2"},
           "bootstrap": {"state_bucket": BUCKET}}
KUBECONFIG = """apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: Q0E=
    server: https://127.0.0.1:6443
  name: default
kind: Config
users:
- name: default
  user:
    client-key-data: S0VZ
"""
SSH_IAP = ["--project", PROJECT, "--zone", "us-west1-b", "--tunnel-through-iap", "--quiet",
           "--ssh-flag=-oConnectTimeout=20", "--ssh-flag=-oServerAliveInterval=30"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Env:
    """A repo copy, a stub PATH, a scenario for the fakes, and helpers to run and inspect."""

    def __init__(self, tmp):
        self.repo, self.home, self.bin = tmp / "repo", tmp / "home", tmp / "bin"
        for rel in SCRIPTS:
            (self.repo / "scripts" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "scripts" / rel, self.repo / "scripts" / rel)
        for stack in ("bootstrap", "base"):
            (self.repo / "infra" / stack).mkdir(parents=True)
        shutil.copytree(ROOT / "node", self.repo / "node")
        self.home.mkdir()
        self.bin.mkdir()
        for tool in ("terraform", "gcloud", "kubectl"):
            (self.bin / tool).write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKES}" {tool} "$@"\n')
            (self.bin / tool).chmod(0o755)
        self.log = tmp / "calls.jsonl"
        self.scenario_file = tmp / "scenario.json"
        self.scenario = {"outputs": OUTPUTS, "kubeconfig": KUBECONFIG}
        self.port = free_port()
        self.write_env(f"GCP_PROJECT={PROJECT}\nLAMBDA_API_KEY=x\nUNDERLAY_MTU=1460\n")

    def write_env(self, text, mode=0o600):
        env = self.repo / ".env"
        env.write_text(text)
        env.chmod(mode)

    def run(self, args, stdin="", **extra):
        self.scenario_file.write_text(json.dumps(self.scenario))
        env = {k: v for k, v in os.environ.items()
               if k not in ("AUTO_APPROVE", "GCP_PROJECT", "UNDERLAY_MTU", "KUBECONFIG", "REPO_ROOT")
               and not k.startswith("TF_VAR_")}
        env.update(PATH=f"{self.bin}:{os.environ['PATH']}", HOME=str(self.home), FAKE_LOG=str(self.log),
                   FAKE_SCENARIO=str(self.scenario_file), WAIT_INTERVAL_S="0",
                   K3S_TUNNEL_TEST_PORT=str(self.port))
        env.update(extra)
        for tool in ("terraform", "gcloud", "kubectl"):  # never the real CLIs
            assert shutil.which(tool, path=env["PATH"]) == str(self.bin / tool)
        return subprocess.run(args, cwd=self.repo, env=env, input=stdin, capture_output=True, text=True,
                              timeout=60)

    def script(self, name, *args, **kw):
        return self.run([str(self.repo / "scripts" / name), *args], **kw)

    def lib(self, code, **kw):
        return self.run(["bash", "-c", f'set -euo pipefail; source scripts/lib/common.sh; {code}'], **kw)

    def calls(self, tool=None):
        if not self.log.exists():
            return []
        entries = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [e for e in entries if tool is None or e["tool"] == tool]

    def argvs(self, tool):
        return [e["argv"] for e in self.calls(tool)]

    @property
    def pidfile(self):
        return self.repo / ".run/tunnel.pid"

    @property
    def kubeconfig(self):
        return self.home / ".kube/hybrid-gpu.yaml"

    def forward(self):
        return f"127.0.0.1:{self.port}:127.0.0.1:6443"


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    # Never leave a background fake tunnel behind, whatever the test did.
    if e.pidfile.exists():
        pid = int(e.pidfile.read_text())
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def tf_base_dir(env):
    return f"-chdir={env.repo.resolve()}/infra/base"


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill 0; it is gone for our purposes.
    proc = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return proc.returncode == 0 and not proc.stdout.strip().startswith("Z")


def wait_gone(pid, seconds=5):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


# ---- scripts are executable bash -------------------------------------------------------------

@pytest.mark.parametrize("name", ["up.sh", "kubeconfig.sh", "tunnel.sh"])
def test_scripts_are_executable_bash(name):
    path = ROOT / "scripts" / name
    assert path.stat().st_mode & stat.S_IXUSR
    assert path.read_text().startswith("#!/usr/bin/env bash\n")


# ---- common.sh: tf_env, tf_init, tf_out ------------------------------------------------------

def test_tf_env_exports_project_and_state_bucket(env):
    proc = env.lib('load_env; tf_env; printf "%s %s" "$TF_VAR_project_id" "$TF_VAR_state_bucket"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"{PROJECT} {BUCKET}"


def test_tf_env_accepts_a_30_character_project_id(env):
    env.write_env("GCP_PROJECT=project-7b009e6f-dae9-421d-a9f\n")
    proc = env.lib('load_env; tf_env; printf "%s" "$TF_VAR_state_bucket"')
    assert proc.returncode == 0 and proc.stdout == "project-7b009e6f-dae9-421d-a9f-k3s-tfstate"


@pytest.mark.parametrize("value", ["", "Upper-Case-1", "x", "ends-with-", "has space 123", "a;rm -rf x"])
def test_tf_env_refuses_a_bad_project_id(env, value):
    env.write_env(f"GCP_PROJECT='{value}'\n")
    proc = env.lib("load_env; tf_env; echo reached")
    assert proc.returncode == 2 and "reached" not in proc.stdout and "GCP_PROJECT" in proc.stderr


def test_tf_init_uses_the_gcs_backend_with_the_stack_as_prefix(env):
    proc = env.lib("load_env; tf_init base")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "", "terraform init output must go to stderr"
    assert env.argvs("terraform") == [[
        tf_base_dir(env), "init", "-input=false", "-lockfile=readonly", "-reconfigure",
        f"-backend-config=bucket={BUCKET}", "-backend-config=prefix=base"]]


def test_tf_init_skips_a_stack_already_initialised_for_this_bucket(env):
    assert env.lib("load_env; tf_init base").returncode == 0
    assert env.lib("load_env; tf_init base").returncode == 0
    assert len(env.calls("terraform")) == 1


def test_tf_init_reinitialises_when_the_project_changed(env):
    assert env.lib("load_env; tf_init base").returncode == 0
    env.write_env("GCP_PROJECT=other-project-42\n")
    assert env.lib("load_env; tf_init base").returncode == 0
    inits = env.argvs("terraform")
    assert len(inits) == 2 and "-backend-config=bucket=other-project-42-k3s-tfstate" in inits[1]


def test_tf_init_refuses_the_bootstrap_stack_and_unknown_stacks(env):
    assert env.lib("load_env; tf_init bootstrap").returncode != 0
    assert env.lib("load_env; tf_init nosuch").returncode != 0
    assert env.lib("load_env; tf_init ../base").returncode != 0
    assert env.calls("terraform") == []


def test_tf_out_reads_raw_and_json_outputs(env):
    proc = env.lib('load_env; printf "%s|%s" "$(tf_out base cp_ip)" "$(tf_out_json base cp_zone)"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == '34.82.10.10|"us-west1-b"'
    outs = [a for a in env.argvs("terraform") if a[1] == "output"]
    assert outs == [[tf_base_dir(env), "output", "-raw", "cp_ip"],
                    [tf_base_dir(env), "output", "-json", "cp_zone"]]


def test_tf_out_bootstrap_reads_local_state_without_init(env):
    proc = env.lib("tf_out bootstrap state_bucket")
    assert proc.returncode == 0 and proc.stdout == BUCKET
    assert [a[1] for a in env.argvs("terraform")] == ["output"]


# ---- common.sh: kc, cp_ssh, cp_scp -----------------------------------------------------------

def test_kc_uses_the_cluster_kubeconfig(env):
    assert env.lib("kc get nodes -o wide").returncode == 0
    assert env.argvs("kubectl") == [["--kubeconfig", str(env.kubeconfig), "get", "nodes", "-o", "wide"]]


def test_cp_ssh_goes_through_iap_in_the_cp_zone_of_the_project(env):
    proc = env.lib("load_env; cp_ssh 'sudo k3s kubectl get nodes'; cp_ssh true")
    assert proc.returncode == 0, proc.stderr
    assert env.argvs("gcloud") == [
        ["compute", "ssh", "k3s-cp", *SSH_IAP, "--command", "sudo k3s kubectl get nodes"],
        ["compute", "ssh", "k3s-cp", *SSH_IAP, "--command", "true"]]
    # The zone is resolved once per process.
    assert [a for a in env.argvs("terraform") if a[1] == "output"] == [[tf_base_dir(env), "output", "-raw", "cp_zone"]]


def test_cp_scp_copies_recursively_through_iap(env):
    assert env.lib("load_env; cp_scp node /tmp/k3s-node").returncode == 0
    assert env.argvs("gcloud") == [[
        "compute", "scp", "--project", PROJECT, "--zone", "us-west1-b", "--tunnel-through-iap", "--quiet",
        "--recurse", "--scp-flag=-oConnectTimeout=20", "node", "k3s-cp:/tmp/k3s-node"]]


def test_cp_ssh_refuses_an_unexpected_zone_output(env):
    env.scenario["outputs"] = {"base": {"cp_zone": "us-west1-b --command=evil"}}
    proc = env.lib("load_env; cp_ssh true")
    assert proc.returncode != 0 and env.calls("gcloud") == []


# ---- common.sh: confirm, wait_for -------------------------------------------------------------

@pytest.mark.parametrize("answer,rc", [("y\n", 0), ("yes\n", 0), ("Y\n", 0), ("n\n", 1), ("\n", 1), ("", 1),
                                       ("yess\n", 1)])
def test_confirm_needs_an_explicit_yes(env, answer, rc):
    proc = env.lib('if confirm "Apply?"; then echo yes; else echo no; fi', stdin=answer)
    assert proc.stdout.strip() == ("yes" if rc == 0 else "no")
    assert "Apply?" in proc.stderr


def test_confirm_passes_without_reading_under_auto_approve(env):
    proc = env.lib('confirm "Apply?" && read -r rest && echo "$rest"', stdin="still-here\n", AUTO_APPROVE="1")
    assert proc.returncode == 0 and proc.stdout.strip() == "still-here"


def test_wait_for_retries_until_the_command_succeeds(env):
    proc = env.lib('n=0; wait_for "n=\\$((n + 1)); (( n >= 3 ))" 30; echo "tries=$n"')
    assert proc.returncode == 0 and proc.stdout.strip() == "tries=3"


def test_wait_for_gives_up_after_the_timeout(env):
    start = time.time()
    proc = env.lib('if wait_for "echo still-down >&2; false" 1; then echo ok; else echo timeout; fi',
                   WAIT_INTERVAL_S="1")
    assert proc.stdout.strip() == "timeout" and time.time() - start < 10
    assert "still-down" in proc.stderr, "the last attempt's output helps the operator"


# ---- common.sh: tf_apply ---------------------------------------------------------------------

def tf_subcommands(env):
    return [a[1] for a in env.argvs("terraform")]


def test_tf_apply_plans_confirms_then_applies_the_saved_plan(env):
    proc = env.lib("load_env; tf_init base; tf_apply base", stdin="y\n")
    assert proc.returncode == 0, proc.stderr
    plan_file = f"{env.repo.resolve()}/.run/base.tfplan"
    argvs = env.argvs("terraform")
    assert argvs[1] == [tf_base_dir(env), "plan", "-input=false", "-lock-timeout=120s", "-detailed-exitcode",
                        f"-out={plan_file}"]
    assert argvs[2] == [tf_base_dir(env), "show", "-json", plan_file]
    assert argvs[3] == [tf_base_dir(env), "apply", "-input=false", "-lock-timeout=120s", plan_file]
    assert all(c["env"].get("TF_VAR_project_id") == PROJECT and c["env"].get("TF_VAR_state_bucket") == BUCKET
               for c in env.calls("terraform"))
    assert not Path(plan_file).exists()
    assert stat.S_IMODE((env.repo / ".run").stat().st_mode) == 0o700


def test_tf_apply_declined_applies_nothing(env):
    proc = env.lib("load_env; tf_apply base", stdin="n\n")
    assert proc.returncode == 2 and "apply" not in tf_subcommands(env)
    assert not (env.repo / ".run/base.tfplan").exists()


def test_tf_apply_without_changes_neither_asks_nor_applies(env):
    env.scenario["plan_rc"] = {"base": 0}
    proc = env.lib("load_env; tf_apply base")  # stdin is empty: a prompt would read EOF and refuse
    assert proc.returncode == 0, proc.stderr
    assert tf_subcommands(env) == ["plan"] and "[y/N]" not in proc.stderr


def test_tf_apply_stops_on_a_failed_plan(env):
    env.scenario["plan_rc"] = {"base": 1}
    proc = env.lib("load_env; tf_apply base", AUTO_APPROVE="1")
    assert proc.returncode == 1 and tf_subcommands(env) == ["plan"]


def test_tf_apply_auto_approve_refuses_to_destroy_or_replace(env):
    env.scenario["deletes"] = {"base": ["google_compute_instance.cp"]}
    proc = env.lib("load_env; tf_apply base", AUTO_APPROVE="1")
    assert proc.returncode == 2 and "apply" not in tf_subcommands(env)
    assert "google_compute_instance.cp" in proc.stderr


def test_tf_apply_interactive_shows_what_it_destroys_and_still_asks(env):
    env.scenario["deletes"] = {"base": ["google_compute_instance.cp"]}
    proc = env.lib("load_env; tf_apply base", stdin="y\n")
    assert proc.returncode == 0 and tf_subcommands(env)[-1] == "apply"
    assert "google_compute_instance.cp" in proc.stderr


def test_tf_apply_allow_delete_lets_auto_approve_replace(env):
    env.scenario["deletes"] = {"base": ["google_compute_instance.cp"]}
    proc = env.lib("load_env; tf_apply base --allow-delete", AUTO_APPROVE="1")
    assert proc.returncode == 0 and tf_subcommands(env)[-1] == "apply"


# ---- kubeconfig.sh ---------------------------------------------------------------------------

def test_kubeconfig_writes_a_private_file_from_the_cp(env):
    proc = env.script("kubeconfig.sh")
    assert proc.returncode == 0, proc.stderr
    assert env.kubeconfig.read_text() == KUBECONFIG
    assert stat.S_IMODE(env.kubeconfig.stat().st_mode) == 0o600
    assert stat.S_IMODE(env.kubeconfig.parent.stat().st_mode) == 0o700
    assert env.argvs("gcloud") == [["compute", "ssh", "k3s-cp", *SSH_IAP, "--command",
                                    "sudo cat /etc/rancher/k3s/k3s.yaml"]]
    assert "client-key-data" not in proc.stdout + proc.stderr, "never print the credentials"


def test_kubeconfig_replaces_an_existing_world_readable_file(env):
    env.kubeconfig.parent.mkdir()
    env.kubeconfig.write_text("old\n")
    env.kubeconfig.chmod(0o644)
    assert env.script("kubeconfig.sh").returncode == 0
    assert env.kubeconfig.read_text() == KUBECONFIG
    assert stat.S_IMODE(env.kubeconfig.stat().st_mode) == 0o600


@pytest.mark.parametrize("server", ["https://34.82.10.10:6443", "https://127.0.0.1:6444", "", "MISSING", "TWICE"])
def test_kubeconfig_refuses_any_server_but_the_tunnel(env, server):
    if server == "MISSING":
        text = KUBECONFIG.replace("    server: https://127.0.0.1:6443\n", "")
    elif server == "TWICE":
        text = KUBECONFIG + "- cluster:\n    server: https://127.0.0.1:6443\n"
    else:
        text = KUBECONFIG.replace("https://127.0.0.1:6443", server)
    env.scenario["kubeconfig"] = text
    env.kubeconfig.parent.mkdir()
    env.kubeconfig.write_text("previous\n")
    proc = env.script("kubeconfig.sh")
    assert proc.returncode != 0 and "127.0.0.1:6443" in proc.stderr
    assert env.kubeconfig.read_text() == "previous\n", "a bad file must not replace the working one"
    assert sorted(p.name for p in env.kubeconfig.parent.iterdir()) == ["hybrid-gpu.yaml"], "no temp file left"


def test_kubeconfig_failed_ssh_leaves_nothing(env):
    env.bin.joinpath("gcloud").unlink()
    env.bin.joinpath("gcloud").write_text("#!/bin/sh\nexit 255\n")
    env.bin.joinpath("gcloud").chmod(0o755)
    proc = env.script("kubeconfig.sh")
    assert proc.returncode != 0 and not env.kubeconfig.exists()
    assert list(env.kubeconfig.parent.iterdir()) == []


# ---- tunnel.sh -------------------------------------------------------------------------------

def tunnel_argv(env):
    return ["compute", "ssh", "k3s-cp", "--project", PROJECT, "--zone", "us-west1-b", "--tunnel-through-iap",
            "--quiet", "--", "-N", "-L", env.forward(), "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30"]


def tunnels(env):
    return [a for a in env.argvs("gcloud") if "-N" in a]


def make_kubeconfig(env):
    env.kubeconfig.parent.mkdir(exist_ok=True)
    env.kubeconfig.write_text(KUBECONFIG)
    env.kubeconfig.chmod(0o600)


def test_tunnel_start_status_stop(env):
    make_kubeconfig(env)
    proc = env.script("tunnel.sh", "start")
    assert proc.returncode == 0, proc.stderr
    assert tunnels(env) == [tunnel_argv(env)]
    pid = int(env.pidfile.read_text())
    assert alive(pid)
    assert stat.S_IMODE(env.pidfile.stat().st_mode) & 0o077 == 0
    assert ["--kubeconfig", str(env.kubeconfig), "--request-timeout=5s", "get", "--raw", "/readyz"] \
        in env.argvs("kubectl")

    assert env.script("tunnel.sh", "status").returncode == 0

    # start again: already up, no second tunnel
    again = env.script("tunnel.sh", "start")
    assert again.returncode == 0 and len(tunnels(env)) == 1 and int(env.pidfile.read_text()) == pid

    stop = env.script("tunnel.sh", "stop")
    assert stop.returncode == 0, stop.stderr
    assert wait_gone(pid) and not env.pidfile.exists()
    assert env.script("tunnel.sh", "status").returncode == 1
    assert env.script("tunnel.sh", "stop").returncode == 0, "stop is idempotent"


def test_tunnel_stop_kills_the_whole_process_group(env):
    """gcloud runs ssh, which runs the IAP helper: stopping only the leader would leave ssh holding the port."""
    make_kubeconfig(env)
    child_pidfile = env.repo / "child.pid"
    fake = env.bin / "gcloud"
    fake.unlink()
    fake.write_text(f'#!/usr/bin/env bash\nsleep 300 &\necho $! > "{child_pidfile}"\n'
                    f'echo $$ > "{env.log}.tunnel"\nwait\n')
    fake.chmod(0o755)
    assert env.script("tunnel.sh", "start").returncode == 0
    deadline = time.time() + 5
    while not child_pidfile.exists() and time.time() < deadline:
        time.sleep(0.05)
    child = int(child_pidfile.read_text())
    assert alive(child)
    assert env.script("tunnel.sh", "stop").returncode == 0
    assert wait_gone(child)


def test_tunnel_start_requires_the_kubeconfig(env):
    proc = env.script("tunnel.sh", "start")
    assert proc.returncode != 0 and "make kubeconfig" in proc.stderr and tunnels(env) == []


def test_tunnel_start_refuses_a_port_in_use(env):
    make_kubeconfig(env)
    with socket.socket() as s:
        s.bind(("127.0.0.1", env.port))
        s.listen()
        proc = env.script("tunnel.sh", "start")
    assert proc.returncode != 0 and "in use" in proc.stderr
    assert tunnels(env) == [] and not env.pidfile.exists()


def test_tunnel_start_replaces_a_stale_pidfile(env):
    make_kubeconfig(env)
    dead = subprocess.Popen(["true"])
    dead.wait()
    env.pidfile.parent.mkdir()
    env.pidfile.write_text(f"{dead.pid}\n")
    assert env.script("tunnel.sh", "start").returncode == 0
    assert len(tunnels(env)) == 1 and int(env.pidfile.read_text()) != dead.pid


def test_tunnel_never_kills_an_unrelated_process_named_in_the_pidfile(env):
    other = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        env.pidfile.parent.mkdir()
        env.pidfile.write_text(f"{other.pid}\n")
        assert env.script("tunnel.sh", "stop").returncode == 0
        assert other.poll() is None, "a recycled PID must not be killed"
        assert env.script("tunnel.sh", "status").returncode == 1
    finally:
        other.kill()
        other.wait()


def test_tunnel_that_exits_fails_fast_and_cleans_up(env):
    make_kubeconfig(env)
    env.scenario.update(tunnel="exit", readyz="down")
    start = time.time()
    proc = env.script("tunnel.sh", "start")
    assert proc.returncode != 0 and time.time() - start < 15
    assert not env.pidfile.exists() and "Address already in use" in proc.stderr


def test_tunnel_without_a_ready_api_times_out_and_stops(env):
    make_kubeconfig(env)
    env.scenario["readyz"] = "down"
    proc = env.script("tunnel.sh", "start", TUNNEL_TIMEOUT_S="2")
    assert proc.returncode != 0 and "/readyz" in proc.stderr
    assert not env.pidfile.exists()
    # The fake tunnel itself was stopped, not left running.
    procs = subprocess.run(["ps", "-eww", "-o", "pid=,args="], capture_output=True, text=True).stdout
    assert env.forward() not in procs


def test_tunnel_running_but_api_down_is_restarted(env):
    make_kubeconfig(env)
    assert env.script("tunnel.sh", "start").returncode == 0
    old = int(env.pidfile.read_text())
    env.scenario["readyz"] = "down"
    assert env.script("tunnel.sh", "status").returncode == 1
    # The hung tunnel's first /readyz fails; the replacement's succeeds.
    env.scenario.update(readyz="ok", readyz_fail_first=1)
    Path(f"{env.log}.readyz").unlink()  # restart the fake's /readyz call count
    proc = env.script("tunnel.sh", "start")
    assert proc.returncode == 0, proc.stderr
    new = int(env.pidfile.read_text())
    assert new != old and wait_gone(old) and alive(new)
    assert len(tunnels(env)) == 2


def test_tunnel_usage(env):
    proc = env.script("tunnel.sh", "bogus")
    assert proc.returncode == 2 and "usage" in proc.stderr


# ---- up.sh -----------------------------------------------------------------------------------

def base_sequence(env, mtu="1460"):
    """The gcloud/kubectl calls of a full `up.sh base`, without the tunnel and its /readyz probes (non_tf)."""
    return [
        ("gcloud", ["compute", "ssh", "k3s-cp", *SSH_IAP, "--command", "true"]),
        ("gcloud", ["compute", "ssh", "k3s-cp", *SSH_IAP, "--command", "rm -rf /tmp/k3s-node"]),
        ("gcloud", ["compute", "scp", "--project", PROJECT, "--zone", "us-west1-b", "--tunnel-through-iap",
                    "--quiet", "--recurse", "--scp-flag=-oConnectTimeout=20", f"{env.repo.resolve()}/node",
                    "k3s-cp:/tmp/k3s-node"]),
        ("gcloud", ["compute", "ssh", "k3s-cp", *SSH_IAP, "--command",
                    "sudo bash /tmp/k3s-node/server-bootstrap.sh --cp-ip 34.82.10.10 --cp-internal-ip 10.10.0.2"
                    f" --underlay-mtu {mtu}"]),
        ("gcloud", ["compute", "ssh", "k3s-cp", *SSH_IAP, "--command", "sudo cat /etc/rancher/k3s/k3s.yaml"]),
        ("kubectl", ["--kubeconfig", str(env.kubeconfig), "wait", "--for=condition=Ready", "node/k3s-cp",
                     "--timeout=10s"]),
        ("kubectl", ["--kubeconfig", str(env.kubeconfig), "-n", "kube-system", "wait", "--for=condition=Ready",
                     "pod", "-l", "k8s-app=kube-dns", "--timeout=10s"]),
    ]


READYZ = ["--request-timeout=5s", "get", "--raw", "/readyz"]


def non_tf(env):
    """gcloud/kubectl calls in order, without the tunnel and the /readyz probes: the backgrounded
    fake tunnel logs its call concurrently with tunnel.sh's first probes. What is deterministic is
    checked here: one tunnel, started after the kubeconfig fetch, and a successful probe after it
    and before the first `kc wait`."""
    calls = [(c["tool"], c["argv"]) for c in env.calls() if c["tool"] != "terraform"]
    assert tunnels(env) == [tunnel_argv(env)]
    tunnel_at = calls.index(("gcloud", tunnel_argv(env)))
    probes = [i for i, (_, argv) in enumerate(calls) if argv[2:] == READYZ]
    first_wait = next(i for i, (_, argv) in enumerate(calls) if "wait" in argv)
    before = [argv for _, argv in calls[:tunnel_at] if argv[2:] != READYZ]
    assert before[-1][-1] == "sudo cat /etc/rancher/k3s/k3s.yaml"
    assert probes and tunnel_at < probes[-1] < first_wait
    return [c for c in calls if c[1][2:] != READYZ and "-N" not in c[1]]


def test_up_base_runs_the_whole_sequence(env):
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    tf = tf_subcommands(env)
    assert tf[:4] == ["init", "plan", "show", "apply"]
    assert env.argvs("terraform")[0][-2:] == [f"-backend-config=bucket={BUCKET}", "-backend-config=prefix=base"]
    assert non_tf(env) == base_sequence(env)
    assert env.kubeconfig.read_text() == KUBECONFIG
    assert alive(int(env.pidfile.read_text()))


def test_up_base_waits_for_iap_ssh(env):
    env.scenario["ssh_failures"] = 3
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    trues = [a for a in env.argvs("gcloud") if a[-1] == "true"]
    assert len(trues) == 4


def test_up_base_rerun_is_safe(env):
    """Second run: nothing to change, so no prompt and no apply; the running tunnel is reused."""
    assert env.script("up.sh", "base", AUTO_APPROVE="1").returncode == 0
    pid = int(env.pidfile.read_text())
    env.log.unlink()
    env.scenario["plan_rc"] = {"base": 0}
    proc = env.script("up.sh", "base")  # no AUTO_APPROVE and no stdin: any prompt would refuse
    assert proc.returncode == 0, proc.stderr
    assert tf_subcommands(env).count("init") == 0 and "apply" not in tf_subcommands(env)
    assert tunnels(env) == [] and int(env.pidfile.read_text()) == pid
    bootstrap = [a for a in env.argvs("gcloud") if "server-bootstrap.sh" in a[-1]]
    assert len(bootstrap) == 1, "server-bootstrap is idempotent and re-run every time"


def test_up_base_declined_apply_touches_nothing_else(env):
    proc = env.script("up.sh", "base", stdin="n\n")
    assert proc.returncode == 2
    assert "apply" not in tf_subcommands(env) and env.calls("gcloud") == [] and env.calls("kubectl") == []


def test_up_base_refuses_to_replace_the_cp_under_auto_approve(env):
    env.scenario["deletes"] = {"base": ["google_compute_instance.cp"]}
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode == 2 and "apply" not in tf_subcommands(env) and env.calls("gcloud") == []


def test_up_base_passes_underlay_mtu_from_env(env):
    env.write_env(f"GCP_PROJECT={PROJECT}\nUNDERLAY_MTU=1380\n")
    assert env.script("up.sh", "base", AUTO_APPROVE="1").returncode == 0
    assert non_tf(env) == base_sequence(env, mtu="1380")


def test_up_base_defaults_an_empty_underlay_mtu(env):
    env.write_env(f"GCP_PROJECT={PROJECT}\nUNDERLAY_MTU=\n")
    assert env.script("up.sh", "base", AUTO_APPROVE="1").returncode == 0
    assert non_tf(env) == base_sequence(env, mtu="1460")


@pytest.mark.parametrize("mtu", ["1460; reboot", "abc", "0", "99999"])
def test_up_base_refuses_a_bad_underlay_mtu_before_touching_anything(env, mtu):
    env.write_env(f"GCP_PROJECT={PROJECT}\nUNDERLAY_MTU='{mtu}'\n")
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode == 2 and env.calls() == []


@pytest.mark.parametrize("output", ["cp_ip", "cp_internal_ip"])
def test_up_base_refuses_a_non_ipv4_output(env, output):
    env.scenario["outputs"] = {"base": {**OUTPUTS["base"], output: "1.2.3.4; reboot"}}
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode != 0
    assert not [a for a in env.argvs("gcloud") if "server-bootstrap.sh" in a[-1]]


def test_up_init_applies_the_bootstrap_stack_with_local_state(env):
    proc = env.script("up.sh", "init", stdin="y\n")
    assert proc.returncode == 0, proc.stderr
    argvs = env.argvs("terraform")
    chdir = f"-chdir={env.repo.resolve()}/infra/bootstrap"
    assert argvs[0] == [chdir, "init", "-input=false", "-lockfile=readonly"]
    assert [a[1] for a in argvs] == ["init", "plan", "show", "apply"]
    assert all(a[0] == chdir for a in argvs)
    assert all(c["env"].get("TF_VAR_project_id") == PROJECT for c in env.calls("terraform"))
    assert env.calls("gcloud") == [] and env.calls("kubectl") == []


def test_up_init_declined(env):
    proc = env.script("up.sh", "init", stdin="")
    assert proc.returncode == 2 and "apply" not in tf_subcommands(env)


def test_up_all_runs_every_stage_but_init(env):
    proc = env.script("up.sh", "all", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    chdirs = {a[0] for a in env.argvs("terraform")}
    assert chdirs == {tf_base_dir(env)}, "make up must not re-apply the bootstrap stack (make init does)"
    assert non_tf(env) == base_sequence(env)


@pytest.mark.parametrize("args", [[], ["nosuch"], ["base", "extra"]])
def test_up_usage(env, args):
    proc = env.script("up.sh", *args)
    assert proc.returncode == 2 and "usage" in proc.stderr and env.calls() == []


def test_up_requires_a_private_env(env):
    env.write_env(f"GCP_PROJECT={PROJECT}\n", mode=0o644)
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode == 2 and env.calls() == []


# ---- Makefile --------------------------------------------------------------------------------

@pytest.mark.parametrize("target,command", [
    ("init", "scripts/up.sh init"), ("up", "scripts/up.sh all"), ("kubeconfig", "scripts/kubeconfig.sh"),
    ("tunnel", "scripts/tunnel.sh start"), ("tunnel-stop", "scripts/tunnel.sh stop")])
def test_make_targets(target, command):
    proc = subprocess.run(["make", "--no-print-directory", "-n", "-f", str(ROOT / "Makefile"), target],
                          capture_output=True, text=True, cwd="/")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{ROOT}/{command}"
