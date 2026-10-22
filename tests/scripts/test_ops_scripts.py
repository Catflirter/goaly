"""Offline tests for scripts/lib/common.sh (Task 10 helpers), scripts/up.sh, kubeconfig.sh, tunnel.sh.

The scripts run from a copy of the repo layout with terraform, gcloud and kubectl replaced by
tests/scripts/fake_tools.py on PATH. The fakes record every call, so the tests assert the exact
commands and flags the scripts would run. Nothing here reaches GCP, a cluster or the network.
"""
import contextlib
import json
import os
import select
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpuctl.infra import Infra, SubprocessShell

ROOT = Path(__file__).resolve().parents[2]
FAKES = Path(__file__).with_name("fake_tools.py")
PROJECT = "k3s-demo-123456"
BUCKET = f"{PROJECT}-k3s-tfstate"
SCRIPTS = ["up.sh", "kubeconfig.sh", "tunnel.sh", "lib/common.sh", "join-node.sh", "cpu-pool.sh"]
TOOLS = ("terraform", "gcloud", "kubectl", "helm", "ssh", "scp")
OUTPUTS = {"base": {"cp_zone": "us-west1-b", "cp_ip": "34.82.10.10", "cp_internal_ip": "10.10.0.2"},
           "bootstrap": {"state_bucket": BUCKET},
           "workers": {"worker_ips": {"k3s-cpu-1": "34.10.0.11"}, "worker_zones": {"k3s-cpu-1": "us-central1-a"}},
           "gpu-access": {"gpu_ips": ["129.80.1.2"]}}
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
        for stack in ("bootstrap", "base", "workers", "gpu-access"):
            (self.repo / "infra" / stack).mkdir(parents=True)
        shutil.copy2(ROOT / "infra/workers/workers.auto.tfvars.json", self.tfvars)
        shutil.copytree(ROOT / "node", self.repo / "node")
        shutil.copytree(ROOT / "k8s", self.repo / "k8s")
        self.home.mkdir()
        self.bin.mkdir()
        for tool in TOOLS:
            (self.bin / tool).write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKES}" {tool} "$@"\n')
            (self.bin / tool).chmod(0o755)
        # scripts/gpuctl (sync-peers, ip) is the fake too: it would reach Lambda and the cluster.
        (self.repo / "scripts/gpuctl").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKES}" gpuctl "$@"\n')
        (self.repo / "scripts/gpuctl").chmod(0o755)
        self.ssh_key = self.home / ".ssh/lambda"
        self.ssh_key.parent.mkdir(mode=0o700)
        self.ssh_key.write_text("not a key\n")
        self.ssh_key.chmod(0o600)
        self.log = tmp / "calls.jsonl"
        self.scenario_file = tmp / "scenario.json"
        self.tmpdir = tmp / "tmp"  # TMPDIR of the scripts, so tests can see a leaked temp file
        self.tmpdir.mkdir()
        self.scenario = {"outputs": OUTPUTS, "kubeconfig": KUBECONFIG}
        self.port = free_port()
        self.write_env(f"GCP_PROJECT={PROJECT}\nLAMBDA_API_KEY=x\nUNDERLAY_MTU=1460\n"
                       f"SSH_PRIVATE_KEY_PATH={self.ssh_key}\n")

    @property
    def tfvars(self):
        return self.repo / "infra/workers/workers.auto.tfvars.json"

    def write_env(self, text, mode=0o600):
        env = self.repo / ".env"
        env.write_text(text)
        env.chmod(mode)

    def run(self, args, stdin="", **extra):
        return subprocess.run(args, cwd=self.repo, env=self.environ(extra), input=stdin, capture_output=True,
                              text=True, timeout=60)

    def popen(self, args, **extra):
        return subprocess.Popen(args, cwd=self.repo, env=self.environ(extra), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def environ(self, extra):
        self.scenario_file.write_text(json.dumps(self.scenario))
        env = {k: v for k, v in os.environ.items()
               if k not in ("AUTO_APPROVE", "GCP_PROJECT", "UNDERLAY_MTU", "KUBECONFIG", "REPO_ROOT",
                            "LAMBDA_API_KEY", "TUNNEL_TIMEOUT_S", "GPU_SSH_IP", "SSH_PRIVATE_KEY_PATH",
                            "CPU_ZONE", "CPU_TYPE", "WORKERS_TFVARS")
               and not k.startswith("TF_VAR_")}
        env.update(PATH=f"{self.bin}:{os.environ['PATH']}", HOME=str(self.home), FAKE_LOG=str(self.log),
                   FAKE_SCENARIO=str(self.scenario_file), WAIT_INTERVAL_S="0",
                   K3S_TUNNEL_TEST_PORT=str(self.port), TMPDIR=str(self.tmpdir))
        env.update(extra)
        for tool in TOOLS:  # never the real CLIs
            assert shutil.which(tool, path=env["PATH"]) == str(self.bin / tool)
        return env

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
        """Spec §8's -L 6443:127.0.0.1:6443, with the test port in place of the local 6443."""
        return f"{self.port}:127.0.0.1:6443"

    @property
    def lockfile(self):
        return self.repo / ".run/tunnel.lock"

    def assert_no_lambda_key(self):
        """No ops script needs the Lambda API key: terraform, gcloud and kubectl never get it (X0-3)."""
        calls = self.calls()
        assert calls and all("LAMBDA_API_KEY" not in c["env"] for c in calls)


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


def test_load_env_exports_the_terraform_variables(env):
    """Brief interface: common.sh exports both, so a later caller that runs terraform directly after
    `source common.sh; load_env` (Task 20's down.sh) has them (S0-3)."""
    proc = env.lib("load_env; printenv TF_VAR_project_id TF_VAR_state_bucket")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == [PROJECT, BUCKET]


def test_load_env_with_a_bad_project_exports_nothing_and_does_not_fail(env):
    """gpuctl commands that never touch GCP must keep working; tf_env still refuses the value."""
    env.write_env("GCP_PROJECT='Not A Project'\n")
    proc = env.lib('load_env; echo "${TF_VAR_project_id-unset} ${TF_VAR_state_bucket-unset}"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "unset unset"


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


def test_tf_init_reruns_after_a_failed_init(env):
    """terraform writes .terraform/terraform.tfstate before it installs providers: a failed or
    interrupted init must not count as initialised (X0-1)."""
    env.scenario["init_rc"] = {"base": 1}
    proc = env.lib("load_env; tf_init base; echo reached")
    assert proc.returncode != 0 and "reached" not in proc.stdout
    assert "init" in proc.stderr and "base" in proc.stderr
    assert (env.repo / "infra/base/.terraform/terraform.tfstate").exists(), "the fake behaves like terraform"
    env.scenario["init_rc"] = {}
    assert env.lib("load_env; tf_init base").returncode == 0
    assert env.lib("load_env; tf_init base").returncode == 0
    assert tf_subcommands(env) == ["init", "init"], "re-initialised once after the failure, then skipped"


def test_tf_init_reruns_when_the_committed_lock_file_changed(env):
    """A pulled provider bump changes infra/STACK/.terraform.lock.hcl: the next tf_init must install
    it, or every plan fails with 'Inconsistent dependency lock file' until a manual init (X1-2)."""
    lockfile = env.repo / "infra/base/.terraform.lock.hcl"
    lockfile.write_text('provider "registry.terraform.io/hashicorp/google" {\n  version = "6.0.0"\n}\n')
    assert env.lib("load_env; tf_init base").returncode == 0
    assert env.lib("load_env; tf_init base").returncode == 0
    assert tf_subcommands(env) == ["init"], "an unchanged lock file is still skipped"
    lockfile.write_text('provider "registry.terraform.io/hashicorp/google" {\n  version = "6.1.0"\n}\n')
    assert env.lib("load_env; tf_init base").returncode == 0
    assert env.lib("load_env; tf_init base").returncode == 0
    assert tf_subcommands(env) == ["init", "init"]
    assert env.argvs("terraform")[-1][2:4] == ["-input=false", "-lockfile=readonly"]
    lockfile.unlink()
    assert env.lib("load_env; tf_init base").returncode == 0
    assert tf_subcommands(env) == ["init", "init", "init"], "a removed lock file counts as a change"


def test_tf_out_reinitialises_a_stack_whose_init_failed(env):
    env.scenario["init_rc"] = {"base": 1}
    assert env.lib("load_env; tf_init base").returncode != 0
    env.scenario["init_rc"] = {}
    proc = env.lib("load_env; tf_out base cp_ip")
    assert proc.returncode == 0 and proc.stdout == "34.82.10.10"
    assert tf_subcommands(env) == ["init", "init", "output"]


def test_tf_init_reinitialises_when_the_project_changed(env):
    assert env.lib("load_env; tf_init base").returncode == 0
    env.write_env("GCP_PROJECT=other-project-42\n")
    assert env.lib("load_env; tf_init base").returncode == 0
    inits = env.argvs("terraform")
    assert len(inits) == 2 and "-backend-config=bucket=other-project-42-k3s-tfstate" in inits[1]


# tf_init and gpuctl's Infra.ensure_init share infra/<stack>/.terraform/k3s-tf-init.ok (MK-1). If they disagree on
# its text, each runs `terraform init -reconfigure` on the stack the other set up (and gpuctl, should that init fail,
# deletes a .terraform a concurrent `make up` is using). These run both real implementations against the fake
# terraform, so the only terraform calls are the ones each side really makes.

def gpuctl_infra(env):
    """gpuctl's real adapter for env.repo, its terraform the same fake the scripts get (no network, no backend)."""
    return Infra(SubprocessShell(), env.repo, {**env.environ({}), "GCP_PROJECT": PROJECT})


def with_lock_file(env, lock):
    if lock:
        shutil.copy2(ROOT / "infra/base/.terraform.lock.hcl", env.repo / "infra/base/.terraform.lock.hcl")


@pytest.mark.parametrize("lock", [True, False], ids=["lock-file", "no-lock-file"])
def test_gpuctl_trusts_a_stack_tf_init_initialised(env, lock):
    with_lock_file(env, lock)
    assert env.lib("load_env; tf_init base").returncode == 0
    assert tf_subcommands(env) == ["init"]
    gpuctl_infra(env).ensure_init("base")
    assert tf_subcommands(env) == ["init"], "gpuctl re-initialised a stack tf_init had just initialised"


@pytest.mark.parametrize("lock", [True, False], ids=["lock-file", "no-lock-file"])
def test_tf_init_trusts_a_stack_gpuctl_initialised(env, lock):
    with_lock_file(env, lock)
    gpuctl_infra(env).ensure_init("base")
    assert tf_subcommands(env) == ["init"]
    assert env.lib("load_env; tf_init base").returncode == 0
    assert tf_subcommands(env) == ["init"], "tf_init re-initialised a stack gpuctl had just initialised"


def test_both_sides_reinitialise_once_after_a_lock_file_change(env):  # X1-2 holds across the two implementations
    with_lock_file(env, True)
    assert env.lib("load_env; tf_init base").returncode == 0
    lockfile = env.repo / "infra/base/.terraform.lock.hcl"
    bumped = lockfile.read_text().replace('version     = "8.7.0"', 'version     = "8.7.1"')
    assert bumped != lockfile.read_text()
    lockfile.write_text(bumped)
    gpuctl_infra(env).ensure_init("base")
    assert tf_subcommands(env) == ["init", "init"], "gpuctl must install the pulled provider bump"
    assert env.lib("load_env; tf_init base").returncode == 0
    lockfile.write_text(lockfile.read_text() + "\n")
    assert env.lib("load_env; tf_init base").returncode == 0
    gpuctl_infra(env).ensure_init("base")
    assert tf_subcommands(env) == ["init", "init", "init"]


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


def test_cp_scp_copies_several_sources_into_one_destination(env):
    assert env.lib("load_env; cp_scp node/k3s node/guard /tmp/k3s-node").returncode == 0
    assert env.argvs("gcloud") == [[
        "compute", "scp", "--project", PROJECT, "--zone", "us-west1-b", "--tunnel-through-iap", "--quiet",
        "--recurse", "--scp-flag=-oConnectTimeout=20", "node/k3s", "node/guard", "k3s-cp:/tmp/k3s-node"]]


def test_cp_scp_needs_a_source_and_a_destination(env):
    proc = env.lib("load_env; cp_scp /tmp/k3s-node")
    assert proc.returncode != 0 and env.calls("gcloud") == []


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
    # Each attempt runs in a subshell, so the attempts are counted in a file, not a variable.
    proc = env.lib('f="$TMPDIR/tries"; : > "$f"; wait_for "echo . >> $f; (( \\$(wc -l < $f) >= 3 ))" 30; '
                   'echo "tries=$(( $(wc -l < "$f") ))"; rm -f "$f"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "tries=3"
    assert list(env.tmpdir.iterdir()) == []


def test_wait_for_shows_a_die_inside_the_command_and_cleans_up(env):
    """A CMD that dies (cp_ssh -> tf_out -> die) is a failed attempt, not a silent exit of the
    whole script with its message hidden in a leaked temp file (C0-1)."""
    proc = env.lib('if wait_for "die boom" 1; then echo ok; else echo timeout; fi', WAIT_INTERVAL_S="1")
    assert proc.stdout.strip() == "timeout"
    assert "error: boom" in proc.stderr
    assert list(env.tmpdir.iterdir()) == [], "the attempt log is removed"


def test_wait_for_gives_up_after_the_timeout(env):
    start = time.time()
    proc = env.lib('if wait_for "echo still-down >&2; false" 1; then echo ok; else echo timeout; fi',
                   WAIT_INTERVAL_S="1")
    assert proc.stdout.strip() == "timeout" and time.time() - start < 10
    assert "still-down" in proc.stderr, "the last attempt's output helps the operator"


def test_wait_for_stops_at_once_on_the_stop_code(env):
    """wait_for CMD TIMEOUT STOP_RC: an attempt that exits STOP_RC can never succeed (M4-2: a rejected
    host key); the wait ends at once and returns STOP_RC, showing that attempt's output."""
    start = time.time()
    proc = env.lib('f="$TMPDIR/tries"; : > "$f"; rc=0; wait_for "echo . >> $f; echo key-rejected >&2; exit 7" 30 7'
                   ' || rc=$?; echo "rc=$rc tries=$(( $(wc -l < "$f") ))"; rm -f "$f"', WAIT_INTERVAL_S="1")
    assert proc.stdout.strip() == "rc=7 tries=1" and time.time() - start < 10
    assert "key-rejected" in proc.stderr
    assert list(env.tmpdir.iterdir()) == []


def test_wait_for_retries_other_failures_when_given_a_stop_code(env):
    proc = env.lib('f="$TMPDIR/tries"; : > "$f"; wait_for "echo . >> $f; (( \\$(wc -l < $f) >= 3 ))" 30 7; '
                   'echo "tries=$(( $(wc -l < "$f") ))"; rm -f "$f"', WAIT_INTERVAL_S="0")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "tries=3"


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


def test_tf_apply_auto_approve_applies_a_destroying_plan_but_lists_it(env):
    """Brief contract (S0-1): under AUTO_APPROVE=1 confirm passes outright, so Task 14's
    `AUTO_APPROVE=1 scripts/cpu-pool.sh scale 1` can remove a worker with a plain tf_apply."""
    env.scenario["deletes"] = {"workers": ["google_compute_instance.cpu[1]"]}
    (env.repo / "infra/workers").mkdir(exist_ok=True)
    proc = env.lib("load_env; tf_apply workers", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert tf_subcommands(env)[-1] == "apply"
    assert "DESTROYS or REPLACES" in proc.stderr and "google_compute_instance.cpu[1]" in proc.stderr


def test_tf_apply_interactive_shows_what_it_destroys_and_still_asks(env):
    env.scenario["deletes"] = {"base": ["google_compute_instance.cp"]}
    proc = env.lib("load_env; tf_apply base", stdin="y\n")
    assert proc.returncode == 0 and tf_subcommands(env)[-1] == "apply"
    assert "google_compute_instance.cp" in proc.stderr and "[y/N]" in proc.stderr


def test_tf_apply_no_destroy_refuses_a_destroying_plan_under_auto_approve(env):
    env.scenario["deletes"] = {"base": ["google_compute_instance.cp"]}
    proc = env.lib("load_env; tf_apply base --no-destroy", AUTO_APPROVE="1")
    assert proc.returncode == 2 and "apply" not in tf_subcommands(env)
    assert "google_compute_instance.cp" in proc.stderr
    assert not (env.repo / ".run/base.tfplan").exists()


def test_tf_apply_no_destroy_still_asks_interactively(env):
    env.scenario["deletes"] = {"base": ["google_compute_instance.cp"]}
    proc = env.lib("load_env; tf_apply base --no-destroy", stdin="y\n")
    assert proc.returncode == 0 and tf_subcommands(env)[-1] == "apply"
    assert "[y/N]" in proc.stderr


def test_tf_apply_no_destroy_lets_a_plan_without_deletes_through(env):
    proc = env.lib("load_env; tf_apply base --no-destroy", AUTO_APPROVE="1")
    assert proc.returncode == 0 and tf_subcommands(env)[-1] == "apply"


def test_tf_apply_refuses_an_unknown_option(env):
    proc = env.lib("load_env; tf_apply base --allow-delete", AUTO_APPROVE="1")
    assert proc.returncode != 0 and env.calls("terraform") == []


def test_tf_apply_plan_confirmed_applies_without_asking_again(env):
    """S3-m1: a caller that asked the question itself (cpu-pool.sh --guarded) applies the saved plan with
    --confirmed: no second question, and no AUTO_APPROVE line for the operator's own yes (S3-L3)."""
    proc = env.lib("load_env; tf_init base; tf_plan base; tf_apply_plan base --confirmed")  # empty stdin
    assert proc.returncode == 0, proc.stderr
    assert tf_subcommands(env) == ["init", "plan", "show", "apply"]
    assert "[y/N]" not in proc.stderr and "AUTO_APPROVE" not in proc.stderr
    assert not (env.repo / ".run/base.tfplan").exists()


def test_tf_apply_plan_refuses_an_unknown_option(env):
    proc = env.lib("load_env; tf_init base; tf_plan base; tf_apply_plan base --yes", AUTO_APPROVE="1")
    assert proc.returncode != 0 and "apply" not in tf_subcommands(env)


# ---- one Terraform run per stack (C-1: stack_lock) ---------------------------------------------

def lock_free(path):
    """Nobody holds PATH with flock(1)."""
    return subprocess.run(["flock", "-n", str(path), "true"]).returncode == 0


@contextlib.contextmanager
def lock_holder(path):
    """Another run holds PATH: `flock -o PATH sleep 60` in its own process group. -o: sleep does not inherit the lock,
    so once the flock process is gone (reaped) the lock is free, whatever happens to sleep (S3-m3)."""
    path.parent.mkdir(exist_ok=True)
    holder = subprocess.Popen(["flock", "-o", str(path), "sleep", "60"], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while lock_free(path):
            assert time.monotonic() < deadline and holder.poll() is None, "the holder never took the lock"
            time.sleep(0.02)
        yield holder
    finally:
        if holder.poll() is None:
            os.killpg(holder.pid, signal.SIGKILL)
        holder.wait()


def path_without_flock(env, tmp_path):
    """env's PATH with the fakes first, then every other directory's commands linked into one, flock left out."""
    dirs = [d for d in env.environ({})["PATH"].split(":") if d != str(env.bin)]
    tools = tmp_path / "no-flock-bin"
    tools.mkdir()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            target = tools / name
            if name != "flock" and not target.exists() and os.access(os.path.join(d, name), os.X_OK):
                target.symlink_to(os.path.join(d, name))
    path = f"{env.bin}:{tools}"
    assert shutil.which("flock", path=path) is None
    return path


def gate_terraform(env, sub):
    """terraform's SUB (plan, apply) waits while $FAKE_LOG.gate exists, after touching $FAKE_LOG.at_SUB: one run can be
    held inside its plan while a test starts another."""
    gate, at = Path(f"{env.log}.gate"), Path(f"{env.log}.at_{sub}")
    (env.bin / "terraform").write_text(
        f'#!/bin/sh\nif [ "$2" = {sub} ] && [ -e "{gate}" ]; then : > "{at}"; while [ -e "{gate}" ]; do sleep 0.05; done; fi\n'
        f'exec "{sys.executable}" "{FAKES}" terraform "$@"\n')
    gate.write_text("")
    return gate, at


def wait_for_file(path, proc, seconds=30):
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert proc.poll() is None, proc.communicate()
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)


def test_a_second_run_on_a_stack_is_refused_without_touching_terraform(env):
    gate, at_plan = gate_terraform(env, "plan")
    first = env.popen(["bash", "-c", "set -euo pipefail; source scripts/lib/common.sh; load_env; tf_init base;"
                       " tf_apply base"], AUTO_APPROVE="1")
    try:
        wait_for_file(at_plan, first)  # the first run is inside its plan, so it holds the base stack's lock
        before = len(env.calls("terraform"))
        second = env.lib("load_env; tf_init base; tf_apply base", AUTO_APPROVE="1")
        assert second.returncode == 2, second.stderr
        assert "base" in second.stderr and str(env.repo.resolve() / ".run/base.lock") in second.stderr
        assert len(env.calls("terraform")) == before, "the refused run ran no terraform at all (its init was done)"
    finally:
        gate.unlink()
        out, err = first.communicate(timeout=60)
    assert first.returncode == 0, err
    assert tf_subcommands(env) == ["init", "plan", "show", "apply"]
    assert lock_free(env.repo / ".run/base.lock"), "released when the first run exited"


@pytest.mark.parametrize("call", ["tf_apply workers", "tf_plan workers", "stack_lock workers"])
def test_a_held_stack_lock_refuses_naming_the_stack_and_the_file(env, call):
    lock = env.repo / ".run/workers.lock"
    with lock_holder(lock):
        proc = env.lib(f"load_env; {call}", AUTO_APPROVE="1")
    assert proc.returncode == 2
    assert "workers stack" in proc.stderr and str(lock.resolve()) in proc.stderr
    assert env.calls("terraform") == []


def test_a_held_lock_of_another_stack_does_not_block(env):
    with lock_holder(env.repo / ".run/workers.lock"):
        proc = env.lib("load_env; tf_init base; tf_apply base", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert tf_subcommands(env)[-1] == "apply"


def test_stack_lock_is_reentrant_and_held_until_the_process_exits(env):
    lock = env.repo / ".run/base.lock"
    proc = env.lib(f'load_env; stack_lock base; stack_lock base; tf_init base; tf_apply base; stack_lock base;'
                   f' if flock -n "{lock}" true; then echo free; else echo held; fi', AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["held"], "taken once, kept through tf_apply, released only on exit"
    assert tf_subcommands(env)[-1] == "apply"
    assert lock_free(lock)


def test_stack_lock_refuses_a_bad_stack_name(env):
    for name in ("../x", "", "Base", "base/x"):
        proc = env.lib(f"stack_lock '{name}'")
        assert proc.returncode != 0 and "stack" in proc.stderr
    assert not (env.repo / "x.lock").exists() and not list(env.repo.rglob("*.lock"))


def test_without_flock_terraform_runs_unlocked_after_one_warning(env, tmp_path):
    proc = env.lib("load_env; stack_lock base; tf_init base; tf_apply base; tf_init workers; tf_apply workers",
                   AUTO_APPROVE="1", PATH=path_without_flock(env, tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert tf_subcommands(env).count("apply") == 2
    assert proc.stderr.count("WARNING") == 1 and "flock" in proc.stderr


def test_stack_lock_required_refuses_without_flock(env, tmp_path):
    proc = env.lib("stack_lock --required workers", PATH=path_without_flock(env, tmp_path))
    assert proc.returncode == 2 and "flock" in proc.stderr


@pytest.mark.parametrize("version, ok", [(("3", "2"), False), (("4", "0"), False), (("4", "1"), True),
                                         (("4", "4"), True), (("5", "0"), True), (("10", "0"), True)])
def test_bash_version_check(env, version, ok):
    """exec {fd}> needs bash >= 4.1 (macOS ships 3.2)."""
    assert (env.lib(f"_bash_version_ok {' '.join(version)}").returncode == 0) == ok
    assert env.lib("_bash_version_ok").returncode == 0, "the bash running these tests"


def test_stack_lock_checks_the_bash_version_before_it_opens_a_descriptor():
    text = (ROOT / "scripts/lib/common.sh").read_text()
    body = text[text.index("\nstack_lock() {"):]
    body = body[:body.index("\n}\n")]
    assert "_bash_version_ok \\\n    || die" in body
    assert body.index("_bash_version_ok") < body.index("exec {")


def lock_probe_terraform(env):
    """terraform plan/apply first record whether their stack's lock (.run/<stack>.lock) is held: STACK SUB held|free."""
    record = Path(f"{env.log}.locks")
    (env.bin / "terraform").write_text(
        '#!/bin/sh\nstack="${1##*/}"\n'
        f'if [ "$2" = plan ] || [ "$2" = apply ]; then\n'
        f'  if flock -n "{env.repo}/.run/$stack.lock" true; then s=free; else s=held; fi\n'
        f'  echo "$stack $2 $s" >> "{record}"\nfi\n'
        f'exec "{sys.executable}" "{FAKES}" terraform "$@"\n')
    return record


def test_up_base_and_workers_stages_hold_their_stacks_locks(env):
    record = lock_probe_terraform(env)
    proc = env.script("up.sh", "all", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert record.read_text().splitlines() == ["base plan held", "base apply held",
                                               "workers plan held", "workers apply held"]
    # The workers stack's lock is cpu-pool.sh's: one lock for every run that plans or applies the workers stack.
    assert (env.repo / ".run/workers.lock").exists()
    # Released when up.sh exits, although the API tunnel it started runs on (it must not inherit the lock).
    assert alive(int(env.pidfile.read_text()))
    assert lock_free(env.repo / ".run/base.lock") and lock_free(env.repo / ".run/workers.lock")


def test_up_init_holds_the_bootstrap_stacks_lock(env):
    record = lock_probe_terraform(env)
    proc = env.script("up.sh", "init", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert record.read_text().splitlines() == ["bootstrap plan held", "bootstrap apply held"]


def test_up_base_is_refused_while_another_run_holds_the_base_stack(env):
    with lock_holder(env.repo / ".run/base.lock"):
        proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode == 2 and "base stack" in proc.stderr
    assert "plan" not in tf_subcommands(env) and env.calls("gcloud") == []


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
    env.assert_no_lambda_key()


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
    """The brief's ssh arguments verbatim, then -o GatewayPorts=no (S0-2)."""
    return ["compute", "ssh", "k3s-cp", "--project", PROJECT, "--zone", "us-west1-b", "--tunnel-through-iap",
            "--quiet", "--", "-N", "-L", env.forward(), "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30", "-o", "GatewayPorts=no"]


def tunnels(env):
    return [a for a in env.argvs("gcloud") if "-N" in a]


def gate_readyz(env):
    """kubectl's /readyz probes wait while $FAKE_LOG.gate exists, after touching $FAKE_LOG.at_readyz: a tunnel.sh start
    can be held inside its first probe, its lock taken and its tunnel launched."""
    gate, at = Path(f"{env.log}.gate"), Path(f"{env.log}.at_readyz")
    (env.bin / "kubectl").write_text(
        f'#!/bin/sh\ncase "$*" in *--raw*/readyz*) if [ -e "{gate}" ]; then : > "{at}";'
        f' while [ -e "{gate}" ]; do sleep 0.05; done; fi;; esac\n'
        f'exec "{sys.executable}" "{FAKES}" kubectl "$@"\n')
    gate.write_text("")
    return gate, at


def read_stderr_until(proc, text, seconds=60):
    """What PROC wrote to stderr up to and including TEXT (which it must write while it runs)."""
    err, deadline = b"", time.monotonic() + seconds
    while text.encode() not in err:
        ready, _, _ = select.select([proc.stderr], [], [], max(0.0, deadline - time.monotonic()))
        assert ready, f"{text!r} never came; stderr so far: {err.decode()}"
        chunk = os.read(proc.stderr.fileno(), 4096)
        assert chunk, f"exited before {text!r}; stderr: {err.decode()}"
        err += chunk
    return err.decode()


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
    assert "waiting up to 30s for the API's /readyz" in proc.stderr, "brief: default wait 30s"
    env.assert_no_lambda_key()  # also the long-lived tunnel's environment
    assert lock_free(env.lockfile), "the start/stop lock is released, and the tunnel does not hold it"

    assert env.script("tunnel.sh", "status").returncode == 0

    # start again: already up, no second tunnel
    again = env.script("tunnel.sh", "start")
    assert again.returncode == 0 and len(tunnels(env)) == 1 and int(env.pidfile.read_text()) == pid

    stop = env.script("tunnel.sh", "stop")
    assert stop.returncode == 0, stop.stderr
    assert wait_gone(pid) and not env.pidfile.exists()
    assert env.script("tunnel.sh", "status").returncode == 1
    assert env.script("tunnel.sh", "stop").returncode == 0, "stop is idempotent"


def test_tunnel_forward_stays_on_loopback_whatever_the_ssh_config_says(env, tmp_path):
    """Spec §8's -L 6443:127.0.0.1:6443 names no bind address, so with `GatewayPorts yes` in the
    operator's ssh config ssh would listen on every interface and open the cluster-admin API to the
    LAN. tunnel.sh adds -o GatewayPorts=no, and ssh takes a command-line option over any config file
    (S0-2). Checked with the real ssh client: -G prints the configuration it would use and connects
    to nothing. The arguments go after the host, where gcloud puts them."""
    ssh = shutil.which("ssh")
    if ssh is None:
        pytest.skip("no ssh client on PATH")
    make_kubeconfig(env)
    assert env.script("tunnel.sh", "start").returncode == 0
    [argv] = tunnels(env)
    config = tmp_path / "ssh_config"
    config.write_text("Host *\n  GatewayPorts yes\n")
    config.chmod(0o600)

    def effective(ssh_args):
        proc = subprocess.run([ssh, "-G", "-F", str(config), "k3s-cp", *ssh_args],
                              capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.splitlines()

    brief_only = ["-N", "-L", env.forward(), "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30"]
    assert "gatewayports yes" in effective(brief_only), "the config does turn it on for the brief's arguments"
    used = effective(argv[argv.index("--") + 1:])
    assert "gatewayports no" in used
    assert f"localforward {env.port} [127.0.0.1]:6443" in used


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


def test_tunnel_never_kills_another_forward_named_in_the_pidfile(env):
    """Our forward PORT:127.0.0.1:6443 is the tail of others, such as 1PORT:127.0.0.1:6443 (an
    operator's own tunnel to another cluster): a recycled PID that runs one is not ours either."""
    other = subprocess.Popen(["bash", "-c", "sleep 30; :", "other-tunnel", "-N", "-L",
                              f"1{env.port}:127.0.0.1:6443", "-o", "ExitOnForwardFailure=yes"],
                             start_new_session=True)
    try:
        env.pidfile.parent.mkdir()
        env.pidfile.write_text(f"{other.pid}\n")
        assert env.script("tunnel.sh", "stop").returncode == 0
        assert other.poll() is None, "another forward's process must not be killed"
        assert env.script("tunnel.sh", "status").returncode == 1
    finally:
        try:
            os.killpg(other.pid, signal.SIGKILL)  # bash and its sleep
        except ProcessLookupError:
            pass
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


def test_concurrent_starts_launch_one_tunnel(env):
    """`make tunnel` while `make up` is in its tunnel step: without a lock both launch, the loser's
    cleanup drops the PID file and the winner's tunnel is orphaned (C0-3)."""
    make_kubeconfig(env)
    start = [str(env.repo / "scripts/tunnel.sh"), "start"]
    procs = [env.popen(start), env.popen(start)]
    results = [p.communicate(timeout=60) for p in procs]
    assert [p.returncode for p in procs] == [0, 0], results
    assert len(tunnels(env)) == 1, results
    pid = int(env.pidfile.read_text())
    assert alive(pid) and int(Path(f"{env.log}.tunnel").read_text()) == pid
    assert lock_free(env.lockfile), "released by both runs, and never held by the tunnel they started"


@pytest.mark.parametrize("left_behind", ["", "4242\n", "not a pid\n"])
def test_concurrent_starts_with_a_lock_file_left_behind_launch_one_tunnel(env, left_behind):
    """C-2: the lock is flock(1) on .run/tunnel.lock. Whatever an earlier (killed) run left in the file, nothing
    has to break a stale lock, so four waiting runs still start exactly one tunnel."""
    make_kubeconfig(env)
    env.lockfile.parent.mkdir()
    env.lockfile.write_text(left_behind)
    start = [str(env.repo / "scripts/tunnel.sh"), "start"]
    procs = [env.popen(start) for _ in range(4)]
    results = [p.communicate(timeout=60) for p in procs]
    assert [p.returncode for p in procs] == [0, 0, 0, 0], results
    assert len(tunnels(env)) == 1, results
    pid = int(env.pidfile.read_text())
    assert alive(pid) and int(Path(f"{env.log}.tunnel").read_text()) == pid
    assert not any("stale" in err for _, err in results)
    assert not Path(f"{env.lockfile}.break").exists()


def test_tunnel_start_waits_for_a_live_holder(env):
    make_kubeconfig(env)
    with lock_holder(env.lockfile) as holder:
        env.lockfile.write_text(f"{holder.pid}\n")  # what a holding tunnel.sh writes: its PID, for this message
        proc = env.popen([str(env.repo / "scripts/tunnel.sh"), "start"])
        err = read_stderr_until(proc, "waiting for another tunnel.sh")
        assert proc.poll() is None and tunnels(env) == [], "waits while a live holder has the lock"
    _, rest = proc.communicate(timeout=60)
    assert proc.returncode == 0, err + rest
    assert f"waiting for another tunnel.sh (pid {holder.pid}) to finish" in err and len(tunnels(env)) == 1
    assert env.lockfile.read_text() == f"{proc.pid}\n", "the new holder wrote its own PID"


def test_a_lock_file_naming_a_live_unrelated_pid_never_blocks(env):
    """C-2: after a reboot the PID in .run/tunnel.lock can belong to another process. Only a held flock(1) counts."""
    make_kubeconfig(env)
    other = subprocess.Popen(["sleep", "60"])
    try:
        env.lockfile.parent.mkdir()
        env.lockfile.write_text(f"{other.pid}\n")
        for action in ("stop", "start"):
            proc = env.script("tunnel.sh", action)
            assert proc.returncode == 0, proc.stderr
            assert "waiting for another tunnel.sh" not in proc.stderr
        assert len(tunnels(env)) == 1
    finally:
        other.kill()
        other.wait()


def test_a_killed_holder_never_blocks_the_next_run(env):
    """tunnel.sh SIGKILLed while it holds the lock (inside start, its tunnel launched): the kernel released the lock,
    and the tunnel it left running never held it, so the next start takes it at once and reuses that tunnel."""
    make_kubeconfig(env)
    gate, at = gate_readyz(env)
    first = env.popen([str(env.repo / "scripts/tunnel.sh"), "start"])
    wait_for_file(at, first)  # the first run holds the lock and probes its tunnel
    assert not lock_free(env.lockfile)
    first.kill()
    gate.unlink()  # the killed run's probe (it inherited the descriptor and stderr too) ends now
    first.communicate(timeout=60)
    deadline = time.monotonic() + 30
    while not lock_free(env.lockfile):
        assert time.monotonic() < deadline, "the orphaned tunnel must not hold the lock"
        time.sleep(0.05)
    proc = env.script("tunnel.sh", "start")
    assert proc.returncode == 0, proc.stderr
    assert "waiting for another tunnel.sh" not in proc.stderr and "already running" in proc.stderr
    assert len(tunnels(env)) == 1


def test_tunnel_stop_during_a_start_waits_then_stops_the_new_tunnel(env):
    make_kubeconfig(env)
    gate, at = gate_readyz(env)
    start = env.popen([str(env.repo / "scripts/tunnel.sh"), "start"])
    wait_for_file(at, start)  # start holds the lock, its tunnel launched, its first probe held at the gate
    stop = env.popen([str(env.repo / "scripts/tunnel.sh"), "stop"])
    err = read_stderr_until(stop, "waiting for another tunnel.sh")
    assert f"(pid {start.pid})" in err
    gate.unlink()
    _, start_err = start.communicate(timeout=60)
    _, stop_err = stop.communicate(timeout=60)
    assert start.returncode == 0, start_err
    assert stop.returncode == 0, err + stop_err
    tunnel = int(Path(f"{env.log}.tunnel").read_text())
    assert f"tunnel stopped (pid {tunnel})" in stop_err
    assert wait_gone(tunnel) and not env.pidfile.exists()
    assert lock_free(env.lockfile)


def test_tunnel_stop_holds_the_lock(env):
    make_kubeconfig(env)
    assert env.script("tunnel.sh", "start").returncode == 0
    record = Path(f"{env.log}.ps_lock")
    real_ps = shutil.which("ps")
    (env.bin / "ps").write_text(f'#!/bin/sh\nif flock -n "{env.lockfile}" true; then echo free; else echo held; fi'
                                f' >> "{record}"\nexec "{real_ps}" "$@"\n')
    (env.bin / "ps").chmod(0o755)
    proc = env.script("tunnel.sh", "stop")
    assert proc.returncode == 0, proc.stderr
    assert record.read_text().splitlines() == ["held"], "stop looks up and kills the tunnel under the lock"


def test_a_run_that_gives_up_waiting_leaves_the_holders_lock_alone(env):
    with lock_holder(env.lockfile) as holder:
        env.lockfile.write_text(f"{holder.pid}\n")
        proc = env.script("tunnel.sh", "stop", TUNNEL_TIMEOUT_S="0")  # waits TUNNEL_TIMEOUT_S + 30 s, then gives up
        assert proc.returncode != 0 and "still holds" in proc.stderr and f"pid {holder.pid}" in proc.stderr
        assert not lock_free(env.lockfile) and env.lockfile.read_text() == f"{holder.pid}\n"


@pytest.mark.parametrize("action", ["start", "stop"])
def test_tunnel_refuses_without_flock(env, tmp_path, action):
    make_kubeconfig(env)
    proc = env.run([str(env.repo / "scripts/tunnel.sh"), action], PATH=path_without_flock(env, tmp_path))
    assert proc.returncode == 2 and "flock" in proc.stderr
    assert tunnels(env) == []


def test_tunnel_sh_has_no_stale_lock_breaking_left():
    text = (ROOT / "scripts/tunnel.sh").read_text()
    assert "break_stale_lock" not in text and ".break" not in text and "noclobber" not in text


def test_tunnel_usage(env):
    proc = env.script("tunnel.sh", "bogus")
    assert proc.returncode == 2 and "usage" in proc.stderr


# ---- up.sh -----------------------------------------------------------------------------------

def base_sequence(env, mtu="1460"):
    """The gcloud/kubectl calls of a full `up.sh base`, without the tunnel and its /readyz probes (non_tf)."""
    return [
        ("gcloud", ["compute", "ssh", "k3s-cp", *SSH_IAP, "--command", "true"]),
        # A fresh directory the operator creates 0700 (mkdir fails if anyone recreated the path after
        # rm), then node/'s entries copied into it: root runs only what we put there (X0-2).
        ("gcloud", ["compute", "ssh", "k3s-cp", *SSH_IAP, "--command",
                    "rm -rf /tmp/k3s-node && mkdir -m 0700 /tmp/k3s-node && [ ! -L /tmp/k3s-node ]"
                    " && [ -O /tmp/k3s-node ]"]),
        ("gcloud", ["compute", "scp", "--project", PROJECT, "--zone", "us-west1-b", "--tunnel-through-iap",
                    "--quiet", "--recurse", "--scp-flag=-oConnectTimeout=20",
                    *sorted(str(p) for p in (env.repo.resolve() / "node").iterdir()), "k3s-cp:/tmp/k3s-node"]),
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


def non_tf(env, proc):
    """gcloud/kubectl calls in order, without the tunnel and the /readyz probes: the backgrounded
    fake tunnel logs its call concurrently with tunnel.sh's first probes. What is deterministic is
    checked here: one tunnel, started after the kubeconfig fetch; probes before the first `kc wait`;
    and, in up.sh's own stderr (PROC, the up.sh run), tunnel.sh reporting the API ready before up.sh
    starts that wait. That last check replaces `tunnel_at < probes[-1]` (M-3): fake_tools.py logs a
    probe only once it has its answer and answers ok only while the tunnel runs, so a successful last
    probe always came after the tunnel's line and that assertion could not fail."""
    calls = [(c["tool"], c["argv"]) for c in env.calls() if c["tool"] != "terraform"]
    assert tunnels(env) == [tunnel_argv(env)]
    tunnel_at = calls.index(("gcloud", tunnel_argv(env)))
    probes = [i for i, (_, argv) in enumerate(calls) if argv[2:] == READYZ]
    first_wait = next(i for i, (_, argv) in enumerate(calls) if "wait" in argv)
    before = [argv for _, argv in calls[:tunnel_at] if argv[2:] != READYZ]
    assert before[-1][-1] == "sudo cat /etc/rancher/k3s/k3s.yaml"
    assert probes and probes[-1] < first_wait
    ready = proc.stderr.find("tunnel up (pid ")
    assert 0 <= ready < proc.stderr.index("waiting up to 300s for: kc wait --for=condition=Ready node/k3s-cp"), \
        "up.sh waits for k3s-cp only once tunnel.sh has seen the API answer /readyz"
    return [c for c in calls if c[1][2:] != READYZ and "-N" not in c[1]]


def test_up_base_runs_the_whole_sequence(env):
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    tf = tf_subcommands(env)
    assert tf[:4] == ["init", "plan", "show", "apply"]
    assert env.argvs("terraform")[0][-2:] == [f"-backend-config=bucket={BUCKET}", "-backend-config=prefix=base"]
    assert non_tf(env, proc) == base_sequence(env)
    assert env.kubeconfig.read_text() == KUBECONFIG
    assert alive(int(env.pidfile.read_text()))
    # The brief's waits (C0-2): IAP SSH 300s, k3s-cp Ready 300s, CoreDNS Ready 300s, tunnel 30s.
    for line in ("waiting up to 300s for: cp_ssh true",
                 "waiting up to 300s for: kc wait --for=condition=Ready node/k3s-cp --timeout=10s",
                 "waiting up to 300s for: kc -n kube-system wait --for=condition=Ready pod -l k8s-app=kube-dns"
                 " --timeout=10s",
                 "waiting up to 30s for the API's /readyz"):
        assert line in proc.stderr
    env.assert_no_lambda_key()
    assert list(env.tmpdir.iterdir()) == []


def test_up_base_stops_when_the_staging_directory_cannot_be_made_private(env):
    """Someone else recreated /tmp/k3s-node (mkdir fails, or it is not ours): nothing is copied or run."""
    staging = base_sequence(env)[1][1][-1]
    env.scenario["ssh_fail"] = [staging]
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode != 0
    # The operator learns what to do, not just a bare exit status (X1-1).
    assert "cannot create a private /tmp/k3s-node on k3s-cp" in proc.stderr
    assert "remove it on the CP and re-run make up" in proc.stderr
    assert "SSH to k3s-cp failed" not in proc.stderr
    assert not [a for a in env.argvs("gcloud") if a[:2] == ["compute", "scp"]]
    assert not [a for a in env.argvs("gcloud") if "server-bootstrap.sh" in a[-1]]


def test_up_base_reports_an_ssh_failure_while_staging_as_such(env):
    """ssh's own exit 255 (e.g. IAP dropped the connection) is not blamed on /tmp/k3s-node (X2-1)."""
    staging = base_sequence(env)[1][1][-1]
    env.scenario["ssh_fail"] = [staging]
    env.scenario["ssh_fail_rc"] = 255
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode != 0
    assert "SSH to k3s-cp failed while preparing /tmp/k3s-node (exit 255); re-run make up" in proc.stderr
    assert "cannot create a private /tmp/k3s-node" not in proc.stderr
    assert "remove it on the CP" not in proc.stderr
    assert not [a for a in env.argvs("gcloud") if a[:2] == ["compute", "scp"]]
    assert not [a for a in env.argvs("gcloud") if "server-bootstrap.sh" in a[-1]]


def test_up_base_reports_a_bad_cp_zone_before_waiting_for_ssh(env):
    """cp_zone is resolved outside wait_for: a bad output dies at once with its message (C0-1)."""
    env.scenario["outputs"] = {"base": {**OUTPUTS["base"], "cp_zone": "bad zone"}}
    start = time.time()
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode != 0 and time.time() - start < 30
    assert "unexpected cp_zone output: bad zone" in proc.stderr
    assert "waiting up to" not in proc.stderr and env.calls("gcloud") == []
    assert list(env.tmpdir.iterdir()) == []


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
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert non_tf(env, proc) == base_sequence(env, mtu="1380")


def test_up_base_defaults_an_empty_underlay_mtu(env):
    env.write_env(f"GCP_PROJECT={PROJECT}\nUNDERLAY_MTU=\n")
    proc = env.script("up.sh", "base", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert non_tf(env, proc) == base_sequence(env, mtu="1460")


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
    assert chdirs == {tf_base_dir(env), f"-chdir={env.repo.resolve()}/infra/workers"}, \
        "make up must not re-apply the bootstrap stack (make init does)"
    # Every stage's calls, in order: tests/scripts/test_workers_ops.py (Task 14, Ruling R38).
    assert non_tf(env, proc)[:len(base_sequence(env))] == base_sequence(env)


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
