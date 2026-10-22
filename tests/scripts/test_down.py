"""Offline tests for Task 22a: scripts/down.sh and `make down` (Ruling R44).

Same harness as test_ops_scripts.py: a repo copy whose terraform, gcloud, kubectl, helm, ssh, scp and
scripts/gpuctl are tests/scripts/fake_tools.py, which records every call. Nothing reaches a cloud or a cluster.
"""
import shutil
import stat
import subprocess
import time

import pytest

from test_ops_scripts import BUCKET, KUBECONFIG, PROJECT, ROOT, Env, make_kubeconfig, wait_gone

STACKS = ("gpu-access", "workers", "base")  # brief step 4: destroyed in this order
RT = "--request-timeout=60s"
DRAIN = ["--ignore-daemonsets", "--delete-emptydir-data", "--force", "--timeout=120s", RT]
GPU_LINE = "k3s-gpu-a6000-1 129.80.1.2 active gpu_1x_a6000 us-west-1"
KEY = "lambda-secret-key-do-not-print-0123456789"


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    shutil.copy2(ROOT / "scripts/down.sh", e.repo / "scripts/down.sh")
    e.write_env(f"GCP_PROJECT={PROJECT}\nLAMBDA_API_KEY={KEY}\nUNDERLAY_MTU=1460\n"
                f"SSH_PRIVATE_KEY_PATH={e.ssh_key}\n")
    make_kubeconfig(e)
    run = e.repo / ".run"
    run.mkdir(mode=0o700)
    for name in ("k3s-gpu-a6000-1", "k3s-gpu-a6000-2"):
        (run / f"known_hosts_{name}").write_text("129.80.1.2 ssh-ed25519 AAAA\n")
    (run / "keep-me").write_text("not a known_hosts file\n")
    yield e
    if e.pidfile.exists():  # never leave a fake tunnel behind
        subprocess.run(["kill", "-KILL", "--", f"-{e.pidfile.read_text().strip()}"], capture_output=True)


def down(env, stdin="", **extra):
    return env.script("down.sh", stdin=stdin, **extra)


def tf_dir(env, stack):
    return f"-chdir={env.repo.resolve()}/infra/{stack}"


def init(env, stack):
    return ("terraform", [tf_dir(env, stack), "init", "-input=false", "-lockfile=readonly", "-reconfigure",
                          f"-backend-config=bucket={BUCKET}", f"-backend-config=prefix={stack}"])


def destroy(env, stack):
    return ("terraform", [tf_dir(env, stack), "destroy", "-auto-approve", "-input=false"])


def kc(env, *args):
    return ("kubectl", ["--kubeconfig", str(env.kubeconfig), *args])


def drain(env, name):
    return [kc(env, "drain", name, *DRAIN), kc(env, "delete", "node", name, RT)]


def list_nodes(env):
    return kc(env, "get", "nodes", "-o", "name", "--request-timeout=20s")


def destroys(env, stacks=STACKS):
    return [call for stack in stacks for call in (init(env, stack), destroy(env, stack))]


def sequence(env, workers=("k3s-cpu-1",), stacks=STACKS):
    """The calls of a complete make down, in order (brief steps 2-5)."""
    return [("gpuctl", ["down", "--auto-approve"]), list_nodes(env),
            *[call for name in workers for call in drain(env, name)], *destroys(env, stacks)]


def calls(env):
    return [(c["tool"], c["argv"]) for c in env.calls()]


def known_hosts(env):
    return sorted(p.name for p in (env.repo / ".run").glob("known_hosts_*"))


def assert_nothing_changed(env):
    assert env.calls() == []
    assert known_hosts(env) == ["known_hosts_k3s-gpu-a6000-1", "known_hosts_k3s-gpu-a6000-2"]
    assert env.kubeconfig.read_text() == KUBECONFIG


def assert_bootstrap_kept(env):
    """Brief step 6: the bootstrap stack (the state bucket) is never destroyed, applied or emptied."""
    for call in env.calls("terraform"):
        assert call["argv"][0] != tf_dir(env, "bootstrap"), call
        assert call["argv"][1] not in ("apply", "plan"), call
    # No gcloud storage rm, no gcloud call at all, except the test's own fake tunnel.
    assert all(c["argv"][:2] == ["compute", "ssh"] and "-N" in c["argv"] for c in env.calls("gcloud"))


# ---- the whole teardown ------------------------------------------------------------------------

def test_down_runs_every_step_in_order(env):
    """Brief steps 2-6: gpuctl down, drain + delete every CPU worker, destroy gpu-access, workers, base,
    stop the tunnel, remove .run/known_hosts_*; the bootstrap stack stays."""
    env.scenario["nodes"] = ["k3s-cp", "k3s-cpu-10", "k3s-cpu-2", "k3s-cpu-1", "k3s-gpu-a6000-1"]
    assert env.script("tunnel.sh", "start").returncode == 0
    pid = int(env.pidfile.read_text())
    tunnel_calls = len(env.calls())
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    # tunnel.sh start read cp_zone from the base stack, so base is initialised already (tf_init's marker).
    assert init(env, "base") in calls(env)[:tunnel_calls]
    expected = sequence(env, workers=("k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-10"))
    expected.remove(init(env, "base"))
    assert calls(env)[tunnel_calls:] == expected
    # Step 5: the tunnel is stopped, the GPU nodes' known_hosts files are gone, nothing else in .run is.
    assert wait_gone(pid) and not env.pidfile.exists()
    assert known_hosts(env) == []
    assert (env.repo / ".run/keep-me").exists()
    # The cluster that ~/.kube/hybrid-gpu.yaml points at is gone: its admin credential goes too.
    assert not env.kubeconfig.exists()
    assert_bootstrap_kept(env)
    assert "bootstrap" in proc.stderr and f"gs://{BUCKET}" in proc.stderr


def test_every_destroy_gets_the_terraform_variables(env):
    """Each stack is planned for destroy with the same TF_VAR_* tf_env sets (the stacks need project_id and
    state_bucket even to destroy)."""
    assert down(env, AUTO_APPROVE="1").returncode == 0
    destroyed = [c for c in env.calls("terraform") if c["argv"][1] == "destroy"]
    assert [c["argv"][0] for c in destroyed] == [tf_dir(env, s) for s in STACKS]
    for call in destroyed:
        assert call["env"]["TF_VAR_project_id"] == PROJECT
        assert call["env"]["TF_VAR_state_bucket"] == BUCKET


def test_the_lambda_api_key_reaches_only_gpuctl(env):
    """down.sh unsets LAMBDA_API_KEY after load_env, as up.sh does; scripts/gpuctl loads .env itself. Nothing
    the script prints holds .env, the key or the kubeconfig."""
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    env.assert_no_lambda_key()
    assert KEY not in proc.stdout + proc.stderr
    assert "client-key-data" not in proc.stdout + proc.stderr and "S0VZ" not in proc.stdout + proc.stderr


def test_a_second_run_finds_nothing_left_and_succeeds(env):
    """Idempotent: once everything is gone, make down runs every step again (each is a no-op) and exits 0."""
    assert down(env, AUTO_APPROVE="1").returncode == 0
    env.log.unlink()
    env.scenario["nodes"] = []
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    # tf_init's marker: the stacks are not initialised again.
    assert calls(env) == [("gpuctl", ["down", "--auto-approve"]), list_nodes(env),
                          *[destroy(env, s) for s in STACKS]]


# ---- step 1: the one confirmation --------------------------------------------------------------

def test_typing_the_project_id_confirms(env):
    proc = down(env, stdin=f"{PROJECT}\n")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == sequence(env)
    assert f"Type the GCP project id ({PROJECT})" in proc.stderr
    # Asked once: nothing after it asks again (gpuctl and terraform are told not to).
    assert proc.stderr.count("Type the GCP project id") == 1


@pytest.mark.parametrize("answer", ["", "\n", "y\n", "yes\n", f"{PROJECT}x\n", f"{PROJECT[:-1]}\n",
                                    "another-project-1\n"])
def test_anything_but_the_project_id_refuses_and_changes_nothing(env, answer):
    proc = down(env, stdin=answer)
    assert proc.returncode == 2
    assert "nothing was changed" in proc.stderr
    assert_nothing_changed(env)


def test_auto_approve_skips_the_question(env):
    proc = down(env, stdin="", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert "Type the GCP project id" not in proc.stderr
    assert calls(env) == sequence(env)


def test_only_auto_approve_1_skips_the_question(env):
    """Same rule as common.sh's confirm: AUTO_APPROVE=1, nothing else."""
    proc = down(env, stdin="", AUTO_APPROVE="yes")
    assert proc.returncode == 2
    assert_nothing_changed(env)


def test_a_missing_env_file_refuses_before_asking(env):
    (env.repo / ".env").unlink()
    proc = down(env, stdin=f"{PROJECT}\n")
    assert proc.returncode == 2 and "Type the GCP project id" not in proc.stderr
    assert_nothing_changed(env)


def test_an_invalid_project_id_refuses_before_asking(env):
    env.write_env("GCP_PROJECT='Not A Project'\n")
    proc = down(env, stdin="Not A Project\n")
    assert proc.returncode == 2 and "Type the GCP project id" not in proc.stderr
    assert_nothing_changed(env)


def test_extra_arguments_are_a_usage_error(env):
    proc = env.script("down.sh", "now", AUTO_APPROVE="1")
    assert proc.returncode == 2 and "usage" in proc.stderr
    assert_nothing_changed(env)


# ---- step 2: gpuctl down -----------------------------------------------------------------------

def test_a_failed_gpu_down_with_gpu_nodes_left_stops_before_any_destroy(env):
    """Destroying gpu-access or base with Lambda nodes still running would leave them billing, outside the
    firewall model and unable to reach any cluster: stop, keep everything, say how to recover."""
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[GPU_LINE])
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == [("gpuctl", ["down", "--auto-approve"]), ("gpuctl", ["list-nodes"])]
    assert GPU_LINE in proc.stderr
    assert "make tunnel" in proc.stderr and "Lambda console" in proc.stderr and "make down" in proc.stderr
    assert "nothing was destroyed" in proc.stderr
    assert known_hosts(env) == ["known_hosts_k3s-gpu-a6000-1", "known_hosts_k3s-gpu-a6000-2"]
    assert env.kubeconfig.exists()


def test_a_failed_gpu_down_stops_when_lambda_cannot_be_asked(env):
    env.scenario.update(gpu_down_rc=2, list_nodes_rc=1)
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == [("gpuctl", ["down", "--auto-approve"]), ("gpuctl", ["list-nodes"])]
    assert "nothing was destroyed" in proc.stderr


def test_a_failed_gpu_down_with_no_gpu_node_left_goes_on(env):
    """A re-run after workers or base were destroyed: gpuctl down's firewall step (sync-peers reads the workers
    and base outputs, Ruling R29) fails, but Lambda lists no GPU node of this cluster, so nothing can be
    orphaned: the teardown goes on and completes."""
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[])
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [("gpuctl", ["down", "--auto-approve"]), ("gpuctl", ["list-nodes"]), *sequence(env)[1:]]
    assert "no GPU node" in proc.stderr


# ---- step 3: drain and delete the workers ------------------------------------------------------

def test_an_unreachable_cluster_skips_the_drains_and_still_destroys(env):
    env.scenario["api_down"] = True
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [("gpuctl", ["down", "--auto-approve"]), list_nodes(env), *destroys(env)]
    assert "cannot reach the cluster" in proc.stderr


def test_a_missing_kubeconfig_skips_the_drains_and_still_destroys(env):
    env.kubeconfig.unlink()
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert [c for c in calls(env) if c[0] == "terraform"] == destroys(env)


def test_a_hanging_cluster_is_bounded_and_still_destroys(env):
    """A half-open tunnel can hang a kubectl request: every kubectl call is killed after DOWN_KC_KILL_S."""
    env.scenario["kubectl_hang"] = True
    start = time.monotonic()
    proc = down(env, AUTO_APPROVE="1", DOWN_KC_KILL_S="1")
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - start < 30
    assert calls(env) == [("gpuctl", ["down", "--auto-approve"]), list_nodes(env), *destroys(env)]


def test_a_failed_drain_still_deletes_the_node_and_goes_on(env):
    env.scenario.update(nodes=["k3s-cp", "k3s-cpu-1", "k3s-cpu-2"], drain_fail=["k3s-cpu-1"], gone=["k3s-cpu-2"])
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == sequence(env, workers=("k3s-cpu-1", "k3s-cpu-2"))
    assert "k3s-cpu-1" in proc.stderr


def test_only_cpu_workers_are_drained(env):
    """k3s-cp is never drained (the base destroy removes it); GPU nodes are gpuctl down's."""
    env.scenario["nodes"] = ["k3s-cp", "k3s-gpu-a6000-1", "not-ours", "k3s-cpu-01", "k3s-cpu-3"]
    assert down(env, AUTO_APPROVE="1").returncode == 0
    assert calls(env) == sequence(env, workers=("k3s-cpu-3",))


# ---- step 4: the destroys, failures and re-runs ------------------------------------------------

@pytest.mark.parametrize("failing", STACKS)
def test_a_failed_destroy_stops_there_and_a_rerun_completes(env, failing):
    env.scenario["destroy_rc"] = {failing: 1}
    first = down(env, AUTO_APPROVE="1")
    assert first.returncode == 1
    done = STACKS[:STACKS.index(failing) + 1]
    assert calls(env) == sequence(env, stacks=done)
    assert f"terraform destroy failed for {failing}" in first.stderr and "make down" in first.stderr
    # Steps 5 and after did not run: the GPU known_hosts files and the kubeconfig are still there.
    assert len(known_hosts(env)) == 2 and env.kubeconfig.exists()

    env.log.unlink()
    env.scenario["destroy_rc"] = {}
    second = down(env, AUTO_APPROVE="1")
    assert second.returncode == 0, second.stderr
    assert [c for c in calls(env) if c[1][1:2] == ["destroy"]] == [destroy(env, s) for s in STACKS]
    assert known_hosts(env) == [] and not env.kubeconfig.exists()
    assert_bootstrap_kept(env)


def test_a_failed_init_stops_before_that_stacks_destroy(env):
    env.scenario["init_rc"] = {"workers": 1}
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == [*sequence(env, stacks=("gpu-access",)), init(env, "workers")]
    assert len(known_hosts(env)) == 2 and env.kubeconfig.exists()


def test_a_failed_tunnel_stop_fails_the_run_after_the_destroys(env):
    """tunnel.sh stop can die (another tunnel.sh holds its lock): down says so and exits 1; a re-run finishes."""
    tunnel = env.repo / "scripts/tunnel.sh"
    tunnel.write_text("#!/bin/sh\necho 'error: another tunnel.sh still holds the lock' >&2\nexit 1\n")
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == sequence(env)
    assert "tunnel.sh stop failed" in proc.stderr and "make down" in proc.stderr


# ---- the script and the Makefile ---------------------------------------------------------------

def test_down_sh_is_executable_bash():
    path = ROOT / "scripts/down.sh"
    assert path.stat().st_mode & stat.S_IXUSR
    assert path.read_text().startswith("#!/usr/bin/env bash\n")


def test_make_down_runs_down_sh():
    proc = subprocess.run(["make", "--no-print-directory", "-n", "-f", str(ROOT / "Makefile"), "down"],
                          capture_output=True, text=True, cwd="/")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{ROOT}/scripts/down.sh"


def test_make_help_lists_down():
    proc = subprocess.run(["make", "--no-print-directory", "-f", str(ROOT / "Makefile"), "help"],
                          capture_output=True, text=True, cwd="/")
    assert proc.returncode == 0, proc.stderr
    assert any(line.split()[:1] == ["down"] for line in proc.stdout.splitlines())
