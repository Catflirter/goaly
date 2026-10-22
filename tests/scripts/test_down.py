"""Offline tests for Task 22a: scripts/down.sh and `make down` (Ruling R44).

Same harness as test_ops_scripts.py: a repo copy whose terraform, gcloud, kubectl, helm, ssh, scp and
scripts/gpuctl are tests/scripts/fake_tools.py, which records every call. Nothing reaches a cloud or a cluster.
"""
import os
import shutil
import signal
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
CP_EXTERNAL_IP = 'jsonpath={.status.addresses[?(@.type=="ExternalIP")].address}'


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


def destroys(env, stacks=STACKS, base_initialised=False):
    return [call for stack in stacks for call in (init(env, stack), destroy(env, stack))
            if not (base_initialised and call == init(env, "base"))]


def same_cluster(env):
    """Before draining: Node k3s-cp's ExternalIP against the base stack's cp_ip (X0-4)."""
    return [kc(env, "get", "node", "k3s-cp", "-o", CP_EXTERNAL_IP, "--request-timeout=20s"), init(env, "base"),
            ("terraform", [tf_dir(env, "base"), "output", "-raw", "cp_ip"])]


GPU_DOWN = ("gpuctl", ["down", "--auto-approve"])
LIST_GPU = ("gpuctl", ["list-nodes"])


def sequence(env, workers=("k3s-cpu-1",), stacks=STACKS, final=True):
    """The calls of a complete make down, in order (brief steps 2-5, then Lambda is asked once more)."""
    check = same_cluster(env) if workers else []
    return [GPU_DOWN, list_nodes(env), *check, *[call for name in workers for call in drain(env, name)],
            *destroys(env, stacks, base_initialised=bool(workers)), *([LIST_GPU] if final else [])]


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
    assert "Lambda lists no GPU node of this cluster" in proc.stderr
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
    assert calls(env) == [GPU_DOWN, list_nodes(env), *[destroy(env, s) for s in STACKS], LIST_GPU]


# ---- step 1: the one confirmation --------------------------------------------------------------

def test_typing_the_project_id_confirms(env):
    proc = down(env, stdin=f"{PROJECT}\n")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == sequence(env)
    assert f"Type the GCP project id ({PROJECT})" in proc.stderr
    # Asked once: nothing after it asks again (gpuctl and terraform are told not to).
    assert proc.stderr.count("Type the GCP project id") == 1
    # The stretch autoscaler could start a GPU node again during the teardown (spec §10.4).
    assert "autoscaler" in proc.stderr


def test_gpuctl_never_reads_the_operators_input(env):
    """gpuctl down and list-nodes get /dev/null: neither the terminal nor what follows the typed answer."""
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[])
    proc = down(env, stdin=f"{PROJECT}\nleftover\nmore\n")
    assert proc.returncode == 0, proc.stderr
    gpuctl = env.calls("gpuctl")
    assert [c["argv"] for c in gpuctl] == [["down", "--auto-approve"], ["list-nodes"], ["list-nodes"]]
    assert [c["stdin"] for c in gpuctl] == ["", "", ""]


@pytest.mark.parametrize("answer", ["", "\n", "y\n", "yes\n", f"{PROJECT}x\n", f"{PROJECT[:-1]}\n",
                                    "another-project-1\n"])
def test_anything_but_the_project_id_refuses_and_changes_nothing(env, answer):
    proc = down(env, stdin=answer)
    assert proc.returncode == 2
    assert "nothing was changed" in proc.stderr
    # Declining is part of the demo (spec §7 step 8): it reads as a cancel, not as an error.
    assert "cancelled" in proc.stderr and "error" not in proc.stderr
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


def test_auto_approve_in_env_does_not_skip_the_question(env):
    """Only the caller's AUTO_APPROVE=1 skips the question, never a line in .env (X0-3)."""
    env.write_env(f"GCP_PROJECT={PROJECT}\nAUTO_APPROVE=1\n")
    proc = down(env, stdin="")
    assert proc.returncode == 2 and "Type the GCP project id" in proc.stderr
    assert_nothing_changed(env)


def test_env_cannot_redirect_the_teardown(env, tmp_path):
    """.env is sourced with set -a: it cannot change which stacks are destroyed, which Nodes are drained or
    which files are removed."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "known_hosts_x").write_text("keep\n")
    (other / "kubeconfig").write_text("keep\n")
    env.write_env(f"GCP_PROJECT={PROJECT}\nDESTROY_STACKS=bootstrap\nRUN_DIR={other}\n"
                  f"KUBECONFIG_FILE={other}/kubeconfig\nCPU_NODE_RE='.*'\nREPO_ROOT={other}\n")
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == sequence(env)
    assert (other / "known_hosts_x").exists() and (other / "kubeconfig").exists()
    assert known_hosts(env) == [] and not env.kubeconfig.exists()
    assert_bootstrap_kept(env)


@pytest.mark.parametrize("value", ["abc", "0", "0s5", "3m x", "-1", "1.5"])
def test_a_bad_kubectl_time_limit_refuses_before_asking(env, value):
    """DOWN_KC_KILL_S must be whole seconds: a bad value would make every kubectl call fail like an unreachable
    cluster."""
    proc = down(env, stdin=f"{PROJECT}\n", DOWN_KC_KILL_S=value)
    assert proc.returncode == 2 and "DOWN_KC_KILL_S" in proc.stderr
    assert "Type the GCP project id" not in proc.stderr
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
    assert calls(env) == [GPU_DOWN, LIST_GPU]
    assert GPU_LINE in proc.stderr
    assert "make tunnel" in proc.stderr and "Lambda console" in proc.stderr and "make down" in proc.stderr
    assert "nothing was destroyed" in proc.stderr
    assert known_hosts(env) == ["known_hosts_k3s-gpu-a6000-1", "known_hosts_k3s-gpu-a6000-2"]
    assert env.kubeconfig.exists()


def test_a_failed_gpu_down_stops_when_lambda_cannot_be_asked(env):
    env.scenario.update(gpu_down_rc=2, list_nodes_rc=1)
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == [GPU_DOWN, LIST_GPU]
    assert "nothing was destroyed" in proc.stderr


def test_a_failed_gpu_down_with_no_gpu_node_left_goes_on(env):
    """A re-run after workers or base were destroyed: gpuctl down's firewall step (sync-peers reads the workers
    and base outputs, Ruling R29) fails, but Lambda lists no GPU node of this cluster, so nothing can be
    orphaned: the teardown goes on and completes."""
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[])
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [GPU_DOWN, LIST_GPU, *sequence(env)[1:]]
    assert "no GPU node" in proc.stderr
    assert ".autoscale-state.json" not in proc.stderr


def test_a_failed_gpu_down_warns_about_leftover_autoscaler_overrides(env):
    """gpuctl down removes lambda/.autoscale-state.json only when it succeeds (C0-2)."""
    (env.repo / "lambda").mkdir()
    (env.repo / "lambda/.autoscale-state.json").write_text("{}\n")
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[])
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert "lambda/.autoscale-state.json" in proc.stderr


# ---- the last step: Lambda is asked once more (spec §13) ---------------------------------------

def test_a_gpu_node_still_on_lambda_after_the_teardown_fails_the_run(env):
    """gpuctl down only removes what it listed: a node started meanwhile (a stretch autoscaler still running, a
    concurrent gpu-apply) would keep billing after base is gone. make down says so and exits 1."""
    env.scenario["gpu_nodes"] = [GPU_LINE]
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == sequence(env)
    assert GPU_LINE in proc.stderr and "still billing" in proc.stderr and "Lambda console" in proc.stderr
    assert "make down: done" not in proc.stderr
    # Everything else was finished: the local cleanup ran.
    assert known_hosts(env) == [] and not env.kubeconfig.exists()


def test_a_gpu_node_started_after_a_failed_gpu_down_is_caught(env):
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[], late_gpu_nodes=[GPU_LINE])
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == [GPU_DOWN, LIST_GPU, *sequence(env)[1:]]
    assert GPU_LINE in proc.stderr and "still billing" in proc.stderr


def test_a_failed_last_lambda_check_fails_the_run(env):
    env.scenario["list_nodes_rc"] = 1
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == sequence(env)
    assert "could not" in proc.stderr and "scripts/gpuctl list-nodes" in proc.stderr
    assert "make down: done" not in proc.stderr


# ---- Ctrl-C ------------------------------------------------------------------------------------

def start_down(env, **extra):
    """down.sh in its own session, so a test can send the terminal's Ctrl-C (SIGINT to the process group)."""
    return subprocess.Popen([str(env.repo / "scripts/down.sh")], cwd=env.repo, env=env.environ(extra),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)


def interrupt_after(env, proc, call, seconds=20):
    """Ctrl-C (SIGINT to down.sh's process group) once CALL is logged; returns rc, stderr, seconds it took."""
    deadline = time.monotonic() + seconds
    while call not in calls(env):
        assert proc.poll() is None and time.monotonic() < deadline, "the call never came"
        time.sleep(0.05)
    time.sleep(0.3)  # the fake is inside its sleep
    os.killpg(proc.pid, signal.SIGINT)
    start = time.monotonic()
    try:
        _, err = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    return proc.returncode, err, time.monotonic() - start


def test_ctrl_c_during_the_drains_stops_before_any_destroy(env):
    """kubectl runs under timeout --foreground, so Ctrl-C reaches it at once, and the INT trap stops the run:
    nothing after the interrupted step is destroyed."""
    env.scenario["kubectl_hang"] = True
    proc = start_down(env, AUTO_APPROVE="1", DOWN_KC_KILL_S="60")
    rc, err, took = interrupt_after(env, proc, list_nodes(env))
    assert rc == 130, err
    assert took < 15
    assert calls(env) == [GPU_DOWN, list_nodes(env)]
    assert "interrupted" in err and "make down" in err
    assert len(known_hosts(env)) == 2 and env.kubeconfig.exists()


def test_ctrl_c_during_gpuctl_down_stops_the_run(env):
    """gpuctl catches KeyboardInterrupt and exits 1; the run must not take that for an ordinary failure and go on
    when Lambda lists no node."""
    env.scenario["gpu_down_hang"] = True
    proc = start_down(env, AUTO_APPROVE="1")
    rc, err, _ = interrupt_after(env, proc, GPU_DOWN)
    assert rc == 130, err
    assert calls(env) == [GPU_DOWN]
    assert "interrupted" in err


# ---- step 3: drain and delete the workers ------------------------------------------------------

def test_an_unreachable_cluster_skips_the_drains_and_still_destroys(env):
    env.scenario["api_down"] = True
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [GPU_DOWN, list_nodes(env), *destroys(env), LIST_GPU]
    assert "cannot reach the cluster" in proc.stderr


def test_a_missing_kubeconfig_skips_the_drains_and_still_destroys(env):
    env.kubeconfig.unlink()
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert [c for c in calls(env) if c[1][1:2] == ["destroy"]] == [destroy(env, s) for s in STACKS]


def test_a_hanging_cluster_is_bounded_and_still_destroys(env):
    """A half-open tunnel can hang a kubectl request: every kubectl call is killed after DOWN_KC_KILL_S."""
    env.scenario["kubectl_hang"] = True
    start = time.monotonic()
    proc = down(env, AUTO_APPROVE="1", DOWN_KC_KILL_S="1")
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - start < 30
    assert calls(env) == [GPU_DOWN, list_nodes(env), *destroys(env), LIST_GPU]


def test_a_failed_drain_still_deletes_the_node_and_goes_on(env):
    env.scenario.update(nodes=["k3s-cp", "k3s-cpu-1", "k3s-cpu-2"], drain_fail=["k3s-cpu-1"], gone=["k3s-cpu-2"])
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == sequence(env, workers=("k3s-cpu-1", "k3s-cpu-2"))
    assert "k3s-cpu-1" in proc.stderr


@pytest.mark.parametrize("node_ip", ["34.99.0.1", ""])
def test_another_projects_cluster_is_not_drained(env, node_ip):
    """KUBECONFIG_FILE and the tunnel port are one per user: if they reach another project's cluster (Node k3s-cp's
    ExternalIP is not this project's cp_ip), its workers are not drained; this project's stacks are still
    destroyed."""
    env.scenario["cp_node_ip"] = node_ip
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [GPU_DOWN, list_nodes(env), *same_cluster(env),
                          *destroys(env, base_initialised=True), LIST_GPU]
    assert "34.82.10.10" in proc.stderr and "skipping the drains" in proc.stderr
    if node_ip:
        assert node_ip in proc.stderr


def test_a_base_stack_without_outputs_skips_the_drains(env):
    env.scenario["outputs"] = {**env.scenario["outputs"], "base": {}}
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [GPU_DOWN, list_nodes(env), *same_cluster(env),
                          *destroys(env, base_initialised=True), LIST_GPU]
    assert "skipping the drains" in proc.stderr


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
    assert calls(env) == sequence(env, stacks=done, final=False)
    assert f"terraform destroy failed for {failing}" in first.stderr and "make down" in first.stderr
    # A state lock held by a concurrent gpuctl or terraform run clears on a re-run (C0-5).
    assert "state lock" in first.stderr
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
    assert calls(env) == [*sequence(env, stacks=("gpu-access",), final=False), init(env, "workers")]
    assert len(known_hosts(env)) == 2 and env.kubeconfig.exists()


def test_a_failed_tunnel_stop_fails_the_run_after_the_destroys(env):
    """tunnel.sh stop can die (another tunnel.sh holds its lock): down says so and exits 1; a re-run finishes."""
    tunnel = env.repo / "scripts/tunnel.sh"
    tunnel.write_text("#!/bin/sh\necho 'error: another tunnel.sh still holds the lock' >&2\nexit 1\n")
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == sequence(env, final=False)
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
