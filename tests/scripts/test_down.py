"""Offline tests for Task 22a: scripts/down.sh and `make down` (Ruling R44).

Same harness as test_ops_scripts.py: a repo copy whose terraform, gcloud, kubectl, helm, ssh, scp and
scripts/gpuctl are tests/scripts/fake_tools.py, which records every call. Nothing reaches a cloud or a cluster.
"""
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time

import pytest

from test_ops_scripts import BUCKET, KUBECONFIG, OUTPUTS, PROJECT, ROOT, Env, make_kubeconfig, wait_gone

STACKS = ("gpu-access", "workers", "base")  # brief step 4: destroyed in this order
RT = "--request-timeout=60s"
DRAIN = ["--ignore-daemonsets", "--delete-emptydir-data", "--force", "--timeout=120s", RT]
GPU_LINE = "k3s-gpu-a6000-1 129.80.1.2 active gpu_1x_a6000 us-west-1"
GPU_LINE_GOING = "k3s-gpu-a6000-1 129.80.1.2 terminating gpu_1x_a6000 us-west-1"
GPU_LINE_BOOTING = "k3s-gpu-a6000-2 - booting gpu_1x_a6000 us-west-1"
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
    """down.sh; Lambda is asked once per check unless the test sets DOWN_LAMBDA_WAIT_S (F2-3)."""
    extra.setdefault("DOWN_LAMBDA_WAIT_S", "0")
    return env.script("down.sh", stdin=stdin, **extra)


def tf_dir(env, stack):
    return f"-chdir={env.repo.resolve()}/infra/{stack}"


def init(env, stack):
    return ("terraform", [tf_dir(env, stack), "init", "-input=false", "-lockfile=readonly", "-reconfigure",
                          f"-backend-config=bucket={BUCKET}", f"-backend-config=prefix={stack}"])


def destroy(env, stack, *extra):
    return ("terraform", [tf_dir(env, stack), "destroy", "-auto-approve", "-input=false", *extra])


def state_list(env, stack):
    """Before each destroy (D-3): a stack whose state lists no resource is not destroyed again. The fake terraform
    fails `state list` unless a test scripts it, and a state that cannot be listed is destroyed as before."""
    return ("terraform", [tf_dir(env, stack), "state", "list"])


def kc(env, *args):
    return ("kubectl", ["--kubeconfig", str(env.kubeconfig), *args])


def drain(env, name):
    return [kc(env, "drain", name, *DRAIN), kc(env, "delete", "node", name, RT)]


def list_nodes(env):
    return kc(env, "get", "nodes", "-o", "name", "--request-timeout=20s")


def destroys(env, stacks=STACKS, base_initialised=False, workers_listed=False):
    """`workers_listed`: the drains listed the workers state already (review F2b-3), and its destroy reuses that."""
    listed_once = [init(env, "workers"), state_list(env, "workers")] if workers_listed else []
    return [call for stack in stacks for call in (init(env, stack), state_list(env, stack), destroy(env, stack))
            if not (base_initialised and call == init(env, "base")) and call not in listed_once]


def same_cluster(env):
    """Before draining: Node k3s-cp's ExternalIP against the base stack's cp_ip (X0-4)."""
    return [kc(env, "get", "node", "k3s-cp", "-o", CP_EXTERNAL_IP, "--request-timeout=20s"), init(env, "base"),
            ("terraform", [tf_dir(env, "base"), "output", "-raw", "cp_ip"])]


GPU_DOWN = ("gpuctl", ["down", "--auto-approve"])
LIST_GPU = ("gpuctl", ["list-nodes"])


def workers_state(env):
    """Before draining (review F2b-3): a workers stack whose state lists no resource has no VMs left to drain. Listed
    once: the workers destroy reuses the answer."""
    return [init(env, "workers"), state_list(env, "workers")]


def sequence(env, workers=("k3s-cpu-1",), stacks=STACKS, final=True):
    """The calls of a complete make down, in order (brief steps 2-5, then Lambda is asked once more)."""
    check = [*same_cluster(env), *workers_state(env)] if workers else []
    return [GPU_DOWN, list_nodes(env), *check, *[call for name in workers for call in drain(env, name)],
            *destroys(env, stacks, base_initialised=bool(workers), workers_listed=bool(workers)),
            *([LIST_GPU] if final else [])]


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
    assert calls(env) == [GPU_DOWN, list_nodes(env), *[c for s in STACKS for c in (state_list(env, s), destroy(env, s))],
                          LIST_GPU]


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


def assert_env_refused(env, proc, name):
    assert proc.returncode == 2, proc.stderr
    assert "Type the GCP project id" not in proc.stderr and "AUTO_APPROVE=1: going on" not in proc.stderr
    assert name in proc.stderr and ".env" in proc.stderr and "nothing was changed" in proc.stderr
    assert_nothing_changed(env)


@pytest.mark.parametrize("line", ["AUTO_APPROVE=1", "export AUTO_APPROVE=1", "declare -g AUTO_APPROVE=1",
                                  "AUTO_APPROVE=", "unset AUTO_APPROVE"])
@pytest.mark.parametrize("caller", [None, "1"])
def test_env_cannot_set_auto_approve(env, line, caller):
    """Only the caller's AUTO_APPROVE=1 skips the question, never a line in .env (X0-3, F2-1): make down
    refuses to run (exit 2, nothing changed) while .env touches AUTO_APPROVE, whatever the caller set."""
    env.write_env(f"GCP_PROJECT={PROJECT}\n{line}\n")
    proc = down(env, stdin="", **({"AUTO_APPROVE": caller} if caller else {}))
    assert_env_refused(env, proc, "AUTO_APPROVE")


@pytest.mark.parametrize("line", ["REPO_ROOT={other}", "RUN_DIR={other}", "KUBECONFIG_FILE={other}/kubeconfig",
                                  "CPU_NODE_RE='.*'", "CP_NAME=k3s-cpu-1", "KC_REQUEST_TIMEOUT=1s",
                                  "DESTROY_STACKS=bootstrap", "export RUN_DIR={other}"])
def test_env_cannot_redirect_the_teardown(env, tmp_path, line):
    """.env is sourced with set -a: a line that sets what down.sh destroys, drains or removes refuses the run
    before the question (F2-1): nothing is drained, destroyed or removed, here or in the other directory."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "known_hosts_x").write_text("keep\n")
    (other / "kubeconfig").write_text("keep\n")
    env.write_env(f"GCP_PROJECT={PROJECT}\n{line.format(other=other)}\n")
    proc = down(env, AUTO_APPROVE="1")
    assert_env_refused(env, proc, line.split("=")[0].split()[-1])
    assert (other / "known_hosts_x").exists() and (other / "kubeconfig").exists()


@pytest.mark.parametrize("line", ["false", "FOO=$(cat /no/such/file)", "BAR=$(exit 3)"])
def test_a_failing_env_line_is_not_blamed_on_a_forbidden_variable(env, line):
    """A .env line that fails for its own reason refuses the run (exit 2, nothing changed), but the message does
    not say .env set AUTO_APPROVE or a readonly name it never mentions (C2-2, X2-2)."""
    env.write_env(f"GCP_PROJECT={PROJECT}\n{line}\n")
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 2, proc.stderr
    assert "a line of" in proc.stderr and ".env failed" in proc.stderr and "nothing was changed" in proc.stderr
    assert "AUTO_APPROVE" not in proc.stderr and "DESTROY_STACKS" not in proc.stderr
    assert_nothing_changed(env)


@pytest.mark.parametrize("comment", ["# AUTO_APPROVE belongs on the command line, never here",
                                     "  # DESTROY_STACKS is decided by make down"])
def test_a_comment_naming_a_forbidden_variable_is_not_blamed(env, comment):
    """Only a line that is not a comment can set a variable: a comment that mentions a readonly name plus an
    unrelated failing line gets the generic message, not 'remove that line'."""
    env.write_env(f"GCP_PROJECT={PROJECT}\n{comment}\nFOO=$(cat /no/such/file)\n")
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 2, proc.stderr
    assert "a line of" in proc.stderr and ".env failed" in proc.stderr and "nothing was changed" in proc.stderr
    assert "names AUTO_APPROVE" not in proc.stderr and "names DESTROY_STACKS" not in proc.stderr
    assert "remove that line" not in proc.stderr
    assert_nothing_changed(env)


def test_a_non_private_env_keeps_load_envs_own_message(env):
    """load_env's own refusal (exit 2, .env not 600) is not rewritten as a failed .env line."""
    env.write_env(f"GCP_PROJECT={PROJECT}\n", mode=0o644)
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 2 and "must be 600" in proc.stderr and "a line of" not in proc.stderr
    assert_nothing_changed(env)


def test_a_forbidden_line_is_named_when_it_is_there(env):
    """When .env does name a readonly variable, the message says which one to remove."""
    env.write_env(f"GCP_PROJECT={PROJECT}\n# a comment\nRUN_DIR=/tmp/elsewhere\n")
    proc = down(env, AUTO_APPROVE="1")
    assert_env_refused(env, proc, "RUN_DIR")
    assert "names RUN_DIR" in proc.stderr and "AUTO_APPROVE" not in proc.stderr


@pytest.mark.parametrize("line", ["TF_WORKSPACE=other", "TF_DATA_DIR=/tmp/elsewhere", "TF_CLI_ARGS=-target=x",
                                  "TF_CLI_ARGS_destroy=-target=google_compute_firewall.x",
                                  "TF_CLI_ARGS_init=-backend-config=prefix=other", "export TF_WORKSPACE=other"])
def test_env_cannot_point_terraform_elsewhere(env, line):
    """.env is sourced with set -a, so it could export Terraform's own overrides: TF_WORKSPACE (another
    workspace's empty state 'destroys' fine), TF_DATA_DIR, TF_CLI_ARGS[_*] (-target). make down would report
    every stack destroyed and leave the resources running. It refuses before the question instead (S2-1)."""
    env.write_env(f"GCP_PROJECT={PROJECT}\n{line}\n")
    proc = down(env, AUTO_APPROVE="1")
    assert_env_refused(env, proc, line.split("=")[0].split()[-1])


def test_the_callers_terraform_overrides_are_refused_too(env):
    """terraform sees the caller's environment as well: TF_WORKSPACE there refuses the run the same way."""
    proc = down(env, AUTO_APPROVE="1", TF_WORKSPACE="other")
    assert proc.returncode == 2 and "TF_WORKSPACE" in proc.stderr and "nothing was changed" in proc.stderr
    assert_nothing_changed(env)


@pytest.mark.parametrize("line", ["APPROVE=1", "PINNED_REPO_ROOT={other}", "PINNED_RUN_DIR={other}",
                                  "PINNED_KUBECONFIG_FILE={other}/kubeconfig"])
def test_env_cannot_reach_the_old_pinned_names(env, tmp_path, line):
    """The names round 1 pinned with (APPROVE, PINNED_*) mean nothing to down.sh any more: the question is still
    asked, and with AUTO_APPROVE=1 from the caller the teardown removes exactly this project's files (F2-1)."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "known_hosts_x").write_text("keep\n")
    (other / "kubeconfig").write_text("keep\n")
    env.write_env(f"GCP_PROJECT={PROJECT}\n{line.format(other=other)}\n")
    proc = down(env, stdin="")
    assert proc.returncode == 2 and "Type the GCP project id" in proc.stderr
    assert_nothing_changed(env)
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
    # PR #104 review X0-4: gpuctl down may have refused another cluster (D-2): make kubeconfig points the kubeconfig
    # back here; --no-drain abandons the cluster through gpuctl (firewall and peers kept right); the console last.
    err = " ".join(proc.stderr.split())
    assert "another cluster" in err and "make kubeconfig" in err
    assert err.index("make kubeconfig") < err.index("scripts/gpuctl down --no-drain") < err.index("Lambda console")
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

def assert_final_hint_names_gpu_down(err):
    """The last Lambda check's hint (Ruling R54, review F8-2): with the cluster gone (base destroyed, kubeconfig
    removed) make gpu-down terminates this cluster's GPU nodes without draining them, so it is the way out;
    scripts/gpuctl down --no-drain if gpuctl still refuses an undrained terminate; the Lambda console last."""
    assert "make gpu-down cannot" not in err and "drains each node through the cluster first" not in err
    assert "run make gpu-down" in err and "without draining" in err
    assert "scripts/gpuctl down --no-drain" in err
    assert err.index("make gpu-down") < err.index("scripts/gpuctl down --no-drain") < err.index("Lambda console")


def test_a_gpu_node_still_on_lambda_after_the_teardown_fails_the_run(env):
    """gpuctl down only removes what it listed: a node started meanwhile (a stretch autoscaler still running, a
    concurrent gpu-apply) would keep billing after base is gone. make down says so and exits 1."""
    env.scenario["gpu_nodes"] = [GPU_LINE]
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == sequence(env)
    assert GPU_LINE in proc.stderr and "still billing" in proc.stderr and "Lambda console" in proc.stderr
    # The cluster is gone, so make gpu-down terminates its nodes without the drain (Ruling R54, review F8-2).
    assert_final_hint_names_gpu_down(proc.stderr)
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
    assert_final_hint_names_gpu_down(proc.stderr)
    assert "make down: done" not in proc.stderr


# ---- a node Lambda still lists right after its terminate (F2-3) ---------------------------------

@pytest.mark.parametrize("then", [[GPU_LINE_GOING], []])
def test_a_just_terminated_node_still_listed_as_active_is_not_a_false_alarm(env, then):
    """gpuctl down terminated the node, then failed in sync-peers; Lambda may list the node as active for a
    moment after the terminate call (reconcile.py _live_gpu_ips). down.sh asks again within its window and goes
    on once the node is terminating or gone."""
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[GPU_LINE], gpu_nodes_then=then)
    proc = down(env, AUTO_APPROVE="1", DOWN_LAMBDA_WAIT_S="2")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [GPU_DOWN, LIST_GPU, LIST_GPU, *sequence(env)[1:]]
    assert "asking again" in proc.stderr


def test_a_terminating_node_is_not_live(env):
    """terminating and terminated instances no longer bill: they never stop the teardown."""
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[GPU_LINE_GOING.replace("terminating", "terminated"),
                                                  GPU_LINE_GOING])
    proc = down(env, AUTO_APPROVE="1", DOWN_LAMBDA_WAIT_S="30")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [GPU_DOWN, LIST_GPU, *sequence(env)[1:]]


@pytest.mark.parametrize("line", [GPU_LINE, GPU_LINE_BOOTING])
def test_a_node_still_live_after_the_window_stops_before_any_destroy(env, line):
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[line])
    start = time.monotonic()
    proc = down(env, AUTO_APPROVE="1", DOWN_LAMBDA_WAIT_S="1")
    assert proc.returncode == 1
    assert time.monotonic() - start < 10
    # A 1 s window: asked once, a 1 s pause, asked once more (C2-1, X2-1: exactly, not "at least").
    assert calls(env) == [GPU_DOWN, LIST_GPU, LIST_GPU]
    assert "asking again in 1 s, for up to 1 s" in proc.stderr
    assert line in proc.stderr and "nothing was destroyed" in proc.stderr
    assert env.kubeconfig.exists()


@pytest.mark.parametrize("wait, asks", [("1", 2), ("6", 3)])
def test_a_slow_lambda_api_does_not_shorten_the_window(env, wait, asks):
    """The window counts the pauses (5 s at most each), not the wall clock: however long each list-nodes call
    takes, a DOWN_LAMBDA_WAIT_S=N window asks 1 + ceil(N/5) times (C2-1, X2-1). Before, a 1.1 s list-nodes
    call used up a 1 s window measured with bash's whole-second SECONDS, and Lambda was asked only once."""
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[GPU_LINE], list_nodes_sleep=1.1)
    proc = down(env, AUTO_APPROVE="1", DOWN_LAMBDA_WAIT_S=wait)
    assert proc.returncode == 1
    assert calls(env) == [GPU_DOWN, *[LIST_GPU] * asks]
    assert "nothing was destroyed" in proc.stderr


def test_the_window_defaults_to_60_seconds(env):
    env.scenario.update(gpu_down_rc=1, gpu_nodes=[GPU_LINE], gpu_nodes_then=[])
    proc = env.script("down.sh", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert "asking again in 5 s, for up to 60 s" in proc.stderr
    assert calls(env) == [GPU_DOWN, LIST_GPU, LIST_GPU, *sequence(env)[1:]]


def test_the_last_check_asks_again_too(env):
    """The last check uses the same window: a node gpuctl down terminated may still be listed as active."""
    env.scenario.update(gpu_nodes=[GPU_LINE], gpu_nodes_then=[])
    proc = down(env, AUTO_APPROVE="1", DOWN_LAMBDA_WAIT_S="2")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [*sequence(env), LIST_GPU]
    assert "make down: done" in proc.stderr


@pytest.mark.parametrize("value", ["abc", "-1", "1.5", "5s", "60 x"])
def test_a_bad_lambda_window_refuses_before_asking(env, value):
    proc = down(env, stdin=f"{PROJECT}\n", DOWN_LAMBDA_WAIT_S=value)
    assert proc.returncode == 2 and "DOWN_LAMBDA_WAIT_S" in proc.stderr
    assert "Type the GCP project id" not in proc.stderr
    assert_nothing_changed(env)


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

def test_workers_with_nothing_in_their_state_are_not_drained(env):
    """PR #104 review F2b-3: a make down stopped after the workers destroy leaves the workers' Nodes listed, NotReady,
    with no VM behind them; each drain would wait out its 120 s timeout. The workers state is listed once, before
    the drains: it lists no resource, so the drains are skipped with one line, and the workers destroy reuses the
    answer (no second state list)."""
    env.scenario["nodes"] = ["k3s-cp", "k3s-cpu-1", "k3s-cpu-2"]
    env.scenario["answers"] = [listed("gpu-access"), listed("workers"), listed("base", BASE_RESOURCES)]
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [GPU_DOWN, list_nodes(env), *same_cluster(env), *workers_state(env),
                          init(env, "gpu-access"), state_list(env, "gpu-access"), state_list(env, "base"),
                          destroy(env, "base"), LIST_GPU]
    assert "the workers stack's state lists no resource" in proc.stderr
    assert "skipping the drains of k3s-cpu-1 k3s-cpu-2" in proc.stderr
    assert "workers: nothing in its state" in proc.stderr and "make down: done" in proc.stderr


def test_workers_with_resources_in_their_state_are_drained(env):
    env.scenario["answers"] = [listed("workers", WORKER_RESOURCES)]
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    expected = sequence(env)
    # A workers state with resources: its destroy asks whether base is destroyed already (-refresh=false).
    expected.insert(expected.index(destroy(env, "workers")), state_list(env, "base"))
    assert calls(env) == expected
    assert [c for c in calls(env) if c == state_list(env, "workers")] == [state_list(env, "workers")]
    assert "skipping the drains" not in proc.stderr


def test_a_workers_state_that_cannot_be_listed_still_drains(env):
    """Not knowing is not "empty": the Nodes are drained as before, and the workers stack is destroyed."""
    env.scenario["answers"] = [listed("workers", rc=1, stderr="Error: Failed to load state: Error 503")]
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == sequence(env)
    assert "draining the CPU workers as usual" in proc.stderr


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
    """The drains' look at the workers state (review F2b-3) cannot init workers: they drain as usual; the workers
    destroy fails on the same init and stops the run before workers and base."""
    env.scenario["init_rc"] = {"workers": 1}
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == [GPU_DOWN, list_nodes(env), *same_cluster(env), init(env, "workers"),
                          *drain(env, "k3s-cpu-1"), *destroys(env, ("gpu-access",)), init(env, "workers")]
    assert "draining the CPU workers as usual" in proc.stderr and "terraform init failed for workers" in proc.stderr
    assert len(known_hosts(env)) == 2 and env.kubeconfig.exists()


def test_a_failed_tunnel_stop_fails_the_run_after_the_destroys(env):
    """tunnel.sh stop can die (another tunnel.sh holds its lock): down says so and exits 1; a re-run finishes."""
    tunnel = env.repo / "scripts/tunnel.sh"
    tunnel.write_text("#!/bin/sh\necho 'error: another tunnel.sh still holds the lock' >&2\nexit 1\n")
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == sequence(env, final=False)
    assert "tunnel.sh stop failed" in proc.stderr and "make down" in proc.stderr


# ---- re-running a make down that stopped between its destroys (minors 2b, D-3) ------------------

WORKER_RESOURCES = 'google_compute_address.worker["k3s-cpu-1"]\ngoogle_compute_instance.worker["k3s-cpu-1"]\n'
BASE_RESOURCES = "google_compute_instance.cp\ngoogle_compute_network.core\n"


def listed(stack, stdout="", rc=0, stderr=""):
    """An "answers" rule for fake terraform: `terraform -chdir=.../infra/STACK state list` prints STDOUT."""
    return {"tool": "terraform", "has": [f"/infra/{stack} ", "state list"], "stdout": stdout, "stderr": stderr,
            "rc": rc}


def tf_outputs(stack, outputs=None, rc=0, stderr=""):
    """An "answers" rule for gpuctl's `terraform output -no-color -json` of STACK (the fake answers only one name)."""
    stdout = "" if outputs is None else json.dumps({k: {"value": v} for k, v in outputs.items()})
    return {"tool": "terraform", "has": [f"/infra/{stack} ", "output -no-color -json"], "stdout": stdout,
            "stderr": stderr, "rc": rc}


def destroyed(env):
    return [c for c in calls(env) if c[1][1:2] == ["destroy"]]


def test_a_stack_with_nothing_in_its_state_is_not_destroyed_again(env):
    """Interrupted after the workers destroy: the re-run skips gpu-access and workers, whose states list no
    resource, and destroys base as usual."""
    env.scenario["answers"] = [listed("gpu-access"), listed("workers"), listed("base", BASE_RESOURCES)]
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert destroyed(env) == [destroy(env, "base")]
    for stack in ("gpu-access", "workers"):
        assert state_list(env, stack) in calls(env)
        assert f"{stack}: nothing in its state (destroyed by an earlier run, or never applied); skipping" in proc.stderr
    assert "make down: done" in proc.stderr


def test_data_sources_alone_are_nothing_to_destroy(env):
    """terraform_remote_state is read, not created: a state that lists only data sources has no resource."""
    env.scenario["answers"] = [listed("workers", "data.terraform_remote_state.base\n")]
    assert down(env, AUTO_APPROVE="1").returncode == 0
    assert destroyed(env) == [destroy(env, "gpu-access"), destroy(env, "base")]


def test_a_state_that_cannot_be_listed_is_still_destroyed(env):
    """Not knowing is not "empty": the destroy runs, and says what is wrong if the state really is unreadable."""
    env.scenario["answers"] = [listed("workers", rc=1, stderr="Error: Failed to load state: Error 503")]
    assert down(env, AUTO_APPROVE="1").returncode == 0
    assert destroyed(env) == [destroy(env, s) for s in STACKS]


def test_workers_left_after_base_are_destroyed_without_a_refresh(env):
    """Base destroyed, workers not (an order make down never leaves, but a hand-run destroy can): a refreshing
    destroy of workers cannot evaluate local.base.* without base's outputs ("Unsupported attribute", measured
    with terraform 1.17), so workers is destroyed from its state as recorded (-refresh=false)."""
    env.scenario["answers"] = [listed("gpu-access"), listed("workers", WORKER_RESOURCES), listed("base")]
    env.scenario["api_down"] = True  # k3s-cp is gone, so is the tunnel's far end
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert destroyed(env) == [destroy(env, "workers", "-refresh=false")]
    assert "the base stack is already destroyed" in proc.stderr and "-refresh=false" in proc.stderr


@pytest.mark.parametrize("base", ["base-has-resources", "base-unknown"])
def test_workers_are_refreshed_as_usual_while_base_may_exist(env, base):
    rule = listed("base", BASE_RESOURCES) if base == "base-has-resources" else listed("base", rc=1)
    env.scenario["answers"] = [listed("workers", WORKER_RESOURCES), rule]
    assert down(env, AUTO_APPROVE="1").returncode == 0
    assert destroy(env, "workers") in destroyed(env)
    assert all("-refresh=false" not in c[1] for c in destroyed(env))


# The same re-runs end to end with the real gpuctl (lambda/gpuctl; Lambda is tests' in-memory FakeLambda) in place
# of the fake scripts/gpuctl: its terraform, gcloud and kubectl calls reach the fake tools like down.sh's own.
REAL_GPUCTL = """\
import json
import os
import sys
from pathlib import Path

sys.path[:0] = [{lambda_dir!r}, {tests_dir!r}]
import gpuctl.__main__ as cli  # noqa: E402
from fakes import FakeLambda  # noqa: E402
from gpuctl.infra import Infra, SubprocessShell  # noqa: E402

with open(os.environ["FAKE_LOG"], "a") as log:  # logged like the fake gpuctl's calls
    log.write(json.dumps({{"tool": "gpuctl", "argv": sys.argv[1:], "env": {{}}}}) + "\\n")
repo = Path({repo!r})
cli.OVERRIDES_PATH = repo / "lambda/.autoscale-state.json"
env = {{**os.environ, "LAMBDA_API_KEY": "k", "OPERATOR_IP": "73.15.40.40"}}  # scripts/gpuctl loads .env
sys.exit(cli.main(sys.argv[1:], env=env, client=FakeLambda(), infra=Infra(SubprocessShell(), repo, env)))
"""


@pytest.fixture
def real_gpuctl(env, tmp_path):
    launcher = tmp_path / "real_gpuctl.py"
    launcher.write_text(REAL_GPUCTL.format(lambda_dir=str(ROOT / "lambda"), tests_dir=str(ROOT / "lambda/tests"),
                                           repo=str(env.repo)))
    (env.repo / "scripts/gpuctl").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{launcher}" "$@"\n')
    return env


def guard_pushes(env):
    return [c for c in env.calls("gcloud") if "sudo /usr/local/sbin/k3s-guard-peers" in c["argv"]]


def test_rerun_after_the_workers_destroy_completes_end_to_end(real_gpuctl):
    """The D-3 state: gpu-access and workers destroyed, base not. gpu-down reads the emptied workers stack as no
    CPU peers ({} from terraform output -json), the destroys skip what is gone, base goes, exit 0."""
    env = real_gpuctl
    env.scenario["answers"] = [tf_outputs("base", OUTPUTS["base"]), tf_outputs("workers", {}),
                               listed("gpu-access"), listed("workers"), listed("base", BASE_RESOURCES)]
    env.scenario["nodes"] = ["k3s-cp"]  # the stale k3s-cpu-1 Node is not listed here: no drain wait
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert "0 CPU worker(s)" in proc.stdout  # gpuctl down's sync-peers, on make down's stdout
    assert len(guard_pushes(env)) == 1  # k3s-cp's peer set was rewritten without the worker
    assert destroyed(env) == [destroy(env, "base")]
    assert "workers: nothing in its state" in proc.stderr and "make down: done" in proc.stderr
    assert "scripts/gpuctl down failed" not in proc.stderr
    assert "Traceback" not in proc.stdout + proc.stderr


def test_rerun_after_the_workers_destroy_skips_the_stale_worker_node_end_to_end(real_gpuctl):
    """PR #104 review S0-3/C0-4, F2b-3: in the real D-3 state k3s-cp still lists the destroyed worker's Node,
    NotReady. The workers state lists no resource, so it is not drained (a drain would wait out its 120 s timeout
    on a real cluster): one line says so, base goes, exit 0."""
    env = real_gpuctl
    env.scenario["answers"] = [tf_outputs("base", OUTPUTS["base"]), tf_outputs("workers", {}),
                               listed("gpu-access"), listed("workers"), listed("base", BASE_RESOURCES)]
    env.scenario["nodes"] = ["k3s-cp", "k3s-cpu-1"]
    env.scenario["drain_fail"] = ["k3s-cpu-1"]
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert "skipping the drains of k3s-cpu-1" in proc.stderr
    assert not [c for c in env.calls("kubectl") if c["argv"][2:3] in (["drain"], ["delete"])]
    assert destroyed(env) == [destroy(env, "base")]
    assert "make down: done" in proc.stderr
    assert "Traceback" not in proc.stdout + proc.stderr


def test_rerun_with_a_real_terraform_error_fails_loudly_end_to_end(real_gpuctl):
    """Not "no outputs": a backend error reading workers fails gpu-down (no peer-list write), and the workers
    destroy, which hits the same error, stops make down before base. The operator sees both errors."""
    env = real_gpuctl
    error = "Error: Failed to load state: googleapi: Error 503: Backend Error"
    env.scenario["answers"] = [tf_outputs("base", OUTPUTS["base"]), tf_outputs("workers", rc=1, stderr=error),
                               listed("gpu-access"), listed("workers", rc=1, stderr=error)]
    env.scenario["destroy_rc"] = {"workers": 1}
    env.scenario["nodes"] = ["k3s-cp"]
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert guard_pushes(env) == []
    assert "cannot read infra/workers outputs" in proc.stderr and "Error 503: Backend Error" in proc.stderr
    assert "scripts/gpuctl down failed (exit 1)" in proc.stderr
    assert "terraform destroy failed for workers" in proc.stderr
    assert destroy(env, "base") not in destroyed(env)
    assert "Traceback" not in proc.stdout + proc.stderr


def test_rerun_with_base_destroyed_and_workers_left_completes_end_to_end(real_gpuctl):
    env = real_gpuctl
    env.scenario["answers"] = [tf_outputs("base", {}), listed("gpu-access"), listed("workers", WORKER_RESOURCES),
                               listed("base")]
    env.scenario["api_down"] = True
    proc = down(env, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert "k3s-cp k3s_peers: skipped" in proc.stdout and guard_pushes(env) == []
    assert destroyed(env) == [destroy(env, "workers", "-refresh=false")]
    assert "make down: done" in proc.stderr
    assert "Traceback" not in proc.stdout + proc.stderr


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
