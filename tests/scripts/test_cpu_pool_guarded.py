"""Offline tests for stretch S3: `cpu-pool.sh scale|remove ... --guarded` (design spec sections 10.2, 10.5).

The core flow (tf_plan with --no-destroy / --destroy-only into .run/workers.tfplan, then tf_apply_plan) runs as
before; --guarded pipes `terraform show -json` of that saved plan into `scripts/gpuctl tfguard --add ... --remove ...`
and applies it only when the guard passes. The fake gpuctl runs the real tfguard on the plan it is piped
(tests/scripts/fake_tools.py), so these tests also prove the pipe carries the plan the fake terraform showed.
Same harness as test_workers_ops.py: nothing reaches a cloud or a cluster.
"""
import contextlib
import json
import os
import select
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from test_ops_scripts import OUTPUTS, lock_holder, path_without_flock
from test_ops_scripts import lock_free as path_lock_free
from test_workers_ops import (
    DRAIN_ARGS, RT, calls, cpu_join, env, get_node, index_of, is_drain, is_tf, kc, leftovers, planned, pool,  # noqa: F401
    set_pool, tf, worker_resources,
)


def rc(address, actions):
    return {"address": address, "change": {"actions": actions}}


def creates(name):
    return [rc(a, ["create"]) for a in worker_resources(name)]


def deletes(name):
    return [rc(a, ["delete"]) for a in worker_resources(name)]


SHARED = [rc("google_compute_firewall.cp_to_workers", ["no-op"]),
          rc("google_compute_firewall.workers_iap_ssh", ["no-op"]),
          rc("data.terraform_remote_state.base", ["read"])]
CP_UPDATE = rc("google_compute_firewall.cp_to_workers", ["update"])


def workers(env, ips, after=None):
    """worker_ips/worker_zones of the workers stack now, and (AFTER) once an apply ran."""
    def outputs(ips):
        return {"worker_ips": ips, "worker_zones": {n: "us-central1-a" for n in ips}}
    env.scenario["outputs"] = {**OUTPUTS, "workers": outputs(ips)}
    if after is not None:
        env.scenario["outputs_after_apply"] = {"workers": outputs(after)}


def guard_calls(env):
    return [c for c in env.calls("gpuctl") if c["argv"][:1] == ["tfguard"]]


def guarded(env, *args, **kw):
    return env.script("cpu-pool.sh", *args, "--guarded", **kw)


IP = {f"k3s-cpu-{i}": f"34.10.0.{10 + i}" for i in range(1, 6)}


def ips(*numbers):
    return {f"k3s-cpu-{i}": IP[f"k3s-cpu-{i}"] for i in numbers}


# ---- scale up --------------------------------------------------------------------------------------

def test_guarded_scale_up_checks_the_saved_plan_then_applies_it(env):
    workers(env, ips(1), after=ips(1, 2))
    env.scenario["changes"] = {"workers": [*creates("k3s-cpu-2"), *SHARED]}
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = guarded(env, "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert [c["argv"] for c in guard_calls(env)] == [["tfguard", "--add", "k3s-cpu-2"]]
    # The guard read `terraform show -json` of the saved plan (the very JSON the fake printed).
    assert json.loads(guard_calls(env)[0]["stdin"])["resource_changes"] == env.scenario["changes"]["workers"]
    show = [c["argv"] for c in env.calls("terraform") if c["argv"][1:2] == ["show"]]
    assert show and all(a[2:] == ["-json", str(env.repo.resolve() / ".run/workers.tfplan")] for a in show)
    # Core's flow: one plan (of the new inventory, as a -var-file), the guard, then that saved plan is applied.
    assert tf(env) == ["init", "plan", "show", "show", "apply"]
    apply = next(c["argv"] for c in env.calls("terraform") if c["argv"][1:2] == ["apply"])
    assert apply[-1] == str(env.repo.resolve() / ".run/workers.tfplan")
    assert index_of(env, lambda c: c["argv"][:1] == ["tfguard"]) < index_of(env, is_tf("apply"))
    assert planned(env) == pool(["k3s-cpu-1", "k3s-cpu-2"])
    assert json.loads(env.tfvars.read_text()) == pool(["k3s-cpu-1", "k3s-cpu-2"])
    assert [c for c in calls(env) if c[1][:1] != ["tfguard"]] == [
        get_node(env, "k3s-cpu-1"), ("gpuctl", ["sync-peers"]),
        *cpu_join(env, "k3s-cpu-2", IP["k3s-cpu-2"], sync=False), ("gpuctl", ["sync-peers"])]
    assert leftovers(env) == []
    assert all("LAMBDA_API_KEY" not in c["env"] for c in env.calls())


def test_guarded_scale_up_violation_applies_and_writes_nothing(env):
    workers(env, ips(1), after=ips(1, 2))
    env.scenario["changes"] = {"workers": [*creates("k3s-cpu-2"), CP_UPDATE]}
    before = env.tfvars.read_text()
    proc = guarded(env, "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert "violation: google_compute_firewall.cp_to_workers" in proc.stderr
    assert "apply" not in tf(env)
    assert env.tfvars.read_text() == before, "the inventory is written only after the guard passes"
    assert [c["argv"][:1] for c in env.calls("gpuctl")] == [["tfguard"]], "no sync-peers, no join"
    assert not env.calls("gcloud") and not env.calls("kubectl")
    assert leftovers(env) == [] and not (env.repo / ".run/workers.tfplan").exists()


def test_guarded_scale_up_refuses_an_extra_worker(env):
    workers(env, ips(1), after=ips(1, 2, 3))
    env.scenario["changes"] = {"workers": [*creates("k3s-cpu-2"), *creates("k3s-cpu-3")]}
    proc = guarded(env, "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 1 and 'worker["k3s-cpu-3"]' in proc.stderr
    assert "apply" not in tf(env)


def test_guarded_scale_up_adds_an_inventory_worker_the_stack_never_created(env):
    """The add set is the new inventory minus what the stack has deployed (worker_ips): k3s-cpu-2 is in the file
    from an earlier scale whose apply failed, so this plan creates it and the guard expects exactly that."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1), after=ips(1, 2))
    env.scenario["changes"] = {"workers": creates("k3s-cpu-2")}
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = guarded(env, "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert [c["argv"] for c in guard_calls(env)] == [["tfguard", "--add", "k3s-cpu-2"]]
    assert tf(env)[-1] == "apply"


def test_guarded_scale_to_the_current_size_needs_no_guard(env):
    """No changes: no saved plan, nothing applied, so there is nothing to check."""
    env.scenario["plan_rc"] = {"workers": 0}
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = guarded(env, "scale", "1")
    assert proc.returncode == 0, proc.stderr
    assert guard_calls(env) == [] and "apply" not in tf(env)


def test_guarded_scale_up_keeps_core_no_destroy_check(env):
    """tf_plan's --no-destroy still refuses a destroying plan (exit 2) before the guard runs."""
    workers(env, ips(1), after=ips(1, 2))
    env.scenario["changes"] = {"workers": [*creates("k3s-cpu-2"), rc('google_compute_instance.worker["k3s-cpu-1"]',
                                                                      ["delete", "create"])]}
    proc = guarded(env, "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 2 and guard_calls(env) == [] and "apply" not in tf(env)


# ---- scale down ------------------------------------------------------------------------------------

def test_guarded_scale_down_guards_before_the_drain(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-3"])
    workers(env, ips(1, 2, 3), after=ips(1))
    env.scenario["changes"] = {"workers": [*deletes("k3s-cpu-3"), *deletes("k3s-cpu-2"), *SHARED]}
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = guarded(env, "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert [c["argv"] for c in guard_calls(env)] == [["tfguard", "--remove", "k3s-cpu-2", "k3s-cpu-3"]]
    assert index_of(env, lambda c: c["argv"][:1] == ["tfguard"]) < index_of(env, is_drain)
    drains = [c for c in calls(env) if c[1][2:3] in (["drain"], ["delete"])]
    assert drains == [kc(env, "drain", "k3s-cpu-2", *DRAIN_ARGS), kc(env, "delete", "node", "k3s-cpu-2", RT),
                      kc(env, "drain", "k3s-cpu-3", *DRAIN_ARGS), kc(env, "delete", "node", "k3s-cpu-3", RT)]
    assert tf(env) == ["init", "plan", "show", "show", "apply"]
    assert json.loads(env.tfvars.read_text()) == pool(["k3s-cpu-1"])
    assert leftovers(env) == []


@pytest.mark.parametrize("extra", [CP_UPDATE, rc('google_compute_instance.worker["k3s-cpu-1"]', ["update"]),
                                   *creates("k3s-cpu-4")[:1]])
def test_guarded_scale_down_violation_drains_nothing(env, extra):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2), after=ips(1))
    env.scenario["changes"] = {"workers": [*deletes("k3s-cpu-2"), extra]}
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    before = env.tfvars.read_text()
    proc = guarded(env, "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert f"violation: {extra['address']}" in proc.stderr
    assert not any(is_drain(c) for c in env.calls()), "nothing drained: a re-run never drains twice"
    assert env.tfvars.read_text() == before and "apply" not in tf(env)
    assert "nothing was drained" in proc.stderr
    assert leftovers(env) == []


def test_guarded_scale_down_keeps_core_destroy_only_check(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2))
    env.scenario["changes"] = {"workers": [*deletes("k3s-cpu-2"),
                                           rc('google_compute_instance.worker["k3s-cpu-1"]', ["delete"])]}
    proc = guarded(env, "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 2 and guard_calls(env) == []
    assert not any(is_drain(c) for c in env.calls())


def test_guarded_scale_adds_and_removes_in_one_plan(env):
    """k3s-cpu-5 left the inventory by hand but is still deployed: scale 3 drains and destroys it and creates
    k3s-cpu-3. The guard gets both sets, from what the stack has deployed."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2, 5), after=ips(1, 2, 3))
    env.scenario["changes"] = {"workers": [*creates("k3s-cpu-3"), *deletes("k3s-cpu-5")]}
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    proc = guarded(env, "scale", "3", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert [c["argv"] for c in guard_calls(env)] == [["tfguard", "--add", "k3s-cpu-3", "--remove", "k3s-cpu-5"]]
    assert [c for c in calls(env) if c[1][2:3] == ["drain"]] == [kc(env, "drain", "k3s-cpu-5", *DRAIN_ARGS)]
    assert tf(env)[-1] == "apply"


def test_guarded_scale_fails_closed_when_the_guard_cannot_run(env):
    """Any failure of the guard pipe (here: tfguard exits 2) is a refusal: exit 1, nothing drained or applied."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2), after=ips(1))
    env.scenario["changes"] = {"workers": deletes("k3s-cpu-2")}
    env.scenario["tfguard_rc"] = 2
    proc = guarded(env, "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 1 and "apply" not in tf(env)
    assert not any(is_drain(c) for c in env.calls())


def test_guarded_scale_fails_closed_when_the_guards_show_fails(env):
    """The other half of the guard pipe (C0-2): `terraform show -json` prints a plan that passes, then exits 1.
    pipefail makes that a refusal too: exit 1, nothing drained, written or applied."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2), after=ips(1))
    env.scenario["changes"] = {"workers": deletes("k3s-cpu-2")}
    env.scenario["guard_show_rc"] = {"workers": 1}
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    before = env.tfvars.read_text()
    proc = guarded(env, "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 1, proc.stderr
    assert [c["argv"] for c in guard_calls(env)] == [["tfguard", "--remove", "k3s-cpu-2"]], "tfguard itself passed"
    assert tf(env) == ["init", "plan", "show", "show"] and "failed the guard" in proc.stderr
    assert not any(is_drain(c) for c in env.calls())
    assert env.tfvars.read_text() == before
    assert leftovers(env) == []


# ---- one guarded run at a time, and only the plan that passed the guard is applied (X0-1) --------------

LOCK = ".run/workers.lock"


def lock_free(env):
    """Nobody holds .run/workers.lock (flock(1), as the guarded run takes it)."""
    return path_lock_free(env.repo / LOCK)


def other_guarded_run(env):
    """Another run holds the workers stack's lock: `flock -o .run/workers.lock sleep 60` in its own process group.
    -o: sleep does not inherit the lock, so the lock is free as soon as the flock process is reaped (S3-m3)."""
    return lock_holder(env.repo / LOCK)


@pytest.mark.parametrize("args", [["scale", "2"], ["scale", "1"], ["remove", "k3s-cpu-2"]])
def test_a_guarded_run_refuses_while_another_holds_the_lock(env, args):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    before = env.tfvars.read_text()
    with other_guarded_run(env):
        proc = guarded(env, *args, AUTO_APPROVE="1")
        assert not lock_free(env), "the other run still holds it"
    assert proc.returncode == 2
    assert "another run holds" in proc.stderr and "workers.lock" in proc.stderr
    assert "changed nothing in the workers stack" in proc.stderr
    assert env.calls() == [], "nothing planned, drained or applied"
    assert env.tfvars.read_text() == before


@pytest.mark.parametrize("left_behind", ["", "4242\n"])
def test_a_crashed_holder_leaves_no_stale_lock(env, left_behind):
    """A guarded run SIGKILLed (or a reboot): the kernel released its lock, so the next guarded run proceeds, whatever
    the file holds (an old PID-style lock included). The S4 loop is never stuck on it (S3-L1)."""
    with other_guarded_run(env) as holder:
        os.killpg(holder.pid, signal.SIGKILL)
        holder.wait()
    (env.repo / LOCK).write_text(left_behind)
    workers(env, ips(1), after=ips(1, 2))
    env.scenario["changes"] = {"workers": creates("k3s-cpu-2")}
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = guarded(env, "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert tf(env)[-1] == "apply"
    assert lock_free(env)


def test_a_guarded_run_holds_the_lock_from_plan_to_apply_then_releases_it(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2), after=ips(1))
    env.scenario["changes"] = {"workers": deletes("k3s-cpu-2")}
    env.scenario["drain_plan"] = "keep"
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    proc = guarded(env, "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert Path(str(env.log) + ".lock_at_drain").read_text() == "held", "the lock is held during the drain"
    assert tf(env)[-1] == "apply"
    assert lock_free(env)


def test_a_failed_guarded_run_releases_the_lock(env):
    workers(env, ips(1), after=ips(1, 2))
    env.scenario["changes"] = {"workers": [*creates("k3s-cpu-2"), CP_UPDATE]}
    proc = guarded(env, "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert lock_free(env)


@pytest.mark.parametrize("args", [["scale", "2"], ["scale", "1"], ["remove", "k3s-cpu-2"]])
def test_an_unguarded_run_is_refused_too_while_another_holds_the_lock(env, args):
    """C-1: the workers stack's lock is core's (stack_lock), so an unguarded make cpu-scale / cpu-remove takes it too,
    before it reads the inventory: it can neither race a guarded run nor make up's workers stage."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2))
    before = env.tfvars.read_text()
    with other_guarded_run(env):
        proc = env.script("cpu-pool.sh", *args, AUTO_APPROVE="1")
    assert proc.returncode == 2, proc.stderr
    assert "another run holds" in proc.stderr and "workers.lock" in proc.stderr
    assert env.calls() == [] and env.tfvars.read_text() == before


@pytest.mark.parametrize("args", [["scale", "1"], ["remove", "k3s-cpu-2"]])
def test_an_unguarded_run_holds_the_lock_through_the_drain_to_the_apply(env, args):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2), after=ips(1))
    env.scenario["changes"] = {"workers": deletes("k3s-cpu-2")}
    env.scenario["drain_plan"] = "keep"
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    proc = env.script("cpu-pool.sh", *args, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert Path(str(env.log) + ".lock_at_drain").read_text() == "held", "the lock is held during the drain"
    assert tf(env)[-1] == "apply"
    assert lock_free(env)


def test_guarded_refuses_without_flock_and_unguarded_runs_unchanged(env, tmp_path):
    path = path_without_flock(env, tmp_path)
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    before = env.tfvars.read_text()
    proc = env.run([str(env.repo / "scripts/cpu-pool.sh"), "scale", "1", "--guarded"], AUTO_APPROVE="1", PATH=path)
    assert proc.returncode == 2
    assert "--guarded needs flock" in proc.stderr and "nothing was changed" in proc.stderr
    assert env.calls() == [] and env.tfvars.read_text() == before
    workers(env, ips(1, 2))
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    proc = env.run([str(env.repo / "scripts/cpu-pool.sh"), "scale", "2"], AUTO_APPROVE="1", PATH=path)
    assert proc.returncode == 0, proc.stderr
    # An unguarded run without flock(1) runs unlocked, after exactly one warning (its take_lock, tf_plan and
    # tf_apply_plan all ask for the workers stack's lock).
    assert proc.stderr.count("WARNING") == 1 and "flock" in proc.stderr
    assert tf(env)[-1] == "apply"


@pytest.mark.parametrize("args, changes", [(["scale", "1"], deletes("k3s-cpu-2")),
                                           (["remove", "k3s-cpu-2"], deletes("k3s-cpu-2"))])
def test_a_plan_replaced_after_the_guard_is_never_applied(env, args, changes):
    """Another run (an unguarded make cpu-scale, say) overwrites .run/workers.tfplan while this one drains: the
    bytes about to be applied are not the bytes the guard passed, so nothing is applied (exit 1), and the other run's
    plan is left where it was."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2), after=ips(1))
    env.scenario["changes"] = {"workers": changes}
    env.scenario["drain_plan"] = "replace"
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    proc = guarded(env, *args, AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert "changed after the guard" in proc.stderr and "nothing was applied" in proc.stderr
    assert "apply" not in tf(env)
    assert (env.repo / ".run/workers.tfplan").read_text() == "another run's plan"
    assert ("gpuctl", ["sync-peers"]) not in calls(env)
    assert lock_free(env)


def test_a_plan_replaced_while_the_guard_reads_it_is_never_applied(env):
    """The guard passed the JSON of one plan, but the file now holds another: refused before the drain."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2), after=ips(1))
    env.scenario["changes"] = {"workers": deletes("k3s-cpu-2")}
    env.scenario["guard_show_replaces"] = True
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    before = env.tfvars.read_text()
    proc = guarded(env, "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert "changed while the guard read it" in proc.stderr and "nothing was drained" in proc.stderr
    assert "apply" not in tf(env) and not any(is_drain(c) for c in env.calls())
    assert env.tfvars.read_text() == before
    assert (env.repo / ".run/workers.tfplan").read_text() == "another run's plan"


@pytest.mark.parametrize("args", [["scale", "1"], ["remove", "k3s-cpu-2"]])
def test_a_failed_run_leaves_a_plan_another_run_wrote_after_the_guard(env, args):
    """S3-m2: the EXIT trap removes the saved plan only while it is still the bytes this run's guard read. Here
    another run replaces it during the drain, and the drain then fails: the other run's plan stays."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2), after=ips(1))
    env.scenario["changes"] = {"workers": deletes("k3s-cpu-2")}
    env.scenario["drain_plan"] = "replace"
    env.scenario["drain_fail"] = ["k3s-cpu-2"]
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    proc = guarded(env, *args, AUTO_APPROVE="1")
    assert proc.returncode == 1 and "cannot drain k3s-cpu-2" in proc.stderr
    assert "apply" not in tf(env)
    assert (env.repo / ".run/workers.tfplan").read_text() == "another run's plan"


@pytest.mark.parametrize("args", [["scale", "1"], ["remove", "k3s-cpu-2"]])
def test_a_failed_run_removes_its_own_plan(env, args):
    """The same failure with the plan untouched: it is this run's, and the EXIT trap removes it."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2), after=ips(1))
    env.scenario["changes"] = {"workers": deletes("k3s-cpu-2")}
    env.scenario["drain_fail"] = ["k3s-cpu-2"]
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    proc = guarded(env, *args, AUTO_APPROVE="1")
    assert proc.returncode == 1 and "cannot drain k3s-cpu-2" in proc.stderr
    assert "apply" not in tf(env) and not (env.repo / ".run/workers.tfplan").exists()
    assert leftovers(env) == []


def test_a_plan_removed_after_the_guard_is_a_clear_failure(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    workers(env, ips(1, 2), after=ips(1))
    env.scenario["changes"] = {"workers": deletes("k3s-cpu-2")}
    env.scenario["drain_plan"] = "remove"
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    proc = guarded(env, "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert "changed after the guard" in proc.stderr and "apply" not in tf(env)


# ---- interactive guarded runs: the question comes before the plan check (C1-1) ---------------------------

APPLY_Q = "Apply this plan to the workers stack? [y/N] "


def interactive(env, args, answer):
    """Run `cpu-pool.sh ARGS --guarded` without AUTO_APPROVE. At each [y/N] prompt, while the run waits, call
    answer(question) and type what it returns. Returns (exit code, stderr)."""
    proc = subprocess.Popen([str(env.repo / "scripts/cpu-pool.sh"), *args, "--guarded"], cwd=env.repo,
                            env=env.environ({}), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    err, answered, deadline = b"", 0, time.monotonic() + 60
    try:
        while True:
            ready, _, _ = select.select([proc.stderr], [], [], max(0.0, deadline - time.monotonic()))
            assert ready, f"timed out; stderr so far: {err.decode()}"
            chunk = os.read(proc.stderr.fileno(), 4096)
            if not chunk:
                break
            err += chunk
            text = err.decode()
            while text.count("[y/N] ") > answered:
                end = [i for i in range(len(text)) if text.startswith("[y/N] ", i)][answered] + len("[y/N] ")
                question = text[text.rfind("\n", 0, end) + 1:end]
                proc.stdin.write(answer(question).encode())
                proc.stdin.flush()
                answered += 1
        proc.stdin.close()
        return proc.wait(timeout=60), err.decode()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


SCENARIOS = {
    "scale up": (["scale", "2"], [1], [1, 2], creates("k3s-cpu-2")),
    "scale down": (["scale", "1"], [1, 2], [1], deletes("k3s-cpu-2")),
    "remove": (["remove", "k3s-cpu-2"], [1, 2], [1], deletes("k3s-cpu-2")),
}


def scenario(env, key):
    args, now, after, changes = SCENARIOS[key]
    set_pool(env, [f"k3s-cpu-{i}" for i in now])
    workers(env, ips(*now), after=ips(*after))
    env.scenario["changes"] = {"workers": changes}
    env.scenario["ready"] = [f"k3s-cpu-{i}" for i in now]
    return args


@pytest.mark.parametrize("key", list(SCENARIOS))
def test_an_interactive_guarded_run_never_applies_a_plan_replaced_while_it_asks(env, key):
    """The operator sits at "Apply this plan ...?" while an unguarded make cpu-scale rewrites .run/workers.tfplan:
    the bytes are checked AFTER the answer, so the plan the guard never saw is not applied (exit 1)."""
    args = scenario(env, key)
    plan = env.repo / ".run/workers.tfplan"

    def answer(question):
        if question == APPLY_Q:
            plan.write_text("another run's plan")
        return "y\n"

    rc, err = interactive(env, args, answer)
    assert rc == 1, err
    assert "changed after the guard" in err and "nothing was applied" in err
    assert "apply" not in tf(env)
    assert plan.read_text() == "another run's plan", "the other run's plan is left where it was"
    assert ("gpuctl", ["sync-peers"]) not in calls(env)
    assert lock_free(env)


@pytest.mark.parametrize("key", list(SCENARIOS))
def test_an_interactive_guarded_run_asks_once_then_applies_the_checked_plan(env, key):
    args = scenario(env, key)
    questions = []

    def answer(question):
        questions.append(question)
        return "y\n"

    rc, err = interactive(env, args, answer)
    assert rc == 0, err
    assert questions.count(APPLY_Q) == 1, "the apply is asked exactly once"
    # The evidence log says the operator approved it, not AUTO_APPROVE (S3-L3).
    assert "AUTO_APPROVE" not in err
    assert err.count("the operator confirmed the apply of the workers plan") == 1
    assert tf(env)[-1] == "apply"
    assert not (env.repo / ".run/workers.tfplan").exists()


def test_an_interactive_guarded_run_declined_at_the_apply_applies_nothing(env):
    args = scenario(env, "scale up")
    rc, err = interactive(env, args, lambda q: "n\n" if q == APPLY_Q else "y\n")
    assert rc == 2 and "not applied: workers" in err
    assert "apply" not in tf(env)
    assert not (env.repo / ".run/workers.tfplan").exists()
    assert "workers.tfplan was removed" in err  # S3-m4: the log says what happened to it


def test_an_interactive_decline_says_so_when_the_plan_is_already_gone(env):
    """S3-m4: another run removed the saved plan while this one asked: the decline says it was gone already, not that
    it was removed or left where it is."""
    args = scenario(env, "scale up")
    plan = env.repo / ".run/workers.tfplan"

    def answer(question):
        if question == APPLY_Q:
            plan.unlink()
            return "n\n"
        return "y\n"

    rc, err = interactive(env, args, answer)
    assert rc == 2 and "not applied: workers" in err
    assert "workers.tfplan is gone already" in err
    assert "was removed" not in err and "left where it is" not in err
    assert "apply" not in tf(env) and not plan.exists()


def test_an_interactive_decline_leaves_a_plan_another_run_wrote_while_it_asked(env):
    """S3-L2: the saved plan is removed on a decline only while it is still the one this run's guard passed."""
    args = scenario(env, "scale up")
    plan = env.repo / ".run/workers.tfplan"

    def answer(question):
        if question == APPLY_Q:
            plan.write_text("another run's plan")
            return "n\n"
        return "y\n"

    rc, err = interactive(env, args, answer)
    assert rc == 2 and "not applied: workers" in err
    assert "not this run's plan" in err and "left" in err
    assert "apply" not in tf(env)
    assert plan.read_text() == "another run's plan"
    assert lock_free(env)


def test_an_unattended_guarded_run_says_the_apply_question_once(env):
    args = scenario(env, "scale up")
    proc = guarded(env, *args, AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.count("Apply this plan to the workers stack?") == 1
    assert proc.stderr.count("Apply this plan to the workers stack? [AUTO_APPROVE=1: yes]") == 1
    assert "operator confirmed" not in proc.stderr


# ---- remove ----------------------------------------------------------------------------------------

def test_guarded_remove_guards_before_the_drain(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-3"])
    env.scenario["changes"] = {"workers": [*deletes("k3s-cpu-2"), *SHARED]}
    proc = guarded(env, "remove", "k3s-cpu-2", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert [c["argv"] for c in guard_calls(env)] == [["tfguard", "--remove", "k3s-cpu-2"]]
    assert index_of(env, lambda c: c["argv"][:1] == ["tfguard"]) < index_of(env, is_drain)
    assert [c for c in calls(env) if c[1][:1] != ["tfguard"]] == [
        kc(env, "drain", "k3s-cpu-2", *DRAIN_ARGS), kc(env, "delete", "node", "k3s-cpu-2", RT),
        ("gpuctl", ["sync-peers"])]
    assert tf(env) == ["init", "plan", "show", "show", "apply"]
    assert json.loads(env.tfvars.read_text()) == pool(["k3s-cpu-1", "k3s-cpu-3"])
    assert leftovers(env) == []


def test_guarded_remove_violation_drains_and_writes_nothing(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-3"])
    env.scenario["changes"] = {"workers": [*deletes("k3s-cpu-2"), rc('google_compute_instance.cp', ["update"])]}
    before = env.tfvars.read_text()
    proc = guarded(env, "remove", "k3s-cpu-2", AUTO_APPROVE="1")
    assert proc.returncode == 1 and "violation: google_compute_instance.cp" in proc.stderr
    assert not any(is_drain(c) for c in env.calls())
    assert env.tfvars.read_text() == before and "apply" not in tf(env)
    assert [c["argv"][:1] for c in env.calls("gpuctl")] == [["tfguard"]], "no sync-peers"
    assert leftovers(env) == []


def test_guarded_remove_of_a_worker_already_out_of_the_inventory(env):
    """A second run after a declined apply: the plan (no -var-file) still deletes k3s-cpu-2; guarded too."""
    set_pool(env, ["k3s-cpu-1"])
    env.scenario["changes"] = {"workers": deletes("k3s-cpu-2")}
    proc = guarded(env, "remove", "k3s-cpu-2", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert [c["argv"] for c in guard_calls(env)] == [["tfguard", "--remove", "k3s-cpu-2"]]
    assert tf(env)[-1] == "apply"


def test_guarded_remove_already_done_needs_no_guard(env):
    set_pool(env, ["k3s-cpu-1"])
    env.scenario["plan_rc"] = {"workers": 0}
    env.scenario["gone"] = ["k3s-cpu-2"]
    proc = guarded(env, "remove", "k3s-cpu-2")
    assert proc.returncode == 0, proc.stderr
    assert guard_calls(env) == [] and "apply" not in tf(env)


# ---- without --guarded, and usage ------------------------------------------------------------------

def test_without_guarded_no_guard_runs(env):
    """Core behaviour is unchanged: the fake's default plan (a bucket create) would fail the guard."""
    workers(env, ips(1, 2))
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = env.script("cpu-pool.sh", "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert guard_calls(env) == [] and tf(env) == ["init", "plan", "show", "apply"]
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    proc = env.script("cpu-pool.sh", "remove", "k3s-cpu-2", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert guard_calls(env) == []


@pytest.mark.parametrize("args", [["scale", "2", "--guard"], ["scale", "--guarded", "2"], ["scale", "2", "--guarded", "x"],
                                  ["remove", "k3s-cpu-2", "--yes"], ["render", "2", "--guarded"],
                                  ["stop", "--guarded"], ["remove", "k3s-cpu-1", "--guarded"]])
def test_guarded_usage(env, args):
    proc = env.script("cpu-pool.sh", *args)
    assert proc.returncode == 2 and env.calls() == []


def test_cpu_pool_applies_only_through_core_tf_apply_plan():
    """S3-m1: the guarded apply is core's tf_apply_plan (--confirmed once apply_plan asked), not a copy of it."""
    text = (Path(__file__).resolve().parents[2] / "scripts/cpu-pool.sh").read_text()
    assert "tf_apply_plan workers --confirmed" in text
    assert " apply -input" not in text and "apply_checked_plan" not in text
