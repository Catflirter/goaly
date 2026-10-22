"""Offline tests for Task 14 (join-node.sh, cpu-pool.sh, the node SSH helpers in common.sh, the up.sh
workers stage) and the up.sh system (Task 15) and workloads (Task 16) stages (Ruling R38).

Same harness as test_ops_scripts.py: a repo copy whose terraform, gcloud, kubectl, helm, ssh, scp and
scripts/gpuctl are tests/scripts/fake_tools.py, which records every call. Nothing reaches a cloud or a cluster.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from gpuctl.infra import Infra

from test_ops_scripts import BUCKET, KUBECONFIG, OUTPUTS, PROJECT, READYZ, SSH_IAP, Env, base_sequence, non_tf

TOKEN = "K10fakejointoken0123456789abcdef::server:s3cr3t-join"
STAGING = "rm -rf /tmp/k3s-node && mkdir -m 0700 /tmp/k3s-node && [ ! -L /tmp/k3s-node ] && [ -O /tmp/k3s-node ]"
INSTALL = "sudo install -m 600 /dev/stdin /etc/rancher/k3s/token && sudo systemctl enable --now k3s-agent"
CONTAINERD_GREP = "sudo grep -A3 nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml"
READY = 'jsonpath={.status.conditions[?(@.type=="Ready")].status}'
RT = "--request-timeout=60s"
STOP_AGENT = "sudo systemctl stop k3s-agent"
GPU = "k3s-gpu-a6000-1"
GPU_IP = "129.80.1.2"
NODE_TOOLS = ("gcloud", "kubectl", "gpuctl", "helm", "ssh", "scp")


def worker_iap(zone="us-central1-a"):
    return ["--project", PROJECT, "--zone", zone, "--tunnel-through-iap", "--quiet",
            "--ssh-flag=-oConnectTimeout=20", "--ssh-flag=-oServerAliveInterval=30"]


def token_delete(name):
    """Runs on k3s-cp: the ids of every join token whose DESCRIPTION is exactly NAME, deleted there. The
    token list never leaves k3s-cp (test_token_delete_command_deletes_only_this_nodes_tokens runs it)."""
    return ("list=$(sudo k3s token list) && ids=$(printf '%s\\n' \"$list\" | awk -v d=" + name
            + " '!h { if (!NF) next; h = 1; for (i = 1; i <= NF; i++) if ($i == \"DESCRIPTION\") c = i; next }"
            " c && $c == d { split($1, t, \".\"); print t[1] } END { if (h && !c) exit 1 }') && for id in $ids;"
            " do sudo k3s token delete \"$id\" || exit 1; done")


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def kc(env, *args):
    return ("kubectl", ["--kubeconfig", str(env.kubeconfig), *args])


def get_node(env, name):
    """node_state's one kubectl call: the Ready condition of NAME, bounded by --request-timeout."""
    return kc(env, "get", "node", name, RT, "-o", READY)


def cp(command):
    return ("gcloud", ["compute", "ssh", "k3s-cp", *SSH_IAP, "--command", command])


def worker(name, command, zone="us-central1-a"):
    return ("gcloud", ["compute", "ssh", name, *worker_iap(zone), "--command", command])


def node_files(env):
    return sorted(str(p) for p in (env.repo.resolve() / "node").iterdir())


def gpu_opts(env, name=GPU):
    return ["-i", str(env.ssh_key), "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={env.repo.resolve()}/.run/known_hosts_{name}",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes"]


def gpu(env, command, name=GPU, ip=GPU_IP):
    return ("ssh", [*gpu_opts(env, name), f"ubuntu@{ip}", command])


def calls(env):
    """Every non-terraform call in order. The token pipeline starts `k3s token create` on the cp and the
    install on the node at the same time, so that pair is put in pipeline order whichever logged first."""
    out = [(c["tool"], c["argv"]) for c in env.calls() if c["tool"] in NODE_TOOLS]
    for i in range(len(out) - 1):
        if out[i][1][-1] == INSTALL and out[i + 1][1][-1].startswith("sudo k3s token create"):
            out[i], out[i + 1] = out[i + 1], out[i]
    return out


def cpu_join(env, name="k3s-cpu-1", ip="34.10.0.11", zone="us-central1-a", mtu="1460", sync=True):
    """The calls of a full join-node.sh for CPU worker NAME (spec §5.2, Task 14 brief steps 1-6)."""
    return [
        *([("gpuctl", ["sync-peers"])] if sync else []),
        get_node(env, name),
        worker(name, "true", zone),  # SSH is up (a VM made seconds ago may not have sshd yet: L3-1)
        worker(name, STAGING, zone),
        ("gcloud", ["compute", "scp", "--project", PROJECT, "--zone", zone, "--tunnel-through-iap", "--quiet",
                    "--recurse", "--scp-flag=-oConnectTimeout=20", *node_files(env), f"{name}:/tmp/k3s-node"]),
        worker(name, f"sudo bash /tmp/k3s-node/node-bootstrap.sh --server-ip 34.82.10.10 --node-name {name}"
                     f" --external-ip {ip} --underlay-mtu {mtu} --no-gpu", zone),
        worker(name, STOP_AGENT, zone),
        cp(f"sudo k3s token create --ttl 15m --description {name}"),
        worker(name, INSTALL, zone),
        kc(env, "wait", "--for=condition=Ready", f"node/{name}", "--timeout=10s"),
        cp(token_delete(name)),
    ]


def gpu_join(env):
    return [
        get_node(env, GPU),
        gpu(env, "true"),
        gpu(env, STAGING),
        ("scp", ["-r", *gpu_opts(env), *node_files(env), f"ubuntu@{GPU_IP}:/tmp/k3s-node"]),
        gpu(env, f"sudo bash /tmp/k3s-node/node-bootstrap.sh --server-ip 34.82.10.10 --node-name {GPU}"
                 f" --external-ip {GPU_IP} --underlay-mtu 1460 --gpu --gpu-type a6000"),
        gpu(env, STOP_AGENT),
        cp(f"sudo k3s token create --ttl 15m --description {GPU}"),
        gpu(env, INSTALL),
        kc(env, "wait", "--for=condition=Ready", f"node/{GPU}", "--timeout=10s"),
        cp(token_delete(GPU)),
        kc(env, "label", "node", GPU, "node-restriction.kubernetes.io/accelerator=nvidia", "--overwrite", RT),
        gpu(env, CONTAINERD_GREP),
    ]


def piped(env):
    path = Path(str(env.log) + ".stdin")
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def assert_token_only_piped(env, proc, targets):
    """The join token reached each target's install command on stdin and nowhere else: not in any
    tool's argv or logged environment, not in the script's output, not on the local disk."""
    assert [p["target"] for p in piped(env)] == targets
    assert all(p["stdin"] == TOKEN + "\n" for p in piped(env))
    assert TOKEN not in env.log.read_text()
    assert "s3cr3t" not in proc.stdout + proc.stderr
    for path in [*env.repo.rglob("*"), *env.home.rglob("*"), *env.tmpdir.rglob("*")]:
        if path.is_file():
            assert "s3cr3t" not in path.read_text(errors="replace"), path


def join(env, *args, **kw):
    return env.script("join-node.sh", *args, **kw)


# ---- join-node.sh: CPU workers --------------------------------------------------------------------

def test_join_cpu_worker_runs_the_whole_sequence(env):
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == cpu_join(env)
    assert_token_only_piped(env, proc, ["k3s-cpu-1"])
    # Step 1 reads the firewall inventory (terraform output) before anything else happens.
    first = env.calls()[0]
    assert first["tool"] == "terraform" and first["argv"][1:] == ["init", "-input=false", "-lockfile=readonly",
                                                                  "-reconfigure", f"-backend-config=bucket={BUCKET}",
                                                                  "-backend-config=prefix=workers"]
    outputs = [c["argv"][-1] for c in env.calls("terraform") if c["argv"][1] == "output"]
    assert outputs[0] == "worker_ips"
    assert "waiting up to 300s for: kc wait --for=condition=Ready node/k3s-cpu-1 --timeout=10s" in proc.stderr
    env.assert_no_lambda_key()


def test_join_waits_for_the_node_to_register(env):
    """`kubectl wait` fails at once for a node that has not registered yet, so it is retried for 300s."""
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--skip-sync")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == cpu_join(env, sync=False)


def test_join_passes_underlay_mtu_from_env(env):
    env.write_env(f"GCP_PROJECT={PROJECT}\nUNDERLAY_MTU=1380\n")
    assert join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11").returncode == 0
    assert calls(env) == cpu_join(env, mtu="1380")


def test_join_of_a_ready_node_only_syncs_and_exits(env):
    """Idempotent: a Ready node is left alone (no bootstrap, no token) unless --force."""
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [("gpuctl", ["sync-peers"]), get_node(env, "k3s-cpu-1")]
    assert piped(env) == []


def test_join_force_rejoins_a_ready_node(env):
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--force")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == cpu_join(env)


def test_join_of_a_registered_but_not_ready_node_bootstraps_again(env):
    env.scenario["registered"] = ["k3s-cpu-1"]
    assert join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11").returncode == 0
    assert calls(env) == cpu_join(env)


def test_join_refuses_an_ip_the_firewall_does_not_know(env):
    """Step 1: the IP must be the worker's own in the workers stack, or nothing happens."""
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.99")
    assert proc.returncode == 1
    assert "34.10.0.99" in proc.stderr and calls(env) == []


def test_join_refuses_a_worker_missing_from_the_workers_stack(env):
    proc = join(env, "--node", "k3s-cpu-2", "--external-ip", "34.10.0.11")
    assert proc.returncode == 1 and calls(env) == []


def test_join_stops_when_sync_peers_fails(env):
    env.scenario["sync_rc"] = 1
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11")
    assert proc.returncode != 0 and calls(env) == [("gpuctl", ["sync-peers"])]


def test_join_stops_when_the_api_cannot_be_reached(env):
    """A dead tunnel is not "node not registered": nothing is bootstrapped and no token is created."""
    env.scenario["api_down"] = True
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--skip-sync")
    assert proc.returncode == 1 and "make tunnel" in proc.stderr
    assert calls(env) == [get_node(env, "k3s-cpu-1")]


def test_join_reports_an_ssh_failure_while_staging(env):
    env.scenario["ssh_fail"] = [STAGING]
    env.scenario["ssh_fail_rc"] = 255
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--skip-sync")
    assert proc.returncode == 1
    assert "SSH to k3s-cpu-1 failed while preparing /tmp/k3s-node (exit 255)" in proc.stderr
    assert "make cpu-start" in proc.stderr, "a stopped worker has no SSH"
    assert "cannot create a private" not in proc.stderr
    assert calls(env)[-1] == worker("k3s-cpu-1", STAGING)


def test_join_gpu_node_ssh_failure_names_the_stale_host_key_file(env):
    """Lambda reuses IPs and names: a host key mismatch in .run/known_hosts_NAME is exit 255 too (X0-6)."""
    env.scenario["ssh_fail"] = [STAGING]
    env.scenario["ssh_fail_rc"] = 255
    proc = join(env, "--node", GPU, "--external-ip", GPU_IP, "--gpu-type", "a6000", "--skip-sync")
    assert proc.returncode == 1
    assert f"known_hosts_{GPU}" in proc.stderr


def test_join_waits_for_ssh_on_a_vm_created_seconds_ago(env):
    """L3-1: a VM made by the apply just before the join has no sshd yet (exit 255). The join waits for
    SSH (JOIN_SSH_TIMEOUT_S, default 300s, as up.sh base waits for k3s-cp) and then stages exactly once."""
    env.scenario["ssh_failures"] = 3
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--skip-sync")
    assert proc.returncode == 0, proc.stderr
    probe = worker("k3s-cpu-1", "true")
    expected = cpu_join(env, sync=False)
    assert calls(env) == [*expected[:1], probe, probe, probe, *expected[1:]]
    assert calls(env).count(worker("k3s-cpu-1", STAGING)) == 1
    assert "waiting up to 300s for: node_ssh k3s-cpu-1 true" in proc.stderr
    assert_token_only_piped(env, proc, ["k3s-cpu-1"])


def test_join_gpu_node_waits_for_ssh_after_the_instance_turns_active(env):
    env.scenario["ssh_failures"] = 2
    proc = gpuctl_join(env)
    assert proc.returncode == 0, proc.stderr
    probe = gpu(env, "true")
    expected = gpu_join(env)
    assert calls(env) == [*expected[:1], probe, probe, *expected[1:]]
    assert calls(env).count(gpu(env, STAGING)) == 1


def test_join_gives_up_when_ssh_never_comes_up(env):
    """The SSH wait is bounded: after JOIN_SSH_TIMEOUT_S the join dies naming the node, before any staging,
    bootstrap or token."""
    env.scenario["ssh_failures"] = 10**6
    start = time.time()
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--skip-sync",
                JOIN_SSH_TIMEOUT_S="1", WAIT_INTERVAL_S="1")
    assert time.time() - start < 15
    assert proc.returncode == 1
    assert "SSH to k3s-cpu-1 not available after 1s" in proc.stderr
    assert "make cpu-start" in proc.stderr, "a stopped worker has no SSH"
    seq = calls(env)
    assert seq[0] == get_node(env, "k3s-cpu-1")
    assert len(seq) >= 2 and all(c == worker("k3s-cpu-1", "true") for c in seq[1:]), "only SSH probes, retried"
    assert piped(env) == []


def test_join_gpu_node_gives_up_when_ssh_never_comes_up(env):
    env.scenario["ssh_failures"] = 10**6
    proc = gpuctl_join(env, JOIN_SSH_TIMEOUT_S="0")
    assert proc.returncode == 1
    assert f"SSH to {GPU} not available after 0s" in proc.stderr and f"known_hosts_{GPU}" in proc.stderr
    assert calls(env) == [get_node(env, GPU), gpu(env, "true")]


def test_join_of_a_ready_node_needs_no_ssh(env):
    """The SSH wait comes after the Ready check: a Ready node is left alone even while SSH is down."""
    env.scenario["ready"] = ["k3s-cpu-1"]
    env.scenario["ssh_failures"] = 10**6
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", JOIN_SSH_TIMEOUT_S="0")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [("gpuctl", ["sync-peers"]), get_node(env, "k3s-cpu-1")]


def test_join_stops_when_the_staging_directory_cannot_be_made_private(env):
    env.scenario["ssh_fail"] = [STAGING]
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--skip-sync")
    assert proc.returncode == 1
    assert "cannot create a private /tmp/k3s-node on k3s-cpu-1" in proc.stderr
    assert calls(env)[-1] == worker("k3s-cpu-1", STAGING)


def test_join_stops_when_the_bootstrap_fails_before_any_token(env):
    bootstrap = cpu_join(env)[5][1][-1]
    env.scenario["ssh_fail"] = [bootstrap]
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11")
    assert proc.returncode != 0
    assert calls(env) == cpu_join(env)[:6] and piped(env) == []


def test_join_deletes_the_token_when_the_node_never_becomes_ready(env):
    """The cleanup runs on failure too: a join that times out leaves no join token behind on k3s-cp."""
    env.scenario["never_ready"] = ["k3s-cpu-1"]
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", JOIN_READY_TIMEOUT_S="0")
    assert proc.returncode == 1 and "k3s-cpu-1 not Ready" in proc.stderr
    assert calls(env)[-1] == cp(token_delete("k3s-cpu-1"))
    assert [c for c in calls(env) if c == cp(token_delete("k3s-cpu-1"))] == [cp(token_delete("k3s-cpu-1"))]


def test_join_rerun_after_a_ready_timeout_restarts_the_agent_from_the_new_token(env):
    """A join that timed out leaves k3s-agent running with a token its cleanup deleted; k3s reads the
    token-file only at start and `enable --now` does not restart an active unit. The re-run stops
    the agent before the new token goes in, so `enable --now` starts it from that token."""
    env.scenario["never_ready"] = ["k3s-cpu-1"]
    first = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", JOIN_READY_TIMEOUT_S="0")
    assert first.returncode == 1 and "k3s-cpu-1 not Ready" in first.stderr
    env.log.unlink()
    env.scenario["never_ready"] = []
    env.scenario["registered"] = ["k3s-cpu-1"]  # it registered, but never became Ready
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == cpu_join(env)
    seq = calls(env)
    stop = seq.index(worker("k3s-cpu-1", STOP_AGENT))
    assert stop < seq.index(cp("sudo k3s token create --ttl 15m --description k3s-cpu-1")) < seq.index(
        worker("k3s-cpu-1", INSTALL))


def test_join_force_restarts_the_agent_from_the_new_token(env):
    env.scenario["ready"] = ["k3s-cpu-1"]
    assert join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--force").returncode == 0
    seq = calls(env)
    assert seq.index(worker("k3s-cpu-1", STOP_AGENT)) < seq.index(worker("k3s-cpu-1", INSTALL))


def test_join_stops_when_the_agent_cannot_be_stopped(env):
    """No new token is created for an agent that would keep running with the old one."""
    env.scenario["ssh_fail"] = [STOP_AGENT]
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--skip-sync")
    assert proc.returncode == 1 and "cannot stop k3s-agent" in proc.stderr
    assert calls(env) == cpu_join(env, sync=False)[:6] and piped(env) == []


def test_join_of_a_ready_node_ignores_kubectl_noise_on_stderr(env):
    """kubectl can succeed and still print a discovery error or a warning on stderr: only stdout is
    the Ready condition, so a Ready node is still left alone (C0-2)."""
    env.scenario["ready"] = ["k3s-cpu-1"]
    env.scenario["kubectl_stderr"] = ("E1022 10:00:00.000000 memcache.go:265] couldn't get current server API group"
                                      " list: the server is currently unable to handle the request")
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [("gpuctl", ["sync-peers"]), get_node(env, "k3s-cpu-1")]
    assert piped(env) == []


def test_join_deletes_the_token_when_installing_it_fails(env):
    env.scenario["install_rc"] = 1
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11")
    assert proc.returncode != 0
    assert calls(env)[-1] == cp(token_delete("k3s-cpu-1"))
    assert not [c for c in calls(env) if "wait" in c[1]]


def test_join_fails_when_the_token_cannot_be_deleted(env):
    env.scenario["ssh_fail"] = [token_delete("k3s-cpu-1")]
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11")
    assert proc.returncode == 1
    assert "expires" in proc.stderr


def test_token_delete_command_deletes_only_this_nodes_tokens(tmp_path):
    """Run the exact command join-node sends to k3s-cp against a stub `k3s token list`. DESCRIPTION is
    matched whole: k3s-cpu-10's token survives the deletion of k3s-cpu-1's (anchored name match)."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    deleted = tmp_path / "deleted"
    (stubs / "sudo").write_text('#!/bin/sh\nexec "$@"\n')
    (stubs / "k3s").write_text(f"""#!/bin/sh
if [ "$1 $2" = "token list" ]; then cat <<'EOF'
TOKEN                     TTL         EXPIRES                USAGES                   DESCRIPTION   EXTRA GROUPS
abc123.0123456789abcdef   14m         2026-10-22T10:15:00Z   authentication,signing   k3s-cpu-1     system:bootstrappers:k3s:default-node-token
def456                    14m         2026-10-22T10:15:00Z   authentication,signing   k3s-cpu-10    system:bootstrappers:k3s:default-node-token
ghi789                    <forever>   <never>                authentication,signing   <none>        system:bootstrappers:k3s:default-node-token
jkl012                    3m          2026-10-22T10:04:00Z   authentication,signing   k3s-cpu-1     system:bootstrappers:k3s:default-node-token
EOF
elif [ "$1 $2" = "token delete" ]; then echo "$3" >> "{deleted}"
else exit 1; fi
""")
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    run_env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}"}
    proc = subprocess.run(["bash", "-c", token_delete("k3s-cpu-1")], env=run_env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    # Only the id part of an id.secret TOKEN column: the secret never reaches sudo's log (X0-4).
    assert deleted.read_text().split() == ["abc123", "jkl012"]
    deleted.unlink()
    proc = subprocess.run(["bash", "-c", token_delete("k3s-cpu-2")], env=run_env, capture_output=True, text=True)
    assert proc.returncode == 0 and not deleted.exists(), "no token of that node: nothing to delete"
    # No tokens at all (no output): nothing to delete, which is not an error.
    (stubs / "k3s").write_text('#!/bin/sh\nexit 0\n')
    proc = subprocess.run(["bash", "-c", token_delete("k3s-cpu-1")], env=run_env, capture_output=True, text=True)
    assert proc.returncode == 0 and not deleted.exists()
    # `k3s token list` itself fails: so does the deletion.
    (stubs / "k3s").write_text('#!/bin/sh\nexit 1\n')
    proc = subprocess.run(["bash", "-c", token_delete("k3s-cpu-1")], env=run_env, capture_output=True, text=True)
    assert proc.returncode != 0
    # A list without a DESCRIPTION column (an unexpected k3s format) fails instead of deleting nothing quietly.
    (stubs / "k3s").write_text('#!/bin/sh\necho "ID NAME"\necho "x k3s-cpu-1"\n')
    proc = subprocess.run(["bash", "-c", token_delete("k3s-cpu-1")], env=run_env, capture_output=True, text=True)
    assert proc.returncode != 0


# ---- join-node.sh: GPU nodes (scripts/gpuctl's call) ---------------------------------------------

def gpuctl_join(env, *extra, **kw):
    """Exactly what lambda/gpuctl/infra.py Infra.join_node runs."""
    return join(env, "--node", GPU, "--external-ip", GPU_IP, "--gpu-type", "a6000", "--skip-sync", *extra, **kw)


def test_join_gpu_node_as_gpuctl_calls_it(env):
    proc = gpuctl_join(env)
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == gpu_join(env)
    assert_token_only_piped(env, proc, [f"ubuntu@{GPU_IP}"])
    outputs = [c["argv"][-1] for c in env.calls("terraform") if c["argv"][1] == "output"]
    assert outputs[0] == "gpu_ips", "step 1 checks the gpu-access stack"
    # The SSH target is the IP join-node was given (GPU_SSH_IP): no Lambda API call through `gpuctl ip`.
    assert not env.calls("gpuctl")
    assert (env.repo / ".run").is_dir() and oct((env.repo / ".run").stat().st_mode & 0o777) == "0o700"
    env.assert_no_lambda_key()


def test_join_node_accepts_exactly_the_call_gpuctl_makes(env):
    """The argv comes from lambda/gpuctl/infra.py Infra.join_node itself, not a copy of it."""
    recorded = []

    class Recorder:
        def run(self, args, **kw):
            recorded.append(list(args))
            return subprocess.CompletedProcess(args, 0, "", "")

    Infra(Recorder(), env.repo, {"GCP_PROJECT": PROJECT}).join_node(GPU, GPU_IP, "a6000")
    [argv] = recorded
    assert argv[0] == str(env.repo / "scripts/join-node.sh")
    proc = env.run(argv)
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == gpu_join(env)


def test_join_gpu_node_restarts_k3s_agent_once_when_containerd_lacks_nvidia(env):
    env.scenario["containerd"] = ["", "[plugins.nvidia]"]
    proc = gpuctl_join(env)
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [*gpu_join(env), gpu(env, "sudo systemctl restart k3s-agent"),
                          kc(env, "wait", "--for=condition=Ready", f"node/{GPU}", "--timeout=10s"),
                          gpu(env, CONTAINERD_GREP)]


def test_join_gpu_node_fails_when_containerd_never_has_nvidia(env):
    env.scenario["containerd"] = [""]
    proc = gpuctl_join(env)
    assert proc.returncode == 1 and "nvidia" in proc.stderr
    assert calls(env)[-1] == gpu(env, CONTAINERD_GREP)
    assert len([c for c in calls(env) if c == gpu(env, "sudo systemctl restart k3s-agent")]) == 1


def test_join_gpu_node_fails_when_the_containerd_config_cannot_be_read(env):
    env.scenario["ssh_fail"] = [CONTAINERD_GREP]
    env.scenario["ssh_fail_rc"] = 255
    proc = gpuctl_join(env)
    assert proc.returncode == 1
    assert gpu(env, "sudo systemctl restart k3s-agent") not in calls(env)


def test_join_gpu_node_refuses_an_ip_without_firewall_rules(env):
    proc = join(env, "--node", GPU, "--external-ip", "129.80.1.3", "--gpu-type", "a6000", "--skip-sync")
    assert proc.returncode == 1 and calls(env) == []


def test_join_gpu_node_ready_is_left_alone(env):
    env.scenario["ready"] = [GPU]
    proc = gpuctl_join(env)
    assert proc.returncode == 0 and calls(env) == [get_node(env, GPU)]


def test_join_gpu_node_needs_the_ssh_key(env):
    env.write_env(f"GCP_PROJECT={PROJECT}\nSSH_PRIVATE_KEY_PATH={env.home}/nope\n")
    proc = gpuctl_join(env)
    assert proc.returncode == 2 and "SSH_PRIVATE_KEY_PATH" in proc.stderr
    assert not env.calls("ssh") and piped(env) == []


@pytest.mark.parametrize("args", [
    [],
    ["--node", "k3s-cpu-1"],
    ["--external-ip", "34.10.0.11"],
    ["--node", "k3s-cp", "--external-ip", "34.10.0.11"],             # anchored: k3s-cp is not k3s-cpu-<n>
    ["--node", "k3s-cpu-1x", "--external-ip", "34.10.0.11"],
    ["--node", "k3s-cpu-01", "--external-ip", "34.10.0.11"],
    ["--node", "xk3s-cpu-1", "--external-ip", "34.10.0.11"],
    ["--node", "k3s-cpu-1", "--external-ip", "34.10.0.11; reboot"],
    ["--node", "k3s-cpu-1", "--external-ip", "300.10.0.11"],
    ["--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--gpu-type", "a6000"],  # a CPU node has no GPU type
    ["--node", GPU, "--external-ip", GPU_IP],                                      # a GPU node needs one
    ["--node", GPU, "--external-ip", GPU_IP, "--gpu-type", "A6000; reboot"],
    ["--node", "k3s-gpu-a6000", "--external-ip", GPU_IP, "--gpu-type", "a6000"],
    ["--node", "k3s-cpu-1", "--external-ip", "34.10.0.11", "--bogus"],
    ["--node", "k3s-cpu-1", "--external-ip"],
])
def test_join_usage_errors_touch_nothing(env, args):
    proc = join(env, *args)
    assert proc.returncode == 2 and "usage" in proc.stderr and env.calls() == []


@pytest.mark.parametrize("mtu", ["1460; reboot", "0"])
def test_join_refuses_a_bad_underlay_mtu(env, mtu):
    env.write_env(f"GCP_PROJECT={PROJECT}\nUNDERLAY_MTU='{mtu}'\n")
    proc = join(env, "--node", "k3s-cpu-1", "--external-ip", "34.10.0.11")
    assert proc.returncode == 2 and env.calls() == []


# ---- common.sh: node SSH helpers ------------------------------------------------------------------

def test_worker_ssh_goes_through_iap_in_the_workers_zone(env):
    env.scenario["outputs"] = {**OUTPUTS, "workers": {**OUTPUTS["workers"],
                                                      "worker_zones": {"k3s-cpu-1": "us-central1-f"}}}
    proc = env.lib("load_env; worker_ssh k3s-cpu-1 'uptime'; node_ssh k3s-cpu-1 'id'")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [worker("k3s-cpu-1", "uptime", "us-central1-f"), worker("k3s-cpu-1", "id", "us-central1-f")]


def test_worker_ssh_refuses_a_worker_without_a_zone(env):
    proc = env.lib("load_env; worker_ssh k3s-cpu-2 true")
    assert proc.returncode != 0 and env.calls("gcloud") == []


def test_worker_scp_copies_recursively_through_iap(env):
    proc = env.lib("load_env; node_scp k3s-cpu-1 a b /tmp/x")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [("gcloud", ["compute", "scp", "--project", PROJECT, "--zone", "us-central1-a",
                                      "--tunnel-through-iap", "--quiet", "--recurse",
                                      "--scp-flag=-oConnectTimeout=20", "a", "b", "k3s-cpu-1:/tmp/x"])]


def test_gpu_ssh_asks_gpuctl_for_the_ip_without_gpu_ssh_ip(env):
    env.scenario["gpu_ip"] = "129.80.9.9"
    proc = env.lib(f"load_env; gpu_ssh {GPU} uptime")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [("gpuctl", ["ip", GPU]), gpu(env, "uptime", ip="129.80.9.9")]


def test_gpu_ssh_uses_gpu_ssh_ip(env):
    proc = env.lib(f"load_env; node_ssh {GPU} uptime && node_scp {GPU} a /tmp/x", GPU_SSH_IP=GPU_IP)
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [gpu(env, "uptime"), ("scp", ["-r", *gpu_opts(env), "a", f"ubuntu@{GPU_IP}:/tmp/x"])]


@pytest.mark.parametrize("answer", ["", "null", "10.0.0.1; reboot", "1.2.3"])
def test_gpu_ssh_refuses_a_bad_ip(env, answer):
    env.scenario["gpu_ip"] = answer
    proc = env.lib(f"load_env; gpu_ssh {GPU} uptime")
    assert proc.returncode != 0 and not env.calls("ssh")


# k3s-cp is accepted since Task 18 (node_ssh runs cp_ssh for it: checks B3/B5 run on every node).
@pytest.mark.parametrize("name", ["k3s-cp2", "k3s-cpu-", "k3s-cpu-1 x", "k3s-gpu-a6000", "node-1", "k3s-cpu-1;id"])
def test_node_ssh_refuses_names_outside_the_worker_pools(env, name):
    proc = env.lib(f"load_env; node_ssh '{name}' true", GPU_SSH_IP=GPU_IP)
    assert proc.returncode != 0
    assert not env.calls("gcloud") and not env.calls("ssh")


# ---- cpu-pool.sh ---------------------------------------------------------------------------------

def pool(names, custom=None):
    return {"cpu_workers": {n: (custom or {}).get(n, {"zone": "us-central1-a", "machine_type": "e2-small"})
                            for n in names}}


def set_pool(env, names):
    env.tfvars.write_text(json.dumps(pool(names)))


def set_workers(env, ips):
    env.scenario["outputs"] = {**OUTPUTS, "workers": {
        "worker_ips": ips, "worker_zones": {n: "us-central1-a" for n in ips}}}


def tf(env, stack="workers"):
    chdir = f"-chdir={env.repo.resolve()}/infra/{stack}"
    return [a[1] for a in env.argvs("terraform") if a[0] == chdir and a[1] != "output"]


def index_of(env, predicate):
    return next(i for i, c in enumerate(env.calls()) if predicate(c))


def is_tf(sub):
    return lambda c: c["tool"] == "terraform" and c["argv"][1:2] == [sub]


def is_drain(c):
    return c["tool"] == "kubectl" and c["argv"][2:3] in (["drain"], ["delete"])


def planned(env):
    """The inventory the last plan was given with -var-file (fake_tools copies it), or None."""
    path = Path(str(env.log) + ".var-file")
    return json.loads(path.read_text()) if path.exists() else None


def leftovers(env):
    """Saved plans and planned inventories left in .run/ (each run removes its own)."""
    run = env.repo / ".run"
    return sorted(p.name for p in run.iterdir() if p.suffix in (".tfplan", ".json")) if run.exists() else []


def vm_status(name, zone="us-central1-a"):
    return ("gcloud", ["compute", "instances", "describe", name, "--project", PROJECT, "--zone", zone,
                       "--format=value(status)"])


def test_scale_up_writes_applies_syncs_then_joins_the_new_worker(env):
    set_workers(env, {"k3s-cpu-1": "34.10.0.11", "k3s-cpu-2": "34.10.0.12"})
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = env.script("cpu-pool.sh", "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(env.tfvars.read_text()) == pool(["k3s-cpu-1", "k3s-cpu-2"])
    assert tf(env) == ["init", "plan", "show", "apply"]
    # After the apply: k3s-cpu-1 was in the pool and is registered, so it is not joined again (scale
    # step 4: only workers that have not joined). One sync-peers before the join (the new IP is in the
    # peer set before its agent connects), join-node --skip-sync for k3s-cpu-2, sync-peers last (step 5).
    assert calls(env) == [get_node(env, "k3s-cpu-1"), ("gpuctl", ["sync-peers"]),
                          *cpu_join(env, "k3s-cpu-2", "34.10.0.12", sync=False), ("gpuctl", ["sync-peers"])]
    assert index_of(env, lambda c: c["argv"][1:2] == ["apply"]) < index_of(env, lambda c: c["tool"] == "gpuctl")
    assert_token_only_piped(env, proc, ["k3s-cpu-2"])
    env.assert_no_lambda_key()


def test_scale_up_never_destroys_under_auto_approve(env):
    """Growing the pool must not destroy anything: a plan that does is refused unattended (--no-destroy)."""
    env.scenario["deletes"] = {"workers": ["google_compute_instance.worker[\"k3s-cpu-1\"]"]}
    proc = env.script("cpu-pool.sh", "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 2
    assert "apply" not in tf(env) and not env.calls("gpuctl") and not env.calls("gcloud")


def test_scale_down_drains_and_deletes_before_the_apply(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-3"])
    env.scenario["deletes"] = {"workers": ["google_compute_instance.worker[\"k3s-cpu-3\"]",
                                           "google_compute_instance.worker[\"k3s-cpu-2\"]"]}
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = env.script("cpu-pool.sh", "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(env.tfvars.read_text()) == pool(["k3s-cpu-1"])
    drains = [c for c in calls(env) if c[1][2:3] in (["drain"], ["delete"])]
    assert drains == [
        kc(env, "drain", "k3s-cpu-2", "--ignore-daemonsets", "--delete-emptydir-data", "--force", "--timeout=120s"),
        kc(env, "delete", "node", "k3s-cpu-2", RT),
        kc(env, "drain", "k3s-cpu-3", "--ignore-daemonsets", "--delete-emptydir-data", "--force", "--timeout=120s"),
        kc(env, "delete", "node", "k3s-cpu-3", RT)]
    last_delete = max(i for i, c in enumerate(env.calls()) if c["argv"][2:4] == ["delete", "node"])
    assert last_delete < index_of(env, is_tf("apply")), "drain before the VM goes"
    # X1-2/C1-1: the plan (with the new inventory) is checked before anything is drained.
    assert index_of(env, is_tf("show")) < index_of(env, is_drain)
    assert planned(env) == pool(["k3s-cpu-1"])
    assert tf(env) == ["init", "plan", "show", "apply"]
    assert calls(env)[-2:] == [get_node(env, "k3s-cpu-1"), ("gpuctl", ["sync-peers"])]
    assert leftovers(env) == []


DRAIN_ARGS = ("--ignore-daemonsets", "--delete-emptydir-data", "--force", "--timeout=120s")


def worker_resources(name):
    """The workers stack's per-worker resources (for_each over var.cpu_workers, keyed by name)."""
    return [f'google_compute_instance.worker["{name}"]', f'google_compute_address.worker["{name}"]',
            f'google_compute_firewall.worker_to_cp["{name}"]', f'google_compute_firewall.worker_mesh["{name}"]']


def test_scale_down_rerun_after_a_declined_apply_drains_again(env):
    """The drain prompt said yes, the apply prompt no: k3s-cpu-2's VM keeps running (its kubelet
    re-registers the Node) while the file already says 1. The re-run takes the workers to remove from
    what the stack has deployed, so it drains and deletes k3s-cpu-2 again before the destroy."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    set_workers(env, {"k3s-cpu-1": "34.10.0.11", "k3s-cpu-2": "34.10.0.12"})
    env.scenario["deletes"] = {"workers": worker_resources("k3s-cpu-2")}
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    first = env.script("cpu-pool.sh", "scale", "1", stdin="y\nn\n")
    assert first.returncode == 2 and "apply" not in tf(env)
    assert json.loads(env.tfvars.read_text()) == pool(["k3s-cpu-1"])
    env.log.unlink()
    proc = env.script("cpu-pool.sh", "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert [c for c in calls(env) if c[1][2:3] in (["drain"], ["delete"])] == [
        kc(env, "drain", "k3s-cpu-2", *DRAIN_ARGS), kc(env, "delete", "node", "k3s-cpu-2", RT)]
    last_delete = max(i for i, c in enumerate(env.calls()) if c["argv"][2:4] == ["delete", "node"])
    assert last_delete < index_of(env, is_tf("apply"))
    assert tf(env)[-1] == "apply"


def test_scale_down_refuses_a_plan_that_also_destroys_k3s_cpu_1(env):
    """Under AUTO_APPROVE=1 the apply may destroy only the workers just drained (X0-1). The plan is
    checked before the drain (X1-2/C1-1): a refused plan drains nothing and leaves the inventory alone."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    before = env.tfvars.read_text()
    env.scenario["deletes"] = {"workers": [*worker_resources("k3s-cpu-2"),
                                           'google_compute_instance.worker["k3s-cpu-1"]']}
    proc = env.script("cpu-pool.sh", "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 2 and "apply" not in tf(env)
    assert not any(is_drain(c) for c in env.calls()), "nothing drained for a refused plan"
    assert env.tfvars.read_text() == before
    assert planned(env) == pool(["k3s-cpu-1"])
    assert "nothing was drained" in proc.stderr
    assert leftovers(env) == []
    unexpected = proc.stderr.split("NOT expected", 1)[1]
    assert 'google_compute_instance.worker["k3s-cpu-1"]' in unexpected
    assert 'worker["k3s-cpu-2"]' not in unexpected
    assert not (env.repo / ".run/workers.tfplan").exists()


def test_scale_down_with_an_unexpected_destroy_still_asks_interactively(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    env.scenario["deletes"] = {"workers": ['google_compute_instance.worker["k3s-cpu-1"]']}
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = env.script("cpu-pool.sh", "scale", "1", stdin="y\ny\n")
    assert proc.returncode == 0, proc.stderr
    assert "NOT expected" in proc.stderr and tf(env)[-1] == "apply"


def test_scale_never_removes_k3s_cpu_1(env):
    """An inventory edited by hand without k3s-cpu-1: the deployed k3s-cpu-1 would be removed; refused."""
    set_pool(env, ["k3s-cpu-2", "k3s-cpu-3"])
    proc = env.script("cpu-pool.sh", "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 2 and "k3s-cpu-1 is never removed" in proc.stderr
    assert calls(env) == [] and "plan" not in tf(env)


@pytest.mark.parametrize("status", ["TERMINATED", "STOPPED", "SUSPENDED"])
def test_scale_does_not_join_a_stopped_worker(env, status):
    """Scale step 4 joins only workers that have not joined: k3s-cpu-2, stopped with make cpu-stop, is
    registered but NotReady and its VM is not running (no SSH); it is left alone and the new k3s-cpu-3
    is joined."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    set_workers(env, {"k3s-cpu-1": "34.10.0.11", "k3s-cpu-2": "34.10.0.12", "k3s-cpu-3": "34.10.0.13"})
    env.scenario["ready"] = ["k3s-cpu-1"]
    env.scenario["registered"] = ["k3s-cpu-2"]
    env.scenario["vm_status"] = {"k3s-cpu-2": status}
    proc = env.script("cpu-pool.sh", "scale", "3", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [get_node(env, "k3s-cpu-1"), get_node(env, "k3s-cpu-2"), vm_status("k3s-cpu-2"),
                          ("gpuctl", ["sync-peers"]),
                          *cpu_join(env, "k3s-cpu-3", "34.10.0.13", sync=False), ("gpuctl", ["sync-peers"])]
    assert f"k3s-cpu-2 is registered but NotReady and its VM is {status}" in proc.stderr


def test_scale_rerun_after_a_join_timeout_joins_the_running_worker_again(env):
    """X1-1: the first scale to 3 times out waiting for k3s-cpu-3 to be Ready (it registered, NotReady).
    The re-run sees k3s-cpu-3 registered but NotReady with its VM RUNNING: that is a join that never
    finished, not a stopped worker, so join-node.sh runs again (agent stopped, then a fresh token)."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    set_workers(env, {"k3s-cpu-1": "34.10.0.11", "k3s-cpu-2": "34.10.0.12", "k3s-cpu-3": "34.10.0.13"})
    env.scenario["ready"] = ["k3s-cpu-1", "k3s-cpu-2"]
    env.scenario["never_ready"] = ["k3s-cpu-3"]
    first = env.script("cpu-pool.sh", "scale", "3", AUTO_APPROVE="1", JOIN_READY_TIMEOUT_S="0")
    assert first.returncode != 0 and "join-node.sh failed for k3s-cpu-3" in first.stderr
    assert json.loads(env.tfvars.read_text()) == pool(["k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-3"])
    env.log.unlink()
    env.scenario["never_ready"] = []
    env.scenario["registered"] = ["k3s-cpu-3"]
    env.scenario["plan_rc"] = {"workers": 0}
    proc = env.script("cpu-pool.sh", "scale", "3", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    rejoin = cpu_join(env, "k3s-cpu-3", "34.10.0.13", sync=False)
    assert calls(env) == [get_node(env, "k3s-cpu-1"), get_node(env, "k3s-cpu-2"), get_node(env, "k3s-cpu-3"),
                          vm_status("k3s-cpu-3"), ("gpuctl", ["sync-peers"]), *rejoin, ("gpuctl", ["sync-peers"])]
    stop = calls(env).index(worker("k3s-cpu-3", STOP_AGENT))
    assert stop < calls(env).index(cp("sudo k3s token create --ttl 15m --description k3s-cpu-3"))


@pytest.mark.parametrize("status", ["STOPPING", "PROVISIONING", "STAGING", "REPAIRING"])
def test_scale_fails_for_a_not_ready_worker_whose_vm_is_neither_running_nor_stopped(env, status):
    """X1-1: a NotReady worker whose VM is on its way somewhere is not joined (no SSH yet), but the scale
    does not report success either: the rest is done, then it exits 1 naming the worker."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    set_workers(env, {"k3s-cpu-1": "34.10.0.11", "k3s-cpu-2": "34.10.0.12"})
    env.scenario["ready"] = ["k3s-cpu-1"]
    env.scenario["registered"] = ["k3s-cpu-2"]
    env.scenario["vm_status"] = {"k3s-cpu-2": status}
    env.scenario["plan_rc"] = {"workers": 0}
    proc = env.script("cpu-pool.sh", "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 1
    assert calls(env) == [get_node(env, "k3s-cpu-1"), get_node(env, "k3s-cpu-2"), vm_status("k3s-cpu-2"),
                          ("gpuctl", ["sync-peers"])]
    assert f"k3s-cpu-2 ({status})" in proc.stderr


def test_scale_refuses_an_unexpected_vm_status(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    set_workers(env, {"k3s-cpu-1": "34.10.0.11", "k3s-cpu-2": "34.10.0.12"})
    env.scenario["ready"] = ["k3s-cpu-1"]
    env.scenario["registered"] = ["k3s-cpu-2"]
    env.scenario["vm_status"] = {"k3s-cpu-2": "RUNNING; reboot"}
    env.scenario["plan_rc"] = {"workers": 0}
    proc = env.script("cpu-pool.sh", "scale", "2", AUTO_APPROVE="1")
    assert proc.returncode == 1 and "unexpected status" in proc.stderr
    assert not env.calls("gpuctl")


def test_scale_joins_a_pool_worker_the_cluster_does_not_know(env):
    """A worker in the pool whose earlier join never registered it is joined again."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    set_workers(env, {"k3s-cpu-1": "34.10.0.11", "k3s-cpu-2": "34.10.0.12"})
    env.scenario["ready"] = ["k3s-cpu-1"]
    env.scenario["plan_rc"] = {"workers": 0}
    proc = env.script("cpu-pool.sh", "scale", "2")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [get_node(env, "k3s-cpu-1"), get_node(env, "k3s-cpu-2"), ("gpuctl", ["sync-peers"]),
                          *cpu_join(env, "k3s-cpu-2", "34.10.0.12", sync=False), ("gpuctl", ["sync-peers"])]


def test_scale_down_declined_changes_nothing(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    before = env.tfvars.read_text()
    env.scenario["deletes"] = {"workers": worker_resources("k3s-cpu-2")}
    proc = env.script("cpu-pool.sh", "scale", "1", stdin="n\n")
    assert proc.returncode == 2
    # The plan is shown before the question (X1-2); declining drains, writes and applies nothing.
    assert 'google_compute_instance.worker["k3s-cpu-2"]' in proc.stderr.split("[y/N]", 1)[0]
    assert env.tfvars.read_text() == before and calls(env) == [] and "apply" not in tf(env)
    assert "nothing was drained" in proc.stderr and leftovers(env) == []


def test_scale_down_of_a_node_already_gone_from_the_cluster_succeeds(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    env.scenario["gone"] = ["k3s-cpu-2"]
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = env.script("cpu-pool.sh", "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(env.tfvars.read_text()) == pool(["k3s-cpu-1"])


def test_scale_down_stops_when_a_drain_fails(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    before = env.tfvars.read_text()
    env.scenario["drain_fail"] = ["k3s-cpu-2"]
    proc = env.script("cpu-pool.sh", "scale", "1", AUTO_APPROVE="1")
    assert proc.returncode == 1 and "disruption budget" in proc.stderr
    assert env.tfvars.read_text() == before and "apply" not in tf(env)
    assert leftovers(env) == []


def test_scale_to_the_current_size_is_quiet(env):
    """Re-run: nothing written, a plan without changes applies nothing, every Ready worker is left alone."""
    before = env.tfvars.read_text()
    mtime = env.tfvars.stat().st_mtime_ns
    env.scenario["plan_rc"] = {"workers": 0}
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = env.script("cpu-pool.sh", "scale", "1")  # no AUTO_APPROVE, no stdin: any prompt would refuse
    assert proc.returncode == 0, proc.stderr
    assert env.tfvars.read_text() == before and env.tfvars.stat().st_mtime_ns == mtime
    assert "apply" not in tf(env)
    assert calls(env) == [get_node(env, "k3s-cpu-1"), ("gpuctl", ["sync-peers"])]


def test_scale_joins_workers_in_numeric_order(env):
    names = [f"k3s-cpu-{i}" for i in (1, 10, 9, 2)]
    set_pool(env, names)
    set_workers(env, {n: f"34.10.0.{i}" for i, n in enumerate(names, start=11)})
    env.scenario["ready"] = names
    env.scenario["plan_rc"] = {"workers": 0}
    proc = env.script("cpu-pool.sh", "scale", "4")
    assert proc.returncode == 0, proc.stderr
    gets = [c[1][4] for c in calls(env) if c[1][2:4] == ["get", "node"]]
    assert gets == ["k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-9", "k3s-cpu-10"]


@pytest.mark.parametrize("n", ["", "0", "x", "-1"])
def test_scale_usage(env, n):
    proc = env.script("cpu-pool.sh", "scale", n)
    assert proc.returncode == 2 and env.calls() == []


def test_remove_drains_deletes_then_applies_and_syncs(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-3"])
    env.scenario["deletes"] = {"workers": ["google_compute_instance.worker[\"k3s-cpu-2\"]"]}
    proc = env.script("cpu-pool.sh", "remove", "k3s-cpu-2", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(env.tfvars.read_text()) == pool(["k3s-cpu-1", "k3s-cpu-3"])
    assert calls(env) == [
        kc(env, "drain", "k3s-cpu-2", "--ignore-daemonsets", "--delete-emptydir-data", "--force", "--timeout=120s"),
        kc(env, "delete", "node", "k3s-cpu-2", RT),
        ("gpuctl", ["sync-peers"])]
    assert tf(env) == ["init", "plan", "show", "apply"]
    assert index_of(env, lambda c: c["argv"][2:4] == ["delete", "node"]) < index_of(env, is_tf("apply"))
    assert index_of(env, is_tf("show")) < index_of(env, is_drain)
    assert planned(env) == pool(["k3s-cpu-1", "k3s-cpu-3"])
    assert leftovers(env) == []


def test_remove_applies_every_resource_of_the_removed_worker_unattended(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    env.scenario["deletes"] = {"workers": worker_resources("k3s-cpu-2")}
    proc = env.script("cpu-pool.sh", "remove", "k3s-cpu-2", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert tf(env)[-1] == "apply" and "NOT expected" not in proc.stderr


@pytest.mark.parametrize("other", ['google_compute_instance.worker["k3s-cpu-1"]',
                                   'google_compute_address.worker["k3s-cpu-3"]',
                                   'google_compute_firewall.worker_mesh["k3s-cpu-20"]',
                                   "google_compute_firewall.cp_to_workers"])
def test_remove_refuses_a_plan_that_destroys_anything_else_unattended(env, other):
    """X0-1: only k3s-cpu-2's own resources may go; k3s-cpu-20 is not k3s-cpu-2 (whole key match)."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-3"])
    env.scenario["deletes"] = {"workers": [*worker_resources("k3s-cpu-2"), other]}
    before = env.tfvars.read_text()
    proc = env.script("cpu-pool.sh", "remove", "k3s-cpu-2", AUTO_APPROVE="1")
    assert proc.returncode == 2 and "apply" not in tf(env)
    assert other in proc.stderr.split("NOT expected", 1)[1]
    assert not env.calls("gpuctl")
    # X1-2: refused before the drain; the inventory and the cluster are untouched.
    assert not any(is_drain(c) for c in env.calls()) and env.tfvars.read_text() == before
    assert "nothing was drained" in proc.stderr and leftovers(env) == []


def test_tf_apply_destroy_only_needs_a_name(env):
    (env.repo / "infra/workers").mkdir(exist_ok=True)
    proc = env.lib("load_env; tf_apply workers --destroy-only", AUTO_APPROVE="1")
    assert proc.returncode != 0 and env.calls("terraform") == []


def test_join_workers_refuses_a_name_without_an_ip(env):
    proc = env.lib("load_env; join_workers k3s-cpu-9")
    assert proc.returncode != 0 and "k3s-cpu-9 is not in" in proc.stderr
    assert calls(env) == []


def test_remove_twice_the_second_run_changes_nothing(env):
    """Review Focus #3: a target that is already gone counts as success; the second run is a no-op."""
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    assert env.script("cpu-pool.sh", "remove", "k3s-cpu-2", AUTO_APPROVE="1").returncode == 0
    after_first = env.tfvars.read_text()
    env.log.unlink()
    env.scenario["gone"] = ["k3s-cpu-2"]
    env.scenario["plan_rc"] = {"workers": 0}
    proc = env.script("cpu-pool.sh", "remove", "k3s-cpu-2")  # no prompt may be needed
    assert proc.returncode == 0, proc.stderr
    assert env.tfvars.read_text() == after_first
    assert "apply" not in tf(env)


def test_remove_declined_changes_nothing(env):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    before = env.tfvars.read_text()
    proc = env.script("cpu-pool.sh", "remove", "k3s-cpu-2", stdin="")
    assert proc.returncode == 2 and env.tfvars.read_text() == before
    assert calls(env) == [] and "apply" not in tf(env) and leftovers(env) == []


@pytest.mark.parametrize("name", ["k3s-cpu-1", "k3s-cp", "", "k3s-cpu-2x", "k3s-gpu-a6000-1"])
def test_remove_refuses_k3s_cpu_1_and_non_workers(env, name):
    set_pool(env, ["k3s-cpu-1", "k3s-cpu-2"])
    proc = env.script("cpu-pool.sh", "remove", name, AUTO_APPROVE="1")
    assert proc.returncode == 2 and env.calls() == []


@pytest.mark.parametrize("action", ["stop", "start"])
def test_stop_start_every_worker_per_zone(env, action):
    env.scenario["outputs"] = {**OUTPUTS, "workers": {
        "worker_ips": {}, "worker_zones": {"k3s-cpu-10": "us-central1-a", "k3s-cpu-1": "us-central1-a",
                                           "k3s-cpu-9": "us-central1-a", "k3s-cpu-2": "us-central1-b"}}}
    proc = env.script("cpu-pool.sh", action)
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == [
        ("gcloud", ["compute", "instances", action, "k3s-cpu-1", "k3s-cpu-9", "k3s-cpu-10", "--project", PROJECT,
                    "--zone", "us-central1-a", "--quiet"]),
        ("gcloud", ["compute", "instances", action, "k3s-cpu-2", "--project", PROJECT, "--zone", "us-central1-b",
                    "--quiet"])]


def test_stop_refuses_a_bad_zone_output(env):
    env.scenario["outputs"] = {**OUTPUTS, "workers": {"worker_ips": {}, "worker_zones": {"k3s-cpu-1": "x; reboot"}}}
    proc = env.script("cpu-pool.sh", "stop")
    assert proc.returncode != 0 and env.calls("gcloud") == []


@pytest.mark.parametrize("args", [[], ["nosuch"], ["stop", "extra"], ["remove"]])
def test_cpu_pool_usage(env, args):
    proc = env.script("cpu-pool.sh", *args)
    assert proc.returncode == 2 and "usage" in proc.stderr and env.calls() == []


# ---- up.sh: workers, system, workloads ------------------------------------------------------------

def test_up_workers_applies_syncs_and_joins_every_worker(env):
    proc = env.script("up.sh", "workers", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    assert tf(env) == ["init", "plan", "show", "apply"]
    # sync-peers before the joins (spec §5.2 step 1) and last (Task 14 brief, up.sh workers stage).
    assert calls(env) == [*cpu_join(env), ("gpuctl", ["sync-peers"])]
    assert_token_only_piped(env, proc, ["k3s-cpu-1"])
    env.assert_no_lambda_key()


def test_up_workers_rerun_is_quiet(env):
    env.scenario["plan_rc"] = {"workers": 0}
    env.scenario["ready"] = ["k3s-cpu-1"]
    proc = env.script("up.sh", "workers")
    assert proc.returncode == 0, proc.stderr
    assert "apply" not in tf(env)
    assert calls(env) == [("gpuctl", ["sync-peers"]), get_node(env, "k3s-cpu-1"), ("gpuctl", ["sync-peers"])]


def test_up_workers_never_destroys_a_worker_under_auto_approve(env):
    """A worker removed from the inventory by hand was never drained: make up refuses to destroy it."""
    env.scenario["deletes"] = {"workers": ["google_compute_instance.worker[\"k3s-cpu-2\"]"]}
    proc = env.script("up.sh", "workers", AUTO_APPROVE="1")
    assert proc.returncode == 2 and "apply" not in tf(env) and calls(env) == []


def system_sequence(env):
    sys_dir = env.repo.resolve() / "k8s/system"
    return [
        kc(env, "apply", "-f", f"{sys_dir}/metrics-server.yaml", "-f", f"{sys_dir}/nvidia-gpu-runtimeclass.yaml",
           "-f", f"{sys_dir}/nvidia-device-plugin-namespace.yaml", RT),
        ("helm", ["repo", "add", "nvdp", "https://nvidia.github.io/k8s-device-plugin"]),
        ("helm", ["repo", "update", "nvdp"]),
        ("helm", ["--kubeconfig", str(env.kubeconfig), "upgrade", "-i", "nvdp", "nvdp/nvidia-device-plugin",
                  "--namespace", "nvidia-device-plugin", "--version", "0.20.1", "-f", f"{sys_dir}/nvdp-values.yaml",
                  "--wait", "--timeout", "5m"]),
        kc(env, "-n", "kube-system", "rollout", "status", "deployment/metrics-server", "--timeout=300s"),
    ]


def workloads_sequence(env):
    k8s = env.repo.resolve() / "k8s"
    return [
        kc(env, "apply", "-f", f"{k8s}/nettest/", "-f", f"{k8s}/hello/", "-f", f"{k8s}/gpu-test/namespace.yaml", RT),
        kc(env, "-n", "nettest", "rollout", "status", "daemonset/echo", "--timeout=300s"),
        kc(env, "-n", "nettest", "rollout", "status", "daemonset/probe", "--timeout=300s"),
        kc(env, "-n", "hello", "rollout", "status", "deployment/hello", "--timeout=300s"),
    ]


def test_up_system_applies_the_manifests_and_the_device_plugin(env):
    proc = env.script("up.sh", "system")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == system_sequence(env)
    assert env.calls("terraform") == []
    env.assert_no_lambda_key()


def test_up_system_bounds_the_helm_repo_commands():
    """helm repo add/update have no timeout of their own (X0-5); upgrade has --wait --timeout 5m."""
    text = (ROOT / "scripts/up.sh").read_text()
    assert "timeout 120 helm repo add nvdp https://nvidia.github.io/k8s-device-plugin\n" in text
    assert "timeout 120 helm repo update nvdp\n" in text


def test_up_system_stops_when_helm_fails(env):
    env.scenario["helm_rc"] = 1
    proc = env.script("up.sh", "system")
    assert proc.returncode != 0 and calls(env) == system_sequence(env)[:4]


def test_up_workloads_applies_and_waits_for_the_rollouts(env):
    proc = env.script("up.sh", "workloads")
    assert proc.returncode == 0, proc.stderr
    assert calls(env) == workloads_sequence(env)
    assert env.calls("terraform") == []


def test_up_all_runs_base_workers_system_workloads_in_order(env):
    proc = env.script("up.sh", "all", AUTO_APPROVE="1")
    assert proc.returncode == 0, proc.stderr
    chdirs = [a[0] for a in env.argvs("terraform") if a[1] == "apply"]
    assert chdirs == [f"-chdir={env.repo.resolve()}/infra/base", f"-chdir={env.repo.resolve()}/infra/workers"]
    non_tf(env)  # its checks: one tunnel, started after the kubeconfig fetch and probed before the first wait
    everything = [c for c in calls(env) if c[1][2:] != READYZ and "-N" not in c[1]]
    assert everything == [*base_sequence(env), *cpu_join(env), ("gpuctl", ["sync-peers"]), *system_sequence(env),
                          *workloads_sequence(env)]
    assert env.kubeconfig.read_text() == KUBECONFIG


# ---- Makefile ------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("args,command", [
    (["cpu-scale", "N=3"], 'scripts/cpu-pool.sh scale "3"'),
    (["cpu-remove", "NAME=k3s-cpu-2"], 'scripts/cpu-pool.sh remove "k3s-cpu-2"'),
    (["cpu-stop"], "scripts/cpu-pool.sh stop"),
    (["cpu-start"], "scripts/cpu-pool.sh start"),
    (["sync-peers"], "scripts/gpuctl sync-peers")])
def test_make_cpu_pool_targets(args, command):
    proc = subprocess.run(["make", "--no-print-directory", "-n", "-f", str(ROOT / "Makefile"), *args],
                          capture_output=True, text=True, cwd="/")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{ROOT}/{command}"


def test_make_help_lists_one_sync_peers_target():
    text = (ROOT / "Makefile").read_text()
    assert text.count("\nsync-peers:") == 1
