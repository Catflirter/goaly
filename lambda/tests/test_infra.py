"""gpuctl.infra: every subprocess the reconciler runs, checked through a scripted FakeShell (no cloud, no cluster)."""
import inspect
import json
import sys
from pathlib import Path

import pytest

from fakes import FakeInfra, FakeShell, fail, ok
from gpuctl.infra import CommandError, Infra, SubprocessShell

ROOT = Path("/srv/goaly")
ENV = {"GCP_PROJECT": "demo-proj"}
BASE_OUTPUTS = ok(json.dumps({
    "cp_ip": {"value": "34.82.10.10", "type": "string", "sensitive": False},
    "cp_zone": {"value": "us-west1-b", "type": "string", "sensitive": False},
}))
NOT_FOUND = 'Error from server (NotFound): nodes "x" not found'


def public_methods(cls):
    return {n for n, v in vars(cls).items() if callable(v) and not n.startswith("_")}


def test_fake_infra_has_exactly_the_infra_methods_with_the_same_signatures():
    assert public_methods(FakeInfra) == public_methods(Infra)
    for name in public_methods(Infra):
        assert inspect.signature(getattr(FakeInfra, name)) == inspect.signature(getattr(Infra, name)), name


# --- terraform ----------------------------------------------------------------------------------

def test_tf_output_unwraps_values():
    sh = FakeShell([ok('{"cp_ip":{"value":"34.82.10.10","type":"string"}}')])
    assert Infra(sh, ROOT, {}).tf_output("base") == {"cp_ip": "34.82.10.10"}
    assert sh.calls[0].args == ["terraform", f"-chdir={ROOT}/infra/base", "output", "-json"]


def test_terraform_env_carries_project_and_state_bucket():
    sh = FakeShell([ok("{}")])
    Infra(sh, ROOT, {**ENV, "PATH": "/usr/bin"}).tf_output("workers")
    assert sh.calls[0].env == {"GCP_PROJECT": "demo-proj", "PATH": "/usr/bin",
                               "TF_VAR_project_id": "demo-proj", "TF_VAR_state_bucket": "demo-proj-k3s-tfstate"}


@pytest.mark.parametrize("stdout", ["not json", "[]", '{"cp_ip": "34.82.10.10"}'])
def test_tf_output_rejects_unexpected_output(stdout):
    with pytest.raises(ValueError):
        Infra(FakeShell([ok(stdout)]), ROOT, {}).tf_output("base")


def test_tf_output_refuses_unknown_stacks():
    with pytest.raises(ValueError):
        Infra(FakeShell([]), ROOT, {}).tf_output("../base")


def test_ensure_init_runs_init_once_per_uninitialised_stack(tmp_path):
    sh = FakeShell([ok()])
    infra = Infra(sh, tmp_path, ENV)
    infra.ensure_init("gpu-access")
    assert sh.calls[0].args == ["terraform", f"-chdir={tmp_path}/infra/gpu-access", "init", "-input=false",
                                "-backend-config=bucket=demo-proj-k3s-tfstate", "-backend-config=prefix=gpu-access"]
    assert sh.calls[0].env["TF_VAR_state_bucket"] == "demo-proj-k3s-tfstate"
    (tmp_path / "infra/gpu-access/.terraform").mkdir(parents=True)
    infra.ensure_init("gpu-access")
    assert len(sh.calls) == 1


def test_ensure_init_needs_gcp_project(tmp_path):
    sh = FakeShell([])
    with pytest.raises(ValueError, match="GCP_PROJECT"):
        Infra(sh, tmp_path, {}).ensure_init("base")
    assert sh.calls == []


def test_apply_gpu_access_sorted_compact_json(tmp_path):
    sh = FakeShell([ok(), ok()])
    Infra(sh, tmp_path, ENV).apply_gpu_access(["129.80.30.31", "129.80.30.30"])
    assert [c.args[2] for c in sh.calls] == ["init", "apply"]
    assert sh.calls[-1].args[:2] == ["terraform", f"-chdir={tmp_path}/infra/gpu-access"]
    assert sh.calls[-1].args[-1] == '-var=gpu_ips=["129.80.30.30","129.80.30.31"]'
    assert "-auto-approve" in sh.calls[-1].args and "-input=false" in sh.calls[-1].args


def test_apply_gpu_access_empty_list(tmp_path):
    (tmp_path / "infra/gpu-access/.terraform").mkdir(parents=True)
    sh = FakeShell([ok()])
    Infra(sh, tmp_path, ENV).apply_gpu_access([])
    assert sh.calls[-1].args[-1] == "-var=gpu_ips=[]"


@pytest.mark.parametrize("bad", [None, "", "129.80.30.0/24", "10.0.0.5", "2001:db8::1", "1.2.3.4 -x"])
def test_apply_gpu_access_never_passes_a_bad_ip_to_terraform(tmp_path, bad):  # Review Focus #2
    sh = FakeShell([])
    with pytest.raises(ValueError):
        Infra(sh, tmp_path, ENV).apply_gpu_access(["129.80.30.30", bad])
    assert sh.calls == []


def test_terraform_failure_raises_command_error(tmp_path):
    (tmp_path / "infra/gpu-access/.terraform").mkdir(parents=True)
    sh = FakeShell([fail(1, "Error: Invalid value for variable")])
    with pytest.raises(CommandError) as e:
        Infra(sh, tmp_path, ENV).apply_gpu_access(["129.80.30.30"])
    assert e.value.returncode == 1 and "Invalid value" in e.value.stderr and e.value.args[0] == "terraform"


# --- cp over IAP, kubectl -----------------------------------------------------------------------

def test_cp_ssh_uses_iap_and_zone():
    sh = FakeShell([BASE_OUTPUTS, ok("hello\n"), ok("again\n")])
    infra = Infra(sh, ROOT, ENV)
    assert infra.cp_ssh("sudo true", input="payload") == "hello\n"
    infra.cp_ssh("uptime")
    ssh = sh.calls[1]
    assert ssh.args == ["gcloud", "compute", "ssh", "k3s-cp", "--project", "demo-proj", "--zone", "us-west1-b",
                        "--tunnel-through-iap", "--quiet", "--command", "sudo true"]
    assert ssh.input == "payload"
    assert len(sh.calls) == 3 and sh.calls[2].args[-1] == "uptime"  # the zone is read once


def test_cp_ssh_failure_raises():
    sh = FakeShell([BASE_OUTPUTS, fail(255, "ssh: connect to host failed")])
    with pytest.raises(CommandError):
        Infra(sh, ROOT, ENV).cp_ssh("sudo true")


def test_kubectl_uses_the_cluster_kubeconfig():
    sh = FakeShell([ok(), ok()])
    Infra(sh, ROOT, {"HOME": "/home/op"}).kubectl("get", "nodes")
    Infra(sh, ROOT, {"HOME": "/home/op", "KUBECONFIG": "/tmp/kc"}).kubectl("get", "nodes")
    assert sh.calls[0].args == ["kubectl", "--kubeconfig", "/home/op/.kube/hybrid-gpu.yaml", "get", "nodes"]
    assert sh.calls[1].args == ["kubectl", "--kubeconfig", "/tmp/kc", "get", "nodes"]


def test_cluster_gpu_nodes_maps_gpu_nodes_to_their_external_ip():
    nodes = {"items": [
        {"metadata": {"name": "k3s-cp"}, "status": {"addresses": [{"type": "ExternalIP", "address": "34.82.10.10"}]}},
        {"metadata": {"name": "k3s-gpu-a6000-1"}, "status": {"addresses": [
            {"type": "InternalIP", "address": "10.0.0.5"}, {"type": "ExternalIP", "address": "129.80.30.30"}]}},
        {"metadata": {"name": "k3s-gpu-a6000-2"}, "status": {"addresses": [{"type": "Hostname", "address": "x"}]}},
    ]}
    sh = FakeShell([ok(json.dumps(nodes))])
    assert Infra(sh, ROOT, {"HOME": "/h"}).cluster_gpu_nodes() == {"k3s-gpu-a6000-1": "129.80.30.30",
                                                                   "k3s-gpu-a6000-2": None}
    assert sh.calls[0].args[-4:] == ["get", "nodes", "-o", "json"]


def test_drain_and_delete_commands():
    sh = FakeShell([ok(), ok()])
    Infra(sh, ROOT, {"HOME": "/h"}).drain_and_delete_node("k3s-gpu-a6000-1")
    assert [c.args[3:] for c in sh.calls] == [
        ["drain", "k3s-gpu-a6000-1", "--ignore-daemonsets", "--delete-emptydir-data", "--force", "--timeout=120s"],
        ["delete", "node", "k3s-gpu-a6000-1"],
    ]


def test_drain_and_delete_tolerate_notfound():  # Review Focus #3
    sh = FakeShell([fail(1, NOT_FOUND)] * 2)
    Infra(sh, ROOT, {}).drain_and_delete_node("x")
    assert len(sh.calls) == 2


def test_drain_other_errors_raise():
    sh = FakeShell([fail(1, "The connection to the server 127.0.0.1:6443 was refused - connection refused")])
    with pytest.raises(CommandError):
        Infra(sh, ROOT, {}).drain_and_delete_node("x")
    assert len(sh.calls) == 1  # never deletes a node it could not drain


def test_delete_other_errors_raise():
    sh = FakeShell([ok(), fail(1, "connection refused")])
    with pytest.raises(CommandError):
        Infra(sh, ROOT, {}).drain_and_delete_node("x")


# --- join-node, host keys, operator IP, cp guard ------------------------------------------------

def test_join_node_runs_the_script_and_raises_on_failure():
    sh = FakeShell([ok(), fail(3, "kubectl wait timed out")])
    infra = Infra(sh, ROOT, {})
    infra.join_node("k3s-gpu-a6000-1", "129.80.30.30", "a6000")
    assert sh.calls[0].args == [f"{ROOT}/scripts/join-node.sh", "--node", "k3s-gpu-a6000-1",
                                "--external-ip", "129.80.30.30", "--gpu-type", "a6000", "--skip-sync"]
    with pytest.raises(CommandError):
        infra.join_node("k3s-gpu-a6000-1", "129.80.30.30", "a6000")


@pytest.mark.parametrize("name, ip, gpu_type", [
    ("k3s-gpu-a6000-1", "1.2.3.4 -oProxyCommand=evil", "a6000"),
    ("k3s-gpu-a6000-1", None, "a6000"),
    ("k3s-gpu-a6000-1", "10.0.0.5", "a6000"),
    ("--help", "129.80.30.30", "a6000"),
    ("k3s-gpu-a6000-1", "129.80.30.30", "a6000; id"),
])
def test_join_node_refuses_bad_arguments(name, ip, gpu_type):
    sh = FakeShell([])
    with pytest.raises(ValueError):
        Infra(sh, ROOT, {}).join_node(name, ip, gpu_type)
    assert sh.calls == []


def test_forget_host_key(tmp_path):
    run = tmp_path / ".run"
    run.mkdir()
    (run / "known_hosts_k3s-gpu-a6000-1").write_text("129.80.30.30 ssh-ed25519 AAAA\n")
    (run / "known_hosts_k3s-gpu-a6000-2").write_text("keep\n")
    infra = Infra(FakeShell([]), tmp_path, {})
    infra.forget_host_key("k3s-gpu-a6000-1")
    infra.forget_host_key("k3s-gpu-a6000-1")  # already gone: no error
    assert [p.name for p in run.iterdir()] == ["known_hosts_k3s-gpu-a6000-2"]
    with pytest.raises(ValueError):
        infra.forget_host_key("../../etc/passwd")


def test_operator_ip_env_override_and_validation():
    assert Infra(FakeShell([]), ROOT, {"OPERATOR_IP": "73.15.40.40"}).detect_operator_ip() == "73.15.40.40"
    with pytest.raises(ValueError):
        Infra(FakeShell([]), ROOT, {"OPERATOR_IP": "10.0.0.1"}).detect_operator_ip()
    with pytest.raises(ValueError):
        Infra(FakeShell([ok("garbage")]), ROOT, {}).detect_operator_ip()


def test_operator_ip_is_detected_when_not_set():
    sh = FakeShell([ok("73.15.40.40\n")])
    assert Infra(sh, ROOT, {"OPERATOR_IP": ""}).detect_operator_ip() == "73.15.40.40"
    assert sh.calls[0].args == ["curl", "-fsS", "--max-time", "5", "https://checkip.amazonaws.com"]


@pytest.mark.parametrize("bad", ["127.0.0.1", "192.168.1.10", "2001:db8::1", "224.0.0.1", "73.15.40.0/24"])
def test_operator_ip_must_be_a_global_ipv4_address(bad):
    with pytest.raises(ValueError):
        Infra(FakeShell([]), ROOT, {"OPERATOR_IP": bad}).detect_operator_ip()


def test_cp_guard_elements_parses_nft_json():
    listing = {"nftables": [
        {"metainfo": {"version": "1.0.9", "json_schema_version": 1}},
        {"set": {"family": "inet", "name": "k3s_peers", "table": "k3s_guard", "type": "ipv4_addr", "handle": 2,
                 "elem": ["34.82.10.10", "10.10.0.2", "129.80.30.30"]}},
    ]}
    sh = FakeShell([BASE_OUTPUTS, ok(json.dumps(listing))])
    assert Infra(sh, ROOT, ENV).cp_guard_elements() == ["10.10.0.2", "34.82.10.10", "129.80.30.30"]
    assert sh.calls[1].args[-1] == "sudo nft -j list set inet k3s_guard k3s_peers"


def test_cp_guard_elements_of_an_empty_set():
    listing = {"nftables": [{"set": {"family": "inet", "name": "k3s_peers", "table": "k3s_guard"}}]}
    assert Infra(FakeShell([BASE_OUTPUTS, ok(json.dumps(listing))]), ROOT, ENV).cp_guard_elements() == []


@pytest.mark.parametrize("stdout", ["garbage", '{"nftables": []}', '{"nftables": [{"set": {"elem": [{"x": 1}]}}]}'])
def test_cp_guard_elements_rejects_unexpected_output(stdout):
    with pytest.raises(ValueError):
        Infra(FakeShell([BASE_OUTPUTS, ok(stdout)]), ROOT, ENV).cp_guard_elements()


# --- SubprocessShell ----------------------------------------------------------------------------

def test_subprocess_shell_runs_with_input_and_env():
    r = SubprocessShell().run([sys.executable, "-c", "import os, sys; print(sys.stdin.read() + os.environ['X'])"],
                              input="in-", env={"X": "env"})
    assert r.returncode == 0 and r.stdout == "in-env\n"


def test_subprocess_shell_failure_raises_unless_unchecked():
    args = [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
    with pytest.raises(CommandError) as e:
        SubprocessShell().run(args)
    assert (e.value.args, e.value.returncode, e.value.stderr) == (tuple(args), 3, "boom")
    assert "boom" in str(e.value) and "exit code 3" in str(e.value)
    assert SubprocessShell().run(args, check=False).returncode == 3


def test_subprocess_shell_missing_program_and_timeout_raise_command_error():
    with pytest.raises(CommandError):
        SubprocessShell().run(["gpuctl-no-such-program"])
    with pytest.raises(CommandError, match="timed out"):
        SubprocessShell().run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2)
