"""gpuctl.infra: every subprocess the reconciler runs, checked through a scripted FakeShell (no cloud, no cluster)."""
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fakes import FakeInfra, FakeShell, fail, ok
from gpuctl.infra import (
    CP_SSH_TIMEOUT_S, TF_INIT_MARKER, TF_INIT_TIMEOUT_S, TF_READ_TIMEOUT_S, CommandError, Infra, MissingStackError,
    SubprocessShell, posix_cksum,
)
from gpuctl.planner import gpu_type_label

ROOT = Path("/srv/goaly")
REPO = Path(__file__).resolve().parents[2]  # this checkout, for its committed lock files
ENV = {"GCP_PROJECT": "demo-proj"}
BASE_OUTPUTS = ok(json.dumps({
    "cp_ip": {"value": "34.82.10.10", "type": "string", "sensitive": False},
    "cp_zone": {"value": "us-west1-b", "type": "string", "sensitive": False},
}))
NOT_FOUND = 'Error from server (NotFound): nodes "x" not found'


def initialised(root, *stacks, bucket="demo-proj-k3s-tfstate"):
    """Mark infra/<stack> as initialised against `bucket`, the way a successful ensure_init (or tf_init) leaves it:
    "gcs <bucket> <stack> lock <cksum of .terraform.lock.hcl>", here for a stack without a lock file."""
    for stack in stacks:
        (root / "infra" / stack / ".terraform").mkdir(parents=True, exist_ok=True)
        marker = root / "infra" / stack / ".terraform" / TF_INIT_MARKER
        marker.write_text(f"gcs {bucket} {stack} lock no-lock-file\n")
    return root


def cksum_binary(path):
    """What scripts/lib/common.sh tf_init records: `cksum < FILE`, its trailing newline stripped as $(...) does."""
    with open(path, "rb") as f:
        return subprocess.run(["cksum"], stdin=f, capture_output=True, text=True, check=True).stdout.rstrip("\n")


def init_args(root, stack, bucket="demo-proj-k3s-tfstate"):
    return ["terraform", f"-chdir={root}/infra/{stack}", "init", "-no-color", "-input=false",
            "-lockfile=readonly",  # never rewrite the committed .terraform.lock.hcl (R26(b))
            "-reconfigure", f"-backend-config=bucket={bucket}", f"-backend-config=prefix={stack}"]


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
    assert sh.calls[0].args == ["terraform", f"-chdir={ROOT}/infra/base", "output", "-no-color", "-json"]


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


def test_ensure_init_runs_init_once_and_records_a_marker(tmp_path):
    (tmp_path / "infra/gpu-access").mkdir(parents=True)
    sh = FakeShell([ok()])
    infra = Infra(sh, tmp_path, ENV)
    infra.ensure_init("gpu-access")
    assert sh.calls[0].args == init_args(tmp_path, "gpu-access")
    assert sh.calls[0].env["TF_VAR_state_bucket"] == "demo-proj-k3s-tfstate"
    # the same file and text scripts/lib/common.sh tf_init writes, so both agree on "initialised"
    marker = tmp_path / "infra/gpu-access/.terraform" / TF_INIT_MARKER
    assert marker.read_text() == "gcs demo-proj-k3s-tfstate gpu-access lock no-lock-file\n"
    infra.ensure_init("gpu-access")
    Infra(sh, tmp_path, ENV).ensure_init("gpu-access")  # a later run trusts the marker too
    assert len(sh.calls) == 1


def test_a_terraform_dir_from_an_offline_validate_is_not_initialised(tmp_path):  # B-1
    # make tf-test used to run `terraform init -backend=false` in the stack, leaving .terraform/providers and no
    # backend: every later terraform output/apply then failed with "Backend initialization required".
    providers = tmp_path / "infra/workers/.terraform/providers/registry.terraform.io/hashicorp/google"
    providers.mkdir(parents=True)
    sh = FakeShell([ok()])
    Infra(sh, tmp_path, ENV).ensure_init("workers")
    assert [c.args for c in sh.calls] == [init_args(tmp_path, "workers")]  # -reconfigure: whatever was there
    assert ((tmp_path / "infra/workers/.terraform" / TF_INIT_MARKER).read_text()
            == "gcs demo-proj-k3s-tfstate workers lock no-lock-file\n")


@pytest.mark.parametrize("marker", [
    "gcs other-proj-k3s-tfstate base lock no-lock-file",
    "gcs demo-proj-k3s-tfstate workers lock no-lock-file",
    "gcs demo-proj-k3s-tfstate base lock 4294967295 0",  # initialised with a lock file that is gone since
    "gcs demo-proj-k3s-tfstate base",  # the lock-less text gpuctl wrote before MK-1
    "",
])
def test_a_marker_for_another_bucket_stack_or_lock_file_triggers_a_reinit(tmp_path, marker):  # B-1, MK-1
    (tmp_path / "infra/base/.terraform").mkdir(parents=True)
    (tmp_path / "infra/base/.terraform" / TF_INIT_MARKER).write_text(marker + "\n")
    sh = FakeShell([ok()])
    Infra(sh, tmp_path, ENV).ensure_init("base")
    assert [c.args for c in sh.calls] == [init_args(tmp_path, "base")]
    assert ((tmp_path / "infra/base/.terraform" / TF_INIT_MARKER).read_text()
            == "gcs demo-proj-k3s-tfstate base lock no-lock-file\n")


# --- the init marker's lock-file checksum (MK-1) -------------------------------------------------
# scripts/lib/common.sh tf_init records `cksum < .terraform.lock.hcl` (Task 10 X1-2: init again after a provider
# bump). gpuctl computes the very same text in pure Python, or each would re-initialise the stack the other set up.

@pytest.mark.parametrize("data, want", [
    (b"", "4294967295 0"),
    (b"123456789", "930766865 9"),  # the CRC-32/CKSUM check value
    (b"\x00" * 256, None),  # the length bytes go in least-significant first: checked against the binary below
], ids=["empty", "check-value", "two-length-bytes"])
def test_posix_cksum_known_values(tmp_path, data, want):
    if want is None:
        (tmp_path / "data").write_bytes(data)
        want = cksum_binary(tmp_path / "data")
    assert posix_cksum(data) == want


@pytest.mark.parametrize("path", [*sorted(REPO.glob("infra/*/.terraform.lock.hcl")), REPO / "Makefile", None],
                         ids=lambda p: "empty-file" if p is None else str(p.relative_to(REPO)))
def test_posix_cksum_matches_the_cksum_binary(tmp_path, path):
    if path is None:
        path = tmp_path / "empty"
        path.write_bytes(b"")
    assert path.is_file()
    assert posix_cksum(path.read_bytes()) == cksum_binary(path)


def test_the_cksum_comparison_covers_the_committed_lock_files():  # the glob above must not silently match nothing
    assert {p.parent.name for p in REPO.glob("infra/*/.terraform.lock.hcl")} >= {"base", "gpu-access", "workers"}


def test_ensure_init_records_the_lock_file_cksum_like_tf_init(tmp_path):
    stack_dir = tmp_path / "infra/gpu-access"
    stack_dir.mkdir(parents=True)
    shutil.copy2(REPO / "infra/gpu-access/.terraform.lock.hcl", stack_dir / ".terraform.lock.hcl")
    sh = FakeShell([ok()])
    Infra(sh, tmp_path, ENV).ensure_init("gpu-access")
    lock_sum = cksum_binary(stack_dir / ".terraform.lock.hcl")
    marker = stack_dir / ".terraform" / TF_INIT_MARKER
    assert marker.read_text() == f"gcs demo-proj-k3s-tfstate gpu-access lock {lock_sum}\n"


def test_ensure_init_reinitialises_when_the_committed_lock_file_changed(tmp_path):  # X1-2, on the gpuctl side
    stack_dir = tmp_path / "infra/base"
    stack_dir.mkdir(parents=True)
    lockfile = stack_dir / ".terraform.lock.hcl"
    lockfile.write_text('provider "registry.terraform.io/hashicorp/google" {\n  version = "8.6.0"\n}\n')
    sh = FakeShell([ok(), ok(), ok()])
    infra = Infra(sh, tmp_path, ENV)
    infra.ensure_init("base")
    infra.ensure_init("base")
    assert len(sh.calls) == 1, "an unchanged lock file is still trusted"
    lockfile.write_text('provider "registry.terraform.io/hashicorp/google" {\n  version = "8.7.0"\n}\n')
    infra.ensure_init("base")
    infra.ensure_init("base")
    assert [c.args for c in sh.calls] == [init_args(tmp_path, "base")] * 2, "a pulled provider bump is installed"
    lockfile.unlink()
    infra.ensure_init("base")
    assert len(sh.calls) == 3, "a removed lock file counts as a change"
    marker = stack_dir / ".terraform" / TF_INIT_MARKER
    assert marker.read_text() == "gcs demo-proj-k3s-tfstate base lock no-lock-file\n"


def test_a_failed_reinit_removes_the_stale_marker(tmp_path):  # B-1: never trusted again after GCP_PROJECT changed
    initialised(tmp_path, "base", bucket="other-proj-k3s-tfstate")
    sh = FakeShell([fail(1, "Error: Failed to get existing workspaces"), ok()])
    infra = Infra(sh, tmp_path, ENV)
    with pytest.raises(CommandError):
        infra.ensure_init("base")
    assert not (tmp_path / "infra/base/.terraform").exists()
    infra.ensure_init("base")
    assert [c.args for c in sh.calls] == [init_args(tmp_path, "base")] * 2


def test_ensure_init_needs_gcp_project(tmp_path):
    (tmp_path / "infra/base").mkdir(parents=True)
    sh = FakeShell([])
    with pytest.raises(ValueError, match="GCP_PROJECT"):
        Infra(sh, tmp_path, {}).ensure_init("base")
    assert sh.calls == []


def test_ensure_init_of_a_stack_missing_from_the_checkout_raises_command_error(tmp_path):
    sh = FakeShell([])
    with pytest.raises(MissingStackError, match="infra/workers does not exist") as raised:
        Infra(sh, tmp_path, ENV).ensure_init("workers")
    assert isinstance(raised.value, CommandError)
    assert sh.calls == []


class InitThenFail(FakeShell):
    """terraform init that writes .terraform/terraform.tfstate (the backend config) and then fails or is killed."""

    def __init__(self, stack_dir, error):
        super().__init__([ok()])  # the retry succeeds
        self.stack_dir, self.error = stack_dir, error

    def run(self, args, **kwargs):
        if self.error is not None:
            (self.stack_dir / ".terraform").mkdir()
            (self.stack_dir / ".terraform" / "terraform.tfstate").write_text("{}")
            error, self.error = self.error, None
            self.calls.append(args)
            raise error
        return super().run(args, **kwargs)


@pytest.mark.parametrize("error", [CommandError(["terraform", "init"], -1, "timed out after 900.0s"),
                                   KeyboardInterrupt()], ids=["timeout", "interrupt"])
def test_a_failed_init_leaves_no_partial_terraform_dir(tmp_path, error):  # C1-1 / X1-1
    stack_dir = tmp_path / "infra/gpu-access"
    stack_dir.mkdir(parents=True)
    (stack_dir / "main.tf").write_text("")
    sh = InitThenFail(stack_dir, error)
    infra = Infra(sh, tmp_path, ENV)
    with pytest.raises(type(error)):
        infra.ensure_init("gpu-access")
    assert not (stack_dir / ".terraform").exists() and (stack_dir / "main.tf").exists()
    infra.ensure_init("gpu-access")  # the next call runs init again instead of trusting the leftover
    assert len(sh.calls) == 2 and sh.calls[1].args[2] == "init"


def test_apply_gpu_access_sorted_compact_json(tmp_path):
    (tmp_path / "infra/gpu-access").mkdir(parents=True)
    sh = FakeShell([ok(), ok()])
    Infra(sh, tmp_path, ENV).apply_gpu_access(["129.80.30.31", "129.80.30.30"], max_nodes=3)
    assert [c.args[2:4] for c in sh.calls] == [["init", "-no-color"], ["apply", "-no-color"]]
    assert sh.calls[-1].args[:2] == ["terraform", f"-chdir={tmp_path}/infra/gpu-access"]
    assert sh.calls[-1].args[-1] == '-var=gpu_ips=["129.80.30.30","129.80.30.31"]'
    assert "-auto-approve" in sh.calls[-1].args and "-input=false" in sh.calls[-1].args


def test_apply_gpu_access_passes_max_nodes_alongside_gpu_ips(tmp_path):  # Ruling R27
    initialised(tmp_path, "gpu-access")
    sh = FakeShell([ok()])
    Infra(sh, tmp_path, ENV).apply_gpu_access(["129.80.30.30"], max_nodes=5)
    assert sh.calls[-1].args[-2:] == ["-var=max_nodes=5", '-var=gpu_ips=["129.80.30.30"]']


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "3", None])
def test_apply_gpu_access_refuses_a_bad_max_nodes(tmp_path, bad):
    sh = FakeShell([])
    with pytest.raises(ValueError, match="max_nodes"):
        Infra(sh, tmp_path, ENV).apply_gpu_access(["129.80.30.30"], max_nodes=bad)
    with pytest.raises(ValueError, match="max_nodes"):
        FakeInfra().apply_gpu_access(["129.80.30.30"], max_nodes=bad)
    assert sh.calls == []


def test_apply_gpu_access_deduplicates_and_sorts_numerically(tmp_path):
    initialised(tmp_path, "gpu-access")
    sh = FakeShell([ok()])
    Infra(sh, tmp_path, ENV).apply_gpu_access(["129.80.30.30", "9.1.1.1", "129.80.30.30"], max_nodes=3)
    assert sh.calls[-1].args[-1] == '-var=gpu_ips=["9.1.1.1","129.80.30.30"]'


def test_fake_infra_normalises_gpu_ips_like_infra():  # S0-2: same list as the real adapter passes to Terraform
    infra = FakeInfra()
    infra.apply_gpu_access(["129.80.30.30", "9.1.1.1", "129.80.30.30"], max_nodes=3)
    assert infra.applied_gpu_ips == [["9.1.1.1", "129.80.30.30"]]
    assert infra.applied_max_nodes == [3]
    assert infra.events == ["tf:['9.1.1.1', '129.80.30.30']"]


@pytest.mark.parametrize("bad", [None, "10.0.0.5", "not-an-ip"])
def test_fake_infra_refuses_bad_gpu_ips_like_infra(bad):  # Review Focus #2
    infra = FakeInfra()
    with pytest.raises(ValueError):
        infra.apply_gpu_access([bad], max_nodes=3)
    with pytest.raises(ValueError):
        infra.join_node("k3s-gpu-a6000-1", bad, "a6000")
    assert infra.applied_gpu_ips == [] and infra.joined == [] and infra.events == []


def test_fake_infra_join_node_accepts_underscore_gpu_types():
    infra = FakeInfra()
    infra.join_node("k3s-gpu-a100-1", "129.80.30.30", gpu_type_label("gpu_8x_a100_80gb_sxm4"))
    assert infra.joined == [("k3s-gpu-a100-1", "129.80.30.30", "a100_80gb_sxm4")]


def test_apply_gpu_access_empty_list(tmp_path):
    initialised(tmp_path, "gpu-access")
    sh = FakeShell([ok()])
    Infra(sh, tmp_path, ENV).apply_gpu_access([], max_nodes=3)
    assert sh.calls[-1].args[-1] == "-var=gpu_ips=[]"


@pytest.mark.parametrize("bad", [None, "", "129.80.30.0/24", "10.0.0.5", "2001:db8::1", "1.2.3.4 -x"])
def test_apply_gpu_access_never_passes_a_bad_ip_to_terraform(tmp_path, bad):  # Review Focus #2
    sh = FakeShell([])
    with pytest.raises(ValueError):
        Infra(sh, tmp_path, ENV).apply_gpu_access(["129.80.30.30", bad], max_nodes=3)
    assert sh.calls == []


def test_terraform_failure_raises_command_error(tmp_path):
    initialised(tmp_path, "gpu-access")
    sh = FakeShell([fail(1, "Error: Invalid value for variable")])
    with pytest.raises(CommandError) as e:
        Infra(sh, tmp_path, ENV).apply_gpu_access(["129.80.30.30"], max_nodes=3)
    assert e.value.returncode == 1 and "Invalid value" in e.value.stderr and e.value.args[0] == "terraform"


# --- cp over IAP, kubectl -----------------------------------------------------------------------

def test_cp_ssh_uses_iap_and_zone(tmp_path):
    sh = FakeShell([BASE_OUTPUTS, ok("hello\n"), ok("again\n")])
    infra = Infra(sh, initialised(tmp_path, "base"), ENV)
    assert infra.cp_ssh("sudo true", input="payload") == "hello\n"
    infra.cp_ssh("uptime")
    ssh = sh.calls[1]
    assert ssh.args == ["gcloud", "compute", "ssh", "k3s-cp", "--project", "demo-proj", "--zone", "us-west1-b",
                        "--tunnel-through-iap", "--quiet", "--command", "sudo true"]
    assert ssh.input == "payload"
    assert len(sh.calls) == 3 and sh.calls[2].args[-1] == "uptime"  # the zone is read once


def test_cp_ssh_in_a_fresh_checkout_initialises_base_before_reading_the_zone(tmp_path):  # B-1 (c)
    (tmp_path / "infra/base").mkdir(parents=True)
    sh = FakeShell([ok(), BASE_OUTPUTS, ok("hello\n")])
    assert Infra(sh, tmp_path, ENV).cp_ssh("sudo true") == "hello\n"
    assert [c.args[:3] for c in sh.calls] == [
        ["terraform", f"-chdir={tmp_path}/infra/base", "init"],
        ["terraform", f"-chdir={tmp_path}/infra/base", "output"], ["gcloud", "compute", "ssh"]]


def test_reading_base_outputs_caches_the_cp_zone():  # C0-4: sync-peers reads infra/base once
    sh = FakeShell([BASE_OUTPUTS, ok("hello\n")])
    infra = Infra(sh, ROOT, ENV)
    assert infra.tf_output("base")["cp_zone"] == "us-west1-b"
    infra.cp_ssh("sudo true")
    assert len(sh.calls) == 2 and sh.calls[1].args[:4] == ["gcloud", "compute", "ssh", "k3s-cp"]


def test_an_invalid_cp_zone_in_base_outputs_is_not_cached(tmp_path):
    bad = ok(json.dumps({"cp_zone": {"value": "us-west1-b; id", "type": "string"}}))
    sh = FakeShell([bad, bad])
    infra = Infra(sh, initialised(tmp_path, "base"), ENV)
    infra.tf_output("base")
    with pytest.raises(ValueError, match="cp_zone"):
        infra.cp_ssh("sudo true")


def test_reads_are_bounded_by_a_timeout_and_apply_is_not(tmp_path):  # X0-4
    (tmp_path / "infra/gpu-access").mkdir(parents=True)
    initialised(tmp_path, "base")
    sh = FakeShell([BASE_OUTPUTS, ok(), ok(), ok()])
    infra = Infra(sh, tmp_path, ENV)
    infra.cp_ssh("sudo true")
    infra.apply_gpu_access(["129.80.30.30"], max_nodes=3)
    assert [(c.args[0], c.args[2], c.timeout) for c in sh.calls] == [
        ("terraform", "output", TF_READ_TIMEOUT_S), ("gcloud", "ssh", CP_SSH_TIMEOUT_S),
        ("terraform", "init", TF_INIT_TIMEOUT_S), ("terraform", "apply", None),
    ]


def test_child_processes_never_get_the_lambda_api_key(tmp_path):  # X0-3
    initialised(tmp_path, "gpu-access", "base")
    env = {**ENV, "HOME": "/h", "PATH": "/usr/bin", "LAMBDA_API_KEY": "secret_abc"}
    sh = FakeShell([BASE_OUTPUTS, ok(), ok(), ok(), ok(), ok("73.15.40.40\n")])
    infra = Infra(sh, tmp_path, env)
    infra.cp_ssh("sudo true")
    infra.apply_gpu_access(["129.80.30.30"], max_nodes=3)
    infra.kubectl("get", "nodes")
    infra.join_node("k3s-gpu-a6000-1", "129.80.30.30", "a6000")
    infra.detect_operator_ip()
    assert [c.args[0] for c in sh.calls] == ["terraform", "gcloud", "terraform", "kubectl",
                                             f"{tmp_path}/scripts/join-node.sh", "curl"]
    for call in sh.calls:
        assert "LAMBDA_API_KEY" not in call.env and call.env["PATH"] == "/usr/bin", call.args
    assert env["LAMBDA_API_KEY"] == "secret_abc"  # the caller's mapping is untouched


def test_cp_ssh_failure_raises(tmp_path):
    sh = FakeShell([BASE_OUTPUTS, fail(255, "ssh: connect to host failed")])
    with pytest.raises(CommandError, match="connect to host failed"):
        Infra(sh, initialised(tmp_path, "base"), ENV).cp_ssh("sudo true")
    # the failure came from the IAP SSH itself, not from an earlier step (e.g. a missing infra/base)
    assert [c.args[:3] for c in sh.calls] == [["terraform", f"-chdir={tmp_path}/infra/base", "output"],
                                              ["gcloud", "compute", "ssh"]]


def test_kubectl_uses_the_cluster_kubeconfig():
    sh = FakeShell([ok(), ok()])
    Infra(sh, ROOT, {"HOME": "/home/op"}).kubectl("get", "nodes")
    Infra(sh, ROOT, {"HOME": "/home/op", "KUBECONFIG": "/tmp/kc"}).kubectl("get", "nodes")
    assert sh.calls[0].args == ["kubectl", "--kubeconfig", "/home/op/.kube/hybrid-gpu.yaml", "get", "nodes"]
    assert sh.calls[1].args == ["kubectl", "--kubeconfig", "/tmp/kc", "get", "nodes"]


def test_kubectl_refuses_a_kubeconfig_list():  # X0-2: kubectl --kubeconfig takes one file
    sh = FakeShell([])
    with pytest.raises(ValueError, match="KUBECONFIG"):
        Infra(sh, ROOT, {"KUBECONFIG": f"/home/op/.kube/a.yaml{os.pathsep}/home/op/.kube/b.yaml"}).kubectl("version")
    assert sh.calls == []


def test_cluster_gpu_nodes_reports_a_kubeconfig_list_as_such():  # S1-1: not as "unexpected output"
    with pytest.raises(ValueError, match="^KUBECONFIG must name one file"):
        Infra(FakeShell([]), ROOT, {"KUBECONFIG": f"/a{os.pathsep}/b"}).cluster_gpu_nodes()


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


@pytest.mark.parametrize("stderr", [
    "Error from server (NotFound): the server could not find the requested resource",  # e.g. a stale tunnel
    'Error from server (NotFound): nodes "k3s-gpu-other-9" not found',                   # another node
    'Error from server (NotFound): nodes "k3s-gpu-a6000-10" not found',                  # a longer name
])
@pytest.mark.parametrize("step", ["drain", "delete"])
def test_only_this_node_being_gone_counts_as_success(step, stderr):  # C0-1
    results = [fail(1, stderr)] if step == "drain" else [ok(), fail(1, stderr)]
    sh = FakeShell(results)
    with pytest.raises(CommandError):
        Infra(sh, ROOT, {}).drain_and_delete_node("k3s-gpu-a6000-1")
    assert len(sh.calls) == len(results)


def test_this_node_not_found_is_tolerated_by_both_steps():
    gone = 'Error from server (NotFound): nodes "k3s-gpu-a6000-1" not found'
    sh = FakeShell([fail(1, gone), fail(1, gone)])
    Infra(sh, ROOT, {}).drain_and_delete_node("k3s-gpu-a6000-1")
    assert [c.args[3] for c in sh.calls] == ["drain", "delete"]


# --- api_unreachable: may gpu-down terminate a node it could not drain? (review F2-3 / G-1) --------

def kubeconfig(tmp_path):
    path = tmp_path / "hybrid-gpu.yaml"
    path.write_text("apiVersion: v1\n")
    return {"KUBECONFIG": str(path)}


def test_api_unreachable_without_a_kubeconfig_runs_nothing():
    sh = FakeShell([])
    reason = Infra(sh, ROOT, {"HOME": "/h"}).api_unreachable()
    assert reason == "no kubeconfig at /h/.kube/hybrid-gpu.yaml" and sh.calls == []


def test_api_unreachable_asks_the_api_server_bounded(tmp_path):
    env = kubeconfig(tmp_path)
    sh = FakeShell([ok("ok")])
    assert Infra(sh, ROOT, env).api_unreachable() is None
    [call] = sh.calls
    assert call.args == ["kubectl", "--kubeconfig", env["KUBECONFIG"], "get", "--raw=/readyz",
                         "--request-timeout=15s"]
    assert call.timeout == 30 and call.check is False


@pytest.mark.parametrize("stderr", [
    "The connection to the server 127.0.0.1:6443 was refused - did you specify the right host or port?",
    'Unable to connect to the server: dial tcp 127.0.0.1:6443: connect: connection refused',
    "Unable to connect to the server: net/http: TLS handshake timeout",
    'Get "https://127.0.0.1:6443/readyz?timeout=15s": dial tcp 127.0.0.1:6443: i/o timeout',
    "Unable to connect to the server: net/http: request canceled (Client.Timeout exceeded while awaiting headers)",
])
def test_api_unreachable_when_nothing_answers(tmp_path, stderr):
    reason = Infra(FakeShell([fail(1, stderr)]), ROOT, kubeconfig(tmp_path)).api_unreachable()
    assert reason is not None and "API server" in reason and stderr[:30] in reason


def test_api_unreachable_when_kubectl_hangs(tmp_path):
    sh = FakeShell([CommandError(["kubectl"], -1, "timed out after 30s")])
    assert "did not answer within 30s" in Infra(sh, ROOT, kubeconfig(tmp_path)).api_unreachable()


@pytest.mark.parametrize("stderr", [
    "Error from server (Forbidden): forbidden: User \"x\" cannot get path \"/readyz\"",
    "Error from server (InternalError): an error on the server (\"[-]etcd failed\") has prevented the request",
    "error: You must be logged in to the server (Unauthorized)",
])
def test_an_api_server_that_answers_is_reachable(tmp_path, stderr):
    """An answer, even an error, means a cluster is there: gpu-down must not terminate an undrained node of it."""
    assert Infra(FakeShell([fail(1, stderr)]), ROOT, kubeconfig(tmp_path)).api_unreachable() is None


def test_api_unreachable_raises_when_kubectl_cannot_run(tmp_path):
    sh = FakeShell([CommandError(["kubectl"], 127, "kubectl: command not found")])
    with pytest.raises(CommandError):
        Infra(sh, ROOT, kubeconfig(tmp_path)).api_unreachable()


# --- join-node, host keys, operator IP, cp guard ------------------------------------------------

def test_join_node_runs_the_script_and_raises_on_failure():
    sh = FakeShell([ok(), fail(3, "kubectl wait timed out")])
    infra = Infra(sh, ROOT, {})
    infra.join_node("k3s-gpu-a6000-1", "129.80.30.30", "a6000")
    assert sh.calls[0].args == [f"{ROOT}/scripts/join-node.sh", "--node", "k3s-gpu-a6000-1",
                                "--external-ip", "129.80.30.30", "--gpu-type", "a6000", "--skip-sync"]
    with pytest.raises(CommandError):
        infra.join_node("k3s-gpu-a6000-1", "129.80.30.30", "a6000")


@pytest.mark.parametrize("instance_type, label", [
    ("gpu_8x_a100_80gb_sxm4", "a100_80gb_sxm4"), ("gpu_1x_h100_pcie", "h100_pcie"), ("gpu_1x_a6000", "a6000"),
])
def test_join_node_accepts_every_gpu_type_label(instance_type, label):  # S0-1: Task 20 passes gpu_type_label()
    sh = FakeShell([ok()])
    Infra(sh, ROOT, {}).join_node("k3s-gpu-a100-1", "129.80.30.30", gpu_type_label(instance_type))
    args = sh.calls[0].args
    assert args[args.index("--gpu-type") + 1] == label


@pytest.mark.parametrize("name, ip, gpu_type", [
    ("k3s-gpu-a6000-1", "1.2.3.4 -oProxyCommand=evil", "a6000"),
    ("k3s-gpu-a6000-1", None, "a6000"),
    ("k3s-gpu-a6000-1", "10.0.0.5", "a6000"),
    ("--help", "129.80.30.30", "a6000"),
    ("k3s-gpu-a6000-1", "129.80.30.30", "a6000; id"),
    ("k3s-gpu-a6000-1", "129.80.30.30", "--help"),
    ("k3s-gpu-a6000-1", "129.80.30.30", "_a100"),
    ("k3s-gpu-a6000-1", "129.80.30.30", "a100_"),
    ("k3s-gpu-a6000-1", "129.80.30.30", "A100"),
    ("k3s-gpu-a6000-1", "129.80.30.30", "a" * 64),  # longer than a K8s label value
    ("k3s-gpu-a6000-1", "129.80.30.30", None),
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


def test_cp_guard_elements_parses_nft_json(tmp_path):
    listing = {"nftables": [
        {"metainfo": {"version": "1.0.9", "json_schema_version": 1}},
        {"set": {"family": "inet", "name": "k3s_peers", "table": "k3s_guard", "type": "ipv4_addr", "handle": 2,
                 "elem": ["34.82.10.10", "10.10.0.2", "129.80.30.30"]}},
    ]}
    sh = FakeShell([BASE_OUTPUTS, ok(json.dumps(listing))])
    assert Infra(sh, initialised(tmp_path, "base"), ENV).cp_guard_elements() == ["10.10.0.2", "34.82.10.10", "129.80.30.30"]
    assert sh.calls[1].args[-1] == "sudo nft -j list set inet k3s_guard k3s_peers"


def test_cp_guard_elements_of_an_empty_set(tmp_path):
    listing = {"nftables": [{"set": {"family": "inet", "name": "k3s_peers", "table": "k3s_guard"}}]}
    infra = Infra(FakeShell([BASE_OUTPUTS, ok(json.dumps(listing))]), initialised(tmp_path, "base"), ENV)
    assert infra.cp_guard_elements() == []


@pytest.mark.parametrize("stdout", ["garbage", '{"nftables": []}', '{"nftables": [{"set": {"elem": [{"x": 1}]}}]}'])
def test_cp_guard_elements_rejects_unexpected_output(tmp_path, stdout):
    with pytest.raises(ValueError):
        Infra(FakeShell([BASE_OUTPUTS, ok(stdout)]), initialised(tmp_path, "base"), ENV).cp_guard_elements()


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


def test_subprocess_shell_non_executable_file_raises_command_error(tmp_path):  # X0-5: exit 1, not a traceback
    script = tmp_path / "join-node.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n")
    script.chmod(0o644)
    with pytest.raises(CommandError) as e:
        SubprocessShell().run([str(script)])
    assert e.value.returncode == 126 and "cannot run" in e.value.stderr


def test_subprocess_shell_missing_program_and_timeout_raise_command_error():
    with pytest.raises(CommandError):
        SubprocessShell().run(["gpuctl-no-such-program"])
    with pytest.raises(CommandError, match="timed out"):
        SubprocessShell().run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2)
