"""gpuctl drains and deletes Nodes only in this project's cluster (minors batch 2b, ledger finding D-2).

~/.kube/hybrid-gpu.yaml and the tunnel port 127.0.0.1:6443 are one per operator, not one per project, so the
kubeconfig can reach another project's cluster (or another cluster on the forwarded port). Before its first drain
of an apply or a down, gpuctl checks what scripts/down.sh's same_cluster checks: Node k3s-cp's ExternalIP must be
infra/base's cp_ip. A mismatch is handled like an API server that does not answer (Ruling R54): the undrained
terminate is refused, with both IPs in the message (callers' evidence redaction masks them), and nothing in the
other cluster is drained or deleted. --no-drain still terminates without draining. A base stack without its
cluster outputs (no cp_ip) is R54's "cluster gone": gpu-down terminates without draining, through whatever
cluster the kubeconfig still reaches.
"""
import pytest

from fakes import DEFAULT_OUTPUTS, FakeInfra, FakeLambda
from gpuctl.__main__ import main
from gpuctl.infra import CommandError
from test_cli import ENV
from test_reconcile import SPEC, TYPES, inst, make_rec

CP_IP = DEFAULT_OUTPUTS["base"]["cp_ip"]  # 34.82.10.10
OTHER_CP_IP = "35.199.7.7"  # k3s-cp of another project's cluster, reached through the same kubeconfig/tunnel
REFUSED = "The connection to the server 127.0.0.1:6443 was refused"


def two_nodes():
    return [inst("k3s-gpu-a6000-1", ip="129.80.30.30"), inst("k3s-gpu-a6000-2", ip="129.80.30.31")]


# --- this project's cluster ----------------------------------------------------------------------

def test_down_drains_when_node_k3s_cp_is_this_projects():
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 0
    assert sorted(infra.deleted_nodes) == ["k3s-gpu-a6000-1", "k3s-gpu-a6000-2"]
    assert sorted(fake.terminated) == ["id-k3s-gpu-a6000-1", "id-k3s-gpu-a6000-2"]
    assert infra.identity_reads == 1  # checked once per run, before the first drain


def test_apply_with_nothing_to_delete_never_reads_the_cluster_identity():
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    assert infra.identity_reads == 0 and infra.deleted_nodes == []


# --- another cluster -----------------------------------------------------------------------------

def test_down_refuses_and_never_drains_in_another_cluster(capsys):
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    infra.cp_node_ip = OTHER_CP_IP
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=infra) == 1
    assert infra.deleted_nodes == [] and fake.terminated == [] and infra.forgotten == []
    assert infra.events == []  # nothing was terminated, so no allow-list changed
    assert infra.identity_reads == 1  # the second node is refused without a second read
    err = capsys.readouterr().err
    assert OTHER_CP_IP in err and CP_IP in err  # both IPs named
    assert "not this project's" in err and "--no-drain" in err
    assert "delete k3s-gpu-a6000-1" in err and "delete k3s-gpu-a6000-2" in err  # both reported, both kept


def test_down_no_drain_terminates_without_reading_another_cluster():
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    infra.cp_node_ip = OTHER_CP_IP
    assert make_rec(fake, infra).down(SPEC, auto_approve=True, no_drain=True) == 0
    assert sorted(fake.terminated) == ["id-k3s-gpu-a6000-1", "id-k3s-gpu-a6000-2"]
    assert infra.deleted_nodes == [] and infra.identity_reads == 0


def test_apply_never_drains_in_another_cluster(capsys):
    old = inst("k3s-gpu-old-1", ip="129.80.30.40", id="id-old")
    fake, infra = FakeLambda(types=TYPES, instances=[old]), FakeInfra()
    infra.cp_node_ip = OTHER_CP_IP
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert infra.deleted_nodes == [] and fake.terminated == [] and fake.launched == []
    err = capsys.readouterr().err
    assert OTHER_CP_IP in err and CP_IP in err and "not this project's" in err
    assert "--no-drain" not in err  # gpu-apply has no such flag


def test_apply_cleanup_never_deletes_a_node_in_another_cluster(capsys):
    """A new node that failed to join is still terminated (it bills), but its Node is not deleted elsewhere."""
    fake, infra = FakeLambda(types=TYPES), FakeInfra(join_fail={"k3s-gpu-a6000-1"})
    infra.cp_node_ip = OTHER_CP_IP
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.terminated == [fake.launched_ids[0]] and infra.forgotten == ["k3s-gpu-a6000-1"]
    assert infra.deleted_nodes == []
    err = capsys.readouterr().err
    assert "cleanup: delete node k3s-gpu-a6000-1" in err and OTHER_CP_IP in err


def test_down_refuses_when_the_cluster_has_no_node_k3s_cp(capsys):
    """An API server that answers but has no Node k3s-cp is not this project's cluster either."""
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    infra.cp_node_ip = CommandError(["kubectl", "get", "node", "k3s-cp"], 1,
                                    'Error from server (NotFound): nodes "k3s-cp" not found')
    infra.unreachable = None  # the API server answers
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert infra.deleted_nodes == [] and fake.terminated == []
    err = capsys.readouterr().err
    assert 'nodes "k3s-cp" not found' in err and "--no-drain" in err


def test_down_refuses_when_node_k3s_cp_has_no_external_ip():
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    infra.cp_node_ip = None
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert infra.deleted_nodes == [] and fake.terminated == []


def test_down_refuses_like_r54_when_the_identity_read_finds_no_api_server(capsys):
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    infra.cp_node_ip = CommandError(["kubectl", "get", "node", "k3s-cp"], 1, REFUSED)
    infra.unreachable = "the API server does not answer: " + REFUSED
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert infra.deleted_nodes == [] and fake.terminated == [] and infra.probes == 1
    err = capsys.readouterr().err
    assert "start make tunnel and re-run" in err and "--no-drain" in err


# --- the cluster is gone (R54): no cp_ip in infra/base -------------------------------------------

@pytest.mark.parametrize("base", [{}, {"cp_zone": "us-west1-b"}], ids=["no-outputs", "no-cp_ip"])
def test_down_after_make_down_never_drains_through_a_leftover_kubeconfig(base, capsys):
    """make down destroyed infra/base, but the kubeconfig and the tunnel port now reach some other cluster: the
    orphan is terminated without a drain, and nothing in that cluster is drained or deleted."""
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra(outputs={"base": base, "workers": {}})
    infra.cp_node_ip = OTHER_CP_IP
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 0
    assert fake.terminated == ["id-k3s-gpu-a6000-1"]
    assert infra.deleted_nodes == [] and infra.identity_reads == 0
    err = capsys.readouterr().err
    assert "without draining" in err and "infra/base has no cluster outputs" in err
