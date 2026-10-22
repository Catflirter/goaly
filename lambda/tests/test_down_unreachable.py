"""gpuctl down when the cluster cannot be reached (Task 21a review r2, F2-3 = ledger finding G-1).

Spec section 5.4 deletes a GPU node as drain -> delete node -> terminate. With the cluster gone (make down destroyed
the base stack, the kubeconfig removed, the tunnel down) the drain can never succeed, and before this fix the billed
instance was never terminated, so make gpu-down could not clean up an orphan. Now gpu-down terminates an owned
instance whose drain failed when the API server provably does not answer (Infra.api_unreachable), loudly; when the
cluster answers and refused the drain, nothing is terminated (unchanged). `gpuctl down --no-drain` skips the drain on
purpose. gpu-apply keeps refusing: only gpu-down terminates without a drain.
"""
from fakes import FakeInfra, FakeLambda
from gpuctl.__main__ import main
from gpuctl.infra import CommandError
from test_cli import ENV
from test_reconcile import SPEC, TYPES, inst, make_rec

NO_KUBECONFIG = "no kubeconfig at /h/.kube/hybrid-gpu.yaml"


def failing_drain(infra, stderr="The connection to the server 127.0.0.1:6443 was refused"):
    drains = []

    def drain(name):
        drains.append(name)
        raise CommandError(["kubectl", "drain", name], 1, stderr)

    infra.drain_and_delete_node = drain
    return drains


def two_nodes():
    return [inst("k3s-gpu-a6000-1", ip="129.80.30.30"), inst("k3s-gpu-a6000-2", ip="129.80.30.31")]


def test_down_terminates_an_owned_instance_when_the_cluster_is_unreachable(capsys):
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    failing_drain(infra)
    infra.unreachable = NO_KUBECONFIG
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 0
    assert fake.terminated == ["id-k3s-gpu-a6000-1"]
    assert infra.forgotten == ["k3s-gpu-a6000-1"]
    assert infra.applied_gpu_ips[-1] == [] and infra.events[-1] == "nft"  # its IP leaves every allow-list
    err = capsys.readouterr().err
    assert "WARNING" in err and NO_KUBECONFIG in err and "without draining" in err
    # the node object may remain in a cluster that comes back: the operator is told how to remove it
    assert "kubectl --kubeconfig ~/.kube/hybrid-gpu.yaml delete node k3s-gpu-a6000-1" in err


def test_down_probes_the_cluster_once_for_all_nodes():
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    drains = failing_drain(infra)
    infra.unreachable = "the API server does not answer"
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 0
    assert sorted(fake.terminated) == ["id-k3s-gpu-a6000-1", "id-k3s-gpu-a6000-2"]
    assert infra.probes == 1 and len(drains) == 1  # the second node is not drained (and waited for) again


def test_down_does_not_terminate_when_a_reachable_cluster_refuses_the_drain(capsys):
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    failing_drain(infra, "error: cannot evict pod as it would violate the pod's disruption budget")
    infra.unreachable = None  # the API server answers
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert fake.terminated == [] and infra.events == []
    err = capsys.readouterr().err
    assert "disruption budget" in err
    assert "the API server answers" in err and "--no-drain" in err  # what to do next


def test_down_does_not_terminate_when_the_probe_itself_fails(capsys):
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    failing_drain(infra)

    def probe_fails():
        raise CommandError(["kubectl", "get", "--raw=/readyz"], 127, "kubectl: command not found")

    infra.api_unreachable = probe_fails
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert fake.terminated == []
    assert "was refused" in capsys.readouterr().err  # the drain's own error is reported


def test_apply_never_terminates_an_undrained_node():
    """Only gpu-down terminates without a drain; a gpu-apply that scales down still refuses."""
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    failing_drain(infra)
    infra.unreachable = NO_KUBECONFIG
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.terminated == [] and fake.launched == []
    assert infra.probes == 0


def test_down_no_drain_terminates_without_touching_the_cluster(capsys):
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    drains = failing_drain(infra)
    assert make_rec(fake, infra).down(SPEC, auto_approve=True, no_drain=True) == 0
    assert drains == [] and infra.probes == 0
    assert sorted(fake.terminated) == ["id-k3s-gpu-a6000-1", "id-k3s-gpu-a6000-2"]
    err = capsys.readouterr().err
    assert "--no-drain" in err and "delete node k3s-gpu-a6000-1" in err and "delete node k3s-gpu-a6000-2" in err


def test_down_unreachable_never_touches_a_foreign_instance():
    foreign = inst("my-notebook", ip="129.80.30.99", id="id-foreign")
    fake, infra = FakeLambda(types=TYPES, instances=[inst(), foreign]), FakeInfra()
    failing_drain(infra)
    infra.unreachable = NO_KUBECONFIG
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 0
    assert fake.terminated == ["id-k3s-gpu-a6000-1"]


def test_cli_down_no_drain(tmp_path, monkeypatch):
    import gpuctl.__main__ as cli
    monkeypatch.setattr(cli, "OVERRIDES_PATH", tmp_path / ".autoscale-state.json")
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    drains = failing_drain(infra)
    assert main(["down", "--auto-approve", "--no-drain"], env=ENV, client=fake, infra=infra) == 0
    assert drains == [] and fake.terminated == ["id-k3s-gpu-a6000-1"]


def test_cli_apply_has_no_no_drain(capsys):
    fake = FakeLambda(types=TYPES)
    assert main(["apply", "--auto-approve", "--no-drain"], env=ENV, client=fake, infra=FakeInfra()) == 2
    assert fake.calls == []
