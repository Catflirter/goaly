"""gpuctl down when the cluster cannot be drained (Task 21a reviews r2 F2-3 = ledger finding G-1, r3 F3-1/F3-3).

Spec section 5.4 deletes a GPU node as drain -> delete node -> terminate. Ruling R54 amends it: gpu-down terminates
an owned instance WITHOUT the drain only when the cluster is known to be gone (no kubeconfig, or infra/base has no
cluster outputs: make down destroyed it) or the operator passes --no-drain. An API server that does not answer
while the cluster still exists (`make tunnel` not running) is not "gone": that node's terminate is refused with
"start make tunnel and re-run, or pass --no-drain", since an undrained terminate would leave its Node object and
k3s's node-password secret behind for the next instance of that name. After the terminates, a gpu-down whose cluster
is gone skips the k3s-cp peer-set update, still rewrites the Lambda rules (operator SSH only) and exits 0.
gpu-apply never terminates an undrained node.
"""
import io

import pytest

from fakes import DEFAULT_OUTPUTS, FakeInfra, FakeLambda
from gpuctl.__main__ import main
from gpuctl.infra import CommandError
from gpuctl.lambda_api import FirewallRule, Ruleset
from gpuctl.peers import global_rules
from test_cli import APPLY_ENV, ENV
from test_reconcile import SPEC, TYPES, inst, make_rec

NO_KUBECONFIG = "no kubeconfig at /h/.kube/hybrid-gpu.yaml"
REFUSED = "The connection to the server 127.0.0.1:6443 was refused"
OPERATOR_SSH = global_rules("73.15.40.40")
# What make down leaves: the base and workers stacks destroyed (terraform output -json prints {}).
GONE_OUTPUTS = {"base": {}, "workers": {}}
OLD_RULES = (FirewallRule("tcp", "73.15.40.40/32", (22, 22), "operator ssh"),
             FirewallRule("udp", "34.82.10.10/32", (51820, 51820), "wireguard 34.82.10.10"))


def failing_drain(infra, stderr=REFUSED):
    drains = []

    def drain(name):
        drains.append(name)
        raise CommandError(["kubectl", "drain", name], 1, stderr)

    infra.drain_and_delete_node = drain
    return drains


def two_nodes():
    return [inst("k3s-gpu-a6000-1", ip="129.80.30.30"), inst("k3s-gpu-a6000-2", ip="129.80.30.31")]


# --- the cluster is gone: no kubeconfig ----------------------------------------------------------

def test_down_terminates_an_owned_instance_when_there_is_no_kubeconfig(capsys):
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    failing_drain(infra)
    infra.no_kubeconfig = NO_KUBECONFIG
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 0
    assert fake.terminated == ["id-k3s-gpu-a6000-1"]
    assert infra.forgotten == ["k3s-gpu-a6000-1"]
    assert infra.applied_gpu_ips[-1] == [] and infra.events[-1] == "nft"  # its IP leaves every allow-list
    assert infra.probes == 0  # known gone: the API server is not asked
    err = capsys.readouterr().err
    assert "WARNING" in err and NO_KUBECONFIG in err and "without draining" in err
    # if that cluster comes back after all, the operator is told how to remove the node object
    assert "kubectl --kubeconfig ~/.kube/hybrid-gpu.yaml delete node k3s-gpu-a6000-1" in err


def test_down_drains_a_gone_cluster_once_for_all_nodes():
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    drains = failing_drain(infra)
    infra.no_kubeconfig = NO_KUBECONFIG
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 0
    assert sorted(fake.terminated) == ["id-k3s-gpu-a6000-1", "id-k3s-gpu-a6000-2"]
    assert len(drains) == 1  # the second node is not drained (and waited for) again


def test_down_unreachable_never_touches_a_foreign_instance():
    foreign = inst("my-notebook", ip="129.80.30.99", id="id-foreign")
    fake, infra = FakeLambda(types=TYPES, instances=[inst(), foreign]), FakeInfra()
    failing_drain(infra)
    infra.no_kubeconfig = NO_KUBECONFIG
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 0
    assert fake.terminated == ["id-k3s-gpu-a6000-1"]


# --- the cluster still exists, its API server does not answer (review F3-1) -----------------------

def test_down_refuses_the_undrained_terminate_while_the_cluster_exists(capsys):
    """infra/base has outputs and the kubeconfig is there, but nothing answers on 127.0.0.1:6443 (make tunnel is
    not running): that looks exactly like a destroyed cluster to kubectl, so gpu-down keeps the instance."""
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    failing_drain(infra)
    infra.unreachable = "the API server does not answer: " + REFUSED
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert fake.terminated == [] and infra.events == [] and infra.forgotten == []
    err = capsys.readouterr().err
    assert "start make tunnel and re-run" in err and "--no-drain" in err and "was refused" in err


def test_down_refuses_every_undrained_node_after_one_probe(capsys):
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    drains = failing_drain(infra)
    infra.unreachable = "the API server does not answer"
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert fake.terminated == []
    assert infra.probes == 1 and len(drains) == 1  # no second wait on a dead tunnel
    err = capsys.readouterr().err
    assert "delete k3s-gpu-a6000-1" in err and "delete k3s-gpu-a6000-2" in err  # both reported, both kept


def test_down_refuses_when_infra_base_cannot_be_read():
    """Not knowing is not "gone": an unreadable base stack (GCS down, init failing) never allows the terminate."""
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra(init_fail={"base"})
    failing_drain(infra)
    infra.unreachable = "the API server does not answer"
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert fake.terminated == []


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
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra(outputs=GONE_OUTPUTS)
    failing_drain(infra)
    infra.no_kubeconfig = NO_KUBECONFIG
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


# --- the cluster is gone: infra/base destroyed (review F3-3, the main G-1 scenario) -------------------

def gone_account(*instances):
    return FakeLambda(types=TYPES, instances=list(instances), global_rules=OLD_RULES,
                      rulesets=[Ruleset("rs-old", "k3s-gpu-us-west-2", "us-west-2", OLD_RULES)])


def test_down_after_make_down_terminates_the_orphan_and_rewrites_the_lambda_rules(tmp_path, monkeypatch, capsys):
    """End to end through the CLI: make down destroyed infra/base and infra/workers, the kubeconfig still points at
    the dead tunnel. gpu-down terminates the orphan, applies the empty gpu-access stack, skips k3s-cp's peer set
    (there is no k3s-cp), leaves only the operator's SSH in Lambda's global rules and in the owned ruleset, creates
    no new ruleset, and exits 0."""
    import gpuctl.__main__ as cli
    monkeypatch.setattr(cli, "OVERRIDES_PATH", tmp_path / ".autoscale-state.json")
    fake, infra = gone_account(inst()), FakeInfra(outputs=GONE_OUTPUTS)
    drains = failing_drain(infra)
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=infra) == 0
    assert drains == ["k3s-gpu-a6000-1"] and fake.terminated == ["id-k3s-gpu-a6000-1"]
    assert infra.probes == 0
    assert infra.applied_gpu_ips == [[]]
    assert "nft" not in infra.events and infra.ssh_calls == []
    assert fake.global_rules == OPERATOR_SSH
    assert [(r.id, r.rules) for r in fake.rulesets] == [("rs-old", tuple(OPERATOR_SSH))]
    assert not [c for c in fake.calls if c[0] == "create_ruleset"]
    captured = capsys.readouterr()
    assert "infra/base has no cluster outputs" in captured.err
    assert "k3s-cp k3s_peers: skipped" in captured.out
    assert "Lambda global rules: operator ssh only" in captured.out


def test_down_rerun_after_make_down_finds_nothing_and_exits_0():
    fake, infra = gone_account(), FakeInfra(outputs=GONE_OUTPUTS)
    out = io.StringIO()
    assert make_rec(fake, infra, out=out).down(SPEC, auto_approve=True) == 0
    assert fake.terminated == [] and infra.applied_gpu_ips == [[]] and "nft" not in infra.events
    assert fake.global_rules == OPERATOR_SSH
    assert "k3s-cp k3s_peers: skipped" in out.getvalue()


def test_down_no_drain_after_make_down_exits_0():
    fake, infra = gone_account(*two_nodes()), FakeInfra(outputs=GONE_OUTPUTS)
    drains = failing_drain(infra)
    assert make_rec(fake, infra).down(SPEC, auto_approve=True, no_drain=True) == 0
    assert drains == [] and sorted(fake.terminated) == ["id-k3s-gpu-a6000-1", "id-k3s-gpu-a6000-2"]
    assert "nft" not in infra.events and fake.global_rules == OPERATOR_SSH


def test_down_after_make_down_still_fails_when_an_instance_could_not_be_terminated():
    """Exit 0 only when no owned instance remains: a failed terminate is still an error (and the rules that can
    be rewritten still are)."""
    fake, infra = gone_account(inst(), inst("k3s-gpu-a6000-2", ip="129.80.30.31")), FakeInfra(outputs=GONE_OUTPUTS)
    failing_drain(infra)
    real_terminate = fake.terminate

    def terminate(ids):
        if ids == ["id-k3s-gpu-a6000-2"]:
            raise CommandError(["terminate"], 1, "Lambda API 500")
        real_terminate(ids)

    fake.terminate = terminate
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert fake.terminated == ["id-k3s-gpu-a6000-1"]
    assert fake.global_rules == OPERATOR_SSH and "nft" not in infra.events


def test_sync_peers_still_needs_the_base_stack(capsys):
    """Only gpu-down's final step goes on without a cluster: sync-peers on its own still refuses."""
    fake, infra = gone_account(), FakeInfra(outputs=GONE_OUTPUTS)
    assert main(["sync-peers"], env=ENV, client=fake, infra=infra) != 0
    assert fake.global_rules == list(OLD_RULES)
    assert "apply the base stack first" in capsys.readouterr().err


@pytest.mark.parametrize("instances", [[], [inst()]], ids=["would-launch", "matches-the-spec"])
def test_apply_still_needs_the_base_stack(instances, tmp_path, monkeypatch, capsys):
    """Review F4-2: a gpu-apply against a destroyed base stack fails with "apply the base stack first". It launches
    nothing and rewrites no Lambda rule: the cluster-gone rules (operator SSH only) are for gpu-down alone. With the
    instance already matching the spec, step 7 is all that runs, and it must not exit 0 on stripped rules."""
    import gpuctl.__main__ as cli
    monkeypatch.setattr(cli, "OVERRIDES_PATH", tmp_path / ".autoscale-state.json")
    fake, infra = gone_account(*instances), FakeInfra(outputs=GONE_OUTPUTS)
    assert main(["apply", "--auto-approve"], env=APPLY_ENV, client=fake, infra=infra) == 1
    assert fake.launched == [] and fake.terminated == []
    assert not [c for c in fake.calls if c[0] in ("launch", "update_global_rules", "update_ruleset",
                                                  "create_ruleset")]
    assert fake.global_rules == list(OLD_RULES)
    assert [(r.id, r.rules) for r in fake.rulesets] == [("rs-old", OLD_RULES)]
    assert "apply the base stack first" in capsys.readouterr().err


def test_partial_base_outputs_are_not_a_gone_cluster():
    """Only a base stack with neither cp_ip nor cp_internal_ip counts as destroyed; a half-read one is an error."""
    outputs = {**DEFAULT_OUTPUTS, "base": {"cp_ip": "34.82.10.10"}}
    fake, infra = gone_account(inst()), FakeInfra(outputs=outputs)
    failing_drain(infra)
    infra.unreachable = "the API server does not answer"
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert fake.terminated == []


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


def test_cli_down_no_drain_help_states_when_gpu_down_terminates_undrained(capsys):
    """Review F4-1: the help says what Ruling R54 does. gpu-down terminates undrained on its own only when the cluster
    is gone (no kubeconfig, or infra/base has no cluster outputs); an API server that does not answer is refused, and
    --no-drain is the operator's choice to abandon a cluster that exists but cannot be reached."""
    assert main(["down", "--help"], env=ENV, client=FakeLambda(types=TYPES), infra=FakeInfra()) == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "already does this when the API server does not answer" not in text
    assert "no kubeconfig" in text and "infra/base has no cluster outputs" in text
    assert "make tunnel" in text and "abandon" in text
