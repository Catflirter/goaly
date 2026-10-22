"""gpuctl drains and deletes Nodes only in this project's cluster (minors batch 2b, ledger finding D-2).

~/.kube/hybrid-gpu.yaml and the tunnel port 127.0.0.1:6443 are one per operator, not one per project, so the
kubeconfig can reach another project's cluster (or another cluster on the forwarded port). Before its first drain
of an apply or a down, gpuctl checks what scripts/down.sh's same_cluster checks: Node k3s-cp's ExternalIP must be
infra/base's cp_ip. A mismatch is handled like an API server that does not answer (Ruling R54): the undrained
terminate is refused, with both IPs in the message (callers' evidence redaction masks them), and nothing in the
other cluster is drained or deleted. --no-drain still terminates without draining. A base stack without its
cluster outputs (neither cp_ip nor cp_internal_ip) is R54's "cluster gone": gpu-down terminates without draining,
and drains nothing through whatever cluster the kubeconfig still reaches. One with cp_internal_ip but no cp_ip is a
broken stack, refused (review S0-2).

A gpu-apply that launches checks before its first launch too (review S0-1/X0-1): a new node joins through the same
kubeconfig, and join-node.sh deletes a stale NotReady Node of its name there (Ruling R55). Each refusal says what
to fix for its cause (review X0-3), and a check that fails part-way refuses the rest of the run (review C0-2).
"""
import pytest

from fakes import DEFAULT_OUTPUTS, FakeInfra, FakeLambda
from gpuctl.__main__ import main
from gpuctl.infra import CommandError
from test_cli import ENV
from test_reconcile import SPEC, TYPES, inst, make_rec, mutations, spec_with
from test_spec import VALID

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


def test_apply_with_nothing_to_change_never_reads_the_cluster_identity():
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()  # SPEC wants exactly this node
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    assert infra.identity_reads == 0 and infra.deleted_nodes == [] and fake.launched == []


def test_apply_reads_the_cluster_identity_once_before_its_first_launch():
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    assert [n.name for n in fake.launched] == ["k3s-gpu-a6000-1"] and infra.identity_reads == 1


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
    assert "not this project's" in err and "--no-drain" in err and "make kubeconfig" in err  # review X0-3
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


def test_apply_launches_nothing_to_join_another_cluster(capsys):
    """Review S0-1/X0-1: the join goes through the kubeconfig, and join-node.sh deletes a stale NotReady Node of the
    new node's name there (and so its node-password secret, Ruling R55). Checked before the first launch, so a
    refused apply starts nothing that bills."""
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    infra.cp_node_ip = OTHER_CP_IP
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.launched == [] and mutations(fake) == [] and infra.joined == [] and infra.deleted_nodes == []
    assert infra.events == [] and infra.identity_reads == 1  # no launch, so no firewall change either
    err = capsys.readouterr().err
    assert OTHER_CP_IP in err and CP_IP in err and "not this project's" in err
    assert "make kubeconfig" in err and "make tunnel" in err and "re-run make gpu-apply" in err
    assert "does not terminate" not in err and "--no-drain" not in err


def test_apply_without_a_kubeconfig_says_to_fetch_it(capsys):
    """Review S0-4: no kubeconfig is not a missing base stack; make kubeconfig (make up writes it) fixes it."""
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    infra.no_kubeconfig = "no kubeconfig at /home/op/.kube/hybrid-gpu.yaml"
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.launched == [] and mutations(fake) == [] and infra.identity_reads == 0
    err = capsys.readouterr().err
    assert "no kubeconfig at" in err and "make kubeconfig" in err and "apply the base stack" not in err


def test_apply_with_infra_base_destroyed_says_to_apply_it(capsys):
    fake, infra = FakeLambda(types=TYPES), FakeInfra(outputs={"base": {}, "workers": {}})
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.launched == [] and mutations(fake) == [] and infra.identity_reads == 0
    err = capsys.readouterr().err
    assert "infra/base has no cluster outputs" in err and "apply the base stack" in err and "make up" in err


def test_a_refused_cleanup_node_delete_says_the_node_is_left_and_how_to_remove_it():
    """Review C0-3: step 9 terminates the instance whether or not its Node can be deleted, so the refusal does not
    say "does not terminate"; it says the Node stays and how to remove it once this project's cluster answers.
    (A launching apply is refused before its launches, so only a refusal that comes later reaches step 9.)"""
    infra = FakeInfra()
    infra.cp_node_ip = OTHER_CP_IP
    rec = make_rec(FakeLambda(types=TYPES), infra)
    with pytest.raises(CommandError) as refused:
        rec._delete_node("k3s-gpu-a6000-1")
    message = str(refused.value)
    assert infra.deleted_nodes == [] and OTHER_CP_IP in message and CP_IP in message
    assert "does not terminate" not in message and "make tunnel" in message
    assert "kubectl --kubeconfig ~/.kube/hybrid-gpu.yaml delete node k3s-gpu-a6000-1" in message


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


def test_down_never_drains_when_infra_base_cannot_be_read(capsys):
    """Review C0-1 (a): not knowing infra/base's cp_ip is neither "gone" nor "same": nothing is drained through a
    kubeconfig that may reach another cluster, and nothing is terminated undrained."""
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra(outputs={"workers": {}})  # no base
    infra.cp_node_ip = OTHER_CP_IP
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert infra.deleted_nodes == [] and fake.terminated == [] and infra.identity_reads == 0
    err = capsys.readouterr().err
    assert "cannot read infra/base outputs" in err
    assert "terraform -chdir=infra/base init" in err and "--no-drain" in err  # review X0-3: the fix for this cause


def test_down_refuses_when_infra_base_has_no_cp_ip(capsys):
    """Review C0-1 (c), S0-2: cp_internal_ip without cp_ip is a broken base stack, not R54's "no base stack outputs"
    (test_partial_base_outputs_are_not_a_gone_cluster): refused, never terminated undrained."""
    base = {"cp_internal_ip": "10.10.0.2", "cp_zone": "us-west1-b"}
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra(outputs={"base": base, "workers": {}})
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert infra.deleted_nodes == [] and fake.terminated == [] and infra.identity_reads == 0
    err = capsys.readouterr().err
    assert "no cp_ip output" in err and "make up" in err and "--no-drain" in err  # review X0-3


def test_down_refuses_when_neither_the_identity_read_nor_the_probe_works(capsys):
    """Review C0-1 (b)."""
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    infra.cp_node_ip = CommandError(["kubectl", "get", "node", "k3s-cp"], 1, "Unauthorized")

    def probe_fails():
        raise CommandError(["kubectl", "get", "--raw", "/readyz"], 1, "x509: certificate signed by unknown authority")
    infra.api_unreachable = probe_fails
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert infra.deleted_nodes == [] and fake.terminated == [] and infra.identity_reads == 1
    err = capsys.readouterr().err
    assert "could not be probed either" in err and "make kubeconfig" in err


def test_a_cluster_check_that_fails_part_way_refuses_the_later_nodes(capsys):
    """Review C0-2/X0-2: an error out of the check fails that node, and the next node is refused too, never drained
    unchecked."""
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()

    def kubeconfig_list():
        raise ValueError("KUBECONFIG lists several files")
    infra.kubeconfig_missing = kubeconfig_list
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 1
    assert infra.deleted_nodes == [] and fake.terminated == []
    err = capsys.readouterr().err
    assert "KUBECONFIG lists several files" in err
    assert "delete k3s-gpu-a6000-1" in err and "delete k3s-gpu-a6000-2" in err


@pytest.mark.parametrize("error", [ValueError("KUBECONFIG lists several files"),
                                   FileNotFoundError(2, "No such file or directory", "kubectl")],
                         ids=["kubeconfig", "no-kubectl"])
def test_a_launch_check_that_raises_launches_nothing(error, capsys):
    """PR #104 review F2b-1: an error out of the check before the first launch refuses the launches, with the error
    and what to do in the message; nothing bills and nothing joins."""
    fake, infra = FakeLambda(types=TYPES), FakeInfra()

    def check_fails():
        raise error
    if isinstance(error, ValueError):
        infra.kubeconfig_missing = check_fails
    else:
        infra.cp_external_ip = check_fails
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.launched == [] and mutations(fake) == [] and infra.joined == [] and infra.events == []
    err = capsys.readouterr().err
    assert "launched nothing: the check of which cluster the kubeconfig reaches failed" in err
    assert str(error) in err and "fix that error and re-run make gpu-apply" in err


def test_ctrl_c_during_the_launch_check_is_an_interrupt_not_a_refusal(capsys):
    """PR #104 review F2b-1: Ctrl-C during the check before the first launch ends gpu-apply as an interrupt (the
    KeyboardInterrupt goes on, as everywhere else gpuctl does not have to finish a cleanup), not as a refusal
    telling the operator to fix an error. Nothing was launched."""
    fake, infra = FakeLambda(types=TYPES), FakeInfra()

    def ctrl_c():
        raise KeyboardInterrupt
    infra.cp_external_ip = ctrl_c
    with pytest.raises(KeyboardInterrupt):
        make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False)
    assert fake.launched == [] and mutations(fake) == [] and infra.joined == [] and infra.events == []
    err = capsys.readouterr().err
    assert "interrupted" in err and "launched nothing" in err
    assert "fix that error" not in err and "check of which cluster the kubeconfig reaches failed" not in err


def test_ctrl_c_during_the_launch_check_still_drops_a_terminated_ip_from_the_allow_lists(capsys):
    """A duplicate is terminated by id, without a drain (Ruling R17), so the cluster is first checked before the
    launch. Ctrl-C there still takes the terminated IP out of every allow-list before the interrupt goes on, as a
    refused launch does (Lambda may hand that IP to another tenant)."""
    keep = inst(id="a", ip="129.80.30.30")
    dup = inst(id="b", ip="129.80.30.31")
    fake, infra = FakeLambda(types=TYPES, instances=[dup, keep]), FakeInfra()

    def ctrl_c():
        raise KeyboardInterrupt
    infra.cp_external_ip = ctrl_c
    spec = spec_with(budget_usd_per_hour=5.0, pools={"a6000": {**VALID["pools"]["a6000"], "count": 2}})
    with pytest.raises(KeyboardInterrupt):
        make_rec(fake, infra).apply(spec, auto_approve=True, keep_on_failure=False)
    assert fake.terminated == ["b"] and fake.launched == [] and infra.joined == []
    assert infra.applied_gpu_ips[-1] == ["129.80.30.30"]
    assert "fix that error" not in capsys.readouterr().err


def test_each_run_checks_the_cluster_again():
    """Review C0-1 (d): one Reconciler for several runs (the stretch autoscaler loop) reads the identity each run."""
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    rec = make_rec(fake, infra)
    assert rec.apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    assert infra.identity_reads == 1
    infra.cp_node_ip = OTHER_CP_IP  # the tunnel now reaches another cluster
    assert rec.down(SPEC, auto_approve=True) == 1
    assert infra.identity_reads == 2 and infra.deleted_nodes == [] and fake.terminated == []


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
