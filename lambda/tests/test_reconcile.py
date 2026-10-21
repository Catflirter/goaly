"""Reconciler.apply / down / wait_active (design spec section 5.4): FakeLambda + FakeInfra, no cloud."""
import copy
import io
import json
from pathlib import Path

import pytest

from fakes import FakeInfra, FakeLambda
from gpuctl.lambda_api import Instance, InstanceType, InsufficientCapacity, LambdaAPIError
from gpuctl.planner import PlanError
from gpuctl.reconcile import OverridesError, Reconciler, Settings, load_overrides
from gpuctl.spec import parse_spec
from test_spec import VALID

SPEC = parse_spec(VALID)
TYPES = {
    "gpu_1x_a6000": InstanceType("gpu_1x_a6000", 109, ("us-west-2",)),
    "gpu_1x_a10": InstanceType("gpu_1x_a10", 129, ()),
    "gpu_1x_a100": InstanceType("gpu_1x_a100", 199, ("us-west-1",)),
}
NO_OVERRIDES = Path(__file__).resolve().parent / "no-such-dir" / ".autoscale-state.json"
MUTATIONS = ("launch", "terminate", "create_ruleset", "update_ruleset", "update_global_rules")


def spec_with(**changes):
    data = copy.deepcopy(VALID)
    data.update(changes)
    return parse_spec(data)


def inst(name="k3s-gpu-a6000-1", status="active", ip="129.80.30.30", id=None, region="us-west-2",
         instance_type="gpu_1x_a6000"):
    return Instance(id=id or f"id-{name}", name=name, status=status, ip=ip, region=region,
                    instance_type=instance_type)


def booting(ip=None):
    return inst(status="booting", ip=ip, id="i-1")


def active(ip):
    return inst(status="active", ip=ip, id="i-1")


class Clock:
    """A fake monotonic clock that only moves when the reconciler sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_rec(fake, infra, *, overrides_path=NO_OVERRIDES, clock=None, confirm=None, out=None, **settings):
    clock = clock or Clock()
    return Reconciler(fake, infra, Settings(spec_path=Path("lambda/gpu-nodes.yaml"), overrides_path=overrides_path,
                                            ssh_key_name="operator", **settings),
                      out=out if out is not None else io.StringIO(), clock=clock, sleep=clock.sleep,
                      confirm=confirm)


def mutations(fake):
    return [c for c in fake.calls if c[0] in MUTATIONS]


# --- apply --------------------------------------------------------------------------------------

def test_apply_launches_into_region_ruleset_then_firewall_then_join():
    fake = FakeLambda(types=TYPES); infra = FakeInfra(); rec = make_rec(fake, infra)
    assert rec.apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    l = fake.launched[0]
    assert (l.name, l.region, l.instance_type, l.image_family) == ("k3s-gpu-a6000-1", "us-west-2", "gpu_1x_a6000", "lambda-stack-22-04")
    assert l.ruleset_id == fake.ruleset_by_name("k3s-gpu-us-west-2").id
    assert l.ssh_key_name == "operator"
    assert infra.events == ["tf:['129.80.30.30']", "nft", "join:k3s-gpu-a6000-1:a6000"]


def test_apply_passes_the_spec_max_nodes_to_the_gpu_access_stack():  # Ruling R27
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    assert make_rec(fake, infra).apply(spec_with(max_nodes=2), auto_approve=True, keep_on_failure=False) == 0
    assert infra.applied_max_nodes and set(infra.applied_max_nodes) == {2}


def test_the_ruleset_a_node_launches_into_already_admits_the_cluster():
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False)
    [launch] = [c for c in fake.calls if c[0] == "launch"]
    before = fake.calls[:fake.calls.index(launch)]
    created = [c for c in before if c[0] == "create_ruleset" and c[1] == "k3s-gpu-us-west-2"]
    assert created, "the region ruleset must exist before launch"
    sources = {rule.source_network for rule in created[0][3]}
    assert {"73.15.40.40/32", "34.82.10.10/32", "34.29.20.20/32"} <= sources


def test_second_apply_changes_no_instances_but_reconciles_firewall():
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    infra.events.clear()
    calls_before = len(fake.calls)
    out = io.StringIO()
    assert make_rec(fake, infra, out=out).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    assert not [c for c in fake.calls[calls_before:] if c[0] in ("launch", "terminate")]
    assert infra.events == ["tf:['129.80.30.30']", "nft"]  # firewall reconciled, no join
    assert "No instance changes" in out.getvalue()


def test_capacity_fallback_on_insufficient_capacity():
    fake = FakeLambda(types=TYPES, launch_errors={("gpu_1x_a6000", "us-west-2"): InsufficientCapacity("x", 400)})
    infra = FakeInfra()
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    tried = [(c[1]["instance_type"], c[1]["region"]) for c in fake.calls if c[0] == "launch"]
    assert tried == [("gpu_1x_a6000", "us-west-2"), ("gpu_1x_a100", "us-west-1")]
    [l] = fake.launched
    assert (l.instance_type, l.region) == ("gpu_1x_a100", "us-west-1")
    assert l.ruleset_id == fake.ruleset_by_name("k3s-gpu-us-west-1").id
    assert infra.joined == [("k3s-gpu-a6000-1", "129.80.30.30", "a100")]


def test_no_capacity_anywhere_returns_1_and_launches_nothing():
    errors = {("gpu_1x_a6000", "us-west-2"): InsufficientCapacity("x", 400),
              ("gpu_1x_a100", "us-west-1"): InsufficientCapacity("x", 400)}
    fake, infra = FakeLambda(types=TYPES, launch_errors=errors), FakeInfra()
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.launched_ids == [] and infra.joined == [] and fake.terminated == []
    assert infra.applied_gpu_ips[-1] == []  # the firewall still matches the (empty) instance list


def test_no_listed_capacity_returns_1_without_trying_to_launch():
    types = {name: InstanceType(name, t.price_cents_per_hour, ()) for name, t in TYPES.items()}
    fake, infra = FakeLambda(types=types), FakeInfra()
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert not [c for c in fake.calls if c[0] == "launch"]


def test_join_failure_terminates_new_instance():
    fake = FakeLambda(types=TYPES)
    infra = FakeInfra(join_fail={"k3s-gpu-a6000-1"})
    rec = make_rec(fake, infra)
    assert rec.apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.terminated == [fake.launched_ids[0]] and infra.applied_gpu_ips[-1] == []
    assert "k3s-gpu-a6000-1" in infra.deleted_nodes and "k3s-gpu-a6000-1" in infra.forgotten
    assert infra.events[-2:] == ["tf:[]", "nft"]  # step 7 again after the cleanup


def test_keep_on_failure_keeps_instance():
    fake = FakeLambda(types=TYPES)
    infra = FakeInfra(join_fail={"k3s-gpu-a6000-1"})
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=True) == 1
    assert fake.terminated == [] and infra.deleted_nodes == [] and infra.forgotten == []
    assert infra.applied_gpu_ips[-1] == ["129.80.30.30"]


def test_wait_active_never_passes_null_ip():  # Review Focus #2
    fake = FakeLambda(types=TYPES, instance_progress={"i-1": [booting(None), active(None), active("129.80.30.30")]})
    infra = FakeInfra()
    clock = Clock()
    assert make_rec(fake, infra, clock=clock).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    assert None not in sum(infra.applied_gpu_ips, [])
    assert infra.applied_gpu_ips == [["129.80.30.30"]]
    assert [c for c in fake.calls if c[0] == "get_instance"] == [("get_instance", "i-1")] * 3
    assert clock.sleeps == [10.0, 10.0]


def test_wait_active_returns_the_active_instance_with_its_ip():
    fake = FakeLambda(instances=[booting()], instance_progress={"i-1": [booting(), active("129.80.30.30")]})
    assert make_rec(fake, FakeInfra()).wait_active("i-1") == active("129.80.30.30")


@pytest.mark.parametrize("status", ["terminated", "terminating"])
def test_wait_active_fails_when_the_instance_goes_away(status):
    gone = inst(status=status, ip=None, id="i-1")
    fake = FakeLambda(instances=[booting()], instance_progress={"i-1": [booting(), gone]})
    with pytest.raises(LambdaAPIError, match=status):
        make_rec(fake, FakeInfra()).wait_active("i-1")


def test_wait_active_times_out():
    fake = FakeLambda(instances=[booting()], instance_progress={"i-1": [booting()]})
    clock = Clock()
    with pytest.raises(TimeoutError, match="i-1"):
        make_rec(fake, FakeInfra(), clock=clock, launch_timeout_s=60.0, poll_s=10.0).wait_active("i-1")
    assert 60.0 <= clock.now <= 70.0


def test_wait_timeout_cleans_up():
    fake = FakeLambda(types=TYPES, instance_progress={"i-1": [booting(None)]})
    infra, clock = FakeInfra(), Clock()
    assert make_rec(fake, infra, clock=clock).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert clock.now >= 900.0
    assert fake.terminated == ["i-1"] and infra.joined == []
    assert "k3s-gpu-a6000-1" in infra.forgotten
    assert infra.applied_gpu_ips[-1] == []


def test_failure_after_wait_cleans_up_the_new_instances():  # e.g. Terraform fails in step 7
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    applied = infra.apply_gpu_access
    attempts = []

    def flaky_apply(gpu_ips, *, max_nodes):
        attempts.append(list(gpu_ips))
        if len(attempts) == 1:
            raise ValueError("terraform exploded")
        applied(gpu_ips, max_nodes=max_nodes)

    infra.apply_gpu_access = flaky_apply
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.terminated == fake.launched_ids == ["i-1"]
    assert infra.joined == [] and infra.applied_gpu_ips == [[]]


def test_unknown_launch_outcome_is_found_by_name_and_cleaned_up():
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    launch = fake.launch

    def lost_response(**kwargs):  # Lambda started the instance, but the response never arrived
        launch(**kwargs)
        raise LambdaAPIError("http/504", 504, "gateway timeout")

    fake.launch = lost_response
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.terminated == ["i-1"] and infra.joined == []


def test_cleanup_continues_when_one_step_fails():
    fake = FakeLambda(types=TYPES)
    infra = FakeInfra(join_fail={"k3s-gpu-a6000-1"})

    def drain_fails(name):
        raise ValueError("kubectl unreachable")

    infra.drain_and_delete_node = drain_fails
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.terminated == ["i-1"] and infra.forgotten == ["k3s-gpu-a6000-1"]
    assert infra.applied_gpu_ips[-1] == []


def test_deletes_happen_before_creates():
    old = inst("k3s-gpu-old-1", ip="129.80.30.40", id="id-old")
    fake, infra = FakeLambda(types=TYPES, instances=[old]), FakeInfra()
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    names = [c[0] for c in fake.calls]
    assert names.index("terminate") < names.index("launch")
    assert fake.terminated == ["id-old"]
    assert infra.deleted_nodes == ["k3s-gpu-old-1"] and infra.forgotten == ["k3s-gpu-old-1"]


def test_delete_drains_before_terminate_then_forgets_the_host_key():
    extra = inst("k3s-gpu-a6000-2", ip="129.80.30.31")
    fake, infra = FakeLambda(types=TYPES, instances=[inst(), extra]), FakeInfra()
    log = []
    drain, forget, terminate = infra.drain_and_delete_node, infra.forget_host_key, fake.terminate
    infra.drain_and_delete_node = lambda name: (log.append(("drain", name)), drain(name))[1]
    infra.forget_host_key = lambda name: (log.append(("forget", name)), forget(name))[1]
    fake.terminate = lambda ids: (log.append(("terminate", ids)), terminate(ids))[1]
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    assert log == [("drain", "k3s-gpu-a6000-2"), ("terminate", ["id-k3s-gpu-a6000-2"]), ("forget", "k3s-gpu-a6000-2")]
    assert infra.applied_gpu_ips[-1] == ["129.80.30.30"]


def test_duplicates_are_terminated_by_id_only():  # Ruling R17
    keep = inst(id="a", ip="129.80.30.30")
    dup = inst(id="b", ip="129.80.30.31")
    fake, infra = FakeLambda(types=TYPES, instances=[dup, keep]), FakeInfra()
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    assert fake.terminated == ["b"]
    assert infra.deleted_nodes == [] and infra.forgotten == []
    assert infra.applied_gpu_ips[-1] == ["129.80.30.30"]


def test_over_budget_returns_2_without_mutations(capsys):
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    assert make_rec(fake, infra).apply(spec_with(budget_usd_per_hour=1.0), auto_approve=True,
                                       keep_on_failure=False) == 2
    assert mutations(fake) == [] and infra.events == [] and infra.deleted_nodes == []
    assert "budget" in capsys.readouterr().err


def test_plan_error_returns_2_without_mutations(capsys):
    unpriced = inst(instance_type="gpu_1x_mystery")  # make_plan cannot price the kept instance
    fake, infra = FakeLambda(types=TYPES, instances=[unpriced, inst("k3s-gpu-a6000-2", ip="129.80.30.31")]), FakeInfra()
    assert make_rec(fake, infra).apply(SPEC, auto_approve=True, keep_on_failure=False) == 2
    assert mutations(fake) == [] and infra.events == [] and infra.deleted_nodes == []
    assert "gpu_1x_mystery" in capsys.readouterr().err


@pytest.mark.parametrize("wanted, launched", [(2, 2), (9, 2), (0, 1)])
def test_overrides_file_raises_target_within_autoscale_bounds(tmp_path, wanted, launched):
    path = tmp_path / ".autoscale-state.json"
    path.write_text(json.dumps({"pools": {"a6000": wanted}}))
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    rec = make_rec(fake, infra, overrides_path=path)
    assert rec.apply(spec_with(budget_usd_per_hour=5.0), auto_approve=True, keep_on_failure=False) == 0
    assert [l.name for l in fake.launched] == [f"k3s-gpu-a6000-{n}" for n in range(1, launched + 1)]


@pytest.mark.parametrize("text", ["{not json", "[]", '{"pools": []}', '{"pools": {"a6000": "two"}}',
                                  '{"pools": {"a6000": -1}}', '{"pools": {"a6000": true}}', '{"other": {}}'])
def test_malformed_overrides_file_returns_2(tmp_path, text, capsys):
    path = tmp_path / ".autoscale-state.json"
    path.write_text(text)
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    assert make_rec(fake, infra, overrides_path=path).apply(SPEC, auto_approve=True, keep_on_failure=False) == 2
    assert mutations(fake) == [] and infra.events == []
    assert str(path) in capsys.readouterr().err


def test_load_overrides():
    assert load_overrides(NO_OVERRIDES) == {}
    assert issubclass(OverridesError, PlanError)


def test_load_overrides_reads_pool_counts(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"pools": {"a6000": 2, "a100": 0}}')
    assert load_overrides(path) == {"a6000": 2, "a100": 0}


def test_declined_confirmation_returns_2():
    prompts = []
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    rec = make_rec(fake, infra, confirm=lambda prompt: prompts.append(prompt) or False)
    assert rec.apply(SPEC, auto_approve=False, keep_on_failure=False) == 2
    assert len(prompts) == 1
    assert mutations(fake) == [] and infra.events == []


def test_accepted_confirmation_applies():
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    rec = make_rec(fake, infra, confirm=lambda prompt: True)
    assert rec.apply(SPEC, auto_approve=False, keep_on_failure=False) == 0
    assert len(fake.launched) == 1


def test_without_a_way_to_confirm_apply_is_refused():
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    assert make_rec(fake, infra).apply(SPEC, auto_approve=False, keep_on_failure=False) == 2
    assert mutations(fake) == []


def test_no_changes_needs_no_confirmation():
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    rec = make_rec(fake, infra, confirm=lambda prompt: pytest.fail("asked to confirm"))
    assert rec.apply(SPEC, auto_approve=False, keep_on_failure=False) == 0
    assert infra.events == ["tf:['129.80.30.30']", "nft"]


def test_apply_prints_the_plan():
    out = io.StringIO()
    make_rec(FakeLambda(types=TYPES), FakeInfra(), out=out).apply(SPEC, auto_approve=True, keep_on_failure=False)
    assert "create k3s-gpu-a6000-1 (pool a6000)" in out.getvalue()


# --- down ---------------------------------------------------------------------------------------

def test_down_terminates_owned_only_and_removes_overrides(tmp_path):
    path = tmp_path / ".autoscale-state.json"
    path.write_text(json.dumps({"pools": {"a6000": 2}}))
    foreign = inst("my-notebook", ip="198.51.100.7", id="id-foreign")
    starting = inst("k3s-gpu-a6000-2", status="booting", ip=None, id="id-2")
    fake = FakeLambda(types=TYPES, instances=[inst(id="id-1"), starting, foreign])
    infra = FakeInfra(cluster_nodes={"k3s-gpu-a6000-1": "129.80.30.30"})
    assert make_rec(fake, infra, overrides_path=path).down(SPEC, auto_approve=True) == 0
    assert not path.exists()
    assert sorted(fake.terminated) == ["id-1", "id-2"]
    assert sorted(infra.deleted_nodes) == ["k3s-gpu-a6000-1", "k3s-gpu-a6000-2"]
    assert sorted(infra.forgotten) == ["k3s-gpu-a6000-1", "k3s-gpu-a6000-2"]
    assert infra.applied_gpu_ips[-1] == [] and infra.events[-2:] == ["tf:[]", "nft"]
    assert not [c for c in fake.calls if c[0] == "launch"]


def test_down_twice_is_noop():  # Review Focus #3
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    assert make_rec(fake, infra).down(SPEC, auto_approve=True) == 0
    terminated, deleted = list(fake.terminated), list(infra.deleted_nodes)
    calls_before = len(fake.calls)
    out = io.StringIO()
    assert make_rec(fake, infra, out=out).down(SPEC, auto_approve=True) == 0
    assert not [c for c in fake.calls[calls_before:] if c[0] == "terminate"]
    assert fake.terminated == terminated and infra.deleted_nodes == deleted
    assert "No instance changes" in out.getvalue()
    assert infra.applied_gpu_ips[-1] == []


def test_down_ignores_a_budget_it_does_not_need():
    fake = FakeLambda(types=TYPES, instances=[inst()])
    assert make_rec(fake, FakeInfra()).down(spec_with(budget_usd_per_hour=0.5), auto_approve=True) == 0
    assert fake.terminated == ["id-k3s-gpu-a6000-1"]


def test_declined_down_changes_no_instances():
    fake, infra = FakeLambda(types=TYPES, instances=[inst()]), FakeInfra()
    assert make_rec(fake, infra, confirm=lambda prompt: False).down(SPEC, auto_approve=False) == 2
    assert mutations(fake) == [] and infra.events == []
