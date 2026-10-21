"""The test double must stay call-compatible with the real client, or tests pass against an API that does not exist."""
import inspect

import pytest

from fakes import ANY_IPV4, FakeLambda
from gpuctl.lambda_api import FirewallRule, Instance, InsufficientCapacity, LambdaAPIError, LambdaClient

RULE = FirewallRule("tcp", "73.15.40.40/32", (22, 22), "operator ssh")
LAUNCH = dict(name="k3s-gpu-a6000-1", region="us-west-2", instance_type="gpu_1x_a6000", ssh_key_name="op",
              image_family="lambda-stack-22-04", ruleset_id="rs-1")


def public_methods(cls):
    return {n for n, v in vars(cls).items() if callable(v) and not n.startswith("_")}


def test_fake_has_exactly_the_client_methods_with_the_same_signatures():
    assert public_methods(FakeLambda) == public_methods(LambdaClient)
    for name in public_methods(LambdaClient):
        assert inspect.signature(getattr(FakeLambda, name)) == inspect.signature(getattr(LambdaClient, name)), name


def test_calls_are_recorded_with_the_method_name_first():
    fake = FakeLambda()
    fake.list_instances()
    fake.update_global_rules([RULE])
    fake.launch(**LAUNCH)
    assert [c[0] for c in fake.calls] == ["list_instances", "update_global_rules", "launch"]
    assert fake.calls[1] == ("update_global_rules", [RULE])
    assert fake.calls[2] == ("launch", LAUNCH)


def test_launch_adds_a_booting_instance_and_terminate_marks_it():
    fake = FakeLambda()
    instance_id = fake.launch(**LAUNCH)
    assert fake.launched_ids == [instance_id]
    assert fake.list_instances() == [Instance(instance_id, "k3s-gpu-a6000-1", "booting", None, "us-west-2",
                                              "gpu_1x_a6000")]
    fake.terminate([instance_id])
    assert fake.terminated == [instance_id] and fake.list_instances()[0].status == "terminated"


def test_launch_errors_are_raised_per_type_and_region():
    fake = FakeLambda(launch_errors={("gpu_1x_a6000", "us-west-2"): InsufficientCapacity("x", 400)})
    with pytest.raises(InsufficientCapacity):
        fake.launch(**LAUNCH)
    assert fake.launched_ids == [] and fake.list_instances() == []
    assert fake.launch(**{**LAUNCH, "region": "us-west-1"})


def test_instance_progress_is_returned_in_order_then_repeats_the_last():
    booting = Instance("i-1", "k3s-gpu-a6000-1", "booting", None, "us-west-2", "gpu_1x_a6000")
    active = Instance("i-1", "k3s-gpu-a6000-1", "active", "129.80.30.30", "us-west-2", "gpu_1x_a6000")
    fake = FakeLambda(instances=[booting], instance_progress={"i-1": [booting, active]})
    assert [fake.get_instance("i-1") for _ in range(3)] == [booting, active, active]
    assert fake.list_instances() == [active]


def test_unknown_ids_raise_api_errors():
    fake = FakeLambda()
    for call in (lambda: fake.get_instance("nope"), lambda: fake.terminate(["nope"]),
                 lambda: fake.update_ruleset("nope", [RULE])):
        with pytest.raises(LambdaAPIError):
            call()


def test_rulesets_and_global_rules_are_stored():
    fake = FakeLambda()
    created = fake.create_ruleset("k3s-gpu-us-west-2", "us-west-2", [RULE])
    assert fake.list_rulesets() == [created] and created.rules == (RULE,)
    fake.update_ruleset(created.id, [])
    assert fake.list_rulesets()[0].rules == ()
    fake.update_global_rules([RULE])
    assert fake.global_rules == [RULE] and fake.get_global_rules() == [RULE]


@pytest.mark.parametrize("source", [ANY_IPV4, "10.0.0.0/8", "2001:db8::1/128"])
def test_fake_refuses_rules_the_real_client_would_never_send(source):
    fake = FakeLambda()
    bad = [FirewallRule("tcp", source, (22, 22), "x")]
    for call in (lambda: fake.update_global_rules(bad), lambda: fake.create_ruleset("k3s-gpu-us-west-2", "us-west-2", bad)):
        with pytest.raises(ValueError):
            call()
    assert fake.global_rules == [] and fake.rulesets == []
