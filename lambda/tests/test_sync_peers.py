"""Reconciler.sync_peers (design spec section 3.3): FakeLambda + FakeInfra, no cloud."""
import io
from pathlib import Path

import pytest

from fakes import DEFAULT_OUTPUTS, FakeInfra, FakeLambda
from gpuctl.__main__ import main
from gpuctl.lambda_api import FirewallRule, Instance, Ruleset
from gpuctl.peers import PeerSet, global_rules, nft_replace_script, region_ruleset_rules
from gpuctl.reconcile import Reconciler, ReconcileError, Settings
from gpuctl.spec import parse_spec
from test_spec import VALID

SPEC = parse_spec(VALID)
SETTINGS = Settings(spec_path=Path("lambda/gpu-nodes.yaml"), overrides_path=Path("lambda/.autoscale-state.json"),
                    ssh_key_name="operator")
REGIONS = {"k3s-gpu-us-west-1", "k3s-gpu-us-west-2", "k3s-gpu-us-east-1"}
OLD_RULE = FirewallRule("tcp", "73.15.40.40/32", (22, 22), "operator ssh")


def inst(name, ip="129.80.30.30", status="active", id=None, region="us-west-2"):
    return Instance(id=id or f"id-{name}-{status}", name=name, status=status, ip=ip, region=region,
                    instance_type="gpu_1x_a6000")


GPU = inst("k3s-gpu-a6000-1")


def rec(fake, infra, out=None):
    return Reconciler(fake, infra, SETTINGS, out=out if out is not None else io.StringIO())


def test_sync_creates_missing_rulesets_and_sets_global_rules():
    fake, infra = FakeLambda(instances=[GPU]), FakeInfra()
    p = rec(fake, infra).sync_peers(SPEC)
    assert p == PeerSet("34.82.10.10", "10.10.0.2", ("34.29.20.20",), ("129.80.30.30",), "73.15.40.40")
    assert {r.name for r in fake.rulesets} == REGIONS
    assert {r.region for r in fake.rulesets} == {"us-west-1", "us-west-2", "us-east-1"}
    assert all(r.rules == tuple(region_ruleset_rules(p)) for r in fake.rulesets)
    assert fake.global_rules == global_rules("73.15.40.40")


def test_sync_updates_existing_rulesets_in_place():
    stale = Ruleset("rs-9", "k3s-gpu-us-west-2", "us-west-2", (OLD_RULE,))
    fake, infra = FakeLambda(instances=[GPU], rulesets=[stale]), FakeInfra()
    p = rec(fake, infra).sync_peers(SPEC)
    assert len(fake.rulesets) == 3
    [updated] = [r for r in fake.rulesets if r.id == "rs-9"]
    assert updated.rules == tuple(region_ruleset_rules(p))
    assert not any(c[0] == "create_ruleset" and c[1] == "k3s-gpu-us-west-2" for c in fake.calls)


def test_ensure_ruleset_creates_then_updates():
    fake = FakeLambda()
    r = rec(fake, FakeInfra())
    created = r.ensure_ruleset("us-west-2", [OLD_RULE])
    assert (created.name, created.region, created.rules) == ("k3s-gpu-us-west-2", "us-west-2", (OLD_RULE,))
    new_rule = FirewallRule("tcp", "73.15.99.99/32", (22, 22), "operator ssh")
    updated = r.ensure_ruleset("us-west-2", [new_rule])
    assert updated == Ruleset(created.id, "k3s-gpu-us-west-2", "us-west-2", (new_rule,))
    assert fake.rulesets == [updated]


def test_sync_replaces_old_operator_ip():  # Review Focus #1
    fake, infra = FakeLambda(instances=[GPU]), FakeInfra()
    r = rec(fake, infra)
    infra.env["OPERATOR_IP"] = "73.15.40.40"
    r.sync_peers(SPEC)
    infra.env["OPERATOR_IP"] = "73.15.99.99"
    r.sync_peers(SPEC)
    assert len(fake.rulesets) == 3
    assert all("73.15.40.40" not in r.source_network for rs in fake.rulesets for r in rs.rules)
    assert all("73.15.40.40" not in r.source_network for r in fake.global_rules)
    assert any("73.15.99.99" in r.source_network for r in fake.global_rules)


def test_sync_also_refreshes_owned_rulesets_of_regions_no_longer_in_the_spec():  # Review Focus #1
    old = Ruleset("rs-7", "k3s-gpu-us-south-2", "us-south-2", (OLD_RULE,))
    other = Ruleset("rs-8", "my-team-rules", "us-west-2", (OLD_RULE,))
    fake, infra = FakeLambda(rulesets=[old, other]), FakeInfra(env={"OPERATOR_IP": "73.15.99.99"})
    rec(fake, infra).sync_peers(SPEC)
    [refreshed] = [r for r in fake.rulesets if r.id == "rs-7"]
    assert all("73.15.40.40" not in rule.source_network for rule in refreshed.rules)
    assert [r for r in fake.rulesets if r.id == "rs-8"] == [other]  # not ours: never touched


def test_sync_pushes_full_peer_set_to_cp():
    fake, infra = FakeLambda(instances=[GPU]), FakeInfra()
    rec(fake, infra).sync_peers(SPEC)
    command, script = infra.ssh_calls[-1]
    assert command == "sudo /usr/local/sbin/k3s-guard-peers"  # Ruling R14: the persisting helper, not nft -f -
    assert script == nft_replace_script(["10.10.0.2", "34.29.20.20", "34.82.10.10", "129.80.30.30"])


def test_sync_order_is_cp_then_global_then_regions():
    fake, infra = FakeLambda(instances=[GPU]), FakeInfra()
    rec(fake, infra).sync_peers(SPEC)
    writes = [c[0] for c in fake.calls if c[0] in ("update_global_rules", "create_ruleset", "update_ruleset")]
    assert infra.events == ["nft"]
    assert writes == ["update_global_rules", "create_ruleset", "create_ruleset", "create_ruleset"]


def test_sync_tolerates_uninitialised_workers_stack():
    out = io.StringIO()
    fake, infra = FakeLambda(instances=[GPU]), FakeInfra(outputs={"base": DEFAULT_OUTPUTS["base"]})
    p = rec(fake, infra, out).sync_peers(SPEC)
    assert p.worker_ips == ()
    assert infra.ssh_calls[-1][1] == nft_replace_script(["10.10.0.2", "34.82.10.10", "129.80.30.30"])
    assert "warning" in out.getvalue().lower() and "workers" in out.getvalue()


def test_sync_treats_a_workers_stack_without_workers_as_empty():
    outputs = {"base": DEFAULT_OUTPUTS["base"], "workers": {}}
    p = rec(FakeLambda(), FakeInfra(outputs=outputs)).sync_peers(SPEC)
    assert p.worker_ips == () and p.gpu_ips == ()


def test_sync_ignores_terminated_and_foreign_instances():
    instances = [
        GPU,
        inst("k3s-gpu-a6000-2", ip="129.80.30.31", status="terminated"),
        inst("k3s-gpu-a6000-3", ip="129.80.30.32", status="terminating"),
        inst("k3s-gpu-a6000-4", ip=None, status="booting"),
        inst("my-notebook", ip="198.51.100.7"),
    ]
    fake, infra = FakeLambda(instances=instances), FakeInfra()
    p = rec(fake, infra).sync_peers(SPEC)
    assert p.gpu_ips == ("129.80.30.30",)
    assert infra.ssh_calls[-1][1] == nft_replace_script(["10.10.0.2", "34.29.20.20", "34.82.10.10", "129.80.30.30"])
    sources = {rule.source_network for rs in fake.rulesets for rule in rs.rules}
    assert not sources & {"129.80.30.31/32", "129.80.30.32/32", "198.51.100.7/32"}


@pytest.mark.parametrize("missing", ["cp_ip", "cp_internal_ip"])
def test_sync_fails_without_cp_outputs_and_changes_nothing(missing):
    base = {k: v for k, v in DEFAULT_OUTPUTS["base"].items() if k != missing}
    fake, infra = FakeLambda(instances=[GPU]), FakeInfra(outputs={"base": base})
    with pytest.raises(ReconcileError, match=missing):
        rec(fake, infra).sync_peers(SPEC)
    assert infra.ssh_calls == [] and fake.rulesets == [] and fake.global_rules == []


def test_sync_with_a_bad_operator_ip_changes_nothing():
    fake, infra = FakeLambda(instances=[GPU]), FakeInfra(env={"OPERATOR_IP": "10.0.0.1"})
    with pytest.raises(ValueError):
        rec(fake, infra).sync_peers(SPEC)
    assert infra.ssh_calls == [] and fake.rulesets == [] and fake.global_rules == []


def test_summary_masks_the_operator_ip():
    out = io.StringIO()
    rec(FakeLambda(instances=[GPU]), FakeInfra(), out).sync_peers(SPEC)
    text = out.getvalue()
    assert "73.15.40.x" in text and "73.15.40.40" not in text
    assert "k3s_peers" in text and "k3s-gpu-us-west-2" in text


def test_out_defaults_to_the_stdout_current_at_call_time(capsys):  # Ruling R18
    r = Reconciler(FakeLambda(), FakeInfra(), SETTINGS)
    r.sync_peers(SPEC)
    assert "73.15.40.x" in capsys.readouterr().out


# --- CLI ----------------------------------------------------------------------------------------

def test_cli_sync_peers(capsys):
    fake, infra = FakeLambda(instances=[GPU]), FakeInfra()
    assert main(["sync-peers"], env={"LAMBDA_API_KEY": "k"}, client=fake, infra=infra) == 0
    assert infra.events == ["nft"] and {r.name for r in fake.rulesets} == REGIONS
    assert "73.15.40.x" in capsys.readouterr().out


def test_cli_sync_peers_failure_exits_1(capsys):
    infra = FakeInfra(outputs={"base": {"cp_zone": "us-west1-b"}})
    assert main(["sync-peers"], env={"LAMBDA_API_KEY": "k"}, client=FakeLambda(), infra=infra) == 1
    assert "cp_ip" in capsys.readouterr().err


def test_cli_sync_peers_bad_operator_ip_exits_1(capsys):
    infra = FakeInfra(env={"OPERATOR_IP": "192.168.1.5"})
    assert main(["sync-peers"], env={"LAMBDA_API_KEY": "k"}, client=FakeLambda(), infra=infra) == 1
    assert "192.168.1.5" in capsys.readouterr().err


def test_cli_sync_peers_needs_the_api_key(capsys):
    infra = FakeInfra()
    assert main(["sync-peers"], env={}, client=FakeLambda(), infra=infra) == 2
    assert infra.ssh_calls == []
