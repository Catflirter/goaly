"""gpu-status after gpu-down (live finding L21-1, progress.md 2026-10-21): `make gpu-status` reported "pool a6000 has
0, wants 1" and exited 3, although Task 21 Step 7 expects 0 instances and no drift.

Spec section 5.4 keeps gpuctl free of local state (Task 21a review S0-1), so status cannot know that gpu-down ran.
Instead the drift check is the spec's four-way comparison (Lambda instances, cluster nodes, gpu-access state,
k3s-cp's k3s_peers), plus surplus instances: a pool below its count is a note pointing at gpu-apply, not drift. So
an empty account after gpu-down is clean, while anything gpu-down left behind is still drift (exit 3).
"""
import pytest

import gpuctl.__main__ as cli
from fakes import FakeInfra, FakeLambda
from gpuctl.__main__ import main
from test_cli import APPLY_ENV, ENV, NODE, TYPES, in_sync

IP = "129.80.30.30"  # NODE's


@pytest.fixture
def lambda_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "OVERRIDES_PATH", tmp_path / ".autoscale-state.json")
    return tmp_path


def test_status_after_down_reports_zero_instances_and_no_drift(lambda_dir, capsys):  # Task 21 Step 7
    fake = FakeLambda(types=TYPES, instances=[NODE])
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=FakeInfra()) == 0
    capsys.readouterr()
    assert main(["status"], env=ENV, client=fake, infra=in_sync()) == 0
    out = capsys.readouterr().out
    assert "GPU instances (k3s-gpu-*): 0" in out
    assert "  note: pool a6000: 0 of 1 running (make gpu-apply starts the rest)\n  No drift\n" in out
    assert "problem" not in out


def test_down_keeps_no_local_state(lambda_dir):
    """Spec section 5.4: no local state. gpu-down only removes the autoscaler's override file (section 10.4)."""
    (lambda_dir / ".autoscale-state.json").write_text('{"pools": {"a6000": 2}}')
    fake = FakeLambda(types=TYPES, instances=[NODE])
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=FakeInfra()) == 0
    assert list(lambda_dir.iterdir()) == []


def test_an_instance_gpu_down_could_not_terminate_is_drift(lambda_dir, capsys):
    """gpu-down drained and deleted the node, then the terminate failed: the instance still bills."""
    fake = FakeLambda(types=TYPES, instances=[NODE])
    infra = in_sync(NODE)
    infra.cluster_nodes.clear()
    assert main(["status"], env=ENV, client=fake, infra=infra) == 3
    assert "k3s-gpu-a6000-1 is not in the cluster" in capsys.readouterr().out


def stale_firewall_rule(infra):
    infra.outputs["gpu-access"] = {"gpu_ips": [IP]}


def stale_peer(infra):
    infra.nft_elements.append(IP)


def stale_node(infra):
    infra.cluster_nodes["k3s-gpu-a6000-1"] = IP


@pytest.mark.parametrize("leave, line", [
    (stale_firewall_rule, f"SECURITY: firewall rule for {IP} has no live instance"),
    (stale_peer, f"SECURITY: cp k3s_peers entry for {IP} has no live instance"),
    (stale_node, "stale node k3s-gpu-a6000-1"),
])
def test_what_gpu_down_left_behind_is_drift(lambda_dir, capsys, leave, line):  # review C0-7
    """0 instances and a pool below its count is no drift, but a leftover allow-list entry or node still is."""
    infra = in_sync()
    leave(infra)
    assert main(["status"], env=ENV, client=FakeLambda(types=TYPES), infra=infra) == 3
    out = capsys.readouterr().out
    assert f"  {line}\n" in out and "note: pool a6000: 0 of 1 running" in out
    assert "1 problem(s)" in out


def test_the_drift_hint_names_both_ways_to_reconcile(lambda_dir, capsys):  # review C0-4
    infra = in_sync()
    stale_node(infra)
    assert main(["status"], env=ENV, client=FakeLambda(types=TYPES), infra=infra) == 3
    out = capsys.readouterr().out
    assert "make gpu-apply reconciles them to the spec; make gpu-down removes every GPU node" in out


def test_apply_after_down_brings_the_spec_back(lambda_dir):
    fake = FakeLambda(types=TYPES, instances=[NODE])
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=FakeInfra()) == 0
    assert main(["apply", "--auto-approve"], env=APPLY_ENV, client=fake, infra=FakeInfra()) == 0
    assert len(fake.launched) == 1


def test_plan_after_down_shows_what_apply_would_do(lambda_dir, capsys):
    assert main(["plan"], env=ENV, client=FakeLambda(types=TYPES)) == 0
    assert "create k3s-gpu-a6000-1" in capsys.readouterr().out
