import copy

import pytest

from fakes import DEFAULT_OUTPUTS, FakeInfra, FakeLambda
from gpuctl.__main__ import main
from gpuctl.lambda_api import Instance, InstanceType
from gpuctl.spec import parse_spec
from gpuctl.status import CapacityRow, capacity_rows, find_drift, render_capacity
from test_spec import VALID

SPEC = parse_spec(VALID)
TYPES = {
    "gpu_1x_a6000": InstanceType("gpu_1x_a6000", 109, ("us-west-2",)),
    "gpu_1x_a100": InstanceType("gpu_1x_a100", 199, ("us-west-1", "us-east-1")),
}


def test_rows_follow_pool_then_candidate_type_then_region():
    rows = capacity_rows(SPEC, TYPES)
    assert [(r.pool, r.instance_type, r.region) for r in rows] == [
        ("a6000", t, r)
        for t in ("gpu_1x_a6000", "gpu_1x_a10", "gpu_1x_a100")
        for r in ("us-west-1", "us-west-2", "us-east-1")
    ]


def test_availability_and_price_come_from_instance_types():
    rows = {(r.instance_type, r.region): r for r in capacity_rows(SPEC, TYPES)}
    assert rows[("gpu_1x_a6000", "us-west-2")] == CapacityRow("a6000", "gpu_1x_a6000", "us-west-2", True, 1.09)
    assert rows[("gpu_1x_a6000", "us-west-1")] == CapacityRow("a6000", "gpu_1x_a6000", "us-west-1", False, 1.09)
    assert rows[("gpu_1x_a100", "us-east-1")].available is True


def test_type_missing_from_lambda_is_unavailable_without_price():
    rows = [r for r in capacity_rows(SPEC, TYPES) if r.instance_type == "gpu_1x_a10"]
    assert len(rows) == 3 and all(not r.available and r.price_usd is None for r in rows)


def test_every_pool_is_listed():
    data = copy.deepcopy(VALID)
    data["pools"]["a100"] = {**data["pools"]["a6000"], "count": 0, "instance_type": "gpu_1x_a100",
                             "fallback_types": [], "regions": ["us-east-1"]}
    rows = capacity_rows(parse_spec(data), TYPES)
    assert rows[-1] == CapacityRow("a100", "gpu_1x_a100", "us-east-1", True, 1.99)
    assert len(rows) == 10


def test_render_marks_yes_no_and_prices_one_line_per_row():
    rows = [CapacityRow("a6000", "gpu_1x_a6000", "us-west-2", True, 1.09),
            CapacityRow("a6000", "gpu_1x_a10", "us-west-1", False, None)]
    lines = render_capacity(rows).splitlines()
    assert lines[0].split() == ["POOL", "TYPE", "REGION", "AVAILABLE", "PRICE"]
    assert lines[1].split() == ["a6000", "gpu_1x_a6000", "us-west-2", "yes", "$1.09/h"]
    assert lines[2].split() == ["a6000", "gpu_1x_a10", "us-west-1", "no", "-"]
    assert len(lines) == 3


def test_render_aligns_columns():
    rows = [CapacityRow("a6000", "gpu_1x_a6000", "us-west-2", True, 1.09),
            CapacityRow("a6000", "gpu_1x_a10", "us-east-1", False, None)]
    lines = render_capacity(rows).splitlines()
    starts = {line.index(word) for line, word in zip(lines, ("REGION", "us-west-2", "us-east-1"))}
    assert len(starts) == 1


# --- drift (Task 20) ----------------------------------------------------------------------------

def active_inst(ip, name="k3s-gpu-a6000-1", status="active", id=None):
    return Instance(id or f"id-{name}", name, status, ip, "us-west-2", "gpu_1x_a6000")


IP = "129.80.30.30"
CLEAN = dict(counts={"a6000": 1}, instances=[active_inst(IP)], cluster_nodes={"k3s-gpu-a6000-1": IP},
             tf_ips=[IP], nft_elements=[IP])


def problems_in(lines):
    return [line for line in lines if not line.startswith("skipped:")]


def test_drift_none_when_everything_agrees():
    assert find_drift(**CLEAN) == []


def test_drift_flags_stale_firewall_ip():
    problems = find_drift(counts={"a6000": 1}, instances=[active_inst("129.80.30.30")],
                          cluster_nodes={"k3s-gpu-a6000-1": "129.80.30.30"},
                          tf_ips=["129.80.30.30", "198.51.100.99"], nft_elements=["129.80.30.30"])
    assert any("198.51.100.99" in p and "no live instance" in p for p in problems)
    [stale] = [p for p in problems if "198.51.100.99" in p]
    assert stale.startswith("SECURITY:") and stale == problems[0]  # flagged loudly, listed first


def test_drift_flags_stale_and_missing_peer_set_entries():
    lines = find_drift(**{**CLEAN, "nft_elements": ["198.51.100.99"]})
    assert any(p.startswith("SECURITY:") and "198.51.100.99" in p and "no live instance" in p for p in lines)
    assert any("missing" in p and IP in p and "k3s_peers" in p for p in lines)
    assert len(lines) == 2


def test_drift_flags_missing_firewall_rule():
    assert find_drift(**{**CLEAN, "tf_ips": []}) == [f"missing firewall rule for {IP}"]


def test_drift_flags_missing_node_and_stale_node():
    lines = find_drift(**{**CLEAN, "cluster_nodes": {"k3s-gpu-a6000-7": "129.80.30.37"}})
    assert "k3s-gpu-a6000-1 is not in the cluster" in lines
    assert "stale node k3s-gpu-a6000-7" in lines
    assert len(lines) == 2


def test_drift_flags_pool_count():
    assert find_drift(**{**CLEAN, "counts": {"a6000": 2}}) == ["pool a6000 has 1, wants 2"]


def test_drift_counts_pools_removed_from_the_spec():
    old = active_inst("129.80.30.31", name="k3s-gpu-old-1")
    lines = find_drift(counts={"a6000": 1}, instances=[active_inst(IP), old],
                       cluster_nodes={"k3s-gpu-a6000-1": IP, "k3s-gpu-old-1": "129.80.30.31"},
                       tf_ips=[IP, "129.80.30.31"], nft_elements=[IP, "129.80.30.31"])
    assert lines == ["pool old has 1, wants 0"]


def test_drift_ignores_terminated_foreign_and_ip_less_instances():
    instances = [active_inst(IP), active_inst("129.80.30.31", name="my-notebook"),
                 active_inst("129.80.30.32", name="k3s-gpu-a6000-2", status="terminated"),
                 active_inst(None, name="k3s-gpu-a6000-3", status="booting")]
    lines = find_drift(**{**CLEAN, "counts": {"a6000": 2}, "instances": instances})
    assert lines == ["k3s-gpu-a6000-3 is not in the cluster"]  # booting: no IP yet, so no rule is missing


@pytest.mark.parametrize("source", ["counts", "instances", "cluster_nodes", "tf_ips", "nft_elements"])
def test_drift_unavailable_sources_are_skipped_not_problems(source):
    lines = find_drift(**{**CLEAN, source: None})
    assert len(lines) == 1 and lines[0].startswith("skipped: ") and lines[0].endswith(" unavailable")
    assert problems_in(lines) == []


def test_drift_unavailable_sources_still_compare_the_rest():
    lines = find_drift(**{**CLEAN, "cluster_nodes": None, "tf_ips": ["198.51.100.99", IP]})
    assert problems_in(lines) == ["SECURITY: firewall rule for 198.51.100.99 has no live instance"]


# --- gpuctl status: drift and exit code 3 --------------------------------------------------------

ENV = {"LAMBDA_API_KEY": "k"}
NODE = active_inst(IP)
PEERS = ["10.10.0.2", "34.29.20.20", "34.82.10.10", IP]  # cp internal, worker, cp, GPU


def consistent_infra(**changes):
    kwargs = dict(outputs={**DEFAULT_OUTPUTS, "gpu-access": {"gpu_ips": [IP]}},
                  cluster_nodes={"k3s-gpu-a6000-1": IP}, nft_elements=PEERS)
    kwargs.update(changes)
    return FakeInfra(**kwargs)


@pytest.fixture(autouse=True)
def no_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr("gpuctl.__main__.OVERRIDES_PATH", tmp_path / ".autoscale-state.json")


def test_status_without_drift_exits_0(capsys):
    fake = FakeLambda(types=TYPES, instances=[NODE])
    assert main(["status"], env=ENV, client=fake, infra=consistent_infra()) == 0
    assert "No drift" in capsys.readouterr().out


def test_status_exit_code_3_on_drift(capsys):
    fake = FakeLambda(types=TYPES, instances=[NODE])
    infra = consistent_infra(outputs={**DEFAULT_OUTPUTS, "gpu-access": {"gpu_ips": [IP, "198.51.100.99"]}})
    assert main(["status"], env=ENV, client=fake, infra=infra) == 3
    assert "SECURITY: firewall rule for 198.51.100.99 has no live instance" in capsys.readouterr().out


def test_status_compares_only_gpu_entries_of_the_peer_set(capsys):
    # cp and worker addresses belong in k3s_peers; a GPU IP that is no longer live does not
    fake = FakeLambda(types=TYPES, instances=[NODE])
    infra = consistent_infra(nft_elements=[*PEERS, "198.51.100.99"])
    assert main(["status"], env=ENV, client=fake, infra=infra) == 3
    out = capsys.readouterr().out
    assert "198.51.100.99" in out and "34.29.20.20" not in out and "34.82.10.10" not in out


def test_status_reports_an_instance_terminated_behind_gpuctls_back(capsys):  # spec section 12, check 3
    gone = active_inst(IP, status="terminated")
    fake = FakeLambda(types=TYPES, instances=[gone])
    assert main(["status"], env=ENV, client=fake, infra=consistent_infra()) == 3
    out = capsys.readouterr().out
    assert f"firewall rule for {IP} has no live instance" in out and "stale node k3s-gpu-a6000-1" in out
    assert "pool a6000 has 0, wants 1" in out


def test_status_unreadable_sources_are_skipped_with_exit_0(capsys):
    fake = FakeLambda(types=TYPES, instances=[NODE])
    infra = consistent_infra(outputs={})  # no Terraform outputs: gpu-access state and the peer set are unknown

    def kubectl_down():
        raise ValueError("kubectl get nodes -o json: unexpected output")

    infra.cluster_gpu_nodes = kubectl_down
    assert main(["status"], env=ENV, client=fake, infra=infra) == 0
    captured = capsys.readouterr()
    assert captured.out.count("skipped: ") == 3
    assert "kubectl get nodes" in captured.err  # the reason goes to stderr
    # B-3: a skipped source is not an all-clear, so the summary never reads as a bare "No drift"
    assert "  No drift in the 2 sources read; 3 source(s) skipped (see above)\n" in captured.out
    assert "  No drift\n" not in captured.out


def test_status_with_the_gpu_access_state_skipped_says_so_in_the_summary(capsys):  # B-3
    fake = FakeLambda(types=TYPES, instances=[NODE])
    infra = consistent_infra(init_fail={"gpu-access"})  # e.g. GCS unreachable: the stale-firewall check is lost
    assert main(["status"], env=ENV, client=fake, infra=infra) == 0  # skipped sources stay exit 0 (brief)
    out = capsys.readouterr().out
    assert "skipped: gpu-access state unavailable" in out
    assert out.rstrip().endswith("No drift in the 4 sources read; 1 source(s) skipped (see above)")


def test_status_drift_uses_the_override_target(tmp_path, monkeypatch, capsys):
    path = tmp_path / "overrides.json"
    path.write_text('{"pools": {"a6000": 2}}')
    monkeypatch.setattr("gpuctl.__main__.OVERRIDES_PATH", path)
    fake = FakeLambda(types=TYPES, instances=[NODE])
    assert main(["status"], env=ENV, client=fake, infra=consistent_infra()) == 3
    assert "pool a6000 has 1, wants 2" in capsys.readouterr().out


def test_status_is_read_only():
    fake = FakeLambda(types=TYPES, instances=[NODE])
    infra = consistent_infra()
    main(["status"], env=ENV, client=fake, infra=infra)
    assert all(c[0] in ("instance_types", "list_instances") for c in fake.calls)
    assert infra.events == [] and infra.ssh_calls == [] and infra.deleted_nodes == [] and infra.joined == []
