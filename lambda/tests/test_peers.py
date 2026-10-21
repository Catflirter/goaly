import dataclasses

import pytest

from gpuctl.lambda_api import FirewallRule
from gpuctl.peers import (
    PeerSet, cp_guard_elements, global_rules, nft_replace_script, region_ruleset_rules, ruleset_name,
)

CP, CP_INTERNAL, WORKER, GPU, OP = "34.82.10.10", "10.10.0.2", "34.29.20.20", "129.80.30.30", "73.15.40.40"


def peers(**overrides) -> PeerSet:
    fields = dict(cp_ip=CP, cp_internal_ip=CP_INTERNAL, worker_ips=(WORKER,), gpu_ips=(GPU,), operator_ip=OP)
    return PeerSet(**{**fields, **overrides})


def test_guard_elements_sorted_unique():
    p = PeerSet("34.82.10.10", "10.10.0.2", ("34.29.20.20", "34.29.20.20"), ("129.80.30.30",), "73.15.40.40")
    assert cp_guard_elements(p) == ["10.10.0.2", "34.29.20.20", "34.82.10.10", "129.80.30.30"]


def test_guard_elements_exclude_operator_and_survive_no_gpus():
    assert cp_guard_elements(peers(gpu_ips=())) == ["10.10.0.2", "34.29.20.20", "34.82.10.10"]


def test_nft_script_exact():
    assert nft_replace_script(["10.10.0.2", "34.82.10.10"]) == (
        "flush set inet k3s_guard k3s_peers\n"
        "add element inet k3s_guard k3s_peers { 10.10.0.2, 34.82.10.10 }\n")


@pytest.mark.parametrize("elements", [[], ["34.82.10.10; flush ruleset"], ["34.82.10.10 }"]])
def test_nft_script_rejects_empty_or_non_ip_elements(elements):
    with pytest.raises(ValueError):
        nft_replace_script(elements)


def test_global_rules_exact():
    assert global_rules(OP) == [FirewallRule("tcp", "73.15.40.40/32", (22, 22), "operator ssh")]


def test_global_rules_reject_non_global_operator_ip():
    with pytest.raises(ValueError):
        global_rules("192.168.1.5")


def test_region_rules_exact():
    assert region_ruleset_rules(peers()) == [
        FirewallRule("tcp", "73.15.40.40/32", (22, 22), "operator ssh"),
        FirewallRule("tcp", "34.82.10.10/32", (10250, 10250), "metrics-server from cp"),
        FirewallRule("udp", "34.29.20.20/32", (51820, 51820), "wireguard 34.29.20.20"),
        FirewallRule("icmp", "34.29.20.20/32", None, "icmp 34.29.20.20"),
        FirewallRule("udp", "34.82.10.10/32", (51820, 51820), "wireguard 34.82.10.10"),
        FirewallRule("icmp", "34.82.10.10/32", None, "icmp 34.82.10.10"),
        FirewallRule("udp", "129.80.30.30/32", (51820, 51820), "wireguard 129.80.30.30"),
        FirewallRule("icmp", "129.80.30.30/32", None, "icmp 129.80.30.30"),
    ]


def test_region_rules_dedupe_and_never_expose_cp_internal_ip():
    rules = region_ruleset_rules(peers(worker_ips=(WORKER, WORKER), gpu_ips=(GPU, CP)))
    assert [r.description for r in rules].count("wireguard 34.82.10.10") == 1
    assert len(rules) == 2 + 2 * 3
    assert not any(CP_INTERNAL in r.source_network for r in rules)


def test_operator_ip_change_leaves_no_trace_of_old_ip():  # Review Focus #1
    old = region_ruleset_rules(peers(operator_ip="73.15.40.40"))
    new = region_ruleset_rules(peers(operator_ip="73.15.99.99"))
    assert any("73.15.40.40" in r.source_network for r in old)
    assert not any("73.15.40.40" in r.source_network for r in new + global_rules("73.15.99.99"))
    assert FirewallRule("tcp", "73.15.99.99/32", (22, 22), "operator ssh") in new


@pytest.mark.parametrize("field, value", [
    ("cp_ip", "10.0.0.1"), ("operator_ip", "192.168.1.5"), ("gpu_ips", ("not-an-ip",)),
    ("cp_internal_ip", "34.82.10.10"), ("worker_ips", ("100.64.0.1",)), ("cp_ip", "192.0.2.1"),
    ("operator_ip", "2001:4860:4860::8888"), ("cp_ip", ""), ("worker_ips", (None,)),
])
def test_invalid_addresses_rejected(field, value):
    with pytest.raises(ValueError):
        peers(**{field: value})


def test_peerset_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        peers().operator_ip = "73.15.99.99"


def test_ruleset_name():
    assert ruleset_name("us-west-1") == "k3s-gpu-us-west-1"
