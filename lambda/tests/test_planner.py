import copy

import pytest

from gpuctl.lambda_api import Instance, InstanceType
from gpuctl.planner import (
    LIVE_STATUSES, PREFIX, BudgetExceeded, Plan, PlanError, check_limits, choose_placement, gpu_type_label,
    make_plan, node_name, owned_instances, parse_node_name, target_counts,
)
from gpuctl.spec import parse_spec
from test_spec import VALID

SPEC = parse_spec(VALID)
POOL = SPEC.pools["a6000"]
TYPES = {
    "gpu_1x_a6000": InstanceType("gpu_1x_a6000", 109, ("us-west-2",)),
    "gpu_1x_a10": InstanceType("gpu_1x_a10", 129, ()),
    "gpu_1x_a100": InstanceType("gpu_1x_a100", 199, ("us-west-1",)),
}


def inst(name, id=None, status="active", ip="203.0.113.10", instance_type="gpu_1x_a6000", region="us-west-2"):
    return Instance(id=id or f"id-{name}", name=name, status=status, ip=ip, region=region,
                    instance_type=instance_type)


def spec_with(**changes):
    data = copy.deepcopy(VALID)
    data.update(changes)
    return parse_spec(data)


def test_constants():
    assert PREFIX == "k3s-gpu-"
    assert LIVE_STATUSES == frozenset({"booting", "active", "unhealthy"})


def test_empty_account_plans_one_create_at_worst_case_price():
    plan = make_plan(SPEC, [], TYPES, {"a6000": 1})
    assert plan.create == (("k3s-gpu-a6000-1", "a6000"),) and plan.delete == () and plan.hourly_usd == 1.99


def test_active_instance_is_kept():
    existing = inst("k3s-gpu-a6000-1")
    plan = make_plan(SPEC, [existing], TYPES, {"a6000": 1})
    assert plan == Plan(create=(), delete=(), keep=(existing,), hourly_usd=1.09)


def test_booting_without_ip_counts_as_existing():
    booting = inst("k3s-gpu-a6000-1", status="booting", ip=None)
    plan = make_plan(SPEC, [booting], TYPES, {"a6000": 1})
    assert plan.create == () and plan.delete == () and plan.keep == (booting,)
    assert plan.hourly_usd == 1.09


def test_unhealthy_counts_as_existing():
    unhealthy = inst("k3s-gpu-a6000-1", status="unhealthy")
    plan = make_plan(SPEC, [unhealthy], TYPES, {"a6000": 1})
    assert plan.create == () and plan.delete == () and plan.keep == (unhealthy,)


def test_terminating_and_terminated_are_ignored():
    gone = [inst("k3s-gpu-a6000-1", id="t1", status="terminating"),
            inst("k3s-gpu-a6000-1", id="t2", status="terminated")]
    plan = make_plan(SPEC, gone, TYPES, {"a6000": 1})
    assert plan.create == (("k3s-gpu-a6000-1", "a6000"),)
    assert plan.delete == () and plan.keep == ()
    assert owned_instances(gone) == []


def test_foreign_instances_are_never_touched():
    plan = make_plan(SPEC, [inst("my-notebook"), inst("k3s-gpux-1")], TYPES, {"a6000": 0})
    assert plan.delete == () and plan.keep == ()
    assert plan.create == () and plan.hourly_usd == 0


def test_owned_instances_filters_prefix_and_status():
    ours = inst("k3s-gpu-a6000-1")
    instances = [ours, inst("my-notebook"), inst("k3s-gpux-1"), inst("k3s-gpu-a6000-2", status="terminated")]
    assert owned_instances(instances) == [ours]


def test_scale_down_deletes_highest_index_first():
    existing = [inst("k3s-gpu-a6000-1"), inst("k3s-gpu-a6000-2"), inst("k3s-gpu-a6000-3")]
    plan = make_plan(SPEC, existing, TYPES, {"a6000": 1})
    assert [i.name for i in plan.delete] == ["k3s-gpu-a6000-3", "k3s-gpu-a6000-2"]
    assert [i.name for i in plan.keep] == ["k3s-gpu-a6000-1"] and plan.create == ()


def test_pool_removed_from_spec_is_deleted():
    old = inst("k3s-gpu-old-1")
    plan = make_plan(SPEC, [old], TYPES, {"a6000": 0})
    assert plan.delete == (old,) and plan.keep == () and plan.create == ()


def test_duplicate_names_keep_smallest_id():
    b = inst("k3s-gpu-a6000-1", id="b")
    a = inst("k3s-gpu-a6000-1", id="a")
    plan = make_plan(SPEC, [b, a], TYPES, {"a6000": 1})
    assert plan.keep == (a,) and plan.delete == (b,) and plan.create == ()
    assert plan.duplicates == (b,)


def test_duplicates_are_marked_so_the_executor_spares_the_shared_name():
    # Duplicates share a name (= K8s node name, host key) with another instance: Plan.duplicates tells the
    # executor to terminate them by id only. Ordinary deletes and keeps are never marked.
    b = inst("k3s-gpu-a6000-1", id="b", status="active")
    a = inst("k3s-gpu-a6000-1", id="a", status="booting", ip=None)
    extra = inst("k3s-gpu-a6000-2")
    plan = make_plan(SPEC, [b, a, extra], TYPES, {"a6000": 1})
    assert plan.keep == (a,) and plan.delete == (extra, b) and plan.duplicates == (b,)
    assert {i.name for i in plan.duplicates} <= {i.name for i in plan.keep}


def test_duplicates_of_a_deleted_name_are_marked_too():
    # The smallest id carries the name (drain + forget once); the other copy is terminated by id only.
    first = inst("k3s-gpu-old-1", id="a")
    second = inst("k3s-gpu-old-1", id="b")
    plan = make_plan(SPEC, [second, first], TYPES, {"a6000": 0})
    assert plan.delete == (first, second) and plan.duplicates == (second,) and plan.keep == ()


def test_plan_without_duplicates_has_none_marked():
    plan = make_plan(SPEC, [inst("k3s-gpu-a6000-1"), inst("k3s-gpu-a6000-2")], TYPES, {"a6000": 1})
    assert plan.duplicates == ()


def test_override_is_clamped_to_autoscale_bounds():
    assert target_counts(SPEC, overrides={"a6000": 5}) == {"a6000": 2}
    assert target_counts(SPEC, overrides={"a6000": 0}) == {"a6000": 1}


def test_set_counts_wins_over_overrides():
    assert target_counts(SPEC, overrides={"a6000": 2}, set_counts={"a6000": 0}) == {"a6000": 0}


def test_unknown_pool_in_set_counts_is_rejected():
    with pytest.raises(PlanError, match="nope"):
        target_counts(SPEC, set_counts={"nope": 1})


def test_override_ignored_without_autoscale():
    data = copy.deepcopy(VALID)
    del data["pools"]["a6000"]["autoscale"]
    assert target_counts(parse_spec(data), overrides={"a6000": 2}) == {"a6000": 1}


def test_target_counts_defaults_to_count():
    assert target_counts(SPEC) == {"a6000": 1}


def test_choose_placement_is_type_major():
    assert choose_placement(POOL, TYPES) == ("gpu_1x_a6000", "us-west-2")


def test_choose_placement_respects_exclude():
    assert choose_placement(POOL, TYPES, exclude=frozenset({("gpu_1x_a6000", "us-west-2")})) == (
        "gpu_1x_a100", "us-west-1")


def test_choose_placement_skips_a_candidate_type_lambda_does_not_list():
    assert choose_placement(POOL, {"gpu_1x_a100": TYPES["gpu_1x_a100"]}) == ("gpu_1x_a100", "us-west-1")


def test_choose_placement_none_without_capacity():
    no_capacity = {name: InstanceType(t.name, t.price_cents_per_hour, ()) for name, t in TYPES.items()}
    assert choose_placement(POOL, no_capacity) is None
    # Capacity only in a region the pool does not list does not count.
    elsewhere = {"gpu_1x_a6000": InstanceType("gpu_1x_a6000", 109, ("us-east-3",))}
    assert choose_placement(POOL, elsewhere) is None


def test_budget_exceeded():
    spec = spec_with(budget_usd_per_hour=1.5)
    plan = make_plan(spec, [], TYPES, {"a6000": 1})
    with pytest.raises(BudgetExceeded):
        check_limits(plan, spec)


def test_budget_equal_to_plan_is_allowed():
    spec = spec_with(budget_usd_per_hour=1.99)
    check_limits(make_plan(spec, [], TYPES, {"a6000": 1}), spec)


def test_max_nodes_enforced():
    spec = spec_with(budget_usd_per_hour=100.0)
    plan = make_plan(spec, [], TYPES, target_counts(spec, set_counts={"a6000": 4}))
    assert len(plan.create) == 4
    with pytest.raises(PlanError, match="max_nodes") as excinfo:
        check_limits(plan, spec)
    assert not isinstance(excinfo.value, BudgetExceeded)


def test_max_nodes_equal_is_allowed():
    spec = spec_with(budget_usd_per_hour=100.0)
    plan = make_plan(spec, [], TYPES, {"a6000": 3})
    assert len(plan.create) == spec.max_nodes == 3
    check_limits(plan, spec)


@pytest.mark.parametrize("count", [1001, 10**9])
def test_absurd_count_fails_fast_without_building_the_plan(count):
    # A --set-count typo must not allocate one create entry per requested node before check_limits runs.
    counts = target_counts(SPEC, set_counts={"a6000": count})
    with pytest.raises(PlanError, match="max_nodes") as excinfo:
        make_plan(SPEC, [], TYPES, counts)
    assert not isinstance(excinfo.value, BudgetExceeded)


def test_max_nodes_checked_before_budget():
    plan = make_plan(SPEC, [], TYPES, target_counts(SPEC, set_counts={"a6000": 4}))
    with pytest.raises(PlanError, match="max_nodes"):
        check_limits(plan, SPEC)


def test_gpu_type_label():
    assert gpu_type_label("gpu_1x_a6000") == "a6000"
    assert gpu_type_label("gpu_8x_a100_80gb_sxm4") == "a100_80gb_sxm4"


@pytest.mark.parametrize("name", ["xgpu_1x_a6000", "cpu_gpu_2x_a", "a6000", "gpu_x_a6000"])
def test_gpu_type_label_strips_only_a_leading_prefix(name):
    assert gpu_type_label(name) == name


def test_node_name_roundtrip():
    assert parse_node_name(node_name("a6000", 12)) == ("a6000", 12) and parse_node_name("k3s-gpu-a6000") is None


@pytest.mark.parametrize("name", ["k3s-gpu-a6000-0", "k3s-gpu-a6000-01", "k3s-gpu-A6000-1", "k3s-gpu--1",
                                  "k3s-gpu-a-6000-1", "my-k3s-gpu-a6000-1", "k3s-gpu-a6000-1 "])
def test_parse_node_name_rejects_non_canonical(name):
    assert parse_node_name(name) is None


# --- Planner rules the brief leaves to the implementation ---

def test_scale_up_fills_lowest_free_index():
    existing = inst("k3s-gpu-a6000-2")
    plan = make_plan(SPEC, [existing], TYPES, {"a6000": 2})
    assert plan.create == (("k3s-gpu-a6000-1", "a6000"),) and plan.keep == (existing,)


def test_scale_up_from_one_creates_the_next_index():
    existing = inst("k3s-gpu-a6000-1")
    plan = make_plan(SPEC, [existing], TYPES, {"a6000": 2})
    assert plan.create == (("k3s-gpu-a6000-2", "a6000"),) and plan.keep == (existing,)
    assert plan.hourly_usd == pytest.approx(1.09 + 1.99)


def test_scale_up_fills_a_gap_between_existing_indices():
    existing = [inst("k3s-gpu-a6000-1"), inst("k3s-gpu-a6000-3")]
    plan = make_plan(spec_with(budget_usd_per_hour=100.0), existing, TYPES, {"a6000": 3})
    assert plan.create == (("k3s-gpu-a6000-2", "a6000"),) and plan.keep == tuple(existing)


def test_owned_name_outside_the_naming_scheme_is_deleted():
    stray = inst("k3s-gpu-scratch")
    plan = make_plan(SPEC, [stray], TYPES, {"a6000": 0})
    assert plan.delete == (stray,) and plan.keep == ()


def test_hourly_prices_keep_by_actual_type_and_create_by_worst_case():
    kept = inst("k3s-gpu-a6000-1", instance_type="gpu_1x_a10")
    plan = make_plan(SPEC, [kept], TYPES, {"a6000": 2})
    assert plan.hourly_usd == pytest.approx(1.29 + 1.99)


def test_kept_instance_with_unknown_price_is_rejected():
    with pytest.raises(PlanError, match="gpu_1x_h100"):
        make_plan(SPEC, [inst("k3s-gpu-a6000-1", instance_type="gpu_1x_h100")], TYPES, {"a6000": 1})


def test_create_without_any_priced_candidate_is_rejected():
    with pytest.raises(PlanError, match="a6000"):
        make_plan(SPEC, [], {}, {"a6000": 1})


@pytest.mark.parametrize("counts", [{}, {"a6000": 1, "nope": 1}, {"a6000": -1}, {"a6000": True}])
def test_make_plan_rejects_counts_not_matching_spec(counts):
    with pytest.raises(PlanError):
        make_plan(SPEC, [], TYPES, counts)


def test_negative_set_count_is_rejected():
    with pytest.raises(PlanError, match="a6000"):
        target_counts(SPEC, set_counts={"a6000": -1})
