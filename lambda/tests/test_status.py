import copy

from gpuctl.lambda_api import InstanceType
from gpuctl.spec import parse_spec
from gpuctl.status import CapacityRow, capacity_rows, render_capacity
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
