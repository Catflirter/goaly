import copy, pytest
from pathlib import Path
from gpuctl.spec import parse_spec, load_spec, SpecError

ROOT = Path(__file__).resolve().parents[2]
VALID = {"budget_usd_per_hour": 3.0, "max_nodes": 3, "pools": {"a6000": {
    "count": 1, "instance_type": "gpu_1x_a6000", "fallback_types": ["gpu_1x_a10", "gpu_1x_a100"],
    "regions": ["us-west-1", "us-west-2", "us-east-1"], "image_family": "lambda-stack-22-04",
    "autoscale": {"min": 1, "max": 2, "idle_minutes": 5}}}}

def test_repo_spec_file_matches_design():
    assert load_spec(ROOT / "lambda/gpu-nodes.yaml") == parse_spec(VALID)

def test_candidate_types_keep_order():
    assert parse_spec(VALID).pools["a6000"].candidate_types() == ("gpu_1x_a6000", "gpu_1x_a10", "gpu_1x_a100")

def test_regions_union_preserves_order():
    data = copy.deepcopy(VALID)
    data["pools"]["a100"] = {**data["pools"]["a6000"], "count": 0, "regions": ["us-east-1", "us-midwest-1"]}
    assert parse_spec(data).regions() == ("us-west-1", "us-west-2", "us-east-1", "us-midwest-1")

def test_autoscale_is_optional():
    data = copy.deepcopy(VALID); del data["pools"]["a6000"]["autoscale"]
    assert parse_spec(data).pools["a6000"].autoscale is None

@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["pools"]["a6000"].update(instance_type="gpu_1x_gh200"), "arm64"),
    (lambda d: d["pools"]["a6000"]["fallback_types"].append("gpu_1x_gh200"), "arm64"),
    (lambda d: d["pools"]["a6000"]["regions"].append("us-south-1"), "us-south-1"),
    (lambda d: d["pools"]["a6000"].update(regions=[]), "regions"),
    (lambda d: d["pools"]["a6000"].update(count=-1), "count"),
    (lambda d: d["pools"]["a6000"].update(count=4), "max_nodes"),
    (lambda d: d.update(budget_usd_per_hour=0), "budget"),
    (lambda d: d["pools"]["a6000"].update(autoscale={"min": 2, "max": 1, "idle_minutes": 5}), "autoscale"),
    (lambda d: d["pools"]["a6000"].update(instance_typo="x"), "unknown key"),
    (lambda d: d["pools"].update({"A6000!": d["pools"]["a6000"]}), "pool name"),
    (lambda d: d["pools"]["a6000"].update(instance_type="a6000"), "instance_type"),
])
def test_invalid_specs_are_rejected(mutate, message):
    data = copy.deepcopy(VALID); mutate(data)
    with pytest.raises(SpecError, match=message):
        parse_spec(data)
