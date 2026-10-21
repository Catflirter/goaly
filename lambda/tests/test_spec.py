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


# --- Regression tests beyond the brief's cases (review round 0: S0-1, C0-1, C0-2, X0-1..X0-5) ---

def test_pool_counts_may_sum_to_exactly_max_nodes():
    data = copy.deepcopy(VALID)
    data["pools"]["a6000"]["count"] = 2
    data["pools"]["b"] = {**copy.deepcopy(data["pools"]["a6000"]), "count": 1}
    assert sum(p.count for p in parse_spec(data).pools.values()) == 3

@pytest.mark.parametrize("mutate, message", [
    # max_nodes caps the sum across pools, not each pool on its own.
    (lambda d: (d["pools"]["a6000"].update(count=2),
                d["pools"].update(b={**copy.deepcopy(d["pools"]["a6000"]), "count": 2})), "max_nodes"),
    (lambda d: d.update(max_nodes=0), "max_nodes"),
    (lambda d: d["pools"]["a6000"]["autoscale"].update(idle_minutes=0), "idle_minutes"),
    (lambda d: d["pools"]["a6000"]["autoscale"].update(min=-1), "min"),
    (lambda d: d.update(budget=3), "unknown key"),
    (lambda d: d["pools"]["a6000"]["autoscale"].update(idle=3), "unknown key"),
    (lambda d: d["pools"]["a6000"].update(count=True), "count"),
    (lambda d: d.update(budget_usd_per_hour=float("nan")), "budget"),
    (lambda d: d.update(budget_usd_per_hour=float("inf")), "budget"),
    (lambda d: d.update(pools=[]), "pools"),
    (lambda d: d["pools"]["a6000"].update(regions="us-west-1"), "list of strings"),
    (lambda d: d["pools"]["a6000"].update(regions=["US West 1"]), "invalid region"),
    (lambda d: d["pools"]["a6000"].update(regions=["us-west-1;x"]), "invalid region"),
    (lambda d: d["pools"]["a6000"].update(regions=["us-" + "a" * 30]), "invalid region"),
    (lambda d: d["pools"]["a6000"].pop("image_family"), "missing key"),
    (lambda d: d["pools"]["a6000"].update(image_family="x; rm -rf / \n$(id)"), "image_family"),
    (lambda d: d["pools"]["a6000"].update(image_family=""), "image_family"),
    (lambda d: d["pools"]["a6000"].update(image_family="a" * 64), "image_family"),
    # arm64 is reported even when the name does not fit the instance-type pattern.
    (lambda d: d["pools"]["a6000"].update(instance_type="gh200"), "arm64"),
    (lambda d: d["pools"]["a6000"].update(instance_type="gpu_1x_GH200"), "arm64"),
    (lambda d: d["pools"]["a6000"]["fallback_types"].append("gpu_1x_gh200 "), "arm64"),
    # autoscale bounds must fit inside the max_nodes guardrail.
    (lambda d: d["pools"]["a6000"].update(autoscale={"min": 1, "max": 4, "idle_minutes": 5}), "max_nodes"),
    (lambda d: d["pools"].update(b={**copy.deepcopy(d["pools"]["a6000"]), "count": 0,
                                     "autoscale": {"min": 3, "max": 3, "idle_minutes": 5}}), "max_nodes"),
])
def test_extra_guards_reject(mutate, message):
    data = copy.deepcopy(VALID); mutate(data)
    with pytest.raises(SpecError, match=message):
        parse_spec(data)

def test_autoscale_bounds_may_reach_max_nodes():
    data = copy.deepcopy(VALID)
    data["pools"]["a6000"]["autoscale"] = {"min": 3, "max": 3, "idle_minutes": 5}
    assert parse_spec(data).pools["a6000"].autoscale.max == 3

REPO_YAML = (ROOT / "lambda/gpu-nodes.yaml").read_text(encoding="utf-8")

def test_repo_spec_file_is_english():
    # Spec section 0.3: deliverables are in English; the operator edits this file live (chapter 6).
    assert REPO_YAML.isascii()

@pytest.mark.parametrize("text, message", [
    (REPO_YAML + "  a6000:\n    count: 0\n    instance_type: gpu_1x_a10\n    fallback_types: []\n"
                 "    regions: [us-west-1]\n    image_family: lambda-stack-22-04\n", "duplicate key 'a6000'"),
    (REPO_YAML.replace("    count: 1", "    count: 3\n    count: 0", 1), "duplicate key 'count'"),
    (REPO_YAML + "max_nodes: 2\n", "duplicate key 'max_nodes'"),
    ("pools: [unclosed\n", "invalid YAML"),
], ids=["duplicate-pool", "duplicate-count", "duplicate-top-level", "malformed"])
def test_load_spec_rejects_bad_yaml(tmp_path, text, message):
    path = tmp_path / "gpu-nodes.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SpecError, match=message) as info:
        load_spec(path)
    assert str(path) in str(info.value)

def test_load_spec_rejects_non_utf8(tmp_path):
    path = tmp_path / "gpu-nodes.yaml"
    path.write_bytes(REPO_YAML.encode() + b"# caf\xe9\n")
    with pytest.raises(SpecError, match="UTF-8") as info:
        load_spec(path)
    assert str(path) in str(info.value)
