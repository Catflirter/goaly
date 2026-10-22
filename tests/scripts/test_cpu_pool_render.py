"""scripts/cpu-pool.sh render N: the CPU pool inventory (infra/workers tfvars) for N workers (Task 14).

render only reads ${WORKERS_TFVARS} and prints the new JSON: no .env, no cloud, nothing written.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ONE = {"k3s-cpu-1": {"zone": "us-central1-a", "machine_type": "e2-small"}}
CUSTOM = {"zone": "us-central1-b", "machine_type": "e2-standard-4"}


def render(tmp_path, current, n, **env):
    f = tmp_path / "w.json"
    f.write_text(json.dumps({"cpu_workers": current}))
    clean = {k: v for k, v in os.environ.items() if k not in ("CPU_ZONE", "CPU_TYPE", "WORKERS_TFVARS")}
    return subprocess.run(["bash", str(ROOT / "scripts/cpu-pool.sh"), "render", str(n)], capture_output=True,
                          text=True, env={**clean, "WORKERS_TFVARS": str(f), **env}, timeout=30)


def workers(proc):
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["cpu_workers"]


def test_scale_up_adds_defaults(tmp_path):
    out = workers(render(tmp_path, ONE, 3))
    assert list(out) == ["k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-3"] and out["k3s-cpu-3"] == ONE["k3s-cpu-1"]


def test_scale_up_honours_env_type(tmp_path):
    out = workers(render(tmp_path, ONE, 2, CPU_TYPE="e2-standard-2", CPU_ZONE="us-central1-c"))
    assert out["k3s-cpu-1"] == ONE["k3s-cpu-1"], "CPU_TYPE/CPU_ZONE only affect the nodes render adds"
    assert out["k3s-cpu-2"] == {"zone": "us-central1-c", "machine_type": "e2-standard-2"}


def test_scale_down_removes_highest_keeps_custom(tmp_path):
    current = {**ONE, "k3s-cpu-2": CUSTOM, "k3s-cpu-3": ONE["k3s-cpu-1"]}
    out = workers(render(tmp_path, current, 2))
    assert out == {**ONE, "k3s-cpu-2": CUSTOM}


def test_same_n_changes_nothing(tmp_path):
    current = {**ONE, "k3s-cpu-2": CUSTOM}
    assert workers(render(tmp_path, current, 2)) == current


def test_numeric_order_k3s_cpu_9_before_k3s_cpu_10(tmp_path):
    """Names sort by number, not as strings: scaling 11 -> 10 removes k3s-cpu-11, and 10 -> 9 removes 10."""
    current = {f"k3s-cpu-{i}": ONE["k3s-cpu-1"] for i in range(1, 12)}
    out = workers(render(tmp_path, current, 10))
    assert list(out) == [f"k3s-cpu-{i}" for i in range(1, 11)]
    out = workers(render(tmp_path, current, 9))
    assert list(out) == [f"k3s-cpu-{i}" for i in range(1, 10)]


def test_output_is_ordered_numerically_whatever_the_file_order(tmp_path):
    current = {"k3s-cpu-10": CUSTOM, "k3s-cpu-1": ONE["k3s-cpu-1"], "k3s-cpu-9": CUSTOM}
    assert list(workers(render(tmp_path, current, 3))) == ["k3s-cpu-1", "k3s-cpu-9", "k3s-cpu-10"]


def test_scale_up_fills_the_lowest_free_numbers(tmp_path):
    """After `cpu-remove k3s-cpu-2` the pool is {1, 3}; growing it reuses 2 instead of jumping to 4,
    so the pool stays k3s-cpu-1..N (spec §5.3) and the surviving nodes keep their names."""
    current = {**ONE, "k3s-cpu-3": CUSTOM}
    out = workers(render(tmp_path, current, 4))
    assert list(out) == ["k3s-cpu-1", "k3s-cpu-2", "k3s-cpu-3", "k3s-cpu-4"]
    assert out["k3s-cpu-3"] == CUSTOM and out["k3s-cpu-2"] == ONE["k3s-cpu-1"]
    assert workers(render(tmp_path, current, 2)) == current, "two nodes already: nothing changes"


def test_other_top_level_keys_are_kept(tmp_path):
    f = tmp_path / "w.json"
    f.write_text(json.dumps({"cpu_workers": ONE, "disk_size_gb": 30}))
    proc = subprocess.run(["bash", str(ROOT / "scripts/cpu-pool.sh"), "render", "1"], capture_output=True,
                          text=True, env={**os.environ, "WORKERS_TFVARS": str(f)}, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"cpu_workers": ONE, "disk_size_gb": 30}


def test_render_of_the_committed_inventory_is_byte_identical():
    """make cpu-scale N=1 on a fresh checkout leaves infra/workers/workers.auto.tfvars.json unchanged."""
    committed = ROOT / "infra/workers/workers.auto.tfvars.json"
    proc = subprocess.run(["bash", str(ROOT / "scripts/cpu-pool.sh"), "render", "1"], capture_output=True,
                          text=True, env={**os.environ, "WORKERS_TFVARS": str(committed)}, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == committed.read_text()


@pytest.mark.parametrize("n", [0, "x", -1, "1.5", "", "01", "3; reboot", 101])
def test_invalid_n_is_usage_error(tmp_path, n):
    proc = render(tmp_path, ONE, n)
    assert proc.returncode == 2 and proc.stdout == ""


@pytest.mark.parametrize("var,value", [("CPU_ZONE", "us-central1"), ("CPU_ZONE", "a b"),
                                       ("CPU_TYPE", "e2 small"), ("CPU_TYPE", "E2-SMALL")])
def test_invalid_zone_or_type_is_usage_error(tmp_path, var, value):
    proc = render(tmp_path, ONE, 2, **{var: value})
    assert proc.returncode == 2 and proc.stdout == ""


@pytest.mark.parametrize("current", [{"k3s-cp": ONE["k3s-cpu-1"]}, {"k3s-cpu-01": ONE["k3s-cpu-1"]},
                                     {"worker": ONE["k3s-cpu-1"]}, {"k3s-cpu-1x": ONE["k3s-cpu-1"]}])
def test_a_file_with_names_outside_the_pool_is_refused(tmp_path, current):
    """Anchored names: k3s-cp (or anything but k3s-cpu-<n>) in the inventory is an error, never renamed."""
    proc = render(tmp_path, current, 1)
    assert proc.returncode == 1 and proc.stdout == ""
    assert "k3s-cpu-<n>" in proc.stderr


def test_a_missing_file_is_an_error(tmp_path):
    proc = subprocess.run(["bash", str(ROOT / "scripts/cpu-pool.sh"), "render", "1"], capture_output=True,
                          text=True, env={**os.environ, "WORKERS_TFVARS": str(tmp_path / "nope.json")}, timeout=30)
    assert proc.returncode == 1 and proc.stdout == ""
