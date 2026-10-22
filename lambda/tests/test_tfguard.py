"""Stretch S3: the Terraform plan guardrail for unattended worker scaling (design spec sections 3.1, 10.2, 10.5).

check_workers_plan is pure; `gpuctl tfguard` reads `terraform show -json` on stdin and needs no Lambda API key.
"""
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gpuctl.__main__ import main
from gpuctl.tfguard import check_workers_plan

ROOT = Path(__file__).resolve().parents[2]
TYPES = ("instance.worker", "address.worker", "firewall.worker_to_cp", "firewall.worker_mesh")


def rc(address, actions):
    return {"address": address, "change": {"actions": actions}}


def plan(*changes):
    return {"format_version": "1.2", "resource_changes": list(changes)}


def worker(name, actions):
    return [rc(f'google_compute_{t}["{name}"]', actions) for t in TYPES]


ADD2 = worker("k3s-cpu-2", ["create"])
DEL2 = worker("k3s-cpu-2", ["delete"])
SHARED_NOOP = [rc("google_compute_firewall.cp_to_workers", ["no-op"]),
               rc("google_compute_firewall.workers_iap_ssh", ["no-op"]),
               rc("data.terraform_remote_state.base", ["read"])]


# ---- check_workers_plan --------------------------------------------------------------------------

def test_clean_scale_up_passes():
    assert check_workers_plan(plan(*ADD2, rc("google_compute_firewall.cp_to_workers", ["no-op"])),
                              add={"k3s-cpu-2"}, remove=set()) == []


def test_clean_scale_down_passes():
    assert check_workers_plan(plan(*DEL2, *SHARED_NOOP), add=set(), remove={"k3s-cpu-2"}) == []


def test_add_and_remove_together_pass():
    changes = [*worker("k3s-cpu-3", ["create"]), *worker("k3s-cpu-5", ["delete"]), *SHARED_NOOP]
    assert check_workers_plan(plan(*changes), add={"k3s-cpu-3"}, remove={"k3s-cpu-5"}) == []


def test_two_workers_added_pass():
    changes = [*worker("k3s-cpu-2", ["create"]), *worker("k3s-cpu-10", ["create"])]
    assert check_workers_plan(plan(*changes), add={"k3s-cpu-2", "k3s-cpu-10"}, remove=set()) == []


def test_touching_control_plane_is_a_violation():
    v = check_workers_plan(plan(*ADD2, rc("google_compute_instance.cp", ["update"])), add={"k3s-cpu-2"}, remove=set())
    assert v and "google_compute_instance.cp" in v[0]


@pytest.mark.parametrize("actions", [["delete", "create"], ["create", "delete"]])
def test_replace_is_a_violation(actions):
    changes = [*ADD2[:3], rc(ADD2[3]["address"], actions)]
    v = check_workers_plan(plan(*changes), add={"k3s-cpu-2"}, remove=set())
    assert any(ADD2[3]["address"] in line for line in v)


def test_replace_of_a_worker_being_removed_is_a_violation():
    changes = [*DEL2[1:], rc(DEL2[0]["address"], ["delete", "create"])]
    v = check_workers_plan(plan(*changes), add=set(), remove={"k3s-cpu-2"})
    assert any(DEL2[0]["address"] in line for line in v)


def test_update_of_another_worker_is_a_violation():
    v = check_workers_plan(plan(*ADD2, rc('google_compute_instance.worker["k3s-cpu-1"]', ["update"])),
                           add={"k3s-cpu-2"}, remove=set())
    assert any('worker["k3s-cpu-1"]' in line for line in v)


def test_update_of_a_shared_rule_is_a_violation():
    """cp_to_workers and workers_iap_ssh are shared: they may only appear as no-op or read."""
    v = check_workers_plan(plan(*ADD2, rc("google_compute_firewall.cp_to_workers", ["update"])),
                           add={"k3s-cpu-2"}, remove=set())
    assert any("cp_to_workers" in line for line in v)


def test_unexpected_extra_create_is_a_violation():
    v = check_workers_plan(plan(*ADD2, *worker("k3s-cpu-3", ["create"])), add={"k3s-cpu-2"}, remove=set())
    assert any('worker["k3s-cpu-3"]' in line for line in v)


def test_count_mismatch_is_a_violation():
    v = check_workers_plan(plan(*ADD2[:3]), add={"k3s-cpu-2"}, remove=set())
    assert any("3 create" in line and "expected 4" in line for line in v)
    assert any(ADD2[3]["address"] in line for line in v), "the missing resource is named"


def test_delete_count_mismatch_is_a_violation():
    v = check_workers_plan(plan(*DEL2[:2]), add=set(), remove={"k3s-cpu-2"})
    assert any("2 delete" in line and "expected 4" in line for line in v)


def test_duplicate_address_is_a_violation():
    v = check_workers_plan(plan(*ADD2, ADD2[0]), add={"k3s-cpu-2"}, remove=set())
    assert any("more than once" in line for line in v)


def test_delete_of_a_worker_that_should_be_created_is_a_violation():
    v = check_workers_plan(plan(*DEL2), add={"k3s-cpu-2"}, remove=set())
    assert v and all("k3s-cpu-2" in line for line in v[:4])


def test_a_worker_resource_inside_a_module_is_not_a_worker_resource():
    changes = [rc(f'module.x.google_compute_{t}["k3s-cpu-2"]', ["create"]) for t in TYPES]
    assert check_workers_plan(plan(*changes), add={"k3s-cpu-2"}, remove=set())


@pytest.mark.parametrize("actions", [["forget"], ["create_then_delete"], [], "create", None])
def test_unknown_or_malformed_actions_are_violations(actions):
    v = check_workers_plan(plan(*ADD2[:3], {"address": ADD2[3]["address"], "change": {"actions": actions}}),
                           add={"k3s-cpu-2"}, remove=set())
    assert v


@pytest.mark.parametrize("doc", [[], "x", None, {"resource_changes": {}}, {"resource_changes": [1]},
                                 {"resource_changes": [{"change": {"actions": ["create"]}}]},
                                 {"resource_changes": [{"address": "a"}]}])
def test_a_malformed_plan_is_a_violation_not_an_exception(doc):
    assert check_workers_plan(doc, add={"k3s-cpu-2"}, remove=set())
    assert check_workers_plan(doc, add=set(), remove=set())


def test_an_empty_plan_with_nothing_expected_passes():
    assert check_workers_plan(plan(*SHARED_NOOP), add=set(), remove=set()) == []


@pytest.mark.parametrize("add,remove", [({"k3s-cp"}, set()), ({"k3s-cpu-0"}, set()), ({"k3s-cpu-02"}, set()),
                                        ({"k3s-cpu-2 "}, set()), ({'k3s-cpu-2"]'}, set()), (set(), {"k3s-cpu-1"}),
                                        ({"k3s-cpu-2"}, {"k3s-cpu-2"}), ({"k3s-gpu-a6000-1"}, set())])
def test_bad_names_are_refused(add, remove):
    with pytest.raises(ValueError):
        check_workers_plan(plan(), add=add, remove=remove)


# ---- gpuctl tfguard ------------------------------------------------------------------------------

def run_cli(monkeypatch, capsys, text, *args):
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    code = main(["tfguard", *args], env={})
    return code, capsys.readouterr()


def test_cli_exit_codes(monkeypatch, capsys):
    code, cap = run_cli(monkeypatch, capsys, json.dumps(plan(*ADD2, *SHARED_NOOP)), "--add", "k3s-cpu-2")
    assert code == 0, cap.err
    code, cap = run_cli(monkeypatch, capsys, json.dumps(plan(*ADD2, rc("google_compute_instance.cp", ["update"]))),
                        "--add", "k3s-cpu-2")
    assert code == 1
    assert "google_compute_instance.cp" in cap.out


def test_cli_prints_every_violation_on_its_own_line(monkeypatch, capsys):
    doc = plan(*ADD2, rc("google_compute_instance.cp", ["update"]), rc("google_compute_network.core", ["delete"]))
    code, cap = run_cli(monkeypatch, capsys, json.dumps(doc), "--add", "k3s-cpu-2")
    assert code == 1
    lines = [line for line in cap.out.splitlines() if line.startswith("violation: ")]
    assert len(lines) >= 2
    assert any("google_compute_instance.cp" in line for line in lines)
    assert any("google_compute_network.core" in line for line in lines)


def test_cli_takes_several_names_and_both_sets(monkeypatch, capsys):
    doc = plan(*worker("k3s-cpu-2", ["create"]), *worker("k3s-cpu-3", ["create"]), *worker("k3s-cpu-4", ["delete"]))
    code, cap = run_cli(monkeypatch, capsys, json.dumps(doc), "--add", "k3s-cpu-2", "k3s-cpu-3",
                        "--remove", "k3s-cpu-4")
    assert code == 0, cap.err


@pytest.mark.parametrize("text", ["", "not json", '{"resource_changes": [', "[]", "null", '"x"'])
def test_cli_fails_closed_on_malformed_input(monkeypatch, capsys, text):
    code, cap = run_cli(monkeypatch, capsys, text, "--add", "k3s-cpu-2")
    assert code == 1
    assert "violation: " in cap.out


def test_cli_fails_closed_on_input_that_is_not_utf8(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"\xff\xfe{"), encoding="utf-8"))
    assert main(["tfguard", "--add", "k3s-cpu-2"], env={}) == 1


@pytest.mark.parametrize("args", [["--remove", "k3s-cpu-1"], ["--add", "k3s-cp"], ["--add", "k3s-cpu-x"],
                                  ["--add", "k3s-cpu-2", "--remove", "k3s-cpu-2"], ["--frobnicate"]])
def test_cli_refuses_bad_names_with_exit_2(monkeypatch, capsys, args):
    code, cap = run_cli(monkeypatch, capsys, json.dumps(plan()), *args)
    assert code == 2
    assert cap.out == ""


def test_cli_needs_no_lambda_api_key(monkeypatch, capsys):
    """env={} in every call above already: tfguard never builds a Lambda client."""
    code, cap = run_cli(monkeypatch, capsys, json.dumps(plan(*DEL2)), "--remove", "k3s-cpu-2")
    assert code == 0, cap.err
    assert "LAMBDA_API_KEY" not in cap.err


def test_cli_cleans_terminal_control_characters_from_addresses(monkeypatch, capsys):
    code, cap = run_cli(monkeypatch, capsys, json.dumps(plan(rc("google_x.y\x1b[2J", ["update"]))))
    assert code == 1
    assert "\x1b" not in cap.out


# ---- the scripts/gpuctl wrapper ------------------------------------------------------------------

@pytest.fixture
def wrapper_repo(tmp_path):
    """A repo copy holding only the real scripts/gpuctl, common.sh and lambda/gpuctl, with no .env at all."""
    repo = tmp_path / "repo"
    (repo / "scripts/lib").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/gpuctl", repo / "scripts/gpuctl")
    shutil.copy2(ROOT / "scripts/lib/common.sh", repo / "scripts/lib/common.sh")
    shutil.copytree(ROOT / "lambda/gpuctl", repo / "lambda/gpuctl")
    (repo / ".venv/bin").mkdir(parents=True)
    os.symlink(sys.executable, repo / ".venv/bin/python")
    return repo


def test_the_wrapper_runs_tfguard_without_env_file_or_lambda_key(wrapper_repo):
    env = {k: v for k, v in os.environ.items() if k not in ("LAMBDA_API_KEY", "PYTHONPATH")}
    proc = subprocess.run([str(wrapper_repo / "scripts/gpuctl"), "tfguard", "--add", "k3s-cpu-2"],
                          input=json.dumps(plan(*ADD2)), capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run([str(wrapper_repo / "scripts/gpuctl"), "tfguard", "--add", "k3s-cpu-2"],
                          input=json.dumps(plan(*ADD2[:3])), capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 1 and "violation: " in proc.stdout


def test_the_wrapper_hides_the_lambda_key_from_tfguard(wrapper_repo, tmp_path):
    """tfguard needs no secret, so the wrapper does not hand it LAMBDA_API_KEY even when the caller's environment has it."""
    probe = tmp_path / "probe"
    probe.write_text("")
    (wrapper_repo / "lambda/gpuctl/__main__.py").write_text(
        "import os, sys\n"
        f"open({str(probe)!r}, 'w').write(os.environ.get('LAMBDA_API_KEY', '<unset>'))\n")
    env = {**os.environ, "LAMBDA_API_KEY": "secret-key"}
    subprocess.run([str(wrapper_repo / "scripts/gpuctl"), "tfguard"], input="", capture_output=True, text=True,
                   env=env, timeout=60)
    assert probe.read_text() == "<unset>"
