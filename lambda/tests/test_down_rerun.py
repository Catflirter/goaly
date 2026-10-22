"""gpuctl down when make down stopped between its stack destroys (minors batch 2b, ledger finding D-3).

make down runs gpu-down, then destroys gpu-access, workers and base. Interrupted after the workers destroy, its
re-run starts with gpu-down again while infra/base still exists: the workers stack then answers `terraform output
-json` with {} (measured with terraform 1.17 on an emptied state), which sync-peers takes as no CPU peers, and only
a real terraform error fails it (Ruling R29). With infra/base destroyed and infra/workers not, gpu-down takes R54's
cluster-gone path and never reads the workers stack. These tests drive the real Infra over a scripted shell (the
terraform, gcloud and kubectl answers), so the commands and the operator's messages are the ones make down sees.
"""
import json

import pytest

from fakes import FakeLambda, FakeShell, fail, ok
from gpuctl.__main__ import main
from gpuctl.infra import Infra
from gpuctl.peers import global_rules, nft_replace_script
from test_reconcile import TYPES

PROJECT = "k3s-demo-123456"
BASE = {"cp_ip": "34.82.10.10", "cp_internal_ip": "10.10.0.2", "cp_zone": "us-west1-b"}
LOAD_ERROR = ("Error: Failed to load state: Failed to open state file at "
              "gs://k3s-demo-123456-k3s-tfstate/workers/default.tfstate: googleapi: Error 503: Backend Error")


def tf_json(outputs):
    return json.dumps({name: {"value": value} for name, value in outputs.items()})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    import gpuctl.__main__ as cli
    monkeypatch.setattr(cli, "OVERRIDES_PATH", tmp_path / "lambda/.autoscale-state.json")
    for stack in ("base", "workers", "gpu-access"):
        (tmp_path / "infra" / stack).mkdir(parents=True)
    kubeconfig = tmp_path / "hybrid-gpu.yaml"
    kubeconfig.write_text("apiVersion: v1\n")
    return tmp_path, {"GCP_PROJECT": PROJECT, "OPERATOR_IP": "73.15.40.40", "KUBECONFIG": str(kubeconfig),
                      "LAMBDA_API_KEY": "k"}


def gpu_down(repo, results):
    root, env = repo
    sh = FakeShell(results)
    fake = FakeLambda(types=TYPES)
    rc = main(["down", "--auto-approve"], env=env, client=fake, infra=Infra(sh, root, env))
    return rc, sh, fake


def commands(sh):
    """Each call as "terraform <stack> <subcommand>" or the tool name."""
    return [f"terraform {c.args[1].rsplit('/', 1)[1]} {c.args[2]}" if c.args[0] == "terraform" else c.args[0]
            for c in sh.calls]


def test_rerun_after_the_workers_destroy_treats_the_empty_workers_stack_as_no_cpu_peers(repo, capsys):
    rc, sh, fake = gpu_down(repo, [
        ok(), ok(),                              # gpu-access: init, apply with no GPU IP
        ok(), ok(tf_json(BASE)),                 # base: init, output (still there)
        ok(), ok("{}"),                          # workers: init, output of the destroyed stack
        ok(),                                    # k3s-cp's peer set over IAP
    ])
    assert rc == 0
    assert commands(sh) == ["terraform gpu-access init", "terraform gpu-access apply", "terraform base init",
                            "terraform base output", "terraform workers init", "terraform workers output", "gcloud"]
    # k3s-cp's peer set keeps only k3s-cp itself: no CPU worker (and no GPU node) is left in it
    assert sh.calls[-1].input == nft_replace_script(["10.10.0.2", "34.82.10.10"])
    assert fake.global_rules == global_rules("73.15.40.40")
    assert "0 CPU worker(s)" in capsys.readouterr().out


def test_rerun_still_fails_loudly_on_a_real_terraform_error_reading_the_workers_stack(repo, capsys):
    """Not "no outputs": an error that terraform reports fails gpu-down before any peer-list write (R29)."""
    rc, sh, fake = gpu_down(repo, [
        ok(), ok(),
        ok(), ok(tf_json(BASE)),
        ok(), fail(1, LOAD_ERROR),
    ])
    assert rc == 1
    assert "gcloud" not in commands(sh)  # k3s-cp's peer set is not rewritten without the workers
    assert not [c for c in fake.calls if c[0] in ("update_global_rules", "update_ruleset", "create_ruleset")]
    captured = capsys.readouterr()
    assert "cannot read infra/workers outputs" in captured.err and "Error 503: Backend Error" in captured.err
    assert "Traceback" not in captured.err + captured.out


def test_rerun_with_base_destroyed_and_workers_left_never_reads_the_workers_stack(repo, capsys):
    """The order make down never leaves but an operator can (base destroyed by hand first): R54's cluster gone."""
    rc, sh, fake = gpu_down(repo, [
        ok(), ok(),                              # gpu-access: an empty apply needs no base output
        ok(), ok("{}"),                          # base: destroyed
    ])
    assert rc == 0
    assert "terraform workers output" not in commands(sh) and "gcloud" not in commands(sh)
    assert fake.global_rules == global_rules("73.15.40.40")
    assert "k3s-cp k3s_peers: skipped" in capsys.readouterr().out
