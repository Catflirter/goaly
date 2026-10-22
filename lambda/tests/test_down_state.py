"""gpu-down's state (live finding L21-1, progress.md 2026-10-21): after `make gpu-down`, `make gpu-status` reported
"pool a6000 has 0, wants 1" and exited 3, although the brief (Task 21 Step 7, Task 24) expects 0 instances and no
drift. gpu-down now records that every pool is down (lambda/.gpu-down); status's drift check then wants 0 per pool,
so an instance still running after gpu-down is drift (it bills), while an empty account is not. gpu-apply clears the
record and the spec's counts apply again; gpu-plan previews what gpu-apply does, so it plans the spec's counts too.
"""
import io
from pathlib import Path

import pytest

import gpuctl.__main__ as cli
from fakes import FakeInfra, FakeLambda
from gpuctl.__main__ import main
from gpuctl.reconcile import Reconciler, Settings, read_down_marker
from test_cli import APPLY_ENV, ENV, NODE, TYPES, in_sync, inst
from test_reconcile import SPEC, make_rec


@pytest.fixture
def marker(tmp_path, monkeypatch):
    path = tmp_path / ".gpu-down"
    monkeypatch.setattr(cli, "DOWN_MARKER_PATH", path)
    monkeypatch.setattr(cli, "OVERRIDES_PATH", tmp_path / ".autoscale-state.json")
    return path


def rec(fake, infra, path, **kw):
    return make_rec(fake, infra, down_marker_path=path, **kw)


# --- the Reconciler records and clears it -------------------------------------------------------

def test_down_records_that_every_pool_is_down(tmp_path):
    path = tmp_path / ".gpu-down"
    fake = FakeLambda(types=TYPES, instances=[inst("k3s-gpu-a6000-1")])
    assert rec(fake, FakeInfra(), path).down(SPEC, auto_approve=True) == 0
    assert path.is_file() and read_down_marker(path)


def test_a_second_down_with_nothing_to_delete_still_records_it(tmp_path):
    path = tmp_path / ".gpu-down"
    assert rec(FakeLambda(types=TYPES), FakeInfra(), path).down(SPEC, auto_approve=True) == 0
    assert path.is_file()


def test_down_records_it_before_the_first_delete(tmp_path):
    """A down that fails half way still counts as down: status then reports what is left as drift."""
    path = tmp_path / ".gpu-down"
    fake, infra = FakeLambda(types=TYPES, instances=[inst("k3s-gpu-a6000-1")]), FakeInfra()
    seen = []
    drain = infra.drain_and_delete_node

    def drain_and_look(name):
        seen.append(path.is_file())
        drain(name)

    infra.drain_and_delete_node = drain_and_look
    assert rec(fake, infra, path).down(SPEC, auto_approve=True) == 0
    assert seen == [True]


def test_declined_down_records_nothing(tmp_path):
    path = tmp_path / ".gpu-down"
    fake = FakeLambda(types=TYPES, instances=[inst("k3s-gpu-a6000-1")])
    assert rec(fake, FakeInfra(), path, confirm=lambda prompt: False).down(SPEC, auto_approve=False) == 2
    assert not path.exists()


def test_apply_clears_it(tmp_path):
    path = tmp_path / ".gpu-down"
    path.write_text("2026-10-21T22:33:00-07:00\n")
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    assert rec(fake, infra, path).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    assert not path.exists() and len(fake.launched) == 1


def test_declined_apply_keeps_it(tmp_path):
    path = tmp_path / ".gpu-down"
    path.write_text("2026-10-21T22:33:00-07:00\n")
    fake = FakeLambda(types=TYPES)
    assert rec(fake, FakeInfra(), path, confirm=lambda prompt: False).apply(
        SPEC, auto_approve=False, keep_on_failure=False) == 2
    assert path.exists() and fake.launched == []


def test_an_unwritable_record_never_stops_gpu_down(tmp_path, capsys):
    """Billing comes first: gpu-down terminates even when it cannot write the record (a warning says so)."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    fake = FakeLambda(types=TYPES, instances=[inst("k3s-gpu-a6000-1")])
    assert rec(fake, FakeInfra(), blocker / ".gpu-down").down(SPEC, auto_approve=True) == 0
    assert fake.terminated == ["id-k3s-gpu-a6000-1"]
    assert "gpu-down" in capsys.readouterr().err


def test_without_a_record_path_nothing_is_written(tmp_path):
    fake = FakeLambda(types=TYPES, instances=[inst("k3s-gpu-a6000-1")])
    Reconciler(fake, FakeInfra(), Settings(spec_path=Path("x"), overrides_path=tmp_path / "o.json", ssh_key_name="k"),
               out=io.StringIO()).down(SPEC, auto_approve=True)
    assert list(tmp_path.iterdir()) == []


def test_drift_wants_zero_while_down(tmp_path):
    path = tmp_path / ".gpu-down"
    path.write_text("2026-10-21T22:33:00-07:00\n")
    assert [line for line in rec(FakeLambda(types=TYPES), in_sync(), path).drift(SPEC, []) if "pool" in line] == []
    node = inst("k3s-gpu-a6000-1")
    assert "pool a6000 has 1, wants 0" in rec(FakeLambda(types=TYPES), in_sync(node), path).drift(SPEC, [node])


@pytest.mark.parametrize("text, expected", [("2026-10-21T22:33:00-07:00\n", "2026-10-21T22:33:00-07:00"),
                                            ("", "(time not recorded)"), ("\x1b[31mx\n", "?[31mx")])
def test_read_down_marker(tmp_path, text, expected):
    path = tmp_path / ".gpu-down"
    assert read_down_marker(path) is None
    path.write_text(text)
    assert read_down_marker(path) == expected


# --- the commands: gpu-down, then gpu-status, gpu-plan, gpu-apply ------------------------------

def test_status_after_down_reports_zero_instances_and_no_drift(marker, capsys):  # brief Task 21 Step 7
    fake = FakeLambda(types=TYPES, instances=[NODE])
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=FakeInfra()) == 0
    capsys.readouterr()
    assert main(["status"], env=ENV, client=fake, infra=in_sync()) == 0
    out = capsys.readouterr().out
    assert "GPU instances (k3s-gpu-*): 0" in out
    assert "down since" in out and "make gpu-apply" in out
    assert "No drift" in out and "wants 1" not in out


def test_status_after_down_reports_an_instance_still_running(marker, capsys):
    marker.write_text("2026-10-21T22:33:00-07:00\n")
    assert main(["status"], env=ENV, client=FakeLambda(types=TYPES, instances=[NODE]), infra=in_sync(NODE)) == 3
    assert "pool a6000 has 1, wants 0" in capsys.readouterr().out


def test_status_after_down_still_reports_a_stale_firewall_rule(marker, capsys):
    marker.write_text("2026-10-21T22:33:00-07:00\n")
    infra = in_sync()
    infra.outputs["gpu-access"] = {"gpu_ips": ["129.80.30.30"]}  # gpu-down's terraform apply did not happen
    assert main(["status"], env=ENV, client=FakeLambda(types=TYPES), infra=infra) == 3
    assert "SECURITY: firewall rule for 129.80.30.30 has no live instance" in capsys.readouterr().out


def test_status_without_down_still_wants_the_spec_count(marker, capsys):
    assert main(["status"], env=ENV, client=FakeLambda(types=TYPES), infra=in_sync()) == 3
    assert "pool a6000 has 0, wants 1" in capsys.readouterr().out


def test_apply_after_down_brings_the_spec_back(marker, capsys):
    fake = FakeLambda(types=TYPES, instances=[NODE])
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=FakeInfra()) == 0
    assert marker.exists()
    assert main(["apply", "--auto-approve"], env=APPLY_ENV, client=fake, infra=FakeInfra()) == 0
    assert not marker.exists() and len(fake.launched) == 1


def test_plan_while_down_shows_what_apply_would_do_and_says_so(marker, capsys):
    marker.write_text("2026-10-21T22:33:00-07:00\n")
    assert main(["plan"], env=ENV, client=FakeLambda(types=TYPES)) == 0
    out = capsys.readouterr().out
    assert "create k3s-gpu-a6000-1" in out  # gpu-apply clears the record and launches the spec's count
    assert "2026-10-21T22:33:00-07:00" in out and "make gpu-down" in out
