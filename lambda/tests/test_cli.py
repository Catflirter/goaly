import copy
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import gpuctl.__main__ as cli
from fakes import FakeInfra, FakeLambda
from gpuctl.__main__ import main
from gpuctl.lambda_api import FirewallRule, Instance, InstanceType, LambdaAPIError, Ruleset
from test_spec import VALID

ENV = {"LAMBDA_API_KEY": "k"}
TYPES = {
    "gpu_1x_a6000": InstanceType("gpu_1x_a6000", 109, ("us-west-2",)),
    "gpu_1x_a10": InstanceType("gpu_1x_a10", 129, ()),
    "gpu_1x_a100": InstanceType("gpu_1x_a100", 199, ("us-west-1",)),
}
READS = ("instance_types", "list_instances", "get_instance", "list_rulesets", "get_global_rules")


def inst(name, ip="129.80.30.30", status="active", id=None, region="us-west-2", instance_type="gpu_1x_a6000"):
    return Instance(id=id or f"id-{name}", name=name, status=status, ip=ip, region=region,
                    instance_type=instance_type)


NODE = inst("k3s-gpu-a6000-1")
FOREIGN = inst("my-notebook", ip="198.51.100.7")


def spec_file(tmp_path, **changes):
    data = copy.deepcopy(VALID)
    data.update(changes)
    path = tmp_path / "gpu-nodes.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def assert_read_only(fake):
    assert fake.calls and all(c[0] in READS for c in fake.calls), fake.calls


@pytest.fixture(autouse=True)
def overrides(monkeypatch, tmp_path):
    """The autoscaler's override file for this test (absent unless the test writes it), never the checkout's."""
    path = tmp_path / ".autoscale-state.json"
    monkeypatch.setattr(cli, "OVERRIDES_PATH", path)
    return path


def in_sync(*nodes):
    """FakeInfra whose cluster, gpu-access state and k3s_peers set all match these instances (no drift)."""
    ips = [n.ip for n in nodes]
    return FakeInfra(outputs={"base": {"cp_ip": "34.82.10.10", "cp_internal_ip": "10.10.0.2", "cp_zone": "us-west1-b"},
                              "workers": {}, "gpu-access": {"gpu_ips": ips}},
                     cluster_nodes={n.name: n.ip for n in nodes}, nft_elements=["10.10.0.2", "34.82.10.10", *ips])


# --- plan ---------------------------------------------------------------------------------------

def test_plan_prints_creates_and_cost(capsys):
    rc = main(["plan"], env=ENV, client=FakeLambda(types=TYPES))
    out = capsys.readouterr().out
    assert rc == 0 and "create k3s-gpu-a6000-1" in out and "$1.99/h" in out


def test_plan_set_count_is_preview_only(tmp_path):
    # budget 5.0: two new a6000 nodes ($3.98 worst case) fit, so the preview runs to completion
    fake = FakeLambda(types=TYPES)
    rc = main(["--spec", spec_file(tmp_path, budget_usd_per_hour=5.0), "plan", "--set-count", "a6000=2"],
              env=ENV, client=fake)
    assert rc == 0 and fake.calls
    assert all(c[0] in ("instance_types", "list_instances") for c in fake.calls)


def test_plan_set_count_over_the_repo_budget_is_refused_and_still_read_only(capsys):
    fake = FakeLambda(types=TYPES)
    rc = main(["plan", "--set-count", "a6000=2"], env=ENV, client=fake)
    assert rc == 2 and "budget" in capsys.readouterr().err
    assert fake.calls and all(c[0] in ("instance_types", "list_instances") for c in fake.calls)


def test_plan_set_count_changes_the_target(tmp_path, capsys):
    # $1.09 kept + $1.99 worst case for the new node is over the repo spec's $3.00 budget
    rc = main(["--spec", spec_file(tmp_path, budget_usd_per_hour=5.0), "plan", "--set-count", "a6000=2"],
              env=ENV, client=FakeLambda(types=TYPES, instances=[NODE]))
    out = capsys.readouterr().out
    assert rc == 0 and "create k3s-gpu-a6000-2" in out and "create k3s-gpu-a6000-1" not in out


def test_plan_lists_deletes_and_keeps(capsys):
    fake = FakeLambda(types=TYPES, instances=[NODE, inst("k3s-gpu-a6000-2", ip="129.80.30.31")])
    rc = main(["plan"], env=ENV, client=fake)
    out = capsys.readouterr().out
    assert rc == 0
    assert "delete k3s-gpu-a6000-2" in out and "keep k3s-gpu-a6000-1" in out and "create" not in out


def test_plan_marks_duplicates(capsys):
    dup = inst("k3s-gpu-a6000-1", id="id-z", ip="129.80.30.31")
    rc = main(["plan"], env=ENV, client=FakeLambda(types=TYPES, instances=[NODE, dup]))
    out = capsys.readouterr().out
    assert rc == 0 and "delete k3s-gpu-a6000-1 id-z" in out and "duplicate" in out


def test_plan_with_nothing_to_change_says_so(capsys):
    rc = main(["plan"], env=ENV, client=FakeLambda(types=TYPES, instances=[NODE]))
    out = capsys.readouterr().out
    assert rc == 0 and "No instance changes" in out and "$1.09/h" in out


def test_plan_over_budget_exits_2(tmp_path, capsys):
    rc = main(["--spec", spec_file(tmp_path, budget_usd_per_hour=1.0), "plan"], env=ENV,
              client=FakeLambda(types=TYPES))
    captured = capsys.readouterr()
    assert rc == 2 and "budget" in captured.err
    assert "create k3s-gpu-a6000-1" in captured.out  # the refused plan is still shown


def test_plan_over_max_nodes_exits_2(capsys):
    rc = main(["plan", "--set-count", "a6000=4"], env=ENV, client=FakeLambda(types=TYPES))
    assert rc == 2 and "max_nodes" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["plan", "--set-count", "a6000"],
    ["plan", "--set-count", "a6000=two"],
    ["plan", "--set-count", "a6000=-1"],
    ["plan", "--set-count", "a6000=1", "--set-count", "a6000=2"],
    ["plan", "--set-count", "nosuchpool=1"],
])
def test_plan_bad_set_count_exits_2(argv, capsys):
    fake = FakeLambda(types=TYPES)
    assert main(argv, env=ENV, client=fake) == 2
    assert capsys.readouterr().err


def test_invalid_spec_exits_2(tmp_path, capsys):
    rc = main(["--spec", spec_file(tmp_path, max_nodes=0), "plan"], env=ENV, client=FakeLambda(types=TYPES))
    assert rc == 2 and "max_nodes" in capsys.readouterr().err


def test_missing_spec_file_exits_2(tmp_path, capsys):
    rc = main(["--spec", str(tmp_path / "nope.yaml"), "status"], env=ENV, client=FakeLambda(types=TYPES))
    assert rc == 2 and "nope.yaml" in capsys.readouterr().err


# --- api key, usage, client construction --------------------------------------------------------

@pytest.mark.parametrize("env", [{}, {"LAMBDA_API_KEY": ""}, {"LAMBDA_API_KEY": "  "}])
def test_missing_api_key_exits_2(env, capsys):
    fake = FakeLambda(types=TYPES)
    rc = main(["plan"], env=env, client=fake)
    captured = capsys.readouterr()
    assert rc == 2 and "LAMBDA_API_KEY" in captured.err and fake.calls == []


@pytest.mark.parametrize("argv, message", [
    ([], "gpuctl: the following arguments are required: COMMAND"),
    (["frobnicate"], "gpuctl: argument COMMAND: invalid choice: 'frobnicate'"),
    (["ip"], "gpuctl: ip: the following arguments are required: NAME"),
    (["status", "extra"], "gpuctl: unrecognized arguments: extra"),
    (["--bogus", "status"], "gpuctl: unrecognized arguments: --bogus"),
])
def test_usage_errors_exit_2(argv, message, capsys):
    assert main(argv, env=ENV, client=FakeLambda(types=TYPES)) == 2
    err = capsys.readouterr().err
    assert message in err and "gpuctl: gpuctl" not in err


def test_client_is_built_from_the_env_key(monkeypatch):
    built = []

    def factory(api_key):
        built.append(api_key)
        return FakeLambda(instances=[NODE])

    monkeypatch.setattr(cli, "LambdaClient", factory)
    assert main(["list-nodes"], env={"LAMBDA_API_KEY": "secret-key"}) == 0
    assert built == ["secret-key"]


def test_malformed_api_key_exits_2_without_echoing_it(capsys):
    rc = main(["list-nodes"], env={"LAMBDA_API_KEY": "abc def\x1b"})
    captured = capsys.readouterr()
    assert rc == 2 and "abc def" not in captured.err + captured.out and "LAMBDA_API_KEY" in captured.err


def test_api_error_exits_1(capsys):
    fake = FakeLambda(types=TYPES)

    def boom():
        raise LambdaAPIError("global/internal-error", 500, "try later")

    fake.list_instances = boom
    rc = main(["list-nodes"], env=ENV, client=fake)
    assert rc == 1 and "global/internal-error" in capsys.readouterr().err


# --- status -------------------------------------------------------------------------------------

def test_status_shows_capacity_rows(capsys):
    fake = FakeLambda(types=TYPES, instances=[NODE])
    rc = main(["status"], env=ENV, client=fake, infra=in_sync(NODE))
    lines = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rc == 0
    for t in ("gpu_1x_a6000", "gpu_1x_a10", "gpu_1x_a100"):
        for region in ("us-west-1", "us-west-2", "us-east-1"):
            row = [line for line in lines if line[:3] == ["a6000", t, region]]
            expected = "yes" if region in TYPES[t].regions_with_capacity else "no"
            assert len(row) == 1 and row[0][3] == expected
    assert_read_only(fake)


def test_status_lists_owned_instances_and_counts_them(capsys):
    fake = FakeLambda(types=TYPES, instances=[NODE, FOREIGN, inst("k3s-gpu-a6000-2", status="terminated")])
    rc = main(["status"], env=ENV, client=fake, infra=in_sync(NODE))
    out = capsys.readouterr().out
    assert rc == 0
    assert "GPU instances (k3s-gpu-*): 1" in out
    assert "k3s-gpu-a6000-1 129.80.30.30 active gpu_1x_a6000 us-west-2" in out
    assert "my-notebook" not in out and "k3s-gpu-a6000-2" not in out
    assert "1 other running instance(s)" in out


def test_status_with_no_instances_reports_zero(capsys):
    rc = main(["status"], env=ENV, client=FakeLambda(types=TYPES), infra=in_sync())
    out = capsys.readouterr().out
    assert rc == 0 and "GPU instances (k3s-gpu-*): 0" in out and "other running" not in out
    # count 1 in the spec: a note, not drift (Task 21 Step 7 after gpu-down; Task 21a review S0-1)
    assert "  note: pool a6000: 0 of 1 running (make gpu-apply starts the rest)\n  No drift\n" in out


# --- list-nodes / ip ----------------------------------------------------------------------------

def test_list_nodes_omits_foreign_instances(capsys):
    fake = FakeLambda(instances=[FOREIGN, NODE])
    rc = main(["list-nodes"], env=ENV, client=fake)
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines() == ["k3s-gpu-a6000-1 129.80.30.30 active gpu_1x_a6000 us-west-2"]
    assert "my-notebook" not in out
    assert_read_only(fake)


def test_list_nodes_shows_dash_for_missing_ip_and_skips_gone_instances(capsys):
    fake = FakeLambda(instances=[
        inst("k3s-gpu-a6000-2", ip=None, status="booting", region="us-west-1"), NODE,
        inst("k3s-gpu-a6000-3", status="terminating")])
    assert main(["list-nodes"], env=ENV, client=fake) == 0
    assert capsys.readouterr().out.splitlines() == [
        "k3s-gpu-a6000-1 129.80.30.30 active gpu_1x_a6000 us-west-2",
        "k3s-gpu-a6000-2 - booting gpu_1x_a6000 us-west-1",
    ]


def test_list_nodes_on_empty_account_prints_nothing(capsys):
    assert main(["list-nodes"], env=ENV, client=FakeLambda()) == 0
    assert capsys.readouterr().out == ""


def test_output_never_carries_control_characters(capsys):
    fake = FakeLambda(instances=[inst("k3s-gpu-a6000-1\x1b[2J", region="us-west-2\r")])
    assert main(["list-nodes"], env=ENV, client=fake) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\r" not in out and out.count("\n") == 1


def test_ip_prints_only_the_address(capsys):
    fake = FakeLambda(instances=[FOREIGN, NODE])
    assert main(["ip", "k3s-gpu-a6000-1"], env=ENV, client=fake) == 0
    assert capsys.readouterr().out == "129.80.30.30\n"
    assert_read_only(fake)


def test_ip_unknown_node_exits_1(capsys):
    fake = FakeLambda(instances=[FOREIGN, NODE, inst("k3s-gpu-a6000-2", status="terminated")])
    for name in ("k3s-gpu-a6000-9", "my-notebook", "k3s-gpu-a6000-2"):
        assert main(["ip", name], env=ENV, client=fake) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and name in captured.err


def test_ip_of_node_without_address_exits_1(capsys):
    fake = FakeLambda(instances=[inst("k3s-gpu-a6000-1", ip=None, status="booting")])
    assert main(["ip", "k3s-gpu-a6000-1"], env=ENV, client=fake) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "booting" in captured.err


BAD_IPS = ["1.2.3.4 -oProxyCommand=evil", "1.2.3.4;id", "-oX", "2001:db8::1", "1.2.3", "01.2.3.4", " 1.2.3.4"]


@pytest.mark.parametrize("bad_ip", BAD_IPS)
def test_ip_refuses_an_address_that_is_not_ipv4(bad_ip, capsys):
    fake = FakeLambda(instances=[inst("k3s-gpu-a6000-1", ip=bad_ip)])
    assert main(["ip", "k3s-gpu-a6000-1"], env=ENV, client=fake) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "invalid IP" in captured.err and "k3s-gpu-a6000-1" in captured.err


@pytest.mark.parametrize("command", [["list-nodes"], ["status"], ["plan"]])
def test_node_lines_refuse_an_address_that_is_not_ipv4(command, capsys):
    fake = FakeLambda(types=TYPES, instances=[inst("k3s-gpu-a6000-1", ip="1.2.3.4 -oX")])
    assert main(command, env=ENV, client=fake, infra=FakeInfra()) == 1
    captured = capsys.readouterr()
    assert "-oX" not in captured.out and "invalid IP" in captured.err


def test_ip_with_duplicate_names_uses_the_instance_the_planner_keeps(capsys):
    fake = FakeLambda(instances=[inst("k3s-gpu-a6000-1", id="id-b", ip="129.80.30.31"),
                                 inst("k3s-gpu-a6000-1", id="id-a", ip="129.80.30.30")])
    assert main(["ip", "k3s-gpu-a6000-1"], env=ENV, client=fake) == 0
    assert capsys.readouterr().out == "129.80.30.30\n"


# --- rules --------------------------------------------------------------------------------------

def test_rules_prints_global_and_owned_rulesets(capsys):
    fake = FakeLambda(
        global_rules=[FirewallRule("tcp", "73.15.40.40/32", (22, 22), "operator ssh")],
        rulesets=[
            Ruleset("rs-2", "my-team-rules", "us-west-1", (FirewallRule("tcp", "192.0.2.1/32", (80, 80), "web"),)),
            Ruleset("rs-1", "k3s-gpu-us-west-1", "us-west-1", (
                FirewallRule("udp", "34.1.2.3/32", (51820, 51820), "wireguard 34.1.2.3"),
                FirewallRule("icmp", "34.1.2.3/32", None, "icmp 34.1.2.3"),
                FirewallRule("tcp", "34.1.2.3/32", (30000, 30010), "range"),
            )),
        ])
    rc = main(["rules"], env=ENV, client=fake)
    out = capsys.readouterr().out
    lines = [line.split() for line in out.splitlines()]
    assert rc == 0
    assert ["global"] in lines
    assert ["tcp", "22", "73.15.40.40/32", "operator", "ssh"] in lines
    assert ["ruleset", "k3s-gpu-us-west-1", "rs-1", "us-west-1"] in lines
    assert ["udp", "51820", "34.1.2.3/32", "wireguard", "34.1.2.3"] in lines
    assert ["icmp", "-", "34.1.2.3/32", "icmp", "34.1.2.3"] in lines
    assert ["tcp", "30000-30010", "34.1.2.3/32", "range"] in lines
    assert "my-team-rules" not in out and "192.0.2.1" not in out
    assert out.index("global") < out.index("k3s-gpu-us-west-1")
    assert_read_only(fake)


def test_python_dash_m_entry_point_returns_the_exit_code():
    env = {k: v for k, v in os.environ.items() if k != "LAMBDA_API_KEY"}
    env["PYTHONPATH"] = str(Path(cli.__file__).resolve().parents[1])
    proc = subprocess.run([sys.executable, "-m", "gpuctl", "list-nodes"], env=env, capture_output=True, text=True)
    assert proc.returncode == 2 and "LAMBDA_API_KEY" in proc.stderr


def test_rules_with_nothing_configured(capsys):
    assert main(["rules"], env=ENV, client=FakeLambda()) == 0
    out = capsys.readouterr().out
    assert "global" in out and "(no rules)" in out and "ruleset" not in out


# --- overrides (Ruling R20) ---------------------------------------------------------------------

def test_plan_reads_the_override_file(tmp_path, overrides, capsys):
    overrides.write_text('{"pools": {"a6000": 2}}')
    rc = main(["--spec", spec_file(tmp_path, budget_usd_per_hour=5.0), "plan"], env=ENV,
              client=FakeLambda(types=TYPES, instances=[NODE]))
    out = capsys.readouterr().out
    assert rc == 0 and "create k3s-gpu-a6000-2" in out


def test_plan_set_count_beats_the_override_file(tmp_path, overrides, capsys):
    overrides.write_text('{"pools": {"a6000": 2}}')
    rc = main(["plan", "--set-count", "a6000=1"], env=ENV, client=FakeLambda(types=TYPES, instances=[NODE]))
    assert rc == 0 and "No instance changes" in capsys.readouterr().out


@pytest.mark.parametrize("command", [["plan"], ["apply", "--auto-approve"]])
def test_malformed_override_file_exits_2(command, overrides, capsys):
    overrides.write_text("{not json")
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    assert main(command, env=APPLY_ENV, client=fake, infra=infra) == 2
    assert str(overrides) in capsys.readouterr().err
    assert all(c[0] in READS for c in fake.calls) and infra.events == []


def _changes(out):
    """The create/delete set a plan printed: {("create", name)} and {("delete", name, id)}."""
    changes = set()
    for line in out.splitlines():
        words = line.split()
        if words and words[0] == "create":
            changes.add(("create", words[1]))
        elif words and words[0] == "delete":
            changes.add(("delete", words[1], words[2]))
    return changes


def test_plan_and_apply_agree_when_an_override_file_exists(tmp_path, overrides, capsys):  # Ruling R20
    overrides.write_text('{"pools": {"a6000": 2}}')
    spec = spec_file(tmp_path, budget_usd_per_hour=5.0)
    kept = inst("k3s-gpu-a6000-2", ip="129.80.30.32")
    old = inst("k3s-gpu-old-1", ip="129.80.30.33", id="id-old")
    assert main(["--spec", spec, "plan"], env=ENV, client=FakeLambda(types=TYPES, instances=[kept, old])) == 0
    planned = _changes(capsys.readouterr().out)
    assert planned == {("create", "k3s-gpu-a6000-1"), ("delete", "k3s-gpu-old-1", "id-old")}  # count 1 alone: none

    fake = FakeLambda(types=TYPES, instances=[kept, old])
    assert main(["--spec", spec, "apply", "--auto-approve"], env=APPLY_ENV, client=fake, infra=FakeInfra()) == 0
    done = {("create", l.name) for l in fake.launched} | {
        ("delete", i.name, i.id) for i in [kept, old] if i.id in fake.terminated}
    assert done == planned


# --- apply / down -------------------------------------------------------------------------------

APPLY_ENV = {"LAMBDA_API_KEY": "k", "LAMBDA_SSH_KEY_NAME": "operator"}


def test_cli_apply_auto_approve(capsys):
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    assert main(["apply", "--auto-approve"], env=APPLY_ENV, client=fake, infra=infra) == 0
    assert [l.ssh_key_name for l in fake.launched] == ["operator"]
    assert infra.joined == [("k3s-gpu-a6000-1", "129.80.30.30", "a6000")]
    assert "create k3s-gpu-a6000-1" in capsys.readouterr().out


@pytest.mark.parametrize("answer, rc, launched", [("y\n", 0, 1), ("yes\n", 0, 1), ("n\n", 2, 0), ("\n", 2, 0),
                                                  ("", 2, 0)])
def test_cli_apply_asks_for_confirmation(answer, rc, launched, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(answer))
    fake = FakeLambda(types=TYPES)
    assert main(["apply"], env=APPLY_ENV, client=fake, infra=FakeInfra()) == rc
    assert len(fake.launched) == launched


def test_cli_apply_keep_on_failure(capsys):
    fake, infra = FakeLambda(types=TYPES), FakeInfra(join_fail={"k3s-gpu-a6000-1"})
    assert main(["apply", "--auto-approve", "--keep-on-failure"], env=APPLY_ENV, client=fake, infra=infra) == 1
    assert fake.terminated == []


def test_cli_apply_join_failure_exits_1_and_cleans_up(capsys):
    fake, infra = FakeLambda(types=TYPES), FakeInfra(join_fail={"k3s-gpu-a6000-1"})
    assert main(["apply", "--auto-approve"], env=APPLY_ENV, client=fake, infra=infra) == 1
    assert fake.terminated == fake.launched_ids


def test_cli_apply_needs_the_ssh_key_name(capsys):
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    assert main(["apply", "--auto-approve"], env=ENV, client=fake, infra=infra) == 2
    assert "LAMBDA_SSH_KEY_NAME" in capsys.readouterr().err
    assert fake.calls == [] and infra.events == []


def test_cli_apply_over_budget_exits_2(tmp_path, capsys):
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    rc = main(["--spec", spec_file(tmp_path, budget_usd_per_hour=1.0), "apply", "--auto-approve"], env=APPLY_ENV,
              client=fake, infra=infra)
    assert rc == 2 and "budget" in capsys.readouterr().err
    assert all(c[0] in READS for c in fake.calls) and infra.events == []


def test_cli_apply_infra_error_exits_1(capsys):
    fake = FakeLambda(types=TYPES, instances=[NODE, inst("k3s-gpu-a6000-2", ip="129.80.30.31")])
    infra = FakeInfra()

    def drain_fails(name):
        raise cli.CommandError(["kubectl", "drain", name], 1, "connection refused")

    infra.drain_and_delete_node = drain_fails
    assert main(["apply", "--auto-approve"], env=APPLY_ENV, client=fake, infra=infra) == 1
    assert "connection refused" in capsys.readouterr().err
    assert fake.terminated == []  # never terminate a node that was not drained


def test_cli_down(overrides, capsys):
    overrides.write_text('{"pools": {"a6000": 2}}')
    fake, infra = FakeLambda(types=TYPES, instances=[NODE, FOREIGN]), FakeInfra()
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=infra) == 0  # no ssh key needed
    assert fake.terminated == ["id-k3s-gpu-a6000-1"] and not overrides.exists()
    assert infra.applied_gpu_ips[-1] == []


def test_cli_down_twice_exits_0(capsys):
    fake, infra = FakeLambda(types=TYPES, instances=[NODE]), FakeInfra()
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=infra) == 0
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=infra) == 0
    assert fake.terminated == ["id-k3s-gpu-a6000-1"]


def test_cli_down_has_no_keep_on_failure_flag(capsys):
    assert main(["down", "--keep-on-failure"], env=ENV, client=FakeLambda(), infra=FakeInfra()) == 2
