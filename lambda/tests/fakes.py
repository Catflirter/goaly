"""In-memory test doubles.

FakeLambda mirrors LambdaClient's methods and signatures (test_fakes.py checks this); FakeInfra mirrors Infra's
(test_infra.py checks this). FakeShell replays scripted results for the real Infra.
"""

from __future__ import annotations

import ipaddress
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from gpuctl.infra import CommandError, MissingStackError, check_join_args, normalise_gpu_ips, operator_ip
from gpuctl.lambda_api import (
    FirewallRule, Instance, InstanceType, LambdaAPIError, Ruleset, _serialise_rules,
)
from gpuctl.reconcile import GUARD_PEERS_COMMAND

# The everyone-CIDR is built, never written out: global constraints forbid the literal in any file.
ANY_IPV4 = str(ipaddress.IPv4Network((0, 0)))
assert ipaddress.ip_network(ANY_IPV4).num_addresses == 2**32  # really the whole IPv4 space (IPv4Network(0) is a /32)


def _not_found(what: str) -> LambdaAPIError:
    return LambdaAPIError("global/object-does-not-exist", 404, f"{what} not found")


class FakeLambda:
    """Lambda account state in memory.

    Every call is appended to `calls` as (method name, *positional args); launch, whose arguments are keyword-only,
    is recorded as ("launch", {kwargs}). `launch_errors[(type, region)]` is raised by launch for that combination;
    `instance_progress[id]` is the sequence get_instance returns for that id (the last entry repeats), and each
    result also replaces the instance in `instances`. Rules go through the real client's serialiser, so the fake
    refuses exactly what LambdaClient would never send.
    """

    def __init__(self, *, types: Mapping[str, InstanceType] | None = None,
                 instances: Iterable[Instance] = (), rulesets: Iterable[Ruleset] = (),
                 global_rules: Iterable[FirewallRule] = (),
                 launch_errors: Mapping[tuple[str, str], Exception] | None = None,
                 instance_progress: Mapping[str, list[Instance]] | None = None) -> None:
        self.types = dict(types or {})
        self.instances = list(instances)
        self.rulesets = list(rulesets)
        self.global_rules = list(global_rules)
        self.launch_errors = dict(launch_errors or {})
        self.instance_progress = {k: list(v) for k, v in (instance_progress or {}).items()}
        self.calls: list[tuple] = []
        self.launched_ids: list[str] = []
        self.terminated: list[str] = []
        self._next_id = 0

    def _new_id(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}-{self._next_id}"

    def _replace_instance(self, instance: Instance) -> None:
        self.instances = [instance if i.id == instance.id else i for i in self.instances]

    # --- instances ---

    def instance_types(self) -> dict[str, InstanceType]:
        self.calls.append(("instance_types",))
        return dict(self.types)

    def list_instances(self) -> list[Instance]:
        self.calls.append(("list_instances",))
        return list(self.instances)

    def get_instance(self, instance_id: str) -> Instance:
        self.calls.append(("get_instance", instance_id))
        progress = self.instance_progress.get(instance_id)
        if progress:
            instance = progress.pop(0) if len(progress) > 1 else progress[0]
            self._replace_instance(instance)
            return instance
        for instance in self.instances:
            if instance.id == instance_id:
                return instance
        raise _not_found(f"instance {instance_id}")

    def launch(self, *, name: str, region: str, instance_type: str, ssh_key_name: str, image_family: str,
               ruleset_id: str) -> str:
        self.calls.append(("launch", dict(name=name, region=region, instance_type=instance_type,
                                          ssh_key_name=ssh_key_name, image_family=image_family,
                                          ruleset_id=ruleset_id)))
        error = self.launch_errors.get((instance_type, region))
        if error is not None:
            raise error
        instance_id = self._new_id("i")
        self.instances.append(Instance(instance_id, name, "booting", None, region, instance_type))
        self.launched_ids.append(instance_id)
        return instance_id

    def terminate(self, instance_ids: list[str]) -> None:
        self.calls.append(("terminate", list(instance_ids)))
        known = {i.id: i for i in self.instances}
        missing = [i for i in instance_ids if i not in known]
        if missing:
            raise _not_found(f"instance(s) {missing}")
        for instance_id in instance_ids:
            old = known[instance_id]
            self._replace_instance(Instance(old.id, old.name, "terminated", old.ip, old.region, old.instance_type))
            self.terminated.append(instance_id)

    # --- firewall ---

    def list_rulesets(self) -> list[Ruleset]:
        self.calls.append(("list_rulesets",))
        return list(self.rulesets)

    def create_ruleset(self, name: str, region: str, rules: Iterable[FirewallRule]) -> Ruleset:
        rules = tuple(rules)
        self.calls.append(("create_ruleset", name, region, list(rules)))
        _serialise_rules(rules)
        ruleset = Ruleset(self._new_id("rs"), name, region, rules)
        self.rulesets.append(ruleset)
        return ruleset

    def update_ruleset(self, ruleset_id: str, rules: Iterable[FirewallRule]) -> None:
        rules = tuple(rules)
        self.calls.append(("update_ruleset", ruleset_id, list(rules)))
        _serialise_rules(rules)
        for index, ruleset in enumerate(self.rulesets):
            if ruleset.id == ruleset_id:
                self.rulesets[index] = Ruleset(ruleset.id, ruleset.name, ruleset.region, rules)
                return
        raise _not_found(f"firewall ruleset {ruleset_id}")

    def get_global_rules(self) -> list[FirewallRule]:
        self.calls.append(("get_global_rules",))
        return list(self.global_rules)

    def update_global_rules(self, rules: Iterable[FirewallRule]) -> None:
        rules = list(rules)
        self.calls.append(("update_global_rules", rules))
        _serialise_rules(rules)
        self.global_rules = rules


# --- shell and infra ----------------------------------------------------------------------------

def ok(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    """A scripted FakeShell result: exit code 0."""
    return subprocess.CompletedProcess([], 0, stdout, stderr)


def fail(returncode: int, stderr: str = "", stdout: str = "") -> subprocess.CompletedProcess[str]:
    """A scripted FakeShell result: a non-zero exit code."""
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@dataclass(frozen=True)
class ShellCall:
    args: list[str]
    input: str | None
    check: bool
    timeout: float | None
    env: dict[str, str] | None


class FakeShell:
    """Replays `results` in order, one per run(), and records every call in `calls`.

    Like SubprocessShell, a non-zero exit code raises CommandError when check=True. A result that is an exception
    is raised instead. Running out of results fails the test: every command a test causes must be scripted.
    """

    def __init__(self, results: Iterable[subprocess.CompletedProcess[str] | Exception]) -> None:
        self.results = list(results)
        self.calls: list[ShellCall] = []

    def run(self, args: Sequence[str], *, input: str | None = None, check: bool = True,
            timeout: float | None = None, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        args = list(args)
        self.calls.append(ShellCall(args, input, check, timeout, None if env is None else dict(env)))
        assert self.results, f"unexpected command (no scripted result left): {args}"
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        result = subprocess.CompletedProcess(args, result.returncode, result.stdout, result.stderr)
        if check and result.returncode != 0:
            raise CommandError(args, result.returncode, result.stderr)
        return result


DEFAULT_OUTPUTS = {
    "base": {"cp_ip": "34.82.10.10", "cp_internal_ip": "10.10.0.2", "cp_zone": "us-west1-b", "cp_name": "k3s-cp"},
    "workers": {"worker_ips": {"k3s-cpu-1": "34.29.20.20"}},
}


class FakeInfra:
    """Terraform outputs, cp SSH, kubectl and join-node in memory.

    `outputs[stack]` is what tf_output returns; a stack missing from it raises CommandError (it cannot be read).
    ensure_init raises MissingStackError for a stack in `missing` (not in the checkout, like Infra) and
    CommandError for a stack in `init_fail`. `events` records the side effects in order:
    "tf:<gpu ips>" for apply_gpu_access, "nft" for the cp peer-set push, "ssh:<command>" for any other cp_ssh,
    "join:<name>:<type>". Arguments are validated and normalised exactly like Infra (shared helpers in gpuctl.infra):
    apply_gpu_access records normalise_gpu_ips(), join_node runs check_join_args(), detect_operator_ip validates
    env["OPERATOR_IP"] (there is no network fallback).
    """

    def __init__(self, *, outputs: Mapping[str, Mapping] | None = None, env: Mapping[str, str] | None = None,
                 cluster_nodes: Mapping[str, str | None] | None = None, nft_elements: Iterable[str] = (),
                 join_fail: Iterable[str] = (), missing: Iterable[str] = (), init_fail: Iterable[str] = ()) -> None:
        source = DEFAULT_OUTPUTS if outputs is None else outputs
        self.outputs = {stack: dict(values) for stack, values in source.items()}
        self.env = dict({"OPERATOR_IP": "73.15.40.40"} if env is None else env)
        self.cluster_nodes = dict(cluster_nodes or {})
        self.nft_elements = list(nft_elements)
        self.join_fail = set(join_fail)
        self.missing = set(missing)
        self.init_fail = set(init_fail)
        self.events: list[str] = []
        self.initialised: list[str] = []
        self.ssh_calls: list[tuple[str, str | None]] = []
        self.kubectl_calls: list[tuple[str, ...]] = []
        self.applied_gpu_ips: list[list[str]] = []
        self.deleted_nodes: list[str] = []
        self.joined: list[tuple[str, str, str]] = []
        self.forgotten: list[str] = []

    def ensure_init(self, stack: str) -> None:
        if stack in self.missing:
            raise MissingStackError(["terraform", f"-chdir=infra/{stack}", "init"], 1,
                                    f"infra/{stack} does not exist in this checkout")
        if stack in self.init_fail:
            raise CommandError(["terraform", f"-chdir=infra/{stack}", "init"], -1, "timed out after 900.0s")
        self.initialised.append(stack)

    def tf_output(self, stack: str) -> dict:
        if stack not in self.outputs:
            raise CommandError(["terraform", f"-chdir=infra/{stack}", "output", "-json"], 1,
                               "Backend initialization required")
        return dict(self.outputs[stack])

    def apply_gpu_access(self, gpu_ips: Sequence[str]) -> None:
        ips = normalise_gpu_ips(gpu_ips)  # ValueError on a null or non-public IP, like Infra (Review Focus #2)
        self.applied_gpu_ips.append(ips)
        self.events.append(f"tf:{ips}")

    def cp_ssh(self, command: str, input: str | None = None) -> str:
        self.ssh_calls.append((command, input))
        self.events.append("nft" if command == GUARD_PEERS_COMMAND else f"ssh:{command}")
        return ""

    def kubectl(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        self.kubectl_calls.append(args)
        return subprocess.CompletedProcess(["kubectl", *args], 0, "", "")

    def cluster_gpu_nodes(self) -> dict[str, str | None]:
        return dict(self.cluster_nodes)

    def drain_and_delete_node(self, name: str) -> None:
        self.deleted_nodes.append(name)
        self.cluster_nodes.pop(name, None)

    def join_node(self, name: str, ip: str, gpu_type: str) -> None:
        ip = check_join_args(name, ip, gpu_type)
        self.joined.append((name, ip, gpu_type))
        self.events.append(f"join:{name}:{gpu_type}")
        if name in self.join_fail:
            raise CommandError(["join-node.sh", "--node", name], 1, "join failed")
        self.cluster_nodes[name] = ip

    def forget_host_key(self, name: str) -> None:
        self.forgotten.append(name)

    def detect_operator_ip(self) -> str:
        return operator_ip(self.env.get("OPERATOR_IP", ""))

    def cp_guard_elements(self) -> list[str]:
        return list(self.nft_elements)
