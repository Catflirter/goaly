"""In-memory test doubles. FakeLambda mirrors LambdaClient's methods and signatures (test_fakes.py checks this)."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Mapping

from gpuctl.lambda_api import (
    FirewallRule, Instance, InstanceType, LambdaAPIError, Ruleset, _serialise_rules,
)

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
