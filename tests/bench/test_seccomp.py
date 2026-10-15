"""Tests for the sandbox's seccomp profile (V28): Docker's default, with no IP sockets.

tests/bench/data/docker-29.7.1-seccomp-default.json is Docker 29.7.1's default profile, as
vendored in moby (vendor/github.com/moby/profiles/seccomp/default.json, Apache-2.0).
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from goaly.bench.sandbox import SECCOMP_PROFILE

DEFAULT = Path(__file__).with_name("data") / "docker-29.7.1-seccomp-default.json"

AF_INET, AF_INET6, AF_NETLINK, AF_PACKET = 2, 10, 16, 17
AF_UNIX, AF_ALG, AF_VSOCK = 1, 38, 40

Profile = dict[str, Any]


def load(path: Path) -> Profile:
    loaded: Profile = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def derive(default: Profile) -> Profile:
    """Docker's default profile with the V28 changes, and nothing else."""
    profile = copy.deepcopy(default)
    rules = []
    for rule in profile["syscalls"]:
        if rule["names"] == ["socket"]:
            continue
        if "socketcall" in rule["names"]:  # Its arguments are out of seccomp's sight.
            rule["names"] = [name for name in rule["names"] if name != "socketcall"]
        rules.append(rule)
    blocked = {AF_INET, AF_INET6, AF_NETLINK, AF_PACKET, AF_ALG, AF_VSOCK}
    socket_rules = [
        {
            "names": ["socket"],
            "action": "SCMP_ACT_ALLOW",
            "args": [{"index": 0, "value": family, "op": "SCMP_CMP_EQ"}],
        }
        for family in range(AF_VSOCK + 1)
        if family not in blocked
    ]
    socket_rules.append(
        {
            "names": ["socket"],
            "action": "SCMP_ACT_ALLOW",
            "args": [{"index": 0, "value": AF_VSOCK, "op": "SCMP_CMP_GT"}],
        }
    )
    first = next(i for i, rule in enumerate(default["syscalls"]) if rule["names"] == ["socket"])
    profile["syscalls"] = rules[:first] + socket_rules + rules[first:]
    return profile


def allows_socket(profile: Profile, family: int) -> bool:
    """Whether the profile lets socket() open ``family``, as libseccomp would decide."""
    operations = {
        "SCMP_CMP_EQ": lambda value: family == value,
        "SCMP_CMP_LT": lambda value: family < value,
        "SCMP_CMP_GT": lambda value: family > value,
    }
    for rule in profile["syscalls"]:
        if "socket" in rule["names"] and rule["action"] == "SCMP_ACT_ALLOW":
            conditions = rule.get("args") or []
            if all(operations[c["op"]](c["value"]) for c in conditions):
                return True
    return bool(profile["defaultAction"] == "SCMP_ACT_ALLOW")


def allowed_names(profile: Profile) -> set[str]:
    return {
        name
        for rule in profile["syscalls"]
        if rule["action"] == "SCMP_ACT_ALLOW"
        for name in rule["names"]
    }


def test_the_profile_is_dockers_default_with_only_the_v28_changes() -> None:
    assert load(SECCOMP_PROFILE) == derive(load(DEFAULT))


@pytest.mark.parametrize("family", [AF_INET, AF_INET6, AF_NETLINK, AF_PACKET, AF_ALG, AF_VSOCK])
def test_no_ip_netlink_or_packet_socket_can_be_opened(family: int) -> None:
    assert allows_socket(load(DEFAULT), family) is (family not in {AF_ALG, AF_VSOCK})
    assert not allows_socket(load(SECCOMP_PROFILE), family)


def test_every_other_family_stays_as_docker_allows_it() -> None:
    default, profile = load(DEFAULT), load(SECCOMP_PROFILE)
    changed = {
        family
        for family in range(64)
        if allows_socket(default, family) != allows_socket(profile, family)
    }

    assert changed == {AF_INET, AF_INET6, AF_NETLINK, AF_PACKET}
    assert allows_socket(profile, AF_UNIX)


def test_io_uring_and_socketcall_are_refused() -> None:
    profile = load(SECCOMP_PROFILE)

    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    assert not {"io_uring_setup", "io_uring_enter", "io_uring_register"} & allowed_names(profile)
    assert "socketcall" not in allowed_names(profile)
    assert "socketcall" in allowed_names(load(DEFAULT))
