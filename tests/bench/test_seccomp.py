"""Tests for the sandbox's seccomp profile (V28): Docker's default, with Unix sockets alone.

tests/bench/data/docker-29.7.1-seccomp-default.json is Docker 29.7.1's default profile, as
vendored in moby (vendor/github.com/moby/profiles/seccomp/default.json, Apache-2.0).

seccomp compares a syscall's 64-bit argument, while the kernel reads socket()'s family as a
32-bit int. So an allow rule on socket may only be an exact match of AF_UNIX on the low 32 bits:
any range, such as "greater than 40", would let a family with high bits set through.
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from goaly.bench.sandbox import SECCOMP_PROFILE

DEFAULT = Path(__file__).with_name("data") / "docker-29.7.1-seccomp-default.json"

AF_UNIX, AF_INET, AF_INET6, AF_NETLINK, AF_PACKET = 1, 2, 10, 16, 17
LOW_32_BITS = 0xFFFFFFFF
UNIX_ONLY = {"index": 0, "value": LOW_32_BITS, "valueTwo": AF_UNIX, "op": "SCMP_CMP_MASKED_EQ"}

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
    unix = {"names": ["socket"], "action": "SCMP_ACT_ALLOW", "args": [UNIX_ONLY]}
    first = next(i for i, rule in enumerate(default["syscalls"]) if rule["names"] == ["socket"])
    profile["syscalls"] = [*rules[:first], unix, *rules[first:]]
    return profile


def matches(condition: dict[str, Any], argument: int) -> bool:
    """Whether a 64-bit ``argument`` meets a seccomp condition, as libseccomp compares it."""
    value: int = condition["value"]
    operation: str = condition["op"]
    expected: int = condition.get("valueTwo", 0)
    return {
        "SCMP_CMP_EQ": lambda: argument == value,
        "SCMP_CMP_NE": lambda: argument != value,
        "SCMP_CMP_LT": lambda: argument < value,
        "SCMP_CMP_LE": lambda: argument <= value,
        "SCMP_CMP_GT": lambda: argument > value,
        "SCMP_CMP_GE": lambda: argument >= value,
        "SCMP_CMP_MASKED_EQ": lambda: argument & value == expected,
    }[operation]()


def allows_socket(profile: Profile, family: int) -> bool:
    """Whether the profile lets socket() through with ``family`` as its 64-bit argument."""
    for rule in profile["syscalls"]:
        conditions = rule.get("args") or []
        allowed = "socket" in rule["names"] and rule["action"] == "SCMP_ACT_ALLOW"
        if allowed and all(matches(condition, family) for condition in conditions):
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


def test_every_rule_that_allows_socket_is_an_exact_match_of_af_unix() -> None:
    rules = [
        rule
        for rule in load(SECCOMP_PROFILE)["syscalls"]
        if "socket" in rule["names"] and rule["action"] == "SCMP_ACT_ALLOW"
    ]

    assert rules == [{"names": ["socket"], "action": "SCMP_ACT_ALLOW", "args": [UNIX_ONLY]}]


@pytest.mark.parametrize("high", [0, 1 << 32, 1 << 63, (1 << 64) - (1 << 32)])
@pytest.mark.parametrize("family", [AF_INET, AF_INET6, AF_NETLINK, AF_PACKET, 0, 3, 38, 40, 41])
def test_no_family_but_unix_gets_through_whatever_its_high_bits(high: int, family: int) -> None:
    assert not allows_socket(load(SECCOMP_PROFILE), high | family)


def test_a_range_rule_like_dockers_lets_a_high_bit_family_through() -> None:
    # Why the profile allows no range: Docker's own "greater than 40" passes a 64-bit AF_INET.
    assert allows_socket(load(DEFAULT), (1 << 32) | AF_INET)


@pytest.mark.parametrize("high", [0, 1 << 32])
def test_unix_sockets_stay_open(high: int) -> None:
    assert allows_socket(load(SECCOMP_PROFILE), high | AF_UNIX)


def test_io_uring_and_socketcall_are_refused() -> None:
    profile = load(SECCOMP_PROFILE)

    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    assert not {"io_uring_setup", "io_uring_enter", "io_uring_register"} & allowed_names(profile)
    assert "socketcall" not in allowed_names(profile)
    assert "socketcall" in allowed_names(load(DEFAULT))
