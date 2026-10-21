"""Load and validate the GPU node spec (lambda/gpu-nodes.yaml, design spec section 5.4)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

POOL_NAME_RE = re.compile(r"[a-z0-9]{1,20}")
INSTANCE_TYPE_RE = re.compile(r"gpu_[0-9]+x_[a-z0-9_]+")
# Region names are interpolated into resource names (ruleset k3s-gpu-<region>).
REGION_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
REGION_MAX_LEN = 32
# image_family goes to the Lambda API and into operator output; keep it to one plain token.
IMAGE_FAMILY_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,62}")
FORBIDDEN_REGIONS = frozenset({"us-south-1"})
ARM_MARKER = "gh200"

_SPEC_KEYS = frozenset({"budget_usd_per_hour", "max_nodes", "pools"})
_POOL_KEYS = frozenset({"count", "instance_type", "fallback_types", "regions", "image_family"})
_POOL_OPTIONAL_KEYS = frozenset({"autoscale"})
_AUTOSCALE_KEYS = frozenset({"min", "max", "idle_minutes"})


class SpecError(ValueError):
    """The GPU node spec is malformed or violates a policy rule."""


@dataclass(frozen=True)
class Autoscale:
    min: int
    max: int
    idle_minutes: int


@dataclass(frozen=True)
class Pool:
    name: str
    count: int
    instance_type: str
    fallback_types: tuple[str, ...]
    regions: tuple[str, ...]
    image_family: str
    autoscale: Autoscale | None

    def candidate_types(self) -> tuple[str, ...]:
        """Instance types to try, in preference order."""
        return (self.instance_type, *self.fallback_types)


@dataclass(frozen=True)
class Spec:
    budget_usd_per_hour: float
    max_nodes: int
    pools: dict[str, Pool]

    def regions(self) -> tuple[str, ...]:
        """Union of all pools' regions, in first-seen order."""
        return tuple(dict.fromkeys(r for pool in self.pools.values() for r in pool.regions))


def _check_keys(data: Any, where: str, required: frozenset[str], optional: frozenset[str] = frozenset()) -> Mapping:
    if not isinstance(data, Mapping):
        raise SpecError(f"{where}: must be a mapping, got {type(data).__name__}")
    unknown = sorted(map(repr, set(data) - required - optional))
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {', '.join(unknown)}")
    missing = sorted(required - set(data))
    if missing:
        raise SpecError(f"{where}: missing key(s) {', '.join(missing)}")
    return data


def _int(value: Any, where: str, minimum: int) -> int:
    # bool is an int subclass; `count: true` is a typo, not 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{where}: must be an integer, got {value!r}")
    if value < minimum:
        raise SpecError(f"{where}: must be >= {minimum}, got {value}")
    return value


def _str_list(value: Any, where: str) -> tuple[str, ...]:
    # A bare string would otherwise be iterated character by character.
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise SpecError(f"{where}: must be a list of strings, got {value!r}")
    return tuple(value)


def _instance_type(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise SpecError(f"{where}: must be a string, got {value!r}")
    # Checked before the pattern so any name containing gh200 reports arm64, not a pattern error.
    if ARM_MARKER in value.lower():
        raise SpecError(f"{where}: {value!r} is an arm64 (GH200) instance type; only x86_64 is supported")
    if not INSTANCE_TYPE_RE.fullmatch(value):
        raise SpecError(f"{where}: {value!r} does not match {INSTANCE_TYPE_RE.pattern}")
    return value


def _autoscale(value: Any, where: str) -> Autoscale | None:
    if value is None:
        return None
    data = _check_keys(value, where, _AUTOSCALE_KEYS)
    lo = _int(data["min"], f"{where}.min", 0)
    hi = _int(data["max"], f"{where}.max", 0)
    if lo > hi:
        raise SpecError(f"{where}: min ({lo}) must be <= max ({hi})")
    return Autoscale(min=lo, max=hi, idle_minutes=_int(data["idle_minutes"], f"{where}.idle_minutes", 1))


def _pool(name: Any, value: Any) -> Pool:
    if not isinstance(name, str) or not POOL_NAME_RE.fullmatch(name):
        raise SpecError(f"invalid pool name {name!r}: must match {POOL_NAME_RE.pattern}")
    where = f"pools.{name}"
    data = _check_keys(value, where, _POOL_KEYS, _POOL_OPTIONAL_KEYS)

    instance_type = _instance_type(data["instance_type"], f"{where}.instance_type")
    fallback_types = tuple(
        _instance_type(t, f"{where}.fallback_types")
        for t in _str_list(data["fallback_types"], f"{where}.fallback_types")
    )

    regions = _str_list(data["regions"], f"{where}.regions")
    if not regions:
        raise SpecError(f"{where}.regions: must not be empty")
    for region in regions:
        if len(region) > REGION_MAX_LEN or not REGION_RE.fullmatch(region):
            raise SpecError(
                f"{where}.regions: invalid region {region!r}: must match {REGION_RE.pattern}, "
                f"at most {REGION_MAX_LEN} characters"
            )
        if region in FORBIDDEN_REGIONS:
            raise SpecError(f"{where}.regions: region {region!r} is not allowed")

    image_family = data["image_family"]
    if not isinstance(image_family, str) or not IMAGE_FAMILY_RE.fullmatch(image_family):
        raise SpecError(f"{where}.image_family: must match {IMAGE_FAMILY_RE.pattern}, got {image_family!r}")

    return Pool(
        name=name,
        count=_int(data["count"], f"{where}.count", 0),
        instance_type=instance_type,
        fallback_types=fallback_types,
        regions=regions,
        image_family=image_family,
        autoscale=_autoscale(data.get("autoscale"), f"{where}.autoscale"),
    )


def parse_spec(data: Mapping) -> Spec:
    """Validate a decoded spec document and build a Spec. Raises SpecError."""
    data = _check_keys(data, "spec", _SPEC_KEYS)

    budget = data["budget_usd_per_hour"]
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise SpecError(f"budget_usd_per_hour: must be a finite number > 0, got {budget!r}")
    max_nodes = _int(data["max_nodes"], "max_nodes", 1)

    pools_data = data["pools"]
    if not isinstance(pools_data, Mapping):
        raise SpecError(f"pools: must be a mapping, got {type(pools_data).__name__}")
    pools = {name: _pool(name, value) for name, value in pools_data.items()}

    total = sum(pool.count for pool in pools.values())
    if total > max_nodes:
        raise SpecError(f"pools: total count {total} exceeds max_nodes {max_nodes}")

    # Autoscale bounds that max_nodes can never honour are config errors; catch them at load time.
    # `count` stays independent of autoscale.min: gpu-down sets every target to 0.
    for pool in pools.values():
        if pool.autoscale is not None and pool.autoscale.max > max_nodes:
            raise SpecError(
                f"pools.{pool.name}.autoscale.max: {pool.autoscale.max} exceeds max_nodes {max_nodes}"
            )
    min_total = sum(pool.autoscale.min for pool in pools.values() if pool.autoscale is not None)
    if min_total > max_nodes:
        raise SpecError(f"pools: total autoscale.min {min_total} exceeds max_nodes {max_nodes}")

    return Spec(budget_usd_per_hour=float(budget), max_nodes=max_nodes, pools=pools)


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects a mapping key repeated in the same mapping.

    Plain safe_load keeps the last value, so a copied pool block or a doubled `count`
    would silently replace the first one.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue  # `<<:` merges are resolved (and may be overridden) by the base class
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:
                continue  # unhashable key: the base class raises its own error
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r}", key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _load_yaml(text: str) -> Any:
    loader = _UniqueKeyLoader(text)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def load_spec(path: str | Path) -> Spec:
    """Read a YAML spec file and validate it. Raises SpecError (or OSError if unreadable)."""
    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SpecError(f"{path}: not valid UTF-8: {exc}") from exc
    try:
        data = _load_yaml(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"{path}: invalid YAML: {exc}") from exc
    try:
        return parse_spec(data)
    except SpecError as exc:
        raise SpecError(f"{path}: {exc}") from exc
