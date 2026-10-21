"""Lambda Cloud API client (design spec section 5.4).

Every call goes through one injectable transport and one rate limiter: at least
MIN_REQUEST_INTERVAL_S between any two requests and MIN_LAUNCH_INTERVAL_S between
two launches. Field names follow the design research; Task 7 checks them against
the live API.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

BASE_URL = "https://cloud.lambda.ai/api/v1"
TIMEOUT_S = 30
MIN_REQUEST_INTERVAL_S = 1.0
MIN_LAUNCH_INTERVAL_S = 12.0
USER_AGENT = "gpuctl"
PROTOCOLS = frozenset({"tcp", "udp", "icmp", "all"})
_API_KEY_RE = re.compile(r"[A-Za-z0-9._-]+")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

Transport = Callable[[str, str, dict | None, dict[str, str]], tuple[int, dict]]


@dataclass(frozen=True)
class Instance:
    id: str
    name: str
    status: str
    ip: str | None
    region: str
    instance_type: str


@dataclass(frozen=True)
class InstanceType:
    name: str
    price_cents_per_hour: int
    regions_with_capacity: tuple[str, ...]


@dataclass(frozen=True)
class FirewallRule:
    protocol: str
    source_network: str
    port_range: tuple[int, int] | None = None
    description: str = ""


@dataclass(frozen=True)
class Ruleset:
    id: str
    name: str
    region: str
    rules: tuple[FirewallRule, ...]


class LambdaAPIError(Exception):
    """The Lambda API (or the path to it) returned an error. Never carries the API key."""

    def __init__(self, code: str, status: int, message: str = "") -> None:
        super().__init__(f"{code} (HTTP {status}): {message}" if message else f"{code} (HTTP {status})")
        self.code = code
        self.status = status


class InsufficientCapacity(LambdaAPIError):
    """No capacity for the requested (instance type, region); try the next combination."""


def _read_json(raw: bytes) -> dict:
    try:
        payload = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a 3xx: urllib would re-send the Authorization header to any Location, even plain http."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises HTTPError(3xx), which the transport returns as a status


_OPENER = urllib.request.build_opener(_RefuseRedirects)


def urllib_transport(method: str, url: str, json_body: dict | None, headers: dict[str, str]) -> tuple[int, dict]:
    """Default transport: one HTTPS request, 30 s timeout, JSON in and out, redirects refused."""
    data = None
    headers = dict(headers)
    auth = headers.pop("Authorization", None)
    if json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        if auth is not None:
            req.add_unredirected_header("Authorization", auth)
        try:
            with _OPENER.open(req, timeout=TIMEOUT_S) as resp:
                return resp.status, _read_json(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, _read_json(e.read())
    except (urllib.error.URLError, OSError) as e:
        # The key travels only in a header, which neither the URL nor the reason contains.
        raise LambdaAPIError("gpuctl/network-error", 0, str(getattr(e, "reason", e))) from None
    except http.client.HTTPException as e:
        # IncompleteRead, BadStatusLine, LineTooLong, InvalidURL: their text can quote raw bytes, so name the type.
        raise LambdaAPIError("gpuctl/network-error", 0, type(e).__name__) from None
    except ValueError as e:
        # e.g. http.client refusing a header value: its message quotes the value, which may be the key.
        raise LambdaAPIError("gpuctl/transport-error", 0, type(e).__name__) from None


def _str(value: Any) -> str:
    """A field that must be a string; anything else is an unexpected shape (TypeError, turned into one by _parse)."""
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    return value


def _optional_str(value: Any) -> str:
    """An optional display field (instance name, rule description): absent or null reads as "", other types fail."""
    return "" if value is None else _str(value)


def _name(value: Any) -> str:
    """Lambda returns some references as {"name": ...} objects and some as plain strings."""
    return _str(value["name"] if isinstance(value, dict) else value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_instance(raw: dict) -> Instance:
    ip = raw.get("ip")
    return Instance(id=_str(raw["id"]), name=_optional_str(raw.get("name")), status=_str(raw["status"]),
                    ip=None if ip is None else _str(ip),
                    region=_name(raw["region"]), instance_type=_name(raw["instance_type"]))


def _parse_rule(raw: dict) -> FirewallRule:
    port_range = raw.get("port_range")
    if port_range is not None:
        if not (isinstance(port_range, list) and len(port_range) == 2 and all(_is_int(p) for p in port_range)):
            raise TypeError(f"port_range must be a list of two ints, got {port_range!r}")
        port_range = (port_range[0], port_range[1])
    return FirewallRule(protocol=_str(raw["protocol"]), source_network=_str(raw["source_network"]),
                        port_range=port_range, description=_optional_str(raw.get("description")))


def _parse_ruleset(raw: dict) -> Ruleset:
    return Ruleset(id=_str(raw["id"]), name=_str(raw["name"]), region=_name(raw["region"]),
                   rules=tuple(_parse_rule(r) for r in raw["rules"]))


def _serialise_rules(rules: Iterable[FirewallRule]) -> list[dict]:
    """The last check before any rule reaches Lambda: every source is a single IPv4 /32 (global constraints)."""
    out = []
    for rule in rules:
        try:
            network = ipaddress.ip_network(rule.source_network, strict=False)
        except ValueError:
            raise ValueError(f"source_network {rule.source_network!r} is not a CIDR") from None
        if network.version != 4 or network.prefixlen != 32:
            raise ValueError(f"source_network {rule.source_network!r} is not a single IPv4 address (/32)")
        if rule.protocol not in PROTOCOLS:
            raise ValueError(f"protocol {rule.protocol!r} is not one of {sorted(PROTOCOLS)}")
        item: dict[str, Any] = {"protocol": rule.protocol}
        if rule.port_range is not None:
            pr = rule.port_range
            if not (len(pr) == 2 and all(_is_int(p) for p in pr) and 1 <= pr[0] <= pr[1] <= 65535):
                raise ValueError(f"port_range {pr!r} must be (lo, hi) with 1 <= lo <= hi <= 65535")
            item["port_range"] = [pr[0], pr[1]]
        item["source_network"] = str(network)
        item["description"] = rule.description
        out.append(item)
    return out


def _segment(value: str) -> str:
    return urllib.parse.quote(value, safe="")


class LambdaClient:
    def __init__(self, api_key: str, *, transport: Transport | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 base_url: str = BASE_URL) -> None:
        # Surrounding whitespace (a trailing CR from a CRLF .env, a pasted newline) is never part of a key, so it
        # is stripped. Any other character outside the charset is refused with a message that never quotes the
        # key: http.client would otherwise reject the header with a ValueError that prints it in full.
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("Lambda API key is empty")
        if not _API_KEY_RE.fullmatch(api_key):
            raise ValueError("Lambda API key contains invalid characters (allowed: letters, digits, '.', '_', '-')")
        parts = urllib.parse.urlsplit(base_url)
        if not (parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS)):
            raise ValueError("base_url must be https:// (plain http only for a loopback host)")
        self._api_key = api_key
        self._transport = transport or urllib_transport
        self._clock = clock
        self._sleep = sleep
        self._base_url = base_url.rstrip("/")
        self._last_request: float | None = None
        self._last_launch: float | None = None

    def __repr__(self) -> str:
        return f"LambdaClient(base_url={self._base_url!r})"

    # --- plumbing ---

    def _wait_turn(self, is_launch: bool) -> None:
        # Read the clock, sleep once, then read it once more and record the later of the scheduled and the
        # actual send time: a real sleep oversleeps by a varying amount, and timing the next wait from the
        # scheduled time could then space two real sends under the minimum. Never loop: with an injected no-op
        # sleep (the tests' idiom) re-reading until the time is reached would busy-spin on the real clock.
        now = self._clock()
        ready = []
        if self._last_request is not None:
            ready.append(self._last_request + MIN_REQUEST_INTERVAL_S)
        if is_launch and self._last_launch is not None:
            ready.append(self._last_launch + MIN_LAUNCH_INTERVAL_S)
        wait = max(ready, default=now) - now
        if wait > 0:
            self._sleep(wait)
        sent = max(now + max(wait, 0.0), self._clock())
        self._last_request = sent
        if is_launch:
            self._last_launch = sent

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***")

    def _request(self, method: str, path: str, body: dict | None = None, *, is_launch: bool = False) -> Any:
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json",
                   "User-Agent": USER_AGENT}
        self._wait_turn(is_launch)
        status, payload = self._transport(method, self._base_url + path, body, headers)
        if not 200 <= status < 300:
            error = payload.get("error") if isinstance(payload, dict) else None
            error = error if isinstance(error, dict) else {}
            code = self._redact(str(error.get("code") or f"http/{status}"))
            message = self._redact(str(error.get("message") or ""))
            cls = InsufficientCapacity if code.endswith("insufficient-capacity") else LambdaAPIError
            raise cls(code, status, message)
        if not isinstance(payload, dict) or "data" not in payload:
            raise LambdaAPIError("gpuctl/unexpected-response", status, f"{method} {path}: no 'data' in response")
        return payload["data"]

    def _parse(self, what: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise LambdaAPIError("gpuctl/unexpected-response", 200, f"{what}: unexpected shape ({e!r})") from None

    # --- instances ---

    def instance_types(self) -> dict[str, InstanceType]:
        data = self._request("GET", "/instance-types")
        return self._parse("instance-types", lambda: {
            key: InstanceType(name=entry["instance_type"]["name"],
                              price_cents_per_hour=int(entry["instance_type"]["price_cents_per_hour"]),
                              regions_with_capacity=tuple(_name(r) for r in entry["regions_with_capacity_available"]))
            for key, entry in data.items()})

    def list_instances(self) -> list[Instance]:
        data = self._request("GET", "/instances")
        return self._parse("instances", lambda: [_parse_instance(raw) for raw in data])

    def get_instance(self, instance_id: str) -> Instance:
        data = self._request("GET", f"/instances/{_segment(instance_id)}")
        return self._parse("instance", lambda: _parse_instance(data))

    def launch(self, *, name: str, region: str, instance_type: str, ssh_key_name: str, image_family: str,
               ruleset_id: str) -> str:
        """Launch one instance and return its id.

        InsufficientCapacity means nothing was launched. Any other error (a timeout or lost response after
        Lambda accepted the request, a 5xx, an unexpected body) leaves the outcome unknown: before retrying,
        callers must call list_instances() and look for `name`, or they may start a second billable instance.
        """
        body = {"region_name": region, "instance_type_name": instance_type, "ssh_key_names": [ssh_key_name],
                "name": name, "hostname": name, "image": {"family": image_family},
                "firewall_rulesets": [{"id": ruleset_id}]}
        data = self._request("POST", "/instance-operations/launch", body, is_launch=True)
        ids = self._parse("launch", lambda: data["instance_ids"])
        if not isinstance(ids, list) or not ids or not isinstance(ids[0], str) or not ids[0]:
            raise LambdaAPIError("gpuctl/unexpected-response", 200, "launch returned no instance id")
        return ids[0]

    def terminate(self, instance_ids: list[str]) -> None:
        self._request("POST", "/instance-operations/terminate", {"instance_ids": list(instance_ids)})

    # --- firewall ---

    def list_rulesets(self) -> list[Ruleset]:
        data = self._request("GET", "/firewall-rulesets")
        return self._parse("firewall-rulesets", lambda: [_parse_ruleset(raw) for raw in data])

    def create_ruleset(self, name: str, region: str, rules: Iterable[FirewallRule]) -> Ruleset:
        body = {"name": name, "region": region, "rules": _serialise_rules(rules)}
        data = self._request("POST", "/firewall-rulesets", body)
        return self._parse("firewall-ruleset", lambda: _parse_ruleset(data))

    def update_ruleset(self, ruleset_id: str, rules: Iterable[FirewallRule]) -> None:
        self._request("PATCH", f"/firewall-rulesets/{_segment(ruleset_id)}", {"rules": _serialise_rules(rules)})

    def get_global_rules(self) -> list[FirewallRule]:
        """The account-wide rules that apply to every instance (read side of update_global_rules)."""
        data = self._request("GET", "/firewall-rulesets/global")
        return self._parse("global firewall rules", lambda: [_parse_rule(raw) for raw in data["rules"]])

    def update_global_rules(self, rules: Iterable[FirewallRule]) -> None:
        self._request("PATCH", "/firewall-rulesets/global", {"rules": _serialise_rules(rules)})
