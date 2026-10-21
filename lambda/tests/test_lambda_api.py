import http.client
import http.server
import io
import ipaddress
import threading
import time
import urllib.error
import urllib.request

import pytest

from gpuctl import lambda_api
from gpuctl.lambda_api import (
    BASE_URL, FirewallRule, Instance, InstanceType, InsufficientCapacity, LambdaAPIError, LambdaClient, Ruleset,
)


class FakeTransport:
    """Answers by (method, path) from canned responses and records every call."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, method, url, body, headers):
        self.calls.append((method, url, body, headers))
        assert url.startswith(BASE_URL + "/")
        return self.responses[(method, url[len(BASE_URL):])]


INSTANCE_JSON = {"id": "i-1", "name": "k3s-gpu-a6000-1", "status": "booting", "ip": None,
                 "region": {"name": "us-west-2"}, "instance_type": {"name": "gpu_1x_a6000"}}
LAUNCH_ARGS = dict(name="k3s-gpu-a6000-1", region="us-west-1", instance_type="gpu_1x_a6000",
                   ssh_key_name="op-key", image_family="lambda-stack-22-04", ruleset_id="rs-1")
LAUNCH_OK = (200, {"data": {"instance_ids": ["i-1"]}})


def make_client(responses, api_key="k"):
    t = FakeTransport(responses)
    return LambdaClient(api_key, transport=t, sleep=lambda s: None), t


def test_instance_types_parse_price_and_capacity():
    t = FakeTransport({("GET", "/instance-types"): (200, {"data": {"gpu_1x_a6000": {
        "instance_type": {"name": "gpu_1x_a6000", "price_cents_per_hour": 109},
        "regions_with_capacity_available": [{"name": "us-west-1"}]}}})})
    types = LambdaClient("k", transport=t, sleep=lambda s: None).instance_types()
    assert types["gpu_1x_a6000"] == InstanceType("gpu_1x_a6000", 109, ("us-west-1",))


def test_list_instances_tolerates_missing_ip():
    client, _ = make_client({("GET", "/instances"): (200, {"data": [INSTANCE_JSON]})})
    assert client.list_instances() == [Instance("i-1", "k3s-gpu-a6000-1", "booting", None, "us-west-2", "gpu_1x_a6000")]


def test_list_instances_tolerates_absent_ip_key():
    raw = {k: v for k, v in INSTANCE_JSON.items() if k != "ip"}
    client, _ = make_client({("GET", "/instances"): (200, {"data": [raw]})})
    assert client.list_instances()[0].ip is None


def test_get_instance_path_and_parse():
    active = {**INSTANCE_JSON, "status": "active", "ip": "198.51.100.7"}
    client, t = make_client({("GET", "/instances/i-1"): (200, {"data": active})})
    assert client.get_instance("i-1") == Instance("i-1", "k3s-gpu-a6000-1", "active", "198.51.100.7",
                                                  "us-west-2", "gpu_1x_a6000")
    method, url, body, _ = t.calls[-1]
    assert (method, url, body) == ("GET", BASE_URL + "/instances/i-1", None)


def test_ids_are_escaped_in_paths():
    client, t = make_client({("GET", "/instances/..%2Fx"): (200, {"data": INSTANCE_JSON})})
    client.get_instance("../x")
    assert t.calls[-1][1] == BASE_URL + "/instances/..%2Fx"


def test_launch_sends_exact_body_and_returns_id():
    client, t = make_client({("POST", "/instance-operations/launch"): LAUNCH_OK})
    instance_id = client.launch(**LAUNCH_ARGS)
    method, url, body, headers = t.calls[-1]
    assert (method, url.endswith("/instance-operations/launch"), instance_id) == ("POST", True, "i-1")
    assert body == {"region_name": "us-west-1", "instance_type_name": "gpu_1x_a6000",
                    "ssh_key_names": ["op-key"], "name": "k3s-gpu-a6000-1", "hostname": "k3s-gpu-a6000-1",
                    "image": {"family": "lambda-stack-22-04"}, "firewall_rulesets": [{"id": "rs-1"}]}
    assert headers["Authorization"] == "Bearer k"


def test_launch_without_instance_id_is_an_error():
    client, _ = make_client({("POST", "/instance-operations/launch"): (200, {"data": {"instance_ids": []}})})
    with pytest.raises(LambdaAPIError, match="instance id"):
        client.launch(**LAUNCH_ARGS)


def test_insufficient_capacity_is_typed():
    client, _ = make_client({("POST", "/instance-operations/launch"): (400, {"error": {
        "code": "instance-operations/launch/insufficient-capacity", "message": "none"}})})
    with pytest.raises(InsufficientCapacity) as e:
        client.launch(**LAUNCH_ARGS)
    assert (e.value.code, e.value.status) == ("instance-operations/launch/insufficient-capacity", 400)


def test_other_errors_do_not_leak_key():
    client, _ = make_client({("GET", "/instances"): (401, {"error": {"code": "global/invalid-api-key",
                                                                     "message": "bad"}})}, api_key="sekrit-123")
    with pytest.raises(LambdaAPIError) as e:
        client.list_instances()
    assert not isinstance(e.value, InsufficientCapacity) and "sekrit-123" not in str(e.value)
    assert (e.value.code, e.value.status) == ("global/invalid-api-key", 401)


def test_error_message_echoing_the_key_is_redacted():
    client, _ = make_client({("GET", "/instances"): (401, {"error": {
        "code": "global/invalid-api-key", "message": "key sekrit-123 is invalid"}})}, api_key="sekrit-123")
    with pytest.raises(LambdaAPIError) as e:
        client.list_instances()
    assert "sekrit-123" not in str(e.value) and "sekrit-123" not in repr(e.value)


def test_error_without_error_object_still_raises():
    client, _ = make_client({("GET", "/instances"): (502, {})})
    with pytest.raises(LambdaAPIError) as e:
        client.list_instances()
    assert e.value.status == 502


def test_success_without_data_is_an_error():
    client, _ = make_client({("GET", "/instances"): (200, {"unexpected": []})})
    with pytest.raises(LambdaAPIError, match="data"):
        client.list_instances()


def test_unexpected_response_shape_raises_api_error():
    client, _ = make_client({("GET", "/instances"): (200, {"data": [{"id": "i-1"}]})})
    with pytest.raises(LambdaAPIError, match="unexpected"):
        client.list_instances()


@pytest.mark.parametrize("data", [
    {"instance_ids": "i-123"},  # a string would make launch return "i"
    {"instance_ids": [123]},
    {"instance_ids": [""]},
    {"instance_ids": None},
])
def test_launch_rejects_malformed_instance_ids(data):
    client, _ = make_client({("POST", "/instance-operations/launch"): (200, {"data": data})})
    with pytest.raises(LambdaAPIError) as e:
        client.launch(**LAUNCH_ARGS)
    assert e.value.code == "gpuctl/unexpected-response"


@pytest.mark.parametrize("field, value", [("region", None), ("instance_type", None), ("region", {"name": None}),
                                          ("region", 7), ("id", None), ("ip", 7)])
def test_instance_with_wrongly_typed_field_is_unexpected(field, value):
    client, _ = make_client({("GET", "/instances"): (200, {"data": [{**INSTANCE_JSON, field: value}]})})
    with pytest.raises(LambdaAPIError) as e:
        client.list_instances()
    assert e.value.code == "gpuctl/unexpected-response"


@pytest.mark.parametrize("name", [None, "absent"])
def test_unnamed_foreign_instance_does_not_break_listing(name):
    # Lambda names are optional (e.g. dashboard launches); one foreign unnamed instance must not block gpu-down.
    foreign = {**INSTANCE_JSON, "id": "i-2"}
    if name == "absent":
        del foreign["name"]
    else:
        foreign["name"] = name
    client, _ = make_client({("GET", "/instances"): (200, {"data": [INSTANCE_JSON, foreign]})})
    assert client.list_instances() == [
        Instance("i-1", "k3s-gpu-a6000-1", "booting", None, "us-west-2", "gpu_1x_a6000"),
        Instance("i-2", "", "booting", None, "us-west-2", "gpu_1x_a6000")]


@pytest.mark.parametrize("value", [7, ["x"]])
def test_instance_name_of_wrong_type_is_unexpected(value):
    client, _ = make_client({("GET", "/instances"): (200, {"data": [{**INSTANCE_JSON, "name": value}]})})
    with pytest.raises(LambdaAPIError) as e:
        client.list_instances()
    assert e.value.code == "gpuctl/unexpected-response"


def test_rule_with_null_description_parses():
    rule = {"protocol": "tcp", "port_range": [22, 22], "source_network": "1.2.3.4/32", "description": None}
    client, _ = make_client({("GET", "/firewall-rulesets"): (200, {"data": [
        {"id": "rs-1", "name": "k3s-gpu-us-west-1", "region": "us-west-1", "rules": [rule]}]})})
    assert client.list_rulesets() == [
        Ruleset("rs-1", "k3s-gpu-us-west-1", "us-west-1", (FirewallRule("tcp", "1.2.3.4/32", (22, 22), ""),))]


@pytest.mark.parametrize("port_range", ["22", [22], [22, 22, 22], ["22", "22"], [True, False]])
def test_rule_with_malformed_port_range_is_unexpected(port_range):
    rule = {"protocol": "tcp", "port_range": port_range, "source_network": "1.2.3.4/32"}
    client, _ = make_client({("GET", "/firewall-rulesets"): (200, {"data": [
        {"id": "rs-1", "name": "k3s-gpu-us-west-1", "region": "us-west-1", "rules": [rule]}]})})
    with pytest.raises(LambdaAPIError) as e:
        client.list_rulesets()
    assert e.value.code == "gpuctl/unexpected-response"


@pytest.mark.parametrize("key", ["", "  ", "\r\n"])
def test_empty_api_key_is_rejected(key):
    with pytest.raises(ValueError, match="empty"):
        LambdaClient(key, transport=FakeTransport({}))


@pytest.mark.parametrize("key", ["secret_abc123.XYZ\r", "secret_abc123.XYZ\n", "secret_abc123.XYZ ",
                                 "\tsecret_abc123.XYZ\r\n"])
def test_surrounding_whitespace_is_stripped_from_key(key):
    client, t = make_client({("GET", "/instances"): (200, {"data": []})}, api_key=key)
    client.list_instances()
    assert t.calls[-1][3]["Authorization"] == "Bearer secret_abc123.XYZ"


@pytest.mark.parametrize("key", ["secret_abc\r123", "secret abc123", "secret_abc123\x00", "sekrit:123", "ключ"])
def test_key_with_invalid_characters_is_refused_without_echo(key):
    with pytest.raises(ValueError, match="invalid characters") as e:
        LambdaClient(key, transport=FakeTransport({}))
    assert "abc" not in str(e.value) and "sekrit" not in str(e.value) and "ключ" not in str(e.value)


@pytest.mark.parametrize("base_url", ["http://cloud.lambda.ai/api/v1", "ftp://cloud.lambda.ai/api/v1",
                                      "cloud.lambda.ai/api/v1", "http://127.0.0.1.evil.example/api"])
def test_base_url_must_be_https(base_url):
    with pytest.raises(ValueError, match="https"):
        LambdaClient("k", transport=FakeTransport({}), base_url=base_url)


@pytest.mark.parametrize("base_url", ["https://example.test/api/v1", "http://127.0.0.1:8080/api/v1",
                                      "http://localhost:8080", "http://[::1]:8080"])
def test_https_or_loopback_http_base_url_is_accepted(base_url):
    LambdaClient("k", transport=FakeTransport({}), base_url=base_url)


def test_rules_serialisation():
    client, t = make_client({("PATCH", "/firewall-rulesets/rs-1"): (200, {"data": {}})})
    client.update_ruleset("rs-1", [FirewallRule("tcp", "1.2.3.4/32", (22, 22), "ssh"),
                                   FirewallRule("icmp", "1.2.3.4/32", None, "icmp")])
    method, url, body, _ = t.calls[-1]
    assert method == "PATCH" and url.endswith("/firewall-rulesets/rs-1")
    assert body == {"rules": [
        {"protocol": "tcp", "port_range": [22, 22], "source_network": "1.2.3.4/32", "description": "ssh"},
        {"protocol": "icmp", "source_network": "1.2.3.4/32", "description": "icmp"}]}


# The everyone-CIDR is built, never written out: global constraints forbid the literal in any file.
ANY_IPV4 = str(ipaddress.IPv4Network((0, 0)))
assert ipaddress.ip_network(ANY_IPV4).num_addresses == 2**32  # really the whole IPv4 space (IPv4Network(0) is a /32)


@pytest.mark.parametrize("source", [ANY_IPV4, "::/0", "not-a-cidr", "0.0.0.0/1", "128.0.0.0/1", "1.2.3.4/8",
                                    "1.2.3.4/31", "2001:db8::1/128"])
def test_open_or_invalid_sources_are_never_sent(source):
    client, t = make_client({("PATCH", "/firewall-rulesets/global"): (200, {"data": {}})})
    with pytest.raises(ValueError, match="source_network"):
        client.update_global_rules([FirewallRule("tcp", source, (6443, 6443))])
    assert t.calls == []


@pytest.mark.parametrize("rule, match", [
    (FirewallRule("sctp", "1.2.3.4/32", (1, 1)), "protocol"),
    (FirewallRule("tcp", "1.2.3.4/32", (22, 21)), "port_range"),
    (FirewallRule("tcp", "1.2.3.4/32", (0, 22)), "port_range"),
    (FirewallRule("tcp", "1.2.3.4/32", (1, 65536)), "port_range"),
    (FirewallRule("tcp", "1.2.3.4/32", (22,)), "port_range"),
    (FirewallRule("tcp", "1.2.3.4/32", ("22", "22")), "port_range"),
])
def test_invalid_protocol_or_port_range_is_never_sent(rule, match):
    client, t = make_client({("PATCH", "/firewall-rulesets/global"): (200, {"data": {}})})
    with pytest.raises(ValueError, match=match):
        client.update_global_rules([rule])
    assert t.calls == []


def test_bare_ip_source_is_sent_as_slash_32():
    client, t = make_client({("PATCH", "/firewall-rulesets/global"): (200, {"data": {}})})
    client.update_global_rules([FirewallRule("all", "1.2.3.4", None, "op")])
    assert t.calls[-1][2] == {"rules": [{"protocol": "all", "source_network": "1.2.3.4/32", "description": "op"}]}


def test_global_rules_path():
    client, t = make_client({("PATCH", "/firewall-rulesets/global"): (200, {"data": {}})})
    client.update_global_rules([FirewallRule("udp", "1.2.3.4/32", (51820, 51820), "wg")])
    method, url, body, _ = t.calls[-1]
    assert (method, url) == ("PATCH", BASE_URL + "/firewall-rulesets/global")
    assert body == {"rules": [{"protocol": "udp", "port_range": [51820, 51820], "source_network": "1.2.3.4/32",
                               "description": "wg"}]}


def test_get_global_rules_path_and_parse():
    client, t = make_client({("GET", "/firewall-rulesets/global"): (200, {"data": {
        "id": "global", "name": "Global Firewall Rules", "rules": [
            {"protocol": "tcp", "port_range": [22, 22], "source_network": "73.15.40.40/32",
             "description": "operator ssh"},
            {"protocol": "icmp", "source_network": "73.15.40.40/32", "description": None}]}})})
    assert client.get_global_rules() == [FirewallRule("tcp", "73.15.40.40/32", (22, 22), "operator ssh"),
                                          FirewallRule("icmp", "73.15.40.40/32", None, "")]
    method, url, body, _ = t.calls[-1]
    assert (method, url, body) == ("GET", BASE_URL + "/firewall-rulesets/global", None)


def test_get_global_rules_with_unexpected_shape_is_an_api_error():
    client, _ = make_client({("GET", "/firewall-rulesets/global"): (200, {"data": {"rules": "nope"}})})
    with pytest.raises(LambdaAPIError, match="unexpected"):
        client.get_global_rules()


def test_terminate_body():
    client, t = make_client({("POST", "/instance-operations/terminate"): (200, {"data": {
        "terminated_instances": [INSTANCE_JSON]}})})
    assert client.terminate(["i-1"]) is None
    method, url, body, _ = t.calls[-1]
    assert (method, url.endswith("/instance-operations/terminate")) == ("POST", True)
    assert body == {"instance_ids": ["i-1"]}


def test_list_rulesets_accepts_region_string_or_object():
    rules = [{"protocol": "tcp", "port_range": [6443, 6443], "source_network": "1.2.3.4/32", "description": "api"},
             {"protocol": "icmp", "source_network": "1.2.3.4/32"}]
    client, _ = make_client({("GET", "/firewall-rulesets"): (200, {"data": [
        {"id": "rs-1", "name": "k3s-gpu-us-west-1", "region": "us-west-1", "rules": rules},
        {"id": "rs-2", "name": "k3s-gpu-us-east-1", "region": {"name": "us-east-1", "description": "Virginia"},
         "rules": []}]})})
    assert client.list_rulesets() == [
        Ruleset("rs-1", "k3s-gpu-us-west-1", "us-west-1", (FirewallRule("tcp", "1.2.3.4/32", (6443, 6443), "api"),
                                                          FirewallRule("icmp", "1.2.3.4/32", None, ""))),
        Ruleset("rs-2", "k3s-gpu-us-east-1", "us-east-1", ())]


def test_create_ruleset_body_and_result():
    rule = FirewallRule("tcp", "1.2.3.4/32", (10250, 10250), "kubelet")
    client, t = make_client({("POST", "/firewall-rulesets"): (200, {"data": {
        "id": "rs-9", "name": "k3s-gpu-us-west-1", "region": {"name": "us-west-1"},
        "rules": [{"protocol": "tcp", "port_range": [10250, 10250], "source_network": "1.2.3.4/32",
                   "description": "kubelet"}]}})})
    result = client.create_ruleset("k3s-gpu-us-west-1", "us-west-1", [rule])
    assert result == Ruleset("rs-9", "k3s-gpu-us-west-1", "us-west-1", (rule,))
    method, _, body, _ = t.calls[-1]
    assert method == "POST"
    assert body == {"name": "k3s-gpu-us-west-1", "region": "us-west-1", "rules": [
        {"protocol": "tcp", "port_range": [10250, 10250], "source_network": "1.2.3.4/32", "description": "kubelet"}]}


class FakeClock:
    """clock() reads the time; sleep(s) records s and advances the time by s."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limits():
    fc = FakeClock()
    t = FakeTransport({("GET", "/instances"): (200, {"data": []}),
                       ("POST", "/instance-operations/launch"): LAUNCH_OK})
    sent_at = []

    def transport(*args):
        sent_at.append(fc.now)
        return t(*args)

    client = LambdaClient("k", transport=transport, clock=fc.clock, sleep=fc.sleep)

    client.list_instances()
    client.list_instances()  # immediately after: must wait >= 1.0 s
    assert sent_at[1] - sent_at[0] >= 1.0
    assert any(s >= 1.0 for s in fc.sleeps)

    fc.now += 5.0  # time passes on its own: no extra wait needed
    before = len(fc.sleeps)
    client.list_instances()
    assert fc.sleeps[before:] == []

    client.launch(**LAUNCH_ARGS)
    client.launch(**LAUNCH_ARGS)  # second launch: >= 12.0 s after the first launch
    assert sent_at[-1] - sent_at[-2] >= 12.0
    assert fc.sleeps[-1] >= 11.0  # waited on the launch window, not just the 1 s window


def test_no_op_sleep_with_frozen_clock_never_spins():
    sleeps = []
    t = FakeTransport({("GET", "/instances"): (200, {"data": []}),
                       ("POST", "/instance-operations/launch"): LAUNCH_OK})
    client = LambdaClient("k", transport=t, clock=lambda: 0.0, sleep=sleeps.append)
    start = time.monotonic()
    client.list_instances()
    client.list_instances()
    client.launch(**LAUNCH_ARGS)
    client.launch(**LAUNCH_ARGS)
    assert time.monotonic() - start < 0.5
    # Scheduled sends: 0, 1, 2 (1 s window), 14 (12 s after the first launch at 2).
    assert sleeps == [1.0, 2.0, 14.0]


def test_brief_idiom_real_clock_and_no_op_sleep_returns_immediately():
    t = FakeTransport({("GET", "/instances"): (200, {"data": []}),
                       ("POST", "/instance-operations/launch"): LAUNCH_OK})
    client = LambdaClient("k", transport=t, sleep=lambda s: None)
    wall, cpu = time.monotonic(), time.process_time()
    client.list_instances()
    client.list_instances()
    client.launch(**LAUNCH_ARGS)
    client.launch(**LAUNCH_ARGS)
    assert time.monotonic() - wall < 0.5 and time.process_time() - cpu < 0.5
    assert len(t.calls) == 4


class OversleepingClock(FakeClock):
    """A real sleep returns late by a varying amount; a later, shorter overshoot must not shrink the next gap."""

    def __init__(self, overshoots):
        super().__init__()
        self.overshoots = iter(overshoots)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds + next(self.overshoots, 0.0)


def test_rate_limits_hold_when_sleep_oversleeps():
    fc = OversleepingClock([0.05, 0.0, 0.3, 0.0, 0.2, 0.0, 0.4, 0.0])
    sent, launches = [], []

    def transport(method, url, body, headers):
        sent.append(fc.now)
        if url.endswith("/launch"):
            launches.append(fc.now)
            return LAUNCH_OK
        return (200, {"data": []})

    client = LambdaClient("k", transport=transport, clock=fc.clock, sleep=fc.sleep)
    for _ in range(3):
        client.list_instances()
    for _ in range(3):
        client.launch(**LAUNCH_ARGS)
        client.list_instances()
    assert all(b - a >= 1.0 for a, b in zip(sent, sent[1:])), sent
    assert all(b - a >= 12.0 for a, b in zip(launches, launches[1:])), launches


def test_failed_launch_still_counts_toward_launch_window():
    fc = FakeClock()
    sent_at = []
    responses = iter([(400, {"error": {"code": "instance-operations/launch/insufficient-capacity", "message": ""}}),
                      LAUNCH_OK])

    def transport(method, url, body, headers):
        sent_at.append(fc.now)
        return next(responses)

    client = LambdaClient("k", transport=transport, clock=fc.clock, sleep=fc.sleep)
    with pytest.raises(InsufficientCapacity):
        client.launch(**LAUNCH_ARGS)
    client.launch(**LAUNCH_ARGS)
    assert sent_at[1] - sent_at[0] >= 12.0


def test_failed_request_still_counts_toward_request_window():
    fc = FakeClock()
    sent_at = []

    def transport(method, url, body, headers):
        sent_at.append(fc.now)
        raise LambdaAPIError("gpuctl/network-error", 0, "down")

    client = LambdaClient("k", transport=transport, clock=fc.clock, sleep=fc.sleep)
    for _ in range(2):
        with pytest.raises(LambdaAPIError):
            client.list_instances()
    assert sent_at[1] - sent_at[0] >= 1.0


# --- urllib_transport: exercised with a patched urlopen, never the network ---

class FakeResponse(io.BytesIO):
    def __init__(self, status, payload):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_urllib_transport_builds_request(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen.update(method=req.get_method(), url=req.full_url, data=req.data, timeout=timeout,
                    auth=req.get_header("Authorization"), ctype=req.get_header("Content-type"))
        return FakeResponse(200, b'{"data": {"ok": true}}')

    monkeypatch.setattr(lambda_api._OPENER, "open", fake_urlopen)
    status, payload = lambda_api.urllib_transport("POST", BASE_URL + "/x", {"a": 1}, {"Authorization": "Bearer k"})
    assert (status, payload) == (200, {"data": {"ok": True}})
    assert seen == {"method": "POST", "url": BASE_URL + "/x", "data": b'{"a": 1}', "timeout": 30,
                    "auth": "Bearer k", "ctype": "application/json"}


def test_urllib_transport_returns_http_errors_as_status(monkeypatch):
    def fake_urlopen(req, timeout):
        body = io.BytesIO(b'{"error": {"code": "global/not-found", "message": "nope"}}')
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, body)

    monkeypatch.setattr(lambda_api._OPENER, "open", fake_urlopen)
    assert lambda_api.urllib_transport("GET", BASE_URL + "/x", None, {}) == (
        404, {"error": {"code": "global/not-found", "message": "nope"}})


def test_urllib_transport_tolerates_non_json_error_body(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", {}, io.BytesIO(b"<html>busy</html>"))

    monkeypatch.setattr(lambda_api._OPENER, "open", fake_urlopen)
    assert lambda_api.urllib_transport("GET", BASE_URL + "/x", None, {}) == (503, {})


def test_urllib_transport_wraps_network_errors(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(lambda_api._OPENER, "open", fake_urlopen)
    with pytest.raises(LambdaAPIError) as e:
        lambda_api.urllib_transport("GET", BASE_URL + "/x", None, {"Authorization": "Bearer sekrit-123"})
    assert e.value.status == 0 and "sekrit-123" not in str(e.value)


def test_default_transport_is_urllib():
    assert BASE_URL == "https://cloud.lambda.ai/api/v1"
    assert LambdaClient("k")._transport is lambda_api.urllib_transport


class UnreadableBody(io.BytesIO):
    def read(self, *args):
        raise TimeoutError("timed out")


class TruncatedResponse(FakeResponse):
    def read(self, *args):
        raise http.client.IncompleteRead(b'{"data": [', 100)


def _raises(exc_factory):
    def fake_open(req, timeout):
        raise exc_factory(req)
    return fake_open


@pytest.mark.parametrize("fake_open", [
    _raises(lambda req: urllib.error.HTTPError(req.full_url, 500, "Error", {}, UnreadableBody(b""))),
    lambda req, timeout: TruncatedResponse(200, b""),
    _raises(lambda req: http.client.BadStatusLine("garbage")),
    _raises(lambda req: http.client.InvalidURL("bad url")),
], ids=["httperror-unreadable-body", "incomplete-read", "bad-status-line", "invalid-url"])
def test_urllib_transport_wraps_every_transport_failure(monkeypatch, fake_open):
    monkeypatch.setattr(lambda_api._OPENER, "open", fake_open)
    with pytest.raises(LambdaAPIError) as e:
        lambda_api.urllib_transport("GET", BASE_URL + "/x", None, {"Authorization": "Bearer sekrit-123"})
    assert e.value.status == 0 and "sekrit-123" not in str(e.value)


def test_urllib_transport_header_rejection_does_not_echo_the_key():
    # http.client refuses a CR in a header value before connecting, quoting the value in its ValueError.
    # Loopback port 9 (discard): nothing is ever sent anywhere.
    with pytest.raises(LambdaAPIError) as e:
        lambda_api.urllib_transport("GET", "http://127.0.0.1:9/x", None, {"Authorization": "Bearer sekrit-123\r"})
    assert e.value.code == "gpuctl/transport-error" and "sekrit-123" not in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__


class _Recorder(http.server.BaseHTTPRequestHandler):
    """Loopback server: records each request's method, path and Authorization header; replies as configured."""

    seen: list
    location: str | None = None

    def _reply(self):
        self.seen.append((self.command, self.path, self.headers.get("Authorization")))
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = b"" if self.location else b'{"data": {}}'
        self.send_response(302 if self.location else 200)
        if self.location:
            self.send_header("Location", self.location)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply

    def log_message(self, *args):
        pass


def _serve(seen, location=None):
    handler = type("Handler", (_Recorder,), {"seen": seen, "location": location})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize("method, body", [("GET", None), ("POST", {"a": 1})])
def test_urllib_transport_refuses_redirects_and_never_forwards_the_key(monkeypatch, method, body):
    # The module opener honours HTTP(S)_PROXY from the environment; this test must reach the loopback servers
    # directly, so it runs the real redirect-refusal handler in an opener without proxies.
    monkeypatch.setattr(lambda_api, "_OPENER",
                        urllib.request.build_opener(urllib.request.ProxyHandler({}), lambda_api._RefuseRedirects))
    other_seen, api_seen = [], []
    other = _serve(other_seen)  # a different host:port that the redirect points at
    api = _serve(api_seen, location=f"http://127.0.0.1:{other.server_address[1]}/stolen")
    try:
        status, payload = lambda_api.urllib_transport(
            method, f"http://127.0.0.1:{api.server_address[1]}/x", body, {"Authorization": "Bearer sekrit-123"})
    finally:
        for server in (api, other):
            server.shutdown()
            server.server_close()
    assert (status, payload) == (302, {})
    assert api_seen == [(method, "/x", "Bearer sekrit-123")]
    assert other_seen == []


def test_client_turns_a_redirect_into_an_error():
    client, _ = make_client({("GET", "/instances"): (302, {})})
    with pytest.raises(LambdaAPIError) as e:
        client.list_instances()
    assert (e.value.code, e.value.status) == ("http/302", 302)
