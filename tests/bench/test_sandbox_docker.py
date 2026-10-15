"""Integration tests for goaly.bench.sandbox's solve mode in real containers (spec §6.1, §8.8,
V27, V28).

They need Docker and run only with ``pytest --integration``. Each run has its own run ID, so
cleanup never touches anyone else's containers.
"""

import json
import secrets
import shlex
import socketserver
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import SecretStr

from goaly.bench.sandbox import (
    LABEL_RUN,
    SandboxError,
    SandboxResult,
    SolveEnvironment,
    ensure_image,
    run_solve,
    stop_solve,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN = f"run-token-{secrets.token_hex(8)}"
AGENT_PASSWORD = f"agent-{secrets.token_hex(8)}"
PORTS = (55432, 55433)


@dataclass(frozen=True)
class Layout:
    checkout: Path
    run_dir: Path
    solve_dir: Path
    inputs_dir: Path
    instances: Path
    proxy_socket: Path
    pg_dirs: tuple[Path, Path]


class _Proxy(BaseHTTPRequestHandler):
    """A stand-in for T03's LlmProxy on its Unix socket: it answers a chat completion that
    brings the run's token.
    """

    seen: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        self.seen.append(self.headers.get("Authorization", ""))
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        authorized = self.headers.get("Authorization") == f"Bearer {TOKEN}"
        body = json.dumps({"ok": authorized}).encode()
        self.send_response(200 if authorized else 401)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _UnixHttpServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class _Postgres(socketserver.BaseRequestHandler):
    """A stand-in for a solving instance's socket: it greets whoever connects."""

    def handle(self) -> None:
        self.request.sendall(b"pg-hello")


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve(server: socketserver.BaseServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


@pytest.fixture(scope="module")
def image() -> str:
    return ensure_image(REPO_ROOT)


@pytest.fixture
def run_id() -> Iterator[str]:
    run_id = f"it-{secrets.token_hex(4)}"
    yield run_id
    leftovers = subprocess.run(  # noqa: S603  (docker with arguments built here)
        ["docker", "ps", "--all", "--quiet", f"--filter=label={LABEL_RUN}={run_id}"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    stop_solve(run_id)  # Even after a failed test.
    assert leftovers == [], "a solve container outlived its run"


@pytest.fixture
def layout(tmp_path: Path) -> Iterator[Layout]:
    root = tmp_path.resolve()
    checkout = root / "snapshot"
    (checkout / "src").mkdir(parents=True)
    (checkout / "README.md").write_text("the harness\n", encoding="utf-8")
    run_dir = root / "run"
    solve_dir = run_dir / "solve"
    solve_dir.mkdir(parents=True)
    for name in ("results.jsonl", "manifest.json", "llm_calls.jsonl"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir()
    (inputs_dir / "plans.json").write_text('{"plan": 1}\n', encoding="utf-8")
    instances = root / "data" / "tasks.jsonl"
    instances.parent.mkdir()
    instances.write_text(json.dumps({"id": 1, "base_sql": "SELECT 1"}) + "\n", encoding="utf-8")
    proxy_dir = root / "proxy"
    proxy_dir.mkdir()
    proxy_socket = proxy_dir / "llm.sock"
    servers: list[socketserver.BaseServer] = [_UnixHttpServer(str(proxy_socket), _Proxy)]
    pg_dirs = (root / "pg1", root / "pg2")
    for directory, port in zip(pg_dirs, PORTS, strict=True):
        directory.mkdir()
        servers.append(_UnixServer(str(directory / f".s.PGSQL.{port}"), _Postgres))
    threads = [serve(server) for server in servers]
    yield Layout(checkout, run_dir, solve_dir, inputs_dir, instances, proxy_socket, pg_dirs)
    for server in servers:
        server.shutdown()
        server.server_close()
    for thread in threads:
        thread.join()


def environment(layout: Layout, run_id: str) -> SolveEnvironment:
    return SolveEnvironment(
        base_url="http://localhost",
        token=SecretStr(TOKEN),
        llm_uds=layout.proxy_socket,
        pg_socket_dirs=layout.pg_dirs,
        pg_agent_password=SecretStr(AGENT_PASSWORD),
        run_id=run_id,
    )


def solve(image: str, layout: Layout, run_id: str, script: str) -> SandboxResult:
    return run_solve(
        image,
        layout.checkout,
        layout.solve_dir,
        layout.instances,
        inputs_dir=layout.inputs_dir,
        environment=environment(layout, run_id),
        argv=["bash", "-c", script],
        timeout_s=120,
    )


def python(code: str) -> str:
    return f"python -c {shlex.quote(code)}"


def test_the_container_sees_no_real_key_no_grader_password_and_no_gold(
    image: str, layout: Layout, run_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_key = f"sk-real-{secrets.token_hex(8)}"
    grader = f"grader-{secrets.token_hex(8)}"
    monkeypatch.setenv("GOALY_API_KEY", real_key)
    monkeypatch.setenv("GOALY_PG_GRADER_PASSWORD", grader)
    hidden = {
        "GOLD": Path.home() / "data" / "effisql" / "golden",
        "CONFIG": Path.home() / ".config",
        "DOCKER": Path("/var/run/docker.sock"),
        "RESULTS": layout.run_dir / "results.jsonl",
        "MANIFEST": layout.run_dir / "manifest.json",
        "CALLS": layout.run_dir / "llm_calls.jsonl",
    }
    checks = "; ".join(f"test -e {path} && echo {name}_VISIBLE" for name, path in hidden.items())

    result = solve(image, layout, run_id, f"env; {checks}; echo done")

    assert result.exit_code == 0, result.output
    assert "done" in result.output
    assert f"GOALY_API_KEY={TOKEN}" in result.output
    assert f"GOALY_PG_AGENT_PASSWORD={AGENT_PASSWORD}" in result.output
    assert real_key not in result.output
    assert grader not in result.output
    assert "GRADER" not in result.output
    assert "_VISIBLE" not in result.output


def test_a_write_outside_the_solve_directory_fails(image: str, layout: Layout, run_id: str) -> None:
    targets = {
        "checkout": layout.checkout / "planted",
        "instances": layout.instances,
        "inputs": layout.inputs_dir / "plans.json",
        "proxy": layout.proxy_socket.parent / "planted",
        "pg": layout.pg_dirs[0] / "planted",
        "root": Path("/planted"),
    }
    script = "".join(f"echo x >> {path}; echo {name}=$?\n" for name, path in targets.items())
    script += f"echo solved > {layout.solve_dir}/predictions.jsonl; echo solve=$?\n"
    script += f"cat {layout.inputs_dir}/plans.json\n"

    result = solve(image, layout, run_id, script)

    assert "Read-only file system" in result.output
    for name in targets:
        assert f"{name}=1" in result.output, result.output
    assert "solve=0" in result.output
    assert '{"plan": 1}' in result.output  # The staged inputs can be read.
    assert not (layout.checkout / "planted").exists()
    assert (layout.solve_dir / "predictions.jsonl").read_text(encoding="utf-8") == "solved\n"


def test_no_ip_netlink_or_packet_socket_can_be_opened(
    image: str, layout: Layout, run_id: str
) -> None:
    code = (
        "import socket\n"
        "for name in ['AF_INET', 'AF_INET6', 'AF_NETLINK', 'AF_PACKET']:\n"
        "    family = getattr(socket, name)\n"
        "    kind = socket.SOCK_STREAM if family in (2, 10) else socket.SOCK_RAW\n"
        "    try:\n"
        "        socket.socket(family, kind).close()\n"
        "        print(name, 'opened')\n"
        "    except PermissionError:\n"
        "        print(name, 'refused')\n"
        "try:\n"
        "    socket.create_connection(('127.0.0.1', 13337), timeout=5)\n"
        "    print('tcp connected')\n"
        "except PermissionError:\n"
        "    print('tcp refused')\n"
    )

    result = solve(image, layout, run_id, python(code))

    assert result.exit_code == 0, result.output
    for name in ("AF_INET", "AF_INET6", "AF_NETLINK", "AF_PACKET", "tcp"):
        assert f"{name} refused" in result.output, result.output


def test_the_llm_proxy_is_reachable_through_its_socket_with_the_runs_token(
    image: str, layout: Layout, run_id: str
) -> None:
    code = (
        "import http.client, os, socket\n"
        "class Unix(http.client.HTTPConnection):\n"
        "    def connect(self):\n"
        "        self.sock = socket.socket(socket.AF_UNIX)\n"
        "        self.sock.connect(os.environ['GOALY_LLM_UDS'])\n"
        "connection = Unix('localhost')\n"
        "token = os.environ['GOALY_API_KEY']\n"
        "connection.request('POST', '/chat/completions', body=b'{}',\n"
        "                   headers={'Authorization': 'Bearer ' + token})\n"
        "print(connection.getresponse().read().decode())\n"
    )

    result = solve(image, layout, run_id, python(code))

    assert result.exit_code == 0, result.output
    assert '{"ok": true}' in result.output
    assert _Proxy.seen[-1] == f"Bearer {TOKEN}"


def test_the_solving_instances_are_reachable_through_their_sockets(
    image: str, layout: Layout, run_id: str
) -> None:
    code = (
        "import os, socket\n"
        "directories = os.environ['GOALY_PG_SOCKET_DIRS'].split(',')\n"
        "for directory, port in zip(directories, [55432, 55433]):\n"
        "    client = socket.socket(socket.AF_UNIX)\n"
        "    client.connect(f'{directory}/.s.PGSQL.{port}')\n"
        "    print(port, client.recv(16).decode())\n"
    )

    result = solve(image, layout, run_id, python(code))

    assert result.exit_code == 0, result.output
    for port in PORTS:
        assert f"{port} pg-hello" in result.output


def test_stop_solve_cancels_a_running_solve(image: str, layout: Layout, run_id: str) -> None:
    outcome: list[BaseException | SandboxResult] = []

    def run() -> None:
        try:
            outcome.append(solve(image, layout, run_id, "sleep 300"))
        except BaseException as error:  # noqa: BLE001  (the test inspects it)
            outcome.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        running = subprocess.run(  # noqa: S603  (docker with arguments built here)
            ["docker", "ps", "--quiet", f"--filter=label={LABEL_RUN}={run_id}"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        if running:
            break
        time.sleep(0.2)
    started = time.monotonic()

    assert stop_solve(run_id) == 1

    worker.join(timeout=60)
    assert not worker.is_alive()
    assert time.monotonic() - started < 30
    assert isinstance(outcome[0], SandboxError)
