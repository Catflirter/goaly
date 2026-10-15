"""Integration tests for goaly.bench.sandbox's solve mode in real containers (spec §6.1).

They need Docker and run only with ``pytest --integration``. Each run has its own run ID, so
cleanup never touches anyone else's containers.
"""

import json
import secrets
import shlex
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import SecretStr

from goaly.bench.sandbox import (
    LABEL_RUN,
    SandboxResult,
    SolveEnvironment,
    ensure_image,
    run_solve,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN = f"run-token-{secrets.token_hex(8)}"
AGENT_PASSWORD = f"agent-{secrets.token_hex(8)}"


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
    assert leftovers == [], "a solve container outlived its run"


@pytest.fixture
def layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    harness = tmp_path / "harness"
    (harness / "src").mkdir(parents=True)
    (harness / "README.md").write_text("the harness\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "solve"
    run_dir.mkdir(parents=True)
    data = tmp_path / "data" / "tasks.jsonl"
    data.parent.mkdir()
    data.write_text(json.dumps({"id": 1, "base_sql": "SELECT 1"}) + "\n", encoding="utf-8")
    return harness, run_dir, data


def solve(
    image: str, layout: tuple[Path, Path, Path], run_id: str, command: str, base_url: str
) -> SandboxResult:
    harness, run_dir, data = layout
    environment = SolveEnvironment(
        base_url=base_url,
        token=SecretStr(TOKEN),
        pg_ports=(55432, 55433),
        pg_agent_password=SecretStr(AGENT_PASSWORD),
        run_id=run_id,
        seed=1,
    )
    return run_solve(
        image, harness, run_dir, data, environment=environment, command=command, timeout_s=120
    )


def test_the_container_sees_no_real_key_no_grader_password_and_no_gold(
    image: str,
    layout: tuple[Path, Path, Path],
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_key = f"sk-real-{secrets.token_hex(8)}"
    grader = f"grader-{secrets.token_hex(8)}"
    monkeypatch.setenv("GOALY_API_KEY", real_key)
    monkeypatch.setenv("GOALY_PG_GRADER_PASSWORD", grader)
    golden = Path.home() / "data" / "effisql" / "golden"
    config = Path.home() / ".config"

    result = solve(
        image,
        layout,
        run_id,
        f"env; test -e {golden} && echo GOLD_VISIBLE; test -e {config} && echo CONFIG_VISIBLE;"
        " test -e /var/run/docker.sock && echo DOCKER_VISIBLE; echo done",
        "http://127.0.0.1:9",
    )

    assert result.exit_code == 0, result.output
    assert "done" in result.output
    assert f"GOALY_API_KEY={TOKEN}" in result.output
    assert f"GOALY_PG_AGENT_PASSWORD={AGENT_PASSWORD}" in result.output
    assert real_key not in result.output
    assert grader not in result.output
    assert "GRADER" not in result.output
    assert "GOLD_VISIBLE" not in result.output
    assert "CONFIG_VISIBLE" not in result.output
    assert "DOCKER_VISIBLE" not in result.output


def test_a_write_outside_the_run_directory_fails(
    image: str, layout: tuple[Path, Path, Path], run_id: str
) -> None:
    harness, run_dir, data = layout

    result = solve(
        image,
        layout,
        run_id,
        f"touch {harness}/planted; echo harness=$?\n"
        f"echo x >> {data}; echo data=$?\n"
        f"touch /planted; echo root=$?\n"
        f"echo solved > {run_dir}/predictions.jsonl; echo run=$?\n",
        "http://127.0.0.1:9",
    )

    assert "Read-only file system" in result.output
    assert "harness=1" in result.output
    assert "data=1" in result.output
    assert "root=1" in result.output
    assert "run=0" in result.output
    assert not (harness / "planted").exists()
    assert (run_dir / "predictions.jsonl").read_text(encoding="utf-8") == "solved\n"


class _Proxy(BaseHTTPRequestHandler):
    """A stand-in for T03's LlmProxy: it answers a chat completion that brings the run's token."""

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


@pytest.fixture
def proxy() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_the_llm_proxy_is_reachable_with_the_runs_token(
    image: str, layout: tuple[Path, Path, Path], run_id: str, proxy: str
) -> None:
    call = (
        "import os, urllib.request\n"
        "request = urllib.request.Request(\n"
        "    os.environ['GOALY_BASE_URL'] + '/chat/completions', data=b'{}',\n"
        "    headers={'Authorization': 'Bearer ' + os.environ['GOALY_API_KEY']})\n"
        "print(urllib.request.urlopen(request, timeout=10).read().decode())\n"
    )

    result = solve(image, layout, run_id, f"python -c {shlex.quote(call)}", proxy)

    assert result.exit_code == 0, result.output
    assert '{"ok": true}' in result.output
    assert _Proxy.seen[-1] == f"Bearer {TOKEN}"
