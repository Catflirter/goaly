"""Tests for goaly.bench.effisql.launcher: the solve phase in its container, with fakes.

The sandbox, the proxy and core's SolveRequest are stand-ins: nothing here starts Docker or
reaches the gateway.
"""

import asyncio
import json
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import SecretStr

from goaly.bench.effisql.harness import EffiSqlPaths, EffiSqlWiring, open_host
from goaly.bench.effisql.launcher import (
    EffiSqlLauncher,
    SolveContainerError,
    SolveRequestLike,
    checkout_of,
    stage_inputs,
)
from goaly.bench.effisql.layout import SolveInputs
from goaly.bench.effisql.pg_settings import PgCredentials
from goaly.bench.sandbox import SandboxResult, SolveEnvironment
from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord
from goaly.contracts.gen.task_pb2 import SPLIT_EXTERNAL, SPLIT_HELDOUT, SPLIT_VAL
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core.llm import CallRecorder
from goaly.harness.core.runner import RunFiles
from tests.bench.effisql.fake_run import write_inputs
from tests.bench.effisql.synthetic import GOLD_SQL, task

pytestmark = pytest.mark.anyio

CHECKOUT = Path(__file__).resolve().parents[3]
PACK = CHECKOUT / "src" / "goaly" / "harness" / "packs" / "effisql"
TOKEN = "per-run-token"  # noqa: S105 - a fake dummy token.


@dataclass(frozen=True)
class Request:
    """Core's SolveRequest, as the launcher reads it."""

    run_id: str
    solve_dir: Path
    inputs_dir: Path
    llm_calls: Path
    tasks: tuple[tuple[str, int], ...]
    pack_dir: Path | None = PACK
    instances_file: Path | None = None
    max_tasks: int = 2
    task_timeout_s: float | None = None

    def goaly_args(
        self, *, solve_dir: Path, inputs_dir: Path, instances_file: Path | None = None
    ) -> list[str]:
        where = ["--instances", str(instances_file)] if instances_file else ["--split", "val"]
        return [
            "run",
            "--bench",
            "effisql",
            "--out",
            str(solve_dir),
            "--inputs",
            str(inputs_dir),
            *where,
        ]


@dataclass
class FakeSandbox:
    exit_code: int = 0
    hang: bool = False
    prepared: list[Path] = field(default_factory=list)
    runs: list[tuple[str, SolveEnvironment, float]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def prepare(self, checkout: Path) -> None:
        self.prepared.append(checkout)

    async def run(
        self,
        checkout: Path,
        request: SolveRequestLike,
        command: str,
        environment: SolveEnvironment,
        timeout_s: float,
    ) -> SandboxResult:
        self.runs.append((command, environment, timeout_s))
        self.started.set()
        if self.hang:
            await asyncio.Event().wait()
        return SandboxResult(self.exit_code, "container output", timed_out=False)

    async def remove(self, run_id: str) -> None:
        self.removed.append(run_id)


@dataclass
class FakeProxy:
    recorders: list[CallRecorder] = field(default_factory=list)
    stopped: bool = False

    base_url = "http://127.0.0.1:41234"

    def container_env(self) -> dict[str, str]:
        return {"GOALY_API_KEY": TOKEN, "GOALY_BASE_URL": self.base_url}


def credentials() -> PgCredentials:
    secret = SecretStr("agent-secret")
    return PgCredentials(admin_password=secret, agent_password=secret, grader_password=secret)


def setup(
    tmp_path: Path, sandbox: FakeSandbox
) -> tuple[EffiSqlLauncher, FakeProxy, Request, EffiSqlPaths]:
    paths = write_inputs(tmp_path)
    proxy = FakeProxy()

    @asynccontextmanager
    async def open_proxy(recorder: CallRecorder, run_id: str) -> AsyncIterator[FakeProxy]:
        proxy.recorders.append(recorder)
        recorder.record(LlmCallRecord(call_id="via-proxy"))
        try:
            yield proxy
        finally:
            proxy.stopped = True

    launcher = EffiSqlLauncher(
        paths=lambda: paths,
        sandbox=lambda: sandbox,
        proxy=open_proxy,
        credentials=credentials,
        solving_ports=(55432, 55433),
    )
    run = tmp_path / "run"
    request = Request(
        run_id="run-1",
        solve_dir=run / "solve",
        inputs_dir=run / "inputs",
        llm_calls=run / "llm_calls.jsonl",
        tasks=(("1", 0), ("3", 0), ("1", 1)),
    )
    return launcher, proxy, request, paths


async def test_it_stages_the_inputs_and_runs_the_container(tmp_path: Path) -> None:
    sandbox = FakeSandbox()
    launcher, proxy, request, paths = setup(tmp_path, sandbox)

    await launcher(request)

    assert sandbox.prepared == [CHECKOUT]
    ((command, environment, timeout_s),) = sandbox.runs
    assert shlex.split(command) == [
        "python", "-m", "goaly", "run", "--bench", "effisql", "--out", str(request.solve_dir),
        "--inputs", str(request.inputs_dir), "--split", "val",
    ]  # fmt: skip
    assert environment.variables() == {
        "GOALY_BASE_URL": "http://127.0.0.1:41234",
        "GOALY_API_KEY": TOKEN,
        "GOALY_PG_HOST": "127.0.0.1",
        "GOALY_PG_PORTS": "55432,55433",
        "GOALY_PG_AGENT_PASSWORD": "agent-secret",
        "GOALY_RUN_ID": "run-1",
        "GOALY_SEED": "0",
    }
    assert timeout_s == 2 * 600 * 1.5 + 600  # 3 tasks, 2 at a time.
    inputs = SolveInputs(request.inputs_dir)
    assert inputs.snapshot.read_bytes() == paths.snapshot.read_bytes()
    assert inputs.plans.read_bytes() == paths.plans.read_bytes()
    assert GOLD_SQL not in inputs.tasks.read_text()
    assert "optimized_sql" not in inputs.tasks.read_text()
    assert [r.call_id for r in read_jsonl(request.llm_calls, LlmCallRecord)] == ["via-proxy"]
    assert proxy.stopped
    log = request.solve_dir.parent / "solve-container.log"
    assert log.read_text() == "exit code 0; timed out: False\ncontainer output"
    assert not request.solve_dir.exists()  # Core makes it; the launcher writes nothing there.


async def test_an_organizers_file_is_staged_without_its_gold(tmp_path: Path) -> None:
    sandbox = FakeSandbox()
    launcher, _, request, _ = setup(tmp_path, sandbox)
    organizers = tmp_path / "heldout.jsonl"
    organizers.write_text(json.dumps({**task(id=901), "extra": "kept on the host"}) + "\n")
    external = Request(**{**request.__dict__, "instances_file": organizers})

    await launcher(external)

    staged = SolveInputs(request.inputs_dir).tasks
    assert shlex.split(sandbox.runs[0][0])[-2:] == ["--instances", str(staged)]
    record = json.loads(staged.read_text())
    assert record["id"] == 901
    assert GOLD_SQL not in staged.read_text()
    assert "extra" not in record


async def test_a_failed_container_raises_after_the_proxy_stops(tmp_path: Path) -> None:
    sandbox = FakeSandbox(exit_code=3)
    launcher, proxy, request, _ = setup(tmp_path, sandbox)

    with pytest.raises(SolveContainerError, match="exited 3"):
        await launcher(request)

    assert proxy.stopped


async def test_cancelling_removes_the_container_then_stops_the_proxy(tmp_path: Path) -> None:
    sandbox = FakeSandbox(hang=True)
    launcher, proxy, request, _ = setup(tmp_path, sandbox)

    running = asyncio.create_task(launcher(request))
    await sandbox.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert sandbox.removed == ["run-1"]
    assert proxy.stopped


async def test_a_request_needs_tasks_and_a_pack(tmp_path: Path) -> None:
    launcher, _, request, _ = setup(tmp_path, FakeSandbox())

    with pytest.raises(ValueError, match="at least one task"):
        await launcher(Request(**{**request.__dict__, "tasks": ()}))
    with pytest.raises(ValueError, match="needs its pack"):
        checkout_of(None)
    with pytest.raises(ValueError, match="not a pack"):
        checkout_of(tmp_path / "a" / "b" / "c" / "d" / "e")


def test_staging_twice_replaces_the_inputs(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    inputs = SolveInputs(tmp_path / "inputs")

    assert stage_inputs(inputs, paths, None) is None
    assert stage_inputs(inputs, paths, None) is None
    assert inputs.tasks.exists()


class TestHost:
    @dataclass(frozen=True)
    class Options:
        files: RunFiles
        instances_file: Path | None
        split: int
        budget_phase: str | None

    async def test_the_host_side_grades_by_the_split_and_launches_the_container(
        self, tmp_path: Path
    ) -> None:
        paths = write_inputs(tmp_path)
        wiring = EffiSqlWiring(paths=lambda: paths)
        options = self.Options(RunFiles(tmp_path / "run"), None, SPLIT_HELDOUT, "baseline")

        async with open_host(options, wiring) as host:  # type: ignore[arg-type]
            fields = host.benchmark.manifest_fields()
            assert fields.environment_settings["grading_mode"] == "official"
            assert fields.environment_settings["grading_priority"] == "baseline_heldout"
            assert host.cache_namespace.startswith("official-")
            assert isinstance(host.launcher, EffiSqlLauncher)

    async def test_an_external_run_has_no_grader(self, tmp_path: Path) -> None:
        paths = write_inputs(tmp_path)
        organizers = tmp_path / "heldout.jsonl"
        organizers.write_text(json.dumps(task(id=901)) + "\n")
        wiring = EffiSqlWiring(paths=lambda: paths)
        options = self.Options(RunFiles(tmp_path / "run"), organizers, SPLIT_EXTERNAL, "final")

        async with open_host(options, wiring) as host:  # type: ignore[arg-type]
            assert host.cache_namespace == ""
            assert host.benchmark.grade_function is None
            assert [i.instance_id for i in host.benchmark.load_instances(SPLIT_EXTERNAL)] == ["901"]

    async def test_a_val_run_grades_in_dev_mode(self, tmp_path: Path) -> None:
        paths = write_inputs(tmp_path)
        options = self.Options(RunFiles(tmp_path / "run"), None, SPLIT_VAL, "rsi")

        async with open_host(options, EffiSqlWiring(paths=lambda: paths)) as host:  # type: ignore[arg-type]
            assert host.benchmark.mode is not None
            assert host.benchmark.mode.value == "dev"
