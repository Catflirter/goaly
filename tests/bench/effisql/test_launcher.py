"""Tests for goaly.bench.effisql.launcher: the solve phase in its container, with fakes.

The sandbox, the proxy and core's SolveRequest are stand-ins: nothing here starts Docker or
reaches the gateway.
"""

import asyncio
import fcntl
import json
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import SecretStr

from goaly.bench.effisql.benchmark import EffiSqlBenchmark
from goaly.bench.effisql.harness import EffiSqlPaths, EffiSqlWiring, open_host
from goaly.bench.effisql.launcher import (
    EffiSqlLauncher,
    SolveContainerError,
    SolveRequestLike,
    checkout_of,
    hold_solving_instances,
    proxy_socket,
    snapshot_checkout,
    stage_inputs,
)
from goaly.bench.effisql.layout import SolveInputs
from goaly.bench.effisql.pg_settings import AgentCredentials, load_agent_credentials
from goaly.bench.sandbox import SandboxResult, SolveEnvironment
from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord
from goaly.contracts.gen.task_pb2 import SPLIT_EXTERNAL, SPLIT_HELDOUT, SPLIT_VAL
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core.llm import CallRecorder
from goaly.harness.core.registry import HostBenchmark, HostOptions
from goaly.harness.core.runner import RunFiles
from tests.bench.effisql.fake_run import write_inputs
from tests.bench.effisql.synthetic import GOLD_SQL, task

pytestmark = pytest.mark.anyio

REPO = Path(__file__).resolve().parents[3]
TOKEN = "per-run-token"  # noqa: S105 - a fake dummy token.


@dataclass(frozen=True)
class Request:
    """Core's SolveRequest, as the launcher reads it."""

    run_id: str
    solve_dir: Path
    inputs_dir: Path
    llm_calls: Path
    tasks: tuple[tuple[str, int], ...]
    pack_dir: Path | None
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
    runs: list[tuple[list[str], Path | None, SolveEnvironment, float]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def prepare(self, checkout: Path) -> None:
        self.prepared.append(checkout)

    async def run(
        self,
        checkout: Path,
        request: SolveRequestLike,
        argv: list[str],
        staged_instances: Path | None,
        environment: SolveEnvironment,
        timeout_s: float,
    ) -> SandboxResult:
        self.runs.append((argv, staged_instances, environment, timeout_s))
        self.started.set()
        if self.hang:
            await asyncio.Event().wait()
        return SandboxResult(self.exit_code, "container output", timed_out=False)

    async def stop(self, run_id: str) -> None:
        self.removed.append(run_id)


@dataclass
class FakeProxy:
    recorders: list[CallRecorder] = field(default_factory=list)
    opened: list[tuple[str, Path]] = field(default_factory=list)
    stopped: bool = False

    uds: Path = Path("/run/goaly/proxy/llm.sock")

    def container_env(self) -> dict[str, str]:
        return {
            "GOALY_API_KEY": TOKEN,
            "GOALY_BASE_URL": "http://localhost",
            "GOALY_LLM_UDS": str(self.uds),
        }


def credentials() -> AgentCredentials:
    return AgentCredentials(agent_password=SecretStr("agent-secret"))


def fake_checkout(root: Path) -> Path:
    """A checkout without .git, as an RSI snapshot is: used as it is."""
    pack = root / "src" / "goaly" / "harness" / "packs" / "effisql"
    pack.mkdir(parents=True)
    (pack / "policy.py").write_text("x = 1\n")
    return pack


def setup(
    tmp_path: Path, sandbox: FakeSandbox
) -> tuple[EffiSqlLauncher, FakeProxy, Request, EffiSqlPaths]:
    paths = write_inputs(tmp_path)
    pack = fake_checkout(tmp_path / "checkout")
    proxy = FakeProxy()

    @asynccontextmanager
    async def open_proxy(
        recorder: CallRecorder, run_id: str, uds: Path
    ) -> AsyncIterator[FakeProxy]:
        proxy.recorders.append(recorder)
        proxy.opened.append((run_id, uds))
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
        socket_dirs=lambda: (Path("/sock/solve-1"), Path("/sock/solve-2")),
        lock_dir=lambda: tmp_path / "locks",
    )
    run = tmp_path / "run"
    request = Request(
        run_id="run-1",
        solve_dir=run / "solve",
        inputs_dir=run / "inputs",
        llm_calls=run / "llm_calls.jsonl",
        tasks=(("1", 0), ("3", 0), ("1", 1)),
        pack_dir=pack,
    )
    return launcher, proxy, request, paths


async def test_it_stages_the_inputs_and_runs_the_container(tmp_path: Path) -> None:
    sandbox = FakeSandbox()
    launcher, proxy, request, paths = setup(tmp_path, sandbox)

    await launcher(request)

    assert sandbox.prepared == [(tmp_path / "checkout").resolve()]
    ((argv, staged, environment, timeout_s),) = sandbox.runs
    assert staged is None
    assert argv == [
        "python", "-m", "goaly", "run", "--bench", "effisql", "--out", str(request.solve_dir),
        "--inputs", str(request.inputs_dir), "--split", "val",
    ]  # fmt: skip
    assert environment.variables() == {
        "GOALY_BASE_URL": "http://localhost",
        "GOALY_API_KEY": TOKEN,
        "GOALY_LLM_UDS": "/run/goaly/proxy/llm.sock",
        "GOALY_PG_SOCKET_DIRS": "/sock/solve-1,/sock/solve-2",
        "GOALY_PG_AGENT_PASSWORD": "agent-secret",
    }
    assert timeout_s == 2 * 600 * 1.5 + 600  # 3 tasks, 2 at a time.
    inputs = SolveInputs(request.inputs_dir)
    assert inputs.snapshot.read_bytes() == paths.snapshot.read_bytes()
    assert inputs.plans.read_bytes() == paths.plans.read_bytes()
    assert GOLD_SQL not in inputs.tasks.read_text()
    assert "optimized_sql" not in inputs.tasks.read_text()
    assert [r.call_id for r in read_jsonl(request.llm_calls, LlmCallRecord)] == ["via-proxy"]
    assert proxy.stopped
    assert proxy.opened == [("run-1", proxy_socket(request.solve_dir.parent, "run-1"))]
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
    assert sandbox.runs[0][0][-2:] == ["--instances", str(staged)]
    assert sandbox.runs[0][1] == staged
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


def test_a_git_checkout_is_snapshotted_without_git_with_its_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    pack = fake_checkout(repo)
    run = ["git", "-C", str(repo), "-c", "user.email=t@e", "-c", "user.name=t"]
    subprocess.run([*run, "init", "-q"], check=True)  # noqa: S603
    subprocess.run([*run, "add", "."], check=True)  # noqa: S603
    subprocess.run([*run, "commit", "-q", "-m", "c"], check=True)  # noqa: S603
    (pack / "policy.py").write_text("x = 2\n")  # An uncommitted change the host hashed.
    (repo / "untracked.txt").write_text("not tracked")

    snapshot = snapshot_checkout(repo, tmp_path / "run" / "checkout")

    copied = snapshot / "src" / "goaly" / "harness" / "packs" / "effisql" / "policy.py"
    assert copied.read_text() == "x = 2\n"
    assert not (snapshot / ".git").exists()
    assert not (snapshot / "untracked.txt").exists()
    assert snapshot_checkout(snapshot, tmp_path / "unused") == snapshot


async def test_a_solve_waits_for_no_all_instances_grading(tmp_path: Path) -> None:
    sandbox = FakeSandbox()
    launcher, proxy, request, _ = setup(tmp_path, sandbox)
    locks = tmp_path / "locks"
    locks.mkdir()
    with (locks / "solve-2.lock").open("w") as grading:
        fcntl.flock(grading, fcntl.LOCK_EX | fcntl.LOCK_NB)  # As T10's all-instances grader.

        with pytest.raises(SolveContainerError, match="solve-2 is grading"):
            await launcher(request)

    assert sandbox.runs == []
    assert proxy.stopped


def test_solve_phases_share_the_locks_and_keep_graders_out(tmp_path: Path) -> None:
    with (
        hold_solving_instances(tmp_path, ["solve-1", "solve-2"]),
        hold_solving_instances(tmp_path, ["solve-1", "solve-2"]),
        (tmp_path / "solve-1.lock").open("w") as grader,
        pytest.raises(BlockingIOError),
    ):
        fcntl.flock(grader, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (tmp_path / "solve-1.lock").open("w") as grader:
        fcntl.flock(grader, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Free once the solves end.


async def test_the_container_gets_the_agent_password_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        raise AssertionError("the launcher must never load every password")

    monkeypatch.setattr("goaly.bench.effisql.pg_settings.load_credentials", refuse)
    sandbox = FakeSandbox()
    launcher, _, request, _ = setup(tmp_path, sandbox)

    await launcher(request)

    (_, _, environment, _) = sandbox.runs[0]
    variables = environment.variables()
    assert [name for name in variables if name.startswith("GOALY_PG_")] == [
        "GOALY_PG_SOCKET_DIRS",
        "GOALY_PG_AGENT_PASSWORD",
    ]
    assert not any("pg.env" in value for value in variables.values())
    assert EffiSqlLauncher().credentials is load_agent_credentials


def test_the_proxy_socket_is_beside_solve_unless_too_long(tmp_path: Path) -> None:
    assert proxy_socket(Path("/runs/r1"), "r1") == Path("/runs/r1/proxy/llm.sock")
    deep = Path("/" + "d" * 120)

    fallback = proxy_socket(deep, "r1")

    assert fallback.name == "llm.sock"
    assert fallback.parent.name == "goaly-proxy-r1"
    assert len(str(fallback)) <= 100


def test_staging_twice_replaces_the_inputs(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    inputs = SolveInputs(tmp_path / "inputs")

    assert stage_inputs(inputs, paths, None) is None
    assert stage_inputs(inputs, paths, None) is None
    assert inputs.tasks.exists()


class TestHost:
    Options = HostOptions

    async def test_the_host_side_grades_by_the_split_and_launches_the_container(
        self, tmp_path: Path
    ) -> None:
        paths = write_inputs(tmp_path)
        wiring = EffiSqlWiring(paths=lambda: paths)
        options = self.Options(RunFiles(tmp_path / "run"), None, SPLIT_HELDOUT, "baseline")

        async with open_host(options, wiring) as opened:
            assert isinstance(opened, HostBenchmark)
            assert isinstance(opened.benchmark, EffiSqlBenchmark)
            host, benchmark = opened, opened.benchmark
            fields = benchmark.manifest_fields()
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

        async with open_host(options, wiring) as opened:
            assert isinstance(opened, HostBenchmark)
            assert isinstance(opened.benchmark, EffiSqlBenchmark)
            host, benchmark = opened, opened.benchmark
            assert host.cache_namespace == ""
            assert benchmark.grade_function is None
            assert [i.instance_id for i in benchmark.load_instances(SPLIT_EXTERNAL)] == ["901"]

    async def test_a_val_run_grades_in_dev_mode(self, tmp_path: Path) -> None:
        paths = write_inputs(tmp_path)
        options = self.Options(RunFiles(tmp_path / "run"), None, SPLIT_VAL, "rsi")

        async with open_host(options, EffiSqlWiring(paths=lambda: paths)) as opened:
            assert isinstance(opened, HostBenchmark)
            assert isinstance(opened.benchmark, EffiSqlBenchmark)
            benchmark = opened.benchmark
            assert benchmark.mode is not None
            assert benchmark.mode.value == "dev"
