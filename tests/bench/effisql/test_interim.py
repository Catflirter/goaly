"""Tests for the TEMPORARY interim driver: the whole isolated run, with a fake container.

The fake container runs the real in-container command, `goaly run --phase solve --in-container`,
in this process, with a scripted model and fake solving instances, so the plan, the launcher,
the solve phase and the host's grading all run as they do for real, without Docker, a
database or the gateway.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from goaly import cli as goaly_cli
from goaly.bench.effisql.harness import EffiSqlWiring, open_harness
from goaly.bench.effisql.interim import (
    InterimRequest,
    RecordingTransport,
    SolveRequest,
    _lines,  # The reader of the container's untrusted files, tested directly.
    run_interim,
)
from goaly.bench.effisql.launcher import EffiSqlLauncher, SolveRequestLike
from goaly.bench.effisql.pg_settings import PgCredentials
from goaly.bench.effisql.splits import load_splits
from goaly.bench.sandbox import SandboxResult, SolveEnvironment
from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord
from goaly.contracts.gen.result_pb2 import TASK_ERROR_TYPE_POLICY_CRASH, TaskResult
from goaly.contracts.gen.task_pb2 import GRADE_STATUS_PREDICTION_FAILED, SPLIT_TRAIN
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core import cli
from goaly.harness.core.llm import CallRecorder
from goaly.harness.core.registry import HarnessOptions, Registry, RunnableHarness
from tests.bench.effisql.fake_run import OFFICIAL_FORM, FakeWiring, fake_wiring

pytestmark = pytest.mark.anyio

TRAIN = load_splits().train[:2]


class FakeContainer:
    """Runs the container's command in this process, as the fake wiring's in-container side."""

    def __init__(self, fakes: FakeWiring, *, solve: bool = True) -> None:
        self.fakes = fakes
        self.solve = solve
        self.commands: list[list[str]] = []

    async def prepare(self, checkout: Path) -> None:
        pass

    async def run(
        self,
        checkout: Path,
        request: SolveRequestLike,
        argv: list[str],
        staged_instances: Path | None,
        environment: SolveEnvironment,
        timeout_s: float,
    ) -> SandboxResult:
        assert argv[:3] == ["python", "-m", "goaly"]
        self.commands.append(argv[3:])
        if not self.solve:
            return SandboxResult(0, "solved nothing", timed_out=False)
        registry = Registry()
        registry.add("effisql", "v0", self._open)
        status = await asyncio.to_thread(
            cli.main, argv[3:], registry, budgets_file=goaly_cli.BUDGETS_FILE
        )
        return SandboxResult(status, "", timed_out=False)

    def _open(self, options: HarnessOptions) -> AbstractAsyncContextManager[RunnableHarness]:
        assert options.in_container
        assert options.inputs_dir is not None
        inside = EffiSqlWiring.in_container(options.inputs_dir)
        wiring = EffiSqlWiring(
            paths=inside.paths, llm=self.fakes.wiring.llm, solving=self.fakes.wiring.solving
        )
        return open_harness(options, wiring)

    async def stop(self, run_id: str) -> None:
        pass


class FakeProxy:
    def container_env(self) -> dict[str, str]:
        return {
            "GOALY_API_KEY": "token",
            "GOALY_BASE_URL": "http://localhost",
            "GOALY_LLM_UDS": "/run/goaly/proxy/llm.sock",
        }


@asynccontextmanager
async def fake_proxy(recorder: CallRecorder, run_id: str) -> AsyncIterator[FakeProxy]:
    yield FakeProxy()


def launcher(fakes: FakeWiring, container: FakeContainer, locks: Path) -> EffiSqlLauncher:
    secret = SecretStr("agent")
    return EffiSqlLauncher(
        paths=fakes.wiring.paths,
        sandbox=lambda: container,
        proxy=fake_proxy,
        credentials=lambda: PgCredentials(
            admin_password=secret, agent_password=secret, grader_password=secret
        ),
        socket_dirs=lambda: (Path("/sock/solve-1"), Path("/sock/solve-2")),
        lock_dir=lambda: locks,
    )


def request(tmp_path: Path, seeds: tuple[int, ...] = (0,)) -> InterimRequest:
    return InterimRequest(
        out=tmp_path / "run", split=SPLIT_TRAIN, only=TRAIN, seeds=seeds, run_id="run-1"
    )


async def test_the_whole_run_solves_in_the_container_and_grades_on_the_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("goaly.cli.confine_to_python_cpus", lambda: None)
    fakes = fake_wiring(tmp_path)
    container = FakeContainer(fakes)

    summary = await run_interim(
        request(tmp_path, (0, 1)), fakes.wiring, launcher(fakes, container, tmp_path / "locks")
    )

    (argv,) = container.commands
    assert argv[argv.index("--out") + 1] == str(tmp_path / "run" / "solve")
    assert argv[argv.index("--seeds") + 1] == "2"
    assert summary["container_failure"] is None
    tasks = summary["tasks"]
    assert isinstance(tasks, list)
    assert [(t["instance_id"], t["seed"], t["credit"]) for t in tasks] == [
        (TRAIN[0], 0, 0.5),
        (TRAIN[1], 0, 0.5),
        (TRAIN[0], 1, 0.5),
        (TRAIN[1], 1, 0.5),
    ]
    assert sorted((g[0], g[1]) for g in fakes.grader.graded) == sorted(
        (i, OFFICIAL_FORM) for i in TRAIN for _ in (0, 1)
    )
    run = tmp_path / "run"
    host = list(read_jsonl(run / "results.jsonl", TaskResult))
    assert len(host) == 4
    assert all(r.HasField("grade") for r in host)
    for seed in (0, 1):
        official = [
            json.loads(x)
            for x in (run / "official" / f"seed-{seed}.jsonl").read_text().splitlines()
        ]
        assert [str(r["id"]) for r in official] == list(TRAIN)
    assert (run / "inputs" / "tasks.jsonl").exists()


async def test_a_task_the_container_left_no_record_for_fails_with_credit_zero(
    tmp_path: Path,
) -> None:
    fakes = fake_wiring(tmp_path)

    summary = await run_interim(
        request(tmp_path),
        fakes.wiring,
        launcher(fakes, FakeContainer(fakes, solve=False), tmp_path / "locks"),
    )

    tasks = summary["tasks"]
    assert isinstance(tasks, list)
    assert [t["credit"] for t in tasks] == [0.0, 0.0]
    host = list(read_jsonl(tmp_path / "run" / "results.jsonl", TaskResult))
    assert all(r.error_type == TASK_ERROR_TYPE_POLICY_CRASH for r in host)
    assert all(r.grade.status == GRADE_STATUS_PREDICTION_FAILED for r in host)
    assert fakes.grader.graded == []


def test_the_request_names_every_task_and_seed(tmp_path: Path) -> None:
    solve = SolveRequest(
        run_id="r",
        pack_dir=tmp_path,
        solve_dir=tmp_path / "solve",
        inputs_dir=tmp_path / "inputs",
        llm_calls=tmp_path / "calls.jsonl",
        tasks=(("10", 1), ("2", 1), ("10", 2), ("2", 2)),
        split=SPLIT_TRAIN,
        budget_phase="baseline",
        budget_share="effisql",
        max_tasks=4,
        task_timeout_s=30,
    )

    args = solve.goaly_args(solve_dir=tmp_path / "solve", inputs_dir=tmp_path / "inputs")

    assert args[args.index("--only") + 1 : args.index("--first-seed")] == ["2", "10"]
    assert args[args.index("--first-seed") + 1] == "1"
    assert args[args.index("--seeds") + 1] == "2"
    assert args[-2:] == ["--task-timeout", "30"]
    assert args[args.index("--split") : args.index("--split") + 2] == ["--split", "train"]


async def test_the_recording_transport_records_each_forwarded_call(tmp_path: Path) -> None:
    def gateway(request: httpx.Request) -> httpx.Response:
        body = {"usage": {"prompt_tokens": 10, "completion_tokens": 4,
                          "completion_tokens_details": {"reasoning_tokens": 3}}}  # fmt: skip
        return httpx.Response(200, json=body, headers={"x-litellm-response-cost": "0.002"})

    path = tmp_path / "calls.jsonl"
    transport = RecordingTransport(path, "run-1")
    transport._inner = httpx.MockTransport(gateway)  # type: ignore[assignment]  # The test's gateway.
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        response = await client.post("/chat/completions", json={"model": "qwen3.6-35b-a3b"})

    assert response.json()["usage"]["prompt_tokens"] == 10
    (record,) = read_jsonl(path, LlmCallRecord)
    assert (record.run_id, record.model, record.http_status) == ("run-1", "qwen3.6-35b-a3b", 200)
    assert (record.usage.input_tokens, record.usage.output_tokens) == (10, 4)
    assert record.usage.reasoning_tokens == 3
    assert record.usage.cost_usd == pytest.approx(0.002)


def test_a_planted_symlink_in_solve_is_refused(tmp_path: Path) -> None:
    secret = tmp_path / "secret.jsonl"
    secret.write_text("{}\n")
    planted = tmp_path / "results.jsonl"
    planted.symlink_to(secret)

    assert _lines(planted, TaskResult) == [None]
    assert _lines(tmp_path / "absent.jsonl", TaskResult) == []
    fifo = tmp_path / "fifo.jsonl"
    os.mkfifo(fifo)
    assert _lines(fifo, TaskResult) == [None]
