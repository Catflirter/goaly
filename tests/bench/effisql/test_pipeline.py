"""The isolated Effi-SQL run end to end, through T06's host pipeline, with a fake container.

`goaly run --bench effisql` runs on the host as it does for real: core plans the run, Effi-SQL's
launcher stages the inputs, starts the proxy and runs the container, and core grades every task
with the host side. The "container" runs the in-container command (`goaly run --phase solve
--in-container`) in this process, with a scripted model and fake solving instances, so nothing
here starts Docker, a database or the gateway.
"""

import json
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

import pytest
from pydantic import SecretStr

from goaly import cli as goaly_cli
from goaly.bench.effisql.harness import (
    PACK_DIR,
    EffiSqlWiring,
    open_harness,
    open_host,
    pack_models,
)
from goaly.bench.effisql.launcher import EffiSqlLauncher, SolveRequestLike
from goaly.bench.effisql.pg_settings import AgentCredentials
from goaly.bench.effisql.splits import load_splits
from goaly.bench.sandbox import SandboxResult, SolveEnvironment
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core import cli
from goaly.harness.core.llm import CallRecorder
from goaly.harness.core.registry import (
    HarnessOptions,
    HostOptions,
    Registry,
    RunnableHarness,
    RunnableHost,
)
from tests.bench.effisql.fake_run import OFFICIAL_FORM, FakeWiring, fake_wiring

pytestmark = pytest.mark.anyio

TRAIN = load_splits().train[:2]


class FakeContainer:
    """Runs the container's command in this process, with the fakes behind it."""

    def __init__(self, fakes: FakeWiring, *, solve: bool = True) -> None:
        self.fakes = fakes
        self.solve = solve
        self.argv: list[list[str]] = []

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
        self.argv.append(argv[3:])
        if self.solve:
            registry = Registry()
            registry.add("effisql", "v0", self._open, pack_dir=PACK_DIR, models=pack_models())
            await cli.run(argv[3:], registry, budgets_file=goaly_cli.BUDGETS_FILE)
        return SandboxResult(0, "", timed_out=False)

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
async def fake_proxy(recorder: CallRecorder, run_id: str, uds: Path) -> AsyncIterator[FakeProxy]:
    yield FakeProxy()


def host_registry(fakes: FakeWiring, container: FakeContainer, locks: Path) -> Registry:
    launcher = EffiSqlLauncher(
        paths=fakes.wiring.paths,
        sandbox=lambda: container,
        proxy=fake_proxy,
        credentials=lambda: AgentCredentials(agent_password=SecretStr("agent")),
        socket_dirs=lambda: (Path("/sock/solve-1"), Path("/sock/solve-2")),
        lock_dir=lambda: locks,
    )

    def host(options: HostOptions) -> AbstractAsyncContextManager[RunnableHost]:
        return open_host(options, fakes.wiring, launcher)

    registry = Registry()
    registry.add("effisql", "v0", open_harness, pack_dir=PACK_DIR, models=pack_models())
    registry.add_benchmark("effisql", host)
    return registry


async def goaly_run(registry: Registry, run: Path, *args: str) -> None:
    argv = ["run", "--bench", "effisql", "--split", "train", "--only", *TRAIN, *args]
    await cli.run(
        [*argv, "--budget-phase", "baseline", "--out", str(run)],
        registry,
        budgets_file=goaly_cli.BUDGETS_FILE,
    )


async def test_the_isolated_run_solves_in_the_container_and_grades_on_the_host(
    tmp_path: Path,
) -> None:
    fakes = fake_wiring(tmp_path)
    container = FakeContainer(fakes)
    run = (tmp_path / "run").resolve()

    await goaly_run(host_registry(fakes, container, tmp_path / "locks"), run, "--seeds", "2")

    (argv,) = container.argv
    assert argv[argv.index("--phase") + 1] == "solve"
    assert "--in-container" in argv
    assert argv[argv.index("--inputs") + 1] == str(run / "inputs")
    graded = {(r.instance_id, r.seed): r for r in read_jsonl(run / "results.jsonl", TaskResult)}
    assert sorted(graded) == sorted((i, s) for i in TRAIN for s in (0, 1))
    assert all(r.grade.score == 0.5 for r in graded.values())
    assert sorted((g[0], g[1]) for g in fakes.grader.graded) == sorted(
        (i, OFFICIAL_FORM) for i in TRAIN for _ in (0, 1)
    )
    for seed in (0, 1):
        official = (run / "official" / f"seed-{seed}.jsonl").read_text().splitlines()
        assert [str(json.loads(x)["id"]) for x in official] == list(TRAIN)
        assert all(json.loads(x)["prediction"].startswith("```sql\n") for x in official)
        plans = (run / "plans" / f"seed-{seed}.jsonl").read_text().splitlines()
        assert sorted(json.loads(x)["instance_id"] for x in plans) == sorted(TRAIN)
    assert sorted(p.name for p in (run / "inputs").iterdir()) == [
        "plans.json",
        "schema_snapshot.json",
        "tasks.jsonl",
    ]


async def test_a_task_the_container_left_no_record_for_fails(tmp_path: Path) -> None:
    fakes = fake_wiring(tmp_path)
    run = (tmp_path / "run").resolve()

    await goaly_run(
        host_registry(fakes, FakeContainer(fakes, solve=False), tmp_path / "locks"), run
    )

    graded = list(read_jsonl(run / "results.jsonl", TaskResult))
    assert len(graded) == len(TRAIN)
    assert all(r.grade.score == 0 for r in graded)
    assert fakes.grader.graded == []
