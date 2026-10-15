"""Tests for goaly.bench.effisql.harness: the Effi-SQL factories, through `goaly run`.

The runs here are the FakeLLM dry run: synthetic tasks, a scripted model, fake solving
instances and a fake grader, wired through the real CLI, runner and grade phase.
"""

import json
import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from goaly import cli as goaly_cli
from goaly.bench.effisql.data import (
    DATASET_REVISION,
    DATASET_SHA256,
    RUNTIME_FILE,
    EffiSqlInstance,
)
from goaly.bench.effisql.grading import Mode
from goaly.bench.effisql.grading_pool import Priority
from goaly.bench.effisql.harness import (  # Private helpers.
    PACK_DIR,
    EffiSqlPaths,
    EffiSqlWiring,
    PackRunConfig,
    RunCallLog,
    classify_error,
    context_factory,
    grading_priority,
    load_pack,
    model_settings,
    open_grading,
    open_harness,
    recorded_priority,
)
from goaly.bench.effisql.pg_instances import SOLVING
from goaly.bench.effisql.splits import load_splits
from goaly.bench.effisql.sql_env import SolvingPool
from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.llm_call_pb2 import CALL_PURPOSE_EXEC, LlmCallRecord
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import (
    TASK_ERROR_TYPE_ENVIRONMENT,
    TASK_ERROR_TYPE_LLM,
    TASK_ERROR_TYPE_POLICY_CRASH,
    TaskResult,
)
from goaly.contracts.gen.task_pb2 import SPLIT_EXTERNAL, SPLIT_HELDOUT, SPLIT_TRAIN, SPLIT_VAL
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core import cli
from goaly.harness.core.budgets import Budgets
from goaly.harness.core.jsonl_writer import JsonlWriter
from goaly.harness.core.llm import CallTags
from goaly.harness.core.llm_errors import (
    ClientError,
    ConnectionFailed,
    ContextOverflow,
    InvalidResponse,
    ModelNotAllowed,
    RateLimited,
    ServerError,
    TimedOut,
)
from goaly.harness.core.llm_models import EXECUTOR_MODEL, ModelParams
from goaly.harness.core.manifest import read_manifest, write_manifest
from goaly.harness.core.registry import GradingOptions, HarnessOptions, Registry
from goaly.harness.core.runner import RunFiles, RunIdentity
from tests.bench.effisql.fake_run import OFFICIAL_FORM, FakeClient, FakeWiring, fake_wiring
from tests.bench.effisql.synthetic import task

pytestmark = pytest.mark.anyio

_PACK_MODULES = "goaly.harness.packs"
VAL_ID = load_splits().val[0]


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """cli.main configures logging for the process: put it back after each test."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


def registry(fakes: FakeWiring) -> Registry:
    registry = Registry()
    registry.add("effisql", "v0", lambda options: open_harness(options, fakes.wiring))
    registry.add_benchmark("effisql", lambda options: open_grading(options, fakes.wiring))
    return registry


def goaly_run(fakes: FakeWiring, *args: str) -> int:
    return cli.main(
        ["run", "--bench", "effisql", *args], registry(fakes), budgets_file=goaly_cli.BUDGETS_FILE
    )


def results(run: Path) -> list[TaskResult]:
    return list(read_jsonl(run / "results.jsonl", TaskResult))


class TestPack:
    def test_the_pack_config_loads_into_core_types(self) -> None:
        config, prompt = load_pack(PACK_DIR)

        assert config.model == ModelParams(
            enable_thinking=True, temperature=0.6, top_p=0.95, top_k=20, max_tokens=16000
        )
        assert config.budgets == Budgets(steps=30, wall_clock_s=600)
        assert config.pack["execute_sql"] == {
            "statement_timeout_s": 120,
            "rows_shown": 20,
            "value_chars": 200,
        }
        assert "compute its result from the tables" in prompt

    def test_an_unknown_section_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text(
            "model: {}\nbudgets: {steps: 1, wall_clock_s: 1}\nx: 1\n"
        )
        (tmp_path / "prompt.md").write_text("p")

        with pytest.raises(ValidationError):
            load_pack(tmp_path)

    def test_importing_the_composition_root_imports_no_pack_code(self) -> None:
        code = (
            "import sys, goaly.cli; "
            f"print(sorted(m for m in sys.modules if m.startswith({_PACK_MODULES!r})))"
        )
        # A fixed command: this interpreter running a constant string.
        out = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
        )

        assert out.stdout.strip() == "[]"


class TestPieces:
    @pytest.mark.parametrize("error", [RateLimited, ServerError, TimedOut, ConnectionFailed])
    def test_a_gateway_outage_is_an_llm_error(self, error: type[Exception]) -> None:
        assert classify_error(error("down")) == TASK_ERROR_TYPE_LLM

    @pytest.mark.parametrize(
        "error", [ContextOverflow, InvalidResponse, ClientError, ModelNotAllowed, ValueError]
    )
    def test_other_failures_end_the_task_as_a_crash(self, error: type[Exception]) -> None:
        assert classify_error(error("bad")) == TASK_ERROR_TYPE_POLICY_CRASH

    def test_an_environment_failure_stays_the_environments(self) -> None:
        assert classify_error(EnvironmentFailure("pg")) == TASK_ERROR_TYPE_ENVIRONMENT
        group = ExceptionGroup("g", [RateLimited("down")])
        assert classify_error(group) == TASK_ERROR_TYPE_LLM

    @pytest.mark.parametrize(
        ("split", "phase", "priority"),
        [
            (SPLIT_TRAIN, "baseline", Priority.BASELINE),
            (SPLIT_HELDOUT, "baseline", Priority.BASELINE_HELDOUT),
            (SPLIT_VAL, "rsi", Priority.RSI),
            (SPLIT_HELDOUT, "final", Priority.FINAL),
            (None, None, Priority.RSI),
        ],
    )
    def test_grading_priority(
        self, split: int | None, phase: str | None, priority: Priority
    ) -> None:
        assert grading_priority(split, phase) is priority  # type: ignore[arg-type]

    async def test_each_task_binds_the_executor_model_to_itself(self) -> None:
        client = FakeClient()
        config = PackRunConfig(
            model=ModelParams(temperature=0.6), budgets=Budgets(steps=3, wall_clock_s=9)
        )
        factory = context_factory(client, config)
        inst = EffiSqlInstance.model_validate(task(id=5))

        async with factory(inst, 2, RunIdentity("run-1", "effisql", SPLIT_TRAIN)) as ctx:
            assert ctx.budgets == Budgets(steps=3, wall_clock_s=9)

        model, tags = client.bound[0]
        assert model == EXECUTOR_MODEL
        assert (tags.purpose, tags.run_id, tags.instance_id, tags.seed) == (
            CALL_PURPOSE_EXEC,
            "run-1",
            "5",
            2,
        )

    async def test_the_call_log_writes_records_and_never_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "llm_calls.jsonl"
        writer = JsonlWriter[LlmCallRecord](path)
        log = RunCallLog(writer)
        async with writer:
            log.record(LlmCallRecord(call_id="a"))
        log.record(LlmCallRecord(call_id="after-close"))  # Logged and dropped.

        assert [r.call_id for r in read_jsonl(path, LlmCallRecord)] == ["a"]


class TestDryRun:
    def test_one_process_run_solves_grades_and_writes_official_records(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fakes = fake_wiring(tmp_path)
        run = tmp_path / "run"

        status = goaly_run(
            fakes, "--split", "train", "--only", "1", "3", "--budget-phase", "baseline",
            "--out", str(run),
        )  # fmt: skip

        assert status == 0, capsys.readouterr().err
        assert fakes.solving.preflights == 1
        assert sorted(r.instance_id for r in results(run)) == ["1", "3"]
        assert all(r.grade.score == 0.5 for r in results(run))
        assert sorted(fakes.grader.graded) == [
            ("1", OFFICIAL_FORM, False, Mode.DEV, Priority.BASELINE),
            ("3", OFFICIAL_FORM, False, Mode.DEV, Priority.BASELINE),
        ]
        official = [
            json.loads(line)
            for line in (run / "official" / "seed-0.jsonl").read_text().splitlines()
        ]
        assert [record["id"] for record in official] == [1, 3]
        assert all(
            record["prediction"].startswith(f"```sql\n{OFFICIAL_FORM};") for record in official
        )
        assert "optimized_sql" not in official[0]
        manifest = read_manifest(run / "manifest.json")
        assert manifest.dataset_sha256 == DATASET_SHA256
        assert manifest.dataset_revision == DATASET_REVISION
        assert dict(manifest.environment_settings) == {
            "grading_mode": "dev",
            "grading_priority": "baseline",
        }
        assert [m.model for m in manifest.models] == [EXECUTOR_MODEL]
        assert manifest.models[0].max_tokens == 16000
        assert fakes.shares == [{EXECUTOR_MODEL: 40}]
        assert fakes.grader.closed

    def test_two_phases_grade_with_no_pack_code_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fakes = fake_wiring(tmp_path)
        run = tmp_path / "run"
        solved = goaly_run(
            fakes, "--phase", "solve", "--split", "val", "--only", VAL_ID, "--budget-phase", "rsi",
            "--out", str(run),
        )  # fmt: skip
        assert solved == 0, capsys.readouterr().err
        assert [r.HasField("grade") for r in results(run)] == [False]
        assert fakes.grader.graded == []
        for name in [m for m in sys.modules if m.startswith(_PACK_MODULES)]:
            monkeypatch.delitem(sys.modules, name)

        graded = goaly_run(fakes, "--phase", "grade", "--out", str(run))

        assert graded == 0, capsys.readouterr().err
        assert not [m for m in sys.modules if m.startswith(_PACK_MODULES)]
        assert [r.HasField("grade") for r in results(run)] == [False, True]
        assert fakes.grader.graded == [(VAL_ID, OFFICIAL_FORM, False, Mode.DEV, Priority.RSI)]

    def test_a_held_out_run_grades_in_official_mode(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fakes = fake_wiring(tmp_path)
        run = tmp_path / "run"
        heldout = load_splits().heldout[0]

        status = goaly_run(
            fakes, "--split", "heldout", "--only", heldout, "--budget-phase", "baseline",
            "--out", str(run),
        )  # fmt: skip

        assert status == 0, capsys.readouterr().err
        assert fakes.grader.graded[0][3:] == (Mode.OFFICIAL, Priority.BASELINE_HELDOUT)

    def test_the_grade_phase_refuses_a_manifest_that_names_another_mode(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fakes = fake_wiring(tmp_path)
        run = tmp_path / "run"
        goaly_run(
            fakes, "--phase", "solve", "--split", "val", "--only", VAL_ID, "--budget-phase", "rsi",
            "--out", str(run),
        )  # fmt: skip
        manifest = run / "manifest.json"
        text = manifest.read_text().replace('"grading_mode": "dev"', '"grading_mode": "official"')
        manifest.write_text(text)

        status = goaly_run(fakes, "--phase", "grade", "--out", str(run))

        assert status == 2
        assert "grading mode 'official'" in capsys.readouterr().err
        assert fakes.grader.graded == []

    def test_an_organizers_file_is_solved_but_not_graded(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fakes = fake_wiring(tmp_path)
        run = tmp_path / "run"
        organizers = tmp_path / "heldout.jsonl"
        organizers.write_text(json.dumps(task(id=901)) + "\n")

        status = goaly_run(
            fakes, "--instances", str(organizers), "--phase", "solve", "--budget-phase", "final",
            "--out", str(run),
        )  # fmt: skip

        assert status == 0, capsys.readouterr().err
        official = [
            json.loads(line)
            for line in (run / "official" / "seed-0.jsonl").read_text().splitlines()
        ]
        assert [record["id"] for record in official] == [901]
        assert "optimized_sql" not in official[0]
        manifest = read_manifest(run / "manifest.json")
        assert manifest.environment_settings["grading_mode"] == "none"
        assert manifest.dataset_revision == ""
        assert fakes.grader.graded == []


class TestRealWiring:
    async def test_the_real_client_reads_the_settings_and_fills_every_share(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GOALY_API_KEY", "test-key-not-real")
        monkeypatch.setenv("GOALY_BASE_URL", "http://127.0.0.1:9/v1")
        recorder = RunCallLog(JsonlWriter[LlmCallRecord](tmp_path / "calls.jsonl"))

        client = await EffiSqlWiring().llm(recorder, {EXECUTOR_MODEL: 7})

        async with client as opened:
            bound = opened.bind(model=EXECUTOR_MODEL, tags=CallTags(purpose=CALL_PURPOSE_EXEC))
            assert bound is not None

    def test_the_default_paths_are_the_hosts(self) -> None:
        paths = EffiSqlPaths.default()

        assert paths.tasks_file.name == RUNTIME_FILE
        assert paths.snapshot.name == "schema_snapshot.json"
        assert paths.plans.name == "plans.json"
        assert paths.grading_cache_root.name == "grader"

    def testmodel_settings_leave_out_what_the_profile_does_not_set(self) -> None:
        settings = model_settings(ModelParams(enable_thinking=False))

        assert settings.model == EXECUTOR_MODEL
        assert not settings.thinking
        assert settings.temperature == 0.6  # The profile's default.

    @pytest.mark.parametrize(
        ("recorded", "priority"),
        [
            ("final", Priority.FINAL),
            ("setup", Priority.RSI),
            ("nonsense", Priority.RSI),
            ("", Priority.RSI),
        ],
    )
    def test_therecorded_priority(self, recorded: str, priority: Priority) -> None:
        manifest = RunManifest(environment_settings={"grading_priority": recorded})

        assert recorded_priority(manifest) is priority

    async def test_grading_refuses_an_external_run(self, tmp_path: Path) -> None:
        fakes = fake_wiring(tmp_path)
        files = RunFiles(tmp_path / "run")
        files.root.mkdir()
        write_manifest(files.manifest, RunManifest(run_id="r", split=SPLIT_EXTERNAL))

        with pytest.raises(ValueError, match="SPLIT_EXTERNAL is not graded"):
            async with open_grading(GradingOptions(files=files), fakes.wiring):
                pass


class TestInContainer:
    def test_the_container_reads_its_inputs_directory(self, tmp_path: Path) -> None:
        paths = EffiSqlWiring.in_container(tmp_path / "inputs").paths()

        assert paths.tasks_file == tmp_path / "inputs" / "tasks.jsonl"
        assert paths.snapshot == tmp_path / "inputs" / "schema_snapshot.json"
        assert paths.plans == tmp_path / "inputs" / "plans.json"

    async def test_the_container_reaches_the_solving_instances_as_the_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GOALY_PG_HOST", "127.0.0.1")
        monkeypatch.setenv("GOALY_PG_PORTS", ",".join(str(i.port) for i in SOLVING))
        monkeypatch.setenv("GOALY_PG_AGENT_PASSWORD", "agent-secret")

        pool = await EffiSqlWiring.in_container(tmp_path).solving()

        assert isinstance(pool, SolvingPool)
        assert sorted(pool.load()) == sorted(i.name for i in SOLVING)

    @pytest.mark.parametrize(
        ("ports", "message"),
        [(None, "GOALY_PG_PORTS"), ("1,x", "invalid literal"), ("1", "unknown")],
    )
    async def test_missing_or_wrong_settings_fail_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ports: str | None, message: str
    ) -> None:
        monkeypatch.setenv("GOALY_PG_HOST", "127.0.0.1")
        monkeypatch.setenv("GOALY_PG_AGENT_PASSWORD", "agent-secret")
        if ports is None:
            monkeypatch.delenv("GOALY_PG_PORTS", raising=False)
        else:
            monkeypatch.setenv("GOALY_PG_PORTS", ports)

        with pytest.raises(EnvironmentFailure, match=message):
            await EffiSqlWiring.in_container(tmp_path).solving()

    def test_the_container_grades_nothing(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="grades nothing"):
            EffiSqlWiring.in_container(tmp_path).grader(tmp_path, None)

    async def test_the_container_needs_its_inputs(self, tmp_path: Path) -> None:
        options = HarnessOptions(
            files=RunFiles(tmp_path), instances_file=None, llm_shares={}, in_container=True
        )

        with pytest.raises(ValueError, match="--inputs"):
            async with open_harness(options):
                pass
