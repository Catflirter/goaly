"""Tests for goaly.contracts.interfaces.

- Typing tests: each fake in tests/fakes.py is assigned to a variable typed with its Protocol,
  so ``make typecheck`` proves that it conforms, and is then exercised at runtime.
- The gRPC-ready rule (acceptance criterion 4): Protocol methods take and return only contract
  messages and plain values. mypy --strict already requires every annotation to be present.
"""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import NoneType
from typing import Any, Protocol, TypeVar, assert_type, get_args, get_origin, get_type_hints

import pytest
from google.protobuf.message import Message
from pydantic import BaseModel

from goaly.contracts import interfaces
from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.environment_pb2 import ShellResult, SqlResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    SPLIT_TRAIN,
    SPLIT_VAL,
    BirdPrediction,
    BirdTaskInput,
    Grade,
    Prediction,
    SwePrediction,
    SweTaskInput,
    TaskInput,
)
from goaly.contracts.gen.trajectory_pb2 import ToolResult
from goaly.contracts.interfaces import (
    Benchmark,
    Environment,
    JsonValue,
    Policy,
    ShellEnvironment,
    SqlEnvironment,
    Tool,
)
from goaly.contracts.jsonl import to_json_line
from tests.fakes import (
    COMMAND_NOT_FOUND,
    TOY_INSTANCES,
    DoNothingPolicy,
    EchoArgs,
    EchoTool,
    FakeEnvironment,
    FakeShellEnvironment,
    FakeSqlEnvironment,
    ToyBenchmark,
    ToyInstance,
)

CLASSES = [
    value
    for value in vars(interfaces).values()
    if isinstance(value, type) and value.__module__ == interfaces.__name__
]
PROTOCOLS = [cls for cls in CLASSES if vars(cls).get("_is_protocol", False)]

# The Python types of protobuf's scalars. An enum value's annotation, such as Split.ValueType, is
# int at runtime.
SCALARS = (bool, bytes, float, int, str)

# The one documented exception (spec §3.3, and Tool's docstring): a tool's arguments are a
# pydantic model, so ArgsT is bound to BaseModel and args_model returns the model class.
DOCUMENTED_EXCEPTIONS = {
    "Tool": ["Tool[ArgsT] is bound to BaseModel", "Tool.args_model returns type[ArgsT]"],
}


def describe(hint: object) -> str:
    return hint.__qualname__ if isinstance(hint, type) else repr(hint)


def is_plain(hint: object, type_params: Sequence[object]) -> bool:
    """Whether ``hint`` is a type parameter, a contract message, a builtin scalar, None or JSON."""
    # dict[str, JsonValue] is a JSON object: the branch of JsonValue that to_official returns.
    if hint in type_params or hint in (*SCALARS, NoneType, JsonValue, dict[str, JsonValue]):
        return True
    return (
        isinstance(hint, type)
        and issubclass(hint, Message)
        and hint.__module__.startswith("goaly.contracts.gen.")
    )


def is_allowed(hint: object, type_params: Sequence[object]) -> bool:
    """Whether ``hint`` is plain, or a Sequence or AbstractAsyncContextManager of plain types."""
    if get_origin(hint) in (Sequence, AbstractAsyncContextManager):
        return all(is_plain(arg, type_params) for arg in get_args(hint))
    return is_plain(hint, type_params)


def annotations(protocol: type) -> Iterator[tuple[str, object]]:
    """Yield each annotation of ``protocol``'s own members, after a note of where it is."""
    owner = protocol.__name__
    members = vars(protocol)["__protocol_attrs__"]  # What typing.get_protocol_members() reads.
    for name, value in vars(protocol).items():
        if name not in members:
            continue
        function = value.fget if isinstance(value, property) else value
        assert inspect.isfunction(function), f"{owner}.{name} is neither a method nor a property"
        for parameter, hint in get_type_hints(function).items():
            role = "returns" if parameter == "return" else f"takes {parameter}:"
            yield f"{owner}.{name} {role}", hint
    for name, hint in inspect.get_annotations(protocol).items():
        yield f"{owner}.{name} is", hint


def violations(protocol: type) -> list[str]:
    """Describe each type in ``protocol``'s interface that breaks the gRPC-ready rule."""
    type_params = protocol.__type_params__
    found = [
        f"{protocol.__name__}[{param.__name__}] is bound to {describe(param.__bound__)}"
        for param in type_params
        if isinstance(param, TypeVar) and param.__bound__ not in (None, *PROTOCOLS)
    ]
    found += [
        f"{where} {describe(hint)}"
        for where, hint in annotations(protocol)
        if not is_allowed(hint, type_params)
    ]
    return found


def test_the_module_defines_the_six_protocols_and_nothing_else() -> None:
    assert PROTOCOLS == CLASSES
    assert sorted(cls.__name__ for cls in PROTOCOLS) == [
        "Benchmark",
        "Environment",
        "Policy",
        "ShellEnvironment",
        "SqlEnvironment",
        "Tool",
    ]


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=lambda protocol: protocol.__name__)
def test_protocols_take_and_return_only_contract_messages_and_plain_values(protocol: type) -> None:
    assert violations(protocol) == DOCUMENTED_EXCEPTIONS.get(protocol.__name__, [])


class Leaky[ModelT: BaseModel](Protocol):
    """Breaks the gRPC-ready rule in each way the rule test must catch."""

    limit: Any

    async def run(self, model: ModelT, options: dict[str, str]) -> Any: ...

    def rows(self) -> Sequence[Sequence[str]]: ...

    @property
    def path(self) -> Path: ...


def test_the_rule_names_each_type_that_breaks_it() -> None:
    assert violations(Leaky) == [
        "Leaky[ModelT] is bound to BaseModel",
        "Leaky.run takes options: dict[str, str]",
        "Leaky.run returns Any",
        "Leaky.rows returns collections.abc.Sequence[collections.abc.Sequence[str]]",
        "Leaky.path returns Path",
        "Leaky.limit is Any",
    ]


def test_fake_environment_closes_and_closing_again_does_nothing() -> None:
    fake = FakeEnvironment()
    env: Environment = fake

    async def close_twice() -> None:
        await env.close()
        await env.close()

    asyncio.run(close_twice())

    assert fake.closed


def test_fake_sql_environment_answers_from_its_canned_results() -> None:
    canned = SqlResult(columns=["n"], row_count=1)
    fake = FakeSqlEnvironment({"SELECT 1": canned})
    env: SqlEnvironment = fake

    async def scenario() -> tuple[SqlResult, SqlResult]:
        return (
            await env.execute("SELECT 1", timeout_s=5.0),
            await env.execute("SELEC 1", timeout_s=1.0),
        )

    known, unknown = asyncio.run(scenario())

    assert known == canned
    assert known is not canned
    assert unknown == SqlResult(error="no canned result for 'SELEC 1'")
    assert fake.executed == [("SELECT 1", 5.0), ("SELEC 1", 1.0)]


def test_a_closed_fake_sql_environment_refuses_statements() -> None:
    fake = FakeSqlEnvironment()
    env: SqlEnvironment = fake

    async def scenario() -> None:
        await env.close()
        await env.execute("SELECT 1", timeout_s=1.0)

    with pytest.raises(RuntimeError, match="the environment is closed"):
        asyncio.run(scenario())
    assert fake.executed == []


def test_fake_shell_environment_runs_canned_commands() -> None:
    canned = ShellResult(stdout="1 passed\n", elapsed_s=0.5)
    fake = FakeShellEnvironment({"pytest -q": canned})
    env: ShellEnvironment = fake

    async def scenario() -> tuple[ShellResult, ShellResult]:
        return (
            await env.exec("pytest -q", timeout_s=60.0),
            await env.exec("pytset -q", timeout_s=1.0),
        )

    known, unknown = asyncio.run(scenario())

    assert known == canned
    assert known is not canned
    assert unknown == ShellResult(
        exit_code=COMMAND_NOT_FOUND, stderr="no canned result for 'pytset -q'"
    )
    assert fake.commands == [("pytest -q", 60.0), ("pytset -q", 1.0)]


def test_fake_shell_environment_keeps_files_in_memory() -> None:
    fake = FakeShellEnvironment(files={"/testbed/a.py": "a = 1\n"})
    env: ShellEnvironment = fake

    async def scenario() -> tuple[str, str]:
        await env.write_file("/testbed/b.py", "b = 2\n")
        await env.write_file("/testbed/a.py", "a = 3\n")
        return await env.read_file("/testbed/a.py"), await env.read_file("/testbed/b.py")

    assert asyncio.run(scenario()) == ("a = 3\n", "b = 2\n")
    assert fake.files == {"/testbed/a.py": "a = 3\n", "/testbed/b.py": "b = 2\n"}


def test_fake_shell_environment_has_no_file_it_was_not_given() -> None:
    env: ShellEnvironment = FakeShellEnvironment()

    with pytest.raises(FileNotFoundError) as raised:
        asyncio.run(env.read_file("/testbed/missing.py"))
    assert raised.value.args == ("/testbed/missing.py",)


SHELL_CALLS: dict[str, Callable[[ShellEnvironment], Awaitable[object]]] = {
    "exec": lambda env: env.exec("true", timeout_s=1.0),
    "read_file": lambda env: env.read_file("notes.txt"),
    "write_file": lambda env: env.write_file("notes.txt", "new"),
}


@pytest.mark.parametrize("call", SHELL_CALLS.values(), ids=list(SHELL_CALLS))
def test_a_closed_fake_shell_environment_refuses_every_call(
    call: Callable[[ShellEnvironment], Awaitable[object]],
) -> None:
    fake = FakeShellEnvironment(files={"notes.txt": "old"})
    env: ShellEnvironment = fake

    async def scenario() -> None:
        await env.close()
        await call(env)

    with pytest.raises(RuntimeError, match="the environment is closed"):
        asyncio.run(scenario())
    assert (fake.commands, fake.files) == ([], {"notes.txt": "old"})


def test_fake_environments_raise_a_canned_environment_failure() -> None:
    died = EnvironmentFailure("the container died")
    sql_env: SqlEnvironment = FakeSqlEnvironment({"SELECT 1": died})
    shell_env: ShellEnvironment = FakeShellEnvironment({"ls": died})

    with pytest.raises(EnvironmentFailure) as raised_by_sql:
        asyncio.run(sql_env.execute("SELECT 1", timeout_s=1.0))
    with pytest.raises(EnvironmentFailure) as raised_by_shell:
        asyncio.run(shell_env.exec("ls", timeout_s=1.0))

    assert raised_by_sql.value is died
    assert raised_by_shell.value is died


def test_echo_tool_answers_with_its_text_in_any_environment() -> None:
    # EnvT is contravariant, so a tool for any environment fits a SQL pack and a shell pack.
    sql_tool: Tool[EchoArgs, SqlEnvironment] = EchoTool()
    shell_tool: Tool[EchoArgs, ShellEnvironment] = EchoTool()

    args = sql_tool.args_model.model_validate_json('{"text": "hello"}')

    async def scenario() -> tuple[ToolResult, ToolResult]:
        return (
            await sql_tool.run(args, FakeSqlEnvironment()),
            await shell_tool.run(args, FakeShellEnvironment()),
        )

    assert_type(args, EchoArgs)
    assert asyncio.run(scenario()) == (ToolResult(content="hello"), ToolResult(content="hello"))
    assert (sql_tool.name, sql_tool.description) == ("echo", "Reply with the given text.")


class StandInContext:
    """Stands in for harness core's AgentContext, which contracts cannot import."""


def test_do_nothing_policy_submits_the_issue_sql_of_a_bird_task() -> None:
    fake = DoNothingPolicy()
    # EnvT and CtxT are contravariant, so a policy for any environment and context fits here.
    policy: Policy[SqlEnvironment, StandInContext] = fake
    task = TaskInput(instance_id="t", bird=BirdTaskInput(issue_sql=["SELECT 1", "SELECT 2"]))

    prediction = asyncio.run(policy.solve(task, FakeSqlEnvironment(), StandInContext()))

    assert prediction == Prediction(
        instance_id="t", bird=BirdPrediction(sqls=["SELECT 1", "SELECT 2"])
    )
    assert fake.tasks == [task]


def test_do_nothing_policy_submits_no_patch_for_a_swe_task() -> None:
    policy: Policy[ShellEnvironment, StandInContext] = DoNothingPolicy()
    task = TaskInput(instance_id="t", swe=SweTaskInput(repo="astropy/astropy"))

    prediction = asyncio.run(policy.solve(task, FakeShellEnvironment(), StandInContext()))

    assert prediction == Prediction(instance_id="t", swe=SwePrediction(patch=""))
    assert prediction.WhichOneof("benchmark") == "swe"


def test_toy_benchmark_loads_the_instances_of_a_split() -> None:
    # EnvT is covariant, so a benchmark that provisions fakes fits a SQL benchmark's slot.
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()

    train = benchmark.load_instances(SPLIT_TRAIN)

    assert_type(train, Sequence[ToyInstance])
    assert [inst.instance_id for inst in train] == ["toy-1", "toy-2"]
    assert [inst.instance_id for inst in benchmark.load_instances(SPLIT_VAL)] == ["toy-3"]
    assert benchmark.name == "toy"


def test_toy_benchmark_shows_the_agent_no_gold_answer() -> None:
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()
    inst = TOY_INSTANCES[0]

    task = benchmark.agent_input(inst)

    assert task == TaskInput(
        instance_id="toy-1",
        bird=BirdTaskInput(db_id="toy", query=inst.query, issue_sql=[inst.issue_sql]),
    )
    assert inst.gold_sql not in to_json_line(task)


def test_toy_benchmark_closes_the_environment_it_provisions() -> None:
    fake = ToyBenchmark()
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = fake
    inst = TOY_INSTANCES[0]

    async def scenario() -> tuple[SqlResult, SqlResult]:
        async with benchmark.provision(inst) as env:
            return (
                await env.execute(inst.gold_sql, timeout_s=1.0),
                await env.execute(inst.issue_sql, timeout_s=1.0),
            )

    gold, issue = asyncio.run(scenario())

    assert gold == SqlResult()
    assert issue.error
    assert [env.closed for env in fake.environments] == [True]


@pytest.mark.parametrize(
    ("sqls", "correct"),
    [
        (["SELECT count(*) FROM users"], True),
        (["SELECT 1", "SELECT count(*) FROM users"], True),
        (["SELECT count(*) FROM users", "SELECT 1"], False),
        ([], False),
    ],
)
def test_toy_benchmark_grades_the_last_statement(sqls: list[str], correct: bool) -> None:
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()
    prediction = Prediction(instance_id="toy-1", bird=BirdPrediction(sqls=sqls))

    grade = asyncio.run(benchmark.grade(TOY_INSTANCES[0], prediction))

    assert grade == Grade(
        instance_id="toy-1", score=float(correct), correct=correct, status=GRADE_STATUS_GRADED
    )


def test_toy_benchmark_writes_the_official_bird_format() -> None:
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()
    prediction = Prediction(instance_id="toy-1", bird=BirdPrediction(sqls=["SELECT 1"]))

    record = benchmark.to_official(TOY_INSTANCES[0], prediction)

    assert json.loads(json.dumps(record)) == {"instance_id": "toy-1", "pred_sqls": ["SELECT 1"]}


async def solve_and_grade[InstanceT, EnvT: Environment, CtxT](
    benchmark: Benchmark[InstanceT, EnvT], policy: Policy[EnvT, CtxT], inst: InstanceT, ctx: CtxT
) -> Grade:
    """Solve and grade one task as the runner will, through the Protocols alone."""
    async with benchmark.provision(inst) as env:
        prediction = await policy.solve(benchmark.agent_input(inst), env, ctx)
    return await benchmark.grade(inst, prediction)


def test_the_fakes_run_a_split_end_to_end() -> None:
    benchmark = ToyBenchmark()
    policy = DoNothingPolicy()

    async def run_split() -> list[Grade]:
        return [
            await solve_and_grade(benchmark, policy, inst, StandInContext())
            for inst in benchmark.load_instances(SPLIT_TRAIN)
        ]

    grades = asyncio.run(run_split())

    assert [(grade.instance_id, grade.correct) for grade in grades] == [
        ("toy-1", False),
        ("toy-2", True),
    ]
    assert [task.instance_id for task in policy.tasks] == ["toy-1", "toy-2"]
    assert [env.closed for env in benchmark.environments] == [True, True]
