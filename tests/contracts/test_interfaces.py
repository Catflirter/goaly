"""Tests for goaly.contracts.interfaces.

- Typing tests: each fake in tests/fakes.py is assigned to a variable typed with its Protocol,
  so ``make typecheck`` proves that it conforms, and is then exercised at runtime.
- The gRPC-ready rule (acceptance criterion 4): every annotation of every Protocol member is a
  contract message or a plain value. That covers inherited members, overloads, and property
  setters and deleters, as well as type parameters' bounds and constraints. One leaky Protocol
  per kind proves that the guard catches each.
"""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Iterator, Sequence, Sized
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import NoneType
from typing import (
    Any,
    Protocol,
    SupportsAbs,
    TypeVar,
    assert_type,
    get_args,
    get_origin,
    get_overloads,
    get_type_hints,
    overload,
)

import pytest
from google.protobuf.message import Message
from pydantic import BaseModel

from goaly.contracts import interfaces
from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.environment_pb2 import ShellResult, SqlResult
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
    SPLIT_VAL,
    EffiSqlGrade,
    EffiSqlPrediction,
    EffiSqlTaskInput,
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
    Instance,
    JsonValue,
    Policy,
    ShellEnvironment,
    SqlEnvironment,
    Tool,
)
from goaly.contracts.jsonl import to_json_line
from tests.fakes import (
    COMMAND_NOT_FOUND,
    STATEMENT_TIMEOUT,
    TOY_BASE_RESULT,
    TOY_INSTANCES,
    DoNothingPolicy,
    EchoArgs,
    EchoTool,
    FakeEnvironment,
    FakeShellEnvironment,
    FakeSqlEnvironment,
    SubmitArgs,
    SubmitTool,
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


def is_allowed_limit(limit: object) -> bool:
    """Whether a type parameter may be bound or constrained to ``limit``."""
    return limit in PROTOCOLS or is_plain(limit, ())


def members(protocol: type) -> Iterator[tuple[type, str]]:
    """Yield each member of ``protocol``, inherited or not, with the class that defines it."""
    names = vars(protocol)["__protocol_attrs__"]  # What typing.get_protocol_members() reads.
    found: set[str] = set()
    for base in protocol.__mro__:
        for name in (*vars(base), *inspect.get_annotations(base)):
            if name in names and name not in found:
                found.add(name)
                yield base, name
    assert found == names, f"{protocol.__name__}: no class defines {names - found}"


def overloads(owner: type, name: str) -> Sequence[Callable[..., object]]:
    """The @overload signatures of ``owner.name``, whether it has an implementation or not.

    typing.get_overloads finds them by module and qualified name. Overloads with no
    implementation leave only typing's unannotated placeholder in the class, so the lookup goes
    through a stand-in that carries the member's names.
    """

    def stand_in() -> None: ...

    stand_in.__module__ = owner.__module__
    stand_in.__qualname__ = f"{owner.__qualname__}.{name}"
    return get_overloads(stand_in)


def functions(owner: type, name: str) -> Iterator[tuple[str, object]]:
    """Yield each function that makes up the member ``owner.name``, with a label for it.

    A property gives its getter, setter and deleter. An overloaded method gives its overloads,
    which are its interface; its implementation is not.
    """
    value = vars(owner)[name]
    if isinstance(value, property):
        for part, function in (("", value.fget), (" setter", value.fset), (" deleter", value.fdel)):
            if function is not None:
                yield f"{name}{part}", function
    elif signatures := overloads(owner, name):
        for number, function in enumerate(signatures, start=1):
            yield f"{name} overload {number}", function
    else:
        yield name, value


def limit_violations(protocol: type) -> Iterator[str]:
    """Describe each bound or constraint of ``protocol``'s type parameters that breaks the rule."""
    for param in protocol.__type_params__:
        if not isinstance(param, TypeVar):
            continue  # A ParamSpec or TypeVarTuple shows in the annotations that use it.
        where = f"{protocol.__name__}[{param.__name__}]"
        if param.__bound__ is not None and not is_allowed_limit(param.__bound__):
            yield f"{where} is bound to {describe(param.__bound__)}"
        for constraint in param.__constraints__:
            if not is_allowed_limit(constraint):
                yield f"{where} is constrained to {describe(constraint)}"


def member_violations(protocol: type) -> Iterator[str]:
    """Describe each annotation of ``protocol``'s members that breaks the rule.

    A member defined in another of this module's Protocols is checked there. Any other member,
    even one inherited from outside, is checked against ``protocol``'s own type parameters, so
    a foreign type variable is reported. Every function needs a return annotation.
    """
    owner, type_params = protocol.__name__, protocol.__type_params__
    for definer, name in members(protocol):
        if definer is not protocol and definer in PROTOCOLS:
            continue
        if name not in vars(definer):  # An annotation-only data member.
            hint = inspect.get_annotations(definer)[name]
            if not is_allowed(hint, type_params):
                yield f"{owner}.{name} is {describe(hint)}"
            continue
        for label, function in functions(definer, name):
            if not inspect.isfunction(function):
                yield f"{owner}.{label} is neither a method nor a property"
                continue
            hints = get_type_hints(function)
            for parameter in list(inspect.signature(function).parameters)[1:]:  # Not self.
                if parameter not in hints:
                    yield f"{owner}.{label} takes {parameter} with no annotation"
                elif not is_allowed(hints[parameter], type_params):
                    yield f"{owner}.{label} takes {parameter}: {describe(hints[parameter])}"
            if "return" not in hints:
                yield f"{owner}.{label} has no return annotation"
            elif not is_allowed(hints["return"], type_params):
                yield f"{owner}.{label} returns {describe(hints['return'])}"


def violations(protocol: type) -> list[str]:
    """Describe each type in ``protocol``'s interface that breaks the gRPC-ready rule."""
    return [*limit_violations(protocol), *member_violations(protocol)]


def test_the_module_defines_the_seven_protocols_and_nothing_else() -> None:
    assert PROTOCOLS == CLASSES
    assert sorted(cls.__name__ for cls in PROTOCOLS) == [
        "Benchmark",
        "Environment",
        "Instance",
        "Policy",
        "ShellEnvironment",
        "SqlEnvironment",
        "Tool",
    ]


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=lambda protocol: protocol.__name__)
def test_protocols_take_and_return_only_contract_messages_and_plain_values(protocol: type) -> None:
    assert violations(protocol) == DOCUMENTED_EXCEPTIONS.get(protocol.__name__, [])


# One Protocol per way to break the gRPC-ready rule, each with the violations the rule must name.


class LeakyBound[ModelT: BaseModel](Protocol):
    def get(self) -> ModelT: ...


class LeakyConstraint[ValueT: (str, Path)](Protocol):
    def get(self) -> ValueT: ...


class LeakyParameter(Protocol):
    async def run(self, options: dict[str, str]) -> None: ...


class LeakyReturn(Protocol):
    def run(self) -> Any: ...


class LeakyNested(Protocol):
    def rows(self) -> Sequence[Sequence[str]]: ...


class LeakyGetter(Protocol):
    @property
    def path(self) -> Path: ...


class LeakySetter(Protocol):
    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, value: Path) -> None: ...


class LeakyDeleter(Protocol):
    @property
    def name(self) -> str: ...

    @name.deleter
    def name(self) -> Path: ...


class LeakyAttribute(Protocol):
    limit: Any


class LeakyOverload(Protocol):
    """Without an implementation, the class holds only typing's placeholder."""

    @overload
    def fetch(self, key: str) -> str: ...

    @overload
    def fetch(self, key: int) -> Path: ...


class LeakyImplementedOverload(Protocol):
    """The overloads are the interface; the implementation's wider types are not."""

    @overload
    def fetch(self, key: str) -> str: ...

    @overload
    def fetch(self, key: int) -> Path: ...

    def fetch(self, key: str | int) -> str | Path: ...


class LeakyForeignMember(SupportsAbs[Path], Protocol):
    """Inherits __abs__ from typing, which returns SupportsAbs's own type variable."""


class LeakyUnannotated(Sized, Protocol):
    """Inherits __len__ from collections.abc, which has no annotations at runtime."""


LEAKS: list[tuple[type, list[str]]] = [
    (LeakyBound, ["LeakyBound[ModelT] is bound to BaseModel"]),
    (LeakyConstraint, ["LeakyConstraint[ValueT] is constrained to Path"]),
    (LeakyParameter, ["LeakyParameter.run takes options: dict[str, str]"]),
    (LeakyReturn, ["LeakyReturn.run returns Any"]),
    (
        LeakyNested,
        ["LeakyNested.rows returns collections.abc.Sequence[collections.abc.Sequence[str]]"],
    ),
    (LeakyGetter, ["LeakyGetter.path returns Path"]),
    (LeakySetter, ["LeakySetter.name setter takes value: Path"]),
    (LeakyDeleter, ["LeakyDeleter.name deleter returns Path"]),
    (LeakyAttribute, ["LeakyAttribute.limit is Any"]),
    (LeakyOverload, ["LeakyOverload.fetch overload 2 returns Path"]),
    (LeakyImplementedOverload, ["LeakyImplementedOverload.fetch overload 2 returns Path"]),
    (LeakyForeignMember, ["LeakyForeignMember.__abs__ returns T"]),
    (LeakyUnannotated, ["LeakyUnannotated.__len__ has no return annotation"]),
]


@pytest.mark.parametrize(
    ("protocol", "expected"), LEAKS, ids=[protocol.__name__ for protocol, _ in LEAKS]
)
def test_the_rule_names_each_kind_of_leak(protocol: type, expected: list[str]) -> None:
    assert violations(protocol) == expected


def test_json_value_holds_only_the_json_types() -> None:
    # The rule allows JsonValue by name, so widening it would widen the rule.
    assert JsonValue.__value__ == (
        dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None
    )


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
    canned = ShellResult(stdout="1 passed\n", elapsed_s=0.001)
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


def test_fake_shell_environment_resolves_relative_paths_against_its_root() -> None:
    fake = FakeShellEnvironment(files={"src/a.py": "a = 1\n"})
    env: ShellEnvironment = fake

    async def scenario() -> tuple[str, str]:
        await env.write_file("/testbed/src/b.py", "b = 2\n")
        return await env.read_file("/testbed/src/a.py"), await env.read_file("src/../src/b.py")

    assert asyncio.run(scenario()) == ("a = 1\n", "b = 2\n")
    assert fake.files == {"/testbed/src/a.py": "a = 1\n", "/testbed/src/b.py": "b = 2\n"}


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
    assert (fake.commands, fake.files) == ([], {"/testbed/notes.txt": "old"})


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


def test_a_cancelled_fake_statement_stops_and_leaves_the_environment_usable() -> None:
    fake = FakeSqlEnvironment({"SELECT pg_sleep(60)": SqlResult(elapsed_s=60.0)})
    env: SqlEnvironment = fake

    async def cancel_midway() -> None:
        async with asyncio.timeout(0.01):
            await env.execute("SELECT pg_sleep(60)", timeout_s=120.0)

    async def scenario() -> SqlResult:
        with pytest.raises(TimeoutError):
            await cancel_midway()
        return await env.execute("SELEC 1", timeout_s=1.0)

    assert asyncio.run(scenario()).error == "no canned result for 'SELEC 1'"
    assert fake.stopped == ["SELECT pg_sleep(60)"]


def test_a_cancelled_fake_command_stops_and_leaves_the_environment_usable() -> None:
    fake = FakeShellEnvironment({"sleep 60": ShellResult(elapsed_s=60.0)})
    env: ShellEnvironment = fake

    async def cancel_midway() -> None:
        async with asyncio.timeout(0.01):
            await env.exec("sleep 60", timeout_s=120.0)

    async def scenario() -> ShellResult:
        with pytest.raises(TimeoutError):
            await cancel_midway()
        return await env.exec("slep 60", timeout_s=1.0)

    assert asyncio.run(scenario()).exit_code == COMMAND_NOT_FOUND
    assert fake.stopped == ["sleep 60"]


def test_fake_sql_environment_stops_a_statement_at_its_timeout() -> None:
    fake = FakeSqlEnvironment({"SELECT pg_sleep(60)": SqlResult(elapsed_s=60.0)})
    env: SqlEnvironment = fake

    result = asyncio.run(env.execute("SELECT pg_sleep(60)", timeout_s=0.01))

    assert result == SqlResult(error=STATEMENT_TIMEOUT, elapsed_s=0.01)
    assert fake.stopped == []


def test_fake_shell_environment_stops_a_command_at_its_timeout() -> None:
    fake = FakeShellEnvironment({"sleep 60": ShellResult(stdout="partial", elapsed_s=60.0)})
    env: ShellEnvironment = fake

    result = asyncio.run(env.exec("sleep 60", timeout_s=0.01))

    assert result == ShellResult(timed_out=True, elapsed_s=0.01)
    assert fake.stopped == []


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


def test_submit_tool_ends_the_episode_with_its_submission() -> None:
    tool: Tool[SubmitArgs, SqlEnvironment] = SubmitTool()
    args = tool.args_model.model_validate_json('{"sql": "SELECT 1"}')

    result = asyncio.run(tool.run(args, FakeSqlEnvironment()))

    assert result == ToolResult(
        content="Submitted.", submission=Prediction(effisql=EffiSqlPrediction(sql="SELECT 1"))
    )


class StandInContext:
    """Stands in for harness core's AgentContext, which contracts cannot import."""


def test_do_nothing_policy_submits_the_base_query_of_an_effisql_task() -> None:
    fake = DoNothingPolicy()
    # EnvT and CtxT are contravariant, so a policy for any environment and context fits here.
    policy: Policy[SqlEnvironment, StandInContext] = fake
    task = TaskInput(effisql=EffiSqlTaskInput(db="tpch", base_sql="SELECT 1"))

    prediction = asyncio.run(policy.solve(task, FakeSqlEnvironment(), StandInContext()))

    assert prediction == Prediction(effisql=EffiSqlPrediction(sql="SELECT 1"))
    assert fake.tasks == [task]


def test_do_nothing_policy_submits_no_patch_for_a_swe_task() -> None:
    policy: Policy[ShellEnvironment, StandInContext] = DoNothingPolicy()
    task = TaskInput(swe=SweTaskInput(repo="astropy/astropy"))

    prediction = asyncio.run(policy.solve(task, FakeShellEnvironment(), StandInContext()))

    assert prediction == Prediction(swe=SwePrediction(patch=""))
    assert prediction.WhichOneof("benchmark") == "swe"


def test_a_frozen_dataclass_with_an_instance_id_is_an_instance() -> None:
    inst: Instance = TOY_INSTANCES[0]

    assert inst.instance_id == "toy-1"


def test_toy_benchmark_loads_the_instances_of_a_split() -> None:
    # EnvT is covariant, so a benchmark that provisions fakes fits a SQL benchmark's slot.
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()

    train = benchmark.load_instances(SPLIT_TRAIN)

    assert_type(train, Sequence[ToyInstance])
    assert [inst.instance_id for inst in train] == ["toy-1", "toy-2"]
    assert [inst.instance_id for inst in benchmark.load_instances(SPLIT_VAL)] == ["toy-3"]
    assert benchmark.name == "toy"


def test_toy_benchmark_loads_the_organizers_tasks_for_the_external_split() -> None:
    external = [ToyInstance("ext-1", SPLIT_EXTERNAL, base_sql="SELECT 1", gold_sql=None)]
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark(external=external)

    assert benchmark.load_instances(SPLIT_EXTERNAL) == external


def test_toy_benchmark_without_a_task_file_refuses_the_external_split() -> None:
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()

    with pytest.raises(ValueError, match="without a task file"):
        benchmark.load_instances(SPLIT_EXTERNAL)


def test_toy_benchmark_shows_the_agent_no_gold_answer() -> None:
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()
    inst = TOY_INSTANCES[0]

    task = benchmark.agent_input(inst)

    assert task == TaskInput(effisql=EffiSqlTaskInput(db="toy", base_sql=inst.base_sql))
    assert inst.gold_sql is not None
    assert inst.gold_sql not in to_json_line(task)


def test_toy_benchmark_solves_in_an_environment_that_reveals_no_gold() -> None:
    fake = ToyBenchmark()
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = fake
    inst = TOY_INSTANCES[0]
    gold_sql = inst.gold_sql
    assert gold_sql is not None

    async def scenario() -> tuple[SqlResult, SqlResult]:
        async with benchmark.provision(inst) as env:
            return (
                await env.execute(inst.base_sql, timeout_s=1.0),
                await env.execute(gold_sql, timeout_s=1.0),
            )

    base, gold = asyncio.run(scenario())

    assert base == TOY_BASE_RESULT
    # The gold answer gets what any query the environment does not know gets.
    assert gold == SqlResult(error=f"no canned result for {gold_sql!r}")
    assert list(fake.environments[0].results) == [inst.base_sql]
    assert [env.closed for env in fake.environments] == [True]


def test_toy_benchmark_closes_the_environment_and_lets_an_exception_through() -> None:
    fake = ToyBenchmark()
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = fake

    async def crash() -> None:
        async with benchmark.provision(TOY_INSTANCES[0]):
            raise LookupError("the policy crashed")

    with pytest.raises(LookupError, match="the policy crashed"):
        asyncio.run(crash())
    assert [env.closed for env in fake.environments] == [True]


def test_toy_benchmark_closes_the_environment_when_the_task_is_cancelled() -> None:
    fake = ToyBenchmark()
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = fake

    async def cancelled() -> None:
        async with asyncio.timeout(0.01), benchmark.provision(TOY_INSTANCES[0]):
            await asyncio.sleep(60)

    with pytest.raises(TimeoutError):
        asyncio.run(cancelled())
    assert [env.closed for env in fake.environments] == [True]


@pytest.mark.parametrize(
    ("sql", "credit"),
    [
        ("SELECT count(id) FROM users", 1.0),  # The gold rewrite.
        ("SELECT count(*) FROM users WHERE id IN (SELECT id FROM users)", 0.5),  # The base query.
        ("SELECT 3", 0.0),  # A query that does not match.
    ],
)
def test_toy_benchmark_credits_a_prediction_as_r_ves_does(sql: str, credit: float) -> None:
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()
    prediction = Prediction(instance_id="toy-1", effisql=EffiSqlPrediction(sql=sql))

    grade = asyncio.run(benchmark.grade(TOY_INSTANCES[0], prediction))

    matched = credit > 0
    assert grade == Grade(
        instance_id="toy-1",
        score=credit,
        correct=matched,
        status=GRADE_STATUS_GRADED,
        effisql=EffiSqlGrade(matched=matched, credit=credit),
    )


def test_toy_benchmark_writes_its_official_record() -> None:
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()
    prediction = Prediction(instance_id="toy-1", effisql=EffiSqlPrediction(sql="SELECT 1"))

    record = benchmark.to_official(TOY_INSTANCES[0], prediction)

    assert json.loads(json.dumps(record)) == {"instance_id": "toy-1", "sql": "SELECT 1"}


def test_toy_benchmark_hands_the_runner_its_manifest_fields() -> None:
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()
    manifest = RunManifest(run_id="run-1", benchmark=benchmark.name)

    fields = benchmark.manifest_fields()
    manifest.MergeFrom(fields)  # As the runner merges them in.

    assert {field.name for field, _ in fields.ListFields()} <= {
        *("dataset_revision", "dataset_notes", "environment_settings", "dataset_sha256"),
    }
    assert fields.environment_settings["grading_mode"] == "dev"
    assert (manifest.run_id, manifest.dataset_revision) == ("run-1", fields.dataset_revision)
    assert len(manifest.dataset_sha256) == 64


def test_toy_benchmark_writes_the_empty_answer_for_a_prediction_with_no_branch() -> None:
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()
    inst = TOY_INSTANCES[0]

    record = benchmark.to_official(inst, Prediction())  # As after a crash.

    assert record == {"instance_id": "toy-1", "sql": inst.base_sql}


def test_toy_benchmark_grades_a_prediction_with_no_branch_as_its_empty_answer() -> None:
    benchmark: Benchmark[ToyInstance, SqlEnvironment] = ToyBenchmark()
    inst = TOY_INSTANCES[0]
    base_query = Prediction(instance_id="toy-1", effisql=EffiSqlPrediction(sql=inst.base_sql))

    async def scenario() -> tuple[Grade, Grade]:
        return (
            await benchmark.grade(inst, Prediction(instance_id="toy-1")),  # As after a crash.
            await benchmark.grade(inst, base_query),
        )

    empty, base = asyncio.run(scenario())

    # The same answer to_official writes for it, so the grade and the record agree.
    assert empty == base
    assert (empty.correct, empty.score) == (True, 0.5)


async def solve_and_grade[InstanceT: Instance, EnvT: Environment, CtxT](
    benchmark: Benchmark[InstanceT, EnvT], policy: Policy[EnvT, CtxT], inst: InstanceT, ctx: CtxT
) -> tuple[Prediction, Grade]:
    """Solve and grade one task as the runner will, through the Protocols alone."""
    async with benchmark.provision(inst) as env:
        prediction = await policy.solve(benchmark.agent_input(inst), env, ctx)
    prediction.instance_id = inst.instance_id  # The task input carries no ID.
    return prediction, await benchmark.grade(inst, prediction)


def test_the_fakes_run_a_split_end_to_end() -> None:
    benchmark = ToyBenchmark()
    policy = DoNothingPolicy()

    async def run_split() -> list[tuple[Prediction, Grade]]:
        return [
            await solve_and_grade(benchmark, policy, inst, StandInContext())
            for inst in benchmark.load_instances(SPLIT_TRAIN)
        ]

    results = asyncio.run(run_split())

    # The do-nothing answer matches, but only toy-2's base query is already the fast one.
    assert [
        (prediction.instance_id, grade.instance_id, grade.correct, grade.score)
        for prediction, grade in results
    ] == [("toy-1", "toy-1", True, 0.5), ("toy-2", "toy-2", True, 1.0)]
    assert [task.effisql.base_sql for task in policy.tasks] == [
        inst.base_sql for inst in TOY_INSTANCES[:2]
    ]
    assert [env.closed for env in benchmark.environments] == [True, True]
