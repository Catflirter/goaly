"""Tests for goaly.contracts.interfaces.

- Typing tests: each fake in tests/fakes.py is assigned to a variable typed with its Protocol,
  so ``make typecheck`` proves that it conforms, and is then exercised at runtime.
- The gRPC-ready rule (acceptance criterion 4): Protocol methods take and return only contract
  messages and plain values. mypy --strict already requires every annotation to be present.
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import NoneType
from typing import Any, Protocol, TypeVar, get_args, get_origin, get_type_hints

import pytest
from google.protobuf.message import Message
from pydantic import BaseModel

from goaly.contracts import interfaces
from goaly.contracts.gen.environment_pb2 import ShellResult, SqlResult
from goaly.contracts.interfaces import Environment, ShellEnvironment, SqlEnvironment
from tests.fakes import COMMAND_NOT_FOUND, FakeEnvironment, FakeShellEnvironment, FakeSqlEnvironment

CLASSES = [
    value
    for value in vars(interfaces).values()
    if isinstance(value, type) and value.__module__ == interfaces.__name__
]
PROTOCOLS = [cls for cls in CLASSES if vars(cls).get("_is_protocol", False)]

# The Python types of protobuf's scalars. An enum value's annotation, such as Split.ValueType, is
# int at runtime.
SCALARS = (bool, bytes, float, int, str)


def describe(hint: object) -> str:
    return hint.__qualname__ if isinstance(hint, type) else repr(hint)


def is_plain(hint: object, type_params: Sequence[object]) -> bool:
    """Whether ``hint`` is a type parameter, a contract message, a builtin scalar or None."""
    if hint in type_params or hint in (*SCALARS, NoneType):
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


def test_every_class_in_the_module_is_a_protocol() -> None:
    assert CLASSES
    assert PROTOCOLS == CLASSES


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=lambda protocol: protocol.__name__)
def test_protocols_take_and_return_only_contract_messages_and_plain_values(protocol: type) -> None:
    assert violations(protocol) == []


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
