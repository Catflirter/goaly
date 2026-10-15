"""Tests for goaly.rsi.guards.allowlist: the static AST allowlist on candidate pack code."""

import asyncio
import importlib
import importlib.util
import sys
import textwrap
import types
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import GUARD_PACK_CODE_ALLOWLIST, LANE_EFFISQL
from goaly.rsi.guards import allowlist
from goaly.rsi.guards.allowlist import check_pack_code, violations

# Pack code as a v0 or a candidate might write it: none of it may be refused.
GOOD_POLICY = '''
"""How the Effi-SQL pack solves a task."""

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from goaly.contracts.gen.task_pb2 import EffiSqlPrediction, Prediction, TaskInput
from goaly.harness.core.budgets import Budgets
from goaly.harness.core.tools import Tool

from . import prompts
from .tools import ExecuteSql

PROMPT = """Rewrite the query so that it runs faster. Keep its result exactly the same.
SELECT l_orderkey, sum(l_extendedprice) FROM lineitem GROUP BY l_orderkey;"""


class Mode(Enum):
    EXPLAIN = "explain"
    RUN = "run"


class Args(BaseModel):
    model_config = ConfigDict(frozen=True)

    sql: str = Field(description="One SELECT statement.")
    mode: Mode = Mode.RUN


@dataclass
class Plan:
    steps: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.steps = [step.strip() for step in self.steps]


class Policy(Tool):
    """Best of two rewrites."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        self.tools = tools
        self.__tries = 0  # Name-mangled: private to this class.
        self.name = "policy"
        self.limits = Budgets()
        self.options = {"steps": 3}
        self.steps = self.options.get("steps", 3)

    async def solve(self, task: TaskInput) -> Prediction:
        async with asyncio.timeout(60):
            async with asyncio.TaskGroup() as group:
                first = group.create_task(self.rewrite(task, 0))
                second = group.create_task(self.rewrite(task, 1))
        both = await asyncio.gather(asyncio.sleep(0), asyncio.sleep(0))
        sql = min(first.result(), second.result(), key=len) or json.dumps(both)
        return Prediction(effisql=EffiSqlPrediction(sql=re.sub(r"\\s+", " ", sql)))

    async def rewrite(self, task: TaskInput, seed: int) -> str:
        self.__tries += 1
        shown = f"{seed}: {task.effisql.base_sql}"
        return prompts.render(shown, len(self.tools)) + ExecuteSql.NAME
'''


@pytest.fixture(autouse=True)
def core_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-ins for the core modules T04 and T05 add, as the guard imports them to check."""
    llm = types.ModuleType("goaly.harness.core.llm")
    tools = types.ModuleType("goaly.harness.core.tools")
    tools.Tool = type("Tool", (), {})  # type: ignore[attr-defined]  # a stand-in module
    tools.asyncio = asyncio  # type: ignore[attr-defined]  # as its own import would bind it
    agent = types.ModuleType("goaly.harness.core.agent")
    agent.llm = llm  # type: ignore[attr-defined]
    agent.run_agent = lambda: None  # type: ignore[attr-defined]
    agent.Client = type("Client", (), {"__module__": "goaly.harness.core.llm"})  # type: ignore[attr-defined]
    read_key = types.FunctionType((lambda: None).__code__, {}, "read_key")
    read_key.__module__ = "goaly.harness.core.settings"
    tools.read_key = read_key  # type: ignore[attr-defined]
    budgets = types.ModuleType("goaly.harness.core.budgets")
    budgets.Budgets = type("Budgets", (), {})  # type: ignore[attr-defined]
    for module in (llm, tools, agent, budgets):
        monkeypatch.setitem(sys.modules, module.__name__, module)


def check(source: str, *, depth: int = 0) -> list[str]:
    return violations(textwrap.dedent(source), path="policy.py", depth=depth, benchmark="effisql")


def test_ordinary_pack_code_passes() -> None:
    assert check(GOOD_POLICY) == []


@pytest.mark.parametrize(
    "source",
    [
        "import json, re, typing, dataclasses, math, statistics, collections\n",
        "import itertools, functools, textwrap, enum, asyncio\n",
        "from collections.abc import Sequence\nfrom collections import abc\n",
        "from collections import Counter, defaultdict\n",
        "import pydantic\npydantic.BaseModel\n",
        "from pydantic.fields import FieldInfo\n",
        "from goaly.contracts import interfaces\n",
        "from goaly.contracts.gen import task_pb2\ntask_pb2.TaskInput\n",
        "from goaly.contracts.errors import EnvironmentFailure\n",
        "from goaly.harness.core.agent import run_agent\nrun_agent()\n",
        "from goaly.harness.packs.effisql.tools import ExecuteSql\n",
        "from goaly.harness.core.tools import Tool\nclass T(Tool): ...\n",
        "from pydantic import TypeAdapter\nTypeAdapter(list[int])\n",
        "from typing import NamedTuple\nP = NamedTuple('P', [('x', int), ('y', float)])\n",
        "from typing import TypedDict\nTD = TypedDict('TD', {'x': int}, total=False)\n",
        "from dataclasses import make_dataclass\nC = make_dataclass('C', ['a', ('b', int)])\n",
        "from typing import TypeVar\nT = TypeVar('T', bound=int)\n",
        "from typing import NamedTuple\nclass P(NamedTuple):\n    x: int\n",
        "row = {'k': 1}\nrow.get('k')\nrow.update({'k': 2})\n",
        "SCHEMA = {'type': 'object', 'properties': {'sql': {'type': 'string'}}}\n",
        "class A:\n    __limit = 3\n    def __helper(self):\n        return self.__limit\n",
        "class A:\n    def f(self):\n        def g():\n"
        "            return self.__x\n        return g\n",
        "import asyncio\nasyncio.TaskGroup\nasyncio.gather\nasyncio.timeout\n",
        "class A:\n    def f(self):\n        return self.__x\n",
        "from goaly.harness.packs import effisql\n",
        "import json\njson.not_there\n",  # An AttributeError when it runs, not a way out.
        "f(**options)\n",
        "try:\n    pass\nexcept ValueError:\n    pass\n",
        "TEXT = 'call __init__ first, then len(x)'\n",
        "match p:\n    case [first, *rest]:\n        pass\n    case {'k': v, **others}:\n"
        "        pass\n    case Point(x=0):\n        pass\n",
        "match p:\n    case [*_]:\n        pass\n    case {'k': 1}:\n        pass\n"
        "    case _:\n        pass\n",
        "LIMIT = 'max(1, 2)'\n",  # Python could evaluate it, but it calls nothing banned.
        "import goaly.harness.packs.effisql.tools as tools\n",
    ],
)
def test_allowed_code_passes(source: str) -> None:
    assert check(source) == []


def test_a_relative_import_within_the_pack_passes() -> None:
    assert check("from .. import helpers\n", depth=1) == []


# Every refusal names the file and line, then the reason.
@pytest.mark.parametrize(
    ("source", "problem"),
    [
        # Modules that reach the OS, processes or the network, and every other module: the
        # stdlib list is closed.
        ("import os\n", "policy.py:1: import of os is not allowed"),
        ("from os import path\n", "policy.py:1: import of os is not allowed"),
        ("import subprocess\n", "policy.py:1: import of subprocess is not allowed"),
        ("import importlib\n", "policy.py:1: import of importlib is not allowed"),
        ("import sys\n", "policy.py:1: import of sys is not allowed"),
        ("import socket\n", "policy.py:1: import of socket is not allowed"),
        ("import pathlib\n", "policy.py:1: import of pathlib is not allowed"),
        ("import shutil\n", "policy.py:1: import of shutil is not allowed"),
        ("import ctypes\n", "policy.py:1: import of ctypes is not allowed"),
        ("import operator\n", "policy.py:1: import of operator is not allowed"),
        ("import httpx\n", "policy.py:1: import of httpx is not allowed"),
        ("import json.decoder\n", "policy.py:1: import of json.decoder is not allowed"),
        ("import asyncio.subprocess\n", "policy.py:1: import of asyncio.subprocess is not"),
        ("from pydantic._internal import x\n", "import of pydantic._internal is not allowed"),
        ("from pydantic.v1 import BaseModel\n", "import of pydantic.v1 is not allowed"),
        # Ruling V3: only the listed core modules; never settings or the LLM client.
        ("from goaly.harness.core import settings\n", "from goaly.harness.core import settings"),
        ("from goaly.harness.core.settings import load_settings\n", "goaly.harness.core.settings"),
        ("import goaly.harness.core.llm as llm\n", "reach core only with from"),
        ("from goaly.harness.core import llm\n", "from goaly.harness.core import llm"),
        ("from goaly.harness.core.llm_models import X\n", "goaly.harness.core.llm_models"),
        ("from goaly.harness.core import llm_throttle\n", "import llm_throttle"),
        ("import goaly.harness.core\n", "import goaly.harness.core: reach core only"),
        ("from goaly.contracts.jsonl import write_jsonl\n", "import of goaly.contracts.jsonl"),
        ("from goaly.contracts import jsonl\n", "banned name jsonl"),
        ("from goaly.bench import runner\n", "import of goaly.bench is not allowed"),
        ("from goaly.harness.packs.swefficiency import x\n", "goaly.harness.packs.swefficiency"),
        ("from goaly.harness.core import context_window\n", "import its names instead"),
        # Binding the goaly package would reach every module the process has loaded.
        ("import goaly.contracts.errors\n", "policy.py:1: banned name goaly"),
        ("from .. import helpers\n", "policy.py:1: a relative import that leaves the pack"),
        ("from typing import *\n", "policy.py:1: import * from typing"),
        # Builtins (rule b adds dir and type).
        ("open('/etc/passwd')\n", "policy.py:1: banned builtin open"),
        ("eval('1')\n", "banned builtin eval"),
        ("exec('x = 1')\n", "banned builtin exec"),
        ("compile('1', 'x', 'eval')\n", "banned builtin compile"),
        ("__import__('os')\n", "banned builtin __import__"),
        ("setattr(x, 'y', 1)\n", "banned builtin setattr"),
        ("delattr(x, 'y')\n", "banned builtin delattr"),
        ("vars(x)\n", "banned builtin vars"),
        ("globals()\n", "banned builtin globals"),
        ("locals()\n", "banned builtin locals"),
        ("dir(x)\n", "banned builtin dir"),
        ("type(x)\n", "banned builtin type"),
        ("breakpoint()\n", "banned builtin breakpoint"),
        ("input()\n", "banned builtin input"),
        ("memoryview(b'')\n", "banned builtin memoryview"),
        ("reader = open\n", "banned builtin open"),
        # Dunders, and the bypasses the brief names.
        ("().__class__.__bases__\n", "policy.py:1: dunder name __class__"),
        ("getattr(x, '__globals__')\n", "policy.py:1: banned builtin getattr"),
        ("getattr(x, '__globals__')\n", "policy.py:1: a string names the dunder __globals__"),
        ("__builtins__['open']\n", "dunder name __builtins__"),
        (
            "class A:\n    def __getattr__(self, name): ...\n",
            "policy.py:2: dunder name __getattr__",
        ),
        (
            "match p:\n    case P(__class__=c):\n        pass\n",
            "policy.py:2: dunder name __class__",
        ),
        ("def f(**k):\n    f(__x__=1)\n", "policy.py:2: dunder name __x__"),
        # The spec re-review's escapes.
        ("ctx._llm._client.complete(model='glm-5.3')\n", "policy.py:1: private attribute _llm"),
        ("import typing\ngetattr(typing, 'sy' + 's')\n", "policy.py:2: banned builtin getattr"),
        ("'{0.__class__}'.format(x)\n", "policy.py:1: banned attribute format"),
        ("'{a}'.format_map(m)\n", "policy.py:1: banned attribute format_map"),
        (
            "import dataclasses\ndataclasses.inspect.currentframe().f_back.f_globals\n",
            "policy.py:2: dataclasses.inspect is a module",
        ),
        (
            "import dataclasses\ndataclasses.inspect.currentframe().f_back.f_globals\n",
            "policy.py:2: banned attribute f_back",
        ),
        # Ruling V12: attribute chains to the interpreter, processes and sockets.
        ("import typing\ntyping.sys.modules['os']\n", "policy.py:2: banned name sys"),
        ("import typing\ntyping.sys.modules['os']\n", "policy.py:2: typing.sys is a module"),
        ("import typing\ntyping.sys.modules['os']\n", "policy.py:2: banned name modules"),
        ("import asyncio\nasyncio.create_subprocess_exec('ls')\n", "banned name create_subprocess"),
        ("import asyncio\nasyncio.create_subprocess_shell('ls')\n", "create_subprocess_shell"),
        ("import asyncio\nasyncio.open_connection('h', 1)\n", "banned name open_connection"),
        ("import asyncio\nasyncio.events.get_event_loop()\n", "asyncio.events is a module"),
        ("await loop.sock_connect(s, a)\n", "policy.py:1: banned name sock_connect"),
        ("await loop.subprocess_exec(p)\n", "banned name subprocess_exec"),
        ("await loop.create_connection(p)\n", "banned name create_connection"),
        ("await loop.connect_write_pipe(p, f)\n", "banned name connect_write_pipe"),
        ("from asyncio import start_server\n", "banned name start_server"),
        ("x.builtins\n", "banned name builtins"),
        ("import json\njson.codecs.open('f')\n", "policy.py:2: json.codecs is a module"),
        # Rule e: a module an allowed module imported is out of reach too.
        ("from goaly.harness.core.agent import llm\n", "goaly.harness.core.agent.llm is a module"),
        ("from goaly.harness.core.tools import asyncio\n", "tools.asyncio is a module"),
        ("import collections\ncollections.abc.Sequence\n", "collections.abc is a module"),
        # Frames and coroutines (rule d), getattr by another name, import by string, the key.
        ("next(g).gi_frame\n", "policy.py:1: banned attribute gi_frame"),
        ("coro.cr_frame\n", "banned attribute cr_frame"),
        ("agen.ag_frame\n", "banned attribute ag_frame"),
        ("err.tb_frame\n", "banned attribute tb_frame"),
        ("import string\nstring.Formatter().get_field('0', [x], {})\n", "banned name Formatter"),
        ("from pydantic import ImportString\n", "banned name ImportString"),
        ("api_key.get_secret_value()\n", "banned name get_secret_value"),
        ("ctx.settings\n", "banned name settings"),
        # A string pydantic or typing would evaluate as code.
        ("x: \"open('/etc/passwd').read()\"\n", "evaluate: banned builtin open"),
        # N1: no private attribute, on any object; only self.__x inside a class body.
        ("class A:\n    def f(self):\n        return self._client\n", "private attribute _client"),
        (
            "class A:\n    @classmethod\n    def f(cls):\n        return cls._x\n",
            "private attribute _x",
        ),
        ("A._y\n", "policy.py:1: private attribute _y"),
        ("def f(self):\n    return self.__x\n", "policy.py:2: private attribute __x"),
        ("from typing import _alias\n", "policy.py:1: private attribute _alias"),
        # N2: a string annotation may be evaluated as code.
        ('x: "int" = 1\n', "policy.py:1: a string in an annotation"),
        ('def f(a: "int") -> None: ...\n', "policy.py:1: a string in an annotation"),
        ('def f() -> "int": ...\n', "policy.py:1: a string in an annotation"),
        ('x: list["int"] = []\n', "policy.py:1: a string in an annotation"),
        ('from typing import Literal\nx: Literal["a"] = "a"\n', "a string in an annotation"),
        ('from pydantic import TypeAdapter\nTypeAdapter("int")\n', "TypeAdapter of a string"),
        ("import typing\ntyping.get_type_hints(f)\n", "banned name get_type_hints"),
        ("from typing import ForwardRef\n", "banned name ForwardRef"),
        ("import pydantic\npydantic.create_model('M')\n", "banned name create_model"),
        # N3: a class pattern reads attributes too.
        ("match x:\n    case object(_client=c):\n        pass\n", "private attribute _client"),
        # N4: format and format_map are banned wherever they appear.
        ("fm = '{0._client}'.format\n", "policy.py:1: banned attribute format"),
        # N5: nothing that comes from core's settings or LLM client, under any name.
        (
            "from goaly.harness.core.agent import Client\n",
            "Client comes from goaly.harness.core.llm",
        ),
        (
            "from goaly.harness.core.tools import read_key\n",
            "comes from goaly.harness.core.settings",
        ),
        ("from goaly.harness.core.agent import Client as C\n", "Client comes from"),
        # Names in every place a name can stand.
        ("def f():\n    global sys\n", "policy.py:2: banned name sys"),
        ("def f():\n    nonlocal __x__\n", "policy.py:2: dunder name __x__"),
        ("try:\n    pass\nexcept ValueError as __e__:\n    pass\n", "dunder name __e__"),
        ("match p:\n    case [__x__]:\n        pass\n", "policy.py:2: dunder name __x__"),
        ("match p:\n    case [*__rest__]:\n        pass\n", "dunder name __rest__"),
        ("match p:\n    case {**__rest__}:\n        pass\n", "dunder name __rest__"),
        ('type Ids = list["int"]\n', "policy.py:1: a string in an annotation"),
        # What cannot be imported, and a module behind another module's name.
        ("from json import not_there\n", "cannot import not_there from json"),
        ("import pydantic.not_there as x\n", "import of pydantic.not_there is not allowed"),
        ("from pydantic.not_there import x\n", "import of pydantic.not_there is not allowed"),
        ("from dataclasses import inspect\n", "policy.py:1: dataclasses.inspect is a module"),
        # Code that does not parse cannot be checked.
        ("def broken(:\n", "policy.py:1: cannot be parsed"),
    ],
)
def test_refused_code_names_its_file_and_line(source: str, problem: str) -> None:
    found = check(source)

    assert any(problem in line for line in found), found


def test_a_module_that_fails_to_import_is_refused_with_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(allowlist, "_import", lambda module: "ImportError")

    assert check("from collections import abc\n") == [
        "policy.py:1: cannot import collections.abc: ImportError"
    ]


def test_every_violation_is_listed_in_file_order() -> None:
    source = "import os\nopen('x')\nx.__class__\n"

    assert check(source) == [
        "policy.py:1: import of os is not allowed",
        "policy.py:2: banned builtin open",
        "policy.py:3: dunder name __class__",
    ]


def test_long_text_that_is_not_code_passes() -> None:
    prompt = "Explain (in words) why the join is slow, then call the tool: run(sql).\n" * 50

    assert check(f"PROMPT = {prompt!r}\n") == []


# The guard on a tree.


def write_pack(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / "src/goaly/harness/packs/effisql" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_the_guard_checks_every_python_file_of_the_pack(tmp_path: Path) -> None:
    write_pack(
        tmp_path,
        {
            "policy.py": GOOD_POLICY,
            "prompts.py": "def render(text, count):\n    return text\n",
            "tools.py": "import os\nclass ExecuteSql: ...\n",
            "sub/deep.py": "from ... import x\n",
            "prompt.md": "import os\n",  # Not Python: never imported.
        },
    )
    tests = tmp_path / "tests/harness/packs/effisql"
    tests.mkdir(parents=True)
    (tests / "test_policy.py").write_text("import pytest\n", encoding="utf-8")  # Sandboxed.

    result = check_pack_code(tmp_path, LANE_EFFISQL)

    assert result.guard == GUARD_PACK_CODE_ALLOWLIST
    assert not result.passed
    assert result.detail.splitlines() == [
        "src/goaly/harness/packs/effisql/sub/deep.py:1: a relative import that leaves the pack",
        "src/goaly/harness/packs/effisql/tools.py:1: import of os is not allowed",
    ]


def test_a_module_another_pack_file_imported_is_out_of_reach(tmp_path: Path) -> None:
    write_pack(
        tmp_path,
        {
            "helpers.py": "import json\nimport dataclasses\nLIMIT = 3\n",
            "policy.py": "from . import helpers\nhelpers.LIMIT\nhelpers.dataclasses.field\n"
            "helpers.json.loads('1')\n",
        },
    )

    result = check_pack_code(tmp_path, LANE_EFFISQL)

    assert result.detail.splitlines() == [
        "src/goaly/harness/packs/effisql/policy.py:3: helpers.dataclasses is a module",
        "src/goaly/harness/packs/effisql/policy.py:4: helpers.json is a module",
    ]


def test_a_module_reached_through_a_from_import_of_the_pack_is_out_of_reach(
    tmp_path: Path,
) -> None:
    write_pack(
        tmp_path,
        {
            "__init__.py": "",
            "helpers.py": "import dataclasses\n"
            "import goaly.harness.packs.effisql.policy as policy\n",
            "policy.py": "from .helpers import dataclasses\ndataclasses.inspect\n",
            "broken.py": "def broken(:\n",
        },
    )

    result = check_pack_code(tmp_path, LANE_EFFISQL)

    assert result.detail.splitlines() == [
        "src/goaly/harness/packs/effisql/broken.py:1: cannot be parsed: invalid syntax",
        "src/goaly/harness/packs/effisql/policy.py:2: banned name inspect",
        "src/goaly/harness/packs/effisql/policy.py:2: dataclasses.inspect is a module",
    ]


def test_a_file_that_is_not_python_text_fails_the_guard(tmp_path: Path) -> None:
    write_pack(tmp_path, {"nul.py": "x = 1\0\n"})
    (tmp_path / "src/goaly/harness/packs/effisql/latin.py").write_bytes(b"x = '\xe9'\n")

    result = check_pack_code(tmp_path, LANE_EFFISQL)

    assert not result.passed
    assert result.detail.splitlines() == [
        "src/goaly/harness/packs/effisql/latin.py:1: not UTF-8 text",
        "src/goaly/harness/packs/effisql/nul.py:1: cannot be parsed: "
        "source code string cannot contain null bytes",
    ]


def test_the_guard_passes_a_clean_pack(tmp_path: Path) -> None:
    write_pack(
        tmp_path,
        {
            "policy.py": GOOD_POLICY,
            "prompts.py": "def render(text, count):\n    return text\n",
            "tools.py": "class ExecuteSql: ...\n",
        },
    )

    result = check_pack_code(tmp_path, LANE_EFFISQL)

    assert result.passed, result.detail
    assert result.detail == "3 Python files checked"


# The review wave of 2026-10-15 and spec §8.5 at e71d9f1.
@pytest.mark.parametrize(
    ("source", "problem"),
    [
        # Exactly the spec's stdlib modules, and no __future__ (M4, M5).
        ("import abc\n", "import of abc is not allowed"),
        ("import string\n", "import of string is not allowed"),
        ("import datetime\n", "import of datetime is not allowed"),
        ("from __future__ import annotations\n", "import of __future__ is not allowed"),
        # Modules a candidate names are refused before anything imports them (#40 C1).
        ("import this as t\n", "import of this is not allowed"),
        ("import unittest.__main__ as m\n", "import of unittest.__main__ is not allowed"),
        ("from tkinter import x\n", "import of tkinter is not allowed"),
        # N5: core only as from goaly.harness.core.<module> import <name>.
        ("import goaly.harness.core.tools as tools\n", "reach core only with from"),
        ("from goaly.harness.core import agent\n", "from goaly.harness.core import agent"),
        ("from goaly.harness.core.tools import read_key\n", "comes from goaly.harness.core."),
        # TypeAdapter under another name, or with a keyword (#40 I1).
        ("from pydantic import TypeAdapter as TA\nTA('x')\n", "TypeAdapter imported under"),
        ("from pydantic import TypeAdapter\nTA = TypeAdapter\n", "TypeAdapter used other than"),
        ("from pydantic import TypeAdapter\nTypeAdapter(type='int')\n", "TypeAdapter of a str"),
        ("import pydantic\npydantic.TypeAdapter(list['int'])\n", "TypeAdapter of a string"),
        # Strings typing or pydantic evaluate, wherever they stand (#40 I2, spec).
        ('Rows = list["int"]\n', "policy.py:1: a string in a subscript"),
        ("import typing\nRows = typing.List['int']\n", "a string in a subscript"),
        ("row['k']\n", "policy.py:1: a string in a subscript"),
        ("row['k'] = 1\n", "a string in a subscript"),
        ("row[f'{k}']\n", "a string in a subscript"),
        (
            "X = \"typing.sys.modules['os'].getpid()\"\n",
            "policy.py:1: in a string Python could evaluate: banned name sys",
        ),
        ("X: 'TypeAlias' = \"open('f')\"\n", "evaluate: banned builtin open"),
        ("from typing import TypeVar\nT = TypeVar('T', bound='open(1)')\n", "a string as a type"),
        ("from typing import TypeVar\nT = TypeVar('T', 'int', 'str')\n", "a string as a type"),
        ("from typing import NewType\nU = NewType('U', 'int')\n", "a string as a type in NewType"),
        ("PROMPT = '" + "x " * 2000 + "' + open('f')\n", "banned builtin open"),
        ("X = '" + " " * 3000 + "open(1)'\n", "evaluate: banned builtin open"),
        ("X = 'self._client'\n", "evaluate: private attribute _client"),
        (
            "from typing import NamedTuple\nP = NamedTuple('P', [('x', 'int')])\n",
            "a string as a field type in NamedTuple",
        ),
        ("from typing import NamedTuple\nP = NamedTuple('P', x='int')\n", "field type"),
        ("from typing import NamedTuple\nP = NamedTuple('P', fields)\n", "not listed literally"),
        ("from typing import NamedTuple\nP = NamedTuple('P', *fields)\n", "unpacked arguments"),
        ("from typing import TypedDict\nT = TypedDict('T', {'x': 'int'})\n", "field type"),
        (
            "from dataclasses import make_dataclass\nC = make_dataclass('C', [('x', 'int')])\n",
            "a string as a field type in make_dataclass",
        ),
        ("from pydantic import BaseModel\nclass M(BaseModel):\n    x: 'int'\n", "annotation"),
        # A module only as the value of an attribute it is read from (#40 I3).
        ("import typing\nt = typing\nt.types.FunctionType\n", "the module typing used as a"),
        ("import typing\n[typing][0].types.CodeType\n", "the module typing used as a value"),
        ("import typing\ndef f(m=typing):\n    return m.types\n", "the module typing used"),
        ("import asyncio\nm = asyncio\n", "the module asyncio used as a value"),
        ("import json\njson = 1\n", "the module json rebound"),
        ("from . import prompts\nf(prompts)\n", None),  # Unbound without the pack's imports.
        # Dunders: none but __init__ and __post_init__ defined in a class body (spec).
        ("class A:\n    __match_args__ = ('x',)\n", "policy.py:2: dunder name __match_args__"),
        ("class A:\n    __slots__ = ()\n", "dunder name __slots__"),
        ("class A:\n    def __eq__(self, other): ...\n", "dunder name __eq__"),
        ("def __init__(self): ...\n", "policy.py:1: dunder name __init__"),
        ("class A:\n    def f(self):\n        super().__init__()\n", "dunder name __init__"),
        ("x.__name__\n", "dunder name __name__"),
        ("x.__doc__\n", "dunder name __doc__"),
        # P8: a dataclass's __match_args__ reads class-body names with one underscore.
        (
            "import dataclasses\nimport typing\n@dataclasses.dataclass\n"
            "@typing.runtime_checkable\nclass K(typing.Protocol):\n    _client: typing.Any\n"
            "match obj:\n    case K(c):\n        pass\n",
            "policy.py:6: a class binds _client",
        ),
        ("import typing\n@typing.runtime_checkable\nclass K: ...\n", "banned name runtime"),
        ("class A:\n    _x = 1\n", "policy.py:2: a class binds _x"),
        ("class A:\n    def _helper(self): ...\n", "policy.py:2: a class binds _helper"),
        # Attribute reads through pydantic (spec).
        ("from pydantic import ConfigDict\nConfigDict(from_attributes=True)\n", "from_attr"),
        ("config = {'from_attributes': True}\n", "the string from_attributes"),
        ("M.model_validate(obj, from_attributes=True)\n", "banned keyword from_attributes"),
        ("from pydantic import Field\nField(alias='_client')\n", "banned keyword alias"),
        ("from pydantic import Field\nField(validation_alias='x')\n", "banned keyword validation"),
        ("from pydantic import AliasPath\n", "banned name AliasPath"),
        ("from pydantic import AliasChoices\n", "banned name AliasChoices"),
        # pydantic's deprecated methods, and serialized objects (#47).
        (
            "M.parse_raw(data, proto='pickle', allow_pickle=True)\n",
            "policy.py:1: banned attribute parse_raw",
        ),
        ("M.parse_raw(data, proto='pickle', allow_pickle=True)\n", "banned keyword allow_pickle"),
        ("M.parse_raw(data, proto='pickle', allow_pickle=True)\n", "banned keyword proto"),
        ("M.parse_obj(data)\n", "banned attribute parse_obj"),
        ("M.parse_file(path)\n", "banned name parse_file"),
        ("M.from_orm(obj)\n", "banned attribute from_orm"),
        ("M.construct(x=1)\n", "banned attribute construct"),
        ("M.schema_json()\n", "banned attribute schema_json"),
        ("m.copy(update={'x': 1})\n", "banned attribute copy"),
        ("M.update_forward_refs()\n", "banned attribute update_forward_refs"),
        ("loads_pickle(x)\n", "banned name loads_pickle"),
        ("x.unmarshal()\n", "banned name unmarshal"),
        ("f(shelve_it=1)\n", "banned name shelve_it"),
        # Names (M6, M7) and self (M6).
        ("class _:\n    pass\n", "a class named _, which mangles nothing"),
        ("class __:\n    pass\n", "a class named __"),
        ("class __LlmClient:\n    pass\n", "banned name __LlmClient"),
        ("class A:\n    def f(self, other):\n        self = other\n", "self assigned"),
        ("class A:\n    def f(self, other):\n        return other.__x\n", "private attribute"),
        ("help(x)\n", "banned builtin help"),
        ("exit(1)\n", "banned builtin exit"),
        ("quit()\n", "banned builtin quit"),
        ("license()\n", "banned builtin license"),
        ("copyright\n", "banned builtin copyright"),
        ("credits\n", "banned builtin credits"),
    ],
)
def test_the_review_waves_escapes_are_refused(source: str, problem: str | None) -> None:
    found = check(source)

    if problem is None:
        assert found == []
    else:
        assert any(problem in line for line in found), found


def test_no_module_a_candidate_names_is_imported_or_looked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []
    real_import = importlib.import_module

    def spy(name: str, package: str | None = None) -> types.ModuleType:
        requested.append(name)
        return real_import(name, package)

    def no_lookups(name: str, package: str | None = None) -> None:
        requested.append(f"find_spec {name}")

    monkeypatch.setattr(importlib, "import_module", spy)
    monkeypatch.setattr(importlib.util, "find_spec", no_lookups)
    source = (
        "import this as t\nimport unittest.__main__ as m\nfrom tkinter import x\n"
        "from json import tool\nfrom pydantic import not_there\nimport antigravity as a\n"
    )

    found = check(source)

    assert all(name in {"json", "pydantic"} for name in requested), requested
    assert len(found) >= 6


def test_an_expression_too_deep_to_parse_fails_the_check() -> None:
    found = check("x = " + "+".join(["1"] * 100_000) + "\n")

    assert len(found) == 1
    assert "cannot be" in found[0]


def test_a_directory_named_like_a_python_file_fails_the_guard(tmp_path: Path) -> None:
    write_pack(tmp_path, {"notes.py/readme.md": "x\n", "policy.py": "X = 1\n"})

    result = check_pack_code(tmp_path, LANE_EFFISQL)

    assert result.detail.splitlines() == [
        "src/goaly/harness/packs/effisql/notes.py:1: not a regular Python file"
    ]


def test_violations_are_listed_by_file_then_line(tmp_path: Path) -> None:
    write_pack(
        tmp_path,
        {"a.py": "\n" * 8 + "import os\nimport sys\n", "b.py": "open('x')\n"},
    )

    result = check_pack_code(tmp_path, LANE_EFFISQL)

    assert result.detail.splitlines() == [
        "src/goaly/harness/packs/effisql/a.py:9: import of os is not allowed",
        "src/goaly/harness/packs/effisql/a.py:10: import of sys is not allowed",
        "src/goaly/harness/packs/effisql/b.py:1: banned builtin open",
    ]


def test_a_module_the_pack_imported_is_out_of_reach_as_a_value_too(tmp_path: Path) -> None:
    write_pack(
        tmp_path,
        {
            "helpers.py": "import json\nLIMIT = 3\n",
            "policy.py": "from . import helpers\nhelpers.LIMIT\nuse(helpers)\n",
        },
    )

    result = check_pack_code(tmp_path, LANE_EFFISQL)

    assert result.detail.splitlines() == [
        "src/goaly/harness/packs/effisql/policy.py:3: the module helpers used as a value; "
        "only its attributes may be read"
    ]


@pytest.mark.parametrize(
    ("source", "problem"),
    [
        ("from goaly.harness.core.logs import not_there\n", "cannot import not_there from"),
        ("import pydantic\nf(pydantic.TypeAdapter)\n", "TypeAdapter used other than called"),
        ("import json\njson.loads = f\n", "only its attributes may be read"),  # Not written.
        ("class A(base()):\n    pass\n", None),
        ("from dataclasses import make_dataclass\nmake_dataclass('C', [spec])\n", "literally"),
    ],
)
def test_the_rules_edges(source: str, problem: str | None) -> None:
    found = check(source)

    if problem is None:
        assert found == []
    else:
        assert any(problem in line for line in found), found


@pytest.mark.parametrize(
    "source",
    ["from goaly.harness.core.tools import Tool\n", "from json import loads\n"],
)
def test_a_listed_module_that_fails_to_import_is_refused(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(allowlist, "_import", lambda module: "SyntaxError")

    assert check(source)[0].endswith(": SyntaxError")


def test_the_importer_refuses_any_module_off_the_table() -> None:
    assert allowlist._import("this") == "it is not in the table"
    assert "this" not in sys.modules or sys.modules["this"] is not None


@pytest.mark.parametrize(("raised", "outcome"), [(SystemExit, "SystemExit"), (ValueError, "")])
def test_an_import_that_raises_anything_is_a_reason_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, raised: type[BaseException], outcome: str
) -> None:
    def failing(name: str, package: str | None = None) -> types.ModuleType:
        raise raised("boom")

    monkeypatch.setattr(importlib, "import_module", failing)

    assert allowlist._import("json") == (outcome or "ValueError")


def test_an_interrupt_while_importing_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(name: str, package: str | None = None) -> types.ModuleType:
        raise KeyboardInterrupt

    monkeypatch.setattr(importlib, "import_module", interrupted)

    with pytest.raises(KeyboardInterrupt):
        allowlist._import("json")


def test_code_too_deep_to_check_fails_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def too_deep(self: object, node: object) -> None:
        raise RecursionError

    monkeypatch.setattr(allowlist._Checker, "visit", too_deep)

    assert check("x = 1\n") == ["policy.py:1: cannot be checked: it nests too deeply"]


def test_a_submodule_another_pack_file_imported_is_bound_as_a_module(tmp_path: Path) -> None:
    write_pack(
        tmp_path,
        {
            "helpers.py": "from collections import abc\n",
            "policy.py": "from .helpers import abc\nabc.Sequence\nuse(abc)\n",
        },
    )

    result = check_pack_code(tmp_path, LANE_EFFISQL)

    assert result.detail.splitlines() == [
        "src/goaly/harness/packs/effisql/policy.py:3: the module abc used as a value; "
        "only its attributes may be read"
    ]
