"""GUARD_PACK_CODE_ALLOWLIST (spec §8.5): a static check of a candidate's pack code.

This is a static defense in depth, not a sandbox. Pack code runs on the host, in the process
that holds the API key, during goaly-smoke, screening and validation. So before any of it runs,
this check refuses the plain ways it could reach the key, the model client, files, processes,
the network, other modules or the interpreter's internals. Code written on purpose to slip past a
static check could still get through; the snapshot rule, the sandbox and the other guards stand
behind it.

The rules, for every Python file of the lane's pack (its tests run only in the sandbox):

- **Imports.** Only these: the modules of ALLOWED_STDLIB; pydantic and its public submodules;
  goaly.contracts, except its JSONL file helpers; the core modules of ALLOWED_CORE_MODULES; the
  lane's own pack; and relative imports that stay inside it. No star import. Each name imported
  from an outside module is resolved at check time. It is refused if it is a module that may not
  be imported itself, or if it comes from core's settings or LLM client, under whatever name.
- **Names.** No builtin of BANNED_BUILTINS, no identifier of BANNED_NAMES, and no dunder but
  those of ALLOWED_DUNDERS.
- **Attributes.** The name rules, plus:
  - no attribute of BANNED_ATTRIBUTES or with a prefix of BANNED_ATTRIBUTE_PREFIXES;
  - no attribute with a leading underscore, on any object, except a name-mangled ``self.__x``
    inside a class body;
  - no attribute of an imported module that is itself a module.

  A class pattern's keyword attributes count as attributes.
- **Strings.** Python can evaluate some strings as code, so no string may appear in an
  annotation, no TypeAdapter may be given one, and no string may name a dunder or read as a call
  to a banned builtin.

Each violation reads ``<file>:<line>: <reason>``.
"""

import ast
import importlib
import importlib.util
import re
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from goaly.contracts.gen.experiment_pb2 import GUARD_PACK_CODE_ALLOWLIST, GuardResult, Lane
from goaly.rsi.lanes import lane_info

__all__ = [
    "ALLOWED_CORE_MODULES",
    "ALLOWED_DUNDERS",
    "ALLOWED_STDLIB",
    "BANNED_ATTRIBUTES",
    "BANNED_ATTRIBUTE_PREFIXES",
    "BANNED_BUILTINS",
    "BANNED_NAMES",
    "check_pack_code",
    "violations",
]

# A closed list: pure modules, and asyncio for concurrency.
ALLOWED_STDLIB = frozenset(
    {
        *("abc", "asyncio", "collections", "collections.abc", "dataclasses", "datetime", "enum"),
        *("functools", "itertools", "json", "math", "re", "statistics", "string", "textwrap"),
        "typing",
    }
)
# The core modules a pack may use; never settings or the LLM client's modules (ruling V3).
ALLOWED_CORE_MODULES = frozenset(
    {"agent", "budgets", "context_window", "logs", "tool_calls", "tools", "trajectory"}
)
BANNED_BUILTINS = frozenset(
    {
        *("open", "eval", "exec", "compile", "__import__", "breakpoint", "input", "memoryview"),
        *("getattr", "setattr", "delattr", "vars", "globals", "locals", "dir", "type"),
    }
)
ALLOWED_DUNDERS = frozenset({"__init__", "__post_init__", "__name__", "__doc__"})
# Identifiers banned wherever they appear, as a name, an attribute or an imported name.
BANNED_NAMES = frozenset(
    {
        # The interpreter, the OS, processes, files, the network and code loading, which an
        # allowed module can expose as an attribute, as typing.sys.modules["os"] does.
        *("sys", "os", "modules", "builtins", "subprocess", "socket", "importlib", "inspect"),
        *("ctypes", "shutil", "pathlib", "io", "codecs", "gc", "marshal", "pickle", "signal"),
        *("tempfile", "threading", "multiprocessing", "concurrent", "linecache", "tokenize"),
        *("logging", "urllib", "http", "httpx", "goaly"),
        # asyncio's processes, sockets, pipes and lookups; asyncio is for concurrency only.
        *("create_subprocess_exec", "create_subprocess_shell", "subprocess_exec"),
        *("subprocess_shell", "open_connection", "open_unix_connection", "start_server"),
        *("start_unix_server", "create_connection", "create_server", "create_unix_connection"),
        *("create_unix_server", "create_datagram_endpoint", "connect_read_pipe"),
        *("connect_write_pipe", "connect_accepted_socket", "getaddrinfo", "getnameinfo"),
        *("sendfile", "add_signal_handler"),
        # Attribute access or imports by string, and evaluation of strings as types.
        *("attrgetter", "methodcaller", "Formatter", "get_field", "ImportString", "PyObject"),
        *("get_type_hints", "ForwardRef", "create_model"),
        # The key and the model client (ruling V3).
        *("settings", "load_settings", "get_secret_value", "LlmClient", "llm_models"),
        *("llm_throttle", "llm_errors"),
        # Files: goaly's JSONL helpers, pydantic's file parsing and its path types.
        *("jsonl", "read_jsonl", "write_jsonl", "append_jsonl", "parse_file", "parse_file_as"),
        *("load_file", "FilePath", "DirectoryPath", "NewPath", "SocketPath"),
    }
)
BANNED_NAME_PREFIXES = ("sock_",)  # loop.sock_connect and the rest.
# Attributes banned wherever they appear, but not as plain names.
BANNED_ATTRIBUTES = frozenset({"format", "format_map"})  # Format strings read attributes.
BANNED_ATTRIBUTE_PREFIXES = ("f_", "gi_", "cr_", "ag_", "tb_")  # Frames, coroutines, tracebacks.

_CORE = "goaly.harness.core"
_CORE_SETTINGS = "goaly.harness.core.settings"
_CORE_LLM = "goaly.harness.core.llm"  # A prefix: llm, llm_models, llm_throttle, llm_errors...
_PACKS = "goaly.harness.packs"
_PYDANTIC_BANNED = frozenset({"v1", "deprecated", "plugin"})
_DUNDER = re.compile(r"(?<!\w)__\w+__(?!\w)")
_EVAL_LIMIT = 2000  # Longer strings are prose or data, not types.


@dataclass(frozen=True)
class _Binding:
    """What an imported name stands for: one of the pack's modules, which is never imported
    here, or a module outside the pack, imported to check its attributes.
    """

    module: str
    imported: ModuleType | None = None  # None for the pack's own modules.


def violations(
    source: str,
    *,
    path: str,
    depth: int,
    benchmark: str,
    package: str | None = None,
    pack_imports: Mapping[str, Mapping[str, _Binding]] | None = None,
) -> list[str]:
    """Return every violation in ``source``, a file ``depth`` directories below the root of the
    pack of ``benchmark``, as ``<path>:<line>: <reason>`` in file order.

    ``package`` is the file's package, by default the pack's root. ``pack_imports`` maps each of
    the pack's modules to the modules its imports bind, so that ``helpers.json`` is refused when
    the pack's helpers imported json.
    """
    root = f"{_PACKS}.{benchmark}"
    try:
        tree = ast.parse(source)
    except SyntaxError as error:  # A NUL byte in the source is one too.
        return [f"{path}:{error.lineno or 1}: cannot be parsed: {error.msg}"]
    checker = _Checker(root, package or root, depth, pack_imports or {})
    checker.visit(tree)
    found = sorted(checker.problems, key=lambda problem: problem[0])
    return list(dict.fromkeys(f"{path}:{line}: {reason}" for line, reason in found))


def check_pack_code(root: Path, lane: Lane.ValueType) -> GuardResult:
    """Check every Python file of the lane's pack under ``root``, a candidate's snapshot."""
    info = lane_info(lane)
    pack = root / info.pack_dirs[0]
    package_root = f"{_PACKS}.{info.benchmark}"
    files = sorted(pack.rglob("*.py")) if pack.is_dir() else []
    modules = {_module_name(package_root, file.relative_to(pack)): file for file in files}
    problems = []
    sources = {}
    for name, file in modules.items():
        try:
            sources[name] = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{file.relative_to(root).as_posix()}:1: not UTF-8 text")
    pack_imports = {
        name: _imports_of(name, text, modules, package_root) for name, text in sources.items()
    }
    for name in sources:
        file = modules[name]
        relative = file.relative_to(pack)
        problems += violations(
            sources[name],
            path=file.relative_to(root).as_posix(),
            depth=len(relative.parent.parts),
            benchmark=info.benchmark,
            package=_package_of(name, relative),
            pack_imports=pack_imports,
        )
    if problems:
        return GuardResult(
            guard=GUARD_PACK_CODE_ALLOWLIST, passed=False, detail="\n".join(sorted(problems))
        )
    return GuardResult(
        guard=GUARD_PACK_CODE_ALLOWLIST,
        passed=True,
        detail=f"{len(files)} Python file{'' if len(files) == 1 else 's'} checked",
    )


class _Checker(ast.NodeVisitor):
    def __init__(
        self,
        root: str,
        package: str,
        depth: int,
        pack_imports: Mapping[str, Mapping[str, _Binding]],
    ) -> None:
        self._root = root
        self._package = package
        self._depth = depth
        self._pack_imports = pack_imports
        self._bindings: dict[str, _Binding] = {}
        self._class_depth = 0
        self.problems: list[tuple[int, str]] = []

    def _problem(self, node: ast.AST, reason: str) -> None:
        self.problems.append((getattr(node, "lineno", 1), reason))

    # Imports.

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _allowed(alias.name, self._root):
                self._problem(node, f"import of {alias.name} is not allowed")
                continue
            bound = alias.asname or alias.name.split(".")[0]
            self._identifier(node, bound)  # import goaly.x binds goaly, which reaches everything.
            target = alias.name if alias.asname else bound
            if _in_pack(target, self._root):
                self._bindings[bound] = _Binding(target)
            elif _import(alias.name) is None:
                self._problem(node, f"cannot import {alias.name}")
            else:
                self._bindings[bound] = _Binding(target, _import(target))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            if node.level > self._depth + 1:
                self._problem(node, "a relative import that leaves the pack")
                return
            base = self._package.rsplit(".", node.level - 1)[0]
            module = f"{base}.{node.module}" if node.module else base
        else:
            module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                self._problem(node, f"import * from {module}")
                continue
            self._identifier(node, alias.name, attribute=True)
            self._identifier(node, alias.asname or alias.name)
            if node.level or _in_pack(module, self._root):
                self._import_from_pack(module, alias)
            else:
                self._import_from_outside(node, module, alias)

    def _import_from_pack(self, module: str, alias: ast.alias) -> None:
        """Bind a name imported from the pack itself, whose code is never imported here."""
        bound, full = alias.asname or alias.name, f"{module}.{alias.name}"
        imported = self._pack_imports.get(module, {}).get(alias.name)
        if imported is not None:  # A module that the pack module itself imported.
            self._bindings[bound] = imported
        elif full in self._pack_imports or not self._pack_imports:
            self._bindings[bound] = _Binding(full)

    def _import_from_outside(self, node: ast.ImportFrom, module: str, alias: ast.alias) -> None:
        name, bound = alias.name, alias.asname or alias.name
        full = f"{module}.{name}"
        if module in (_CORE, _PACKS):  # Each of their submodules is allowed, or not, on its own.
            if _in_pack(full, self._root):
                self._bindings[bound] = _Binding(full)
            elif not _allowed(full, self._root):
                self._problem(node, f"import of {full} is not allowed")
            elif not _is_module(full) or (imported := _import(full)) is None:
                self._problem(node, f"cannot import {full}")
            else:
                self._bindings[bound] = _Binding(full, imported)
            return
        if not _allowed(module, self._root):
            self._problem(node, f"import of {module} is not allowed")
            return
        if _is_module(full):  # A submodule, such as collections.abc or pydantic.fields.
            if not _allowed(full, self._root):
                self._problem(node, f"import of {full} is not allowed")
            elif (imported := _import(full)) is None:
                self._problem(node, f"cannot import {full}")
            else:
                self._bindings[bound] = _Binding(full, imported)
            return
        imported = _import(module)
        if imported is None or not hasattr(imported, name):
            self._problem(node, f"cannot import {name} from {module}")
            return
        value = getattr(imported, name)
        if isinstance(value, ModuleType):
            self._problem(node, f"{module}.{name} is a module")
        self._check_origin(node, name, value)

    # Names and attributes.

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_BUILTINS:
            self._problem(node, f"banned builtin {node.id}")
        self._identifier(node, node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._identifier(node, node.attr, attribute=True, owner=node.value)
        if isinstance(node.value, ast.Name) and node.value.id in self._bindings:
            self._check_module_attribute(node, node.value.id, node.attr)
        self.generic_visit(node)

    def _check_module_attribute(self, node: ast.AST, name: str, attribute: str) -> None:
        binding = self._bindings[name]
        if binding.imported is None:  # One of the pack's modules.
            if attribute in self._pack_imports.get(binding.module, {}):
                self._problem(node, f"{name}.{attribute} is a module")
            return
        value = getattr(binding.imported, attribute, None)
        if isinstance(value, ModuleType) or _is_module(f"{binding.module}.{attribute}"):
            self._problem(node, f"{name}.{attribute} is a module")
        elif value is not None:
            self._check_origin(node, f"{name}.{attribute}", value)

    def _check_origin(self, node: ast.AST, name: str, value: object) -> None:
        """Refuse what comes from core's settings or LLM client, under whatever name (N5)."""
        origin = getattr(value, "__module__", None)
        if isinstance(origin, str) and (origin == _CORE_SETTINGS or origin.startswith(_CORE_LLM)):
            self._problem(node, f"{name} comes from {origin}")

    def _identifier(
        self,
        node: ast.AST,
        name: str,
        *,
        attribute: bool = False,
        owner: ast.expr | None = None,
    ) -> None:
        if name.startswith("__") and name.endswith("__") and len(name) > 4:
            if name not in ALLOWED_DUNDERS:
                self._problem(node, f"dunder name {name}")
            return
        if name in BANNED_NAMES or name.startswith(BANNED_NAME_PREFIXES):
            self._problem(node, f"banned name {name}")
        if not attribute:
            return
        if name in BANNED_ATTRIBUTES or name.startswith(BANNED_ATTRIBUTE_PREFIXES):
            self._problem(node, f"banned attribute {name}")
        elif name.startswith("_") and not self._mangled(name, owner):
            self._problem(node, f"private attribute {name}")

    def _mangled(self, name: str, owner: ast.expr | None) -> bool:
        """Whether ``owner.name`` is a name-mangled ``self.__x`` inside a class body."""
        return (
            name.startswith("__")
            and self._class_depth > 0
            and isinstance(owner, ast.Name)
            and owner.id == "self"
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._identifier(node, node.name)
        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._identifier(node, node.name)
        if node.returns is not None:
            self._annotation(node.returns)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._identifier(node, node.arg)
        if node.annotation is not None:
            self._annotation(node.annotation)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg is not None:
            self._identifier(node, node.arg)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._annotation(node.annotation)
        self.generic_visit(node)

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        self._annotation(node.value)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._identifier(node, name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._identifier(node, name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._identifier(node, node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self._identifier(node, node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self._identifier(node, node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self._identifier(node, node.rest)
        self.generic_visit(node)

    def visit_MatchClass(self, node: ast.MatchClass) -> None:
        for attribute in node.kwd_attrs:  # case P(x=...) reads P's attribute x (N3).
            self._identifier(node, attribute, attribute=True)
        self.generic_visit(node)

    # Strings Python may evaluate.

    def _annotation(self, annotation: ast.expr) -> None:
        for inner in ast.walk(annotation):
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                self._problem(inner, "a string in an annotation, which Python may evaluate")

    def visit_Call(self, node: ast.Call) -> None:
        called = node.func
        name = called.id if isinstance(called, ast.Name) else getattr(called, "attr", "")
        if name == "TypeAdapter" and any(
            isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            for argument in node.args
        ):
            self._problem(node, "TypeAdapter of a string, which Python may evaluate")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str):
            return
        for dunder in dict.fromkeys(_DUNDER.findall(node.value)):
            if dunder not in ALLOWED_DUNDERS:
                self._problem(node, f"a string names the dunder {dunder}")
        for called in _calls_in(node.value):
            if called in BANNED_BUILTINS:
                self._problem(node, f"a string Python could evaluate calls {called}")


def _calls_in(text: str) -> Iterator[str]:
    """The names a string calls, if it parses as a Python expression."""
    if "(" not in text or len(text) > _EVAL_LIMIT:
        return
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            yield node.func.id


def _allowed(module: str, root: str) -> bool:
    """Whether pack code may import ``module``, the pack of ``root`` being its own."""
    parts = module.split(".")
    if any(part.startswith("_") for part in parts):
        return False
    if module in ALLOWED_STDLIB:
        return True
    if parts[0] == "pydantic":
        return not _PYDANTIC_BANNED.intersection(parts[1:])
    if parts[:2] == ["goaly", "contracts"]:
        return parts[2:3] != ["jsonl"]
    if parts[:3] == ["goaly", "harness", "core"]:
        return len(parts) >= 4 and parts[3] in ALLOWED_CORE_MODULES
    return _in_pack(module, root)


def _in_pack(module: str, root: str) -> bool:
    return module == root or module.startswith(f"{root}.")


def _import(module: str) -> ModuleType | None:
    """Import an allowed module that lives outside the pack, never the pack's own code."""
    try:
        return importlib.import_module(module)
    except Exception:  # noqa: BLE001  (any failure means the module cannot be used)
        return None


def _is_module(name: str) -> bool:
    """Whether ``name`` names a module, imported or importable, outside the packs."""
    if name.startswith(f"{_PACKS}."):
        return False  # Finding a pack's module would run its package's code.
    if name in sys.modules:
        return isinstance(sys.modules[name], ModuleType)
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # A parent that is no package, or a module with no spec.
        return False


def _module_name(root: str, relative: Path) -> str:
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((root, *parts))


def _package_of(module: str, relative: Path) -> str:
    return module if relative.name == "__init__.py" else module.rsplit(".", 1)[0]


def _imports_of(
    module: str, source: str, modules: Mapping[str, Path], root: str
) -> dict[str, _Binding]:
    """The names the imports of the pack module ``module`` bind to modules, in the pack or not."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}  # Refused when the file itself is checked.
    package = module if modules[module].name == "__init__.py" else module.rsplit(".", 1)[0]
    found: dict[str, _Binding] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                target = alias.name if alias.asname else bound
                if not _in_pack(target, root):
                    found[bound] = _Binding(target, _import(target))
        elif isinstance(node, ast.ImportFrom):
            base = package.rsplit(".", node.level - 1)[0] if node.level else ""
            source_module = ".".join(part for part in (base, node.module or "") if part)
            for alias in node.names:
                full = f"{source_module}.{alias.name}"
                if full in modules:
                    found[alias.asname or alias.name] = _Binding(full)
                elif not _in_pack(source_module, root) and _is_module(full):
                    found[alias.asname or alias.name] = _Binding(full, _import(full))
    return found
