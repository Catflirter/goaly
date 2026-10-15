"""GUARD_PACK_CODE_ALLOWLIST (spec §8.5): a static check of a candidate's pack code.

This is a static defense in depth, not a sandbox. In every real run pack code runs only in the
solve container, which holds no key, no grader credentials and no gold (spec §6.1); only the
one-process mode runs it next to the client's real configuration. This check is a second layer
in front of the container: before pack code runs anywhere, it refuses the plain ways the code
could reach the key, the model client, files, processes, the network, other modules or the
interpreter's internals. Code written on purpose to slip past a static check could still get
through, and a string built at run time is beyond any static check; the container stands behind
it.

The rules, for every Python file of the lane's pack (its tests run only in the sandbox):

- **Imports.** Only the modules of a static table: those of ALLOWED_STDLIB, those of
  ALLOWED_PYDANTIC, goaly.contracts and its modules but its JSONL file helpers, and the lane's
  own pack, with relative imports that stay inside it. No module a candidate names is imported,
  or even looked for, until it has passed that table. Core is reached only as
  ``from goaly.harness.core.<module> import <name>``, for a module of ALLOWED_CORE_MODULES, so
  that pack code never holds a core module object; each name imported that way is refused if it
  is a module, or if it comes from core's settings or LLM client, under whatever name. No star
  import, and no name imported from a module that is itself a module.
- **Names.** No builtin of BANNED_BUILTINS and no identifier of BANNED_NAMES, wherever it
  appears, even as a local or a parameter: so no ``type``, ``input``, ``settings`` or
  ``logging``. No identifier that contains one of BANNED_SUBSTRINGS. No dunder: a class body may
  define only ``__init__`` and ``__post_init__``. A class body binds no name with one leading
  underscore, and a class's name is not underscores alone. ``self`` is never assigned. A module
  is used only to read its attributes: never as a value.
- **Attributes.** The name rules, plus no attribute of BANNED_ATTRIBUTES or with a prefix of
  BANNED_ATTRIBUTE_PREFIXES; no attribute with a leading underscore, on any object, except a
  name-mangled ``self.__x`` inside a class body; and no attribute of an imported module that is
  itself a module. A class pattern's keyword attributes count as attributes.
- **Keywords.** None of BANNED_KEYWORDS, which make pydantic read attributes or unpickle data.
- **Strings.** typing and pydantic can evaluate strings as code, so no string may appear in an
  annotation, inside any subscript (so dict keys are read and written with ``get`` and
  ``update``), as a field type of the functional NamedTuple, TypedDict and make_dataclass forms,
  as a TypeVar's bound or constraints, as NewType's type, or as an argument of TypeAdapter.
  Those forms may only be called directly, never aliased or passed on. A string that parses as
  an expression, more than a bare name or constant, must pass every rule as code would; no
  string names a dunder, and no string is ``from_attributes``.

Each violation reads ``<file>:<line>: <reason>``, in file order.
"""

import ast
import importlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import goaly.contracts
from goaly.contracts.gen.experiment_pb2 import GUARD_PACK_CODE_ALLOWLIST, GuardResult, Lane
from goaly.rsi.lanes import lane_info

__all__ = [
    "ALLOWED_CORE_MODULES",
    "ALLOWED_PYDANTIC",
    "ALLOWED_STDLIB",
    "BANNED_ATTRIBUTES",
    "BANNED_ATTRIBUTE_PREFIXES",
    "BANNED_BUILTINS",
    "BANNED_KEYWORDS",
    "BANNED_NAMES",
    "BANNED_SUBSTRINGS",
    "check_pack_code",
    "violations",
]

# Exactly these, as spec §8.5 lists them, with collections' abc; asyncio is for concurrency.
ALLOWED_STDLIB = frozenset(
    {
        *("json", "re", "typing", "dataclasses", "math", "statistics", "collections"),
        *("collections.abc", "itertools", "functools", "textwrap", "enum", "asyncio"),
    }
)
ALLOWED_PYDANTIC = frozenset(
    {
        *("pydantic", "pydantic.dataclasses", "pydantic.fields", "pydantic.types"),
        *("pydantic.functional_validators", "pydantic.functional_serializers"),
        "pydantic.config",
    }
)
# The core modules a pack may import names from; never settings or the LLM client's (V3).
ALLOWED_CORE_MODULES = frozenset(
    {"agent", "budgets", "context_window", "logs", "tool_calls", "tools", "trajectory"}
)
BANNED_BUILTINS = frozenset(
    {
        *("open", "eval", "exec", "compile", "__import__", "breakpoint", "input", "memoryview"),
        *("getattr", "setattr", "delattr", "vars", "globals", "locals", "dir", "type"),
        *("help", "exit", "quit", "license", "copyright", "credits"),
    }
)
# Identifiers banned wherever they appear, as a name, an attribute, a keyword or an import.
BANNED_NAMES = frozenset(
    {
        # The interpreter, the OS, processes, files, the network and code loading, which an
        # allowed module can expose as an attribute, as typing.sys.modules["os"] does.
        *("sys", "os", "modules", "builtins", "subprocess", "socket", "importlib", "inspect"),
        *("ctypes", "shutil", "pathlib", "io", "codecs", "gc", "signal", "tempfile"),
        *("threading", "multiprocessing", "concurrent", "linecache", "tokenize", "logging"),
        *("urllib", "http", "httpx", "goaly"),
        # asyncio's processes, sockets, pipes and lookups; asyncio is for concurrency only.
        *("create_subprocess_exec", "create_subprocess_shell", "subprocess_exec"),
        *("subprocess_shell", "open_connection", "open_unix_connection", "start_server"),
        *("start_unix_server", "create_connection", "create_server", "create_unix_connection"),
        *("create_unix_server", "create_datagram_endpoint", "connect_read_pipe"),
        *("connect_write_pipe", "connect_accepted_socket", "getaddrinfo", "getnameinfo"),
        *("sendfile", "add_signal_handler"),
        # Attribute access or imports by string, and evaluation of strings as types.
        *("attrgetter", "methodcaller", "Formatter", "get_field", "ImportString", "PyObject"),
        *("get_type_hints", "ForwardRef", "create_model", "runtime_checkable"),
        # pydantic's reads of attributes by alias.
        *("AliasPath", "AliasChoices"),
        # The key and the model client (ruling V3).
        *("settings", "load_settings", "get_secret_value", "LlmClient", "llm_models"),
        *("llm_throttle", "llm_errors"),
        # Files: goaly's JSONL helpers, pydantic's file parsing and its path types.
        *("jsonl", "read_jsonl", "write_jsonl", "append_jsonl", "parse_file", "parse_file_as"),
        *("load_file", "FilePath", "DirectoryPath", "NewPath", "SocketPath"),
    }
)
BANNED_NAME_PREFIXES = ("sock_",)  # loop.sock_connect and the rest.
# Any identifier that contains one of these: serialized objects run code when they load.
BANNED_SUBSTRINGS = ("pickle", "marshal", "shelve")
# Attributes banned wherever they appear, but not as plain names.
BANNED_ATTRIBUTES = frozenset(
    {
        "format",  # Format strings read attributes.
        "format_map",
        # pydantic's deprecated BaseModel methods: parse_raw can unpickle, and the rest read
        # attributes or skip validation.
        *("parse_raw", "parse_obj", "from_orm", "construct", "schema_json", "copy"),
        "update_forward_refs",  # It evaluates strings.
    }
)
BANNED_ATTRIBUTE_PREFIXES = ("f_", "gi_", "cr_", "ag_", "tb_")  # Frames, coroutines, tracebacks.
# Keywords that make pydantic read an object's attributes, or unpickle data.
BANNED_KEYWORDS = frozenset(
    {"from_attributes", "alias", "validation_alias", "allow_pickle", "proto"}
)
# The only dunders a class body may define (spec §8.5).
_CLASS_DUNDERS = frozenset({"__init__", "__post_init__"})
# The forms that evaluate strings, which may only be called directly.
_STRING_FORMS = frozenset(
    {"TypeAdapter", "NamedTuple", "TypedDict", "make_dataclass", "TypeVar", "NewType"}
)

_DUNDER = re.compile(r"(?<!\w)__\w+__(?!\w)")

_CORE = "goaly.harness.core"
_CORE_SETTINGS = "goaly.harness.core.settings"
_CORE_LLM = "goaly.harness.core.llm"  # A prefix: llm, llm_models, llm_throttle, llm_errors...
_PACKS = "goaly.harness.packs"


def _contracts_modules() -> frozenset[str]:
    """goaly.contracts and its modules, as installed, but its JSONL file helpers."""
    root = Path(goaly.contracts.__file__).parent
    found = {"goaly.contracts"}
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not any(part.startswith("_") for part in parts) and parts != ("jsonl",):
            found.add(".".join(("goaly.contracts", *parts)))
    return frozenset(found)


# The static table of the modules outside the pack that pack code may import.
_TABLE = ALLOWED_STDLIB | ALLOWED_PYDANTIC | _contracts_modules()
_CORE_TABLE = frozenset(f"{_CORE}.{name}" for name in ALLOWED_CORE_MODULES)


@dataclass(frozen=True)
class _Binding:
    """What an imported name stands for: one of the pack's modules, which is never imported
    here, or a module of the table, imported to check its attributes.
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
) -> list[str]:
    """Return every violation in ``source``, a file ``depth`` directories below the root of the
    pack of ``benchmark``, as ``<path>:<line>: <reason>`` in file order.

    ``package`` is the file's package, by default the pack's root.
    """
    return _violations(source, path=path, depth=depth, benchmark=benchmark, package=package)


def check_pack_code(root: Path, lane: Lane.ValueType) -> GuardResult:
    """Check every Python file of the lane's pack under ``root``, a candidate's snapshot."""
    info = lane_info(lane)
    pack = root / info.pack_dirs[0]
    package_root = f"{_PACKS}.{info.benchmark}"
    paths = sorted(pack.rglob("*.py")) if pack.is_dir() else []
    problems: dict[str, list[str]] = {}
    files = []
    for path in paths:
        shown = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.is_file():
            problems[shown] = [f"{shown}:1: not a regular Python file"]
        else:
            files.append(path)
    modules = {_module_name(package_root, file.relative_to(pack)): file for file in files}
    sources = {}
    for name, file in modules.items():
        try:
            sources[name] = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems[file.relative_to(root).as_posix()] = [
                f"{file.relative_to(root).as_posix()}:1: not UTF-8 text"
            ]
    pack_imports = {name: _imports_of(name, text, modules) for name, text in sources.items()}
    for name, text in sources.items():
        file = modules[name]
        relative = file.relative_to(pack)
        shown = file.relative_to(root).as_posix()
        problems[shown] = _violations(
            text,
            path=shown,
            depth=len(relative.parent.parts),
            benchmark=info.benchmark,
            package=_package_of(name, relative),
            pack_imports=pack_imports,
        )
    found = [line for shown in sorted(problems) for line in problems[shown]]  # File, then line.
    if found:
        return GuardResult(guard=GUARD_PACK_CODE_ALLOWLIST, passed=False, detail="\n".join(found))
    return GuardResult(
        guard=GUARD_PACK_CODE_ALLOWLIST,
        passed=True,
        detail=f"{len(files)} Python file{'' if len(files) == 1 else 's'} checked",
    )


def _violations(
    source: str,
    *,
    path: str,
    depth: int,
    benchmark: str,
    package: str | None = None,
    pack_imports: Mapping[str, Mapping[str, _Binding]] | None = None,
) -> list[str]:
    """violations, with ``pack_imports``: for each of the pack's modules, the modules its
    imports bind, so that ``helpers.json`` is refused when the pack's helpers imported json.
    """
    root = f"{_PACKS}.{benchmark}"
    tree = _parse(source)
    if isinstance(tree, str):
        return [f"{path}:{tree}"]
    checker = _Checker(root, package or root, depth, pack_imports or {})
    try:
        checker.visit(tree)
    except RecursionError:
        return [f"{path}:1: cannot be checked: it nests too deeply"]
    found = sorted(checker.problems, key=lambda problem: problem[0])
    return list(dict.fromkeys(f"{path}:{line}: {reason}" for line, reason in found))


def _parse(source: str) -> ast.Module | str:
    """The module's syntax tree, or ``<line>: <reason>`` if it cannot be parsed."""
    try:
        return ast.parse(source)
    except SyntaxError as error:  # A NUL byte in the source is one too.
        return f"{error.lineno or 1}: cannot be parsed: {error.msg}"
    except (ValueError, RecursionError, MemoryError) as error:
        return f"1: cannot be parsed: {type(error).__name__}"


class _Checker(ast.NodeVisitor):
    def __init__(
        self,
        root: str,
        package: str,
        depth: int,
        pack_imports: Mapping[str, Mapping[str, _Binding]],
        *,
        in_string: bool = False,
    ) -> None:
        self._root = root
        self._package = package
        self._depth = depth
        self._pack_imports = pack_imports
        self._in_string = in_string
        self._bindings: dict[str, _Binding] = {}
        self._class_depth = 0  # Inside a class: self.__x is mangled, also in its methods.
        self._direct_class: ast.ClassDef | None = None  # The class whose body this is.
        self._allowed_uses: set[int] = set()  # Names in the one place they may be.
        self.problems: list[tuple[int, str]] = []

    def _problem(self, node: ast.AST, reason: str) -> None:
        prefix = "in a string Python could evaluate: " if self._in_string else ""
        self.problems.append((getattr(node, "lineno", 1), prefix + reason))

    # Imports.

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name
            bound = alias.asname or name.split(".")[0]
            if name == _CORE or name.startswith(f"{_CORE}."):
                self._problem(node, f"import {name}: reach core only with from {name} import")
                continue
            if not (_in_pack(name, self._root) or name in _TABLE):
                self._problem(node, f"import of {name} is not allowed")
                continue
            self._identifier(node, bound)  # import goaly.x binds goaly, which reaches everything.
            target = name if alias.asname else bound
            if _in_pack(target, self._root):
                self._bindings[bound] = _Binding(target)
            elif target in _TABLE and (imported := self._import(node, target)) is not None:
                self._bindings[bound] = _Binding(target, imported)

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
            if alias.name in _STRING_FORMS and alias.asname not in (None, alias.name):
                self._problem(node, f"{alias.name} imported under another name")
            if node.level or _in_pack(module, self._root):
                self._import_from_pack(module, alias)
            elif module == _PACKS and f"{module}.{alias.name}" == self._root:
                self._bindings[alias.asname or alias.name] = _Binding(self._root)
            elif module in _CORE_TABLE:
                self._import_from_core(node, module, alias)
            elif module in _TABLE:
                self._import_from_table(node, module, alias)
            elif module == _CORE:
                self._problem(node, f"from {_CORE} import {alias.name}: import its names instead")
            else:
                self._problem(node, f"import of {module} is not allowed")

    def _import_from_pack(self, module: str, alias: ast.alias) -> None:
        """Bind a name imported from the pack itself, whose code is never imported here. A name
        that is neither one of the pack's modules nor a module one of them imported is left
        unbound: a module could reach it only through an alias, which the module rules refuse.
        """
        bound, full = alias.asname or alias.name, f"{module}.{alias.name}"
        imported = self._pack_imports.get(module, {}).get(alias.name)
        if imported is not None:  # A module that the pack module itself imported.
            self._bindings[bound] = imported
        elif full in self._pack_imports:  # One of the pack's modules.
            self._bindings[bound] = _Binding(full)

    def _import_from_core(self, node: ast.ImportFrom, module: str, alias: ast.alias) -> None:
        imported = self._import(node, module)
        if imported is None:
            return
        if not hasattr(imported, alias.name):
            self._problem(node, f"cannot import {alias.name} from {module}")
            return
        value = getattr(imported, alias.name)
        if isinstance(value, ModuleType):
            self._problem(node, f"{module}.{alias.name} is a module")
        self._check_origin(node, alias.name, value)

    def _import_from_table(self, node: ast.ImportFrom, module: str, alias: ast.alias) -> None:
        full = f"{module}.{alias.name}"
        bound = alias.asname or alias.name
        if full in _TABLE:  # A submodule of the table, such as collections.abc.
            if (imported := self._import(node, full)) is not None:
                self._bindings[bound] = _Binding(full, imported)
            return
        imported = self._import(node, module)
        if imported is None:
            return
        if not hasattr(imported, alias.name):  # Never looked for as a submodule.
            self._problem(node, f"cannot import {alias.name} from {module}")
            return
        value = getattr(imported, alias.name)
        if isinstance(value, ModuleType):
            self._problem(node, f"{full} is a module")
        self._check_origin(node, alias.name, value)

    def _import(self, node: ast.AST, module: str) -> ModuleType | None:
        """Import a module of the table, never one a candidate merely named."""
        imported = _import(module)
        if isinstance(imported, str):
            self._problem(node, f"cannot import {module}: {imported}")
            return None
        return imported

    # Names and attributes.

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_BUILTINS:
            self._problem(node, f"banned builtin {node.id}")
        self._identifier(node, node.id)
        if node.id in self._bindings and id(node) not in self._allowed_uses:
            action = "rebound" if isinstance(node.ctx, ast.Store | ast.Del) else "used as a value"
            self._problem(node, f"the module {node.id} {action}; only its attributes may be read")
        if node.id in _STRING_FORMS and id(node) not in self._allowed_uses:
            self._problem(node, f"{node.id} used other than called")
        if node.id == "self" and isinstance(node.ctx, ast.Store | ast.Del):
            self._problem(node, "self assigned")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._identifier(node, node.attr, attribute=True, owner=node.value)
        if node.attr in _STRING_FORMS and id(node) not in self._allowed_uses:
            self._problem(node, f"{node.attr} used other than called")
        if isinstance(node.value, ast.Name) and node.value.id in self._bindings:
            if isinstance(node.ctx, ast.Load):
                self._allowed_uses.add(id(node.value))
            self._check_module_attribute(node, node.value.id, node.attr)
        self.generic_visit(node)

    def _check_module_attribute(self, node: ast.AST, name: str, attribute: str) -> None:
        binding = self._bindings[name]
        if binding.imported is None:  # One of the pack's modules.
            if attribute in self._pack_imports.get(binding.module, {}):
                self._problem(node, f"{name}.{attribute} is a module")
            return
        if f"{binding.module}.{attribute}" in _TABLE:  # Imported, or not yet: a module either way.
            self._problem(node, f"{name}.{attribute} is a module")
            return
        value = getattr(binding.imported, attribute, None)
        if isinstance(value, ModuleType):
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
        if _is_dunder(name):
            self._problem(node, f"dunder name {name}")
            return
        bare = name.lstrip("_")
        if name in BANNED_NAMES or bare in BANNED_NAMES or name.startswith(BANNED_NAME_PREFIXES):
            self._problem(node, f"banned name {name}")
        if any(part in name.lower() for part in BANNED_SUBSTRINGS):
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
        if not node.name.strip("_"):
            self._problem(node, f"a class named {node.name}, which mangles nothing")
        self._identifier(node, node.name)
        for base in node.bases:  # class P(NamedTuple) uses annotations, which are checked.
            if isinstance(base, ast.Name | ast.Attribute):
                self._allowed_uses.add(id(base))
        for statement in node.body:
            # One leading underscore: a dataclass's __match_args__ would read it as an
            # attribute, which no rule sees (P8). A mangled __x is private for real.
            for bound, name in _class_bindings(statement):
                if name.startswith("_") and not name.startswith("__"):
                    self._problem(bound, f"a class binds {name}; use a mangled _{name}")
        outer = self._direct_class
        self._direct_class = node
        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1
        self._direct_class = outer

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # A class body may define __init__ and __post_init__, and no other dunder.
        in_class = self._direct_class is not None and node in self._direct_class.body
        if not (in_class and node.name in _CLASS_DUNDERS):
            self._identifier(node, node.name)
        if node.returns is not None:
            self._annotation(node.returns)
        outer = self._direct_class
        self._direct_class = None  # A function's body is no class body.
        self.generic_visit(node)
        self._direct_class = outer

    def visit_arg(self, node: ast.arg) -> None:
        self._identifier(node, node.arg)
        if node.annotation is not None:
            self._annotation(node.annotation)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg is not None:
            self._identifier(node, node.arg)
            if node.arg in BANNED_KEYWORDS:
                self._problem(node, f"banned keyword {node.arg}")
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

    def visit_Subscript(self, node: ast.Subscript) -> None:
        for inner in ast.walk(node.slice):
            string = isinstance(inner, ast.Constant) and isinstance(inner.value, str)
            if string or isinstance(inner, ast.JoinedStr):
                self._problem(inner, "a string in a subscript, which Python may evaluate")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        form = _called_form(node.func)
        if form is not None:
            self._allowed_uses.add(id(node.func))
            for problem in _string_form_problems(form, node):
                self._problem(node, problem)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str):
            return
        if node.value == "from_attributes":
            self._problem(node, "the string from_attributes")
        for dunder in dict.fromkeys(_DUNDER.findall(node.value)):
            if dunder not in _CLASS_DUNDERS:
                self._problem(node, f"a string names the dunder {dunder}")
        expression = _expression_in(node.value)
        if expression is not None:
            ast.increment_lineno(expression, node.lineno - 1)
            inner = _Checker(
                self._root, self._package, self._depth, self._pack_imports, in_string=True
            )
            inner._bindings = self._bindings
            inner.visit(expression)
            self.problems += inner.problems


def _class_bindings(statement: ast.stmt) -> Iterator[tuple[ast.AST, str]]:
    """The names a statement directly in a class body binds, each with its node."""
    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        yield statement, statement.name
        return
    if isinstance(statement, ast.Assign):
        targets: list[ast.expr] = list(statement.targets)
    elif isinstance(statement, ast.AnnAssign | ast.AugAssign):
        targets = [statement.target]
    else:
        return
    for target in targets:
        for inner in ast.walk(target):
            if isinstance(inner, ast.Name):
                yield inner, inner.id


def _called_form(called: ast.expr) -> str | None:
    """Which form that evaluates strings ``called`` names, if any."""
    if isinstance(called, ast.Name) and called.id in _STRING_FORMS:
        return called.id
    if isinstance(called, ast.Attribute) and called.attr in _STRING_FORMS:
        return called.attr
    return None


def _string_form_problems(form: str, call: ast.Call) -> Iterator[str]:
    """The strings a call of ``form`` would evaluate, and fields it does not list literally."""
    if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        yield f"{form} with unpacked arguments, which cannot be checked"
        return
    if form == "TypeAdapter":
        adapted = [*call.args, *_keywords(call, ("type",))]
        if any(_has_string(argument) for argument in adapted):
            yield "TypeAdapter of a string, which Python may evaluate"
        return
    # The first argument is the name of the type; its fields or types follow.
    rest = call.args[1:]
    if form in ("TypeVar", "NewType"):
        if any(_has_string(argument) for argument in [*rest, *_keywords(call, ("bound",))]):
            yield f"a string as a type in {form}, which Python may evaluate"
        return
    fields = rest[0] if rest else None
    if fields is not None and not isinstance(fields, ast.List | ast.Tuple | ast.Dict):
        yield f"{form} fields that are not listed literally"
        return
    # NamedTuple and TypedDict also take their fields as keywords; total is TypedDict's flag.
    named = [] if form == "make_dataclass" else _keywords(call, None, but=("total",))
    types: list[ast.expr] = [*named]
    if isinstance(fields, ast.Dict):
        types += fields.values
    elif fields is not None:
        for field in fields.elts:
            if isinstance(field, ast.Tuple | ast.List):
                types += field.elts[1:]
            elif not (isinstance(field, ast.Constant) and isinstance(field.value, str)):
                yield f"{form} fields that are not listed literally"
                return
    if any(_has_string(field_type) for field_type in types):
        yield f"a string as a field type in {form}, which Python may evaluate"


def _keywords(
    call: ast.Call, names: tuple[str, ...] | None, *, but: tuple[str, ...] = ()
) -> list[ast.expr]:
    """The values of the call's keywords among ``names``, or of all of them but ``but``."""
    return [
        keyword.value
        for keyword in call.keywords
        if (names is None or keyword.arg in names) and keyword.arg not in but
    ]


def _has_string(node: ast.expr) -> bool:
    return any(
        isinstance(inner, ast.Constant) and isinstance(inner.value, str) for inner in ast.walk(node)
    )


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def _expression_in(text: str) -> ast.expr | None:
    """The expression a string holds, if it parses as one and is more than a bare name or
    constant, which evaluate to nothing a rule could refuse.
    """
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if isinstance(tree.body, ast.Name | ast.Constant):
        return None
    return tree.body


def _in_pack(module: str, root: str) -> bool:
    return module == root or module.startswith(f"{root}.")


def _import(module: str) -> ModuleType | str:
    """Import a module of the table or of the core list, or say why it cannot be imported."""
    if module not in _TABLE and module not in _CORE_TABLE:
        return "it is not in the table"  # Never reached: callers check the table first.
    try:
        return importlib.import_module(module)
    except KeyboardInterrupt:
        raise
    except BaseException as error:  # noqa: BLE001  (an import that fails cannot be used)
        return type(error).__name__


def _module_name(root: str, relative: Path) -> str:
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((root, *parts))


def _package_of(module: str, relative: Path) -> str:
    return module if relative.name == "__init__.py" else module.rsplit(".", 1)[0]


def _imports_of(module: str, source: str, modules: Mapping[str, Path]) -> dict[str, _Binding]:
    """The names the imports of the pack module ``module`` bind to modules, in the pack or in
    the table. Nothing outside the table is imported; such imports are refused when the file
    itself is checked.
    """
    tree = _parse(source)
    if isinstance(tree, str):
        return {}  # Refused when the file itself is checked.
    package = module if modules[module].name == "__init__.py" else module.rsplit(".", 1)[0]
    found: dict[str, _Binding] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                target = alias.name if alias.asname else bound
                if target in _TABLE and not isinstance(imported := _import(target), str):
                    found[bound] = _Binding(target, imported)
        elif isinstance(node, ast.ImportFrom):
            base = package.rsplit(".", node.level - 1)[0] if node.level else ""
            source_module = ".".join(part for part in (base, node.module or "") if part)
            for alias in node.names:
                full = f"{source_module}.{alias.name}"
                if full in modules:
                    found[alias.asname or alias.name] = _Binding(full)
                elif full in _TABLE and not isinstance(imported := _import(full), str):
                    found[alias.asname or alias.name] = _Binding(full, imported)
    return found
