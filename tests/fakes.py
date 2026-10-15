"""Fakes of the goaly.contracts.interfaces Protocols, for unit tests.

They need no database, container or model, as spec §3.3 asks of harness unit tests. Each is a
small, deterministic implementation that records how it was called.
tests/contracts/test_interfaces.py proves, under mypy and at runtime, that each conforms to its
Protocol.

pytest imports tests/ as a namespace package, so any test can import them, e.g.
``from tests.fakes import FakeSqlEnvironment``.
"""

from collections.abc import Mapping

from goaly.contracts.gen.environment_pb2 import ShellResult, SqlResult

# The exit status a shell gives for a command it cannot find.
COMMAND_NOT_FOUND = 127


class FakeEnvironment:
    """An Environment with nothing in it. Once closed, its subclasses refuse every call."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("the environment is closed")


class FakeSqlEnvironment(FakeEnvironment):
    """A SqlEnvironment that answers each statement from a table of canned results.

    A statement missing from the table gets a result with ``error`` set, as a failing statement
    does. ``executed`` records each statement with its timeout.
    """

    def __init__(self, results: Mapping[str, SqlResult] | None = None) -> None:
        super().__init__()
        self.results = dict(results or {})
        self.executed: list[tuple[str, float]] = []

    async def execute(self, sql: str, *, timeout_s: float) -> SqlResult:
        self._ensure_open()
        self.executed.append((sql, timeout_s))
        result = SqlResult()
        if sql in self.results:
            result.CopyFrom(self.results[sql])  # A copy: the caller cannot change the table.
        else:
            result.error = f"no canned result for {sql!r}"
        return result


class FakeShellEnvironment(FakeEnvironment):
    """A ShellEnvironment with canned command results and files kept in memory.

    A command missing from the table exits with status 127, as an unknown command does.
    ``commands`` records each command with its timeout, and ``files`` maps paths to contents.
    """

    def __init__(
        self,
        results: Mapping[str, ShellResult] | None = None,
        files: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.results = dict(results or {})
        self.files = dict(files or {})
        self.commands: list[tuple[str, float]] = []

    async def exec(self, command: str, *, timeout_s: float) -> ShellResult:
        self._ensure_open()
        self.commands.append((command, timeout_s))
        result = ShellResult()
        if command in self.results:
            result.CopyFrom(self.results[command])  # A copy: the caller cannot change the table.
        else:
            result.exit_code = COMMAND_NOT_FOUND
            result.stderr = f"no canned result for {command!r}"
        return result

    async def read_file(self, path: str) -> str:
        self._ensure_open()
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_file(self, path: str, content: str) -> None:
        self._ensure_open()
        self.files[path] = content
