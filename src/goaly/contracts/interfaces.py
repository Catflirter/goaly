"""Behavioral interfaces: the Protocols the layers call each other through (spec §3.3).

bench implements the environments, and core calls them only through these Protocols, so harness
unit tests can use fakes, with no database or container.

gRPC-ready: every method takes and returns only contract messages (``goaly.contracts.gen``) or
plain values, never framework objects. A gRPC-backed implementation could then replace an
in-process one without changing any caller. tests/contracts/test_interfaces.py enforces the rule.
"""

from typing import Protocol

from goaly.contracts.gen.environment_pb2 import ShellResult, SqlResult

__all__ = ["Environment", "ShellEnvironment", "SqlEnvironment"]


class Environment(Protocol):
    """The runtime environment of one task: what the agent's tools act on.

    bench implements it, as a PostgreSQL sandbox (SqlEnvironment) or a Docker container
    (ShellEnvironment); harness unit tests use fakes. It belongs to whoever created it, normally
    the context manager that Benchmark.provision returns, which closes it.
    """

    async def close(self) -> None:
        """Release the environment's resources. Calling it again does nothing."""


class SqlEnvironment(Environment, Protocol):
    """A task's sandbox database, which BIRD-CRITIC's tools query: bench's PostgreSQL sandbox."""

    async def execute(self, sql: str, *, timeout_s: float) -> SqlResult:
        """Run ``sql``, stopping it after ``timeout_s`` seconds, and return its result.

        A statement that fails or times out gives a result with ``error`` set, not an exception.
        An exception means that the environment itself failed.
        """


class ShellEnvironment(Environment, Protocol):
    """A task's container, which SWE-fficiency's tools work in: bench's Docker environment.

    Paths are inside the container, where the repository is at /testbed.
    """

    async def exec(self, command: str, *, timeout_s: float) -> ShellResult:
        """Run ``command`` in a shell, stopping it after ``timeout_s`` seconds.

        A command that fails or times out gives a result, with ``exit_code`` or ``timed_out``
        saying so, not an exception. An exception means that the environment itself failed.
        """

    async def read_file(self, path: str) -> str:
        """Return the text of the file at ``path``. Raise FileNotFoundError if there is none."""

    async def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to the file at ``path``, replacing the file if it exists."""
