"""A fake psycopg connection for unit tests of the Effi-SQL PostgreSQL code: no server needed.

It records every statement as text, answers queries from canned responses chosen by a substring
of the query, and keeps what COPY wrote. Tests pass it where a ``Connect`` is expected through
:meth:`FakeServer.connect`, which also records which database each connection was opened to.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from psycopg import sql

type Answer = Sequence[tuple[Any, ...]] | BaseException | Callable[[Any], Sequence[tuple[Any, ...]]]


def _fake_verifier(password: bytes, user: bytes, algorithm: bytes | None = None) -> bytes:
    """Stands in for libpq's PQencryptPasswordConn; never contains the password."""
    return b"SCRAM-SHA-256$fake:" + user + b":" + str(len(password)).encode()


def as_text(query: object) -> str:
    if isinstance(query, sql.Composable):
        return query.as_string(None)
    if isinstance(query, bytes):
        return query.decode()
    return str(query)


class FakeCursor:
    def __init__(self, connection: "FakeConnection", rows: Sequence[tuple[Any, ...]] = ()) -> None:
        self._connection = connection
        self._rows = list(rows)
        self.adapters = SimpleNamespace(
            register_loader=lambda *args: connection.loaders.append(args)
        )

    def execute(
        self, query: object, params: object = None, *, prepare: bool | None = None
    ) -> "FakeCursor":
        return self._connection.execute(query, params, prepare=prepare)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        return iter(self._rows)

    @contextmanager
    def copy(self, query: object) -> Iterator["FakeCopy"]:
        text = as_text(query)
        self._connection.statements.append(text)
        sink = FakeCopy()
        yield sink
        self._connection.copied[text] = b"".join(sink.blocks)

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeCopy:
    def __init__(self) -> None:
        self.blocks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.blocks.append(bytes(data))


class FakeConnection:
    def __init__(self, dbname: str, responses: Sequence[tuple[str, Answer]] = ()) -> None:
        self.statements: list[str] = []
        self.copied: dict[str, bytes] = {}
        self.responses = list(responses)
        self.prepared: list[str] = []
        self.info = SimpleNamespace(dbname=dbname)
        self.loaders: list[tuple[str, object]] = []
        self.adapters = SimpleNamespace(register_loader=lambda *args: self.loaders.append(args))
        self.pgconn = SimpleNamespace(encrypt_password=_fake_verifier)
        self.closed = False

    def execute(
        self, query: object, params: object = None, *, prepare: bool | None = None
    ) -> FakeCursor:
        text = as_text(query)
        self.statements.append(text)
        if prepare:
            self.prepared.append(text)
        for needle, answer in self.responses:
            if needle in text:
                if isinstance(answer, BaseException):
                    raise answer
                rows = answer(params) if callable(answer) else answer
                return FakeCursor(self, rows)
        return FakeCursor(self)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.statements.append("BEGIN")
        try:
            yield
        except BaseException:
            self.statements.append("ROLLBACK")
            raise
        self.statements.append("COMMIT")

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True


class FakeServer:
    """Hands out FakeConnections; ``responses`` answer queries on every connection."""

    def __init__(self, responses: Sequence[tuple[str, Answer]] = ()) -> None:
        self.responses = list(responses)
        self.connections: list[FakeConnection] = []

    def connect(self, dbname: str) -> Any:  # A psycopg.Connection, as far as the code can tell.
        connection = FakeConnection(dbname, self.responses)
        self.connections.append(connection)
        return connection

    def statements(self, dbname: str) -> list[str]:
        return [s for c in self.connections if c.info.dbname == dbname for s in c.statements]


class AsyncFakeCursor:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    async def fetchone(self) -> tuple[Any, ...] | None:
        return self._cursor.fetchone()

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self._cursor.fetchall()


class AsyncFakeConnection:
    """The async face of a FakeConnection: same canned answers, same record of statements."""

    def __init__(self, dbname: str, responses: Sequence[tuple[str, Answer]] = ()) -> None:
        self.sync = FakeConnection(dbname, responses)
        self.info = self.sync.info

    @property
    def statements(self) -> list[str]:
        return self.sync.statements

    async def execute(
        self, query: object, params: object = None, *, prepare: bool | None = None
    ) -> AsyncFakeCursor:
        return AsyncFakeCursor(self.sync.execute(query, params, prepare=prepare))
