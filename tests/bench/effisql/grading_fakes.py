"""Fake grading sessions: canned plans, results and times per query, and a record of each call."""

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grader import DigestBuilder, ResultDigest
from goaly.bench.effisql.grading import BaseEntry, GradingSessions

type Row = tuple[Any, ...]

BASE_SQL = "SELECT l_returnflag, count(*) FROM lineitem GROUP BY l_returnflag"
INST = EffiSqlInstance.model_validate({"id": 7, "db": "tpch", "base_sql": BASE_SQL})
BASE_ROWS: list[Row] = [("A", 10), ("N", 20), ("R", 30)]
ALT_ROWS: list[Row] = [("A", 1), ("N", 2), ("R", 3)]


def plan(*relations: str, kind: str = "Seq Scan") -> dict[str, Any]:
    scans = [
        {"Node Type": kind, "Relation Name": r, "Parent Relationship": "Outer"} for r in relations
    ]
    if not scans:
        return {"Plan": {"Node Type": "Result"}}
    return {"Plan": {"Node Type": "HashAggregate", "Plans": scans}}


def digest_of(rows: Sequence[Row]) -> ResultDigest:
    builder = DigestBuilder()
    builder.add(rows)
    return builder.digest()


class FakeConnection:
    def __init__(self) -> None:
        self.cancels = 0

    def cancel_safe(self) -> None:
        self.cancels += 1


class FakeSession:
    """Answers plan, digest and execution_ms from canned values, keyed by the exact query."""

    def __init__(
        self,
        *,
        plans: Mapping[str, dict[str, Any] | Exception] | None = None,
        results: Mapping[str, Sequence[Row] | Exception] | None = None,
        times: Mapping[str, Sequence[float | Exception]] | None = None,
    ) -> None:
        self.plans = dict(plans or {})
        self.results = dict(results or {})
        self.times = {sql: list(values) for sql, values in (times or {}).items()}
        self.calls: list[tuple[str, str, int | None]] = []
        self.connection = FakeConnection()
        self.closed = False

    def plan(self, sql: str) -> dict[str, Any]:
        self.calls.append(("plan", sql, None))
        answer = self.plans[sql]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def digest(self, sql: str, *, row_limit: int | None = None) -> ResultDigest:
        self.calls.append(("digest", sql, row_limit))
        answer = self.results[sql]
        if isinstance(answer, Exception):
            raise answer
        builder = DigestBuilder()
        for row in answer:
            builder.add([row])
            if row_limit is not None and builder.row_count > row_limit:
                return builder.digest(complete=False)
        return builder.digest()

    def execution_ms(self, sql: str) -> float:
        self.calls.append(("time", sql, None))
        value = self.times[sql].pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self) -> None:
        self.closed = True

    def kinds(self, sql: str) -> list[str]:
        return [kind for kind, query, _ in self.calls if query == sql]


def sessions(main: FakeSession, alt: FakeSession | None = None) -> GradingSessions:
    return GradingSessions(main, alt or FakeSession(), instance="grade-1")  # type: ignore[arg-type]  # Structural fakes.


def entry(**changes: Any) -> BaseEntry:
    base = digest_of(BASE_ROWS)
    alt = digest_of(ALT_ROWS)
    fields: dict[str, Any] = {
        "instance_id": INST.instance_id,
        "base_sql_sha256": hashlib.sha256(BASE_SQL.encode()).hexdigest(),
        "row_count": base.row_count,
        "row_hash_sum": f"{base.row_hash_sum:064x}",
        "alt_row_count": alt.row_count,
        "alt_row_hash_sum": f"{alt.row_hash_sum:064x}",
        "base_ms": 1000.0,
        "timing_runs": 3,
        "relations": ("lineitem",),
        "identity_speedup": 1.02,
        "measured_on": "grade-1",
        "measured_at": datetime(2026, 10, 15, tzinfo=UTC),
    }
    fields.update(changes)
    return BaseEntry(**fields)
