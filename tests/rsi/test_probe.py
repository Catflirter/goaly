"""Tests for goaly.rsi.probe: the periodic identity probes of the grading instances."""

import logging

import pytest

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grader import GraderError
from goaly.bench.effisql.grading_pool import GRADING_INSTANCES, GradingInstance, Priority
from goaly.rsi.probe import ProbeReading, pick_task, probe

pytestmark = pytest.mark.anyio


def task(task_id: int, base_ms: float | None) -> EffiSqlInstance:
    record: dict[str, object] = {"id": task_id, "db": "tpch", "base_sql": "SELECT 1"}
    if base_ms is not None:
        record["base_time"] = base_ms
    return EffiSqlInstance.model_validate(record)


class FakeGrader:
    """τ by instance name; a missing name fails, as a probe that cannot run does."""

    def __init__(self, taus: dict[str, float], reference: float | None = 1.0) -> None:
        self.taus = taus
        self.reference = reference
        self.calls: list[tuple[str, str, Priority]] = []

    async def identity_probe(
        self, inst: EffiSqlInstance, *, instance: GradingInstance, priority: Priority = Priority.RSI
    ) -> float:
        self.calls.append((inst.instance_id, instance.name, priority))
        if instance.name not in self.taus:
            raise GraderError("the instance is down")
        return self.taus[instance.name]

    def identity_reference(self, inst: EffiSqlInstance) -> float | None:
        return self.reference


async def test_each_instance_is_probed_and_drift_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    grader = FakeGrader({"grade-1": 1.05, "grade-2": 0.85})
    caplog.set_level(logging.WARNING)
    readings = await probe(grader, task(7, 9000), GRADING_INSTANCES)
    assert [r.instance for r in readings] == ["grade-1", "grade-2", "grade-3"]
    assert {c[2] for c in grader.calls} == {Priority.RSI}
    first, second, third = readings
    assert first.drift == pytest.approx(0.05)
    assert second.drift == pytest.approx(-0.15)
    assert third.tau is None
    assert third.error == "GraderError: the instance is down"
    events = [getattr(r, "event", None) for r in caplog.records]
    assert events.count("identity_drift") == 1  # grade-2 only: 5% is within the limit.
    assert events.count("identity_probe_failed") == 1


async def test_no_reference_means_no_drift() -> None:
    (reading,) = await probe(
        FakeGrader({"grade-1": 2.0}, None), task(7, None), GRADING_INSTANCES[:1]
    )
    assert reading.drift is None
    assert ProbeReading(instance="x", task="1", tau=1.0, reference=0.0).drift is None


def test_the_probe_task_has_a_base_near_ten_seconds() -> None:
    tasks = [task(1, 500), task(2, 11_000), task(3, 60_000), task(4, None)]
    assert pick_task(tasks).instance_id == "2"
    assert pick_task([task(5, None)]).instance_id == "5"
    with pytest.raises(ValueError, match="at least one"):
        pick_task([])
