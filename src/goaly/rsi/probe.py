"""The periodic identity probes (spec §6.2): is each grading instance still timing as at setup?

Once per round, before the round's validations, the lane probes every grading instance with one
task's base query timed against itself (T10's EffiSqlGrader.identity_probe), and compares the τ
with the one the setup measured for that task. A drift beyond ±10% is logged as a WARNING, and
the readings go into the round log; the lane goes on either way, since a drifting instance says
nothing about any hypothesis. A probe that cannot run is a reading with its error.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grading_pool import GradingInstance, Priority

__all__ = ["DRIFT_LIMIT", "PROBE_BASE_MS", "IdentityGrader", "ProbeReading", "pick_task", "probe"]

DRIFT_LIMIT: Final = 0.10  # Beyond ±10% of the setup's τ an instance has drifted.
PROBE_BASE_MS: Final = 10_000.0  # A base of about 10 s keeps a probe cheap (T10).

_logger = logging.getLogger(__name__)


class IdentityGrader(Protocol):
    """What the probe needs of T10's EffiSqlGrader: the lane shares the evaluator's own."""

    async def identity_probe(
        self, inst: EffiSqlInstance, *, instance: GradingInstance, priority: Priority = ...
    ) -> float: ...

    def identity_reference(self, inst: EffiSqlInstance) -> float | None: ...


class ProbeReading(BaseModel):
    """One instance's probe: its τ now and the setup's, or why it could not run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance: str
    task: str
    tau: float | None = None
    reference: float | None = None
    error: str = ""

    @property
    def drift(self) -> float | None:
        """τ now over the setup's, less 1; None unless both are known."""
        if self.tau is None or not self.reference:
            return None
        return self.tau / self.reference - 1


def pick_task(instances: Sequence[EffiSqlInstance]) -> EffiSqlInstance:
    """The task whose base time is nearest PROBE_BASE_MS; the first if none has a time."""
    if not instances:
        raise ValueError("an identity probe needs at least one task")
    timed = [inst for inst in instances if inst.base_time is not None]
    if not timed:
        return instances[0]
    return min(timed, key=lambda inst: abs(float(inst.base_time or 0) - PROBE_BASE_MS))


async def probe(
    grader: IdentityGrader, inst: EffiSqlInstance, instances: Sequence[GradingInstance]
) -> list[ProbeReading]:
    """Probe each of ``instances`` with ``inst``, concurrently; never raises for a probe."""
    reference = grader.identity_reference(inst)
    readings = await asyncio.gather(*(_one(grader, inst, gi, reference) for gi in instances))
    return list(readings)


async def _one(
    grader: IdentityGrader,
    inst: EffiSqlInstance,
    instance: GradingInstance,
    reference: float | None,
) -> ProbeReading:
    base: dict[str, object] = {
        "instance": instance.name,
        "task": inst.instance_id,
        "reference": reference,
    }
    try:
        tau = await grader.identity_probe(inst, instance=instance, priority=Priority.RSI)
    except Exception as error:  # noqa: BLE001  (GraderError or other: the next probe retries)
        _logger.warning(
            "the identity probe on %s failed: %s",
            instance.name,
            error,
            extra={"event": "identity_probe_failed", **base},
        )
        return ProbeReading(
            instance=instance.name,
            task=inst.instance_id,
            reference=reference,
            error=f"{type(error).__name__}: {error}",
        )
    reading = ProbeReading(
        instance=instance.name, task=inst.instance_id, reference=reference, tau=tau
    )
    drift = reading.drift
    if drift is not None and abs(drift) > DRIFT_LIMIT:
        _logger.warning(
            "grading instance %s drifted: identity τ %.3f against the setup's %.3f",
            instance.name,
            tau,
            reference,
            extra={"event": "identity_drift", "tau": tau, "drift": drift, **base},
        )
    return reading
