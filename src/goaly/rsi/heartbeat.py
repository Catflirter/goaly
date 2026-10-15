"""The lane process's heartbeat, which the supervisor watches (spec §8.7).

A lane runs :func:`heartbeat` for its whole life, on its own event loop, so a loop that hangs
stops beating. Every 30 s it rewrites the lane's heartbeat file with the current time. The
supervisor kills a lane whose file is 10 minutes old (:data:`STALE_AFTER_S`) and counts it as
crashed.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import Lane
from goaly.rsi.lanes import lane_info

__all__ = ["INTERVAL_S", "STALE_AFTER_S", "beat", "heartbeat", "heartbeat_age", "heartbeat_file"]

INTERVAL_S = 30.0
STALE_AFTER_S = 600.0

_logger = logging.getLogger(__name__)


def heartbeat_file(state_dir: Path, lane: Lane.ValueType) -> Path:
    """Return the heartbeat file of ``lane`` in the supervisor's ``state_dir``."""
    return state_dir / f"{lane_info(lane).name}.heartbeat"


def beat(path: Path) -> None:
    """Write the current time to ``path``, replacing it atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(f"{time.time():.3f}\n", encoding="utf-8")
    temporary.replace(path)


def heartbeat_age(path: Path, *, now: float | None = None) -> float | None:
    """Return how many seconds ago ``path`` last beat, or None if it never has."""
    try:
        last = float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return max(0.0, (time.time() if now is None else now) - last)


@contextlib.asynccontextmanager
async def heartbeat(path: Path, interval_s: float = INTERVAL_S) -> AsyncIterator[None]:
    """Beat on ``path`` now and every ``interval_s`` while the block runs.

    A beat that fails, as on a full disk, is logged and tried again at the next interval; the
    supervisor then sees the file go stale. The supervisor deletes the file before it starts a
    lane, and until the first beat counts the lane's age from its start.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    beat(path)

    async def keep_beating() -> None:
        while True:
            await asyncio.sleep(interval_s)
            try:
                beat(path)
            except Exception:  # Any failure: log it and try at the next beat.
                _logger.exception("could not write the heartbeat", extra={"path": str(path)})

    task = asyncio.create_task(keep_beating(), name="goaly-heartbeat")
    try:
        yield
    finally:
        task.cancel()
        # wait() never raises, so a cancellation of the block's own task is not swallowed.
        await asyncio.wait({task})
