"""Work for ProcessRunner tests: module-level, so a spawned worker can import and run it."""

import os
import time

from goaly.bench.effisql.grading import GradingSlot


def instance_name(slot: GradingSlot) -> str:
    return slot.instance.name


def die(_slot: GradingSlot) -> None:
    os._exit(3)  # A worker crash, as an OOM kill would end it.


def wait_for_abort(slot: GradingSlot) -> str:
    """Wait until this call is aborted, as a long statement would until it is cancelled."""
    deadline = time.monotonic() + 20
    while not slot._aborted.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    return "aborted" if slot._aborted.is_set() else "never aborted"


def was_aborted(slot: GradingSlot) -> bool:
    return slot._aborted.is_set()
