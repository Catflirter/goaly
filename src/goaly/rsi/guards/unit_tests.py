"""GUARD_UNIT_TESTS (spec §8.2 step 3): the pack's tests pass in a fresh, read-only sandbox.

The tests are agent-written code, so they never run on the host. A one-shot sandbox mounts the
candidate's snapshot read-only and runs pytest on the lane's test directory, which writes nothing:
the image turns bytecode off, and the cache plugin is off. A Docker failure raises SandboxError,
which aborts the experiment rather than rejecting the candidate.
"""

import shlex
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import GUARD_UNIT_TESTS, GuardResult, Lane
from goaly.rsi.lanes import lane_info
from goaly.rsi.sandbox import SandboxConfig, run_once

__all__ = ["check_unit_tests"]

_NO_TESTS = frozenset({4, 5})  # pytest: a usage error, such as a missing path; nothing collected.
_REPORT_LINES = 40  # The end of pytest's report the guard keeps.


def check_unit_tests(
    snapshot: Path,
    lane: Lane.ValueType,
    *,
    experiment_id: str,
    image: str,
    timeout_s: float = 600.0,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008  (frozen, so one default is safe)
) -> GuardResult:
    """Run the pack's tests from ``snapshot``, a candidate's, in a sandbox of ``image``."""
    tests = lane_info(lane).pack_dirs[1]
    result = run_once(
        image,
        snapshot,
        lane,
        experiment_id=experiment_id,
        purpose="unit-tests",
        command=f"python -m pytest -q -p no:cacheprovider {shlex.quote(str(tests))}",
        timeout_s=timeout_s,
        config=config,
    )
    report = "\n".join(result.output.strip().splitlines()[-_REPORT_LINES:])
    if result.timed_out:
        detail = f"the pack's tests ran out of time after {timeout_s:g} s"
    elif result.exit_code in _NO_TESTS:
        detail = f"no test of the pack ran: {report}"
    else:
        detail = report
    passed = result.exit_code == 0 and not result.timed_out
    if passed:
        detail = report.splitlines()[-1] if report else "the pack's tests passed"
    return GuardResult(guard=GUARD_UNIT_TESTS, passed=passed, detail=detail)
