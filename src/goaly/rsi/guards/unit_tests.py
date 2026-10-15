"""GUARD_UNIT_TESTS (spec §8.2 step 3): the pack's tests pass in a fresh, read-only sandbox.

The tests are agent-written code, so they never run on the host. A one-shot sandbox mounts the
candidate's snapshot read-only and runs pytest on the lane's test directory, which writes nothing:
the image turns bytecode off, and the cache plugin is off. A Docker failure raises SandboxError,
which aborts the experiment rather than rejecting the candidate.

The guard passes only if pytest exits 0 and its summary counts at least one passed test, so a
suite that only skips, or a conftest that exits early, passes nothing. Its detail is pytest's
summary line alone: the rest of the report is agent-written text, which would reach the ledger
and the analyst's prompt.
"""

import re
import shlex
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import GUARD_UNIT_TESTS, GuardResult, Lane
from goaly.rsi.lanes import lane_info
from goaly.rsi.sandbox import SandboxConfig, run_once

__all__ = ["check_unit_tests"]

_NO_TESTS = frozenset({4, 5})  # pytest: a usage error, such as a missing path; nothing collected.
# pytest -q's summary, such as "1 failed, 2 passed in 0.10s", or "no tests ran in 0.01s".
_SUMMARY = re.compile(
    r"^(?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected)"
    r"(?:, \d+ \w+)*|no tests ran) in \d+(?:\.\d+)?s(?: \([\d:]+\))?$"
)
_PASSED = re.compile(r"(?:^|, )(\d+) passed")


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
    tests = lane_info(lane).tests_dir
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
    summary = _summary(result.output)
    passed_tests = int(found.group(1)) if (found := _PASSED.search(summary or "")) else 0
    if result.timed_out:
        detail = f"the pack's tests ran out of time after {timeout_s:g} s"
    elif result.exit_code in _NO_TESTS:
        detail = f"no test of the pack ran: {summary or f'pytest exited {result.exit_code}'}"
    elif summary is None:
        detail = f"pytest exited {result.exit_code} without its summary"
    else:
        detail = summary
    passed = result.exit_code == 0 and not result.timed_out and passed_tests > 0
    if result.exit_code == 0 and not result.timed_out and passed_tests == 0:
        detail = f"no test of the pack passed: {detail}"
    return GuardResult(guard=GUARD_UNIT_TESTS, passed=passed, detail=detail)


def _summary(output: str) -> str | None:
    """pytest's summary line, the last line of its report that reads as one, or None."""
    for line in reversed(output.splitlines()):
        stripped = line.strip().strip("=").strip()
        if _SUMMARY.fullmatch(stripped):
            return stripped
    return None
