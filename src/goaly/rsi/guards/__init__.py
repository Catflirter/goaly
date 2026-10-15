"""The guards a candidate must pass before it is evaluated (spec §8.2 step 3, §8.5).

Each guard checks a candidate's commit, through git or a read-only snapshot of it
(goaly.rsi.workspace.Workspace.snapshot), never its live worktree, and returns a GuardResult.
:func:`run_guards` runs them all in the spec's order.
"""

from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from goaly.contracts.gen.experiment_pb2 import GuardResult, Lane
from goaly.rsi.guards.allowlist import check_pack_code
from goaly.rsi.guards.checksums import check_checksums
from goaly.rsi.guards.gold_windows import GoldWindows
from goaly.rsi.guards.hardcoding import HardCodingConfig, check_hard_coding
from goaly.rsi.guards.paths import check_allowed_paths
from goaly.rsi.guards.unit_tests import check_unit_tests
from goaly.rsi.sandbox import SandboxConfig
from goaly.rsi.workspace import Workspace

__all__ = ["GuardsConfig", "run_guards"]


class GuardsConfig(BaseModel):
    """The guards' settings, from config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hard_coding: HardCodingConfig = HardCodingConfig()
    unit_test_timeout_s: float = Field(default=600.0, gt=0)
    sandbox: SandboxConfig = SandboxConfig()


def run_guards(
    workspace: Workspace,
    lane: Lane.ValueType,
    *,
    experiment_id: str,
    parent: str,
    candidate: str,
    gold: GoldWindows,
    frozen: Mapping[str, str],
    image: str,
    config: GuardsConfig = GuardsConfig(),  # noqa: B008  (frozen, so one default is safe)
) -> list[GuardResult]:
    """Run the guards on the commit ``candidate``, built on ``parent``, until one fails, and
    return their results in the order they ran, as ExperimentRecord.guards keeps them.

    The order is the spec's: allowed paths, checksums, the hard-coding scan, the pack code
    allowlist and the unit tests. Allowed paths reads git alone; the others read one snapshot
    of the commit. ``frozen`` holds the lane's reference checksums (see
    goaly.rsi.guards.checksums), and ``image`` is the sandbox image (goaly.rsi.sandbox).

    Raises goaly.rsi.sandbox.SandboxError if Docker fails, which aborts the experiment rather
    than rejecting the candidate.
    """
    paths = check_allowed_paths(workspace.repo, lane, parent=parent, candidate=candidate)
    if not paths.passed:
        return [paths]
    results = [paths]
    with workspace.snapshot(candidate) as snapshot:
        guards: list[Callable[[], GuardResult]] = [
            lambda: check_checksums(snapshot, frozen),
            lambda: check_hard_coding(
                workspace.repo,
                lane,
                parent=parent,
                candidate=candidate,
                snapshot=snapshot,
                gold=gold,
                config=config.hard_coding,
            ),
            lambda: check_pack_code(snapshot, lane),
            lambda: check_unit_tests(
                snapshot,
                lane,
                experiment_id=experiment_id,
                image=image,
                timeout_s=config.unit_test_timeout_s,
                config=config.sandbox,
            ),
        ]
        for guard in guards:
            result = guard()
            results.append(result)
            if not result.passed:
                break
    return results
