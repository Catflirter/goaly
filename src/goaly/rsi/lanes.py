"""The RSI lanes (spec §8.1) and the names each one uses for its refs and pack directories."""

from dataclasses import dataclass
from pathlib import PurePosixPath

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, LANE_SWE, Lane

__all__ = ["LaneInfo", "lane_info"]


@dataclass(frozen=True)
class LaneInfo:
    """One lane's names (spec §3.2)."""

    lane: Lane.ValueType
    # The lane's name in its refs, worktree paths and sandbox labels: best/<name>, rsi/<name>/...
    name: str
    # Its benchmark's name, which is also its pack's directory name.
    benchmark: str
    # The name people read, as in lessons.md.
    title: str

    @property
    def best_tag(self) -> str:
        """The tag that marks the lane's best version."""
        return f"best/{self.name}"

    def branch(self, experiment_id: str) -> str:
        """The branch of an experiment's candidate, kept even when the candidate is rejected."""
        return f"rsi/{self.name}/{experiment_id}"

    @property
    def pack_dirs(self) -> tuple[PurePosixPath, PurePosixPath]:
        """The only directories the lane may change, relative to the repository root (spec §8.2)."""
        return (
            PurePosixPath("src/goaly/harness/packs") / self.benchmark,
            PurePosixPath("tests/harness/packs") / self.benchmark,
        )


_LANES = {
    LANE_EFFISQL: LaneInfo(LANE_EFFISQL, name="effisql", benchmark="effisql", title="Effi-SQL"),
    # On hold since 2026-10-15; it keeps its names so that it can resume.
    LANE_SWE: LaneInfo(LANE_SWE, name="swe", benchmark="swefficiency", title="SWE-fficiency"),
}


def lane_info(lane: Lane.ValueType) -> LaneInfo:
    """Return ``lane``'s names; raise ValueError for LANE_UNSPECIFIED or an unknown value."""
    try:
        return _LANES[lane]
    except KeyError:
        raise ValueError(f"{lane!r} is not an RSI lane") from None
