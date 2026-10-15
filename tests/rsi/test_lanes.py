"""Tests for goaly.rsi.lanes: each lane's names, refs and pack directories."""

from pathlib import PurePosixPath

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, LANE_SWE, LANE_UNSPECIFIED, Lane
from goaly.rsi.lanes import lane_info


def test_effisql_lane() -> None:
    info = lane_info(LANE_EFFISQL)

    assert info.lane == LANE_EFFISQL
    assert info.name == "effisql"
    assert info.title == "Effi-SQL"
    assert info.best_tag == "best/effisql"
    assert info.branch("exp-7") == "rsi/effisql/exp-7"
    assert info.pack_dirs == (
        PurePosixPath("src/goaly/harness/packs/effisql"),
        PurePosixPath("tests/harness/packs/effisql"),
    )


def test_the_swe_lane_names_its_refs_swe_and_its_pack_swefficiency() -> None:
    # Spec §3.2: SWE-fficiency keeps its benchmark name for its pack and best/swe for its tag.
    info = lane_info(LANE_SWE)

    assert info.name == "swe"
    assert info.title == "SWE-fficiency"
    assert info.best_tag == "best/swe"
    assert info.branch("exp-7") == "rsi/swe/exp-7"
    assert info.pack_dirs == (
        PurePosixPath("src/goaly/harness/packs/swefficiency"),
        PurePosixPath("tests/harness/packs/swefficiency"),
    )


@pytest.mark.parametrize("lane", [LANE_UNSPECIFIED, Lane.ValueType(99)])
def test_an_unspecified_or_unknown_lane_is_an_error(lane: Lane.ValueType) -> None:
    with pytest.raises(ValueError, match="not an RSI lane"):
        lane_info(lane)
