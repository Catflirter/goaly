"""Tests for goaly.rsi.hypothesis_queue: a durable FIFO of hypotheses, one per lane."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import (
    CHANGE_KIND_PROMPT,
    HYPOTHESIS_SOURCE_ANALYST,
    HYPOTHESIS_SOURCE_UNSPECIFIED,
    LANE_EFFISQL,
    LANE_SWE,
    LANE_UNSPECIFIED,
    Hypothesis,
)
from goaly.contracts.jsonl import to_json_line
from goaly.rsi.hypothesis_queue import HypothesisQueue


def hypothesis(hypothesis_id: str, change: str = "Tighten the prompt.") -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        failure_pattern="timid rewrites",
        target_instance_ids=["101"],
        change_kind=CHANGE_KIND_PROMPT,
        change=change,
        source=HYPOTHESIS_SOURCE_ANALYST,
    )


def ids(queue: HypothesisQueue) -> list[str]:
    return [queued.hypothesis_id for queued in queue.pending()]


def test_an_absent_queue_is_empty_and_reading_creates_nothing(tmp_path: Path) -> None:
    queue = HypothesisQueue(tmp_path / "ledger", LANE_EFFISQL)
    assert queue.pending() == []
    assert not (tmp_path / "ledger").exists()


def test_hypotheses_leave_in_the_order_they_came(tmp_path: Path) -> None:
    queue = HypothesisQueue(tmp_path / "ledger", LANE_EFFISQL)
    queue.push([hypothesis("h-1"), hypothesis("h-2")])
    queue.push([hypothesis("h-3"), hypothesis("h-1", change="Changed."), hypothesis("h-3")])
    assert ids(queue) == ["h-1", "h-2", "h-3"]
    assert queue.pending()[0].change == "Tighten the prompt."  # A queued one keeps its place.
    assert [taken.hypothesis_id for taken in queue.remove(["h-3", "h-1", "h-9"])] == ["h-1", "h-3"]
    assert ids(queue) == ["h-2"]
    assert queue.remove([]) == []


def test_the_queue_survives_a_reopen_and_another_process(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    HypothesisQueue(directory, LANE_EFFISQL).push([hypothesis("h-1")])
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, Hypothesis
        from goaly.contracts.jsonl import from_json_line
        from goaly.rsi.hypothesis_queue import HypothesisQueue

        queue = HypothesisQueue(Path(sys.argv[1]), LANE_EFFISQL)
        assert [h.hypothesis_id for h in queue.pending()] == ["h-1"]
        queue.push([from_json_line(sys.argv[2], Hypothesis)])
        """
    )
    subprocess.run(  # noqa: S603 - the test's own interpreter and script.
        [sys.executable, "-c", script, str(directory), to_json_line(hypothesis("h-2"))],
        check=True,
        timeout=60,
    )
    reopened = HypothesisQueue(directory, LANE_EFFISQL)
    assert reopened.pending() == [hypothesis("h-1"), hypothesis("h-2")]


def test_each_lane_has_its_own_queue(tmp_path: Path) -> None:
    HypothesisQueue(tmp_path, LANE_EFFISQL).push([hypothesis("h-1")])
    assert HypothesisQueue(tmp_path, LANE_SWE).pending() == []
    assert (tmp_path / "hypothesis_queue-effisql.jsonl").exists()
    with pytest.raises(ValueError, match="not an RSI lane"):
        HypothesisQueue(tmp_path, LANE_UNSPECIFIED)


@pytest.mark.parametrize(
    ("invalid", "says"),
    [
        (hypothesis("bad id"), "hypothesis_id"),
        (Hypothesis(hypothesis_id="h-2", source=HYPOTHESIS_SOURCE_ANALYST), "no change kind"),
        (
            Hypothesis(
                hypothesis_id="h-2",
                change_kind=CHANGE_KIND_PROMPT,
                source=HYPOTHESIS_SOURCE_UNSPECIFIED,
            ),
            "no source",
        ),
    ],
)
def test_an_invalid_hypothesis_changes_nothing(
    tmp_path: Path, invalid: Hypothesis, says: str
) -> None:
    queue = HypothesisQueue(tmp_path, LANE_EFFISQL)
    queue.push([hypothesis("h-1")])
    with pytest.raises(ValueError, match=says):
        queue.push([hypothesis("h-3"), invalid])
    assert ids(queue) == ["h-1"]


def test_a_failed_write_keeps_the_old_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "ledger"
    queue = HypothesisQueue(directory, LANE_EFFISQL)
    queue.push([hypothesis("h-1")])

    def broken(descriptor: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("goaly.rsi.hypothesis_queue.os.fsync", broken)
    with pytest.raises(OSError, match="disk full"):
        queue.push([hypothesis("h-2")])
    monkeypatch.undo()
    assert ids(queue) == ["h-1"]
    assert sorted(path.name for path in directory.iterdir()) == [
        ".hypothesis_queue-effisql.lock",
        "hypothesis_queue-effisql.jsonl",
    ]


def test_an_unchanged_queue_is_not_rewritten(tmp_path: Path) -> None:
    queue = HypothesisQueue(tmp_path, LANE_EFFISQL)
    queue.push([hypothesis("h-1")])
    before = queue.path.stat().st_ino
    queue.push([hypothesis("h-1")])
    queue.remove(["h-9"])
    assert queue.path.stat().st_ino == before
