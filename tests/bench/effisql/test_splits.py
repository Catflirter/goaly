"""Tests for goaly.bench.effisql.splits: the stratified split and its exclusions.

The tasks here are synthetic; the integration test compares the committed split with one drawn
again from a local copy of the real dataset, named by GOALY_EFFISQL_DATA.
"""

import json
import random
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError

from goaly.bench.effisql.data import DATASET_SHA256, Difficulty, EffiSqlInstance
from goaly.bench.effisql.splits import (
    EXCLUSIONS_PATH,
    SPLIT_SEED,
    SPLITS_PATH,
    EffiSqlSplits,
    Exclusion,
    dump_splits,
    load_exclusions,
    load_splits,
    split_tasks,
)
from goaly.contracts.gen.task_pb2 import (
    SPLIT_EXTERNAL,
    SPLIT_HELDOUT,
    SPLIT_TRAIN,
    SPLIT_UNSPECIFIED,
    SPLIT_VAL,
    Split,
)
from tests.bench.effisql.synthetic import task

SOURCE = "a" * 64  # A made-up dataset sha256.


def labelled(labels: Sequence[Difficulty]) -> list[EffiSqlInstance]:
    """Return synthetic TPC-H tasks with these difficulties, numbered from 1."""
    return [
        EffiSqlInstance.model_validate(task(id=number, difficulty=label))
        for number, label in enumerate(labels, start=1)
    ]


def tasks(easy: int, medium: int, hard: int) -> list[EffiSqlInstance]:
    """Return synthetic TPC-H tasks with these difficulty counts, numbered from 1."""
    sizes: list[tuple[Difficulty, int]] = [("easy", easy), ("medium", medium), ("hard", hard)]
    return labelled([label for label, size in sizes for _ in range(size)])


def counts(ids: Sequence[str], instances: Sequence[EffiSqlInstance]) -> Counter[str | None]:
    difficulty = {inst.instance_id: inst.difficulty for inst in instances}
    return Counter(difficulty[instance_id] for instance_id in ids)


# The split.


def test_the_seed_is_20261014() -> None:
    assert SPLIT_SEED == 20261014


def test_splits_the_238_tasks_into_148_30_and_60() -> None:
    instances = tasks(71, 111, 56)

    splits = split_tasks(instances, source_sha256=SOURCE)

    assert (len(splits.train), len(splits.val), len(splits.heldout)) == (148, 30, 60)
    assert (splits.seed, splits.source_sha256) == (SPLIT_SEED, SOURCE)


def test_each_split_is_within_one_task_of_proportional_by_difficulty() -> None:
    instances = tasks(71, 111, 56)

    splits = split_tasks(instances, source_sha256=SOURCE)

    assert counts(splits.val, instances) == {"easy": 9, "medium": 14, "hard": 7}
    assert counts(splits.heldout, instances) == {"easy": 18, "medium": 28, "hard": 14}
    assert counts(splits.train, instances) == {"easy": 44, "medium": 69, "hard": 35}


def test_any_split_is_within_one_task_of_proportional() -> None:
    cases = random.Random(7)  # noqa: S311  # Reproducible test cases, not a secret.
    for _ in range(200):
        strata = [cases.randint(1, 40) for _ in range(cases.randint(1, 3))]
        strata += [0] * (3 - len(strata))
        instances = tasks(*strata)
        val = cases.randint(0, len(instances))
        heldout = cases.randint(0, len(instances) - val)

        splits = split_tasks(instances, source_sha256=SOURCE, val_size=val, heldout_size=heldout)

        total = len(instances)
        sizes = {"val": val, "heldout": heldout, "train": total - val - heldout}
        for name, size in sizes.items():
            got = counts(getattr(splits, name), instances)
            assert sum(got.values()) == size
            for label, count in Counter(inst.difficulty for inst in instances).items():
                share = Fraction(count * size, total)
                assert abs(got[label] - share) < 1, (strata, val, heldout, name, label)


def test_the_splits_are_disjoint_and_cover_every_tpch_task() -> None:
    instances = tasks(5, 7, 3)

    splits = split_tasks(instances, source_sha256=SOURCE, val_size=3, heldout_size=4)

    every = [*splits.train, *splits.val, *splits.heldout]
    assert sorted(every, key=int) == [inst.instance_id for inst in instances]


def test_each_split_lists_its_ids_in_numeric_order() -> None:
    splits = split_tasks(tasks(20, 30, 10), source_sha256=SOURCE, val_size=10, heldout_size=20)

    for ids in (splits.train, splits.val, splits.heldout):
        assert list(ids) == sorted(ids, key=int)


def test_the_same_seed_always_gives_the_same_split() -> None:
    instances = tasks(71, 111, 56)

    first = split_tasks(instances, source_sha256=SOURCE)

    assert split_tasks(instances, source_sha256=SOURCE) == first
    assert split_tasks(list(reversed(instances)), source_sha256=SOURCE) == first


def test_the_split_of_a_small_set_is_pinned() -> None:
    # Pins the algorithm: a change to it would also change the committed split. The difficulties
    # are interleaved, so each one mixes one- and two-digit IDs and their order before the
    # shuffle matters: easy is 1, 4, 6, 8 and 11, medium 2, 5, 9 and 12, hard 3, 7 and 10. Val and
    # held-out get one of each, and train 3 easy, 2 medium and 1 hard.
    e: Difficulty = "easy"
    m: Difficulty = "medium"
    h: Difficulty = "hard"
    instances = labelled([e, m, h, e, m, e, h, e, m, h, e, m])

    splits = split_tasks(instances, source_sha256=SOURCE, val_size=3, heldout_size=3)

    assert splits == EffiSqlSplits(
        seed=SPLIT_SEED,
        source_sha256=SOURCE,
        train=("4", "5", "6", "7", "9", "11"),
        val=("2", "3", "8"),
        heldout=("1", "10", "12"),
    )


def test_another_seed_gives_another_split() -> None:
    instances = tasks(71, 111, 56)

    assert split_tasks(instances, source_sha256=SOURCE, seed=1) != split_tasks(
        instances, source_sha256=SOURCE
    )


def test_only_tpch_tasks_are_split() -> None:
    other = [
        EffiSqlInstance.model_validate(task(id=f"solar-{n}", db="solar_panel")) for n in range(9)
    ]
    instances = tasks(5, 7, 3)

    splits = split_tasks([*other, *instances], source_sha256=SOURCE, val_size=3, heldout_size=4)

    assert splits == split_tasks(instances, source_sha256=SOURCE, val_size=3, heldout_size=4)


def test_a_tpch_task_without_a_difficulty_cannot_be_split() -> None:
    instances = [*tasks(5, 7, 3), EffiSqlInstance.model_validate(task(id=99, difficulty=None))]

    with pytest.raises(ValueError, match=r"^task 99 has no difficulty to stratify by$"):
        split_tasks(instances, source_sha256=SOURCE, val_size=3, heldout_size=4)


@pytest.mark.parametrize(("val", "heldout"), [(10, 6), (-1, 4), (3, -1)])
def test_rejects_split_sizes_that_do_not_fit(val: int, heldout: int) -> None:
    with pytest.raises(ValueError, match=r"^cannot draw"):
        split_tasks(tasks(5, 7, 3), source_sha256=SOURCE, val_size=val, heldout_size=heldout)


# The committed split.


def test_the_committed_split_is_drawn_with_the_seed_from_the_pinned_dataset() -> None:
    splits = load_splits()

    assert (splits.seed, splits.source_sha256) == (SPLIT_SEED, DATASET_SHA256)
    assert (len(splits.train), len(splits.val), len(splits.heldout)) == (148, 30, 60)
    every = [*splits.train, *splits.val, *splits.heldout]
    assert all(instance_id.isdigit() for instance_id in every)
    assert len(set(every)) == 238


def test_the_committed_file_is_written_as_dump_splits_writes_it() -> None:
    assert SPLITS_PATH.read_text(encoding="utf-8") == dump_splits(load_splits())


def test_a_dumped_split_loads_back(tmp_path: Path) -> None:
    splits = split_tasks(tasks(5, 7, 3), source_sha256=SOURCE, val_size=3, heldout_size=4)
    path = tmp_path / "effisql.json"
    path.write_text(dump_splits(splits), encoding="utf-8")

    assert load_splits(path) == splits


def test_dump_splits_writes_indented_json_with_one_id_per_line() -> None:
    splits = EffiSqlSplits(seed=1, source_sha256="ab", train=("1", "2"), val=("3",), heldout=())

    assert dump_splits(splits) == (
        '{\n  "seed": 1,\n  "source_sha256": "ab",\n  "train": [\n    "1",\n    "2"\n  ],\n'
        '  "val": [\n    "3"\n  ],\n  "heldout": []\n}\n'
    )


@pytest.mark.parametrize(
    "change",
    [
        {"val": ["3", "1"]},
        {"heldout": ["4", "4"]},
        {"seed": "20261014"},
        {"train": [1, 2]},
        {"extra": True},
    ],
    ids=["across-splits", "within-a-split", "seed-as-text", "integer-ids", "unknown-key"],
)
def test_load_splits_rejects_an_invalid_file(tmp_path: Path, change: dict[str, object]) -> None:
    fields = {"seed": 1, "source_sha256": "ab", "train": ["1", "2"], "val": ["3"], "heldout": ["4"]}
    path = tmp_path / "effisql.json"
    path.write_text(json.dumps({**fields, **change}), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_splits(path)


@pytest.mark.parametrize(
    ("split", "name"), [(SPLIT_TRAIN, "train"), (SPLIT_VAL, "val"), (SPLIT_HELDOUT, "heldout")]
)
def test_ids_gives_the_ids_of_a_split(split: Split.ValueType, name: str) -> None:
    splits = EffiSqlSplits(seed=1, source_sha256="ab", train=("1",), val=("2",), heldout=("3",))

    assert splits.ids(split) == getattr(splits, name)


@pytest.mark.parametrize("split", [SPLIT_EXTERNAL, SPLIT_UNSPECIFIED])
def test_ids_refuses_a_split_that_is_not_ours(split: Split.ValueType) -> None:
    splits = EffiSqlSplits(seed=1, source_sha256="ab", train=("1",), val=("2",), heldout=("3",))

    with pytest.raises(ValueError, match=f"^{Split.Name(split)} is not one of our splits$"):
        splits.ids(split)


# Exclusions.


def write_exclusions(path: Path, entries: object) -> Path:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_the_committed_exclusion_list_is_empty_until_t10_fills_it() -> None:
    assert EXCLUSIONS_PATH.read_text(encoding="utf-8") == "[]\n"
    assert load_exclusions() == ()


def test_load_exclusions_reads_each_task_with_its_reason(tmp_path: Path) -> None:
    path = write_exclusions(
        tmp_path / "exclusions.json",
        [
            {"instance_id": "57", "reason": "the base query took over 300 s"},
            {"instance_id": "8", "reason": "the gold answer is not 1.2x faster"},
        ],
    )

    assert load_exclusions(path) == (
        Exclusion(instance_id="57", reason="the base query took over 300 s"),
        Exclusion(instance_id="8", reason="the gold answer is not 1.2x faster"),
    )


@pytest.mark.parametrize(
    "entries",
    [
        [{"instance_id": "57", "reason": "slow"}, {"instance_id": "57", "reason": "again"}],
        [{"instance_id": "57", "reason": " "}],
        [{"instance_id": "", "reason": "slow"}],
        [{"instance_id": "57"}],
        [{"instance_id": 57, "reason": "slow"}],
        [{"instance_id": "57", "reason": "slow", "by": "T10"}],
        {"instance_id": "57", "reason": "slow"},
    ],
    ids=[
        "repeated",
        "blank-reason",
        "blank-id",
        "no-reason",
        "integer-id",
        "unknown-key",
        "object",
    ],
)
def test_load_exclusions_rejects_an_invalid_list(tmp_path: Path, entries: object) -> None:
    path = write_exclusions(tmp_path / "exclusions.json", entries)

    with pytest.raises(ValidationError):
        load_exclusions(path)


def test_the_fixture_exclusions_drop_exactly_the_listed_tasks(tmp_path: Path) -> None:
    committed = SPLITS_PATH.read_bytes()
    splits = load_splits()
    listed = {splits.train[0], splits.train[-1], splits.val[3], splits.heldout[10]}
    path = write_exclusions(
        tmp_path / "exclusions.json",
        [{"instance_id": instance_id, "reason": "fixture"} for instance_id in sorted(listed)],
    )

    kept = splits.without(load_exclusions(path))

    for name in ("train", "val", "heldout"):
        ids = getattr(splits, name)
        assert getattr(kept, name) == tuple(i for i in ids if i not in listed)
    assert len(kept.train) + len(kept.val) + len(kept.heldout) == 238 - 4
    assert (kept.seed, kept.source_sha256) == (splits.seed, splits.source_sha256)
    assert load_splits() == splits  # The split ID lists stay as they are.
    assert SPLITS_PATH.read_bytes() == committed


def test_no_exclusions_leave_the_split_as_it_is() -> None:
    splits = load_splits()

    assert splits.without(()) == splits


def test_an_exclusion_of_a_task_in_no_split_is_refused() -> None:
    exclusions = [
        Exclusion(instance_id="1001", reason="x"),
        Exclusion(instance_id="99x", reason="y"),
    ]

    with pytest.raises(ValueError, match=r"^excluded tasks in no split: 99x, 1001$"):
        load_splits().without(exclusions)
