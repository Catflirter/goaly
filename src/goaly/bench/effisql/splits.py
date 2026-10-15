"""Effi-SQL's data splits (spec §7): the TPC-H tasks split three ways, stratified by difficulty.

- split_tasks draws the split: seed 20261014, val 30, held-out 60, and train all the rest.
  scripts/effisql/make_splits.py writes it to src/goaly/bench/splits/effisql.json, which is
  committed, so every run uses the same split.
- load_splits reads that file, and EffiSqlSplits.ids gives a split's task IDs.
- load_exclusions reads the committed exclusion list, effisql_exclusions.json next to it, which
  T10's exclusion run fills (spec §4.1). EffiSqlSplits.without drops the excluded tasks from
  their splits after splitting, so the split itself stays reproducible (spec §7).
"""

import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, TypeAdapter, model_validator

from goaly.bench.effisql.data import EffiSqlInstance, NonBlankStr
from goaly.contracts.gen.task_pb2 import SPLIT_HELDOUT, SPLIT_TRAIN, SPLIT_VAL, Split

__all__ = [
    "EXCLUSIONS_PATH",
    "HELDOUT_SIZE",
    "SPLITS_PATH",
    "SPLIT_DB",
    "SPLIT_SEED",
    "VAL_SIZE",
    "EffiSqlSplits",
    "Exclusion",
    "dump_splits",
    "load_exclusions",
    "load_splits",
    "split_tasks",
]

SPLIT_SEED = 20261014
SPLIT_DB = "tpch"  # Our splits use only the TPC-H tasks (spec §1.4, §4.1).
VAL_SIZE = 30
HELDOUT_SIZE = 60
# The committed split, src/goaly/bench/splits/effisql.json, and the committed exclusion list.
SPLITS_PATH = Path(__file__).resolve().parents[1] / "splits" / "effisql.json"
EXCLUSIONS_PATH = SPLITS_PATH.with_name("effisql_exclusions.json")


class Exclusion(BaseModel):
    """A task left out of our splits, and why (spec §4.1): an entry of the exclusion list."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    instance_id: NonBlankStr
    reason: NonBlankStr


def _each_task_once(exclusions: tuple[Exclusion, ...]) -> tuple[Exclusion, ...]:
    listed = Counter(exclusion.instance_id for exclusion in exclusions)
    repeated = sorted((name for name, count in listed.items() if count > 1), key=_id_order)
    if repeated:
        raise ValueError(f"excluded more than once: {', '.join(repeated)}")
    return exclusions


_EXCLUSIONS = TypeAdapter(Annotated[tuple[Exclusion, ...], AfterValidator(_each_task_once)])


class EffiSqlSplits(BaseModel):
    """The Effi-SQL split (spec §7): the IDs of each split, and what they were drawn from.

    ``seed`` is the seed of the draw, and ``source_sha256`` the sha256 of the dataset it was drawn
    from. Each ID list is in numeric order, and no ID is in two of them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    seed: int
    source_sha256: str
    train: tuple[str, ...]
    val: tuple[str, ...]
    heldout: tuple[str, ...]

    @model_validator(mode="after")
    def _check_each_id_is_listed_once(self) -> Self:
        listed = Counter([*self.train, *self.val, *self.heldout])
        repeated = sorted((name for name, count in listed.items() if count > 1), key=_id_order)
        if repeated:
            raise ValueError(f"listed more than once: {', '.join(repeated)}")
        return self

    def ids(self, split: Split.ValueType) -> tuple[str, ...]:
        """Return the IDs of ``split``, which is SPLIT_TRAIN, SPLIT_VAL or SPLIT_HELDOUT."""
        if split == SPLIT_TRAIN:
            return self.train
        if split == SPLIT_VAL:
            return self.val
        if split == SPLIT_HELDOUT:
            return self.heldout
        raise ValueError(f"{Split.Name(split)} is not one of our splits")

    def without(self, exclusions: Iterable[Exclusion]) -> Self:
        """Return this split with the excluded tasks dropped from their splits (spec §7).

        The other tasks keep their splits and their order; this split itself is unchanged. Raises
        ValueError if an excluded task is in no split, which would mean the list is wrong.
        """
        excluded = {exclusion.instance_id for exclusion in exclusions}
        unknown = excluded.difference(self.train, self.val, self.heldout)
        if unknown:
            raise ValueError(f"excluded tasks in no split: {', '.join(_in_order(unknown))}")
        return self.model_copy(
            update={
                name: tuple(i for i in getattr(self, name) if i not in excluded)
                for name in ("train", "val", "heldout")
            }
        )


def load_splits(path: Path = SPLITS_PATH) -> EffiSqlSplits:
    """Read the split in the file at ``path``, by default the committed one.

    Raises pydantic's ValidationError, a ValueError, if the file does not hold a valid split.
    """
    return EffiSqlSplits.model_validate_json(path.read_bytes())


def load_exclusions(path: Path = EXCLUSIONS_PATH) -> tuple[Exclusion, ...]:
    """Read the exclusion list in the file at ``path``, by default the committed one.

    The file holds a JSON array of ``{"instance_id": ..., "reason": ...}`` objects, with string
    IDs. Raises pydantic's ValidationError, a ValueError, if an entry is invalid or a task is
    listed twice.
    """
    return _EXCLUSIONS.validate_json(path.read_bytes())


def dump_splits(splits: EffiSqlSplits) -> str:
    """Return ``splits`` as the committed file holds it: indented JSON, one ID per line."""
    return json.dumps(splits.model_dump(), indent=2) + "\n"


def split_tasks(
    instances: Iterable[EffiSqlInstance],
    *,
    source_sha256: str,
    seed: int = SPLIT_SEED,
    val_size: int = VAL_SIZE,
    heldout_size: int = HELDOUT_SIZE,
) -> EffiSqlSplits:
    """Split the tasks on SPLIT_DB three ways, stratified by difficulty (spec §7).

    val gets ``val_size`` tasks, held-out ``heldout_size``, and train all the rest. Each split's
    count of each difficulty is within one task of proportional (see _apportion). The tasks of
    each difficulty are put in ID order and shuffled with one random.Random(seed), the
    difficulties taken in name order; the first of each go to val, the next to held-out, and the
    rest to train. So the split depends only on the seed and the tasks, never on their order.
    Tasks on other databases are left out. Raises ValueError if a task has no difficulty or the
    sizes do not fit.
    """
    strata: dict[str, list[str]] = {}
    for inst in instances:
        if inst.db != SPLIT_DB:
            continue
        if inst.difficulty is None:
            raise ValueError(f"task {inst.instance_id} has no difficulty to stratify by")
        strata.setdefault(inst.difficulty, []).append(inst.instance_id)
    total = sum(len(ids) for ids in strata.values())
    if min(val_size, heldout_size) < 0 or val_size + heldout_size > total:
        raise ValueError(
            f"cannot draw {val_size} val and {heldout_size} held-out tasks from {total} tasks"
        )

    labels = sorted(strata)
    sizes = (val_size, heldout_size, total - val_size - heldout_size)
    counts = _apportion([len(strata[label]) for label in labels], sizes)
    rng = random.Random(seed)  # noqa: S311  # A reproducible draw, not a secret.
    val: list[str] = []
    heldout: list[str] = []
    train: list[str] = []
    for label, (val_count, heldout_count, _) in zip(labels, counts, strict=True):
        ids = sorted(strata[label], key=_id_order)
        rng.shuffle(ids)
        val += ids[:val_count]
        heldout += ids[val_count : val_count + heldout_count]
        train += ids[val_count + heldout_count :]
    return EffiSqlSplits(
        seed=seed,
        source_sha256=source_sha256,
        train=_in_order(train),
        val=_in_order(val),
        heldout=_in_order(heldout),
    )


def _apportion(strata: Sequence[int], sizes: Sequence[int]) -> list[tuple[int, ...]]:
    """Share out each stratum's tasks among the splits, each count within one of proportional.

    Stratum s's exact share of split k is strata[s] * sizes[k] / sum(strata). Each count is the
    floor or the ceiling of its share, and every stratum and every split keeps its total: a
    controlled rounding, which always exists for a two-way table. Of those roundings, the one
    whose rounded-up shares have the largest remainders in all wins, as in largest-remainder
    apportionment. The search tries every rounding, which is instant for three difficulties.
    """
    total = sum(strata)
    shares = [[Fraction(count * size, total) for size in sizes] for count in strata]
    floors = [[math.floor(share) for share in row] for row in shares]
    # Each stratum rounds up as many shares as its floors leave it short, and only shares with a
    # remainder can round up.
    choices = [
        list(combinations([k for k, share in enumerate(row) if share % 1], count - sum(low)))
        for row, low, count in zip(shares, floors, strata, strict=True)
    ]
    short = [size - sum(low[k] for low in floors) for k, size in enumerate(sizes)]

    def keeps_the_split_sizes(ups: tuple[tuple[int, ...], ...]) -> bool:
        return all(sum(k in up for up in ups) == short[k] for k in range(len(sizes)))

    def remainders(ups: tuple[tuple[int, ...], ...]) -> Fraction:
        return sum((shares[s][k] % 1 for s, up in enumerate(ups) for k in up), Fraction(0))

    best = max(filter(keeps_the_split_sizes, product(*choices)), key=remainders)
    return [
        tuple(low + (k in up) for k, low in enumerate(row))
        for row, up in zip(floors, best, strict=True)
    ]


def _id_order(instance_id: str) -> tuple[int, str]:
    """Order IDs shortest first, then alphabetically: numeric order for the dataset's IDs."""
    return len(instance_id), instance_id


def _in_order(ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(ids, key=_id_order))
