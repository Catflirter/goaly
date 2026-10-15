"""Draw Effi-SQL's split and write it to src/goaly/bench/splits/effisql.json (spec §7).

    uv run python scripts/effisql/make_splits.py [--cache-dir DIR] [--url URL] [--out PATH]

The split is drawn from the pinned dataset, which is fetched into the cache or checked there
first, with the seed and sizes of goaly.bench.effisql.splits. Running the script again writes
the same file. The file is committed: never edit it by hand. To use a local copy of the dataset,
pass its file URL as --url.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from goaly.bench.effisql.data import (
    DATASET_SHA256,
    DATASET_URL,
    DatasetError,
    default_cache_dir,
    fetch_dataset,
    load_instances,
)
from goaly.bench.effisql.splits import SPLITS_PATH, EffiSqlSplits, dump_splits, split_tasks


def draw(dataset: Path) -> EffiSqlSplits:
    """Split the tasks in ``dataset``, the pinned dataset's verified copy."""
    return split_tasks(load_instances(dataset), source_sha256=DATASET_SHA256)


def write(splits: EffiSqlSplits, path: Path) -> None:
    path.write_text(dump_splits(splits), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the Effi-SQL split and write it.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="the dataset cache (default: ~/.cache/goaly/effisql)",
    )
    parser.add_argument(
        "--url",
        default=DATASET_URL,
        help="where to download the dataset from (default: the pinned Hugging Face file); "
        "a file URL copies a local copy, which must have the pinned sha256 too",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SPLITS_PATH,
        help="where to write the split (default: src/goaly/bench/splits/effisql.json)",
    )
    args = parser.parse_args(argv)
    out: Path = args.out

    try:
        splits = draw(fetch_dataset(args.cache_dir, url=args.url))
    except DatasetError as error:
        sys.stderr.write(f"make_splits: {error}\n")
        return 1
    write(splits, out)
    sys.stdout.write(
        f"make_splits: wrote {out}: {len(splits.train)} train, {len(splits.val)} val,"
        f" {len(splits.heldout)} held-out\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
