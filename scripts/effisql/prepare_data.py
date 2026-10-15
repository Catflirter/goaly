"""Prepare Effi-SQL's data: fetch the pinned dataset and write its gold-free runtime copy.

    uv run python scripts/effisql/prepare_data.py [--cache-dir DIR] [--url URL]

The dataset is downloaded into the cache unless a copy is already there, and is checked against
its pinned sha256 either way (spec §4.1). The runtime copy, which candidate evaluations load
instead of the dataset (spec §8.5), is then written next to it. Neither file is ever committed.
To fill the cache from a local copy of the dataset, pass its file URL as --url.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from goaly.bench.effisql.data import (
    DATASET_URL,
    RUNTIME_FILE,
    DatasetError,
    default_cache_dir,
    fetch_dataset,
    write_runtime_file,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the Effi-SQL dataset and write its gold-free runtime copy."
    )
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
    args = parser.parse_args(argv)
    cache_dir: Path = args.cache_dir

    try:
        dataset = fetch_dataset(cache_dir, url=args.url)
        runtime = cache_dir / RUNTIME_FILE
        write_runtime_file(dataset, runtime)
    except (DatasetError, OSError) as error:  # OSError: the cache cannot be written, say.
        sys.stderr.write(f"prepare_data: {error}\n")
        return 1
    sys.stdout.write(f"prepare_data: {dataset} is verified; wrote {runtime}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
