"""Tests for scripts/effisql/make_splits.py, which draws the split and writes the committed file.

The unit tests stand in for the download, since only the real dataset has the pinned sha256. The
integration test draws the split again from a local copy of the real dataset, named by
GOALY_EFFISQL_DATA, and compares it with the committed file.
"""

import os
from collections import Counter
from pathlib import Path

import pytest

from goaly.bench.effisql.data import DATASET_FILE, DATASET_SHA256, load_instances
from goaly.bench.effisql.splits import SPLITS_PATH, dump_splits, load_splits, split_tasks
from scripts.effisql import make_splits
from tests.bench.effisql.synthetic import task, write_tasks


def synthetic_dataset(path: Path) -> Path:
    """Write 238 TPC-H tasks with the dataset's difficulty counts, and 2 on another database."""
    labels = ["easy"] * 71 + ["medium"] * 111 + ["hard"] * 56
    records = [task(id=n, difficulty=label) for n, label in enumerate(labels, start=1)]
    other = [task(id=n, db="solar_panel") for n in (239, 240)]
    return write_tasks(path, *records, *other)


def test_writes_the_split_of_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cache, out = tmp_path / "cache", tmp_path / "effisql.json"

    def fetch(cache_dir: Path, *, url: str) -> Path:
        assert (cache_dir, url) == (cache, "file:///copies/effi-sql-pg.jsonl")
        cache_dir.mkdir()
        return synthetic_dataset(cache_dir / DATASET_FILE)

    monkeypatch.setattr(make_splits, "fetch_dataset", fetch)

    status = make_splits.main(
        ["--cache-dir", str(cache), "--url", "file:///copies/effi-sql-pg.jsonl", "--out", str(out)]
    )

    assert status == 0
    expected = split_tasks(load_instances(cache / DATASET_FILE), source_sha256=DATASET_SHA256)
    assert out.read_text(encoding="utf-8") == dump_splits(expected)
    assert capsys.readouterr().out == f"make_splits: wrote {out}: 148 train, 30 val, 60 held-out\n"


def test_writes_the_committed_file_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[Path] = []
    monkeypatch.setattr(make_splits, "fetch_dataset", lambda cache_dir, *, url: None)
    monkeypatch.setattr(make_splits, "draw", lambda dataset: load_splits())
    monkeypatch.setattr(make_splits, "write", lambda splits, path: written.append(path))

    assert make_splits.main([]) == 0
    assert written == [SPLITS_PATH]


def test_a_file_system_error_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "missing" / "effisql.json"

    def fetch(cache_dir: Path, *, url: str) -> Path:
        cache_dir.mkdir()
        return synthetic_dataset(cache_dir / DATASET_FILE)

    monkeypatch.setattr(make_splits, "fetch_dataset", fetch)

    status = make_splits.main(["--cache-dir", str(tmp_path / "cache"), "--out", str(out)])

    assert status == 1
    err = capsys.readouterr().err
    assert err.startswith("make_splits: [Errno ")
    assert str(out) in err


def test_a_dataset_error_fails_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "copies").mkdir()
    copy = synthetic_dataset(tmp_path / "copies" / DATASET_FILE)
    out = tmp_path / "effisql.json"

    status = make_splits.main(
        ["--cache-dir", str(tmp_path / "cache"), "--url", copy.as_uri(), "--out", str(out)]
    )

    assert status == 1
    assert capsys.readouterr().err.startswith(f"make_splits: {copy.as_uri()} has sha256 ")
    assert not out.exists()


@pytest.mark.integration
def test_the_committed_split_is_the_real_dataset_split(tmp_path: Path) -> None:
    local_copy = os.environ.get("GOALY_EFFISQL_DATA")
    if not local_copy:
        pytest.skip("set GOALY_EFFISQL_DATA to a local copy of effi-sql-pg.jsonl")
    out = tmp_path / "effisql.json"

    status = make_splits.main(
        [
            "--cache-dir",
            str(tmp_path / "cache"),
            "--url",
            Path(local_copy).as_uri(),
            "--out",
            str(out),
        ]
    )

    assert status == 0
    assert out.read_text(encoding="utf-8") == SPLITS_PATH.read_text(encoding="utf-8")
    difficulty = {
        inst.instance_id: inst.difficulty
        for inst in load_instances(tmp_path / "cache" / DATASET_FILE)
        if inst.db == "tpch"
    }
    splits = load_splits(out)
    assert Counter(difficulty[i] for i in splits.val) == {"easy": 9, "medium": 14, "hard": 7}
    assert Counter(difficulty[i] for i in splits.heldout) == {"easy": 18, "medium": 28, "hard": 14}
    assert Counter(difficulty[i] for i in splits.train) == {"easy": 44, "medium": 69, "hard": 35}
