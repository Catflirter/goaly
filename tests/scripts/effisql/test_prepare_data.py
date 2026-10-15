"""Tests for scripts/effisql/prepare_data.py, which prepares the dataset and its runtime copy.

The unit tests stand in for the download, since only the real dataset has the pinned sha256. The
integration test runs the script on a local copy of the real dataset, from GOALY_EFFISQL_DATA.
"""

import os
from collections import Counter
from pathlib import Path

import pytest

from goaly.bench.effisql.data import (
    DATASET_FILE,
    DATASET_URL,
    GOLD_FIELDS,
    RUNTIME_FILE,
    default_cache_dir,
    load_instances,
)
from scripts.effisql import prepare_data
from tests.bench.effisql.synthetic import ids, task, write_tasks


def test_writes_the_runtime_copy_next_to_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"

    def fetch(cache_dir: Path, *, url: str) -> Path:
        assert (cache_dir, url) == (cache, "file:///copies/effi-sql-pg.jsonl")
        cache_dir.mkdir()
        return write_tasks(cache_dir / DATASET_FILE, task(id=1), task(id=2))

    monkeypatch.setattr(prepare_data, "fetch_dataset", fetch)

    status = prepare_data.main(
        ["--cache-dir", str(cache), "--url", "file:///copies/effi-sql-pg.jsonl"]
    )

    assert status == 0
    assert ids(load_instances(cache / RUNTIME_FILE)) == ["1", "2"]
    assert capsys.readouterr().out == (
        f"prepare_data: {cache / DATASET_FILE} is verified; wrote {cache / RUNTIME_FILE}\n"
    )


def test_uses_the_default_cache_and_the_pinned_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def fetch(cache_dir: Path, *, url: str) -> Path:
        assert (cache_dir, url) == (default_cache_dir(), DATASET_URL)
        cache_dir.mkdir(parents=True)
        return write_tasks(cache_dir / DATASET_FILE, task(id=1))

    monkeypatch.setattr(prepare_data, "fetch_dataset", fetch)

    assert prepare_data.main([]) == 0


def test_a_checksum_mismatch_fails_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "copies").mkdir()
    copy = write_tasks(tmp_path / "copies" / DATASET_FILE, task(id=1))
    cache = tmp_path / "cache"

    status = prepare_data.main(["--cache-dir", str(cache), "--url", copy.as_uri()])

    assert status == 1
    assert capsys.readouterr().err.startswith(f"prepare_data: {copy.as_uri()} has sha256 ")
    assert list(cache.iterdir()) == []


def test_a_file_system_error_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    cache.write_text("a file where the cache directory should be\n", encoding="utf-8")

    status = prepare_data.main(["--cache-dir", str(cache), "--url", "file:///copy.jsonl"])

    assert status == 1
    err = capsys.readouterr().err
    assert err.startswith("prepare_data: [Errno ")
    assert str(cache) in err
    assert "Traceback" not in err


@pytest.mark.integration
def test_prepares_the_real_dataset(tmp_path: Path) -> None:
    local_copy = os.environ.get("GOALY_EFFISQL_DATA")
    if not local_copy:
        pytest.skip("set GOALY_EFFISQL_DATA to a local copy of effi-sql-pg.jsonl")
    cache = tmp_path / "cache"

    assert prepare_data.main(["--cache-dir", str(cache), "--url", Path(local_copy).as_uri()]) == 0

    dataset, runtime = load_instances(cache / DATASET_FILE), load_instances(cache / RUNTIME_FILE)
    assert len(dataset) == len(runtime) == 300
    tpch = [inst for inst in runtime if inst.db == "tpch"]
    assert Counter(inst.difficulty for inst in tpch) == {"easy": 71, "medium": 111, "hard": 56}
    assert all(inst.base_explain_analyze is not None for inst in runtime)
    assert all(inst.optimized_sql is not None for inst in dataset)
    text = (cache / RUNTIME_FILE).read_text(encoding="utf-8")
    assert not [name for name in GOLD_FIELDS if name in text]
