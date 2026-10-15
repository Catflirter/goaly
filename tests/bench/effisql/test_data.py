"""Tests for goaly.bench.effisql.data: the pinned dataset, its tasks and their gold-free copy.

Every task here is synthetic. No test reads the real dataset or holds a real gold answer.
"""

import copy
import hashlib
import http.client
import io
import json
import math
import pickle
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import NoReturn

import pytest
from pydantic import ValidationError

from goaly.bench.effisql.data import (
    DATASET_FILE,
    DATASET_REVISION,
    DATASET_SHA256,
    DATASET_URL,
    GOLD_FIELDS,
    RUNTIME_FIELDS,
    RUNTIME_FILE,
    ChecksumMismatchError,
    DatasetError,
    EffiSqlInstance,
    TaskFileError,
    default_cache_dir,
    fetch_dataset,
    load_instances,
    write_runtime_file,
)
from goaly.contracts.interfaces import Instance, JsonValue
from tests.bench.effisql.synthetic import (
    BASE_PLAN,
    BASE_SQL,
    GOLD_PLAN,
    GOLD_SQL,
    ids,
    task,
    without,
    write_tasks,
)

# The instance model.


def test_a_task_record_becomes_an_instance() -> None:
    inst = EffiSqlInstance.model_validate(task())

    assert inst.instance_id == "7"
    assert inst.db == "tpch"
    assert inst.base_sql == BASE_SQL
    assert inst.base_time == 1234.5
    assert inst.base_explain_analyze == BASE_PLAN
    assert inst.difficulty == "medium"
    assert inst.optimized_sql == GOLD_SQL
    assert inst.fast_time == 12.25
    assert inst.optimized_explain_analyze == GOLD_PLAN


def test_an_instance_satisfies_the_instance_protocol() -> None:
    inst: Instance = EffiSqlInstance.model_validate(task(id=42))  # make typecheck proves it.

    assert inst.instance_id == "42"


def test_the_gold_fields_are_the_three_the_agent_never_sees() -> None:
    assert GOLD_FIELDS == ("optimized_sql", "fast_time", "optimized_explain_analyze")


def test_a_string_id_becomes_the_instance_id_as_it_is() -> None:
    inst = EffiSqlInstance.model_validate(task(id="tpch-007"))

    assert inst.instance_id == "tpch-007"
    assert inst.record["id"] == "tpch-007"


def test_the_record_keeps_an_integer_id_as_an_integer() -> None:
    inst = EffiSqlInstance.model_validate(task(id=7))

    assert type(inst.record["id"]) is int
    assert inst.record["id"] == 7


@pytest.mark.parametrize(
    "record",
    [
        without(task(), *GOLD_FIELDS, "difficulty"),
        task(**dict.fromkeys(GOLD_FIELDS), difficulty=None),
    ],
    ids=["missing", "null"],
)
def test_a_task_without_gold_answers_or_difficulty_is_valid(record: dict[str, JsonValue]) -> None:
    inst = EffiSqlInstance.model_validate(record)

    assert inst.optimized_sql is None
    assert inst.fast_time is None
    assert inst.optimized_explain_analyze is None
    assert inst.difficulty is None


@pytest.mark.parametrize(
    "record", [without(task(), "base_time"), task(base_time=None)], ids=["missing", "null"]
)
def test_a_task_without_a_base_time_is_valid(record: dict[str, JsonValue]) -> None:
    assert EffiSqlInstance.model_validate(record).base_time is None


@pytest.mark.parametrize(
    "record",
    [without(task(), "base_explain_analyze"), task(base_explain_analyze=None)],
    ids=["missing", "null"],
)
def test_a_task_without_a_plan_is_valid(record: dict[str, JsonValue]) -> None:
    assert EffiSqlInstance.model_validate(record).base_explain_analyze is None


# An empty plan counts as no plan, so agent_input falls back to plans.json (controller's ruling).
EMPTY_PLANS: list[JsonValue] = [None, {}, [], "", " \n\t"]
EMPTY_PLAN_IDS = ["null", "empty-object", "empty-array", "empty-string", "blank-string"]


@pytest.mark.parametrize("plan", EMPTY_PLANS, ids=EMPTY_PLAN_IDS)
def test_an_empty_plan_counts_as_no_plan(plan: JsonValue) -> None:
    inst = EffiSqlInstance.model_validate(
        task(base_explain_analyze=plan, optimized_explain_analyze=plan)
    )

    assert inst.base_explain_analyze is None
    assert inst.optimized_explain_analyze is None
    assert inst.record["base_explain_analyze"] == plan  # The record keeps it as loaded.


def test_an_empty_gold_answer_is_kept_as_it_is() -> None:
    # Gold answers are never checked here: an organizers' file may blank them.
    assert EffiSqlInstance.model_validate(task(optimized_sql="")).optimized_sql == ""


def test_the_times_accept_integers() -> None:
    inst = EffiSqlInstance.model_validate(task(base_time=1200, fast_time=0))

    assert (inst.base_time, inst.fast_time) == (1200.0, 0.0)
    assert isinstance(inst.base_time, float)


def test_unknown_fields_are_ignored_but_kept_in_the_record() -> None:
    inst = EffiSqlInstance.model_validate(task(dialect="postgresql", slow_sql_type="nested"))

    assert not hasattr(inst, "dialect")
    assert inst.record["dialect"] == "postgresql"
    assert inst.record["slow_sql_type"] == "nested"


def test_the_record_keeps_the_non_gold_fields_as_loaded_in_file_order() -> None:
    record = task(note={"tags": ["a", None], "weight": 0.5}, id="x-1")
    reordered = {name: record[name] for name in reversed(record)}

    inst = EffiSqlInstance.model_validate(reordered)

    expected = [(name, value) for name, value in reordered.items() if name not in GOLD_FIELDS]
    assert list(inst.record.items()) == expected


def test_the_record_holds_no_gold_field() -> None:
    inst = EffiSqlInstance.model_validate(task())

    assert list(inst.record) == [name for name in task() if name not in GOLD_FIELDS]


def test_the_record_is_read_only() -> None:
    inst = EffiSqlInstance.model_validate(task())

    with pytest.raises(TypeError):
        inst.record["db"] = "other"  # type: ignore[index]  # The refused write is the point.


def test_the_record_shares_no_object_with_the_input() -> None:
    record = task(note={"tags": ["a"]})
    inst = EffiSqlInstance.model_validate(record)

    record["db"] = "changed"
    note = record["note"]
    assert isinstance(note, dict)
    note["tags"] = []

    assert inst.record["db"] == "tpch"
    assert inst.record["note"] == {"tags": ["a"]}


def test_an_instance_is_frozen() -> None:
    inst = EffiSqlInstance.model_validate(task())

    with pytest.raises(ValidationError, match="frozen"):
        inst.db = "other"  # type: ignore[misc]  # Tests the runtime check.


def test_the_repr_shows_no_gold_answer() -> None:
    text = repr(EffiSqlInstance.model_validate(task()))

    assert "instance_id='7'" in text
    assert GOLD_SQL not in text
    assert "synthetic gold plan" not in text


def test_an_instance_survives_pickling_and_deep_copying() -> None:
    inst = EffiSqlInstance.model_validate(task(note=[1, 2]))

    for twin in (pickle.loads(pickle.dumps(inst)), copy.deepcopy(inst)):  # noqa: S301
        assert twin == inst
        assert twin.record == inst.record


def test_validating_an_instance_again_keeps_its_record() -> None:
    inst = EffiSqlInstance.model_validate(task(note=[1, 2]))

    again = EffiSqlInstance.model_validate(inst)

    assert again == inst
    assert again.record == inst.record


@pytest.mark.parametrize("name", ["id", "db", "base_sql"])
def test_rejects_a_task_without_a_required_field(name: str) -> None:
    with pytest.raises(ValidationError) as caught:
        EffiSqlInstance.model_validate(without(task(), name))

    errors = caught.value.errors()
    assert [(error["loc"], error["type"]) for error in errors] == [((name,), "missing")]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("id", True),
        ("id", 7.0),
        ("id", ""),
        ("id", " \t"),
        ("id", None),
        ("db", ""),
        ("db", 5),
        ("db", None),
        ("base_sql", " \n"),
        ("base_sql", None),
        ("base_time", -0.5),
        ("base_time", math.nan),
        ("base_time", math.inf),
        ("base_time", "1234.5"),
        ("base_time", True),
        ("base_explain_analyze", [BASE_PLAN]),
        ("base_explain_analyze", '{"Node Type": "Aggregate"}'),
        ("difficulty", "Easy"),
        ("difficulty", ""),
        ("optimized_sql", 5),
        ("fast_time", -1),
        ("fast_time", math.nan),
        ("optimized_explain_analyze", "plan"),
    ],
)
def test_rejects_an_invalid_field(name: str, value: JsonValue) -> None:
    with pytest.raises(ValidationError) as caught:
        EffiSqlInstance.model_validate(task(**{name: value}))

    assert [error["loc"][0] for error in caught.value.errors()] == [name]


def test_rejects_an_unknown_field_that_is_not_json() -> None:
    with pytest.raises(ValidationError) as caught:
        EffiSqlInstance.model_validate({**task(), "note": {1, 2}})

    assert [error["loc"][0] for error in caught.value.errors()] == ["note"]


# Loading a task file.


def test_loads_every_task_in_file_order(tmp_path: Path) -> None:
    path = write_tasks(tmp_path / "tasks.jsonl", task(id=3), task(id=1), task(id=2))

    instances = load_instances(path)

    assert ids(instances) == ["3", "1", "2"]
    assert instances[0] == EffiSqlInstance.model_validate(task(id=3))


def test_loads_a_file_without_gold_answers_or_difficulty(tmp_path: Path) -> None:
    bare = without(task(), *GOLD_FIELDS, "difficulty")
    path = write_tasks(tmp_path / "tasks.jsonl", bare, {**bare, "id": 8})

    instances = load_instances(path)

    assert ids(instances) == ["7", "8"]
    assert {(inst.optimized_sql, inst.fast_time, inst.difficulty) for inst in instances} == {
        (None, None, None)
    }


def test_loads_a_file_without_base_times(tmp_path: Path) -> None:
    path = write_tasks(tmp_path / "tasks.jsonl", without(task(), "base_time"), task(id=8))

    assert [inst.base_time for inst in load_instances(path)] == [None, 1234.5]


def test_loads_a_file_with_string_ids(tmp_path: Path) -> None:
    path = write_tasks(tmp_path / "tasks.jsonl", task(id="tpch-1"), task(id="2"), task(id=3))

    instances = load_instances(path)

    assert ids(instances) == ["tpch-1", "2", "3"]
    assert [inst.record["id"] for inst in instances] == ["tpch-1", "2", 3]


def test_loads_a_file_with_no_plans(tmp_path: Path) -> None:
    no_plan = without(task(id=1), "base_explain_analyze")
    empty = [task(id=n, base_explain_analyze=plan) for n, plan in enumerate(EMPTY_PLANS, start=2)]
    path = write_tasks(tmp_path / "tasks.jsonl", no_plan, *empty)

    instances = load_instances(path)

    assert ids(instances) == ["1", "2", "3", "4", "5", "6"]
    assert {inst.base_explain_analyze is None for inst in instances} == {True}


def test_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    first, second = json.dumps(task(id=1)), json.dumps(task(id=2))
    path.write_text(f"\n{first}\n  \n\t\n{second}\n\n", encoding="utf-8")

    assert ids(load_instances(path)) == ["1", "2"]


def test_reads_windows_line_endings_and_a_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    first, second = json.dumps(task(id=1)).encode(), json.dumps(task(id=2)).encode()
    path.write_bytes(b"\xef\xbb\xbf" + first + b"\r\n" + second + b"\r\n")

    assert ids(load_instances(path)) == ["1", "2"]


def test_only_a_line_feed_ends_a_task(tmp_path: Path) -> None:
    base_sql = "SELECT 1 -- a line separator \u2028 and a paragraph separator \u2029"
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(task(base_sql=base_sql), ensure_ascii=False) + "\n", "utf-8")

    assert [inst.base_sql for inst in load_instances(path)] == [base_sql]


def test_an_empty_file_has_no_tasks(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_bytes(b"")

    assert load_instances(path) == ()


def test_reports_the_file_and_line_of_an_invalid_task(tmp_path: Path) -> None:
    path = write_tasks(tmp_path / "tasks.jsonl", task(id=1), task(id=2, base_time=-1, db=""))

    with pytest.raises(TaskFileError) as caught:
        load_instances(path)

    assert str(caught.value) == (
        f"{path}:2: db: Value error, must not be blank;"
        " base_time: Input should be greater than or equal to 0"
    )
    assert isinstance(caught.value, DatasetError)


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        (b'{"id": 1,', "not valid JSON: Expecting property name enclosed in double quotes"),
        (b'{"id": 1, "base_time": NaN}', "not valid JSON: NaN is not a JSON value"),
        (b'{"id": 1, "base_time": -Infinity}', "not valid JSON: -Infinity is not a JSON value"),
        (b'[{"id": 1}]', "not a JSON object"),
        (b'"tpch"', "not a JSON object"),
        (b'{"id": 1, "db": "caf\xe9"}', "not valid UTF-8 at byte 20"),
    ],
)
def test_reports_a_line_that_is_not_a_task(tmp_path: Path, line: bytes, reason: str) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_bytes(json.dumps(task(id=1)).encode() + b"\n" + line + b"\n")

    with pytest.raises(TaskFileError) as caught:
        load_instances(path)

    assert str(caught.value).startswith(f"{path}:2: {reason}")


def test_rejects_an_id_that_repeats(tmp_path: Path) -> None:
    path = write_tasks(tmp_path / "tasks.jsonl", task(id=1), task(id=2), task(id="1"))

    with pytest.raises(TaskFileError) as caught:
        load_instances(path)

    assert str(caught.value) == f"{path}:3: the id 1 repeats line 1"


def test_an_error_never_quotes_the_task(tmp_path: Path) -> None:
    # A gold answer must not reach a log through an error message or its chained cause.
    gold = "SELECT 'gold that must stay hidden'"
    path = write_tasks(tmp_path / "tasks.jsonl", task(optimized_sql=gold, fast_time=gold))

    with pytest.raises(TaskFileError) as caught:
        load_instances(path)

    assert gold not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_instances(tmp_path / "missing.jsonl")


# Fetching the pinned dataset.


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A synthetic dataset to download, outside the cache."""
    (tmp_path / "source").mkdir()
    return write_tasks(tmp_path / "source" / DATASET_FILE, task(id=1), task(id=2))


def test_pins_the_dataset_revision_and_checksum() -> None:
    assert DATASET_REVISION == "86cbcc9efbd31db32a46897d47cbf865b1cba3a4"
    assert DATASET_SHA256 == "2af3bdc9fe6c1197f23e31089660e7ce10fca6956b32e5970a3113f0c87e9d97"
    assert DATASET_URL == (
        "https://huggingface.co/datasets/birdsql/Effi-SQL/resolve/"
        "86cbcc9efbd31db32a46897d47cbf865b1cba3a4/effi-sql-pg.jsonl"
    )


def test_the_default_cache_is_in_the_home_directory() -> None:
    assert default_cache_dir() == Path.home() / ".cache" / "goaly" / "effisql"


def test_fetch_downloads_the_dataset_into_the_cache(tmp_path: Path, source: Path) -> None:
    cache = tmp_path / "cache" / "effisql"

    path = fetch_dataset(cache, url=source.as_uri(), sha256=sha256_of(source))

    assert path == cache / DATASET_FILE
    assert path.read_bytes() == source.read_bytes()
    assert [entry.name for entry in cache.iterdir()] == [DATASET_FILE]


def test_fetch_uses_the_cached_copy_without_downloading(tmp_path: Path, source: Path) -> None:
    cache, digest = tmp_path / "cache", sha256_of(source)
    fetch_dataset(cache, url=source.as_uri(), sha256=digest)
    source.unlink()  # Any second download would fail.

    assert fetch_dataset(cache, url=source.as_uri(), sha256=digest) == cache / DATASET_FILE


def test_fetch_discards_a_download_with_the_wrong_checksum(tmp_path: Path, source: Path) -> None:
    cache, wrong = tmp_path / "cache", "0" * 64

    with pytest.raises(ChecksumMismatchError) as caught:
        fetch_dataset(cache, url=source.as_uri(), sha256=wrong)

    assert str(caught.value) == (
        f"{source.as_uri()} has sha256 {sha256_of(source)}, not the pinned {wrong};"
        " the download was discarded"
    )
    assert isinstance(caught.value, DatasetError)
    assert list(cache.iterdir()) == []


def test_fetch_refuses_a_cached_copy_with_the_wrong_checksum(tmp_path: Path, source: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = cache / DATASET_FILE
    cached.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ChecksumMismatchError) as caught:
        fetch_dataset(cache, url=source.as_uri(), sha256=sha256_of(source))

    assert str(caught.value) == (
        f"{cached} has sha256 {sha256_of(cached)}, not the pinned {sha256_of(source)};"
        " delete it to download the pinned file again"
    )
    assert cached.read_text(encoding="utf-8") == "changed\n"  # Kept for a person to look at.


def test_fetch_checks_the_pinned_checksum_by_default(tmp_path: Path, source: Path) -> None:
    with pytest.raises(ChecksumMismatchError, match=f"not the pinned {DATASET_SHA256}"):
        fetch_dataset(tmp_path / "cache", url=source.as_uri())


def test_fetch_reports_a_download_that_fails(tmp_path: Path) -> None:
    missing = (tmp_path / "missing.jsonl").as_uri()

    with pytest.raises(DatasetError, match=f"^cannot download {missing}: .*No such file"):
        fetch_dataset(tmp_path / "cache", url=missing, sha256="0" * 64)

    assert list((tmp_path / "cache").iterdir()) == []


@pytest.mark.parametrize(
    "url", ["http://huggingface.co/datasets/x.jsonl", "ftp://example.com/x.jsonl", "/data/x.jsonl"]
)
def test_fetch_refuses_a_url_that_is_not_https_or_file(tmp_path: Path, url: str) -> None:
    with pytest.raises(DatasetError) as caught:
        fetch_dataset(tmp_path / "cache", url=url)

    assert str(caught.value) == f"cannot download {url}: only https and file URLs are allowed"
    assert not (tmp_path / "cache").exists()


def test_fetch_reports_an_http_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(url: str, *, timeout: float) -> NoReturn:
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", not_found)

    with pytest.raises(DatasetError) as caught:
        fetch_dataset(tmp_path / "cache")

    assert str(caught.value) == f"cannot download {DATASET_URL}: HTTP Error 404: Not Found"
    assert list((tmp_path / "cache").iterdir()) == []


class CutShort(io.BytesIO):
    """A response whose connection closes before the whole body has arrived."""

    def read(self, size: int | None = -1) -> bytes:
        chunk = super().read(size)
        if not chunk:
            raise http.client.IncompleteRead(b"", 1000)
        return chunk


def test_fetch_deletes_a_partial_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def cut_short(url: str, *, timeout: float) -> CutShort:
        assert timeout > 0
        return CutShort(b"part of the file")

    monkeypatch.setattr(urllib.request, "urlopen", cut_short)

    with pytest.raises(DatasetError, match=f"^cannot download {DATASET_URL}: IncompleteRead"):
        fetch_dataset(tmp_path / "cache")

    assert list((tmp_path / "cache").iterdir()) == []


# The gold-free runtime file.


def test_the_runtime_fields_are_what_the_agent_sees_plus_the_grading_fields() -> None:
    assert RUNTIME_FIELDS == (
        "id",
        "db",
        "base_sql",
        "base_time",
        "base_explain_analyze",
        "difficulty",
    )
    assert not set(RUNTIME_FIELDS) & set(GOLD_FIELDS)


def test_the_runtime_file_keeps_only_the_runtime_fields(tmp_path: Path) -> None:
    dataset = write_tasks(
        tmp_path / DATASET_FILE,
        task(id=3, dialect="postgresql"),
        without(task(id="1"), "base_time", "difficulty"),
        task(id=2),
    )
    target = tmp_path / RUNTIME_FILE

    write_runtime_file(dataset, target)

    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [list(record) for record in records] == [
        list(RUNTIME_FIELDS),
        ["id", "db", "base_sql", "base_explain_analyze"],
        list(RUNTIME_FIELDS),
    ]
    assert records[0] == {name: task(id=3)[name] for name in RUNTIME_FIELDS}
    assert [record["id"] for record in records] == [3, "1", 2]


def test_the_runtime_file_holds_no_gold_answer(tmp_path: Path) -> None:
    dataset = write_tasks(tmp_path / DATASET_FILE, task(id=1), task(id=2, fast_time=98.765))
    target = tmp_path / RUNTIME_FILE

    write_runtime_file(dataset, target)

    text = target.read_text(encoding="utf-8")
    assert GOLD_SQL not in text
    assert "synthetic gold plan" not in text
    assert "98.765" not in text
    assert not [name for name in GOLD_FIELDS if name in text]


def test_the_runtime_file_loads_as_the_dataset_without_gold(tmp_path: Path) -> None:
    dataset = write_tasks(
        tmp_path / DATASET_FILE,
        task(id=1, extra=True),
        task(id="2"),
        task(id=3, base_explain_analyze={}),
    )
    target = tmp_path / RUNTIME_FILE

    write_runtime_file(dataset, target)

    gold_free = [inst.model_dump(exclude=set(GOLD_FIELDS)) for inst in load_instances(dataset)]
    assert [
        inst.model_dump(exclude=set(GOLD_FIELDS)) for inst in load_instances(target)
    ] == gold_free
    assert {
        (inst.optimized_sql, inst.fast_time, inst.optimized_explain_analyze)
        for inst in load_instances(target)
    } == {(None, None, None)}


def test_the_runtime_file_has_one_json_object_per_line(tmp_path: Path) -> None:
    record: dict[str, JsonValue] = {
        "id": 5,
        "db": "tpch",
        "base_sql": "SELECT 'é'",
        "base_time": 1.5,
    }
    dataset = write_tasks(tmp_path / DATASET_FILE, record)
    target = tmp_path / RUNTIME_FILE

    write_runtime_file(dataset, target)

    expected = """{"id": 5, "db": "tpch", "base_sql": "SELECT 'é'", "base_time": 1.5}\n"""
    assert target.read_text(encoding="utf-8") == expected


def test_the_runtime_file_replaces_an_older_copy(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    dataset = write_tasks(data / DATASET_FILE, task(id=1))
    target = data / RUNTIME_FILE
    target.write_text("older\n", encoding="utf-8")

    write_runtime_file(dataset, target)

    assert ids(load_instances(target)) == ["1"]
    assert sorted(path.name for path in data.iterdir()) == sorted([DATASET_FILE, RUNTIME_FILE])


def test_an_invalid_dataset_leaves_the_runtime_file_as_it_was(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    dataset = write_tasks(data / DATASET_FILE, task(id=1), task(id=1))
    target = data / RUNTIME_FILE
    target.write_text("older\n", encoding="utf-8")

    with pytest.raises(TaskFileError, match="repeats line 1"):
        write_runtime_file(dataset, target)

    assert target.read_text(encoding="utf-8") == "older\n"
    assert sorted(path.name for path in data.iterdir()) == sorted([DATASET_FILE, RUNTIME_FILE])
