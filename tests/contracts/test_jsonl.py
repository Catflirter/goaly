"""Tests for goaly.contracts.jsonl."""

import json
import logging
import math
import pickle
import stat
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any, assert_type

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message_factory
from google.protobuf.json_format import ParseError, SerializeToJsonError
from google.protobuf.message import Message
from google.protobuf.struct_pb2 import ListValue, Struct, Value
from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.environment_pb2 import ShellResult, SqlResult
from goaly.contracts.gen.llm_call_pb2 import LlmCallAttempt, LlmCallRecord, LlmCallStatus, Usage
from goaly.contracts.gen.task_pb2 import (
    EffiSqlPrediction,
    EffiSqlTaskInput,
    Grade,
    GradeStatus,
    SweGrade,
    TaskInput,
)
from goaly.contracts.jsonl import (
    JsonlError,
    append_jsonl,
    from_json_line,
    read_jsonl,
    to_json_line,
    write_jsonl,
)

TEXT = "naïve 日本語\nsecond line\u2028after a line separator"


def reject_constant(name: str) -> None:
    raise ValueError(f"not valid JSON: {name}")


def parse(line: str) -> dict[str, Any]:
    """Parse ``line`` as strict JSON: NaN and Infinity are not JSON."""
    fields = json.loads(line, parse_constant=reject_constant)
    assert isinstance(fields, dict)
    return fields


# to_json_line.


def test_writes_a_compact_line_with_the_proto_field_names() -> None:
    line = to_json_line(ShellResult(exit_code=1, stdout="hi", timed_out=True, elapsed_s=0.5))

    assert " " not in line
    assert parse(line) == {
        "exit_code": 1,
        "stdout": "hi",
        "stderr": "",
        "truncated": False,
        "timed_out": True,
        "elapsed_s": 0.5,
    }


def test_writes_default_values_but_not_unset_messages() -> None:
    line = to_json_line(Grade(instance_id="t1", status=GradeStatus.GRADE_STATUS_GRADED))

    # A failed task shows "correct": false; the benchmark variant, never set, is left out.
    assert parse(line) == {
        "instance_id": "t1",
        "score": 0.0,
        "correct": False,
        "status": "GRADE_STATUS_GRADED",
        "feedback": "",
    }


def test_writes_keys_in_the_order_the_proto_declares_them_whatever_is_set() -> None:
    sparse = Grade(status=GradeStatus.GRADE_STATUS_GRADED, swe=SweGrade(cached=True))
    full = Grade(
        instance_id="t1",
        score=0.5,
        correct=True,
        status=GradeStatus.GRADE_STATUS_GRADED,
        feedback="ok",
        swe=SweGrade(
            patch_applied=True,
            tests_passed=True,
            speedup=2.0,
            gold_speedup=4.0,
            sr=0.5,
            t_pre_s=4.0,
            t_post_s=2.0,
            eval_wall_s=60.0,
            eval_cpu_s=120.0,
            cached=True,
        ),
    )
    swe_keys = [
        *("patch_applied", "tests_passed", "empty_patch", "speedup", "gold_speedup", "sr"),
        *("t_pre_s", "t_post_s", "eval_wall_s", "eval_cpu_s", "cached"),
    ]

    lines = [parse(to_json_line(message)) for message in (sparse, full)]

    for fields in lines:
        assert list(fields) == ["instance_id", "score", "correct", "status", "feedback", "swe"]
        assert list(fields["swe"]) == swe_keys


def test_writes_keys_in_declaration_order_not_in_number_order() -> None:
    record = LlmCallRecord(status=LlmCallStatus.LLM_CALL_STATUS_OK, seed=2, call_id="c")

    # seed is field 13, declared after instance_id.
    assert list(parse(to_json_line(record)))[:5] == [
        *("call_id", "run_id", "instance_id", "seed", "purpose")
    ]


def message_with_a_list_and_a_map() -> type[Message]:
    """Outer, with a list and a map of Inner, which declares field 2 before field 1.

    Built at run time, so the tests do not depend on which contracts have lists or maps.
    """
    file = descriptor_pb2.FileDescriptorProto(
        name="goaly_tests/ordering.proto", package="goaly_tests", syntax="proto3"
    )
    optional, repeated = (
        descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    string, message = (
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
    )
    inner = file.message_type.add(name="Inner")
    inner.field.add(name="late", number=2, type=string, label=optional)
    inner.field.add(name="early", number=1, type=string, label=optional)
    outer = file.message_type.add(name="Outer")
    entry = outer.nested_type.add(name="ByKeyEntry")
    entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=string, label=optional)
    entry.field.add(
        name="value", number=2, type=message, type_name=".goaly_tests.Inner", label=optional
    )
    outer.field.add(
        name="by_key",
        number=1,
        type=message,
        type_name=".goaly_tests.Outer.ByKeyEntry",
        label=repeated,
    )
    outer.field.add(
        name="items", number=2, type=message, type_name=".goaly_tests.Inner", label=repeated
    )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("goaly_tests.Outer"))


def test_writes_the_elements_of_lists_and_the_values_of_maps_in_declaration_order() -> None:
    outer = message_with_a_list_and_a_map()()
    getattr(outer, "items").add().early = "e"  # noqa: B009  # The type is built at run time.
    getattr(outer, "by_key")["k"].early = "e"  # noqa: B009

    fields = parse(to_json_line(outer))

    assert list(fields["items"][0]) == ["late", "early"]
    assert list(fields["by_key"]["k"]) == ["late", "early"]


def test_writes_each_attempt_of_a_call_and_only_the_retry_after_it_was_sent() -> None:
    throttled = LlmCallAttempt(
        wait_s=0.5,
        latency_s=0.2,
        status=LlmCallStatus.LLM_CALL_STATUS_RATE_LIMITED,
        http_status=429,
        retry_after_s=2.0,
    )
    answered = LlmCallAttempt(
        wait_s=2.0, latency_s=3.1, status=LlmCallStatus.LLM_CALL_STATUS_OK, http_status=200
    )
    record = LlmCallRecord(attempts=[throttled, answered], wait_s=2.5, latency_s=5.8)

    line = to_json_line(record)

    attempts = parse(line)["attempts"]
    assert [attempt["http_status"] for attempt in attempts] == [429, 200]
    assert [attempt.get("retry_after_s") for attempt in attempts] == [2.0, None]
    assert from_json_line(line, LlmCallRecord) == record


def test_writes_timestamps_in_rfc_3339() -> None:
    started_at = Timestamp()
    started_at.FromDatetime(datetime(2026, 10, 15, 1, 2, 3, 456000, tzinfo=UTC))

    line = to_json_line(LlmCallRecord(started_at=started_at))

    assert parse(line)["started_at"] == "2026-10-15T01:02:03.456Z"


def test_keeps_text_as_is_on_a_single_line() -> None:
    message = Grade(feedback=TEXT)

    line = to_json_line(message)

    assert "\n" not in line
    assert "naïve 日本語" in line  # Not \u escapes.
    assert "\u2028" in line
    assert from_json_line(line, Grade) == message


def test_writes_int64_values_as_strings_without_losing_precision() -> None:
    message = Usage(input_tokens=2**53 + 1)

    line = to_json_line(message)

    assert parse(line)["input_tokens"] == "9007199254740993"
    assert from_json_line(line, Usage) == message


def test_writes_a_row_count_as_a_json_number() -> None:
    # Counts are int32, which JSON holds as numbers; int64 would be a string such as "3".
    assert parse(to_json_line(SqlResult(row_count=3)))["row_count"] == 3


@pytest.mark.parametrize(
    ("value", "written"), [(math.nan, "NaN"), (math.inf, "Infinity"), (-math.inf, "-Infinity")]
)
def test_writes_non_finite_doubles_as_strings(value: float, written: str) -> None:
    line = to_json_line(Grade(score=value))

    assert parse(line)["score"] == written
    read = from_json_line(line, Grade).score
    assert math.isnan(read) if math.isnan(value) else read == value


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_refuses_a_non_finite_number_inside_a_value(value: float) -> None:
    # Written as the string "NaN", it would read back as a string, so protobuf refuses it.
    row = ListValue()
    row.append(value)

    with pytest.raises(SerializeToJsonError, match="Fail to serialize"):
        to_json_line(SqlResult(rows=[row]))


def test_never_writes_a_nan_token_which_is_not_json(monkeypatch: pytest.MonkeyPatch) -> None:
    # protobuf writes a non-finite double as a string; the line must not depend on that alone.
    monkeypatch.setattr(json_format, "MessageToDict", lambda *_, **__: {"score": math.nan})

    with pytest.raises(ValueError, match="not JSON compliant"):
        to_json_line(Grade())


@pytest.mark.parametrize(
    "message",
    [Timestamp(seconds=1), Value(string_value="x"), ListValue()],
    ids=["Timestamp", "Value", "ListValue"],
)
def test_refuses_a_message_whose_json_is_not_an_object(message: Message) -> None:
    # Their JSON is a string, any value and an array; from_json_line reads only objects.
    with pytest.raises(SerializeToJsonError, match="is not written as a JSON object"):
        to_json_line(message)


def test_writes_a_struct_as_protobuf_does() -> None:
    # A Struct's keys are its data, not fields: ordering them by its descriptor would drop them.
    struct = Struct()
    struct.update({"zeta": 1.0, "alpha": "a"})

    assert parse(to_json_line(struct)) == {"zeta": 1.0, "alpha": "a"}


# from_json_line.


def test_reads_a_line_into_the_requested_type() -> None:
    message = from_json_line('{"effisql":{"db":"tpch"}}', TaskInput)

    assert_type(message, TaskInput)
    assert message == TaskInput(effisql=EffiSqlTaskInput(db="tpch"))


def test_ignores_unknown_fields_at_any_depth() -> None:
    # TaskInput once had an instance_id: lines written before it was removed still read.
    line = '{"dropped":1,"instance_id":"t1","effisql":{"db":"tpch","also_dropped":[true]}}'

    message = from_json_line(line, TaskInput)

    assert message == TaskInput(effisql=EffiSqlTaskInput(db="tpch"))


@pytest.mark.parametrize(
    "line",
    [
        # protobuf alone reads these three as an empty message: no item is a field name.
        '"instance_id"',
        '["db_id"]',
        "[]",
        "7",
        "null",
        "{not json",
        '{"sql":7}',
    ],
)
def test_rejects_a_line_that_is_not_a_valid_message(line: str) -> None:
    with pytest.raises(ParseError):
        from_json_line(line, EffiSqlPrediction)


# read_jsonl.


def test_reads_every_message_in_order_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_bytes(b'{"sql":"a"}\n\n{"sql":"b"}\r\n  \n{"sql":"c"}')

    messages = read_jsonl(path, EffiSqlPrediction)

    assert_type(messages, Iterator[EffiSqlPrediction])
    assert list(messages) == [EffiSqlPrediction(sql=name) for name in "abc"]


def test_splits_records_on_newlines_only(tmp_path: Path) -> None:
    # str.splitlines() would also split at U+2028, U+2029 and U+0085, all valid inside JSON text.
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"sql":"a\u2028b\u2029c\x85d"}\n', encoding="utf-8")

    assert list(read_jsonl(path, EffiSqlPrediction)) == [
        EffiSqlPrediction(sql="a\u2028b\u2029c\x85d")
    ]


def test_names_the_file_and_line_of_a_malformed_record(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"sql":"a"}\n\n{"sql": oops}\n{"sql":"c"}\n', "utf-8")
    read: list[EffiSqlPrediction] = []

    with pytest.raises(JsonlError) as raised:
        read.extend(read_jsonl(path, EffiSqlPrediction))

    assert read == [EffiSqlPrediction(sql="a")]
    assert str(raised.value).startswith(f"{path}:3: not a valid goaly.v1.EffiSqlPrediction: ")
    assert (raised.value.path, raised.value.line_number) == (path, 3)
    assert isinstance(raised.value.__cause__, ParseError)


def test_names_the_file_and_line_of_text_that_is_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_bytes(b'{"sql":"a"}\n{"sql":"\xff"}\n')

    with pytest.raises(JsonlError, match=r":2: not valid UTF-8 at byte 8$") as raised:
        list(read_jsonl(path, EffiSqlPrediction))

    assert (raised.value.path, raised.value.line_number) == (path, 2)


def test_a_jsonl_error_survives_pickling(tmp_path: Path) -> None:
    # Errors cross process boundaries, as from a ProcessPoolExecutor worker.
    error = JsonlError(tmp_path / "tasks.jsonl", 3, "not valid UTF-8 at byte 16")

    copy = pickle.loads(pickle.dumps(error))  # noqa: S301  # Pickled just above.

    assert type(copy) is JsonlError
    assert (str(copy), copy.path, copy.line_number, copy.reason) == (
        f"{tmp_path / 'tasks.jsonl'}:3: not valid UTF-8 at byte 16",
        tmp_path / "tasks.jsonl",
        3,
        "not valid UTF-8 at byte 16",
    )


# write_jsonl and append_jsonl.


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    """An empty directory: tmp_path also holds the isolated HOME of conftest.py."""
    path = tmp_path / "out"
    path.mkdir()
    return path


def test_writes_one_line_per_message(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"

    write_jsonl(path, [EffiSqlPrediction(sql="a"), EffiSqlPrediction(sql="b")])

    assert path.read_text(encoding="utf-8") == '{"sql":"a"}\n{"sql":"b"}\n'


def test_write_replaces_the_whole_file(directory: Path) -> None:
    path = directory / "tasks.jsonl"
    path.write_text('{"sql":"old"}\n' * 3, encoding="utf-8")

    write_jsonl(path, [])

    assert path.read_bytes() == b""
    assert list(directory.iterdir()) == [path]


def test_a_failed_write_leaves_the_file_as_it_was(directory: Path) -> None:
    path = directory / "tasks.jsonl"
    path.write_text('{"sql":"old"}\n', encoding="utf-8")

    def messages() -> Iterator[EffiSqlPrediction]:
        yield EffiSqlPrediction(sql="new")
        raise RuntimeError("the producer failed")

    with pytest.raises(RuntimeError, match="the producer failed"):
        write_jsonl(path, messages())

    assert path.read_text(encoding="utf-8") == '{"sql":"old"}\n'
    assert list(directory.iterdir()) == [path]  # No temporary file is left behind.


def test_write_renames_a_file_from_the_same_directory(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A rename replaces the file atomically only within one file system.
    renames: list[tuple[Path, Path]] = []
    replace = Path.replace

    def recording_replace(source: Path, target: str | PurePath) -> Path:
        renames.append((source, Path(target)))
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    path = directory / "tasks.jsonl"

    write_jsonl(path, [EffiSqlPrediction(sql="a")])

    ((source, target),) = renames
    assert target == path
    assert source.parent == target.parent
    assert path.read_text(encoding="utf-8") == '{"sql":"a"}\n'


def test_a_written_file_gets_the_permissions_of_any_new_file(tmp_path: Path) -> None:
    path, reference = tmp_path / "tasks.jsonl", tmp_path / "reference"
    reference.touch()

    write_jsonl(path, [EffiSqlPrediction(sql="a")])

    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


def test_appends_one_line_per_call(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"

    append_jsonl(path, EffiSqlPrediction(sql="a"))
    append_jsonl(path, EffiSqlPrediction(sql="b"))

    assert path.read_text(encoding="utf-8") == '{"sql":"a"}\n{"sql":"b"}\n'


# Torn tails: a crash loses at most the line being written (spec §6.2).

A_THEN_TORN = b'{"sql":"a"}\n{"sql":'


def test_append_cuts_off_a_torn_last_line(directory: Path) -> None:
    path = directory / "tasks.jsonl"
    path.write_bytes(A_THEN_TORN)

    append_jsonl(path, EffiSqlPrediction(sql="b"))
    append_jsonl(path, EffiSqlPrediction(sql="c"))

    assert path.read_bytes() == b'{"sql":"a"}\n{"sql":"b"}\n{"sql":"c"}\n'
    assert list(read_jsonl(path, EffiSqlPrediction)) == [
        EffiSqlPrediction(sql=name) for name in "abc"
    ]


def test_append_cuts_off_a_file_that_is_one_torn_line(directory: Path) -> None:
    path = directory / "tasks.jsonl"
    path.write_bytes(b'{"sql":"a')

    append_jsonl(path, EffiSqlPrediction(sql="b"))

    assert path.read_bytes() == b'{"sql":"b"}\n'


def test_append_cuts_off_a_torn_line_longer_than_a_read(directory: Path) -> None:
    path = directory / "tasks.jsonl"
    path.write_bytes(b'{"sql":"a"}\n{"sql":"' + b"x" * 200_000)

    append_jsonl(path, EffiSqlPrediction(sql="b"))

    assert path.read_bytes() == b'{"sql":"a"}\n{"sql":"b"}\n'


def test_append_keeps_a_last_record_that_lost_only_its_line_feed(directory: Path) -> None:
    # The write stopped just before the line feed. read_jsonl returns this record, so the next
    # append must keep it too.
    path = directory / "tasks.jsonl"
    path.write_bytes(b'{"sql":"a"}\n{"sql":"b"}')

    append_jsonl(path, EffiSqlPrediction(sql="c"))

    assert path.read_bytes() == b'{"sql":"a"}\n{"sql":"b"}\n{"sql":"c"}\n'


def test_a_message_that_cannot_be_written_leaves_the_file_untouched(directory: Path) -> None:
    path = directory / "results.jsonl"
    path.write_bytes(b'{"columns":["x"]}\n{"colu')
    row = ListValue()
    row.append(math.nan)

    with pytest.raises(SerializeToJsonError):
        append_jsonl(path, SqlResult(rows=[row]))

    assert path.read_bytes() == b'{"columns":["x"]}\n{"colu'  # Not even the torn tail is cut.


@pytest.mark.parametrize(
    "torn", [b'{"sql":', b'{"sql":"\xe6\x97'], ids=["in-the-json", "in-a-character"]
)
def test_read_skips_a_torn_last_line_with_a_warning(
    directory: Path, caplog: pytest.LogCaptureFixture, torn: bytes
) -> None:
    path = directory / "tasks.jsonl"
    path.write_bytes(b'{"sql":"a"}\n' + torn)

    with caplog.at_level(logging.WARNING, logger="goaly.contracts.jsonl"):
        messages = list(read_jsonl(path, EffiSqlPrediction))

    assert messages == [EffiSqlPrediction(sql="a")]
    (warning,) = caplog.records
    assert warning.levelno == logging.WARNING
    assert warning.getMessage().startswith(f"{path}:2: ")


def test_read_still_rejects_a_malformed_last_line_that_ends_in_a_line_feed(
    directory: Path,
) -> None:
    path = directory / "tasks.jsonl"
    path.write_bytes(A_THEN_TORN + b"\n")

    with pytest.raises(JsonlError, match=r":2: not a valid goaly\.v1\.EffiSqlPrediction: "):
        list(read_jsonl(path, EffiSqlPrediction))
