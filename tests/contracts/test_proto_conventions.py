"""The conventions every contract in proto/goaly/v1/ follows (spec §3.3).

One rule no test can check: JSONL files key fields and enum values by name, so never rename one;
when one is removed, reserve its name as well as its number.
"""

import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from google.protobuf.descriptor_pb2 import (
    DescriptorProto,
    EnumDescriptorProto,
    FieldDescriptorProto,
    FileDescriptorProto,
    FileDescriptorSet,
)

PROTO_ROOT = Path(__file__).resolve().parents[2] / "proto"

# Field numbers in descriptor.proto, which make up the paths in SourceCodeInfo.
FILE_MESSAGE, FILE_ENUM = 4, 5
MESSAGE_FIELD, MESSAGE_NESTED, MESSAGE_ENUM, MESSAGE_ONEOF = 2, 3, 4, 8
ENUM_VALUE = 2

type SourcePath = tuple[int, ...]


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> list[FileDescriptorProto]:
    """The descriptors of every .proto, comments included, as protoc parses them."""
    protos = sorted(path.relative_to(PROTO_ROOT).as_posix() for path in PROTO_ROOT.rglob("*.proto"))
    assert protos
    descriptor_set = tmp_path_factory.mktemp("descriptors") / "contracts.binpb"
    # The arguments are fixed flags and the paths of the repository's own .proto files.
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={PROTO_ROOT}",
            "--include_source_info",
            f"--descriptor_set_out={descriptor_set}",
            *protos,
        ],
        check=True,
        timeout=120,
    )
    return list(FileDescriptorSet.FromString(descriptor_set.read_bytes()).file)


def messages(file: FileDescriptorProto) -> Iterator[tuple[DescriptorProto, SourcePath]]:
    """Yield every message type in ``file`` with its source path, nested ones included."""
    pending: list[tuple[DescriptorProto, SourcePath]] = [
        (message, (FILE_MESSAGE, index)) for index, message in enumerate(file.message_type)
    ]
    while pending:
        message, path = pending.pop()
        if message.options.map_entry:  # protoc writes these for map fields; they have no source.
            continue
        yield message, path
        pending.extend(
            (nested, (*path, MESSAGE_NESTED, index))
            for index, nested in enumerate(message.nested_type)
        )


def enums(file: FileDescriptorProto) -> Iterator[tuple[EnumDescriptorProto, SourcePath]]:
    """Yield every enum type in ``file`` with its source path, nested ones included."""
    for index, enum in enumerate(file.enum_type):
        yield enum, (FILE_ENUM, index)
    for message, path in messages(file):
        for index, enum in enumerate(message.enum_type):
            yield enum, (*path, MESSAGE_ENUM, index)


def fields(file: FileDescriptorProto) -> Iterator[tuple[str, FieldDescriptorProto, SourcePath]]:
    """Yield every field in ``file`` with its message's name and its source path."""
    for message, path in messages(file):
        for index, field in enumerate(message.field):
            yield message.name, field, (*path, MESSAGE_FIELD, index)


def documentable(file: FileDescriptorProto) -> Iterator[tuple[str, SourcePath]]:
    """Yield the name and source path of every message, field, oneof, enum and enum value."""
    for message, path in messages(file):
        yield message.name, path
        synthetic = {field.oneof_index for field in message.field if field.proto3_optional}
        for index, oneof in enumerate(message.oneof_decl):
            if index not in synthetic:  # protoc adds one per `optional` field, with no source.
                yield f"{message.name}.{oneof.name}", (*path, MESSAGE_ONEOF, index)
    for message_name, field, path in fields(file):
        yield f"{message_name}.{field.name}", path
    for enum, path in enums(file):
        yield enum.name, path
        for index, value in enumerate(enum.value):
            yield f"{enum.name}.{value.name}", (*path, ENUM_VALUE, index)


def leading_comments(file: FileDescriptorProto) -> dict[SourcePath, str]:
    """The comment above each element of ``file``, by source path."""
    return {tuple(where.path): where.leading_comments for where in file.source_code_info.location}


def test_every_file_is_proto3_in_package_goaly_v1(files: list[FileDescriptorProto]) -> None:
    assert [
        file.name for file in files if (file.syntax, file.package) != ("proto3", "goaly.v1")
    ] == []


def test_every_element_has_a_comment(files: list[FileDescriptorProto]) -> None:
    undocumented = []
    for file in files:
        comments = leading_comments(file)
        undocumented += [
            f"{file.name}: {name}"
            for name, path in documentable(file)
            if not comments.get(path, "").strip()
        ]

    assert undocumented == []


def test_enums_start_unspecified_and_prefix_every_value(files: list[FileDescriptorProto]) -> None:
    for file in files:
        for enum, _ in enums(file):
            prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", enum.name).upper() + "_"
            assert (enum.value[0].name, enum.value[0].number) == (f"{prefix}UNSPECIFIED", 0)
            assert [value.name for value in enum.value if not value.name.startswith(prefix)] == []


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("_at", ".google.protobuf.Timestamp"),  # Points in time.
        ("_s", "double"),  # Durations, in seconds.
        ("_usd", "double"),  # Money.
        ("_tokens", "int64"),  # Token counts.
    ],
)
def test_field_names_carry_their_units(
    files: list[FileDescriptorProto], suffix: str, expected: str
) -> None:
    types = {
        FieldDescriptorProto.TYPE_DOUBLE: "double",
        FieldDescriptorProto.TYPE_INT64: "int64",
    }
    wrong = [
        f"{message}.{field.name}"
        for file in files
        for message, field, _ in fields(file)
        if field.name.endswith(suffix) and (field.type_name or types.get(field.type)) != expected
    ]

    assert wrong == []


def test_every_timestamp_is_named_at(files: list[FileDescriptorProto]) -> None:
    # The other direction of the _at rule: no point in time goes by another name.
    assert [
        f"{message}.{field.name}"
        for file in files
        for message, field, _ in fields(file)
        if field.type_name == ".google.protobuf.Timestamp" and not field.name.endswith("_at")
    ] == []


@pytest.mark.parametrize(
    ("unit", "suffix"),
    [
        (r"\bseconds\b", "_s"),  # Durations.
        (r"\bdollars\b", "_usd"),  # Money.
    ],
)
def test_fields_documented_in_a_unit_carry_its_suffix(
    files: list[FileDescriptorProto], unit: str, suffix: str
) -> None:
    # The other direction of the _s and _usd rules. A double does not say what it measures, so
    # the unit its comment states stands in for its type.
    wrong = []
    for file in files:
        comments = leading_comments(file)
        wrong += [
            f"{message}.{field.name}"
            for message, field, path in fields(file)
            if re.search(unit, comments.get(path, "")) and not field.name.endswith(suffix)
        ]

    assert wrong == []


def test_no_field_uses_a_duration(files: list[FileDescriptorProto]) -> None:
    # Durations are `double <name>_s`, which reads better in JSON than "1.5s".
    assert [
        f"{message}.{field.name}"
        for file in files
        for message, field, _ in fields(file)
        if field.type_name == ".google.protobuf.Duration"
    ] == []


def test_nothing_is_named_benchmark(files: list[FileDescriptorProto]) -> None:
    # Benchmark is the Protocol's name; a benchmark is named by its adapter's string instead.
    names = [message.name for file in files for message, _ in messages(file)]
    names += [enum.name for file in files for enum, _ in enums(file)]

    assert "Benchmark" not in names


def test_enum_values_are_declared_in_numeric_order(files: list[FileDescriptorProto]) -> None:
    # An order that matters, such as the order guards run in, is stated in the enum's comment.
    assert [
        f"{file.name}: {enum.name}"
        for file in files
        for enum, _ in enums(file)
        if [value.number for value in enum.value] != sorted(value.number for value in enum.value)
    ] == []
