"""Tests for goaly.contracts.gen, the code generated from proto/goaly/v1/."""

import importlib
import pickle
import pkgutil
import re
import tomllib
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from google.protobuf.message import Message
from google.protobuf.message_factory import GetMessageClass
from google.protobuf.struct_pb2 import ListValue, Struct, Value
from google.protobuf.timestamp_pb2 import Timestamp

import goaly.contracts.gen
from goaly.contracts.gen.task_pb2 import Grade, Prediction, TaskInput
from goaly.contracts.jsonl import from_json_line, to_json_line

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTO_PACKAGE_DIR = REPO_ROOT / "proto" / "goaly" / "v1"
# protoc's header in every generated module, e.g. "# Protobuf Python Version: 7.35.1".
GENERATED_FOR = re.compile(r"^# Protobuf Python Version: (\S+)$", re.MULTILINE)

# Every field the agent sees, by message: all that TaskInput reaches (spec §3.3, §4). Gold
# answers, and whatever would give them away, must never appear here. So a field added anywhere
# TaskInput reaches fails the test below until someone reviews it and adds it here.
AGENT_VISIBLE = {
    "goaly.v1.TaskInput": {"effisql", "swe"},
    "goaly.v1.EffiSqlTaskInput": {"db", "base_sql", "schema", "base_plan", "difficulty"},
    "goaly.v1.SweTaskInput": {
        *("repo", "base_commit", "workload", "test_cmd", "rebuild_cmd", "covering_tests"),
    },
}


def generated_modules() -> list[ModuleType]:
    """Import every module in goaly.contracts.gen."""
    package = goaly.contracts.gen
    return [
        importlib.import_module(f"{package.__name__}.{module.name}")
        for module in pkgutil.iter_modules(package.__path__)
    ]


def message_types(descriptors: Iterable[Descriptor]) -> Iterator[Descriptor]:
    """Yield ``descriptors`` and the message types nested in them, but not map entry types."""
    for descriptor in descriptors:
        if not descriptor.GetOptions().map_entry:
            yield descriptor
            yield from message_types(descriptor.nested_types)


MODULES = generated_modules()
MESSAGES = [
    GetMessageClass(descriptor)
    for module in MODULES
    for descriptor in message_types(module.DESCRIPTOR.message_types_by_name.values())
]


def test_there_is_one_module_per_proto() -> None:
    protos = sorted(f"{path.stem}_pb2" for path in PROTO_PACKAGE_DIR.glob("*.proto"))

    assert protos
    assert sorted(module.__name__.rpartition(".")[2] for module in MODULES) == protos


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_descriptors_keep_the_proto_names(module: ModuleType) -> None:
    proto = module.__name__.rpartition(".")[2].removesuffix("_pb2")

    assert module.DESCRIPTOR.name == f"goaly/v1/{proto}.proto"
    assert module.DESCRIPTOR.package == "goaly.v1"


def test_the_protobuf_floor_is_the_version_the_code_was_generated_for() -> None:
    # The generated code refuses a runtime older than this version, so it must be the floor.
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floors = [
        requirement.removeprefix("protobuf>=")
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("protobuf")
    ]
    versions = set()
    for module in MODULES:
        assert module.__file__ is not None
        header = GENERATED_FOR.search(Path(module.__file__).read_text(encoding="utf-8"))
        assert header is not None, module.__name__
        versions.add(header[1])

    (version,) = versions  # One protoc generated every module.
    assert floors == [version]


@pytest.mark.parametrize("message_type", MESSAGES, ids=lambda cls: cls.DESCRIPTOR.full_name)
def test_messages_pickle_through_the_module_they_live_in(message_type: type[Message]) -> None:
    message = message_type()

    # pickle finds the class through __module__, which must be a goaly.contracts.gen module.
    assert pickle.loads(pickle.dumps(message)) == message  # noqa: S301  # Pickled just above.


@pytest.mark.parametrize(
    ("message_type", "variants"),
    [
        (TaskInput, {"effisql": "goaly.v1.EffiSqlTaskInput", "swe": "goaly.v1.SweTaskInput"}),
        (Prediction, {"effisql": "goaly.v1.EffiSqlPrediction", "swe": "goaly.v1.SwePrediction"}),
        (Grade, {"effisql": "goaly.v1.EffiSqlGrade", "swe": "goaly.v1.SweGrade"}),
    ],
    ids=["TaskInput", "Prediction", "Grade"],
)
def test_task_messages_have_one_variant_per_benchmark(
    message_type: type[Message], variants: dict[str, str]
) -> None:
    oneof = message_type.DESCRIPTOR.oneofs_by_name["benchmark"]

    found = {
        field.name: field.message_type.full_name if field.message_type else None
        for field in oneof.fields
    }
    assert found == variants


def reachable_fields(start: Descriptor) -> dict[str, set[str]]:
    """Map each message type ``start`` reaches, itself included, to its field names."""
    reachable: dict[str, set[str]] = {}
    pending = [start]
    while pending:
        descriptor = pending.pop()
        if descriptor.full_name not in reachable:  # Visit each type once: types can form cycles.
            reachable[descriptor.full_name] = set(descriptor.fields_by_name)
            pending.extend(field.message_type for field in descriptor.fields if field.message_type)
    return reachable


def test_the_agent_sees_only_the_reviewed_task_fields() -> None:
    assert reachable_fields(TaskInput.DESCRIPTOR) == AGENT_VISIBLE


def test_the_walk_ends_on_types_that_form_a_cycle() -> None:
    # Struct holds Values, a Value can hold a ListValue, and a ListValue holds Values.
    assert reachable_fields(Struct.DESCRIPTOR) == {
        "google.protobuf.Struct": {"fields"},
        "google.protobuf.Struct.FieldsEntry": {"key", "value"},
        "google.protobuf.Value": {
            *("null_value", "number_value", "string_value", "bool_value"),
            *("struct_value", "list_value"),
        },
        "google.protobuf.ListValue": {"values"},
    }


# Round trips through JSONL (acceptance 1). The samples are built from the descriptors, so every
# message added to proto/goaly/v1/ is covered without a change here.

# Two different values per scalar type, the first of them never the default.
SCALARS: dict[int, tuple[object, object]] = {
    FieldDescriptor.CPPTYPE_STRING: ("naïve 日本語\nnew line\u2028line separator", "plain"),
    FieldDescriptor.CPPTYPE_BOOL: (True, False),
    FieldDescriptor.CPPTYPE_DOUBLE: (0.1, -2.5e300),
    FieldDescriptor.CPPTYPE_FLOAT: (1.5, -2.25),
    FieldDescriptor.CPPTYPE_INT32: (2**31 - 1, -(2**31)),
    FieldDescriptor.CPPTYPE_UINT32: (2**32 - 1, 1),
    FieldDescriptor.CPPTYPE_INT64: (2**53 + 1, -(2**63)),
    FieldDescriptor.CPPTYPE_UINT64: (2**64 - 1, 2**53 + 1),
}
BYTES = (b"\x00\xff", b"bytes")


def scalar_samples(field: FieldDescriptor) -> tuple[object, object]:
    """Two different values for a scalar or enum field, the first of them not its default."""
    if field.enum_type is not None:
        numbers = [value.number for value in field.enum_type.values if value.number != 0]
        return numbers[0], numbers[-1]
    if field.type == FieldDescriptor.TYPE_BYTES:
        return BYTES
    return SCALARS[field.cpp_type]


def well_known_sample(full_name: str) -> Message:
    """A value of a google.protobuf message type."""
    if full_name == "google.protobuf.Timestamp":
        return Timestamp(seconds=1_760_486_523, nanos=123_456_789)
    if full_name == "google.protobuf.ListValue":
        row = ListValue()
        row.extend([None, "NULL", 1.5, True])  # SQL NULL beside the string "NULL".
        return row
    if full_name == "google.protobuf.Struct":
        struct = Struct()
        struct.update({"text": "naïve", "number": 1.5, "flag": True, "none": None, "list": ["a"]})
        return struct
    if full_name == "google.protobuf.Value":
        return Value(string_value="NULL")
    raise AssertionError(f"no sample for {full_name}: add one to well_known_sample")


def fill(message: Message, variant: int) -> None:
    """Set ``message`` in place: populate one of ours, copy a sample into a well-known type."""
    if message.DESCRIPTOR.file.package == "google.protobuf":
        message.CopyFrom(well_known_sample(message.DESCRIPTOR.full_name))
    else:
        populate(message, variant)


def set_sample(message: Message, field: FieldDescriptor, variant: int) -> None:
    """Set ``field``: maps and repeated fields get two entries, messages are filled in."""
    value = getattr(message, field.name)
    entry = field.message_type
    if entry is not None and entry.GetOptions().map_entry:
        key_field, value_field = entry.fields_by_name["key"], entry.fields_by_name["value"]
        for key in scalar_samples(key_field):
            if value_field.message_type is None:
                value[key] = scalar_samples(value_field)[0]
            else:
                fill(value[key], variant)
    elif field.is_repeated:
        if entry is None:
            value.extend(scalar_samples(field))
        else:
            fill(value.add(), variant)
            fill(value.add(), variant)
    elif entry is None:
        setattr(message, field.name, scalar_samples(field)[0])
    else:
        fill(value, variant)
        value.SetInParent()  # Present even if its type has no fields.


def populate(message: Message, variant: int) -> Message:
    """Set every field of ``message`` to a value other than its default, and return it.

    In each oneof only one branch is set: branch number ``variant``, modulo the branch count.
    """
    descriptor = message.DESCRIPTOR
    assert isinstance(descriptor, Descriptor)  # Always true; narrows the stubs' upb alternative.
    for field in descriptor.fields:
        oneof = field.containing_oneof
        if oneof is None or oneof.fields[variant % len(oneof.fields)].name == field.name:
            set_sample(message, field, variant)
    return message


def samples(message_type: type[Message]) -> list[Message]:
    """Fully set messages of ``message_type``, one per branch of its largest oneof."""
    oneofs = message_type.DESCRIPTOR.oneofs
    variants = max((len(oneof.fields) for oneof in oneofs), default=1)
    return [populate(message_type(), variant) for variant in range(variants)]


def sample_id(message: Message) -> str:
    """The message's type, and the branch it sets in each oneof that has a choice."""
    descriptor = message.DESCRIPTOR
    branches = [
        message.WhichOneof(oneof.name) or "unset"
        for oneof in descriptor.oneofs
        if len(oneof.fields) > 1
    ]
    return f"{descriptor.full_name}[{','.join(branches)}]" if branches else descriptor.full_name


SAMPLES = [sample for message_type in MESSAGES for sample in samples(message_type)]


@pytest.mark.parametrize("message", SAMPLES, ids=sample_id)
def test_every_message_round_trips_through_jsonl(message: Message) -> None:
    assert from_json_line(to_json_line(message), type(message)) == message


@pytest.mark.parametrize("message_type", MESSAGES, ids=lambda cls: cls.DESCRIPTOR.full_name)
def test_every_message_with_only_defaults_round_trips_through_jsonl(
    message_type: type[Message],
) -> None:
    read = from_json_line(to_json_line(message_type()), message_type)

    assert read == message_type()
    assert read.ListFields() == []  # Reading the written defaults back sets nothing.


@pytest.mark.parametrize("message", SAMPLES, ids=sample_id)
def test_the_samples_set_every_field(message: Message) -> None:
    descriptor = message.DESCRIPTOR
    expected: set[str | None] = {
        field.name for field in descriptor.fields if field.containing_oneof is None
    }
    expected |= {message.WhichOneof(oneof.name) for oneof in descriptor.oneofs}

    # ListFields lists exactly the fields that hold something other than their default.
    assert {field.name for field, _ in message.ListFields()} == expected


def test_the_samples_take_every_oneof_branch() -> None:
    branches = {
        (message_type.DESCRIPTOR.full_name, field.name)
        for message_type in MESSAGES
        for oneof in message_type.DESCRIPTOR.oneofs
        for field in oneof.fields
    }

    taken = {
        (sample.DESCRIPTOR.full_name, sample.WhichOneof(oneof.name))
        for sample in SAMPLES
        for oneof in sample.DESCRIPTOR.oneofs
    }

    assert taken == branches
