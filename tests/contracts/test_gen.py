"""Tests for goaly.contracts.gen, the code generated from proto/goaly/v1/."""

import importlib
import pickle
import pkgutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
from google.protobuf.descriptor import Descriptor
from google.protobuf.message import Message
from google.protobuf.message_factory import GetMessageClass

import goaly.contracts.gen
from goaly.contracts.gen.task_pb2 import Grade, Prediction, TaskInput

PROTO_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "proto" / "goaly" / "v1"

# Dataset fields the agent must never see (spec §4): gold answers and what would give them away.
HIDDEN_FROM_THE_AGENT = {
    # BIRD-CRITIC: the gold SQL and the test cases that check a prediction.
    "sol_sql",
    "test_cases",
    # SWE-fficiency: the expert patch, its tests, the issue text and hints, and the gold speedup.
    "patch",
    "test_patch",
    "problem_statement",
    "hints_text",
    "speedup",
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


@pytest.mark.parametrize("message_type", MESSAGES, ids=lambda cls: cls.DESCRIPTOR.full_name)
def test_messages_pickle_through_the_module_they_live_in(message_type: type[Message]) -> None:
    message = message_type()

    # pickle finds the class through __module__, which must be a goaly.contracts.gen module.
    assert pickle.loads(pickle.dumps(message)) == message  # noqa: S301  # Pickled just above.


@pytest.mark.parametrize(
    ("message_type", "variants"),
    [
        (TaskInput, {"bird": "goaly.v1.BirdTaskInput", "swe": "goaly.v1.SweTaskInput"}),
        (Prediction, {"bird": "goaly.v1.BirdPrediction", "swe": "goaly.v1.SwePrediction"}),
        (Grade, {"bird": "goaly.v1.BirdGrade", "swe": "goaly.v1.SweGrade"}),
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


def test_task_input_holds_nothing_the_agent_must_not_see() -> None:
    names: set[str] = set()
    pending = [TaskInput.DESCRIPTOR]
    while pending:
        descriptor = pending.pop()
        names.update(descriptor.fields_by_name)
        pending.extend(field.message_type for field in descriptor.fields if field.message_type)

    assert {"instance_id", "db_id", "repo"} <= names  # The walk reached both variants.
    assert names.isdisjoint(HIDDEN_FROM_THE_AGENT)
