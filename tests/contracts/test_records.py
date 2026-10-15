"""Tests for the run and experiment records: trajectory, result, manifest and experiment."""

import json

from goaly.contracts.gen.experiment_pb2 import PhaseResult
from goaly.contracts.jsonl import from_json_line, to_json_line


def test_an_optional_statistic_keeps_its_presence_through_jsonl() -> None:
    # A sign-test p of exactly 0 was computed; a gain probability never set was not.
    line = to_json_line(PhaseResult(sign_test_p=0.0))

    fields = json.loads(line)
    read = from_json_line(line, PhaseResult)

    assert fields["sign_test_p"] == 0.0
    assert "gain_probability" not in fields
    assert read.HasField("sign_test_p")
    assert not read.HasField("gain_probability")
