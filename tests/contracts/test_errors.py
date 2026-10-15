"""Tests for goaly.contracts.errors."""

from goaly.contracts import errors
from goaly.contracts.errors import EnvironmentFailure


def test_the_module_defines_one_exception() -> None:
    classes = [
        value
        for value in vars(errors).values()
        if isinstance(value, type) and value.__module__ == errors.__name__
    ]

    assert classes == [EnvironmentFailure]


def test_environment_failure_is_a_plain_exception() -> None:
    # No handler written for OSError, RuntimeError or ValueError catches it by accident, e.g. a
    # tool's handler for a missing file.
    assert EnvironmentFailure.__mro__ == (EnvironmentFailure, Exception, BaseException, object)
