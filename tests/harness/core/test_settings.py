"""Tests for goaly.harness.core.settings."""

import os
import traceback
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError
from pydantic_core import ErrorDetails

from goaly.harness.core.settings import (
    Settings,
    SettingsError,
    _describe,
    default_env_file,
    load_settings,
)

DEFAULT_BASE_URL = "https://api.goaly.ai/inference/v1"


def write_env_file(path: Path, *lines: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


# Loading and source precedence.


def test_default_env_file_is_computed_from_home_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "first"))
    first = default_env_file()
    monkeypatch.setenv("HOME", str(tmp_path / "second"))

    assert first == tmp_path / "first" / ".config" / "goaly" / ".env"
    assert default_env_file() == tmp_path / "second" / ".config" / "goaly" / ".env"


def test_reads_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "env-key")

    settings = load_settings()

    assert settings.api_key.get_secret_value() == "env-key"
    assert settings.base_url == DEFAULT_BASE_URL


def test_reads_the_default_env_file() -> None:
    write_env_file(
        default_env_file(), "GOALY_API_KEY=file-key", "GOALY_BASE_URL=https://file.test/v1"
    )

    settings = load_settings()

    assert settings.api_key.get_secret_value() == "file-key"
    assert settings.base_url == "https://file.test/v1"


def test_reads_an_explicit_env_file(tmp_path: Path) -> None:
    env_file = write_env_file(tmp_path / "custom.env", "GOALY_API_KEY=file-key")

    assert load_settings(env_file).api_key.get_secret_value() == "file-key"


def test_environment_overrides_the_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    write_env_file(
        default_env_file(), "GOALY_API_KEY=file-key", "GOALY_BASE_URL=https://file.test/v1"
    )
    monkeypatch.setenv("GOALY_BASE_URL", "https://env.test/v1")

    settings = load_settings()

    assert settings.api_key.get_secret_value() == "file-key"
    assert settings.base_url == "https://env.test/v1"


def test_init_arguments_override_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "env-key")

    settings = Settings(api_key=SecretStr("init-key"))

    assert settings.api_key.get_secret_value() == "init-key"


def test_a_missing_env_file_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "env-key")

    assert load_settings(tmp_path / "absent.env").api_key.get_secret_value() == "env-key"


def test_never_reads_a_dotenv_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_env_file(tmp_path / ".env", "GOALY_API_KEY=cwd-key")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SettingsError, match="GOALY_API_KEY is not set"):
        load_settings()


def test_ignores_unknown_goaly_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    write_env_file(default_env_file(), "GOALY_API_KEY=file-key", "GOALY_UNKNOWN=file")
    monkeypatch.setenv("GOALY_ANOTHER_UNKNOWN", "env")

    assert load_settings().api_key.get_secret_value() == "file-key"


def test_reads_the_environment_again_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "first-key")
    first = load_settings()
    monkeypatch.setenv("GOALY_API_KEY", "second-key")

    assert first.api_key.get_secret_value() == "first-key"
    assert load_settings().api_key.get_secret_value() == "second-key"


# Errors.


def test_a_missing_key_names_the_variable_and_the_env_file_consulted() -> None:
    with pytest.raises(SettingsError) as raised:
        load_settings()

    assert str(raised.value) == (
        "GOALY_API_KEY is not set. Export it, or add the line GOALY_API_KEY=<value> "
        f"to {Path.home()}/.config/goaly/.env."
    )


def test_a_missing_key_names_an_explicit_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"

    with pytest.raises(SettingsError) as raised:
        load_settings(env_file)

    assert str(raised.value).endswith(f"to {env_file}.")


def test_an_env_file_that_is_not_utf8_names_the_file_and_leaks_nothing(tmp_path: Path) -> None:
    env_file = tmp_path / "latin1.env"
    env_file.write_bytes(b"GOALY_API_KEY=canary-secret-XYZ\n# cl\xe9 API\n")

    with pytest.raises(SettingsError) as raised:
        load_settings(env_file)

    error = raised.value
    assert str(error) == f"Cannot read {env_file}: not valid UTF-8 at byte 36."
    # The decoding error holds the undecoded bytes, key included, so it must not travel along.
    assert error.__cause__ is None
    assert error.__context__ is None
    rendered = "".join(traceback.format_exception(error))
    for text in (str(error), repr(error), repr(error.args), rendered):
        assert "canary" not in text


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a file whatever its mode")
def test_an_unreadable_env_file_names_the_file(tmp_path: Path) -> None:
    env_file = write_env_file(tmp_path / "locked.env", "GOALY_API_KEY=file-key")
    env_file.chmod(0)

    with pytest.raises(SettingsError) as raised:
        load_settings(env_file)

    assert str(raised.value) == f"Cannot read {env_file}: Permission denied."
    assert raised.value.__context__ is None


def test_describes_a_model_level_error_without_a_value(tmp_path: Path) -> None:
    detail: ErrorDetails = {"type": "value_error", "loc": (), "msg": "keys disagree", "input": "x"}

    assert _describe(detail, tmp_path / ".env") == "The settings are invalid: keys disagree."


# Validation.


@pytest.mark.parametrize("blank", ["", "   ", "\t \n"])
def test_a_blank_key_counts_as_missing(blank: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_API_KEY", blank)

    with pytest.raises(SettingsError, match=r"^GOALY_API_KEY is not set"):
        load_settings()


def test_a_blank_key_argument_counts_as_missing() -> None:
    with pytest.raises(ValidationError) as raised:
        Settings(api_key=SecretStr("  "))

    assert [error["type"] for error in raised.value.errors()] == ["missing"]


def test_a_key_that_is_not_a_string_is_rejected() -> None:
    with pytest.raises(ValidationError) as raised:
        Settings(api_key=12345)  # type: ignore[arg-type]  # The wrong type is the point.

    assert [error["type"] for error in raised.value.errors()] == ["string_type"]


def test_a_blank_key_in_the_env_file_counts_as_missing() -> None:
    write_env_file(default_env_file(), 'GOALY_API_KEY="   "')

    with pytest.raises(SettingsError, match=r"^GOALY_API_KEY is not set"):
        load_settings()


def test_an_empty_variable_falls_back_to_the_next_source(monkeypatch: pytest.MonkeyPatch) -> None:
    write_env_file(default_env_file(), "GOALY_API_KEY=file-key")
    monkeypatch.setenv("GOALY_API_KEY", "")
    monkeypatch.setenv("GOALY_BASE_URL", "")

    settings = load_settings()

    assert settings.api_key.get_secret_value() == "file-key"
    assert settings.base_url == DEFAULT_BASE_URL


def test_strips_whitespace_around_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_API_KEY", " \tenv-key \n")

    assert load_settings().api_key.get_secret_value() == "env-key"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://example.test/v1/", "https://example.test/v1"),
        ("https://example.test/", "https://example.test"),
        ("http://localhost:8080/v1", "http://localhost:8080/v1"),
    ],
)
def test_accepts_http_base_urls_without_a_trailing_slash(
    base_url: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "env-key")
    monkeypatch.setenv("GOALY_BASE_URL", base_url)

    assert load_settings().base_url == expected


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.test/v1",
        "example.test/v1",
        "/inference/v1",
        "https://",
        "https:///v1",
        "https://example.test:port/v1",
        "https://example.test:99999/v1",
    ],
)
def test_rejects_an_invalid_base_url(base_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "env-key")
    monkeypatch.setenv("GOALY_BASE_URL", base_url)

    with pytest.raises(SettingsError) as raised:
        load_settings()

    assert str(raised.value) == (
        "GOALY_BASE_URL is invalid: must be an absolute http or https URL with a host."
    )


def test_reports_every_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_BASE_URL", "ftp://example.test")

    with pytest.raises(SettingsError) as raised:
        load_settings()

    assert str(raised.value).splitlines() == [
        "GOALY_API_KEY is not set. Export it, or add the line GOALY_API_KEY=<value> "
        f"to {default_env_file()}.",
        "GOALY_BASE_URL is invalid: must be an absolute http or https URL with a host.",
    ]


# Secret safety and immutability.


def test_repr_and_str_hide_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "canary-key")

    settings = load_settings()

    assert "canary-key" not in repr(settings)
    assert "canary-key" not in str(settings)


def test_errors_echo_no_values_and_chain_no_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "canary-key")
    monkeypatch.setenv("GOALY_BASE_URL", "ftp://canary-host.test")

    with pytest.raises(SettingsError) as raised:
        load_settings()

    # pydantic's ValidationError echoes inputs, so it must not travel with the error at all.
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    rendered = "".join(traceback.format_exception(raised.value))
    assert "canary" not in rendered
    assert "canary" not in repr(raised.value)


def test_settings_are_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "env-key")
    settings = load_settings()

    with pytest.raises(ValidationError, match="frozen"):
        settings.base_url = "https://other.test/v1"  # type: ignore[misc]  # Tests the runtime check.
