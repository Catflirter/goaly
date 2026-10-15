"""Process settings: the model gateway's API key and base URL.

Each value comes from the first of these that provides it: explicit arguments, ``GOALY_*``
environment variables, the env file (``~/.config/goaly/.env`` by default), the field default.
"""

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import SecretStr, ValidationError, field_validator
from pydantic.fields import FieldInfo
from pydantic_core import ErrorDetails, PydanticCustomError, PydanticKnownError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

__all__ = ["Settings", "SettingsError", "default_env_file", "load_settings"]

_ENV_PREFIX = "GOALY_"


class SettingsError(Exception):
    """The settings are missing or invalid; the message says what to fix."""


class Settings(BaseSettings):
    """Validated settings. Build them with :func:`load_settings` and pass them down explicitly."""

    model_config = SettingsConfigDict(
        env_prefix=_ENV_PREFIX,
        env_file_encoding="utf-8",  # Not the locale's encoding.
        extra="ignore",
        frozen=True,
    )

    api_key: SecretStr
    base_url: str = "https://api.goaly.ai/inference/v1"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest priority first, and nothing else: no secrets directory, and the dotenv source
        # reads only the file load_settings passes in, never a `.env` in the working directory.
        # A blank value in the environment or the env file counts as absent, so the next applies.
        return init_settings, _BlankIsAbsent(env_settings), _BlankIsAbsent(dotenv_settings)

    @field_validator("api_key", mode="before")
    @classmethod
    def _strip_api_key(cls, value: object) -> object:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise PydanticKnownError("missing")  # A blank key argument counts as missing.
        return value

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        if not _is_absolute_http_url(value):
            raise PydanticCustomError(
                "invalid_url", "must be an absolute http or https URL with a host"
            )
        return value.rstrip("/")


def default_env_file() -> Path:
    """Return ``~/.config/goaly/.env`` for the current ``HOME``."""
    return Path.home() / ".config" / "goaly" / ".env"


def load_settings(env_file: Path | None = None) -> Settings:
    """Read the settings from the environment and ``env_file`` (default: :func:`default_env_file`).

    A missing env file is not an error. Missing or invalid values, and an env file that cannot be
    read, raise :class:`SettingsError`, whose message names every problem but never echoes a value.
    """
    env_file = default_env_file() if env_file is None else env_file
    details: list[ErrorDetails] = []
    unreadable = ""
    # The except blocks keep only data built here: the exceptions themselves carry the inputs
    # (a decoding error holds the file's raw bytes), so none of them may outlive its block.
    try:
        return Settings(_env_file=env_file)
    except ValidationError as error:
        details = error.errors(include_input=False)
    except UnicodeDecodeError as error:
        unreadable = f"not valid UTF-8 at byte {error.start}"
    except OSError as error:
        unreadable = error.strerror or type(error).__name__
    # Raised outside the except blocks, and from None: no exception is attached as the cause or
    # the context.
    if unreadable:
        raise SettingsError(f"Cannot read {env_file}: {unreadable}.") from None
    raise SettingsError("\n".join(_describe(detail, env_file) for detail in details)) from None


class _BlankIsAbsent(PydanticBaseSettingsSource):
    """Wraps a source and drops its blank values, so that a later source supplies those fields."""

    def __init__(self, source: PydanticBaseSettingsSource) -> None:
        super().__init__(source.settings_cls)
        self._source = source

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._source.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        return {name: value for name, value in self._source().items() if not _is_blank(value)}


def _is_blank(value: object) -> bool:
    return isinstance(value, str) and not value.strip()


def _is_absolute_http_url(value: str) -> bool:
    try:
        url = urlsplit(value)
        _ = url.port  # Raises ValueError for a malformed port.
    except ValueError:
        return False
    return url.scheme in {"http", "https"} and bool(url.hostname)


def _describe(detail: ErrorDetails, env_file: Path) -> str:
    if not detail["loc"]:  # A model-level error belongs to no single variable.
        return f"The settings are invalid: {detail['msg']}."
    variable = _ENV_PREFIX + str(detail["loc"][0]).upper()
    if detail["type"] == "missing":
        return (
            f"{variable} is not set. Export it, or add the line {variable}=<value> to {env_file}."
        )
    return f"{variable} is invalid: {detail['msg']}."
