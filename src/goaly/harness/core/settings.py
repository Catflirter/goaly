"""Process settings: the model gateway's API key and base URL.

Each value comes from the first of these that provides it: explicit arguments, ``GOALY_*``
environment variables, the env file (``~/.config/goaly/.env`` by default), the field default.
"""

from pathlib import Path
from urllib.parse import urlsplit

from pydantic import SecretStr, ValidationError, field_validator
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
        env_ignore_empty=True,  # An empty variable counts as unset, so the next source applies.
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
        return init_settings, env_settings, dotenv_settings

    @field_validator("api_key", mode="before")
    @classmethod
    def _strip_api_key(cls, value: object) -> object:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise PydanticKnownError("missing")  # A blank key counts as missing.
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

    A missing env file is not an error. Missing or invalid values raise :class:`SettingsError`,
    whose message names every problem but never echoes a value.
    """
    env_file = default_env_file() if env_file is None else env_file
    try:
        return Settings(_env_file=env_file)
    except ValidationError as error:
        problems = [_describe(detail, env_file) for detail in error.errors()]
    # Raised outside the except block, and from None, so that pydantic's ValidationError, which
    # echoes the inputs, is attached neither as the cause nor as the context.
    raise SettingsError("\n".join(problems)) from None


def _is_absolute_http_url(value: str) -> bool:
    try:
        url = urlsplit(value)
        _ = url.port  # Raises ValueError for a malformed port.
    except ValueError:
        return False
    return url.scheme in {"http", "https"} and bool(url.hostname)


def _describe(detail: ErrorDetails, env_file: Path) -> str:
    variable = _ENV_PREFIX + str(detail["loc"][0]).upper()
    if detail["type"] == "missing":
        return (
            f"{variable} is not set. Export it, or add the line {variable}=<value> to {env_file}."
        )
    return f"{variable} is invalid: {detail['msg']}."
