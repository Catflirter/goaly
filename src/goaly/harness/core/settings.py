"""Process settings: the model gateway's API key and base URL, and the RSI lane's secrets.

Each value comes from the first of these that provides it: explicit arguments, ``GOALY_*``
environment variables, the env file (``~/.config/goaly/.env`` by default), the field default.
"""

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import SecretStr, ValidationError, field_validator
from pydantic.fields import FieldInfo
from pydantic_core import ErrorDetails, PydanticCustomError, PydanticKnownError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

__all__ = ["Settings", "SettingsError", "default_env_file", "load_settings"]

_ENV_PREFIX = "GOALY_"
_NOT_AN_HTTP_URL = "must be an absolute http or https URL with a host"
_NOT_A_HEADER_VALUE = "must be printable ASCII, with no whitespace or control characters"
# ntfy.sh topics: 1 to 64 letters, digits, "_" or "-".
_NTFY_TOPIC = re.compile(r"[A-Za-z0-9_-]{1,64}")


class SettingsError(Exception):
    """The settings are missing or invalid; the message says what to fix."""


class Settings(BaseSettings):
    """Validated settings. Build them with :func:`load_settings` and pass them down explicitly."""

    model_config = SettingsConfigDict(
        env_prefix=_ENV_PREFIX,
        env_file_encoding="utf-8",  # Not the locale's encoding.
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,  # A ValidationError would quote the key otherwise.
    )

    api_key: SecretStr
    base_url: str = "https://api.goaly.ai/inference/v1"
    # The ntfy.sh topic the RSI lane notifies the project lead on (spec §8.7). Anyone who knows
    # it can read and post, so it is a secret; None turns notifications off.
    ntfy_topic: SecretStr | None = None
    # The GitHub token for results PRs (spec §8.7), read at run time: goaly.rsi reloads the
    # settings for each use, and falls back to the Coder workspace's token when this is unset.
    github_token: SecretStr | None = None

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
    def _check_api_key(cls, value: object) -> object:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise PydanticKnownError("missing")  # A blank key argument counts as missing.
            # The key goes in the Authorization header. httpx refuses any other character, and
            # quotes the whole header, key and all, in its error.
            if not all("!" <= char <= "~" for char in value):
                raise PydanticCustomError("invalid_key", _NOT_A_HEADER_VALUE)
        return value

    @field_validator("ntfy_topic", mode="before")
    @classmethod
    def _check_ntfy_topic(cls, value: object) -> object:
        value = _optional_secret(value)
        if isinstance(value, str) and not _NTFY_TOPIC.fullmatch(value):
            raise PydanticCustomError(
                "invalid_topic", "must be 1 to 64 letters, digits, underscores or hyphens"
            )
        return value

    @field_validator("github_token", mode="before")
    @classmethod
    def _check_github_token(cls, value: object) -> object:
        value = _optional_secret(value)
        if isinstance(value, str) and not all("!" <= char <= "~" for char in value):
            raise PydanticCustomError("invalid_token", _NOT_A_HEADER_VALUE)  # A header value.
        return value

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        url = value.strip()
        problem = _base_url_problem(url)
        if problem is not None:
            raise PydanticCustomError("invalid_url", problem)
        return url.rstrip("/")


def _optional_secret(value: object) -> object:
    """Unwrap and strip an optional secret's value; a blank one counts as unset."""
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


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


def _base_url_problem(url: str) -> str | None:
    """Say why ``url`` cannot be the base URL, or return None. The reason never quotes ``url``."""
    # urlsplit silently drops some of these characters, so it must never see them.
    if any(char.isspace() or not char.isprintable() for char in url):
        return "must not contain whitespace or control characters"
    try:
        parts = urlsplit(url)
    except ValueError:  # For example, an unclosed "[" around an IPv6 host.
        return _NOT_AN_HTTP_URL
    if "@" in parts.netloc:
        return "must not contain credentials"
    if "?" in url or "#" in url:
        return "must not have a query or a fragment"
    try:
        _ = parts.port  # Raises ValueError for a malformed port.
    except ValueError:
        return _NOT_AN_HTTP_URL
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return _NOT_AN_HTTP_URL
    return None


def _describe(detail: ErrorDetails, env_file: Path) -> str:
    if not detail["loc"]:  # A model-level error belongs to no single variable.
        return f"The settings are invalid: {detail['msg']}."
    variable = _ENV_PREFIX + str(detail["loc"][0]).upper()
    if detail["type"] == "missing":
        return (
            f"{variable} is not set. Export it, or add the line {variable}=<value> to {env_file}."
        )
    return f"{variable} is invalid: {detail['msg']}."
