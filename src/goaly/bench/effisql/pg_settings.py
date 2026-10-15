"""Settings of the Effi-SQL PostgreSQL environment (spec §4.1, §6.2).

:class:`PgEnvConfig` says where the environment's data lives and how its servers run; it holds no
secret. :class:`PgCredentials` holds the passwords of its three roles, which live only in
``~/.config/goaly/pg.env`` (mode 0600), outside the repository: :func:`ensure_credentials`
creates that file with random passwords, and :func:`load_credentials` reads it. Neither ever
prints a password, and no error message quotes one.
"""

import os
import secrets
import stat
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

__all__ = [
    "ADMIN_ROLE",
    "AGENT_ROLE",
    "ALT_DB",
    "GRADER_ROLE",
    "MAIN_DB",
    "POSTGRES_IMAGE",
    "PgCredentials",
    "PgEnvConfig",
    "PgSettingsError",
    "default_credentials_file",
    "ensure_credentials",
    "load_credentials",
    "write_private",
]

# postgres:16 as pulled on 2026-10-15 (PostgreSQL 16.15), the major version the official
# evaluator uses. Pinned by digest, so every instance runs the same server build.
POSTGRES_IMAGE = "postgres@sha256:ca0bd484cb98bf4b24eb1010e73fb3fcbd6714d240fbc1a10eea5b7dbecb641d"

MAIN_DB = "tpch"  # TPC-H at scale factor 3: the database the tasks run on.
ALT_DB = "tpch_alt"  # TPC-H at scale factor 0.3: the grader's robustness check (spec §4.1).

ADMIN_ROLE = "postgres"  # The bootstrap superuser; only setup uses it.
AGENT_ROLE = "effisql_agent"  # The agent's read-only role, on the solving instances.
GRADER_ROLE = "effisql_grader"  # The grader's role: SELECT on both databases.

_ENV_PREFIX = "GOALY_PG_"
_PASSWORD_FIELDS = ("admin_password", "agent_password", "grader_password")
_PASSWORD_BYTES = 32  # Each password is 32 random bytes, URL-safe base64: 43 characters.


class PgSettingsError(Exception):
    """The credentials file is missing, unsafe or invalid; the message says what to fix."""


class PgEnvConfig(BaseModel):
    """Where the environment's data lives and how its servers run. It holds no secret."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_root: Path = Field(default_factory=lambda: Path.home() / "data" / "effisql")
    image: str = POSTGRES_IMAGE
    host: str = "127.0.0.1"
    # The one-off server that loads the golden data directory; the instances use other ports.
    build_port: int = Field(default=55439, ge=1024, le=65535)
    # The build runs on the instances' CPUs, before they start, leaving CPUs 0, 1, 12 and 13
    # to the system and our Python processes (spec §6.2).
    build_cpuset: str | None = "2-11,14-23"
    build_container: str = "goaly-effisql-build"

    @property
    def dbgen_dir(self) -> Path:
        """Where the pinned dbgen is checked out and built."""
        return self.data_root / "dbgen"

    def raw_dir(self, scale: str) -> Path:
        """Where dbgen writes the ``.tbl`` files for scale factor ``scale``."""
        return self.data_root / "raw" / f"sf{scale}"

    @property
    def golden_pgdata(self) -> Path:
        """The golden data directory: built once, then copied into every instance."""
        return self.data_root / "golden" / "pgdata"

    @property
    def golden_record(self) -> Path:
        """The golden build's record, written last, so it exists only for a complete build."""
        return self.data_root / "golden" / "golden.json"

    def snapshot_path(self, db: str) -> Path:
        """The schema snapshot of database ``db``, the hand-off to T08."""
        return self.data_root / "snapshots" / db / "schema_snapshot.json"

    @property
    def plans_path(self) -> Path:
        """The default plan cache: our EXPLAIN plans for tasks the task file gives none."""
        return self.data_root / "plans.json"


class PgCredentials(BaseSettings):
    """The passwords of the admin, agent and grader roles. Build it with :func:`load_credentials`.

    Each comes from an explicit argument or else from the credentials file: never from an
    environment variable, so no child process can inherit one.
    """

    model_config = SettingsConfigDict(
        env_prefix=_ENV_PREFIX,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    admin_password: SecretStr = Field(min_length=1)
    agent_password: SecretStr = Field(min_length=1)
    grader_password: SecretStr = Field(min_length=1)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Only the file load_credentials passes in: no environment variables, no secrets
        # directory and no .env in the working directory.
        return init_settings, dotenv_settings


def default_credentials_file() -> Path:
    """Return ``~/.config/goaly/pg.env`` for the current ``HOME``."""
    return Path.home() / ".config" / "goaly" / "pg.env"


def ensure_credentials(path: Path | None = None, *, golden_pgdata: Path | None = None) -> None:
    """Give each role a random password in the credentials file, keeping those it already has.

    Creates the file, mode 0600, and its directory, mode 0700, if they are missing. A file that
    other users could read is refused rather than used. If the data directory ``golden_pgdata``
    already exists, it was built with an admin password, so a missing one is an error rather
    than a reason to make up a new one that the servers would refuse.
    """
    path = default_credentials_file() if path is None else path
    values = _read_values(path) if path.exists() else {}
    missing = [name for name in _PASSWORD_FIELDS if not values.get(_variable(name))]
    if not missing:
        return
    if "admin_password" in missing and golden_pgdata is not None and golden_pgdata.exists():
        raise PgSettingsError(
            f"{path} has lost the admin password that the data directory {golden_pgdata} was "
            "built with: restore the file, or delete the golden build and the instances to "
            "rebuild them."
        )
    for name in missing:
        values[_variable(name)] = secrets.token_urlsafe(_PASSWORD_BYTES)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_private(path, "".join(f"{key}={value}\n" for key, value in values.items()))


def write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, as a new file only its owner can read (0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)  # Make the rename itself durable.
    finally:
        os.close(directory)


def load_credentials(path: Path | None = None) -> PgCredentials:
    """Read the credentials from ``path`` (default: the pg.env file).

    Raise :class:`PgSettingsError` if a password is missing, or if the file is unsafe or cannot
    be read. The message names the variable but never echoes a value.
    """
    path = default_credentials_file() if path is None else path
    if path.exists():
        _check_private(path)
    try:
        return PgCredentials(_env_file=path)
    except ValidationError as error:
        details = error.errors(include_input=False)
    except (OSError, UnicodeDecodeError) as error:
        reason = error.strerror if isinstance(error, OSError) else "not valid UTF-8"
        raise PgSettingsError(f"Cannot read {path}: {reason}.") from None
    problems = [f"{_variable(str(detail['loc'][0]))} is missing or empty." for detail in details]
    hint = f" Run the environment setup, which writes {path}."
    raise PgSettingsError(" ".join(problems) + hint) from None


def _variable(field: str) -> str:
    return _ENV_PREFIX + field.upper()


def _check_private(path: Path) -> None:
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PgSettingsError(
            f"{path} is readable by other users (mode {stat.S_IMODE(mode):o}); run chmod 600 on it."
        )


def _read_values(path: Path) -> dict[str, str]:
    """Read the credentials file with python-dotenv, the parser pydantic-settings reads it with."""
    _check_private(path)
    return {key: value for key, value in dotenv_values(path, encoding="utf-8").items() if value}
