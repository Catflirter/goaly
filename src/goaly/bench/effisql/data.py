"""Effi-SQL's task data (spec §4.1, §8.5): the pinned dataset, its tasks and their gold-free copy.

- fetch_dataset downloads the pinned dataset into a local cache, which is never committed, and
  checks its sha256 before anything reads it. The run manifest records DATASET_REVISION and
  DATASET_SHA256.
- EffiSqlInstance is one task, validated from a line of a task file. A task file is the dataset,
  its gold-free runtime copy, or an organizers' ``--instances`` file, all in the dataset's format.
- load_instances reads a task file.
- write_runtime_file writes the dataset's gold-free runtime copy, which candidate evaluations
  load. Only the setup and exclusion tools, and the analyst for train, read the dataset itself.
"""

import hashlib
import http.client
import json
import secrets
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pydantic import JsonValue as PydanticJsonValue
from pydantic_core import ErrorDetails

from goaly.contracts.interfaces import JsonValue

__all__ = [
    "DATASET_FILE",
    "DATASET_REPO",
    "DATASET_REVISION",
    "DATASET_SHA256",
    "DATASET_URL",
    "GOLD_FIELDS",
    "RUNTIME_FIELDS",
    "RUNTIME_FILE",
    "ChecksumMismatchError",
    "DatasetError",
    "Difficulty",
    "EffiSqlInstance",
    "NonBlankStr",
    "TaskFileError",
    "default_cache_dir",
    "fetch_dataset",
    "load_instances",
    "write_runtime_file",
]

# The dataset (spec §4.1): one file of the Hugging Face dataset, pinned to a revision and a sha256.
DATASET_REPO = "birdsql/Effi-SQL"
DATASET_REVISION = "86cbcc9efbd31db32a46897d47cbf865b1cba3a4"
DATASET_FILE = "effi-sql-pg.jsonl"
DATASET_SHA256 = "2af3bdc9fe6c1197f23e31089660e7ce10fca6956b32e5970a3113f0c87e9d97"
DATASET_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{DATASET_FILE}"
)

# The gold answer's fields, which the agent never sees (spec §4.1).
GOLD_FIELDS = ("optimized_sql", "fast_time", "optimized_explain_analyze")

# The gold-free runtime copy of the dataset (spec §8.5), which the script writes next to it in the
# cache. Its tasks keep only the fields the agent may see plus the grading fields.
RUNTIME_FILE = "effi-sql-pg.runtime.jsonl"
RUNTIME_FIELDS = ("id", "db", "base_sql", "base_time", "base_explain_analyze", "difficulty")

_URL_SCHEMES = ("https", "file")  # file serves a local copy, and the tests.
_DOWNLOAD_TIMEOUT_S = 60.0  # For connecting and for each read, not for the whole download.
_CHUNK_BYTES = 1 << 20

Difficulty = Literal["easy", "medium", "hard"]


class DatasetError(Exception):
    """Effi-SQL's data cannot be fetched or read; the message says why."""


class ChecksumMismatchError(DatasetError):
    """A file's sha256 is not the pinned one, so the file is not used."""


class TaskFileError(DatasetError):
    """A line of a task file holds no valid task. The message starts ``path:line:``.

    It never quotes the line, which may hold a gold answer, and chains no exception that would.
    """


def default_cache_dir() -> Path:
    """Return ``~/.cache/goaly/effisql`` for the current ``HOME``: the default dataset cache."""
    return Path.home() / ".cache" / "goaly" / "effisql"


def fetch_dataset(cache_dir: Path, *, url: str = DATASET_URL, sha256: str = DATASET_SHA256) -> Path:
    """Return the path of the dataset in ``cache_dir``, downloading it first if it is not there.

    The cached copy is checked against ``sha256`` on every call, so a damaged or changed copy is
    never used: it raises ChecksumMismatchError and stays where it is for a person to look at. A
    download goes to a temporary file in ``cache_dir``, which becomes the cached copy only once its
    sha256 matches; otherwise it is deleted and ChecksumMismatchError is raised. ``url`` is an
    https URL, or a file URL for a local copy. A download that fails raises DatasetError.
    """
    if urlsplit(url).scheme not in _URL_SCHEMES:
        raise DatasetError(f"cannot download {url}: only https and file URLs are allowed")
    path = cache_dir / DATASET_FILE
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        _download(url, path, sha256)
        return path
    with path.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != sha256:
        raise ChecksumMismatchError(
            f"{path} has sha256 {actual}, not the pinned {sha256};"
            " delete it to download the pinned file again"
        )
    return path


def _download(url: str, path: Path, sha256: str) -> None:
    """Download ``url`` to ``path``, provided that its sha256 is ``sha256``."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    digest = hashlib.sha256()
    try:
        try:
            # fetch_dataset allows only the https and file schemes.
            with (
                urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response,  # noqa: S310
                temporary.open("xb") as file,
            ):
                while chunk := response.read(_CHUNK_BYTES):
                    digest.update(chunk)
                    file.write(chunk)
        except (OSError, http.client.HTTPException) as error:  # URLError is an OSError.
            raise DatasetError(f"cannot download {url}: {error}") from error
        if digest.hexdigest() != sha256:
            raise ChecksumMismatchError(
                f"{url} has sha256 {digest.hexdigest()}, not the pinned {sha256};"
                " the download was discarded"
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)  # Already gone once it has replaced path.


def _task_id(value: object) -> str:
    """Return a task's ``id``, an integer or a string in the file, as its instance ID."""
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError("must be an integer or a string")  # pydantic reports a ValueError.
    return _not_blank(str(value))


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _no_empty_plan(value: object) -> object:
    """Read an empty plan, ``{}``, ``[]`` or a blank string, as no plan: None."""
    if isinstance(value, dict | list) and not value:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


_TaskId = Annotated[str, BeforeValidator(_task_id)]
# A string that is not empty or only whitespace.
NonBlankStr = Annotated[str, AfterValidator(_not_blank)]
_Millis = Annotated[float, Field(ge=0, allow_inf_nan=False)]  # A time in milliseconds.
# A query plan's root node, as a JSON object (spec §4.1), or None for no plan.
_Plan = Annotated[dict[str, JsonValue] | None, BeforeValidator(_no_empty_plan)]

# Checks that each field of a record is JSON; pydantic's own JsonValue reports one error per field.
_RECORD: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, PydanticJsonValue])


class EffiSqlInstance(BaseModel):
    """One Effi-SQL task (spec §4.1), as its adapter loads it; it satisfies the Instance Protocol.

    Validate a task record from a task file with ``EffiSqlInstance.model_validate(record)``:

    - The record's ``id``, an integer in the dataset, becomes ``instance_id``, a string. A string
      ``id`` is kept as it is; a bool is rejected.
    - A field the model does not know is ignored, as an organizers' file may carry more.
    - Every field an organizers' file may lack is optional, and missing or null gives None: the
      gold answer, ``difficulty``, ``base_time``, and the plan ``base_explain_analyze``. The
      gold answer is never checked beyond its type.
    - An empty plan, ``{}``, ``[]`` or a blank string, counts as no plan and gives None too, so
      agent_input falls back to the plan cache. ``record`` keeps the value as loaded.

    The repr leaves out the gold answer and the plans. Times are in milliseconds, as measured on
    the dataset authors' hardware.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)

    instance_id: _TaskId = Field(validation_alias="id")
    db: NonBlankStr
    base_sql: NonBlankStr
    base_time: _Millis | None = None
    base_explain_analyze: _Plan = Field(default=None, repr=False)
    difficulty: Difficulty | None = None
    optimized_sql: str | None = Field(default=None, repr=False)
    fast_time: _Millis | None = Field(default=None, repr=False)
    optimized_explain_analyze: _Plan = Field(default=None, repr=False)

    _record: dict[str, JsonValue] = PrivateAttr()

    @model_validator(mode="wrap")
    @classmethod
    def _keep_record(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        instance = handler(data)
        if isinstance(data, Mapping):  # An instance validated again keeps the record it has.
            non_gold = {name: value for name, value in data.items() if name not in GOLD_FIELDS}
            instance._record = _RECORD.validate_python(non_gold)  # A copy of the input's objects.
        return instance

    @property
    def record(self) -> Mapping[str, JsonValue]:
        """The task record's non-gold fields as loaded, in file order, unknown fields included.

        The official prediction record is this record plus a ``prediction`` field (spec §3.3),
        which ``to_official`` builds as ``{**inst.record, "prediction": ...}``. The mapping is
        read-only; its values belong to the instance, so never change them.
        """
        return MappingProxyType(self._record)


def load_instances(path: Path) -> tuple[EffiSqlInstance, ...]:
    """Read the tasks in the task file at ``path``, in file order.

    The file holds one task record per line, as a JSON object in the dataset's format. Blank lines
    are skipped, and only a line feed ends a line. Raises TaskFileError for a line that holds no
    valid task and for an ID that repeats, and OSError if the file cannot be read.
    """
    instances: list[EffiSqlInstance] = []
    lines: dict[str, int] = {}  # The line each instance ID is on.
    with path.open("rb") as file:  # A binary file splits at line feeds only.
        for line_number, raw in enumerate(file, start=1):
            if not raw.strip():
                continue
            task = _parse_task(raw)
            if isinstance(task, str):
                raise TaskFileError(f"{path}:{line_number}: {task}")
            first = lines.setdefault(task.instance_id, line_number)
            if first != line_number:
                raise TaskFileError(
                    f"{path}:{line_number}: the id {task.instance_id} repeats line {first}"
                )
            instances.append(task)
    return tuple(instances)


class _NotJsonError(ValueError):
    """A NaN or an infinity, which Python's json reads but JSON does not have."""


def _reject_constant(name: str) -> None:
    raise _NotJsonError(f"{name} is not a JSON value")


def _parse_task(raw: bytes) -> EffiSqlInstance | str:
    """Validate ``raw``, one line of a task file, as a task, or say why it is not one.

    The reason is returned rather than raised, so the caller's error chains none of these
    exceptions: their text and inputs quote the line.
    """
    try:
        record = json.loads(raw, parse_constant=_reject_constant)  # Bytes: a BOM is skipped.
    except UnicodeDecodeError as error:
        return f"not valid UTF-8 at byte {error.start}"
    except json.JSONDecodeError as error:
        return f"not valid JSON: {error.msg} at column {error.colno}"
    except _NotJsonError as error:
        return f"not valid JSON: {error}"
    if not isinstance(record, dict):
        return "not a JSON object"
    try:
        return EffiSqlInstance.model_validate(record)
    except ValidationError as error:
        return "; ".join(_describe(detail) for detail in error.errors(include_input=False))


def _describe(detail: ErrorDetails) -> str:
    location = ".".join(str(part) for part in detail["loc"])
    return f"{location}: {detail['msg']}" if location else detail["msg"]


def write_runtime_file(dataset: Path, target: Path) -> None:
    """Write the gold-free runtime copy of the task file ``dataset`` to ``target`` (spec §8.5).

    Each task keeps only RUNTIME_FIELDS that it has, in its file's order, so the copy holds no gold
    answer and no field this module does not know. ``dataset`` is read with load_instances first,
    so a file it rejects writes nothing. The copy replaces ``target`` atomically.
    """
    lines = [
        json.dumps(
            {name: value for name, value in task.record.items() if name in RUNTIME_FIELDS},
            ensure_ascii=False,
            allow_nan=False,
        )
        for task in load_instances(dataset)
    ]
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            file.writelines(f"{line}\n" for line in lines)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)  # Already gone once it has replaced target.
