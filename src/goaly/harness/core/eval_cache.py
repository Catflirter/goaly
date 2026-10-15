"""The evaluation cache (spec §6.1): one task's finished evaluation, keyed by (harness hash, task,
seed).

A run finds a task's evaluation here and records it again, rather than solving and grading the
task anew; RSI's screening and resumed experiments rely on this (spec §8.2, §8.7). An evaluation
is the task's TaskResult and its Trajectory, which holds the prediction.

The key's task is the benchmark's name, the instance's ID and a digest of the agent's input, so
an instance whose input has changed, say after a dataset fix, gets no stale evaluation. How the
benchmark grades is the cache's namespace, a grading identity that every entry is kept under:
the wiring (T11) names it for the grading mode and a fingerprint of bench and its environment, so
a dev-mode run never gets an official-mode grade, nor a grade from other grading code. Runs on
SPLIT_EXTERNAL bypass the cache, since an organizer's instance IDs may collide with ours.

Each entry is one JSON file, at root/namespace/harness_hash/, which a write replaces atomically
(spec §6.2): readers never see part of an entry, and concurrent writers of one key leave one
whole entry. An entry that cannot be read or used is a miss, logged as a warning, and the next
write replaces it.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from google.protobuf import json_format

from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import TaskInput
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.contracts.jsonl import to_json_line
from goaly.harness.core.atomic import write_text_atomically

__all__ = [
    "CacheKey",
    "CachedEvaluation",
    "EvalCache",
    "evaluation_key",
    "put_evaluation",
    "task_digest",
]

_logger = logging.getLogger(__name__)

# Bumped when an entry's layout changes; an entry of another format is a miss.
_FORMAT = 1
# A namespace and a harness hash name directories, so each must be a plain name.
_PLAIN_NAME = re.compile(r"[0-9A-Za-z_-]+")


@dataclass(frozen=True)
class CacheKey:
    """What identifies one cached evaluation: (harness hash, task, seed)."""

    # The content hash of the harness: core plus the run's pack (RunManifest.harness_hash).
    harness_hash: str
    # The benchmark adapter's name, as RunManifest.benchmark gives it.
    benchmark: str
    instance_id: str
    # The replicate index, as TaskResult.seed.
    seed: int
    # task_digest() of the agent's input for the instance.
    task_digest: str


@dataclass(frozen=True)
class CachedEvaluation:
    """A task's evaluation: its result and its trajectory, of the run that made them."""

    result: TaskResult
    trajectory: Trajectory


def task_digest(task: TaskInput) -> str:
    """Return the SHA-256, in hex, of ``task`` as a JSON line: it changes with anything in it."""
    return hashlib.sha256(to_json_line(task).encode("utf-8")).hexdigest()


def evaluation_key(
    *, harness_hash: str, benchmark: str, instance_id: str, seed: int, task: TaskInput
) -> CacheKey:
    """Return the key of a task's evaluation: the harness hash, the task and the seed.

    ``task`` is the agent's input, which the benchmark builds from its dataset. In a two-phase
    run the host builds it, as it computes the harness hash: no part of a key comes from the
    solve container.
    """
    return CacheKey(harness_hash, benchmark, instance_id, seed, task_digest(task))


async def put_evaluation(cache: "EvalCache", key: CacheKey, evaluation: CachedEvaluation) -> None:
    """Put ``evaluation`` in ``cache`` under ``key``. A failure to write it is logged, never
    raised: the cache only saves work."""
    try:
        await cache.put(key, evaluation)
    except OSError as error:
        _logger.warning(
            "could not cache the evaluation of %s, seed %d: %s", key.instance_id, key.seed, error
        )


class EvalCache:
    """The evaluation cache in the directory ``root``, which writing creates if needed.

    ``namespace`` is the grading identity of the runs that share its entries, such as
    ``f"{mode}-{bench_fingerprint}"``; it must be a plain name, of letters, digits, ``_`` and
    ``-``. The files are read and written in a worker thread, off the event loop.
    """

    def __init__(self, root: Path, namespace: str) -> None:
        _check_plain_name("namespace", namespace)
        self.root = root
        self.namespace = namespace

    def path(self, key: CacheKey) -> Path:
        """Return the file that holds ``key``'s entry, whether or not there is one."""
        _check_plain_name("harness hash", key.harness_hash)
        task = json.dumps(
            [key.benchmark, key.instance_id, key.seed, key.task_digest], ensure_ascii=False
        )
        name = hashlib.sha256(task.encode("utf-8")).hexdigest()
        return self.root / self.namespace / key.harness_hash / f"{name}.json"

    async def get(self, key: CacheKey) -> CachedEvaluation | None:
        """Return the evaluation cached for ``key``, or None if there is none it can use."""
        return await asyncio.to_thread(self._read, self.path(key), key)

    async def put(self, key: CacheKey, evaluation: CachedEvaluation) -> None:
        """Cache ``evaluation`` for ``key``, replacing any entry it has.

        Raise ValueError if the evaluation's result or trajectory is of another task or seed, or
        its trajectory's task does not have the key's digest. The evaluation is copied first, so
        the caller may change it while this waits.
        """
        path = self.path(key)
        problem = _other_task(key, evaluation.result, evaluation.trajectory)
        if problem is not None:
            raise ValueError(f"the evaluation does not match its key: {problem}")
        result, trajectory = TaskResult(), Trajectory()
        result.CopyFrom(evaluation.result)
        trajectory.CopyFrom(evaluation.trajectory)
        await asyncio.to_thread(self._write, path, key, CachedEvaluation(result, trajectory))

    @staticmethod
    def _write(path: Path, key: CacheKey, evaluation: CachedEvaluation) -> None:
        # Each part is JSON already, so joining them makes one JSON object, on one line.
        text = (
            f'{{"format":{_FORMAT},"key":{_key_json(key)},'
            f'"result":{to_json_line(evaluation.result)},'
            f'"trajectory":{to_json_line(evaluation.trajectory)}}}\n'
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomically(path, text)

    @staticmethod
    def _read(path: Path, key: CacheKey) -> CachedEvaluation | None:
        try:
            entry: object = json.loads(path.read_bytes().decode("utf-8"))
            return _evaluation(entry, key)
        except FileNotFoundError:
            return None
        # Anything an unusable entry raises, a RecursionError for deeply nested JSON included.
        except Exception as error:  # noqa: BLE001 - A miss, logged here.
            _logger.warning("ignored the cache entry %s: %s: %s", path, type(error).__name__, error)
            return None


def _key_json(key: CacheKey) -> str:
    return json.dumps(dataclasses.asdict(key), ensure_ascii=False, separators=(",", ":"))


def _check_plain_name(what: str, name: str) -> None:
    if not _PLAIN_NAME.fullmatch(name):
        raise ValueError(f"the {what} must be a plain name, which names a directory: {name!r}")


def _other_task(key: CacheKey, result: TaskResult, trajectory: Trajectory) -> str | None:
    """Say how an evaluation is not of ``key``'s task and seed, or return None if it is."""
    for message in (result, trajectory):
        if (message.instance_id, message.seed) != (key.instance_id, key.seed):
            return (
                f"its {type(message).__name__} is of {message.instance_id!r}, seed"
                f" {message.seed}, not {key.instance_id!r}, seed {key.seed}"
            )
    if task_digest(trajectory.task) != key.task_digest:
        return "its trajectory's task has another digest than the key's"
    return None


def _evaluation(entry: object, key: CacheKey) -> CachedEvaluation:
    """Check that ``entry`` is a whole entry for ``key``, and return its evaluation.

    Raise ValueError, or protobuf's json_format.ParseError, if it is not.
    """
    if not isinstance(entry, dict) or entry.get("format") != _FORMAT:
        raise ValueError(f"not an entry of format {_FORMAT}")
    if entry.get("key") != dataclasses.asdict(key):
        raise ValueError(f"an entry for another key: {entry.get('key')!r}")
    result = _message(entry, "result", TaskResult())
    trajectory = _message(entry, "trajectory", Trajectory())
    problem = _other_task(key, result, trajectory)
    if problem is not None:
        raise ValueError(problem)
    return CachedEvaluation(result, trajectory)


def _message[M: (TaskResult, Trajectory)](entry: dict[str, object], field: str, message: M) -> M:
    value = entry.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"its {field} is not a JSON object")
    return json_format.ParseDict(value, message, ignore_unknown_fields=True)
