"""The isolated solve phase (spec §6.1, V27): what the host asks a benchmark's launcher to run.

Candidate pack code runs only in the solve container, which sees what it needs and no more: the
checkout that holds the pack and the inputs its benchmark staged, both read-only, and the run's
solve directory, its only writable mount. A benchmark's launcher (HostBenchmark.launcher)
starts the LLM proxy, which holds the real key and writes its call records to a host-only file,
and the solve container, for one SolveRequest. In the container it runs
``goaly <request.goaly_args(...)>``: `goaly run --phase solve --in-container`, which solves the
request's tasks and writes the solve directory.

Nothing the container writes decides a score. The host grades every task from the predictions
in the solve directory, which it reads as untrusted input, and keeps every other record of the
run where the container cannot write it: the graded results and trajectories, the official
files, the manifest, the evaluation cache and the proxy's call log.
"""

import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from google.protobuf import json_format
from google.protobuf.message import Message

from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import (
    SPLIT_EXTERNAL,
    SPLIT_HELDOUT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    Prediction,
    Split,
)
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.contracts.jsonl import from_json_line, to_json_line
from goaly.harness.core.atomic import write_text_atomically

__all__ = [
    "MAX_PREDICTION_BYTES",
    "MAX_RESULT_BYTES",
    "MAX_TRAJECTORY_BYTES",
    "SPLIT_NAMES",
    "TASKS_FILE",
    "SolveLauncher",
    "SolveRecord",
    "SolveRequest",
    "SolveTasks",
    "read_solve_records",
    "read_solve_tasks",
    "write_predictions",
    "write_solve_tasks",
]

_logger = logging.getLogger(__name__)

# Our splits, by the name `goaly run --split` takes.
SPLIT_NAMES: Mapping[Split.ValueType, str] = {
    SPLIT_TRAIN: "train",
    SPLIT_VAL: "val",
    SPLIT_HELDOUT: "heldout",
}

# The tasks file, in the solve directory: what the container solves.
TASKS_FILE = "tasks.json"

# The longest line the host reads of each file the container wrote; a longer one is malformed.
MAX_PREDICTION_BYTES = 4 * 2**20
MAX_RESULT_BYTES = 2**20
MAX_TRAJECTORY_BYTES = 64 * 2**20

type _Task = tuple[str, int]


@dataclass(frozen=True)
class SolveRequest:
    """One isolated solve phase, which the host asks a benchmark's launcher to run.

    The launcher starts the LLM proxy, which writes its call records to ``llm_calls``, then the
    solve container. The container mounts the checkout that holds ``pack_dir``, read-only, and
    ``inputs_dir``, read-only, and ``solve_dir``, its only writable mount. In it the launcher
    runs ``goaly`` with the arguments goaly_args() gives. The launcher returns once the container
    has exited with status 0, and raises otherwise; cancelled, it stops the container and the
    proxy before it returns.
    """

    # The run, which the host's manifest describes.
    run_id: str
    benchmark: str
    # The registered harness the container runs, by name.
    harness: str
    # The checkout the container runs, read-only: --harness-checkout's, or the host's own.
    checkout: Path
    # The directory of that harness's pack, inside ``checkout``, which the host hashed with the
    # checkout's core. None for a harness without one, such as a test's.
    pack_dir: Path | None
    # The split the tasks come from; SPLIT_EXTERNAL for instances_file.
    split: Split.ValueType
    # The organizers' task file (--instances) on the host, to mount read-only; or None.
    instances_file: Path | None
    # <run>/solve: the container's only writable mount. The host writes the tasks file in it
    # before it calls the launcher, and then reads the container's records from it.
    solve_dir: Path
    # <run>/inputs: the launcher stages the solve phase's read-only inputs here, such as a
    # gold-free task file, and mounts it read-only. Core never creates, reads or cleans it.
    inputs_dir: Path
    # <run>/llm_calls.jsonl, which only the host writes: the proxy's LlmCallRecords go here.
    llm_calls: Path
    # The (instance ID, seed) pairs to solve, in order; never empty. Tasks taken from the
    # evaluation cache, or already graded, are not among them.
    tasks: tuple[_Task, ...]
    # This run's share of each model's concurrency budget, for the proxy and the container.
    llm_shares: Mapping[str, int]
    # How many tasks solve at once, and the time limit of one task (None: no limit).
    max_tasks: int
    task_timeout_s: float | None
    # Whether the solve directory holds an earlier solve phase of this run, which the container
    # continues; and whether that phase's errors are solved again (--retry-errors).
    resume: bool
    retry_errors: bool
    # The container's log level, as `goaly run --log-level` takes it.
    log_level: str

    def __post_init__(self) -> None:
        if (self.split == SPLIT_EXTERNAL) != (self.instances_file is not None):
            raise ValueError("a solve phase has a task file exactly when its split is external")
        if not self.tasks:
            raise ValueError(f"the solve phase of run {self.run_id} has no task to solve")
        if self.max_tasks < 1:
            raise ValueError(f"max_tasks must be at least 1: {self.max_tasks}")

    def goaly_args(
        self, *, solve_dir: Path, inputs_dir: Path, instances_file: Path | None = None
    ) -> list[str]:
        """The arguments of the `goaly` command the container runs, after the program's name.

        The paths are where the container sees the solve directory, the inputs directory and,
        for an organizers' run, the task file. Raise ValueError if ``instances_file`` is given
        for a run that has none, or missing for one that has one.
        """
        if (instances_file is None) != (self.instances_file is None):
            needs = "needs" if self.instances_file is not None else "has no use for"
            raise ValueError(f"the solve phase of run {self.run_id} {needs} a task file")
        args = [
            "run",
            "--bench",
            self.benchmark,
            "--harness",
            self.harness,
            "--phase",
            "solve",
            "--in-container",
            "--out",
            str(solve_dir),
            "--tasks",
            str(solve_dir / TASKS_FILE),
            "--inputs",
            str(inputs_dir),
        ]
        if instances_file is not None:
            args += ["--instances", str(instances_file)]
        else:
            args += ["--split", SPLIT_NAMES[self.split]]
        args += ["--run-id", self.run_id, "--max-tasks", str(self.max_tasks)]
        if self.task_timeout_s is not None:
            args += ["--task-timeout", repr(self.task_timeout_s)]
        for model, share in sorted(self.llm_shares.items()):
            args += ["--llm-share", f"{model}={share}"]
        if self.resume:
            args.append("--resume")
        if self.retry_errors:
            args.append("--retry-errors")
        args += ["--log-level", self.log_level]
        return args


# Runs one isolated solve phase; see SolveRequest.
type SolveLauncher = Callable[[SolveRequest], Awaitable[None]]


@dataclass(frozen=True)
class SolveTasks:
    """What a solve phase in the container solves, as the host's tasks file says."""

    # The run's instances and seeds, all of them, as its manifest has them.
    instance_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    # The (instance ID, seed) pairs to solve now: some of the run's, never one twice.
    tasks: tuple[_Task, ...]


def write_solve_tasks(path: Path, solve: SolveTasks) -> None:
    """Write ``solve`` to the tasks file at ``path``, replacing it atomically. It blocks."""
    fields = {
        "instance_ids": list(solve.instance_ids),
        "seeds": list(solve.seeds),
        "tasks": [[instance_id, seed] for instance_id, seed in solve.tasks],
    }
    write_text_atomically(path, json.dumps(fields, ensure_ascii=False) + "\n")


def read_solve_tasks(path: Path) -> SolveTasks:
    """Read the tasks file at ``path``. Raise ValueError, naming the file, if it is not one."""
    try:
        fields = json.loads(path.read_text(encoding="utf-8"))
        solve = SolveTasks(
            instance_ids=_strings(fields["instance_ids"]),
            seeds=_seeds(fields["seeds"]),
            tasks=tuple(_task(task) for task in fields["tasks"]),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is not a tasks file: {error}") from error
    run = {(instance_id, seed) for instance_id in solve.instance_ids for seed in solve.seeds}
    if not set(solve.tasks) <= run or len(set(solve.tasks)) != len(solve.tasks):
        raise ValueError(f"{path} lists tasks the run does not have, or one twice")
    return solve


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise TypeError("expected a list of IDs")
    return tuple(value)


def _seeds(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(_is_seed(item) for item in value):
        raise TypeError("expected a list of seeds")
    return tuple(value)


def _task(value: object) -> _Task:
    if not (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)):
        raise TypeError("expected an [instance ID, seed] pair")
    instance_id, seed = value
    if not instance_id or not _is_seed(seed):
        raise TypeError("expected an [instance ID, seed] pair")
    return instance_id, seed


def _is_seed(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


# The predictions files, which the solve phase writes and the host grades.


def write_predictions(path: Path, predictions: Sequence[Prediction]) -> None:
    """Write one seed's predictions, a JSON line each, replacing the file at ``path`` atomically.

    The solve phase writes each seed's file once its tasks are recorded, with one prediction per
    recorded task, its last. It blocks: run it in a worker thread from the event loop.
    """
    write_text_atomically(
        path, "".join(f"{to_json_line(prediction)}\n" for prediction in predictions)
    )


@dataclass(frozen=True)
class SolveRecord:
    """What the solve container recorded of one task, as the host reads it: untrusted input.

    Only ``prediction`` decides the task's score, once the host has graded it. ``result`` and
    ``trajectory``, the container's own, say what the container claims happened: the host keeps
    them for people and for the analyst, never for a decision.
    """

    # The task's one valid prediction; None if it has none, as ``problem`` says.
    prediction: Prediction | None
    # Why there is no valid prediction, for the task's error message; empty if there is one.
    problem: str
    # The container's ungraded result and trajectory of the task, its last; None if it wrote
    # none that could be read.
    result: TaskResult | None
    trajectory: Trajectory | None


type _Key = tuple[str, int]


def read_solve_records(
    files: Callable[[int], Path],
    results: Path,
    trajectories: Path,
    keys: Collection[_Key],
) -> dict[_Key, SolveRecord]:
    """Read the solve directory's record of each task of ``keys``, whatever the container wrote.

    ``files(seed)`` is the predictions file of a seed. A task's prediction is valid when that
    file holds exactly one line for its instance, and the line is a Prediction; a task with none,
    with two or more, or with one that does not parse, has a ``problem`` instead. A line longer
    than the file's bound, or that names no instance, counts for no task. The results and the
    trajectories give each task's last line that parses. It reads files, so it blocks.
    """
    wanted = set(keys)
    valid: dict[_Key, Prediction] = {}
    lines: Counter[_Key] = Counter()
    malformed: set[_Key] = set()
    for seed in sorted({seed for _, seed in wanted}):
        path = files(seed)
        if not path.is_file():
            continue
        for instance_id, prediction in _instance_lines(path, MAX_PREDICTION_BYTES, Prediction):
            key = (instance_id, seed)
            if key not in wanted:
                continue
            lines[key] += 1
            if prediction is None:
                malformed.add(key)
            else:
                valid[key] = prediction
    last_results = _last_lines(results, MAX_RESULT_BYTES, TaskResult, wanted)
    last_trajectories = _last_lines(trajectories, MAX_TRAJECTORY_BYTES, Trajectory, wanted)
    records: dict[_Key, SolveRecord] = {}
    for key in wanted:
        problem = ""
        if lines[key] == 0:
            problem = "the solve phase left no prediction for this task"
        elif lines[key] > 1:
            problem = f"the solve phase left {lines[key]} predictions for this task, not one"
        elif key in malformed:
            problem = "the solve phase left a prediction for this task that does not parse"
        prediction = None if problem else valid[key]
        if prediction is not None:
            prediction.instance_id = key[0]
        records[key] = SolveRecord(
            prediction, problem, last_results.get(key), last_trajectories.get(key)
        )
    return records


def _last_lines[M: (TaskResult, Trajectory)](
    path: Path, max_bytes: int, kind: type[M], wanted: set[_Key]
) -> dict[_Key, M]:
    """Each wanted task's last line in ``path`` that parses as a ``kind``, by its seed field."""
    if not path.is_file():
        return {}
    last: dict[_Key, M] = {}
    for _, record in _instance_lines(path, max_bytes, kind):
        if record is not None and (record.instance_id, record.seed) in wanted:
            last[(record.instance_id, record.seed)] = record
    return last


def _instance_lines[M: Message](
    path: Path, max_bytes: int, kind: type[M]
) -> Iterator[tuple[str, M | None]]:
    """Each line of ``path`` that names an instance: its instance ID, and the line as a ``kind``,
    or None if it does not parse. A line longer than ``max_bytes`` is skipped whole."""
    for line in _bounded_lines(path, max_bytes):
        if line is None or not line.strip():
            continue
        try:
            text = line.decode("utf-8")
            fields = json.loads(text)
        except (UnicodeDecodeError, ValueError, RecursionError):
            continue
        if not isinstance(fields, dict) or not isinstance(fields.get("instance_id"), str):
            continue
        try:
            message: M | None = from_json_line(text, kind)
        except (json_format.ParseError, ValueError, TypeError, RecursionError):
            message = None
        yield fields["instance_id"], message


def _bounded_lines(path: Path, max_bytes: int) -> Iterator[bytes | None]:
    """The lines of ``path``, each with its line feed; None for one longer than ``max_bytes``,
    which is read no further than that, so a huge line never sits in memory whole."""
    with path.open("rb") as file:
        while line := file.readline(max_bytes + 1):
            if len(line) <= max_bytes:
                yield line
                continue
            while line and not line.endswith(b"\n"):  # The rest of the overlong line.
                line = file.readline(max_bytes + 1)
            _logger.warning(
                "%s holds a line longer than %d bytes, which is ignored", path, max_bytes
            )
            yield None
