"""The Effi-SQL solve container's inputs (spec §6.1, ruling V27): ``<run>/inputs/``.

The solve container is untrusted, so it writes only ``<run>/solve/``, its one writable mount, and
core keeps the run's records, manifest, cache and call log host-side. Effi-SQL's launcher stages
one more directory on the host, ``<run>/inputs/`` (SolveRequest.inputs_dir), which the container
mounts read-only and its harness factory reads (HarnessOptions.inputs_dir): the gold-free tasks,
the schema snapshot and the plans the agents see. It holds no gold and no golden data.
"""

from dataclasses import dataclass
from pathlib import Path

__all__ = ["SolveInputs"]


@dataclass(frozen=True)
class SolveInputs:
    """The files of the inputs directory ``root``."""

    root: Path

    @property
    def tasks(self) -> Path:
        """The gold-free task file the container loads its tasks from."""
        return self.root / "tasks.jsonl"

    @property
    def snapshot(self) -> Path:
        return self.root / "schema_snapshot.json"

    @property
    def plans(self) -> Path:
        return self.root / "plans.json"
