"""The sandboxes goaly.rsi runs candidate code in (spec §8.5, §8.8): the implementer's and the
unit-test guard's.

Both mount one worktree, a candidate's or a snapshot of one, read-only at its own path; the
implementer's also mounts the lane's two pack directories read-write over it. Nothing else from
the host is mounted. No git metadata is mounted, so nothing inside can commit. The hardening,
the one-shot runs, the cleanup and the image are goaly.bench.sandbox's, which the solve phase of
goaly run shares (spec §6.1).
"""

import os
import secrets
from pathlib import Path

from goaly.bench.sandbox import (
    DOCKERFILE,
    IMAGE_REPOSITORY,
    SandboxConfig,
    SandboxError,
    SandboxResult,
    bind,
    ensure_image,
    hardened_options,
    image_tag,
    mount_path,
    run_container,
    timed_shell,
)
from goaly.bench.sandbox import remove_containers as remove_labelled
from goaly.contracts.gen.experiment_pb2 import Lane
from goaly.rsi.ids import check_id
from goaly.rsi.lanes import lane_info

__all__ = [
    "DOCKERFILE",
    "IMAGE_REPOSITORY",
    "LABEL_EXPERIMENT",
    "LABEL_LANE",
    "LABEL_PURPOSE",
    "LABEL_SANDBOX",
    "SandboxConfig",
    "SandboxError",
    "SandboxResult",
    "ensure_image",
    "image_tag",
    "remove_containers",
    "run_once",
    "run_options",
    "timed_shell",
]

LABEL_SANDBOX = "goaly.rsi.sandbox"  # "1" on every sandbox container.
LABEL_LANE = "goaly.rsi.lane"
LABEL_EXPERIMENT = "goaly.rsi.experiment"
LABEL_PURPOSE = "goaly.rsi.purpose"  # Such as "implementer" or "unit-tests".


def run_options(
    worktree: Path,
    lane: Lane.ValueType,
    *,
    experiment_id: str,
    purpose: str,
    writable_packs: bool,
    uid: int,
    gid: int,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008  (frozen, so one default is safe)
) -> list[str]:
    """Return the ``docker run`` options for a sandbox on ``worktree``, which ``lane`` works in.

    The worktree is mounted read-only at its own path; with ``writable_packs`` the lane's two pack
    directories are mounted read-write over it, and they must exist. The image and the command
    follow these options. Raises ValueError for a worktree path docker could misread, or an
    invalid label.
    """
    check_id(experiment_id, "experiment_id")
    check_id(purpose, "purpose")
    info = lane_info(lane)
    mount_path(worktree, "worktree")
    labels = {
        LABEL_SANDBOX: "1",
        LABEL_LANE: info.name,
        LABEL_EXPERIMENT: experiment_id,
        LABEL_PURPOSE: purpose,
    }
    options = [
        *hardened_options(uid=uid, gid=gid, workdir=worktree, labels=labels, config=config),
        bind(worktree, what="worktree"),
    ]
    if writable_packs:
        options += [bind(worktree / pack, writable=True) for pack in info.pack_dirs]
    return options


def run_once(
    image: str,
    worktree: Path,
    lane: Lane.ValueType,
    *,
    experiment_id: str,
    purpose: str,
    command: str,
    timeout_s: float,
    writable_packs: bool = False,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008
) -> SandboxResult:
    """Run the bash ``command`` in a fresh sandbox on ``worktree``, as the host user, and return
    how it ended. The container is removed before this returns, whatever happens.

    Raises SandboxError if docker fails or hangs. A command that fails or times out is a result,
    not an error.
    """
    options = run_options(
        worktree,
        lane,
        experiment_id=experiment_id,
        purpose=purpose,
        writable_packs=writable_packs,
        uid=os.getuid(),
        gid=os.getgid(),
        config=config,
    )
    name = f"goaly-rsi-{lane_info(lane).name}-{experiment_id}-{purpose}-{secrets.token_hex(4)}"
    return run_container(image, options, command, name=name, timeout_s=timeout_s, config=config)


def remove_containers(
    *,
    lane: Lane.ValueType | None = None,
    experiment_id: str | None = None,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008
) -> int:
    """Remove every sandbox container, or a lane's or an experiment's, running or not, and
    return how many there were. Containers without the sandbox label are never touched.
    """
    labels = {LABEL_SANDBOX: "1"}
    if lane is not None:
        labels[LABEL_LANE] = lane_info(lane).name
    if experiment_id is not None:
        check_id(experiment_id, "experiment_id")
        labels[LABEL_EXPERIMENT] = experiment_id
    return remove_labelled(labels, config=config)
