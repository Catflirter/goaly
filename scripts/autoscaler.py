#!/usr/bin/env python3
"""Operator-side autoscaler loop (stretch S4; design spec sections 10.2, 10.4, 10.5; Rulings SR6, SR7, SR8).

    scripts/autoscaler.py [--interval 20] [--cpu-max 3] [--once] [--dry-run]
                          [--pods-file F --nodes-file F] [--log DIR]

It runs on the operator machine, never in the cluster: the Lambda API key stays in .env, which only scripts/gpuctl
reads (spec section 10.2); this process neither loads nor prints it. Every --interval seconds one tick:

  1. read `kubectl get pods -A -o json` and `kubectl get nodes -o json` (kc's kubeconfig, ~/.kube/hybrid-gpu.yaml;
     --pods-file/--nodes-file read files instead), the CPU inventory infra/workers/workers.auto.tfvars.json
     (cpu_count), lambda/gpu-nodes.yaml and the override file lambda/.autoscale-state.json (gpu_counts: gpuctl's
     own targets, the override clamped to autoscale.min..max, else count), and `scripts/gpuctl list-nodes`;
  2. gpu_launching (Review Focus #4): a Lambda instance that is booting, or live without a registered Node; decide()
     adds NotReady, unlabeled and joining k3s-gpu-* nodes. GPU nodes that are NotReady or unlabeled, instances that
     launch, a clock skew against a node's Ready time and every reason GPU scaling is off are logged every tick,
     Pending Pods or not (a node NotReady for ever holds GPU scale-up for ever: the log says so each time);
  3. decide() (lambda/gpuctl/autoscale.py) returns at most one action;
  4. run it:
       cpu_scale  -> AUTO_APPROVE=1 scripts/cpu-pool.sh scale <target> --guarded
       cpu_remove -> AUTO_APPROVE=1 scripts/cpu-pool.sh remove <node> --guarded
       gpu_scale  -> `scripts/gpuctl plan --set-count <pool>=<target>` first (budget_usd_per_hour and max_nodes at
                     live prices, spec section 10.5; a refusal skips it, is remembered for COOLDOWN_S so the Lambda
                     API is not asked again every tick, and the CPU rules still get this tick), then, holding gpuctl's
                     write lock (.run/gpuctl.lock, gpuctl.lock), a re-check of the override file and of the live
                     instances, {pool: target} merged into lambda/.autoscale-state.json and `scripts/gpuctl apply
                     --auto-approve`, which inherits the lock (GPUCTL_LOCK_FD). A scale-up that fails or is
                     interrupted puts the override file back as it was (gpuctl terminates the new node it could not
                     join); a failed scale-down keeps the lower target (its terminate may have gone through, and a
                     lower target never relaunches anything) and, while more instances than that target are live,
                     runs gpuctl apply again after each cooldown (a GPU scale-down failure each time) and says so
                     every tick. Neither touches a file someone else changed meanwhile;
  5. save .run/autoscaler-state.json.

The autoscaler writes only lambda/.autoscale-state.json and its own state file, both atomically (temp file and
rename). The state is saved with last_action_at = now before an action starts, so a restart during or right after
it waits out the cooldown (Review Focus #1), and again when it ends: the cooldown counts from the end of the action;
a state file that cannot be written starts no action. Results: exit 0 is ok; cpu-pool.sh exit 2 (another guarded
run holds .run/workers.lock, or a plan was refused) is retried at the next tick without a cooldown, and
REFUSED_RETRIES refusals in a row count as one failure; any other exit is a failure and starts the cooldown. After
FAILURE_LIMITS[kind] failures in a row (GPU scale-up, GPU scale-down, CPU scale-up, CPU removal) that kind of action
is off until the autoscaler is restarted, and every tick says so: a broken join or apply never becomes an endless
loop of paid launches or Terraform runs. Only that kind is off: with GPU scale-down off, GPU nodes get no idle clock
but GPU scale-up still runs, and the reverse. A gpu_scale that started nothing (NotStarted: the lock is held, or what
the tick read changed) is neither a failure nor a cooldown; a child a signal killed (SIGHUP: returncode -1) is a
failure.

make gpu-down always wins (spec section 10.4), and the autoscaler should be stopped (Ctrl-C) before it, which the
startup line says. gpuctl apply, down and sync-peers hold .run/gpuctl.lock, and one started while another runs waits
for it (gpuctl.lock); the autoscaler takes that lock without waiting (held: the gpu_scale is not started and is
decided again next tick) before its re-check, and keeps it through its write and gpuctl apply. What that
guarantees: a gpu-down that holds the lock when a gpu_scale wants it wins (nothing is started), and one that starts
during a gpu_scale runs right after its apply and removes what it launched. A gpu-down that ended between this
tick's read and the lock is seen only through what it changed: the override file (changed or deleted), or live
instances it terminated (gone, and not this process's own failed scale-down finishing: its pool still has its
target live); then nothing is written and GPU scaling is off until a restart. gpuctl keeps no record of a gpu-down
(spec section 5.4; the lock file is empty), so one that changed nothing observable (no override file and no live
GPU instance) is invisible: the gpu_scale goes ahead exactly as the next tick would, and launches the node a Pending
Pod asks for. That is why spec section 10.4 says to stop the autoscaler before make gpu-down. Across ticks the
autoscaler also protects itself:
  - once it has written the override file, finding that file deleted or changed by someone else (make gpu-down, or a
    hand edit) turns GPU scaling off until it is restarted; a write re-reads the file first and stops if it changed
    since the tick read it;
  - an autoscaled pool with fewer live Lambda instances than its target (gpu-down terminated them, or make gpu-apply
    has not run yet) holds GPU scale-up (scaling up from a target nobody runs would launch two nodes at once);
    scale-down, which never launches anything, stays on.

A tick that cannot read its inputs (kubectl failed, a list without items, a malformed state, override, inventory or
spec file) is skipped and logged (Ruling SR7(a)): a failed read must never look like an empty cluster.

Every tick writes JSON lines {"ts", "action", "reason", "result", "duration_s"} (plus target/node/pool) to
<DIR>/autoscaler.log, by default verification/autoscale-<ts>/: one per action, a "none" line when nothing is done
(with why), "notice" lines and "skip_tick" lines. --dry-run prints the action instead of running it, saves no
state (it keeps it in memory, so a dry loop shows the cooldown) and writes a log only with --log.

Ctrl-C between ticks stops at once; while the tick reads (kubectl, gpuctl list-nodes, gpuctl plan) the read
finishes and the loop stops, and a second Ctrl-C stops it at once (nothing is changing then); during an action the
action finishes first (its process runs in its own session, so the terminal's SIGINT does not reach it), then the
loop stops. A second Ctrl-C interrupts the action (SIGINT to its process group); then check with make gpu-status
and the override file, and finish with make cpu-scale N=1 / make gpu-down, or run the autoscaler again. The
signal handler only sets a flag (never a lock), and every first Ctrl-C says what will happen.

Test hooks: AUTOSCALER_STATE (state file), AUTOSCALER_ROOT (repo root for every file and script above),
AUTOSCALER_NOW (a fixed clock, epoch seconds).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TextIO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lambda"))

from gpuctl.autoscale import (  # noqa: E402
    COOLDOWN_S, GPU_ADMIN_LABEL, PROTECTED_CPU_NODE, Action, NodeObs, State, decide, node_observations, pending_pods,
)
from gpuctl.lock import LOCK_FD_ENV, LockError, lock_path, try_lock  # noqa: E402
from gpuctl.planner import LIVE_STATUSES, PlanError, parse_node_name, target_counts  # noqa: E402
from gpuctl.reconcile import load_overrides  # noqa: E402
from gpuctl.spec import load_spec  # noqa: E402

DEFAULT_INTERVAL_S = 20.0
DEFAULT_CPU_MAX = 3
# cpu-pool.sh --guarded exit 2 (lock held, plan refused) is retried at the next tick this many times in a row; then
# the cooldown starts like after any other failure.
REFUSED_RETRIES = 3
# Failures in a row after which a kind of action is off until the autoscaler is restarted: every GPU attempt is a
# paid launch (or a terminate), every CPU attempt a Terraform run.
FAILURE_LIMITS = {"gpu_up": 2, "gpu_down": 2, "cpu_scale": 3, "cpu_remove": 3}
KIND_LABELS = {"gpu_up": "GPU scale-up", "gpu_down": "GPU scale-down", "cpu_scale": "CPU scale-up",
               "cpu_remove": "CPU worker removal"}
# A node's Ready lastTransitionTime this far ahead of the local clock is logged (it keeps the node 'joining').
SKEW_WARN_S = 60.0
READ_TIMEOUT_S = 120
CLEANUP_HINT = ("an action was interrupted: check with make gpu-status and kubectl get nodes, and check "
                "lambda/.autoscale-state.json (the GPU targets make gpu-apply launches) before any make gpu-apply; "
                "then finish with make cpu-scale N=1 (CPU) or make gpu-down (GPU), or run make autoscale again")
STARTUP_HINT = ("autoscaler: stop it with Ctrl-C before make gpu-down or make cpu-scale (spec section 10.4); "
                "Ctrl-C waits for a running action, a second Ctrl-C interrupts it")
# decide()'s wording when no pool is autoscaled; replaced by the real reason when GPU scaling is off
NO_AUTOSCALE = "no GPU pool has autoscale"
# decide()'s wording when the caller says a GPU instance is launching; replaced by the instances' real state
LAUNCHING = "a GPU instance is still booting"


@dataclass(frozen=True)
class Result:
    rc: int
    out: str
    err: str


@dataclass(frozen=True)
class NotStarted:
    """_gpu_scale started nothing (another gpuctl write holds the lock, or what the tick read changed): no cooldown,
    no failure. A type of its own, never a return code: a child killed by SIGHUP has returncode -1."""
    why: str


def _exit_text(rc: int) -> str:
    """'exit 1', or 'killed by SIGHUP' for a child a signal ended (Popen's returncode -N)."""
    if rc < 0:
        try:
            return f"killed by {signal.Signals(-rc).name}"
        except ValueError:
            return f"killed by signal {-rc}"
    return f"exit {rc}"


@dataclass(frozen=True)
class Paths:
    root: Path
    state: Path
    overrides: Path
    spec: Path
    tfvars: Path
    gpuctl: Path
    cpu_pool: Path
    kubeconfig: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Paths:
        root = Path(env.get("AUTOSCALER_ROOT") or REPO)
        state = env.get("AUTOSCALER_STATE")
        return cls(root=root, state=Path(state) if state else root / ".run" / "autoscaler-state.json",
                   overrides=root / "lambda" / ".autoscale-state.json", spec=root / "lambda" / "gpu-nodes.yaml",
                   tfvars=root / "infra" / "workers" / "workers.auto.tfvars.json",
                   gpuctl=root / "scripts" / "gpuctl", cpu_pool=root / "scripts" / "cpu-pool.sh",
                   kubeconfig=Path.home() / ".kube" / "hybrid-gpu.yaml")


class Stopper:
    """SIGINT/SIGTERM: the first asks the loop to stop after the current read or action and says so; a second one
    during an action or a read raises KeyboardInterrupt there (run_command passes it on to the child's process
    group). Elsewhere (saving the state, writing the log) a second one only repeats the message.

    The handler sets plain attributes and writes with os.write: it takes no lock, because a lock the interrupted
    main thread holds (threading.Event.wait's Condition, a buffered stream) would deadlock it."""

    def __init__(self) -> None:
        self.requested = False
        self.in_action = False
        self.in_read = False

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def request(self) -> None:
        self.requested = True

    @contextlib.contextmanager
    def reading(self):
        """Read-only commands run inside: a second Ctrl-C stops them (and the loop) at once."""
        self.in_read = True
        try:
            yield
        finally:
            self.in_read = False

    def _handle(self, signum, frame) -> None:
        if self.requested and (self.in_action or self.in_read):
            raise KeyboardInterrupt
        self.requested = True
        if self.in_action:
            message = "stopping after the current action (Ctrl-C again interrupts it)"
        elif self.in_read:
            message = "stopping after this read (Ctrl-C again stops at once; nothing is changing)"
        else:
            message = "stopping"
        os.write(2, f"autoscaler: {message}\n".encode())

    def wait(self, seconds: float) -> bool:
        """Sleep up to `seconds` in short slices; True as soon as a stop was requested."""
        deadline = time.monotonic() + seconds
        while not self.requested:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            time.sleep(min(left, 0.2))
        return True


def run_command(argv: list[str], *, env: Mapping[str, str] | None = None, stream: bool = False,
                out: TextIO | None = None, pass_fds: tuple[int, ...] = ()) -> Result:
    """Run argv in its own session (the terminal's Ctrl-C does not reach it) with stdin from /dev/null.

    pass_fds: descriptors the child inherits (gpuctl's write lock for its apply); every other one is closed.

    stream=True (actions): its output goes to `out` line by line as it comes, and Result.out keeps the last lines.
    A KeyboardInterrupt (the second Ctrl-C) sends SIGINT to its process group, waits for it, and propagates.
    """
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT if stream else subprocess.PIPE, text=True, errors="replace",
                            env=full_env, start_new_session=True, pass_fds=pass_fds)
    try:
        if not stream:
            try:
                stdout, stderr = proc.communicate(timeout=READ_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _signal_group(proc, signal.SIGKILL)
                proc.communicate()
                return Result(124, "", f"timed out after {READ_TIMEOUT_S}s")
            return Result(proc.returncode, stdout, stderr)
        tail: collections.deque[str] = collections.deque(maxlen=20)
        for line in proc.stdout:
            tail.append(line.rstrip("\n"))
            if out is not None:
                out.write(f"  | {line.rstrip()}\n")
                out.flush()
        return Result(proc.wait(), "\n".join(tail), "")
    except KeyboardInterrupt:
        _signal_group(proc, signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)
            proc.wait()
        raise


def _signal_group(proc: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(proc.pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


@dataclass
class Deps:
    clock: Callable[[], float] = time.time
    run: Callable[..., Result] | None = None  # run(argv, *, env=None, stream=False) -> Result
    wait: Callable[[float], bool] | None = None  # sleep up to N seconds; True when a stop was requested
    out: TextIO = field(default_factory=lambda: sys.stdout)
    stopper: Stopper | None = None  # tests: a stop request without a signal


class SkipTick(Exception):
    """This tick's inputs cannot be read: skip it and log why."""


@dataclass
class Snapshot:
    now: float
    state: State
    pending: list
    nodes: list[NodeObs]
    cpu_count: int
    pools: dict
    gpu_counts: dict[str, int]
    overrides: dict[str, int]
    overrides_text: str | None  # the override file as this tick read it (None: absent)
    live: dict[str, int]  # live Lambda instances per pool
    live_names: frozenset[str] | None  # their names (None: list-nodes failed this tick)
    launching: list[str]
    gpu_off: str | None  # why GPU scaling is off this tick, or None
    gpu_short: str | None  # why GPU scale-up is off this tick (a pool below its target), or None
    notices: list[str]


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def atomic_write(path: Path, text: str, *, mode: int = 0o644) -> None:
    """Write text to path through a temp file in the same directory and a rename: never a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fchmod(f.fileno(), mode)
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


def describe(action: Action) -> str:
    if action.kind == "cpu_scale":
        return f"cpu_scale target={action.target}"
    if action.kind == "cpu_remove":
        return f"cpu_remove node={action.node}"
    return f"gpu_scale pool={action.pool} target={action.target}"


class Autoscaler:
    def __init__(self, args: argparse.Namespace, paths: Paths, deps: Deps, stopper: Stopper) -> None:
        self.args, self.paths, self.deps, self.stopper = args, paths, deps, stopper
        self.out = deps.out
        self.refusals = 0
        self.failures: dict[str, int] = {}  # action kind (FAILURE_LIMITS) -> failures in a row
        self.disabled: dict[str, str] = {}  # action kind -> why it is off until a restart
        self.gpu_disabled: str | None = None  # why GPU scaling is off until a restart (the override file changed)
        # The override file's text as this process last left it ("" = it removed the file); None until it writes it
        self.override_written: str | None = None
        self.unfinished_down: set[str] = set()  # pools whose scale-down by this process failed (review r2-2)
        self.plan_refusals: dict[tuple[str, int], tuple[float, str]] = {}  # (pool, target) -> (when, why)
        self.action_interrupted = False
        self.notes: list[str] = []  # what the current action did to the override file
        self.memory_state: State | None = None  # --dry-run: the state is kept here, never saved
        self.log_path: Path | None = None
        self.log_errors: set[int | None] = set()  # errnos of log writes that failed (each said once)
        if args.log:
            self.log_path = Path(args.log) / "autoscaler.log"
        elif not args.dry_run:
            stamp = datetime.fromtimestamp(deps.clock()).astimezone().strftime("%Y-%m-%dT%H%M%z")
            self.log_path = paths.root / "verification" / f"autoscale-{stamp}" / "autoscaler.log"
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # --- output --------------------------------------------------------------------------------

    def _ts(self, epoch: float) -> str:
        return datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds")

    def record(self, action: str, *, reason: str, result: str, started: float, **extra) -> None:
        line = {"ts": self._ts(started), "action": action, **{k: v for k, v in extra.items() if v is not None},
                "reason": reason, "result": result,
                "duration_s": round(max(self.deps.clock() - started, 0.0), 1)}
        if self.log_path is None:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
        except OSError as e:  # a full disk, a removed directory: never the end of the loop (review S4-r3-6)
            if e.errno not in self.log_errors:  # once per kind of error
                self.log_errors.add(e.errno)
                print(f"autoscaler: cannot write the log {self.log_path}: {e.strerror or e}; the autoscaler goes on "
                      "and this terminal still shows every tick (said once per kind of error)", file=sys.stderr)

    def say(self, text: str) -> None:
        self.out.write(text + "\n")
        self.out.flush()

    def run(self, argv: list, **kw) -> Result:
        argv = [str(a) for a in argv]
        try:
            if self.deps.run is not None:
                return self.deps.run(argv, **kw)
            return run_command(argv, out=self.out if kw.get("stream") else None, **kw)
        except OSError as e:  # not found, not executable: a failed command like any other
            message = f"cannot run {argv[0]}: {e.strerror or e}"
            return Result(127, message, message)

    # --- reading -------------------------------------------------------------------------------

    def _read_state(self) -> State:
        if self.args.dry_run and self.memory_state is not None:
            return self.memory_state
        # malformed -> ValueError -> the tick is skipped (Ruling SR7(a))
        return State.from_json(_read_text(self.paths.state) or "")

    def _save_state(self, state: State) -> str | None:
        """Save the state file; the error text if it cannot be written (never a crash without a log line)."""
        try:
            atomic_write(self.paths.state, state.to_json() + "\n", mode=0o600)
        except OSError as e:
            return f"cannot write {self.paths.state}: {e.strerror or e}"
        return None

    def _kubectl_json(self, what: str, path: str | None) -> dict:
        if path:
            text = Path(path).read_text(encoding="utf-8")
        else:
            r = self.run(["kubectl", "--kubeconfig", self.paths.kubeconfig, "--request-timeout=30s",
                          "get", *(["pods", "-A"] if what == "pods" else ["nodes"]), "-o", "json"])
            if r.rc != 0:
                raise SkipTick(f"kubectl get {what} failed (exit {r.rc}): {_last_line(r.err)}")
            text = r.out
        doc = json.loads(text)
        if not isinstance(doc, dict):
            raise ValueError(f"{what}: not a kubectl list")
        return doc

    def _cpu_count(self) -> int:
        data = json.loads(self.paths.tfvars.read_text(encoding="utf-8"))
        workers = data.get("cpu_workers") if isinstance(data, dict) else None
        if not isinstance(workers, dict) or not workers:
            raise ValueError(f"{self.paths.tfvars}: no cpu_workers inventory")
        return len(workers)

    def _instances(self, registered: set[str]) -> tuple[list[str], dict[str, int], frozenset[str], str | None]:
        """Why a GPU instance is launching (booting, or live without a Node), the live instances per pool, their
        names, and an error if list-nodes failed."""
        r = self.run([self.paths.gpuctl, "list-nodes"])
        if r.rc != 0:
            return [], {}, frozenset(), f"gpuctl list-nodes failed ({_exit_text(r.rc)}): {_last_line(r.err)}"
        reasons, live, names = [], collections.Counter(), set()
        for line in r.out.splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                return [], {}, frozenset(), f"gpuctl list-nodes printed an unexpected line {line[:120]!r}"
            name, _ip, status = fields[:3]
            parsed = parse_node_name(name)
            if parsed is not None and status in LIVE_STATUSES:
                live[parsed[0]] += 1
                names.add(name)
            if status == "booting":
                reasons.append(f"Lambda instance {name} is booting")
            elif status in LIVE_STATUSES and name not in registered:
                reasons.append(f"Lambda instance {name} is {status} but has no registered Node yet")
        return reasons, dict(live), frozenset(names), None

    def observe(self, now: float) -> Snapshot:
        try:
            state = self._read_state()
            pods = self._kubectl_json("pods", self.args.pods_file)
            nodes_doc = self._kubectl_json("nodes", self.args.nodes_file)
            nodes = node_observations(nodes_doc, pods)
            pending = pending_pods(pods, now)
            cpu_count = self._cpu_count()
            spec = load_spec(self.paths.spec)
            overrides_text = _read_text(self.paths.overrides)
            overrides = load_overrides(self.paths.overrides)
            gpu_counts = target_counts(spec, overrides=overrides)
        except (ValueError, PlanError, OSError) as e:  # SpecError and JSONDecodeError are ValueErrors
            raise SkipTick(f"{type(e).__name__}: {e}") from None

        notices = []
        gpu_nodes = [o for o in nodes if o.kind == "gpu"]
        for obs in gpu_nodes:
            if not obs.ready:
                notices.append(f"GPU node {obs.name} is NotReady: GPU scale-up is held until it is Ready or gone "
                               "(make gpu-status)")
            elif not obs.labeled:
                notices.append(f"GPU node {obs.name} has no {GPU_ADMIN_LABEL[0]} label yet: GPU scale-up is held")
        for obs in nodes:
            if obs.ready_since is not None and obs.ready_since - now > SKEW_WARN_S:
                notices.append(f"clock skew: {obs.name}'s Ready lastTransitionTime is {obs.ready_since - now:.0f}s "
                               "ahead of this machine's clock (a GPU node stays 'joining' while it lasts; check NTP)")
        launching, live, live_names, error = self._instances({o.name for o in gpu_nodes})
        if error is not None:
            launching = [f"{error}: GPU scale-up held this tick"]
        notices.extend(launching)

        # make gpu-down always wins (spec section 10.4): see the module docstring
        if self.override_written is not None and (overrides_text or "") != self.override_written:
            self.gpu_disabled = (f"{self.paths.overrides} was {'deleted' if overrides_text is None else 'changed'} "
                                 "outside the autoscaler (make gpu-down, or a hand edit): GPU scaling is off until "
                                 "make autoscale is restarted")
        gpu_off = self.gpu_disabled
        if gpu_off is not None:
            notices.append(gpu_off)
        gpu_short = None  # scale-down (it never launches anything) stays on
        if gpu_off is None and error is None:
            short = [f"pool {name}: {live.get(name, 0)} of {gpu_counts[name]} GPU instances live"
                     for name in sorted(spec.pools)
                     if spec.pools[name].autoscale is not None and live.get(name, 0) < gpu_counts[name]]
            if short:
                gpu_short = (f"{', '.join(short)} (make gpu-down ran, or make gpu-apply has not): no GPU scale-up "
                             "until make gpu-apply brings the pool to its target")
                if any(p.resource == "gpu" for p in pending):  # a CPU-only demo runs with no GPU node at all
                    notices.append(gpu_short)
        if error is None:  # more paid instances than the target: a scale-down that did not finish (review r2-2)
            for name in sorted(spec.pools):
                if spec.pools[name].autoscale is None or live.get(name, 0) <= gpu_counts[name]:
                    self.unfinished_down.discard(name)
                    continue
                retry = name in self.unfinished_down and gpu_off is None and "gpu_down" not in self.disabled
                notices.append(f"pool {name}: {live[name]} GPU instances live, target {gpu_counts[name]}: a "
                               "scale-down has not finished and the extra instances keep billing; "
                               + ("the autoscaler runs gpuctl apply again after the cooldown" if retry else
                                  "make gpu-apply terminates them (make gpu-down removes every GPU node)"))
        return Snapshot(now=now, state=state, pending=pending, nodes=nodes, cpu_count=cpu_count,
                        pools=dict(spec.pools), gpu_counts=gpu_counts, overrides=overrides,
                        overrides_text=overrides_text, live=live, live_names=None if error else live_names,
                        launching=launching, gpu_off=gpu_off, gpu_short=gpu_short, notices=notices)

    # --- one tick ------------------------------------------------------------------------------

    def decide(self, snap: Snapshot, *, gpu: bool = True) -> tuple[Action | None, State]:
        """decide() without GPU scaling when it is off, and without the kinds of action FAILURE_LIMITS turned off."""
        pending, nodes = snap.pending, snap.nodes
        if "gpu_up" in self.disabled or snap.gpu_short:
            pending = [p for p in pending if p.resource != "gpu"]
        if "cpu_scale" in self.disabled:
            pending = [p for p in pending if p.resource == "gpu"]
        if "gpu_down" in self.disabled:  # a busy node gets no idle clock, so rule 5 cannot pick it; scale-up stays
            nodes = [replace(o, busy=True) if o.kind == "gpu" else o for o in nodes]
        if "cpu_remove" in self.disabled:  # likewise for rule 4
            nodes = [replace(o, busy=True) if o.kind == "cpu" and o.name != PROTECTED_CPU_NODE else o for o in nodes]
        return decide(now=snap.now, pending=pending, nodes=nodes, state=snap.state,
                      cpu_count=snap.cpu_count, cpu_max=self.args.cpu_max,
                      gpu_counts=snap.gpu_counts if gpu else {}, pools=snap.pools if gpu else {},
                      gpu_launching=bool(snap.launching))

    @staticmethod
    def _failure_kind(snap: Snapshot, action: Action) -> str:
        if action.kind == "gpu_scale":
            return "gpu_up" if action.target > snap.gpu_counts.get(action.pool, 0) else "gpu_down"
        return action.kind

    def tick(self) -> str:
        """One tick; returns "ok", "none", "dry-run", "skipped", "refused" or "failed"."""
        started = now = self.deps.clock()
        try:
            with self.stopper.reading():
                snap = self.observe(now)
        except SkipTick as e:
            self.say(f"{self._ts(now)} tick skipped: {e}")
            self.record("skip_tick", reason=str(e), result="skipped", started=started)
            return "skipped"
        for notice in snap.notices + [self.disabled[kind] for kind in sorted(self.disabled)]:
            self.say(f"{self._ts(now)} notice: {notice}")
            self.record("notice", reason=notice, result="logged", started=started)

        gpu_off = snap.gpu_off
        action, state = self.decide(snap, gpu=gpu_off is None)
        if action is None and gpu_off is None:
            action, state = self._finish_scale_down(snap, state)
        if action is not None and action.kind == "gpu_scale":
            refused = self._plan_check(action, now, started)
            if refused is not None:
                gpu_off = refused
                action, state = self.decide(snap, gpu=False)  # the CPU rules still get this tick
        if action is None:
            return self._no_action(snap, state, gpu_off, started)
        if self.args.dry_run:
            self.memory_state = state
            self.say(f"{describe(action)} (dry-run): {action.reason}")
            self.record(action.kind, target=action.target, node=action.node, pool=action.pool,
                        reason=action.reason, result="dry-run", started=started)
            return "dry-run"
        if self.stopper.requested:  # Ctrl-C came while this tick read the cluster: start nothing new
            self.say(f"{self._ts(now)} {describe(action)}: not started (stopping)")
            self.record(action.kind, target=action.target, node=action.node, pool=action.pool,
                        reason=action.reason, result="not started (Ctrl-C)", started=started)
            return "none"
        return self._act(snap, action, state, started)

    def _finish_scale_down(self, snap: Snapshot, state: State) -> tuple[Action | None, State]:
        """A scale-down of this process that failed and left more live instances than its target: gpuctl apply again
        (to the same, lower target) once the cooldown is over, counted as a GPU scale-down (review r2-2)."""
        last = state.last_action_at
        if "gpu_down" in self.disabled or snap.live_names is None or (last is not None
                                                                      and snap.now - last < COOLDOWN_S):
            return None, state
        for name in sorted(self.unfinished_down):
            live, target = snap.live.get(name, 0), snap.gpu_counts.get(name)
            if target is not None and live > target:
                return (Action("gpu_scale", target=target, pool=name,
                               reason=f"finish a failed scale-down: {live} GPU instances live, target {target}"),
                        replace(state, last_action_at=snap.now))
        return None, state

    def _plan_check(self, action: Action, now: float, started: float) -> str | None:
        """`gpuctl plan --set-count` for this gpu_scale (budget_usd_per_hour, max_nodes: spec section 10.5): why it
        is refused, or None. A refusal is remembered for COOLDOWN_S: no Lambda API call and no log line every tick."""
        key = (action.pool, action.target)
        if key in self.plan_refusals:
            when, why = self.plan_refusals[key]
            if now - when < COOLDOWN_S:
                return (f"gpu_scale {action.pool}={action.target} refused by gpuctl plan {now - when:.0f}s ago "
                        f"(checked again in {COOLDOWN_S - (now - when):.0f}s): {why}")
        with self.stopper.reading():
            check = self.run([self.paths.gpuctl, "plan", "--set-count", f"{action.pool}={action.target}"])
        if check.rc == 0:
            self.plan_refusals.pop(key, None)
            return None
        why = _last_line(check.err) or _last_line(check.out)
        result = f"refused by gpuctl plan (exit {check.rc}): {why}"
        self.say(f"{self._ts(now)} {describe(action)}: {action.reason} -> {result}")
        self.record("gpu_scale", pool=action.pool, target=action.target, reason=action.reason, result=result,
                    started=started)
        self.plan_refusals[key] = (now, why)
        return f"gpu_scale {action.pool}={action.target} refused by gpuctl plan: {why}"

    def _no_action(self, snap: Snapshot, state: State, gpu_off: str | None, started: float) -> str:
        now = snap.now
        held = state.held or ""
        off = gpu_off  # GPU scale-down off (FAILURE_LIMITS) holds no Pending Pod: it is listed below
        if off:
            held = held.replace(f"waiting: {NO_AUTOSCALE}", f"waiting: {off}")
        if snap.launching:  # decide() knows only that something launches; the log names what
            held = held.replace(f"waiting: {LAUNCHING}", "waiting: " + "; ".join(snap.launching))
        parts = [held] if held else []
        if snap.gpu_short and off is None and any(p.resource == "gpu" for p in snap.pending):
            parts.append(snap.gpu_short)
        parts.extend(self.disabled[kind] for kind in sorted(self.disabled) if self.disabled[kind] != off)
        idle = ", ".join(f"{name} idle {now - since:.0f}s" for name, since in sorted(state.idle_since.items()))
        if idle:
            parts.append(idle)
        last = state.last_action_at
        if last is not None and now - last < COOLDOWN_S:
            head, result = f"no action (cooldown): {COOLDOWN_S - (now - last):.0f}s left", "cooldown"
        else:
            head, result = "no action", "held" if held else "idle"
        reason = "; ".join(parts) or "nothing to do"
        self.say(f"{self._ts(now)} {head}: {reason}")
        self.record("none", reason=f"{head}: {reason}", result=result, started=started)
        if self.args.dry_run:
            self.memory_state = state
        elif (error := self._save_state(state)) is not None:
            self.say(f"{self._ts(now)} notice: {error}")
            self.record("notice", reason=error, result="logged", started=started)
        return "none"

    def _failed(self, kind: str, result: str) -> str:
        """Count a failure of this kind of action; after FAILURE_LIMITS[kind] in a row it is off until a restart."""
        count = self.failures.get(kind, 0) + 1
        self.failures[kind] = count
        limit, label = FAILURE_LIMITS[kind], KIND_LABELS[kind]
        if count < limit:
            return f"{result}; retried after the cooldown ({count}/{limit} failures)"
        self.disabled[kind] = (f"{label} disabled after {count} failures in a row (last: {result}); fix the cause "
                               "and run make autoscale again")
        return f"{result}; {label} disabled after {count} failures in a row until make autoscale is restarted"

    def _act(self, snap: Snapshot, action: Action, state: State, started: float) -> str:
        previous_last = snap.state.last_action_at
        kind = self._failure_kind(snap, action)
        extra = {"target": action.target, "node": action.node, "pool": action.pool}
        self.say(f"{self._ts(snap.now)} {describe(action)}: {action.reason}")
        # Review Focus #1: the cooldown is on disk before anything changes; if it cannot be, nothing starts
        error = self._save_state(state)
        if error is not None:
            result = self._failed(kind, f"failed: {error}; not started")
            self.say(f"{self._ts(self.deps.clock())} {describe(action)} -> {result}")
            self.record(action.kind, **extra, reason=action.reason, result=result, started=started)
            return "failed"
        self.notes = []
        try:
            self.stopper.in_action = True
            if action.kind == "gpu_scale":
                r = self._gpu_scale(snap, action)
            else:
                argv = ([self.paths.cpu_pool, "scale", action.target, "--guarded"] if action.kind == "cpu_scale"
                        else [self.paths.cpu_pool, "remove", action.node, "--guarded"])
                r = self.run(argv, env={"AUTO_APPROVE": "1"}, stream=True)
        except KeyboardInterrupt:
            self.stopper.in_action = False
            self.action_interrupted = True
            self._save_state(replace(state, last_action_at=self.deps.clock()))
            self.record(action.kind, **extra, reason=action.reason,
                        result="; ".join(["interrupted (second Ctrl-C)", *self.notes]), started=started)
            raise
        finally:
            self.stopper.in_action = False

        outcome = "failed"
        if isinstance(r, NotStarted):
            outcome, result = "refused", f"not started: {r.why}"
            state = replace(state, last_action_at=previous_last)
        elif r.rc == 0:
            self.refusals, outcome, result = 0, "ok", "ok"
            self.failures.pop(kind, None)
            if action.kind == "gpu_scale":
                self.unfinished_down.discard(action.pool)
            state = replace(state, last_action_at=self.deps.clock())
        elif r.rc == 2 and action.kind != "gpu_scale":
            self.refusals += 1
            why = _last_line(r.out)
            if self.refusals < REFUSED_RETRIES:
                outcome = "refused"
                result = f"refused (exit 2): {why}; retry next tick ({self.refusals}/{REFUSED_RETRIES})"
                state = replace(state, last_action_at=previous_last)
            else:
                result = self._failed(kind, f"refused (exit 2) {self.refusals} times in a row: {why}; "
                                            f"cooldown {COOLDOWN_S}s")
                self.refusals = 0
                state = replace(state, last_action_at=self.deps.clock())
        else:
            self.refusals = 0
            if kind == "gpu_down":  # the lower target stays (never a relaunch); its apply is run again (r2-2)
                self.unfinished_down.add(action.pool)
            result = self._failed(kind, f"failed ({_exit_text(r.rc)}): {_last_line(r.out)}")
            state = replace(state, last_action_at=self.deps.clock())
        result = "; ".join([result, *self.notes])
        error = self._save_state(state)
        if error is not None:
            result += f" (state not saved: {error})"
        self.say(f"{self._ts(self.deps.clock())} {describe(action)} -> {result}")
        self.record(action.kind, **extra, reason=action.reason, result=result, started=started)
        return outcome

    def _gpu_scale(self, snap: Snapshot, action: Action) -> Result | NotStarted:
        """Under gpuctl's write lock: re-check what the tick read, merge {pool: target} into the override file and
        run gpuctl apply with the lock (see the module docstring)."""
        lock = lock_path(self.paths.root)
        try:
            fd = try_lock(lock)
        except (LockError, OSError) as e:  # a failure of this gpu_scale, counted like any other
            return Result(1, str(e), "")
        if fd is None:  # make gpu-down or make gpu-apply runs in another terminal: decided again next tick
            return NotStarted(f"another gpuctl apply/down/sync-peers holds {lock} (make gpu-down or make gpu-apply "
                              "in another terminal, possibly at its confirmation prompt, or a cpu-pool.sh sync-peers; "
                              f"fuser -v {lock} shows it); decided again next tick")
        try:
            return self._gpu_scale_locked(snap, action, fd)
        finally:
            os.close(fd)  # releases the lock

    def _gpu_scale_locked(self, snap: Snapshot, action: Action, fd: int) -> Result | NotStarted:
        path = self.paths.overrides
        try:
            before = _read_text(path)
        except OSError as e:
            return Result(1, f"cannot read {path}: {e.strerror or e}", "")
        if before != snap.overrides_text:  # make gpu-down (or a hand edit) since this tick read it
            self.gpu_disabled = (f"{path} changed while a tick ran (make gpu-down, or a hand edit): GPU scaling is "
                                 "off until make autoscale is restarted")
            return NotStarted(f"{path} changed since this tick read it (make gpu-down?); GPU scaling is off until "
                              "make autoscale is restarted")
        # A make gpu-down that ended after this tick read the instances leaves no record (spec section 5.4); without
        # an override file only its terminates can show, here. With no live instance either it is invisible (see the
        # module docstring). Any later one waits for this lock, so it runs after this apply and wins (section 10.4).
        _, _, names, error = self._instances(set())
        if error is not None or snap.live_names is None:
            return NotStarted("cannot re-check the GPU instances before writing the target: "
                              + (error or "list-nodes failed when this tick read them"))
        if names != snap.live_names:
            gone, new = sorted(snap.live_names - names), sorted(names - snap.live_names)
            finished = self._own_scale_down_finished(snap, gone, names)
            unexplained = [name for name in gone if name not in finished]
            change = "; ".join(filter(None, [
                unexplained and f"gone: {', '.join(unexplained)} (make gpu-down?)",
                finished and (f"gone: {', '.join(finished)}: this autoscaler's failed scale-down finished "
                              "meanwhile"),
                new and f"new: {', '.join(new)}"]))
            if unexplained:
                self.gpu_disabled = (f"GPU instances went away while a tick ran ({', '.join(unexplained)}; make "
                                     "gpu-down?): GPU scaling is off until make autoscale is restarted")
            return NotStarted(f"the GPU instances changed since this tick read them ({change})"
                              + ("; GPU scaling is off until make autoscale is restarted" if unexplained else
                                 "; decided again next tick"))
        merged = dict(sorted({**snap.overrides, action.pool: action.target}.items()))
        text = json.dumps({"pools": merged}) + "\n"
        scale_up = action.target > snap.gpu_counts.get(action.pool, 0)
        applied = False
        try:  # the write too: a second Ctrl-C right after it still restores the file (review r2-3)
            try:
                atomic_write(path, text)
            except OSError as e:
                return Result(1, f"cannot write {path}: {e.strerror or e}", "")
            self.override_written = text
            r = self.run([self.paths.gpuctl, "apply", "--auto-approve"], stream=True, pass_fds=(fd,),
                         env={LOCK_FD_ENV: str(fd)})
            applied = r.rc == 0
            return r
        finally:  # also on the second Ctrl-C
            if scale_up and not applied:
                # a further Ctrl-C only repeats the message: it must not leave the raised target behind (r2-3)
                self.stopper.in_action = False
                self._restore_override(written=text, before=before)

    def _own_scale_down_finished(self, snap: Snapshot, gone: list[str], names: frozenset[str]) -> list[str]:
        """The gone instances that this process's own failed scale-down explains (review S4-r3-2): their pool's
        scale-down failed earlier (its terminate may still go through) and the pool still has at least its target
        live. A make gpu-down leaves fewer (and deletes the override file)."""
        live_now = collections.Counter(parsed[0] for name in names if (parsed := parse_node_name(name)) is not None)
        finished = []
        for name in gone:
            parsed = parse_node_name(name)
            target = snap.gpu_counts.get(parsed[0]) if parsed is not None else None
            if parsed is not None and parsed[0] in self.unfinished_down and target is not None \
                    and live_now[parsed[0]] >= target:
                finished.append(name)
        return finished

    def _restore_override(self, *, written: str, before: str | None) -> None:
        """Put the override file back as it was before a scale-up that did not complete, unless someone else
        (make gpu-down) changed it meanwhile: then it is left as it is and GPU scaling is off until a restart."""
        path = self.paths.overrides
        try:
            current = _read_text(path)
            if current == before:  # the write never happened (it failed, or a Ctrl-C came first)
                return
            if current != written:
                self.gpu_disabled = (f"{path} changed during gpuctl apply (make gpu-down, or a hand edit): GPU "
                                     "scaling is off until make autoscale is restarted")
                self.notes.append(f"{path} changed during the apply (make gpu-down?): left as it is")
                return
            if before is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write(path, before)
            self.override_written = before or ""
            self.notes.append(f"{path} put back as it was")
        except OSError as e:
            self.notes.append(f"cannot put {path} back: {e.strerror or e}; check it before any make gpu-apply")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoscaler.py", description="Stretch autoscaler loop (spec section 10).")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S, help="seconds between ticks (20)")
    parser.add_argument("--cpu-max", type=int, default=DEFAULT_CPU_MAX, help="CPU workers at most (3)")
    parser.add_argument("--once", action="store_true", help="run one tick and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the action instead of running it")
    parser.add_argument("--pods-file", help="read `kubectl get pods -A -o json` from this file")
    parser.add_argument("--nodes-file", help="read `kubectl get nodes -o json` from this file")
    parser.add_argument("--log", metavar="DIR", help="write DIR/autoscaler.log (default verification/autoscale-<ts>/)")
    return parser


def main(argv: list[str] | None = None, *, env: Mapping[str, str] = os.environ, deps: Deps | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if bool(args.pods_file) != bool(args.nodes_file):
        parser.error("--pods-file and --nodes-file go together")
    if not (math.isfinite(args.interval) and args.interval > 0) or args.cpu_max < 1:
        parser.error("--interval must be > 0 and --cpu-max >= 1")
    stopper = (deps.stopper if deps is not None else None) or Stopper()
    if deps is None:
        fixed = env.get("AUTOSCALER_NOW")
        deps = Deps(clock=(lambda: float(fixed)) if fixed else time.time, wait=stopper.wait)
        stopper.install()
    wait = deps.wait or stopper.wait
    scaler = Autoscaler(args, Paths.from_env(env), deps, stopper)
    if scaler.log_path is not None:
        scaler.say(f"autoscaler: logging to {scaler.log_path}")
    scaler.say(STARTUP_HINT)
    try:
        if args.once:
            return 0 if scaler.tick() in ("ok", "none", "dry-run") else 1
        while True:
            scaler.tick()
            if stopper.requested or wait(args.interval):
                break
    except KeyboardInterrupt:
        if scaler.action_interrupted:
            print(f"autoscaler: {CLEANUP_HINT}", file=sys.stderr)
        else:
            print("autoscaler: stopped while reading the cluster; nothing was changing", file=sys.stderr)
        return 130
    scaler.say("autoscaler: stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
