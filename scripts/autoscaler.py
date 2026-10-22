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
     launch, a clock skew against a node's Ready time and the make gpu-down record are logged every tick, Pending
     Pods or not (a node NotReady for ever holds GPU scale-up for ever: the log says so each time);
  3. decide() (lambda/gpuctl/autoscale.py) returns at most one action;
  4. run it:
       cpu_scale  -> AUTO_APPROVE=1 scripts/cpu-pool.sh scale <target> --guarded
       cpu_remove -> AUTO_APPROVE=1 scripts/cpu-pool.sh remove <node> --guarded
       gpu_scale  -> `scripts/gpuctl plan --set-count <pool>=<target>` first (budget_usd_per_hour and max_nodes at
                     live prices, spec section 10.5; a refusal skips it and the CPU rules still get this tick), then
                     {pool: target} merged into lambda/.autoscale-state.json and `scripts/gpuctl apply --auto-approve`
                     (a failed apply puts the override file back as it was);
  5. save .run/autoscaler-state.json.

The autoscaler writes only lambda/.autoscale-state.json and its own state file, both atomically (temp file and
rename). The state is saved with last_action_at = now before an action starts, so a restart during or right after
it waits out the cooldown (Review Focus #1), and again when it ends: the cooldown counts from the end of the action.
Results: exit 0 is ok; cpu-pool.sh exit 2 (another guarded run holds .run/workers.lock, or a plan was refused) is
retried at the next tick without a cooldown, but after REFUSED_RETRIES refusals in a row the cooldown starts, so it
never loops on an error; any other failure is logged and starts the cooldown (the next decision repeats it after).

make gpu-down always wins (spec section 10.4): it deletes the override file, and while its record lambda/.gpu-down
exists (Ruling R53) the autoscaler makes no GPU decision at all (a gpu-apply would clear the record), which it logs.

A tick that cannot read its inputs (kubectl failed, a list without items, a malformed state, override, inventory or
spec file) is skipped and logged (Ruling SR7(a)): a failed read must never look like an empty cluster.

Every tick writes JSON lines {"ts", "action", "reason", "result", "duration_s"} (plus target/node/pool) to
<DIR>/autoscaler.log, by default verification/autoscale-<ts>/: one per action, a "none" line when nothing is done
(with why), "notice" lines and "skip_tick" lines. --dry-run prints the action instead of running it, saves no
state (it keeps it in memory, so a dry loop shows the cooldown) and writes a log only with --log.

Ctrl-C between ticks stops at once; during an action the action finishes first (its process runs in its own
session, so the terminal's SIGINT does not reach it), then the loop stops. A second Ctrl-C interrupts the action
(SIGINT to its process group); then check with make gpu-status and finish with make cpu-scale N=1 / make gpu-down,
or run the autoscaler again.

Test hooks: AUTOSCALER_STATE (state file), AUTOSCALER_ROOT (repo root for every file and script above),
AUTOSCALER_NOW (a fixed clock, epoch seconds).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TextIO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lambda"))

from gpuctl.autoscale import (  # noqa: E402
    COOLDOWN_S, GPU_ADMIN_LABEL, Action, NodeObs, State, decide, node_observations, pending_pods,
)
from gpuctl.planner import LIVE_STATUSES, PlanError, target_counts  # noqa: E402
from gpuctl.reconcile import load_overrides  # noqa: E402
from gpuctl.spec import load_spec  # noqa: E402

DEFAULT_INTERVAL_S = 20.0
DEFAULT_CPU_MAX = 3
# cpu-pool.sh --guarded exit 2 (lock held, plan refused) is retried at the next tick this many times in a row; then
# the cooldown starts like after any other failure.
REFUSED_RETRIES = 3
# A node's Ready lastTransitionTime this far ahead of the local clock is logged (it keeps the node 'joining').
SKEW_WARN_S = 60.0
READ_TIMEOUT_S = 120
CLEANUP_HINT = ("an action was interrupted: check with make gpu-status and kubectl get nodes, then finish with "
                "make cpu-scale N=1 (CPU) or make gpu-down / make gpu-apply (GPU), or run make autoscale again")
# decide()'s wording when no pool is autoscaled; replaced by the real reason when GPU scaling is off
NO_AUTOSCALE = "no GPU pool has autoscale"


@dataclass(frozen=True)
class Result:
    rc: int
    out: str
    err: str


@dataclass(frozen=True)
class Paths:
    root: Path
    state: Path
    overrides: Path
    gpu_down: Path
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
                   overrides=root / "lambda" / ".autoscale-state.json", gpu_down=root / "lambda" / ".gpu-down",
                   spec=root / "lambda" / "gpu-nodes.yaml",
                   tfvars=root / "infra" / "workers" / "workers.auto.tfvars.json",
                   gpuctl=root / "scripts" / "gpuctl", cpu_pool=root / "scripts" / "cpu-pool.sh",
                   kubeconfig=Path.home() / ".kube" / "hybrid-gpu.yaml")


class Stopper:
    """SIGINT/SIGTERM: the first asks the loop to stop after the current action; a second one during an action
    raises KeyboardInterrupt there (run_command passes it on to the action's process group)."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.in_action = False

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:
        if self.in_action and self.event.is_set():
            raise KeyboardInterrupt
        self.event.set()
        if self.in_action:  # os.write: never re-enter a buffered stream the main code may be writing
            os.write(2, b"autoscaler: stopping after the current action (Ctrl-C again interrupts it)\n")

    def wait(self, seconds: float) -> bool:
        return self.event.wait(seconds)


def run_command(argv: list[str], *, env: Mapping[str, str] | None = None, stream: bool = False,
                out: TextIO | None = None) -> Result:
    """Run argv in its own session (the terminal's Ctrl-C does not reach it) with stdin from /dev/null.

    stream=True (actions): its output goes to `out` line by line as it comes, and Result.out keeps the last lines.
    A KeyboardInterrupt (the second Ctrl-C) sends SIGINT to its process group, waits for it, and propagates.
    """
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT if stream else subprocess.PIPE, text=True, errors="replace",
                            env=full_env, start_new_session=True)
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
    launching: list[str]
    gpu_off: str | None  # why GPU scaling is off this tick, or None
    notices: list[str]


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
        self.memory_state: State | None = None  # --dry-run: the state is kept here, never saved
        self.log_path: Path | None = None
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
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")

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
        try:
            text = self.paths.state.read_text(encoding="utf-8")
        except FileNotFoundError:
            text = ""
        return State.from_json(text)  # malformed -> ValueError -> the tick is skipped (Ruling SR7(a))

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

    def _instances(self, registered: set[str]) -> tuple[list[str], str | None]:
        """Why a GPU instance is launching (booting, or live without a Node), and an error if list-nodes failed."""
        r = self.run([self.paths.gpuctl, "list-nodes"])
        if r.rc != 0:
            return [], f"gpuctl list-nodes failed (exit {r.rc}): {_last_line(r.err)}"
        reasons = []
        for line in r.out.splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                return [], f"gpuctl list-nodes printed an unexpected line {line[:120]!r}"
            name, _ip, status = fields[:3]
            if status == "booting":
                reasons.append(f"Lambda instance {name} is booting")
            elif status in LIVE_STATUSES and name not in registered:
                reasons.append(f"Lambda instance {name} is {status} but has no registered Node yet")
        return reasons, None

    def observe(self, now: float) -> Snapshot:
        try:
            state = self._read_state()
            pods = self._kubectl_json("pods", self.args.pods_file)
            nodes_doc = self._kubectl_json("nodes", self.args.nodes_file)
            nodes = node_observations(nodes_doc, pods)
            pending = pending_pods(pods, now)
            cpu_count = self._cpu_count()
            spec = load_spec(self.paths.spec)
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
        launching, error = self._instances({o.name for o in gpu_nodes})
        if error is not None:
            launching = [f"{error}: GPU scale-up held this tick"]
        notices.extend(launching)
        gpu_off = None
        if self.paths.gpu_down.exists():
            gpu_off = (f"make gpu-down record {self.paths.gpu_down} exists: no GPU scaling until a confirmed "
                       "make gpu-apply removes it")
            notices.append(gpu_off)
        return Snapshot(now=now, state=state, pending=pending, nodes=nodes, cpu_count=cpu_count,
                        pools=dict(spec.pools), gpu_counts=gpu_counts, overrides=overrides, launching=launching,
                        gpu_off=gpu_off, notices=notices)

    # --- one tick ------------------------------------------------------------------------------

    def decide(self, snap: Snapshot, *, gpu: bool = True) -> tuple[Action | None, State]:
        return decide(now=snap.now, pending=snap.pending, nodes=snap.nodes, state=snap.state,
                      cpu_count=snap.cpu_count, cpu_max=self.args.cpu_max,
                      gpu_counts=snap.gpu_counts if gpu else {}, pools=snap.pools if gpu else {},
                      gpu_launching=bool(snap.launching))

    def tick(self) -> str:
        """One tick; returns "ok", "none", "dry-run", "skipped", "refused" or "failed"."""
        started = now = self.deps.clock()
        try:
            snap = self.observe(now)
        except SkipTick as e:
            self.say(f"{self._ts(now)} tick skipped: {e}")
            self.record("skip_tick", reason=str(e), result="skipped", started=started)
            return "skipped"
        for notice in snap.notices:
            self.say(f"{self._ts(now)} notice: {notice}")
            self.record("notice", reason=notice, result="logged", started=started)

        gpu_off = snap.gpu_off
        action, state = self.decide(snap, gpu=gpu_off is None)
        if action is not None and action.kind == "gpu_scale":
            check = self.run([self.paths.gpuctl, "plan", "--set-count", f"{action.pool}={action.target}"])
            if check.rc != 0:  # budget_usd_per_hour / max_nodes (spec section 10.5), or the check failed
                why = _last_line(check.err) or _last_line(check.out)
                result = f"refused by gpuctl plan (exit {check.rc}): {why}"
                self.say(f"{self._ts(now)} {describe(action)}: {action.reason} -> {result}")
                self.record("gpu_scale", pool=action.pool, target=action.target, reason=action.reason,
                            result=result, started=started)
                gpu_off = f"GPU scale-up refused by gpuctl plan: {why}"
                action, state = self.decide(snap, gpu=False)  # the CPU rules still get this tick
        if action is None:
            return self._no_action(snap, state, gpu_off, started)
        if self.args.dry_run:
            self.memory_state = state
            self.say(f"{describe(action)} (dry-run): {action.reason}")
            self.record(action.kind, target=action.target, node=action.node, pool=action.pool,
                        reason=action.reason, result="dry-run", started=started)
            return "dry-run"
        if self.stopper.event.is_set():  # Ctrl-C came while this tick read the cluster: start nothing new
            self.say(f"{self._ts(now)} {describe(action)}: not started (stopping)")
            self.record(action.kind, target=action.target, node=action.node, pool=action.pool,
                        reason=action.reason, result="not started (Ctrl-C)", started=started)
            return "none"
        return self._act(snap, action, state, started)

    def _no_action(self, snap: Snapshot, state: State, gpu_off: str | None, started: float) -> str:
        now = snap.now
        held = state.held or ""
        if gpu_off:
            held = held.replace(f"waiting: {NO_AUTOSCALE}", f"waiting: {gpu_off}")
        parts = [held] if held else []
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
        else:
            atomic_write(self.paths.state, state.to_json() + "\n", mode=0o600)
        return "none"

    def _act(self, snap: Snapshot, action: Action, state: State, started: float) -> str:
        previous_last = snap.state.last_action_at
        # Review Focus #1: the cooldown is on disk before anything changes
        atomic_write(self.paths.state, state.to_json() + "\n", mode=0o600)
        self.say(f"{self._ts(snap.now)} {describe(action)}: {action.reason}")
        extra = {"target": action.target, "node": action.node, "pool": action.pool}
        outcome = "failed"
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
            state = replace(state, last_action_at=self.deps.clock())
            atomic_write(self.paths.state, state.to_json() + "\n", mode=0o600)
            self.record(action.kind, **extra, reason=action.reason, result="interrupted (second Ctrl-C)",
                        started=started)
            raise
        finally:
            self.stopper.in_action = False

        if r.rc == 0:
            self.refusals, outcome, result = 0, "ok", "ok"
            state = replace(state, last_action_at=self.deps.clock())
        elif r.rc == 2 and action.kind != "gpu_scale":
            self.refusals += 1
            why = _last_line(r.out)
            if self.refusals < REFUSED_RETRIES:
                outcome = "refused"
                result = f"refused (exit 2): {why}; retry next tick ({self.refusals}/{REFUSED_RETRIES})"
                state = replace(state, last_action_at=previous_last)
            else:
                result = f"refused (exit 2) {self.refusals} times in a row: {why}; cooldown {COOLDOWN_S}s"
                self.refusals = 0
                state = replace(state, last_action_at=self.deps.clock())
        else:
            self.refusals = 0
            result = f"failed (exit {r.rc}): {_last_line(r.out)}; the next decision repeats it after the cooldown"
            state = replace(state, last_action_at=self.deps.clock())
        atomic_write(self.paths.state, state.to_json() + "\n", mode=0o600)
        self.say(f"{self._ts(self.deps.clock())} {describe(action)} -> {result}")
        self.record(action.kind, **extra, reason=action.reason, result=result, started=started)
        return outcome

    def _gpu_scale(self, snap: Snapshot, action: Action) -> Result:
        """Merge {pool: target} into the override file, run gpuctl apply; put the file back if the apply fails."""
        try:
            before = self.paths.overrides.read_text(encoding="utf-8")
        except FileNotFoundError:
            before = None
        merged = dict(sorted({**snap.overrides, action.pool: action.target}.items()))
        atomic_write(self.paths.overrides, json.dumps({"pools": merged}) + "\n")
        r = self.run([self.paths.gpuctl, "apply", "--auto-approve"], stream=True)
        if r.rc != 0:
            if before is None:
                self.paths.overrides.unlink(missing_ok=True)
            else:
                atomic_write(self.paths.overrides, before)
        return r


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
    try:
        if args.once:
            return 0 if scaler.tick() in ("ok", "none", "dry-run") else 1
        while True:
            scaler.tick()
            if stopper.event.is_set() or wait(args.interval):
                break
    except KeyboardInterrupt:
        print(f"autoscaler: {CLEANUP_HINT}", file=sys.stderr)
        return 130
    scaler.say("autoscaler: stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
