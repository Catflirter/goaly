"""Subprocess adapters for gpuctl: Terraform outputs and applies, gcloud IAP SSH to k3s-cp, kubectl, join-node.

Every command goes through an injectable Shell, so tests replay scripted results (tests/fakes.py) and nothing here
needs a cloud account to be exercised. Commands get argv lists, never a shell string.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from gpuctl.peers import NFT_SET, _global_ipv4
from gpuctl.planner import PREFIX

STACKS = frozenset({"base", "workers", "gpu-access"})  # the stacks gpuctl reads or applies (infra/<stack>)
CP_NAME = "k3s-cp"
DEFAULT_KUBECONFIG = ".kube/hybrid-gpu.yaml"  # under $HOME (global constraints)
OPERATOR_IP_URL = "https://checkip.amazonaws.com"
# Secrets no child process needs: terraform, gcloud, kubectl, curl and join-node.sh never use the Lambda API key.
CHILD_ENV_DENYLIST = frozenset({"LAMBDA_API_KEY"})
# Reads that can stall (IAP tunnel, GCS backend) are bounded. apply never is: a killed apply would orphan the state lock.
CP_SSH_TIMEOUT_S = 120.0
TF_READ_TIMEOUT_S = 300.0
# init downloads providers (the Google provider is large), so it gets longer. A failed or killed init removes the
# .terraform it created, so the next ensure_init runs init again instead of trusting a half-initialised stack.
TF_INIT_TIMEOUT_S = 900.0
# Written into infra/<stack>/.terraform only after an init against the GCS backend succeeded; it holds
# "gcs <bucket> <stack> lock <lock sum>", the lock sum being `cksum < .terraform.lock.hcl` ("<crc> <bytes>") or
# "no-lock-file" (init_marker_text). scripts/lib/common.sh tf_init writes and checks the same file and text, so
# the shell scripts and gpuctl agree on what an initialised stack is: neither re-initialises (-reconfigure) a stack
# the other set up, and both init again once a pulled provider bump changes the lock file (Task 10 X1-2, MK-1).
# A bare .terraform proves nothing: `terraform init -backend=false` (an offline validate) or a killed init leaves
# one without a usable backend.
TF_INIT_MARKER = "k3s-tf-init.ok"
TF_LOCK_FILE = ".terraform.lock.hcl"
_ZONE_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_NODE_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
# A GPU type label as gpu_type_label() returns it (a6000, a100_80gb_sxm4); also a valid K8s label value (<= 63).
_GPU_TYPE_RE = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")
_LABEL_VALUE_MAX = 63
_STDERR_TAIL = 2000


class CommandError(Exception):
    """A command could not be run or exited non-zero. `args` is its argv, as a tuple."""

    def __init__(self, args: Sequence[str], returncode: int, stderr: str) -> None:
        self.returncode = returncode
        self.stderr = stderr or ""
        detail = self.stderr.strip()[-_STDERR_TAIL:]
        self._message = f"{shlex.join(args)} failed with exit code {returncode}" + (f": {detail}" if detail else "")
        super().__init__(*args)

    def __str__(self) -> str:
        return self._message


class MissingStackError(CommandError):
    """infra/<stack> is not in this checkout, so there is nothing to init or read (unlike a stack that fails)."""


class Shell(Protocol):
    def run(self, args: Sequence[str], *, input: str | None = None, check: bool = True,
            timeout: float | None = None, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        ...


class SubprocessShell:
    """Shell backed by subprocess.run: text mode, stdout and stderr captured, no shell interpretation."""

    def run(self, args: Sequence[str], *, input: str | None = None, check: bool = True,
            timeout: float | None = None, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        args = [str(a) for a in args]
        try:
            result = subprocess.run(args, input=input, capture_output=True, text=True, timeout=timeout,
                                    env=None if env is None else dict(env), check=False)
        except FileNotFoundError:
            raise CommandError(args, 127, f"{args[0]}: command not found") from None
        except subprocess.TimeoutExpired:
            raise CommandError(args, -1, f"timed out after {timeout}s") from None
        except OSError as e:  # e.g. PermissionError: a script that is not executable
            raise CommandError(args, 126, f"{args[0]}: cannot run: {e.strerror or e}") from None
        if check and result.returncode != 0:
            raise CommandError(args, result.returncode, result.stderr)
        return result


def operator_ip(text: str) -> str:
    """The operator's public IP from OPERATOR_IP or the lookup service: one global unicast IPv4 address."""
    value = text.strip()
    return str(_global_ipv4(value, "operator IP"))


def gpu_ip(value: object) -> str:
    """One GPU node IP. Review Focus #2: a booting instance's null IP (or anything that is not one public IPv4
    address) never reaches Terraform or an SSH command line."""
    return str(_global_ipv4(value, "gpu_ips"))


def normalise_gpu_ips(gpu_ips: Sequence[object]) -> list[str]:
    """The gpu-access stack's gpu_ips: every IP validated, then deduplicated and sorted numerically."""
    return [str(ip) for ip in sorted({ipaddress.IPv4Address(gpu_ip(ip)) for ip in gpu_ips})]


def check_max_nodes(value: object) -> int:
    """The gpu-access stack's max_nodes (Ruling R27): spec max_nodes, a plain integer >= 1."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"max_nodes must be an integer >= 1, got {value!r}")
    return value


def check_join_args(name: object, ip: object, gpu_type: object) -> str:
    """Validate join_node's arguments (they reach a command line and a node label) and return the IP."""
    if not isinstance(name, str) or not _NODE_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid node name {name!r}")
    if not isinstance(gpu_type, str) or len(gpu_type) > _LABEL_VALUE_MAX or not _GPU_TYPE_RE.fullmatch(gpu_type):
        raise ValueError(f"invalid GPU type {gpu_type!r}")
    return gpu_ip(ip)


def _cksum_table() -> tuple[int, ...]:
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = (crc << 1) ^ 0x04C11DB7 if crc & 0x80000000 else crc << 1
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)


_CKSUM_TABLE = _cksum_table()


def posix_cksum(data: bytes) -> str:
    """What `cksum < FILE` prints for FILE's bytes, without the newline: "<crc> <bytes>".

    POSIX cksum: CRC-32 with polynomial 0x04C11DB7, MSB first, initial value 0, over the data followed by its
    length (least significant byte first, only as many bytes as the length needs), then complemented.
    GNU, BSD/macOS and uutils cksum agree on it, so gpuctl and scripts/lib/common.sh tf_init compute the same text.
    """
    crc = 0
    table = _CKSUM_TABLE
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[(crc >> 24) ^ byte]
    length = len(data)
    while length:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[(crc >> 24) ^ (length & 0xFF)]
        length >>= 8
    return f"{~crc & 0xFFFFFFFF} {len(data)}"


def init_marker_text(bucket: str, stack: str, stack_dir: Path) -> str:
    """The TF_INIT_MARKER text for `stack` initialised against `bucket` with its current lock file, exactly as
    scripts/lib/common.sh tf_init builds it: "gcs <bucket> <stack> lock <cksum of the lock file | no-lock-file>"."""
    lock_file = stack_dir / TF_LOCK_FILE
    lock_sum = posix_cksum(lock_file.read_bytes()) if lock_file.is_file() else "no-lock-file"
    return f"gcs {bucket} {stack} lock {lock_sum}"


class Infra:
    """Terraform, IAP SSH to k3s-cp, kubectl and join-node for one repo checkout.

    `env` is the full environment the commands run with (normally os.environ, with .env loaded by scripts/gpuctl).
    Terraform also gets TF_VAR_project_id and TF_VAR_state_bucket derived from GCP_PROJECT, so a direct call that
    bypasses scripts/gpuctl still configures every stack.
    """

    def __init__(self, shell: Shell, repo_root: Path, env: Mapping[str, str]) -> None:
        self.shell = shell
        self.repo_root = Path(repo_root)
        self.env = env
        self._cp_zone: str | None = None

    # --- terraform ---

    def _project(self) -> str:
        return self.env.get("GCP_PROJECT", "").strip()

    def _child_env(self) -> dict[str, str]:
        """The environment every child process gets: self.env without the secrets none of them needs."""
        return {k: v for k, v in self.env.items() if k not in CHILD_ENV_DENYLIST}

    def _tf_env(self) -> dict[str, str]:
        env = self._child_env()
        project = self._project()
        if project:
            env["TF_VAR_project_id"] = project
            env["TF_VAR_state_bucket"] = f"{project}-k3s-tfstate"
        return env

    def _stack_dir(self, stack: str) -> Path:
        if stack not in STACKS:
            raise ValueError(f"unknown Terraform stack {stack!r} (expected one of {sorted(STACKS)})")
        return self.repo_root / "infra" / stack

    def _terraform(self, stack: str, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        # -no-color right after the subcommand: a CommandError carries terraform's stderr to the operator, and
        # colour codes would reach the terminal as ?[31m junk.
        command, *rest = args
        return self.shell.run(["terraform", f"-chdir={self._stack_dir(stack)}", command, "-no-color", *rest],
                              env=self._tf_env(), timeout=timeout)

    def ensure_init(self, stack: str) -> None:
        """terraform init the stack against its GCS state prefix, unless its last init did exactly that.

        Initialised means .terraform/k3s-tf-init.ok (TF_INIT_MARKER) reads
        "gcs <GCP_PROJECT>-k3s-tfstate <stack> lock <cksum of .terraform.lock.hcl | no-lock-file>"
        (init_marker_text), the marker scripts/lib/common.sh tf_init writes and checks too. A .terraform without it
        (an offline `terraform init -backend=false`, an interrupted init, a marker for another project's bucket, a
        lock file changed by a pulled provider bump) is initialised again with -reconfigure, which switches the
        backend without copying any old state into it.
        A stack that is not in this checkout raises MissingStackError (a CommandError). If init fails or is
        interrupted (a timeout included), .terraform is removed, so nothing half-initialised is left behind.
        -lockfile=readonly (Ruling R26(b)): providers are verified against the committed .terraform.lock.hcl, which
        init never rewrites, so even the read-only `gpuctl status` cannot change a tracked file.
        """
        stack_dir = self._stack_dir(stack)
        if not stack_dir.is_dir():
            raise MissingStackError(["terraform", f"-chdir={stack_dir}", "init"], 1,
                                    f"infra/{stack} does not exist in this checkout")
        project = self._project()
        if not project:
            raise ValueError(f"GCP_PROJECT is not set: cannot locate the state bucket to init infra/{stack}")
        bucket = f"{project}-k3s-tfstate"
        dot_terraform = stack_dir / ".terraform"
        marker = dot_terraform / TF_INIT_MARKER
        # Read before init, as tf_init does; -lockfile=readonly means init cannot change the lock file anyway.
        want = init_marker_text(bucket, stack, stack_dir)
        try:
            if marker.read_text().rstrip("\n") == want:
                return
        except OSError:  # no marker (or no .terraform): not initialised
            pass
        try:
            marker.unlink(missing_ok=True)
            self._terraform(stack, "init", "-input=false", "-lockfile=readonly", "-reconfigure",
                            f"-backend-config=bucket={bucket}", f"-backend-config=prefix={stack}",
                            timeout=TF_INIT_TIMEOUT_S)
            dot_terraform.mkdir(exist_ok=True)
            marker.write_text(want + "\n")
        except BaseException:  # CommandError, a timeout, or Ctrl-C mid-download
            shutil.rmtree(dot_terraform, ignore_errors=True)
            raise

    def tf_output(self, stack: str) -> dict:
        """The stack's outputs as {name: value}. Reading base also caches a valid cp_zone for cp_ssh."""
        stdout = self._terraform(stack, "output", "-json", timeout=TF_READ_TIMEOUT_S).stdout
        try:
            raw = json.loads(stdout)
            if not isinstance(raw, dict):
                raise TypeError("not an object")
            outputs = {name: item["value"] for name, item in raw.items()}
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"terraform output -json for infra/{stack}: unexpected output ({e})") from None
        zone = outputs.get("cp_zone")
        if stack == "base" and isinstance(zone, str) and _ZONE_RE.fullmatch(zone):
            self._cp_zone = zone
        return outputs

    def apply_gpu_access(self, gpu_ips: Sequence[str], *, max_nodes: int) -> None:
        """Apply the gpu-access stack for exactly these GPU IPs (sorted, deduplicated; [] removes every rule).

        `max_nodes` is the spec's max_nodes (Ruling R27): the stack refuses more IPs than its own copy allows, so
        a spec that raises the cap would otherwise launch nodes the firewall never admits.
        """
        value = json.dumps(normalise_gpu_ips(gpu_ips), separators=(",", ":"))
        max_nodes = check_max_nodes(max_nodes)
        self.ensure_init("gpu-access")
        self._terraform("gpu-access", "apply", "-auto-approve", "-input=false", f"-var=max_nodes={max_nodes}",
                        f"-var=gpu_ips={value}")

    # --- k3s-cp over IAP ---

    def _zone(self) -> str:
        if self._cp_zone is None:
            self.ensure_init("base")  # a fresh clone or worktree has never initialised infra/base
            zone = self.tf_output("base").get("cp_zone")
            if not isinstance(zone, str) or not _ZONE_RE.fullmatch(zone):
                raise ValueError(f"infra/base output cp_zone is missing or invalid: {zone!r}")
            self._cp_zone = zone
        return self._cp_zone

    def cp_ssh(self, command: str, input: str | None = None) -> str:
        """Run `command` on k3s-cp through IAP SSH and return its stdout. `input` is sent on stdin."""
        args = ["gcloud", "compute", "ssh", CP_NAME]
        if self._project():
            args += ["--project", self._project()]
        args += ["--zone", self._zone(), "--tunnel-through-iap", "--quiet", "--command", command]
        return self.shell.run(args, input=input, timeout=CP_SSH_TIMEOUT_S, env=self._child_env()).stdout

    # --- kubectl ---

    def _kubeconfig(self) -> str:
        # scripts/gpuctl pins KUBECONFIG to ~/.kube/hybrid-gpu.yaml unless .env sets it (spec section 11), so an
        # ambient kubeconfig for another cluster does not reach here through the wrapper.
        explicit = self.env.get("KUBECONFIG", "").strip()
        if explicit:
            if os.pathsep in explicit:
                raise ValueError(f"KUBECONFIG must name one file, not a list: {explicit!r}")
            return explicit
        home = self.env.get("HOME", "").strip()
        return str((Path(home) if home else Path.home()) / DEFAULT_KUBECONFIG)

    def kubectl(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.shell.run(["kubectl", "--kubeconfig", self._kubeconfig(), *args], check=check,
                              env=self._child_env())

    def cluster_gpu_nodes(self) -> dict[str, str | None]:
        """GPU nodes (k3s-gpu-*) in the cluster, each with its ExternalIP (None if it reports none)."""
        # kubectl runs outside the try: its own errors (CommandError, a KUBECONFIG list) surface unchanged.
        stdout = self.kubectl("get", "nodes", "-o", "json").stdout
        try:
            items = json.loads(stdout)["items"]
            nodes: dict[str, str | None] = {}
            for item in items:
                name = item["metadata"]["name"]
                if not name.startswith(PREFIX):
                    continue
                external = [a["address"] for a in item.get("status", {}).get("addresses", [])
                            if a.get("type") == "ExternalIP"]
                nodes[name] = external[0] if external else None
            return nodes
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            raise ValueError(f"kubectl get nodes -o json: unexpected output ({e!r})") from None

    def _kubectl_tolerating_node_gone(self, name: str, *args: str) -> None:
        result = self.kubectl(*args, check=False)
        # Only "this node is gone" counts as success. A generic NotFound ("the server could not find the requested
        # resource", e.g. a stale tunnel to another server) or another node's name still fails.
        gone = re.compile(rf'\bnodes? "{re.escape(name)}" not found')
        if result.returncode != 0 and not gone.search(result.stderr):
            raise CommandError(result.args, result.returncode, result.stderr)

    def drain_and_delete_node(self, name: str) -> None:
        """Drain then delete the node. A node that is already gone counts as success (Review Focus #3)."""
        self._kubectl_tolerating_node_gone(name, "drain", name, "--ignore-daemonsets", "--delete-emptydir-data",
                                           "--force", "--timeout=120s")
        self._kubectl_tolerating_node_gone(name, "delete", "node", name)

    # --- nodes ---

    def join_node(self, name: str, ip: str, gpu_type: str) -> None:
        """scripts/join-node.sh for a GPU node; the caller has already synced the peers (--skip-sync).

        The IP comes from the Lambda API and becomes an SSH target, so it must be one public IPv4 address.
        """
        ip = check_join_args(name, ip, gpu_type)
        self.shell.run([str(self.repo_root / "scripts" / "join-node.sh"), "--node", name, "--external-ip", ip,
                        "--gpu-type", gpu_type, "--skip-sync"], env=self._child_env())

    def forget_host_key(self, name: str) -> None:
        """Delete .run/known_hosts_<name>: the next instance with this name has a new host key."""
        if not _NODE_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid node name {name!r}")
        (self.repo_root / ".run" / f"known_hosts_{name}").unlink(missing_ok=True)

    def detect_operator_ip(self) -> str:
        """OPERATOR_IP if set, else the public IP checkip.amazonaws.com sees. Must be a global IPv4 address."""
        override = self.env.get("OPERATOR_IP", "").strip()
        if override:
            return operator_ip(override)
        stdout = self.shell.run(["curl", "-fsS", "--max-time", "5", OPERATOR_IP_URL], timeout=10,
                                env=self._child_env()).stdout
        return operator_ip(stdout)

    def cp_guard_elements(self) -> list[str]:
        """The addresses currently in k3s-cp's k3s_peers nft set, sorted."""
        stdout = self.cp_ssh(f"sudo nft -j list set {NFT_SET}")
        try:
            sets = [entry["set"] for entry in json.loads(stdout)["nftables"] if "set" in entry]
            if len(sets) != 1:
                raise ValueError(f"expected one set, got {len(sets)}")
            elements = sets[0].get("elem", [])
            if not isinstance(elements, list) or not all(isinstance(e, str) for e in elements):
                raise ValueError("elements are not plain addresses")
            return [str(ip) for ip in sorted(ipaddress.IPv4Address(e) for e in elements)]
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"nft -j list set {NFT_SET}: unexpected output ({e})") from None
