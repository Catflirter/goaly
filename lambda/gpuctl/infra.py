"""Subprocess adapters for gpuctl: Terraform outputs and applies, gcloud IAP SSH to k3s-cp, kubectl, join-node.

Every command goes through an injectable Shell, so tests replay scripted results (tests/fakes.py) and nothing here
needs a cloud account to be exercised. Commands get argv lists, never a shell string.
"""

from __future__ import annotations

import ipaddress
import json
import re
import shlex
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
_ZONE_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_NODE_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_NOT_FOUND_RE = re.compile(r"\(NotFound\)|nodes? \"[^\"]*\" not found")
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
        if check and result.returncode != 0:
            raise CommandError(args, result.returncode, result.stderr)
        return result


def operator_ip(text: str) -> str:
    """The operator's public IP from OPERATOR_IP or the lookup service: one global unicast IPv4 address."""
    value = text.strip()
    return str(_global_ipv4(value, "operator IP"))


def _gpu_ip(value: object) -> str:
    # Review Focus #2: a booting instance's null IP (or anything that is not one public IPv4) never reaches Terraform.
    return str(_global_ipv4(value, "gpu_ips"))


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

    def _tf_env(self) -> dict[str, str]:
        env = dict(self.env)
        project = self._project()
        if project:
            env["TF_VAR_project_id"] = project
            env["TF_VAR_state_bucket"] = f"{project}-k3s-tfstate"
        return env

    def _stack_dir(self, stack: str) -> Path:
        if stack not in STACKS:
            raise ValueError(f"unknown Terraform stack {stack!r} (expected one of {sorted(STACKS)})")
        return self.repo_root / "infra" / stack

    def _terraform(self, stack: str, *args: str) -> subprocess.CompletedProcess[str]:
        return self.shell.run(["terraform", f"-chdir={self._stack_dir(stack)}", *args], env=self._tf_env())

    def ensure_init(self, stack: str) -> None:
        """terraform init the stack against its GCS state prefix, unless infra/<stack>/.terraform already exists."""
        if (self._stack_dir(stack) / ".terraform").exists():
            return
        project = self._project()
        if not project:
            raise ValueError(f"GCP_PROJECT is not set: cannot locate the state bucket to init infra/{stack}")
        self._terraform(stack, "init", "-input=false", f"-backend-config=bucket={project}-k3s-tfstate",
                        f"-backend-config=prefix={stack}")

    def tf_output(self, stack: str) -> dict:
        """The stack's outputs as {name: value}."""
        stdout = self._terraform(stack, "output", "-json").stdout
        try:
            raw = json.loads(stdout)
            if not isinstance(raw, dict):
                raise TypeError("not an object")
            return {name: item["value"] for name, item in raw.items()}
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"terraform output -json for infra/{stack}: unexpected output ({e})") from None

    def apply_gpu_access(self, gpu_ips: Sequence[str]) -> None:
        """Apply the gpu-access stack for exactly these GPU IPs (sorted, deduplicated; [] removes every rule)."""
        ips = sorted({ipaddress.IPv4Address(_gpu_ip(ip)) for ip in gpu_ips})
        value = json.dumps([str(ip) for ip in ips], separators=(",", ":"))
        self.ensure_init("gpu-access")
        self._terraform("gpu-access", "apply", "-auto-approve", "-input=false", f"-var=gpu_ips={value}")

    # --- k3s-cp over IAP ---

    def _zone(self) -> str:
        if self._cp_zone is None:
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
        return self.shell.run(args, input=input, env=dict(self.env)).stdout

    # --- kubectl ---

    def _kubeconfig(self) -> str:
        explicit = self.env.get("KUBECONFIG", "").strip()
        if explicit:
            return explicit
        home = self.env.get("HOME", "").strip()
        return str((Path(home) if home else Path.home()) / DEFAULT_KUBECONFIG)

    def kubectl(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.shell.run(["kubectl", "--kubeconfig", self._kubeconfig(), *args], check=check,
                              env=dict(self.env))

    def cluster_gpu_nodes(self) -> dict[str, str | None]:
        """GPU nodes (k3s-gpu-*) in the cluster, each with its ExternalIP (None if it reports none)."""
        try:
            items = json.loads(self.kubectl("get", "nodes", "-o", "json").stdout)["items"]
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

    def _kubectl_tolerating_not_found(self, *args: str) -> None:
        result = self.kubectl(*args, check=False)
        if result.returncode != 0 and not _NOT_FOUND_RE.search(result.stderr):
            raise CommandError(result.args, result.returncode, result.stderr)

    def drain_and_delete_node(self, name: str) -> None:
        """Drain then delete the node. A node that is already gone counts as success (Review Focus #3)."""
        self._kubectl_tolerating_not_found("drain", name, "--ignore-daemonsets", "--delete-emptydir-data",
                                           "--force", "--timeout=120s")
        self._kubectl_tolerating_not_found("delete", "node", name)

    # --- nodes ---

    def join_node(self, name: str, ip: str, gpu_type: str) -> None:
        """scripts/join-node.sh for a GPU node; the caller has already synced the peers (--skip-sync).

        The IP comes from the Lambda API and becomes an SSH target, so it must be one public IPv4 address.
        """
        if not _NODE_NAME_RE.fullmatch(name) or not _NODE_NAME_RE.fullmatch(gpu_type):
            raise ValueError(f"invalid node name {name!r} or GPU type {gpu_type!r}")
        ip = _gpu_ip(ip)
        self.shell.run([str(self.repo_root / "scripts" / "join-node.sh"), "--node", name, "--external-ip", ip,
                        "--gpu-type", gpu_type, "--skip-sync"], env=dict(self.env))

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
                                env=dict(self.env)).stdout
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
