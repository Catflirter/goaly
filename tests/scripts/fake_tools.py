"""Stand-in for terraform, gcloud and kubectl in the offline tests of scripts/ (never the real CLIs).

The stub bin/ directory holds one wrapper per tool, `exec python fake_tools.py <tool> "$@"`.
Every call appends one JSON line {"tool", "argv", "env"} to $FAKE_LOG. $FAKE_SCENARIO (a JSON
file) steers the answers:

  outputs        {stack: {name: value}}   terraform output -raw/-json
  init_rc        {stack: rc}              terraform init exit code (default 0). Like the real one, the fake
                                          writes .terraform/terraform.tfstate before it fails
  plan_rc        {stack: rc}              terraform plan -detailed-exitcode (default 2: changes)
  deletes        {stack: [address]}       resource addresses the plan deletes or replaces
  ssh_failures   N                        first N `gcloud compute ssh --command true` calls fail
  ssh_fail       [command]                `gcloud compute ssh --command C` exits ssh_fail_rc for each listed C
  ssh_fail_rc    rc                       ... exit code of those calls (default 1: the remote command
                                          failed; 255 is ssh's own error, e.g. IAP dropped the connection)
  kubeconfig     text                     answer to `sudo cat /etc/rancher/k3s/k3s.yaml`
  readyz         "ok" | anything else     kubectl get --raw /readyz (anything else: exit 1). Like the
                                          real API behind the tunnel, it answers ok only while a fake
                                          tunnel runs (its PID is in $FAKE_LOG.tunnel)
  readyz_fail_first N                     ... and the first N /readyz calls fail
  tunnel         "stay" | "exit"          the backgrounded `gcloud compute ssh ... -N -L` tunnel
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOL = sys.argv[1]
ARGS = sys.argv[2:]
SCENARIO = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text())
LOGGED_ENV = ("TF_VAR_project_id", "TF_VAR_state_bucket", "AUTO_APPROVE", "LAMBDA_API_KEY")


def log():
    entry = {"tool": TOOL, "argv": ARGS, "env": {k: os.environ[k] for k in LOGGED_ENV if k in os.environ}}
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps(entry) + "\n")


def opt(name):
    """Value of --name=VALUE or --name VALUE."""
    for i, a in enumerate(ARGS):
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
        if a == name and i + 1 < len(ARGS):
            return ARGS[i + 1]
    return None


def terraform():
    chdir = Path(ARGS[0].split("=", 1)[1])
    stack, sub, rest = chdir.name, ARGS[1], ARGS[2:]
    if sub == "init":
        cfg = dict(a.split("=", 2)[1:] for a in rest if a.startswith("-backend-config="))
        if cfg:
            state = chdir / ".terraform/terraform.tfstate"
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(json.dumps({"backend": {"type": "gcs", "config": cfg}}))
        return SCENARIO.get("init_rc", {}).get(stack, 0)
    if sub == "output":
        value = SCENARIO["outputs"][stack][rest[-1]]
        sys.stdout.write(json.dumps(value) if "-json" in rest else str(value))
        return 0
    if sub == "plan":
        out = next(a.split("=", 1)[1] for a in rest if a.startswith("-out="))
        Path(out).write_text("fake plan")
        return SCENARIO.get("plan_rc", {}).get(stack, 2)
    if sub == "show":
        changes = [{"address": a, "change": {"actions": ["delete", "create"]}}
                   for a in SCENARIO.get("deletes", {}).get(stack, [])]
        changes.append({"address": "google_storage_bucket.x", "change": {"actions": ["create"]}})
        print(json.dumps({"resource_changes": changes}))
        return 0
    if sub == "apply":
        return 0
    return 1


def gcloud():
    if ARGS[:2] == ["compute", "scp"]:
        return 0
    if ARGS[:2] != ["compute", "ssh"]:
        return 1
    command = opt("--command")
    if command is None:  # the tunnel
        if SCENARIO.get("tunnel", "stay") == "exit":
            print("bind [127.0.0.1]:6443: Address already in use", file=sys.stderr)
            return 255
        Path(os.environ["FAKE_LOG"] + ".tunnel").write_text(str(os.getpid()))
        time.sleep(300)
        return 0
    if command in SCENARIO.get("ssh_fail", []):
        return SCENARIO.get("ssh_fail_rc", 1)
    if command == "true":
        counter = Path(os.environ["FAKE_LOG"] + ".ssh-true")
        n = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(n + 1))
        return 255 if n < SCENARIO.get("ssh_failures", 0) else 0
    if command == "sudo cat /etc/rancher/k3s/k3s.yaml":
        sys.stdout.write(SCENARIO.get("kubeconfig", ""))
    return 0


def tunnel_running():
    marker = Path(os.environ["FAKE_LOG"] + ".tunnel")
    if not marker.exists():
        return False
    state = subprocess.run(["ps", "-o", "stat=", "-p", marker.read_text().strip()], capture_output=True, text=True)
    return state.returncode == 0 and not state.stdout.strip().startswith("Z")


def kubectl():
    if "--raw" in ARGS and "/readyz" in ARGS:
        counter = Path(os.environ["FAKE_LOG"] + ".readyz")
        n = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(n + 1))
        answer = SCENARIO.get("readyz", "ok")
        if answer != "ok" or n < SCENARIO.get("readyz_fail_first", 0) or not tunnel_running():
            return 1
        sys.stdout.write("ok")
    return 0


if __name__ == "__main__":
    log()
    sys.exit({"terraform": terraform, "gcloud": gcloud, "kubectl": kubectl}[TOOL]())
