"""Stand-in for terraform, gcloud, kubectl, helm, ssh, scp and scripts/gpuctl in the offline tests of
scripts/ (never the real CLIs).

The stub bin/ directory holds one wrapper per tool, `exec python fake_tools.py <tool> "$@"` (the repo
copy's scripts/gpuctl is such a wrapper too). Every call appends one JSON line {"tool", "argv", "env"}
to $FAKE_LOG; a /readyz probe does so only once it has its answer, so a probe that found the tunnel up
is always logged after the tunnel's own call (that tunnel runs concurrently, in the background).
$FAKE_SCENARIO (a JSON file) steers the answers:

  outputs        {stack: {name: value}}   terraform output -raw/-json
  init_rc        {stack: rc}              terraform init exit code (default 0). Like the real one, the fake
                                          writes .terraform/terraform.tfstate before it fails
  plan_rc        {stack: rc}              terraform plan -detailed-exitcode (default 2: changes)
  deletes        {stack: [address]}       resource addresses the plan deletes or replaces
  ssh_failures   N                        first N `gcloud compute ssh --command true` calls fail
  ssh_fail       [command]                a remote command C (any target) exits ssh_fail_rc for each listed C
  ssh_fail_rc    rc                       ... exit code of those calls (default 1: the remote command
                                          failed; 255 is ssh's own error, e.g. IAP dropped the connection)
  ssh_stderr     text                     ... and each failing call above (ssh_failures, ssh_fail) prints this
                                          on stderr, as ssh does (e.g. REMOTE HOST IDENTIFICATION HAS CHANGED)
  kubeconfig     text                     answer to `sudo cat /etc/rancher/k3s/k3s.yaml`
  readyz         "ok" | anything else     kubectl get --raw /readyz (anything else: exit 1). Like the
                                          real API behind the tunnel, it answers ok only while a fake
                                          tunnel runs (its PID is in $FAKE_LOG.tunnel)
  readyz_fail_first N                     ... and the first N /readyz calls fail
  tunnel         "stay" | "exit"          the backgrounded `gcloud compute ssh ... -N -L` tunnel

Workers and GPU nodes (Task 14). A remote command reaches the same handler whether it comes through
`gcloud compute ssh NAME --command C` (k3s-cp, CPU workers) or `ssh ... ubuntu@IP C` (GPU nodes):

  token          text                     what `sudo k3s token create ...` prints (the join token)
  install_rc     rc                       exit code of the token install command. Its stdin (the token)
                                          is appended to $FAKE_LOG.stdin as {"target", "stdin"}, never
                                          to $FAKE_LOG, so a test can prove the token came through a pipe
  containerd     [text, ...]              successive answers to the containerd nvidia grep ("" = no match, exit 1)
  ready          [node]                   `kubectl get node N -o jsonpath=...` prints True for these ...
  registered     [node]                   ... False for these, and NotFound (exit 1) for any other node
  api_down       true                     ... or fails like a dead tunnel, for every node
  kubectl_stderr text                     ... and, when it succeeds, also prints this on stderr (a
                                          warning or discovery error kubectl reports with exit 0)
  never_ready    [node]                   `kubectl wait ... node/N` times out (exit 1)
  gone           [node]                   `kubectl drain N` / `kubectl delete node N`: NotFound (exit 1)
  drain_fail     [node]                   `kubectl drain N` fails for another reason (exit 1)
  helm_rc        rc                       exit code of `helm upgrade` (default 0)
  sync_rc        rc                       exit code of `scripts/gpuctl sync-peers` (default 0)
  gpu_ip         text                     what `scripts/gpuctl ip NAME` prints
  vm_status      {name: status}           `gcloud compute instances describe NAME --format=value(status)`
                                          (default RUNNING)

A plan run with -var-file=FILE copies FILE's text to $FAKE_LOG.var-file, so a test can see what it planned.

Teardown (Task 22a, scripts/down.sh):

  destroy_rc     {stack: rc}              terraform destroy exit code (default 0)
  nodes          [node]                   `kubectl get nodes -o name` lists these (default k3s-cp, k3s-cpu-1);
                                          it fails like a dead tunnel under api_down
  kubectl_hang   true                     every kubectl call except /readyz hangs (a half-open tunnel)
  cp_node_ip     ip                       `kubectl get node k3s-cp -o jsonpath=...ExternalIP...` prints it
                                          (default: the base stack's cp_ip output)
  gpu_down_rc    rc                       exit code of `scripts/gpuctl down` (default 0)
  gpu_down_hang  true                     `scripts/gpuctl down` hangs; Ctrl-C ends it with exit 1, as the real
                                          gpuctl does (it catches KeyboardInterrupt)
  gpu_nodes      [line]                   what `scripts/gpuctl list-nodes` prints, one line each (default none)
  late_gpu_nodes [line]                   ... and these too, from its second call on (a node started meanwhile)
  gpu_nodes_then [line]                   from its second call on, printed instead of gpu_nodes (a node Lambda
                                          still listed as active right after its terminate call)
  list_nodes_rc  rc                       exit code of `scripts/gpuctl list-nodes` (default 0)
  list_nodes_sleep s                      `scripts/gpuctl list-nodes` takes this long (seconds; a slow Lambda API)

`scripts/gpuctl down` and `list-nodes` log what they read on stdin as "stdin" (down.sh gives them /dev/null).
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
READYZ_PROBE = TOOL == "kubectl" and "--raw" in ARGS and "/readyz" in ARGS
LOGGED_ENV = ("TF_VAR_project_id", "TF_VAR_state_bucket", "AUTO_APPROVE", "LAMBDA_API_KEY", "GPU_SSH_IP")

TOKEN = "K10fakejointoken0123456789abcdef::server:s3cr3t-join"
INSTALL_TOKEN = "sudo install -m 600 /dev/stdin /etc/rancher/k3s/token && sudo systemctl enable --now k3s-agent"
CONTAINERD_GREP = "sudo grep -A3 nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml"
CONTAINERD_NVIDIA = "[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia']"


def log():
    entry = {"tool": TOOL, "argv": ARGS, "env": {k: os.environ[k] for k in LOGGED_ENV if k in os.environ}}
    if TOOL == "gpuctl" and ARGS[:1] in (["down"], ["list-nodes"]):
        entry["stdin"] = sys.stdin.read()
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps(entry) + "\n")


def counter(name):
    """How often this was asked before in this test: 0, 1, 2, ..."""
    path = Path(os.environ["FAKE_LOG"] + "." + name)
    n = int(path.read_text()) if path.exists() else 0
    path.write_text(str(n + 1))
    return n


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
        for a in rest:
            if a.startswith("-var-file="):
                Path(os.environ["FAKE_LOG"] + ".var-file").write_text(Path(a.split("=", 1)[1]).read_text())
        return SCENARIO.get("plan_rc", {}).get(stack, 2)
    if sub == "show":
        changes = [{"address": a, "change": {"actions": ["delete", "create"]}}
                   for a in SCENARIO.get("deletes", {}).get(stack, [])]
        changes.append({"address": "google_storage_bucket.x", "change": {"actions": ["create"]}})
        print(json.dumps({"resource_changes": changes}))
        return 0
    if sub == "apply":
        return 0
    if sub == "destroy":
        return SCENARIO.get("destroy_rc", {}).get(stack, 0)
    return 1


def gcloud():
    if ARGS[:2] == ["compute", "scp"]:
        return 0
    if ARGS[:2] == ["compute", "instances"] and ARGS[2] in ("stop", "start"):
        return 0
    if ARGS[:3] == ["compute", "instances", "describe"] and opt("--format") == "value(status)":
        print(SCENARIO.get("vm_status", {}).get(ARGS[3], "RUNNING"))
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
    return remote(ARGS[2], command)


def ssh():
    """ssh [options] ubuntu@IP COMMAND (gpu_ssh): the target and the command are the last two arguments."""
    return remote(ARGS[-2], ARGS[-1])


def scp():
    return 0


def remote(target, command):
    """A command run on k3s-cp, a CPU worker or a GPU node."""
    failed = command in SCENARIO.get("ssh_fail", []) or (
        command == "true" and counter("ssh-true") < SCENARIO.get("ssh_failures", 0))
    if failed and "ssh_stderr" in SCENARIO:
        print(SCENARIO["ssh_stderr"], file=sys.stderr)
    if command in SCENARIO.get("ssh_fail", []):
        return SCENARIO.get("ssh_fail_rc", 1)
    if command == "true":
        return 255 if failed else 0
    if command == "sudo cat /etc/rancher/k3s/k3s.yaml":
        sys.stdout.write(SCENARIO.get("kubeconfig", ""))
    if command.startswith("sudo k3s token create "):
        sys.stdout.write(SCENARIO.get("token", TOKEN) + "\n")
    if command == INSTALL_TOKEN:
        with open(os.environ["FAKE_LOG"] + ".stdin", "a") as f:
            f.write(json.dumps({"target": target, "stdin": sys.stdin.read()}) + "\n")
        return SCENARIO.get("install_rc", 0)
    if command == CONTAINERD_GREP:
        answers = SCENARIO.get("containerd", [CONTAINERD_NVIDIA])
        answer = answers[min(counter("containerd"), len(answers) - 1)]
        sys.stdout.write(answer + "\n" if answer else "")
        return 0 if answer else 1
    return 0


def tunnel_running():
    marker = Path(os.environ["FAKE_LOG"] + ".tunnel")
    if not marker.exists():
        return False
    state = subprocess.run(["ps", "-o", "stat=", "-p", marker.read_text().strip()], capture_output=True, text=True)
    return state.returncode == 0 and not state.stdout.strip().startswith("Z")


def not_found(node):
    print(f'Error from server (NotFound): nodes "{node}" not found', file=sys.stderr)
    return 1


def kubectl():
    if READYZ_PROBE:
        n = counter("readyz")
        answer = SCENARIO.get("readyz", "ok")
        up = answer == "ok" and n >= SCENARIO.get("readyz_fail_first", 0) and tunnel_running()
        # Logged after the answer: the tunnel logs its call before it writes $FAKE_LOG.tunnel, so a probe
        # that saw the tunnel up comes after the tunnel in $FAKE_LOG however the two were scheduled.
        log()
        if not up:
            return 1
        sys.stdout.write("ok")
        return 0
    args = ARGS[2:]  # after --kubeconfig FILE
    if SCENARIO.get("kubectl_hang"):
        time.sleep(300)
        return 1
    if args[:2] == ["get", "nodes"]:
        if SCENARIO.get("api_down"):
            print("Unable to connect to the server: dial tcp 127.0.0.1:6443: connect: connection refused",
                  file=sys.stderr)
            return 1
        for node in SCENARIO.get("nodes", ["k3s-cp", "k3s-cpu-1"]):
            print(f"node/{node}")
        return 0
    if args[:3] == ["get", "node", "k3s-cp"] and any("ExternalIP" in a for a in args):
        sys.stdout.write(SCENARIO.get("cp_node_ip", SCENARIO["outputs"]["base"]["cp_ip"]))
        return 0
    if args[:2] == ["get", "node"]:
        node = args[2]
        if SCENARIO.get("api_down"):
            print("Unable to connect to the server: dial tcp 127.0.0.1:6443: connect: connection refused",
                  file=sys.stderr)
            return 1
        if SCENARIO.get("kubectl_stderr") and node in SCENARIO.get("ready", []) + SCENARIO.get("registered", []):
            print(SCENARIO["kubectl_stderr"], file=sys.stderr)
        if node in SCENARIO.get("ready", []):
            sys.stdout.write("True")
        elif node in SCENARIO.get("registered", []):
            sys.stdout.write("False")
        else:
            return not_found(node)
        return 0
    if args[:1] == ["wait"] and len(args) > 2 and args[2].startswith("node/"):
        if args[2].split("/", 1)[1] in SCENARIO.get("never_ready", []):
            print("error: timed out waiting for the condition", file=sys.stderr)
            return 1
        return 0
    node = args[1] if args[:1] == ["drain"] else args[2] if args[:2] == ["delete", "node"] else None
    if node is not None:
        if node in SCENARIO.get("gone", []):
            return not_found(node)
        if args[0] == "drain" and node in SCENARIO.get("drain_fail", []):
            print("error: cannot evict pod as it would violate the pod's disruption budget", file=sys.stderr)
            return 1
    return 0


def helm():
    return SCENARIO.get("helm_rc", 0) if "upgrade" in ARGS else 0


def gpuctl():
    if ARGS[:1] == ["sync-peers"]:
        return SCENARIO.get("sync_rc", 0)
    if ARGS[:1] == ["ip"]:
        sys.stdout.write(SCENARIO.get("gpu_ip", "") + "\n")
        return 0
    if ARGS[:1] == ["down"]:
        if SCENARIO.get("gpu_down_hang"):
            try:
                time.sleep(300)
            except KeyboardInterrupt:
                return 1
        return SCENARIO.get("gpu_down_rc", 0)
    if ARGS[:1] == ["list-nodes"]:
        time.sleep(SCENARIO.get("list_nodes_sleep", 0))
        later = counter("list-nodes") > 0
        late = SCENARIO.get("late_gpu_nodes", []) if later else []
        nodes = SCENARIO["gpu_nodes_then"] if later and "gpu_nodes_then" in SCENARIO else SCENARIO.get("gpu_nodes", [])
        for line in nodes + late:
            print(line)
        return SCENARIO.get("list_nodes_rc", 0)
    return 1


if __name__ == "__main__":
    if not READYZ_PROBE:  # a /readyz probe logs itself once it has its answer (kubectl)
        log()
    sys.exit({"terraform": terraform, "gcloud": gcloud, "kubectl": kubectl, "helm": helm, "ssh": ssh, "scp": scp,
              "gpuctl": gpuctl}[TOOL]())
