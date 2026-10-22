"""Stand-in for terraform, gcloud, kubectl, helm, ssh, scp and scripts/gpuctl in the offline tests of
scripts/ (never the real CLIs).

The stub bin/ directory holds one wrapper per tool, `exec python fake_tools.py <tool> "$@"` (the repo
copy's scripts/gpuctl is such a wrapper too). Every call appends one JSON line {"tool", "argv", "env"}
to $FAKE_LOG. $FAKE_SCENARIO (a JSON file) steers the answers:

  outputs        {stack: {name: value}}   terraform output -raw/-json
  init_rc        {stack: rc}              terraform init exit code (default 0). Like the real one, the fake
                                          writes .terraform/terraform.tfstate before it fails
  plan_rc        {stack: rc}              terraform plan -detailed-exitcode (default 2: changes)
  deletes        {stack: [address]}       resource addresses the plan deletes or replaces
  ssh_failures   N                        first N `gcloud compute ssh --command true` calls fail
  ssh_fail       [command]                a remote command C (any target) exits ssh_fail_rc for each listed C
  ssh_fail_rc    rc                       ... exit code of those calls (default 1: the remote command
                                          failed; 255 is ssh's own error, e.g. IAP dropped the connection)
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

Verification checks (Task 18) script their answers instead, with rules tried before every handler above:

  answers        [rule, ...]              the first rule whose "tool" is this tool, whose "has" substrings
                                          all occur in the call's arguments joined by spaces (for a remote
                                          command that includes the command) and none of whose "not"
                                          substrings occur, answers the call: it prints "stdout" and
                                          "stderr" and exits "rc" (default 0). With "seq": [answer, ...]
                                          successive calls get successive answers (the last one repeats).
                                          An answer with "serve": TEXT (kubectl port-forward L:R) listens on
                                          127.0.0.1:L, prints kubectl's "Forwarding from" line and answers
                                          every HTTP GET with TEXT until it is killed. "sleep": SECONDS
                                          waits that long after printing, before exiting (a capture or a
                                          stalled call that is still running).
"""
import http.server
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TOOL = sys.argv[1]
ARGS = sys.argv[2:]
SCENARIO = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text())
LOGGED_ENV = ("TF_VAR_project_id", "TF_VAR_state_bucket", "AUTO_APPROVE", "LAMBDA_API_KEY", "GPU_SSH_IP")

TOKEN = "K10fakejointoken0123456789abcdef::server:s3cr3t-join"
INSTALL_TOKEN = "sudo install -m 600 /dev/stdin /etc/rancher/k3s/token && sudo systemctl enable --now k3s-agent"
CONTAINERD_GREP = "sudo grep -A3 nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml"
CONTAINERD_NVIDIA = "[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia']"


def log():
    entry = {"tool": TOOL, "argv": ARGS, "env": {k: os.environ[k] for k in LOGGED_ENV if k in os.environ}}
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
    if command in SCENARIO.get("ssh_fail", []):
        return SCENARIO.get("ssh_fail_rc", 1)
    if command == "true":
        return 255 if counter("ssh-true") < SCENARIO.get("ssh_failures", 0) else 0
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
    if "--raw" in ARGS and "/readyz" in ARGS:
        n = counter("readyz")
        answer = SCENARIO.get("readyz", "ok")
        if answer != "ok" or n < SCENARIO.get("readyz_fail_first", 0) or not tunnel_running():
            return 1
        sys.stdout.write("ok")
        return 0
    args = ARGS[2:]  # after --kubeconfig FILE
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
    return 1


def serve(text):
    """kubectl port-forward LOCAL:REMOTE: answer HTTP GETs on 127.0.0.1:LOCAL until killed."""
    local, remote = next(a for a in ARGS if re.fullmatch(r"\d+:\d+", a)).split(":")
    body = text.encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server's name)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", int(local)), Handler)
    print(f"Forwarding from 127.0.0.1:{local} -> {remote}", flush=True)
    server.serve_forever()


def scripted():
    """The exit code of the first matching "answers" rule (see the module docstring), or None."""
    joined = " ".join(ARGS)
    for i, rule in enumerate(SCENARIO.get("answers", [])):
        if rule.get("tool") != TOOL or not all(s in joined for s in rule.get("has", [])):
            continue
        if any(s in joined for s in rule.get("not", [])):
            continue
        answer = rule
        if "seq" in rule:
            answer = rule["seq"][min(counter(f"answer-{i}"), len(rule["seq"]) - 1)]
        sys.stdout.write(answer.get("stdout", ""))
        sys.stdout.flush()
        sys.stderr.write(answer.get("stderr", ""))
        sys.stderr.flush()
        if "serve" in answer:
            serve(answer["serve"])
        time.sleep(answer.get("sleep", 0))
        return answer.get("rc", 0)
    return None


if __name__ == "__main__":
    log()
    rc = scripted()
    if rc is not None:
        sys.exit(rc)
    sys.exit({"terraform": terraform, "gcloud": gcloud, "kubectl": kubectl, "helm": helm, "ssh": ssh, "scp": scp,
              "gpuctl": gpuctl}[TOOL]())
