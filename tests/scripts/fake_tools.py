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
  node_time      [time, ...]              successive answers to a remote command that reads the node's clock
                                          (`date -u +%Y-%m-%dT%H:%M:%SZ`; default 2026-10-22T10:00:00Z)
  ready          [node]                   `kubectl get node N -o jsonpath=...` prints True for these ...
  registered     [node]                   ... False for these, and NotFound (exit 1) for any other node
  api_down       true                     ... or fails like a dead tunnel, for every node
  kubectl_stderr text                     ... and, when it succeeds, also prints this on stderr (a
                                          warning or discovery error kubectl reports with exit 0)
  never_ready    [node]                   `kubectl wait ... node/N` times out (exit 1), and join-node's read
                                          of N's Ready status, heartbeat and ExternalIP says False
  external_ip    {node: ip}               N's ExternalIP (`kubectl get node N -o jsonpath=...ExternalIP...`
                                          alone or in join-node's read). Default: the --external-ip of N's
                                          node-bootstrap.sh run in this test, else none
  heartbeat      time                     the Ready heartbeat in join-node's read (default 2026-10-22T10:00:05Z,
                                          after the default node_time)
  gone           [node]                   `kubectl drain N` / `kubectl delete node N`: NotFound (exit 1)
  drain_fail     [node]                   `kubectl drain N` fails for another reason (exit 1)
  helm_rc        rc                       exit code of `helm upgrade` (default 0)
  sync_rc        rc                       exit code of `scripts/gpuctl sync-peers` (default 0)
  gpu_ip         text                     what `scripts/gpuctl ip NAME` prints
  vm_status      {name: status}           `gcloud compute instances describe NAME --format=value(status)`
                                          (default RUNNING)

A plan run with -var-file=FILE copies FILE's text to $FAKE_LOG.var-file, so a test can see what it planned.

The plan guardrail (stretch S3, cpu-pool.sh --guarded):

  changes        {stack: [resource_change]} `terraform show -json` prints exactly these resource_changes
                                          (instead of the `deletes` replaces plus one bucket create)
  outputs_after_apply {stack: {name: value}} terraform output answers from these once an apply of that stack
                                          ran (the workers a scale-up creates appear in worker_ips)
  tfguard_rc     rc                       `scripts/gpuctl tfguard` exits rc. Without it the fake runs the real
                                          `gpuctl tfguard` (lambda/gpuctl) on the plan it was piped. Its stdin
                                          is logged as "stdin" either way
  guard_show_rc  {stack: rc}              the second and later `terraform show` of that stack (the first is
                                          tf_plan's own check; the next is the guard's) print as usual, then
                                          exit rc
  guard_show_replaces true                 ... and that later show overwrites the saved plan with other bytes
                                          while it is read (another run's plan lands mid-guard)
  drain_plan     "keep" | "replace" | "remove"  `kubectl drain` stands in for another cpu-pool.sh run during the
                                          drain: it leaves the saved plan (the last plan -out file) alone,
                                          overwrites it with other bytes, or deletes it, and writes "held" or
                                          "free" (whether .run/workers.lock is flock(1)ed then) to $FAKE_LOG.lock_at_drain

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
                                          stalled call that is still running). "run_with": {TOOL: rc}
                                          runs the command after `--` (kubectl exec) for real, here, with
                                          only stubs of the named tools (each exits its rc) in front of
                                          /usr/bin:/bin, and exits with its status: the exec'd script's
                                          own logic is tested, not a canned exit code.

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
READYZ_PROBE = TOOL == "kubectl" and "--raw" in ARGS and "/readyz" in ARGS
LOGGED_ENV = ("TF_VAR_project_id", "TF_VAR_state_bucket", "AUTO_APPROVE", "LAMBDA_API_KEY", "GPU_SSH_IP")

TOKEN = "K10fakejointoken0123456789abcdef::server:s3cr3t-join"
INSTALL_TOKEN = "sudo install -m 600 /dev/stdin /etc/rancher/k3s/token && sudo systemctl enable --now k3s-agent"
CONTAINERD_GREP = "sudo grep -A3 nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml"
CONTAINERD_NVIDIA = "[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia']"
NODE_CLOCK = "date -u +%Y-%m-%dT%H:%M:%SZ"


def log():
    entry = {"tool": TOOL, "argv": ARGS, "env": {k: os.environ[k] for k in LOGGED_ENV if k in os.environ}}
    if TOOL == "gpuctl" and ARGS[:1] in (["down"], ["list-nodes"], ["tfguard"]):
        entry["stdin"] = sys.stdin.read()
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


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
        outputs = SCENARIO["outputs"]
        if Path(os.environ["FAKE_LOG"] + ".applied." + stack).exists():
            outputs = {**outputs, **SCENARIO.get("outputs_after_apply", {})}
        value = outputs[stack][rest[-1]]
        sys.stdout.write(json.dumps(value) if "-json" in rest else str(value))
        return 0
    if sub == "plan":
        out = next(a.split("=", 1)[1] for a in rest if a.startswith("-out="))
        Path(out).write_text("fake plan")
        Path(os.environ["FAKE_LOG"] + ".plan_out").write_text(out)
        for a in rest:
            if a.startswith("-var-file="):
                Path(os.environ["FAKE_LOG"] + ".var-file").write_text(Path(a.split("=", 1)[1]).read_text())
        return SCENARIO.get("plan_rc", {}).get(stack, 2)
    if sub == "show":
        shown = counter("show." + stack)
        if stack in SCENARIO.get("changes", {}):
            print(json.dumps({"format_version": "1.2", "planned_values": {},
                              "resource_changes": SCENARIO["changes"][stack]}))
            if shown and SCENARIO.get("guard_show_replaces"):
                Path(rest[-1]).write_text("another run's plan")
            return SCENARIO.get("guard_show_rc", {}).get(stack, 0) if shown else 0
        changes = [{"address": a, "change": {"actions": ["delete", "create"]}}
                   for a in SCENARIO.get("deletes", {}).get(stack, [])]
        changes.append({"address": "google_storage_bucket.x", "change": {"actions": ["create"]}})
        print(json.dumps({"resource_changes": changes}))
        return 0
    if sub == "apply":
        Path(os.environ["FAKE_LOG"] + ".applied." + stack).write_text("")
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


def bootstrapped():
    """{node: external IP} of every node-bootstrap.sh run in this test."""
    path = Path(os.environ["FAKE_LOG"] + ".bootstrapped")
    return json.loads(path.read_text()) if path.exists() else {}


def remote(target, command):
    """A command run on k3s-cp, a CPU worker or a GPU node."""
    failed = command in SCENARIO.get("ssh_fail", []) or (
        command == "true" and counter("ssh-true") < SCENARIO.get("ssh_failures", 0))
    if failed and "ssh_stderr" in SCENARIO:
        print(SCENARIO["ssh_stderr"], file=sys.stderr)
    if command in SCENARIO.get("ssh_fail", []):
        return SCENARIO.get("ssh_fail_rc", 1)
    if "node-bootstrap.sh" in command:
        name, ip = re.search(r"--node-name (\S+)", command), re.search(r"--external-ip (\S+)", command)
        if name and ip:
            Path(os.environ["FAKE_LOG"] + ".bootstrapped").write_text(
                json.dumps({**bootstrapped(), name.group(1): ip.group(1)}))
    if NODE_CLOCK in command:
        times = SCENARIO.get("node_time", ["2026-10-22T10:00:00Z"])
        sys.stdout.write(times[min(counter("node-time"), len(times) - 1)] + "\n")
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
        query = opt("-o") or ""
        if SCENARIO.get("api_down"):
            print("Unable to connect to the server: dial tcp 127.0.0.1:6443: connect: connection refused",
                  file=sys.stderr)
            return 1
        if SCENARIO.get("kubectl_stderr") and node in SCENARIO.get("ready", []) + SCENARIO.get("registered", []):
            print(SCENARIO["kubectl_stderr"], file=sys.stderr)
        if "ExternalIP" in query:
            # join-node's reads of the address (and, with lastHeartbeatTime, of the Ready condition): a node
            # bootstrapped in this test counts as registered.
            if node not in SCENARIO.get("ready", []) + SCENARIO.get("registered", []) + list(bootstrapped()):
                return not_found(node)
            ip = SCENARIO.get("external_ip", {}).get(node, bootstrapped().get(node, ""))
            if "lastHeartbeatTime" in query:
                status = "False" if node in SCENARIO.get("never_ready", []) else "True"
                sys.stdout.write(f"{status}|{SCENARIO.get('heartbeat', '2026-10-22T10:00:05Z')}|{ip}")
            else:
                sys.stdout.write(ip)
            return 0
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
        if args[0] == "drain" and "drain_plan" in SCENARIO:
            other_run()
        if args[0] == "drain" and node in SCENARIO.get("drain_fail", []):
            print("error: cannot evict pod as it would violate the pod's disruption budget", file=sys.stderr)
            return 1
    return 0


def other_run():
    """drain_plan: another cpu-pool.sh run touches the saved plan while this one drains."""
    plan = Path(Path(os.environ["FAKE_LOG"] + ".plan_out").read_text())
    lock = plan.parent / "workers.lock"
    free = subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0
    Path(os.environ["FAKE_LOG"] + ".lock_at_drain").write_text("free" if free else "held")
    if SCENARIO["drain_plan"] == "replace":
        plan.write_text("another run's plan")
    elif SCENARIO["drain_plan"] == "remove":
        plan.unlink()


def helm():
    return SCENARIO.get("helm_rc", 0) if "upgrade" in ARGS else 0


def tfguard(stdin):
    """The real `gpuctl tfguard` (it reads only stdin: no Lambda, no cluster), unless the scenario sets its exit code."""
    if "tfguard_rc" in SCENARIO:
        return SCENARIO["tfguard_rc"]
    import io
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lambda"))
    from gpuctl.__main__ import main
    sys.stdin = io.StringIO(stdin)
    return main(ARGS, env={})


def gpuctl(entry):
    if ARGS[:1] == ["tfguard"]:
        return tfguard(entry["stdin"])
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
        if "run_with" in answer:
            return run_with(answer["run_with"])
        return answer.get("rc", 0)
    return None


def run_with(stubs):
    """Run the command after `--` with stub tools STUBS {name: rc} first on PATH; its exit status."""
    stub_dir = Path(os.environ["FAKE_LOG"] + ".run_with")
    stub_dir.mkdir(exist_ok=True)
    for name, rc in stubs.items():
        stub = stub_dir / name
        stub.write_text(f"#!/bin/sh\nexit {int(rc)}\n")
        stub.chmod(0o755)
    command = ARGS[ARGS.index("--") + 1:]
    return subprocess.run(command, env={"PATH": f"{stub_dir}:/usr/bin:/bin"}).returncode


if __name__ == "__main__":
    ENTRY = None
    if not READYZ_PROBE:  # a /readyz probe logs itself once it has its answer (kubectl)
        ENTRY = log()
    rc = scripted()
    if rc is not None:
        if READYZ_PROBE:  # a scripted answer to a /readyz probe: log it now that it is answered
            log()
        sys.exit(rc)
    if TOOL == "gpuctl":  # never a /readyz probe, so ENTRY is set (gpuctl tfguard reads its logged stdin)
        sys.exit(gpuctl(ENTRY))
    sys.exit({"terraform": terraform, "gcloud": gcloud, "kubectl": kubectl, "helm": helm, "ssh": ssh,
              "scp": scp}[TOOL]())
