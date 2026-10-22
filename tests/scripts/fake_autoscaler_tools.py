"""Stand-in for kubectl, scripts/gpuctl and scripts/cpu-pool.sh in the subprocess tests of scripts/autoscaler.py
(tests/scripts/test_autoscaler_cli.py); never the real tools.

Each stub (`exec python fake_autoscaler_tools.py <tool> "$@"`) appends one JSON line {"tool", "argv",
"auto_approve", "state", "overrides"} to $FAKE_LOG: `state` and `overrides` are the text of the autoscaler's state
file ($AUTOSCALER_STATE) and of lambda/.autoscale-state.json at the moment of the call (None if absent), so a test
sees what was saved before an action ran. $FAKE_SCENARIO (a JSON file) steers the answers:

  pods, nodes      the JSON `kubectl get pods -A -o json` / `kubectl get nodes -o json` prints
  instances        [line, ...]  what `gpuctl list-nodes` prints
  rc               {tool: rc}   exit code of every call of cpu-pool / gpuctl-plan / gpuctl-apply (default 0)
  sleep            {tool: s}    ... after sleeping s seconds; a cpu-pool that finishes writes $FAKE_LOG.done
"""
import json
import os
import sys
import time
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        return None


def main():
    tool, argv = sys.argv[1], sys.argv[2:]
    scenario = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text())
    root = Path(os.environ["AUTOSCALER_ROOT"])
    if tool == "gpuctl" and argv[:1] in (["plan"], ["apply"]):
        tool = f"gpuctl-{argv[0]}"
    entry = {"tool": tool, "argv": argv, "auto_approve": os.environ.get("AUTO_APPROVE"),
             "state": read(os.environ.get("AUTOSCALER_STATE")),
             "overrides": read(root / "lambda" / ".autoscale-state.json")}
    with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    if tool == "kubectl":
        what = "pods" if "pods" in argv else "nodes"
        print(json.dumps(scenario[what]))
        return 0
    if tool == "gpuctl" and argv[:1] == ["list-nodes"]:
        for line in scenario.get("instances", []):
            print(line)
        return 0
    print(f"fake {tool} {' '.join(argv)}")
    sys.stdout.flush()
    time.sleep(scenario.get("sleep", {}).get(tool, 0))
    if tool == "cpu-pool":
        Path(os.environ["FAKE_LOG"] + ".done").write_text("done\n")
    return scenario.get("rc", {}).get(tool, 0)


if __name__ == "__main__":
    sys.exit(main())
