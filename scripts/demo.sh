#!/usr/bin/env bash
# The live demo (spec section 9.1): make demo, make demo-replay CHAPTER=n, make demo-reset.
#
#   demo.sh run        the chapters in CHAPTERS (make demo CHAPTERS=1,2,4,5); default 1 2 3 4 5 6 8,
#                      plus 7 when the stretch plan is installed (k8s/scale-demo/)
#   demo.sh replay N   print the recording of chapter N: ${VERIFICATION_FINAL:-verification/final}/demo/chN.txt
#                      (a relative VERIFICATION_FINAL is read from the repository root); exit 1 without one
#   demo.sh reset      delete what the demo creates (limits-demo Pods, the np-denied namespace, the gpu-test
#                      Pods by name and the gpu-hold Pods by label, the stretch load Job), then scale the CPU
#                      pool back to 1 worker (AUTO_APPROVE=1 scripts/cpu-pool.sh scale 1 --guarded); a Ctrl-C
#                      lets the running command stop, then ends the reset (exit 130)
#   demo.sh list       the chapters, what each shows and its estimated length
#   demo.sh --source-only   (sourced) load the functions only, for the tests
#
# run. Each step prints its title and its command, runs, then waits for Enter (DEMO_NO_PAUSE=1: no wait), so
# the presenter sets the pace. The steps of chapters 1-5 are the check functions of scripts/lib/checks.sh, the
# same ones make verify runs (spec section 9.1: shared, not reimplemented). A step runs as run_check runs a
# check: in a subshell (a check's exit, its set -e failure and the INT/TERM traps _vr_on_signal sets end there,
# not in the demo), with stdin from /dev/null, its stdout and stderr through redact and then tee (screen safety:
# nothing reaches the screen or a recording unredacted), and with its cleanup messages (_vr_tell) on the
# terminal and in the recording. The operator's public IP and the GCP project id are found first
# (screen_masks, scripts/lib/screen.sh), so they are masked too, as <operator_ip> and <GCP_PROJECT>; when
# the IP cannot be found the demo does not start (exit 2) unless DEMO_ALLOW_UNMASKED_IP=1. The filter and tee
# ignore SIGINT and end when the step does: a Ctrl-C stops the step's command, never them first (which would
# kill terraform, cpu-pool.sh or down.sh by SIGPIPE while they stop gracefully); then the demo stops.
# A failed step does not stop the demo: it names the fallback, make demo-replay CHAPTER=n, and the demo goes
# on; a check that SKIPs (no GPU node) says so. The exit status is 1 when a step failed, 2 for an unknown
# chapter or an operator IP that cannot be masked (nothing has run then), 0 otherwise.
#
# Recordings: verification/demo-<YYYY-MM-DDTHHMM-0700>/chN.txt (what was on the screen for chapter N) and
# timings.txt (each chapter's wall time, pauses included, and its command time; then the total of chapters 1-6,
# whose target is 25 minutes). The directory is 0700 and its files 0600, like a verify run's: it is
# unreviewed evidence. make promote-demo RUN=verification/demo-<ts> copies a reviewed run to
# verification/final/demo/ for make demo-replay. A run that finds its directory holding files (a second run in
# the same minute) is refused (exit 2). DEMO_TS replaces the timestamp (tests).
#
# Chapter 6 (Ruling R63): a read-only terraform plan of the workers stack with the inventory scaled to 2 (what
# scaling adds); AUTO_APPROVE=1 scripts/cpu-pool.sh scale 2 --guarded in the background (its own session, so
# a Ctrl-C of the demo does not stop a terraform apply half way; its log goes to ch6.txt, redacted, once it is
# done, or as far as it got when the wait gives up, and meanwhile to .run/demo-scale.log), while kubectl get
# nodes -w shows the new node join, for at most DEMO_SCALE_TIMEOUT_S (default 900) seconds; then
# AUTO_APPROVE=1 scripts/cpu-pool.sh scale 1 --guarded; then scripts/gpuctl plan --set-count <pool>=2, the
# pool being the first in lambda/gpu-nodes.yaml. Chapter 8: gpuctl plan --set-count <pool>=0 and the last
# lines of terraform plan -destroy of the workers stack, then it asks whether to run make down for real: y or
# yes, then the GCP project id typed without echo (make down's own confirmation, asked here so the id never
# shows), runs AUTO_APPROVE=1 scripts/down.sh; anything else runs nothing (never asked under DEMO_NO_PAUSE=1;
# a caller's AUTO_APPROVE skips neither question).
set -euo pipefail

# checks.sh (and common.sh) and screen.sh are linted on their own; pre-commit runs shellcheck without -x.
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/checks.sh"
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/screen.sh"

DEMO_CHAPTERS_ALL=(1 2 3 4 5 6 7 8)
# Seconds between two looks at the background scale of chapter 6.
DEMO_POLL_S="${DEMO_POLL_S:-2}"
# Lines of terraform plan -destroy that chapter 8 shows (the resource count is in the last ones).
DEMO_TAIL_LINES=15

usage() {
  cat >&2 <<'EOF'
usage: scripts/demo.sh run | replay N | reset | list
  run        the chapters in CHAPTERS (make demo CHAPTERS=1,2,4,5); DEMO_NO_PAUSE=1: no wait for Enter
  replay N   print the recording of chapter N (make demo-replay CHAPTER=N)
  reset      delete the demo's objects and scale the CPU pool back to 1 (make demo-reset)
  list       the chapters
EOF
}

# stretch_installed: the stretch plan's scale-demo workload is in the repository (chapter 7 has content).
stretch_installed() {
  [[ -d "$REPO_ROOT/k8s/scale-demo" ]]
}

chapter_title() {
  case "$1" in
    1) echo "Architecture: nodes, describe node (summary), system Pods on k3s-cp (A1, A2, A3)" ;;
    2) echo "Security: firewall inventory, negative tests, only WireGuard on the wire (B6, B5, B2)" ;;
    3) echo "Cross-node connectivity matrix (B1)" ;;
    4) echo "hello-world: cross-node, port-forward, QoS, cgroup, OOMKilled and throttling, NetworkPolicy (C2-C5, C7, C8)" ;;
    5) echo "GPU: capacity, nvidia-smi, vectoradd, the over-request stays Pending (D2, D4, D5, D6)" ;;
    6) echo "Scaling by inventory: workers plan, CPU pool 1 -> 2 -> 1, GPU plan with the pool at 2" ;;
    7) echo "Stretch: autoscaling (requires the stretch plan)" ;;
    8) echo "Wrap-up: the teardown plan; make down only if you say so" ;;
  esac
}

chapter_estimate() {
  case "$1" in
    1) echo "3 min" ;;
    2) echo "5 min" ;;
    3) echo "2 min" ;;
    4) echo "5 min" ;;
    5) echo "4 min" ;;
    6) echo "5 min" ;;
    7) echo "3-10 min" ;;
    8) echo "1 min" ;;
  esac
}

# parse_chapters SPEC: the chapters SPEC names (separated by commas and/or spaces), in that order, each once,
# on one line. An empty SPEC is the default: 1 2 3 4 5 6 8, and 7 when the stretch plan is installed. Anything
# but 1-8 is refused: "unknown chapter X" on stderr, status 2, nothing printed.
parse_chapters() {
  local spec="${1:-}" word out=" "
  local -a words=()
  if [[ -z "${spec//[[:space:],]/}" ]]; then
    for word in "${DEMO_CHAPTERS_ALL[@]}"; do
      if [[ "$word" != 7 ]] || stretch_installed; then out+="$word "; fi
    done
  else
    IFS=$', \t\n' read -r -a words <<< "$spec"
    for word in "${words[@]}"; do
      [[ -n "$word" ]] || continue
      if [[ ! "$word" =~ ^[1-8]$ ]]; then
        log "unknown chapter $word (chapters: ${DEMO_CHAPTERS_ALL[*]}; make demo CHAPTERS=1,3)"
        return 2
      fi
      [[ "$out" == *" $word "* ]] || out+="$word "
    done
  fi
  out="${out# }"
  printf '%s\n' "${out% }"
}

demo_list() {
  local n note
  for n in "${DEMO_CHAPTERS_ALL[@]}"; do
    note=""
    if [[ "$n" == 7 ]]; then
      if stretch_installed; then note="  [stretch: installed]"; else note="  [stretch: not installed, not in the default]"; fi
    fi
    printf '%2s  %-9s %s%s\n' "$n" "$(chapter_estimate "$n")" "$(chapter_title "$n")" "$note"
  done
}

# --- replay ----------------------------------------------------------------------------------------

demo_replay() {
  local n="${1:-}" final file
  (($# == 1)) || { usage; die "usage: replay takes exactly one chapter number" 2; }
  [[ "$n" =~ ^[1-8]$ ]] || die "unknown chapter $n (chapters: ${DEMO_CHAPTERS_ALL[*]})" 2
  final="${VERIFICATION_FINAL:-verification/final}"
  [[ "$final" == /* ]] || final="$REPO_ROOT/$final"
  file="$final/demo/ch$n.txt"
  [[ -f "$file" ]] || die "no recording for chapter $n ($file; make promote-demo RUN=verification/demo-<ts> records one)" 1
  # Recordings were redacted when they were made; once more, as everything the demo puts on the screen, with
  # the operator's IP and the project id as they are now (a recording made while the IP was unknown, or
  # before it changed, may still hold it). Replay is the fallback for a bad network, so it looks nothing up
  # (--local): the IP is masked only when .env sets OPERATOR_IP, and an unknown IP does not stop it playing.
  cd -- "$REPO_ROOT"
  screen_masks --local
  redact <"$file"
}

# --- run: steps ------------------------------------------------------------------------------------

# _demo_show: the screen and the chapter's recording, redacted (spec section 9.1, screen safety). Both
# ignore SIGINT and end when the step closes the pipe (screen_filter in scripts/lib/screen.sh): a Ctrl-C
# then cannot kill terraform, cpu-pool.sh or down.sh by SIGPIPE while they stop gracefully.
_demo_show() {
  (
    trap '' INT
    redact | tee -a -- "$DEMO_FILE"
  )
}

# demo_pause: wait for Enter on the demo's stdin (the terminal); DEMO_NO_PAUSE=1, or no more input, goes on.
demo_pause() {
  local answer
  [[ "${DEMO_NO_PAUSE:-}" != 1 ]] || return 0
  printf '[Enter] next step ' >&2
  read -r answer || true
}

# demo_note TEXT...: a line on the screen and in the recording.
demo_note() {
  printf '%s\n' "$@" | _demo_show
}

# demo_step TITLE CMD...: print the title and the command, run CMD (a check function, another function or a
# program) as run_check runs a check (see the header), print its outcome, then pause. Records a failure in
# DEMO_FAILED and adds the command time to DEMO_RUN_S. A step whose output could not be redacted or recorded
# counts as failed too.
demo_step() {
  local title="$1" rc started errexit=0
  shift
  DEMO_STEP=$((DEMO_STEP + 1))
  DEMO_STEPS=$((DEMO_STEPS + 1))
  {
    printf '\n--- [ch%s.%d] %s\n' "$DEMO_CH" "$DEMO_STEP" "$title"
    _vr_quote "$@"
  } | _demo_show
  started=$SECONDS
  [[ $- == *e* ]] && errexit=1
  set +e
  (
    if ((errexit)); then set -e; fi
    # shellcheck disable=SC2034 # read by _vr_tell in this subshell (checks.sh)
    _VR_CHECK_ID="ch$DEMO_CH" _VR_CHECK_FILE="$DEMO_FILE" _VR_TELL_FD=3
    trap _vr_reap EXIT
    "$@"
  ) 3>&2 </dev/null 2>&1 | _demo_show
  local -a status=("${PIPESTATUS[@]}")
  if ((errexit)); then set -e; fi
  DEMO_RUN_S=$((DEMO_RUN_S + SECONDS - started))
  rc=${status[0]}
  if ((status[1] != 0)); then
    demo_note "(the output of this step could not be redacted or recorded in full)"
    ((rc != 0 && rc != 77)) || rc=1
  fi
  _demo_outcome "$rc" "$title"
  demo_pause
}

# _demo_outcome RC TITLE: the step's outcome on the screen and in the recording; a failure goes to DEMO_FAILED.
_demo_outcome() {
  local rc="$1" title="$2"
  if ((rc == 0)); then
    demo_note "=> OK"
  elif ((rc == 77)); then
    DEMO_SKIPPED=$((DEMO_SKIPPED + 1))
    demo_note "=> SKIP (nothing to show live; the recording: make demo-replay CHAPTER=$DEMO_CH)"
  else
    DEMO_FAILED+=("ch$DEMO_CH: $title (exit $rc)")
    DEMO_CH_FAILED=$((DEMO_CH_FAILED + 1))
    demo_note "=> FAILED (exit $rc); fallback: make demo-replay CHAPTER=$DEMO_CH, then go on"
  fi
}

# --- chapters 1-5: the checks of make verify --------------------------------------------------------

# demo_a2_summary: check A2 (kubectl describe node for every node), shown as a summary: the commands, each
# node's name, roles, taints, addresses, capacity and allocatable (cpu, memory, GPUs), kernel, OS image and
# runtime. The check's status is the step's.
demo_a2_summary() {
  local -a status
  set +e
  check_a2_describe_nodes 2>&1 | grep -E \
    -e '^(\$ |A2: |Name:|Roles:|Taints:|Addresses:|Capacity:|Allocatable:)' \
    -e '^[[:space:]]+(InternalIP|ExternalIP|cpu|memory|nvidia\.com/gpu|Kernel Version|OS Image|Container Runtime Version|Kubelet Version):'
  status=("${PIPESTATUS[@]}")
  set -e
  return "${status[0]}"
}

demo_chapter_1() {
  demo_step "A1 nodes Ready (kubectl get nodes -o wide)" check_a1_nodes
  demo_step "A2 kubectl describe node, every node (summary)" demo_a2_summary
  demo_step "A3 metrics-server and CoreDNS Ready on k3s-cp" check_a3_system_pods
}

demo_chapter_2() {
  demo_step "B6 firewall inventory: every source a /32 (or IAP)" check_b6_firewall_inventory
  demo_step "B5 negative tests: API ports from here, the metadata server, DNS, guard counters" check_b5_negative
  # The brief's "simplified B2" is B2 itself (spec section 9.1: the demo shares the checks, it does not
  # reimplement them): a capture on k3s-cp while marked requests cross; it shows only udp/51820 crossing.
  demo_step "B2 on the wire at k3s-cp: only WireGuard (udp/51820) between the nodes" check_b2_wire
}

demo_chapter_3() {
  demo_step "B1 N x N connectivity matrix (curl, MTU ping, Service DNS)" check_b1_matrix
}

demo_chapter_4() {
  demo_step "C2 hello world from another node" check_c2_cross_node_curl
  demo_step "C3 hello world through kubectl port-forward" check_c3_port_forward
  demo_step "C4 QoS class Guaranteed" check_c4_qos
  demo_step "C5 cgroup memory.max and cpu.max" check_c5_cgroup
  demo_step "C7 limits-demo: OOMKilled and CPU throttling" check_c7_limits_demo
  demo_step "C8 NetworkPolicy: allowed from nettest, denied elsewhere" check_c8_netpol
}

demo_chapter_5() {
  demo_step "D2 GPU capacity and allocatable" check_d2_gpu_capacity
  demo_step "D4 gpu-smoke: nvidia-smi in a Pod" check_d4_gpu_smoke
  demo_step "D5 vectoradd: Test PASSED" check_d5_vectoradd
  demo_step "D6 gpu-overflow stays Pending" check_d6_overflow
}

# --- chapter 6: scaling by inventory (Ruling R63) ------------------------------------------------------

# demo_gpu_pool: the first pool of lambda/gpu-nodes.yaml (the inventory gpu-apply reconciles).
demo_gpu_pool() {
  local pool
  pool="$(awk '/^pools:/ { in_pools = 1; next }
               in_pools && /^[^[:space:]#]/ { exit }
               in_pools && /^  [A-Za-z0-9_-]+:/ { sub(/^  /, ""); sub(/:.*/, ""); print; exit }' \
    "$REPO_ROOT/lambda/gpu-nodes.yaml")" || return 1
  [[ "$pool" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || { echo "no GPU pool found in lambda/gpu-nodes.yaml" >&2; return 1; }
  printf '%s\n' "$pool"
}

# _demo_new_workers N: the workers scaling the CPU pool to N adds (cpu-pool.sh render N reads the inventory
# only), one per line.
_demo_new_workers() {
  local next
  next="$(scripts/cpu-pool.sh render "$1")" || return 1
  jq -r --slurpfile now infra/workers/workers.auto.tfvars.json \
    '.cpu_workers | keys[] | select(. as $n | $now[0].cpu_workers | has($n) | not)' <<< "$next"
}

# demo_workers_plan N: terraform plan of the workers stack with the inventory scaled to N (read-only: what
# make cpu-scale N=N would create), the rendered inventory passed as a -var-file, as cpu-pool.sh plans it.
demo_workers_plan() {
  local n="$1" next rc=0
  vr_env
  run_dir || return 1
  next="$RUN_DIR/demo-workers.$n.tfvars.json"
  echo "\$ scripts/cpu-pool.sh render $n > .run/demo-workers.$n.tfvars.json"
  scripts/cpu-pool.sh render "$n" >"$next" || { rm -f -- "$next"; return 1; }
  (tf_init workers) || { rm -f -- "$next"; return 1; }
  vr_cmd terraform -chdir=infra/workers plan -input=false -lock-timeout=120s -no-color "-var-file=$next" || rc=$?
  rm -f -- "$next"
  return "$rc"
}

# _demo_detach RCFILE LOG CMD...: start CMD in the background in its own session (setsid, where there is
# one: a Ctrl-C of the demo then does not reach it, and no terraform apply stops half way), its output in LOG;
# its exit status is written to RCFILE when it ends. Not a job of the caller's shell: _vr_reap leaves it alone.
_demo_detach() {
  local rcfile="$1" logfile="$2"
  shift 2
  local -a session=()
  if command -v setsid >/dev/null 2>&1; then session=(setsid); fi
  rm -f -- "$rcfile"
  # shellcheck disable=SC2016 # expanded by the inner bash
  (${session[@]+"${session[@]}"} bash -c 'rc=0; "$@" || rc=$?; printf "%s\n" "$rc" >"$0"' "$rcfile" "$@" \
    </dev/null >"$logfile" 2>&1 &)
}

# _demo_node_ready NODE: print and check NODE's Ready condition.
_demo_node_ready() {
  local status
  status="$(vr_cap kc get node "$1" -o 'jsonpath={.status.conditions[?(@.type=="Ready")].status}' "$_VR_RT")" \
    || return 1
  [[ "$status" == True ]]
}

# _demo_watch_lines MARK: copy stdin to stdout line by line and create the file MARK at the first line. Builtins
# only (no child process of its own), so stopping the watch subshell's children stops it.
_demo_watch_lines() {
  local line
  while IFS= read -r line; do
    printf '%s\n' "$line"
    [[ -e "$1" ]] || : >"$1"
  done
}

# demo_scale_up N: AUTO_APPROVE=1 scripts/cpu-pool.sh scale N --guarded in the background (_demo_detach) while
# kubectl get nodes -w shows the new worker(s) join; waits for the scale to end, for at most
# DEMO_SCALE_TIMEOUT_S seconds. Its log goes to the recording (redacted), not the screen. Passes when the scale
# exited 0 and every new worker is Ready.
demo_scale_up() {
  local n="$1" rcfile logfile timeout="${DEMO_SCALE_TIMEOUT_S:-900}" deadline watch started rc fail=0 node names
  local -a new=()
  [[ "$timeout" =~ ^[0-9]+$ ]] || { echo "DEMO_SCALE_TIMEOUT_S is not a number of seconds: '$timeout'"; return 1; }
  names="$(_demo_new_workers "$n")" || { echo "cannot read the CPU pool inventory (scripts/cpu-pool.sh render $n)"; return 1; }
  while read -r node; do
    [[ -n "$node" ]] && new+=("$node")
  done <<< "$names"
  ((${#new[@]})) || echo "the CPU pool has $n worker(s) already: the scale changes nothing"
  run_dir || return 1
  rcfile="$RUN_DIR/demo-scale.rc"
  logfile="$RUN_DIR/demo-scale.log"
  echo "\$ AUTO_APPROVE=1 scripts/cpu-pool.sh scale $n --guarded &   # in the background; its log goes to ch$DEMO_CH.txt"
  _demo_detach "$rcfile" "$logfile" env AUTO_APPROVE=1 scripts/cpu-pool.sh scale "$n" --guarded
  _vr_quote kc get nodes -w | redact
  # The watch's lines pass through _demo_watch_lines, which marks the first one (kubectl has started). kubectl
  # and that reader are both children of $watch, so _vr_stop stops both.
  started="$RUN_DIR/demo-watch.started"
  rm -f -- "$started"
  (kc get nodes -w </dev/null > >(_demo_watch_lines "$started")) &
  watch=$!
  # A scale that ends at once must not stop the watch before it has shown anything: wait (at most 5 s, or
  # until the watch has ended) for its first line.
  deadline=$((SECONDS + 5))
  while [[ ! -e "$started" ]] && kill -0 "$watch" 2>/dev/null && ((SECONDS < deadline)); do
    sleep 0.1
  done
  deadline=$((SECONDS + timeout))
  while [[ ! -s "$rcfile" ]] && ((SECONDS < deadline)); do
    sleep "$DEMO_POLL_S"
  done
  _vr_stop "$watch"
  rm -f -- "$started"
  if [[ ! -s "$rcfile" ]]; then
    # The log so far goes to the recording; the file itself stays, and the scale goes on writing to it.
    {
      printf -- '--- the log so far of AUTO_APPROVE=1 scripts/cpu-pool.sh scale %s --guarded (still running) ---\n' "$n"
      cat -- "$logfile"
      printf -- '--- the rest is in .run/demo-scale.log ---\n'
    } | redact >>"$DEMO_FILE"
    echo "cpu-pool.sh scale $n has not finished after ${timeout}s; it goes on in the background (its log so far"
    echo "is in ch$DEMO_CH.txt, all of it in .run/demo-scale.log). Once it is done, make demo-reset scales the pool back to 1."
    return 1
  fi
  rc="$(head -n 1 -- "$rcfile")"
  {
    printf -- '--- the log of AUTO_APPROVE=1 scripts/cpu-pool.sh scale %s --guarded (exit %s) ---\n' "$n" "$rc"
    cat -- "$logfile"
    printf -- '--- end of the log ---\n'
  } | redact >>"$DEMO_FILE"
  rm -f -- "$rcfile" "$logfile"
  echo "cpu-pool.sh scale $n --guarded finished (exit $rc); its log is in ch$DEMO_CH.txt"
  [[ "$rc" == 0 ]] || return 1
  for node in ${new[@]+"${new[@]}"}; do
    if _demo_node_ready "$node"; then
      echo "$node is Ready"
    else
      echo "$node is not Ready"
      fail=1
    fi
  done
  return "$fail"
}

demo_chapter_6() {
  local pool
  pool="$(demo_gpu_pool 2>&1)" || { demo_note "$pool"; pool=""; }
  demo_step "What scaling the CPU pool to 2 adds: terraform plan of the workers stack (read-only)" \
    demo_workers_plan 2
  demo_step "Scale the CPU pool to 2 in the background; watch the new worker join" demo_scale_up 2
  demo_step "Scale back to 1: drain the new worker, delete it, destroy its VM" \
    env AUTO_APPROVE=1 scripts/cpu-pool.sh scale 1 --guarded
  if [[ -n "$pool" ]]; then
    demo_step "gpu-plan with the inventory's pool $pool at 2 (read-only; nothing is started)" \
      scripts/gpuctl plan --set-count "$pool=2"
  else
    demo_step "gpu-plan with the inventory's pool at 2" demo_gpu_pool
  fi
}

# --- chapters 7 and 8 -------------------------------------------------------------------------------

# Chapter 7 belongs to the stretch plan (Task S6 replaces this function).
demo_chapter_7() {
  if ! stretch_installed; then
    demo_note "Chapter 7 requires the stretch plan (k8s/scale-demo/ is not installed): nothing to show."
    return 0
  fi
  demo_note "Chapter 7: k8s/scale-demo/ is installed, but its steps come with the stretch plan (Task S6)."
}

# demo_destroy_summary: the last lines of terraform plan -destroy of the workers stack (read-only).
demo_destroy_summary() {
  local -a status
  vr_env
  (tf_init workers) || return 1
  echo "\$ terraform -chdir=infra/workers plan -destroy -input=false -lock-timeout=120s -no-color | tail -n $DEMO_TAIL_LINES"
  set +e
  # tail ignores SIGINT, as _demo_show does: a Ctrl-C must not kill the plan (it holds the state lock) by SIGPIPE.
  terraform -chdir=infra/workers plan -destroy -input=false -lock-timeout=120s -no-color 2>&1 \
    | (
      trap '' INT
      tail -n "$DEMO_TAIL_LINES"
    )
  status=("${PIPESTATUS[@]}")
  set -e
  return "${status[0]}"
}

# demo_ask_down: ask whether to run make down now. Only y or yes, then the GCP project id (GCP_PROJECT in
# .env) typed without echo, runs it: make down's own confirmation, asked by the demo so the id never shows
# on the screen (down.sh's prompt would echo it). Then AUTO_APPROVE=1 scripts/down.sh runs in the
# foreground, no stdin, its output redacted. A caller's AUTO_APPROVE never skips either question. Never
# asked under DEMO_NO_PAUSE=1.
demo_ask_down() {
  local answer="" typed="" project rc=0 started
  if [[ "${DEMO_NO_PAUSE:-}" == 1 ]]; then
    demo_note "make down was not run (DEMO_NO_PAUSE=1: nobody to ask); run it by hand when you are done."
    return 0
  fi
  printf 'Tear everything down now with make down? [y/N] ' >&2
  read -r answer || true
  if [[ ! "$answer" =~ ^([yY]|[yY][eE][sS])$ ]]; then
    demo_note "make down was not run; the cluster stays up."
    return 0
  fi
  printf 'Type the GCP project id (GCP_PROJECT in .env; not shown as you type) to destroy everything: ' >&2
  read -rs typed || true
  printf '\n' >&2
  project="$(screen_env_project)" || project=""
  if [[ -z "$project" || "$typed" != "$project" ]]; then
    demo_note "make down was not run: the answer was not the project id (GCP_PROJECT in .env); the cluster stays up."
    return 0
  fi
  DEMO_STEP=$((DEMO_STEP + 1))
  DEMO_STEPS=$((DEMO_STEPS + 1))
  {
    printf '\n--- [ch%s.%d] %s\n' "$DEMO_CH" "$DEMO_STEP" "make down: tear everything down"
    printf '$ AUTO_APPROVE=1 scripts/down.sh   # make down, confirmed above with the project id\n'
  } | _demo_show
  started=$SECONDS
  env AUTO_APPROVE=1 scripts/down.sh </dev/null 2>&1 | _demo_show || rc=$?
  DEMO_RUN_S=$((DEMO_RUN_S + SECONDS - started))
  _demo_outcome "$rc" "make down"
}

demo_chapter_8() {
  local pool
  pool="$(demo_gpu_pool 2>&1)" || { demo_note "$pool"; pool=""; }
  if [[ -n "$pool" ]]; then
    demo_step "GPU nodes: the plan with the pool $pool at 0 (read-only)" scripts/gpuctl plan --set-count "$pool=0"
  else
    demo_step "GPU nodes: the plan with the pool at 0" demo_gpu_pool
  fi
  demo_step "The workers stack: what terraform plan -destroy removes (last lines, read-only)" demo_destroy_summary
  demo_ask_down
}

# --- run ----------------------------------------------------------------------------------------------

# demo_chapter N: run chapter N into $DEMO_DIR/chN.txt and append its timing to timings.txt.
demo_chapter() {
  local n="$1" started wall
  DEMO_CH="$n"
  DEMO_FILE="$DEMO_DIR/ch$n.txt"
  DEMO_STEP=0
  DEMO_STEPS=0
  DEMO_RUN_S=0
  DEMO_CH_FAILED=0
  DEMO_SKIPPED=0
  : >>"$DEMO_FILE"
  printf '\n=== Chapter %s: %s (about %s) ===\n' "$n" "$(chapter_title "$n")" "$(chapter_estimate "$n")" | _demo_show
  started=$SECONDS
  DEMO_CH_STARTED=$started
  "demo_chapter_$n"
  wall=$((SECONDS - started))
  DEMO_CH_STARTED=""
  printf 'ch%s %ds (commands %ds; %d step(s), %d failed, %d skipped) %s\n' "$n" "$wall" "$DEMO_RUN_S" \
    "$DEMO_STEPS" "$DEMO_CH_FAILED" "$DEMO_SKIPPED" "$(chapter_title "$n")" >>"$DEMO_DIR/timings.txt"
  if ((n <= 6)); then DEMO_CORE_S=$((DEMO_CORE_S + wall)); fi
  DEMO_TOTAL_S=$((DEMO_TOTAL_S + wall))
}

# _demo_interrupted: Ctrl-C (the step's own cleanup has run in its subshell): note it, then end by SIGINT.
_demo_interrupted() {
  trap - INT
  if [[ -n "${DEMO_CH_STARTED:-}" && -n "${DEMO_DIR:-}" ]]; then
    printf 'ch%s interrupted after %ds\n' "$DEMO_CH" "$((SECONDS - DEMO_CH_STARTED))" >>"$DEMO_DIR/timings.txt" || true
  fi
  log "" "demo interrupted. make demo-reset deletes what the demo created and scales the CPU pool back to 1."
  if [[ -f "$RUN_DIR/demo-scale.log" && ! -s "$RUN_DIR/demo-scale.rc" ]]; then
    log "chapter 6's CPU scale may still be running in the background (its log: .run/demo-scale.log);"
    log "make demo-reset waits for nothing: run it once that scale is done."
  fi
  kill -s INT "$$"
}

demo_run() {
  local chapters n ts scan rc=0
  (($# == 0)) || { usage; die "usage: run takes no arguments (CHAPTERS=1,3 picks the chapters)" 2; }
  chapters="$(parse_chapters "${CHAPTERS:-}")" || exit 2
  cd -- "$REPO_ROOT"
  # Before the first redact starts: from here on the operator's IP and the project id are masked on the
  # screen and in the recordings. Without the IP (Lambda's operator rule shows it) the demo does not start.
  screen_masks
  if ((!SCREEN_IP_MASKED)); then
    if [[ "${DEMO_ALLOW_UNMASKED_IP:-}" != 1 ]]; then
      log "cannot determine the operator's public IP (OPERATOR_IP in .env is empty or not an IPv4 address, and"
      log "checkip.amazonaws.com did not answer), so the screen and the recordings could not mask it."
      log "Set OPERATOR_IP in .env to the address gpuctl uses and run make preflight; DEMO_ALLOW_UNMASKED_IP=1 runs anyway."
      exit 2
    fi
    log "warning: the operator's public IP is unknown and NOT masked on the screen or in the recordings (DEMO_ALLOW_UNMASKED_IP=1)"
  fi
  if ((!SCREEN_PROJECT_MASKED)); then
    log "warning: GCP_PROJECT in .env is missing or not a project id, so the screen and the recordings cannot mask it"
  fi
  ts="${DEMO_TS:-$(TZ="$VR_TZ" date +%Y-%m-%dT%H%M%z)}"
  DEMO_DIR="$REPO_ROOT/verification/demo-$ts"
  if [[ -d "$DEMO_DIR" && -n "$(ls -A -- "$DEMO_DIR")" ]]; then
    die "$DEMO_DIR already holds files from another run; start again in a minute (new directory name)" 2
  fi
  umask 077
  mkdir -p -- "$DEMO_DIR"
  chmod 700 -- "$DEMO_DIR"
  DEMO_FAILED=()
  DEMO_CORE_S=0
  DEMO_TOTAL_S=0
  trap _demo_interrupted INT
  for n in $chapters; do
    demo_chapter "$n"
  done
  trap - INT
  printf 'total %ds; chapters 1-6: %dm%02ds (target 25 min)\n' "$DEMO_TOTAL_S" "$((DEMO_CORE_S / 60))" \
    "$((DEMO_CORE_S % 60))" >>"$DEMO_DIR/timings.txt"

  echo
  echo "recorded in verification/demo-$ts/ (timings.txt: wall time per chapter, pauses included)"
  if command -v gitleaks >/dev/null 2>&1; then
    if gitleaks dir "$DEMO_DIR" --config "$REPO_ROOT/.gitleaks.toml" --no-banner --redact --exit-code 1 >/dev/null 2>&1; then
      scan="gitleaks found no leaks"
    else
      scan="gitleaks reported leaks (or failed): do not share or promote this run"
      rc=1
    fi
  else
    scan="not scanned (gitleaks is not installed)"
  fi
  echo "secret scan: $scan"
  if ((${#DEMO_FAILED[@]})); then
    echo "failed steps (their recordings: make demo-replay CHAPTER=n):"
    printf '  %s\n' "${DEMO_FAILED[@]}"
    rc=1
  fi
  return "$rc"
}

# --- reset ----------------------------------------------------------------------------------------------

# _demo_reset_do CMD...: print the command, run it (stdin /dev/null), output redacted; a failure is noted in
# RESET_FAILED and the reset goes on.
_demo_reset_do() {
  local rc=0
  (vr_cmd "$@") </dev/null 2>&1 | screen_filter || rc=$?
  if ((rc != 0)); then
    RESET_FAILED+=("$(_vr_quote "$@") (exit $rc)")
  fi
  # Ctrl-C: the command has stopped (the filter outlived it, so nothing was killed by SIGPIPE); no later one runs.
  if ((RESET_INTERRUPTED)); then
    log "demo-reset interrupted; re-run make demo-reset to finish"
    exit 130
  fi
}

demo_reset() {
  (($# == 0)) || { usage; die "usage: reset takes no arguments" 2; }
  cd -- "$REPO_ROOT"
  screen_masks
  if ((!SCREEN_IP_MASKED)); then
    log "warning: cannot determine the operator's public IP, so the output cannot mask it (set OPERATOR_IP in .env)"
  fi
  RESET_FAILED=()
  RESET_INTERRUPTED=0
  trap 'RESET_INTERRUPTED=1' INT
  echo "demo-reset: deleting what the demo created, then scaling the CPU pool back to 1"
  _demo_reset_do kc -n limits-demo delete pod oom cpu-burn --ignore-not-found --timeout=120s "$_VR_RT"
  _demo_reset_do kc delete namespace np-denied --ignore-not-found --timeout=120s "$_VR_RT"
  _demo_reset_do kc -n gpu-test delete pod gpu-smoke vectoradd gpu-overflow --ignore-not-found --timeout=120s "$_VR_RT"
  # The holders have generated names (gpu-hold-xxxxx) and restart; a leftover one keeps a billed GPU allocated.
  _demo_reset_do kc -n gpu-test delete pod -l app=gpu-hold --ignore-not-found --timeout=120s "$_VR_RT"
  if stretch_installed; then
    _demo_reset_do kc -n scale-demo delete job load-generator --ignore-not-found --timeout=120s "$_VR_RT"
  fi
  _demo_reset_do env AUTO_APPROVE=1 scripts/cpu-pool.sh scale 1 --guarded
  if ((${#RESET_FAILED[@]})); then
    log "demo-reset: failed (the rest ran; re-run make demo-reset once the cause is fixed):"
    printf '  %s\n' "${RESET_FAILED[@]}" | screen_filter >&2
    return 1
  fi
  echo "demo-reset: done"
}

# --- main -------------------------------------------------------------------------------------------------

demo_main() {
  local cmd="${1:-}"
  (($# == 0)) || shift
  case "$cmd" in
    run) demo_run "$@" ;;
    replay) demo_replay "$@" ;;
    reset) demo_reset "$@" ;;
    list)
      (($# == 0)) || { usage; die "usage: list takes no arguments" 2; }
      demo_list
      ;;
    -h | --help | help) usage 2>&1 ;;
    *)
      usage
      die "usage: unknown or missing subcommand '$cmd'" 2
      ;;
  esac
}

if [[ "${1:-}" == --source-only ]]; then
  # shellcheck disable=SC2317 # reached only when the file is run, not sourced
  return 0 2>/dev/null || exit 0
fi
demo_main "$@"
