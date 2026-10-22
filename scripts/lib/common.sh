# shellcheck shell=bash
# Shared helpers for the operator scripts. Source it; do not run it.
# Task 7: REPO_ROOT, log, die, load_env. Task 10: the Terraform, IAP SSH, kubectl and prompt helpers
# below. Kept to bash 3.2 (macOS operators): no bash-4-only features, except where stack_lock takes a
# lock (exec {fd}>, bash >= 4.1; checked at run time, so an older bash is refused with a clear message).

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"

log() {
  printf '%s\n' "$*" >&2
}

# die MESSAGE [CODE]: log MESSAGE and exit CODE (default 1). Codes follow gpuctl's: 1 failed, 2 refused.
die() {
  log "error: $1"
  exit "${2:-1}"
}

# Mode of a file (following symlinks) as octal digits; GNU stat first, then BSD/macOS stat.
_file_mode() {
  stat -L -c '%a' -- "$1" 2>/dev/null || stat -L -f '%Lp' -- "$1"
}

# Export every variable in $REPO_ROOT/.env. The file holds the Lambda API key, so it must be private (600).
# A missing or non-private .env exits 2 (refused), as gpuctl does when LAMBDA_API_KEY is missing.
# A valid GCP_PROJECT also exports TF_VAR_project_id and TF_VAR_state_bucket (as tf_env does), so a
# caller that runs terraform directly after load_env has them. An invalid one is left for tf_env to
# refuse, so commands that never touch GCP still work.
load_env() {
  local env_file="$REPO_ROOT/.env" mode
  [[ -f "$env_file" ]] \
    || die "$env_file not found: cp .env.example .env && chmod 600 .env, then fill it in" 2
  mode="$(_file_mode "$env_file")" || die "cannot read the permissions of $env_file" 2
  [[ "$mode" == 600 ]] \
    || die "$env_file has mode $mode; it holds secrets and must be 600: chmod 600 $env_file" 2
  set -a
  # shellcheck source=/dev/null
  source "$env_file"
  set +a
  if _gcp_project_ok; then _export_tf_vars; fi
}

# ---- Terraform, IAP SSH, kubectl (Task 10) ----------------------------------------------------

CP_NAME=k3s-cp
# The cluster kubeconfig (spec §4.6, §8): 0600, server https://127.0.0.1:6443 through `make tunnel`.
KUBECONFIG_FILE="$HOME/.kube/hybrid-gpu.yaml"
# Runtime state (gitignored): the tunnel's PID file and log, saved Terraform plans.
RUN_DIR="$REPO_ROOT/.run"

# run_dir: create $RUN_DIR, private (it holds saved plans).
run_dir() {
  mkdir -p -- "$RUN_DIR" && chmod 700 -- "$RUN_DIR"
}

# GCP_PROJECT must be a GCP project id (6-30 characters: lowercase letters, digits, hyphens; a
# letter first, no hyphen last). It ends up in bucket names and gcloud arguments.
_gcp_project_ok() {
  [[ "${GCP_PROJECT:-}" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]
}

_export_tf_vars() {
  export TF_VAR_project_id="$GCP_PROJECT" TF_VAR_state_bucket="$GCP_PROJECT-k3s-tfstate"
}

# tf_env: export TF_VAR_project_id and TF_VAR_state_bucket from GCP_PROJECT (.env, so call load_env
# first; load_env already exports them for a valid id). Every Terraform helper below calls it. An
# invalid GCP_PROJECT is refused (exit 2).
tf_env() {
  _gcp_project_ok || die "GCP_PROJECT='${GCP_PROJECT:-}' in .env is not a GCP project id" 2
  _export_tf_vars
}

# _bash_version_ok [MAJOR MINOR]: the bash version (default: this bash) is at least 4.1, which
# `exec {fd}>` (a descriptor bash picks and names) needs. macOS ships bash 3.2.
_bash_version_ok() {
  local major="${1:-${BASH_VERSINFO[0]}}" minor="${2:-${BASH_VERSINFO[1]}}"
  (( major > 4 || (major == 4 && minor >= 1) ))
}

# stack_lock [--required] STACK: hold flock(1) on $RUN_DIR/STACK.lock until this process exits, so two
# runs never plan or apply one stack at once (C-1): they would race on its saved plan
# ($RUN_DIR/STACK.tfplan) and, for workers, on the inventory file. tf_plan and tf_apply_plan take it;
# cpu-pool.sh takes the workers lock before it reads the inventory (its --guarded runs since Ruling
# SR8: workers.lock is the one lock of the workers stack); down.sh takes every stack it destroys.
# Another holder: fail fast (exit 2), naming the stack and the file. The kernel drops the lock when the
# holder exits, however it ends, so there is no stale lock and nothing to clean up; the file stays
# (removing it would let a waiting run lock an unlinked file). Re-entrant: a stack this process holds
# already is a no-op (a second open of the file would conflict with this process's own lock); a
# subshell shares the descriptor, so it may call it too. Children inherit the descriptor: a script
# that starts a process outliving it must close it there (tunnel.sh does for the API tunnel).
# Without flock(1) on PATH: one WARNING per process and the run goes on unlocked, or with --required
# exit 2. Locking needs bash >= 4.1; an older bash is refused (exit 2).
stack_lock() {
  local required=0 stack file fd
  if [[ "${1:-}" == --required ]]; then
    required=1
    shift
  fi
  stack="${1:-}"
  [[ "$stack" =~ ^[a-z][a-z0-9-]*$ ]] || die "stack_lock: not a stack name: '$stack'"
  [[ " ${_STACK_LOCKS:-} " != *" $stack "* ]] || return 0
  file="$RUN_DIR/$stack.lock"
  if ! command -v flock > /dev/null 2>&1; then
    (( required == 0 )) \
      || die "flock(1) (util-linux) is not on PATH, and this run must hold the $stack stack's lock $file: nothing was changed" 2
    if [[ -z "${_STACK_LOCK_WARNED:-}" ]]; then
      log "WARNING: flock(1) (util-linux) is not on PATH: this run plans and applies Terraform stacks without their locks; do not run make up, make down or cpu-pool.sh at the same time"
      _STACK_LOCK_WARNED=1
    fi
    return 0
  fi
  _bash_version_ok \
    || die "taking the $stack stack's lock needs bash >= 4.1 (this is bash ${BASH_VERSION}); run it with a newer bash (macOS: brew install bash)" 2
  run_dir || die "cannot create $RUN_DIR"
  exec {fd}>> "$file" || die "cannot open $file"
  if ! flock -n "$fd"; then
    exec {fd}>&-
    die "another run holds $file, the lock of the $stack stack (make up, make down, cpu-pool.sh or the autoscaler): this run changed nothing in the $stack stack; the lock is released when that run exits" 2
  fi
  _STACK_LOCKS="${_STACK_LOCKS:-} $stack"
}

# tf_dir STACK: the stack's directory, $REPO_ROOT/infra/STACK; dies for anything else.
tf_dir() {
  [[ "$1" =~ ^[a-z][a-z0-9-]*$ && -d "$REPO_ROOT/infra/$1" ]] || die "unknown Terraform stack: $1"
  printf '%s\n' "$REPO_ROOT/infra/$1"
}

# tf_init STACK: init STACK on the gcs backend, bucket $GCP_PROJECT-k3s-tfstate, prefix STACK.
# Skipped when the last init of this stack succeeded for exactly that bucket and prefix and the
# committed lock file (.terraform.lock.hcl) is unchanged since: a pulled provider bump is installed
# by the next run instead of failing every plan with "Inconsistent dependency lock file". A marker,
# .terraform/k3s-tf-init.ok, records both (the lock file's cksum, which only has to notice a
# change); it is removed before every init and written only after it succeeds. gpuctl's
# Infra.ensure_init (lambda/gpuctl/infra.py) writes and checks the very same text, so neither
# re-initialises a stack the other set up (tests/scripts/test_ops_scripts.py runs both).
# .terraform/terraform.tfstate cannot tell: terraform writes it before it installs the providers,
# so a failed or interrupted init would otherwise be skipped on every later run. A stack
# initialised for another bucket (GCP_PROJECT changed) is re-initialised with -reconfigure, which
# switches to the new state without copying the old project's state into it. -lockfile=readonly:
# providers are checked against the committed lock file, never rewritten (make tf-lock does that).
# Terraform's output goes to stderr, so $(tf_out ...) stays clean when it initialises on the way.
# The bootstrap stack keeps local state; `scripts/up.sh init` initialises it, not this.
tf_init() {
  local stack="$1" dir marker want lock_sum="no-lock-file"
  [[ "$stack" != bootstrap ]] \
    || die "tf_init: the bootstrap stack keeps local state; scripts/up.sh init initialises it"
  dir="$(tf_dir "$stack")" || exit
  tf_env
  marker="$dir/.terraform/k3s-tf-init.ok"
  if [[ -f "$dir/.terraform.lock.hcl" ]]; then
    lock_sum="$(cksum < "$dir/.terraform.lock.hcl")" || die "cannot read $dir/.terraform.lock.hcl"
  fi
  want="gcs $TF_VAR_state_bucket $stack lock $lock_sum"
  if [[ -f "$marker" && "$(cat -- "$marker")" == "$want" ]]; then
    return 0
  fi
  rm -f -- "$marker"
  log "terraform init $stack (state in gs://$TF_VAR_state_bucket/$stack)"
  terraform -chdir="$dir" init -input=false -lockfile=readonly -reconfigure \
    -backend-config="bucket=$TF_VAR_state_bucket" -backend-config="prefix=$stack" >&2 \
    || die "terraform init failed for $stack; it is retried on the next run until it succeeds"
  printf '%s\n' "$want" > "$marker" || die "cannot write $marker"
}

# tf_out STACK NAME / tf_out_json STACK NAME: one output of STACK, raw or as JSON. A stack on the
# gcs backend is initialised first if needed (fresh checkout), which needs load_env.
tf_out() {
  _tf_output "$1" -raw "$2"
}

tf_out_json() {
  _tf_output "$1" -json "$2"
}

_tf_output() {
  local dir
  dir="$(tf_dir "$1")" || exit
  if [[ "$1" != bootstrap ]]; then tf_init "$1" || return; fi
  terraform -chdir="$dir" output "$2" "$3"
}

# tf_apply STACK [--var-file FILE] [--no-destroy | --destroy-only NAME...]: tf_plan, then
# tf_apply_plan: plan STACK into a saved plan, then apply exactly that plan once confirm agrees (under
# AUTO_APPROVE=1 confirm agrees outright, as the brief says). No changes: nothing is asked or applied,
# so re-runs are quiet. A plan that destroys or replaces anything lists those addresses first.
#
# The two guard options are opt-in for plans that spec §3.1 has a human review (base; workers when
# scaled by hand). Under AUTO_APPROVE=1 they refuse a plan (exit 2) that destroys or replaces
# anything they do not allow; interactively the unexpected addresses are listed again and the
# question is still asked.
#   --no-destroy           allows no destroy at all. Callers: up.sh base (k3s-cp), up.sh workers and
#                          an add-only cpu-pool.sh scale (a worker that left the inventory without
#                          cpu-pool.sh was never drained).
#   --destroy-only NAME... allows only addresses keyed ["NAME"] (for_each keys: a worker's instance,
#                          address and per-worker rules). Callers: cpu-pool.sh scale-down and remove,
#                          with exactly the workers they drain, so k3s-cpu-1 or any other worker is
#                          never destroyed or replaced unattended.
#   --var-file FILE        plans with -var-file=FILE (an absolute path): its values override the
#                          stack's *.auto.tfvars files, and the saved plan keeps them. cpu-pool.sh plans
#                          the new inventory this way before it drains anything or writes the file.
# Declining exits 2. STACK must be initialised already (tf_init; `terraform init` for bootstrap).
# tf_plan takes the stack's lock (stack_lock STACK) before it plans, and this process keeps it until it
# exits: another run that plans or applies STACK meanwhile is refused (exit 2).
tf_apply() {
  tf_plan "$@"
  tf_apply_plan "$1"
}

# tf_plan STACK [options]: the first half of tf_apply (same options): plan, list what the plan
# destroys or replaces, apply the guard (refusing exits 2 and keeps no plan), and set TF_PLAN to the
# saved plan, $RUN_DIR/STACK.tfplan, or to "" when there is nothing to change. Call it outside $(...)
# so TF_PLAN survives. A caller can do work between the check and the apply (cpu-pool.sh drains the
# workers the plan destroys) and then call tf_apply_plan STACK.
tf_plan() {
  local stack="$1" opt dir plan rc=0 doomed unexpected addr name ok
  local -a allowed=() extra=()
  TF_PLAN=""
  shift
  if [[ "${1:-}" == --var-file ]]; then
    (( $# >= 2 )) || die "tf_plan: --var-file needs a file"
    [[ "$2" == /* && -f "$2" ]] || die "tf_plan: --var-file needs an existing absolute path: '$2'"
    extra=("-var-file=$2")
    shift 2
  fi
  opt="${1:-}"
  case "$opt" in
    "" | --no-destroy) (( $# <= 1 )) || die "tf_apply: unexpected arguments after $opt" ;;
    --destroy-only)
      (( $# >= 2 )) || die "tf_apply: --destroy-only needs at least one name"
      allowed=("${@:2}") ;;
    *) die "tf_apply: unknown option $opt" ;;
  esac
  dir="$(tf_dir "$stack")" || exit
  tf_env
  stack_lock "$stack"
  run_dir || die "cannot create $RUN_DIR"
  plan="$RUN_DIR/$stack.tfplan"
  rm -f -- "$plan"
  terraform -chdir="$dir" plan -input=false -lock-timeout=120s -detailed-exitcode ${extra[@]+"${extra[@]}"} \
    -out="$plan" || rc=$?
  case "$rc" in
    0) rm -f -- "$plan"; log "$stack: no changes"; return 0 ;;
    2) ;;
    *) rm -f -- "$plan"; die "terraform plan failed for $stack (exit $rc)" ;;
  esac
  doomed="$(terraform -chdir="$dir" show -json "$plan" \
    | jq -r '.resource_changes[]? | select(.change.actions | index("delete")) | .address')" \
    || { rm -f -- "$plan"; die "cannot read the saved plan for $stack"; }
  if [[ -n "$doomed" ]]; then
    log "WARNING: this plan DESTROYS or REPLACES in the $stack stack:"
    log "$(printf '%s\n' "$doomed" | sed 's/^/  - /')"
    if [[ -n "$opt" ]]; then
      unexpected=""
      while read -r addr; do
        ok=0
        for name in ${allowed[@]+"${allowed[@]}"}; do
          [[ "$addr" == *"[\"$name\"]"* ]] && ok=1
        done
        (( ok )) || unexpected+="  - $addr"$'\n'
      done <<< "$doomed"
      if [[ -n "$unexpected" ]]; then
        log "WARNING: of these, NOT expected by this command (${allowed[*]:-no destroys allowed}):"
        log "${unexpected%$'\n'}"
        if [[ "${AUTO_APPROVE:-}" == 1 ]]; then
          rm -f -- "$plan"
          die "refusing to destroy or replace unexpected resources in $stack under AUTO_APPROVE=1: nothing was applied; run it interactively to review" 2
        fi
      fi
    fi
  fi
  TF_PLAN="$plan"
}

# tf_apply_plan STACK [--confirmed]: the second half of tf_apply: ask, then apply exactly the plan
# tf_plan saved ($TF_PLAN) and remove it. Nothing to do when TF_PLAN is "" (no changes). --confirmed:
# the caller asked this question itself (confirm) and got a yes, so it is not asked again and no
# AUTO_APPROVE line is logged for an operator's yes (cpu-pool.sh --guarded checks the plan in between).
tf_apply_plan() {
  local stack="$1" dir plan="${TF_PLAN:-}" rc=0 confirmed=0
  case "${2:-}" in
    "") ;;
    --confirmed) confirmed=1 ;;
    *) die "tf_apply_plan: unknown option ${2}" ;;
  esac
  (( $# <= 2 )) || die "tf_apply_plan: unexpected arguments after $2"
  [[ -n "$plan" ]] || return 0
  [[ "$plan" == "$RUN_DIR/$stack.tfplan" && -f "$plan" ]] || die "tf_apply_plan: no saved plan for $stack"
  dir="$(tf_dir "$stack")" || exit
  tf_env
  stack_lock "$stack"
  if (( confirmed == 0 )) && ! confirm "Apply this plan to the $stack stack?"; then
    rm -f -- "$plan"
    TF_PLAN=""
    die "not applied: $stack" 2
  fi
  terraform -chdir="$dir" apply -input=false -lock-timeout=120s "$plan" || rc=$?
  rm -f -- "$plan"
  TF_PLAN=""
  [[ "$rc" == 0 ]] || die "terraform apply failed for $stack (exit $rc)"
}

# kc ARGS...: kubectl against this cluster (reached through `make tunnel`).
kc() {
  kubectl --kubeconfig "$KUBECONFIG_FILE" "$@"
}

# cp_resolve: check GCP_PROJECT and look up k3s-cp's zone (tf_out base cp_zone) once per shell,
# into CP_ZONE. Callers that run gcloud themselves (tunnel.sh) call it first.
cp_resolve() {
  tf_env
  if [[ -z "${CP_ZONE:-}" ]]; then
    CP_ZONE="$(tf_out base cp_zone)" || die "cannot read cp_zone from the base stack (run make up first)"
    [[ "$CP_ZONE" =~ ^[a-z]+-[a-z]+[0-9]+-[a-z]$ ]] || die "unexpected cp_zone output: $CP_ZONE"
  fi
}

# cp_ssh CMD: run CMD on k3s-cp over IAP SSH (OS Login); stdin is forwarded. ConnectTimeout bounds a
# hung connect; ServerAliveInterval notices a dead connection during long commands.
cp_ssh() {
  cp_resolve || return
  gcloud compute ssh "$CP_NAME" --project "$GCP_PROJECT" --zone "$CP_ZONE" --tunnel-through-iap --quiet \
    --ssh-flag=-oConnectTimeout=20 --ssh-flag=-oServerAliveInterval=30 --command "$1"
}

# cp_scp SRC... DST: copy local SRCs (files or directories) to DST on k3s-cp over IAP. As with
# scp -r, a directory SRC becomes DST if DST does not exist yet, and DST/<name> if it does.
cp_scp() {
  (( $# >= 2 )) || die "usage: cp_scp SRC... DST"
  local dst="${!#}"
  cp_resolve || return
  gcloud compute scp --project "$GCP_PROJECT" --zone "$CP_ZONE" --tunnel-through-iap --quiet \
    --recurse --scp-flag=-oConnectTimeout=20 "${@:1:$#-1}" "$CP_NAME:$dst"
}

# confirm PROMPT: ask on stdin; only y or yes (any case) agrees, anything else or EOF refuses (1).
# AUTO_APPROVE=1 agrees without reading anything.
confirm() {
  local answer=""
  if [[ "${AUTO_APPROVE:-}" == 1 ]]; then
    log "$1 [AUTO_APPROVE=1: yes]"
    return 0
  fi
  printf '%s [y/N] ' "$1" >&2
  read -r answer || true
  [[ "$answer" =~ ^([yY]|[yY][eE][sS])$ ]]
}

# wait_for CMD TIMEOUT_S [STOP_RC]: eval CMD (a string, so shell functions such as kc and cp_ssh
# work) every WAIT_INTERVAL_S seconds (default 5) until it succeeds. Returns 1 after TIMEOUT_S
# seconds, with the last attempt's output on stderr (TIMEOUT_S counts bash SECONDS, which step on
# whole wall-clock seconds: a TIMEOUT_S of 1 can expire after a few ms; N gives at least N-1 s).
# With STOP_RC, an attempt that exits STOP_RC (a failure that no retry can clear) ends the wait
# at once: wait_for returns STOP_RC, with that attempt's output on stderr. Each attempt runs in a
# subshell: a CMD that dies (cp_ssh -> tf_out -> die) is a failed attempt whose message is shown,
# not a silent exit of the whole script; variables CMD sets do not outlive the attempt. CMD must
# bound its own runtime (kubectl --timeout, ConnectTimeout), or one hung attempt outlasts TIMEOUT_S.
wait_for() {
  local cmd="$1" timeout="$2" stop="${3:-}" interval="${WAIT_INTERVAL_S:-5}" deadline last rc
  [[ "$timeout" =~ ^[0-9]+$ ]] || die "wait_for: bad timeout '$timeout'"
  [[ -z "$stop" || "$stop" =~ ^[1-9][0-9]{0,2}$ ]] || die "wait_for: bad stop code '$stop'"
  [[ "$interval" =~ ^[0-9]+$ ]] || die "wait_for: bad WAIT_INTERVAL_S '$interval'"
  last="$(mktemp)" || die "wait_for: mktemp failed"
  deadline=$((SECONDS + timeout))
  log "waiting up to ${timeout}s for: $cmd"
  while rc=0; (eval "$cmd") > "$last" 2>&1 || rc=$?; (( rc != 0 )); do
    if [[ -n "$stop" ]] && (( rc == stop )); then
      log "stopped waiting for: $cmd (exit $rc, which no retry clears); the attempt said:"
      tail -n 20 -- "$last" >&2
      rm -f -- "$last"
      return "$rc"
    fi
    if (( SECONDS >= deadline )); then
      log "timed out after ${timeout}s waiting for: $cmd; the last attempt said:"
      tail -n 20 -- "$last" >&2
      rm -f -- "$last"
      return 1
    fi
    sleep "$interval"
  done
  rm -f -- "$last"
}

# ---- CPU workers and GPU nodes (Task 14) ------------------------------------------------------

# Node names (global constraints): CPU workers k3s-cpu-<n>, GPU nodes k3s-gpu-<pool>-<n> (gpuctl's
# planner: pool [a-z0-9]{1,20}). Anchored, so k3s-cp is neither, and no number has a leading zero.
CPU_NODE_RE='^k3s-cpu-[1-9][0-9]*$'
GPU_NODE_RE='^k3s-gpu-[a-z0-9]{1,20}-[1-9][0-9]*$'
# The private staging directory for node/ on a host, made fresh by the operator's account with mode
# 0700. mkdir (no -p) fails if anything recreated the path after rm, and the owner check catches a
# mkdir that is not atomic, so nobody else on the host can place files there for root to run.
# Used by up.sh (k3s-cp) and join-node.sh (workers, GPU nodes).
# shellcheck disable=SC2034
NODE_STAGING_CMD='rm -rf /tmp/k3s-node && mkdir -m 0700 /tmp/k3s-node && [ ! -L /tmp/k3s-node ] && [ -O /tmp/k3s-node ]'

# is_ipv4 TEXT: one dotted-quad IPv4 address (each octet 0-255, no leading zeros).
is_ipv4() {
  local octet='(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
  [[ "$1" =~ ^$octet\.$octet\.$octet\.$octet$ ]]
}

# node_kind NAME: prints cpu or gpu; fails (1) for any other name.
node_kind() {
  if [[ "$1" =~ $CPU_NODE_RE ]]; then
    echo cpu
  elif [[ "$1" =~ $GPU_NODE_RE ]]; then
    echo gpu
  else
    return 1
  fi
}

# _worker_zone NAME: set _ZONE to NAME's zone from `tf_out_json workers worker_zones`. The output is
# read once per shell (_WORKER_ZONES); call it outside $(...) so that cache survives.
_worker_zone() {
  [[ "$1" =~ $CPU_NODE_RE ]] || die "not a CPU worker name: '$1'"
  if [[ -z "${_WORKER_ZONES:-}" ]]; then
    _WORKER_ZONES="$(tf_out_json workers worker_zones)" || die "cannot read worker_zones from the workers stack"
  fi
  _ZONE="$(jq -r --arg n "$1" 'if type == "object" then .[$n] // "" else error("not an object") end' \
    <<< "$_WORKER_ZONES")" || die "unexpected worker_zones output from the workers stack"
  [[ -n "$_ZONE" ]] || die "$1 is not in the workers stack (worker_zones)"
  [[ "$_ZONE" =~ ^[a-z]+-[a-z]+[0-9]+-[a-z]$ ]] || die "unexpected zone for $1: '$_ZONE'"
}

# vm_status NAME: the GCE status of CPU worker NAME's VM (RUNNING, TERMINATED, SUSPENDED, ...) in
# VM_STATUS, and its zone in _ZONE. Call it outside $(...): _worker_zone caches worker_zones for the
# shell. Used by cpu-pool.sh (scale step 4) and join-node.sh (a stopped VM fails at once).
vm_status() {
  tf_env
  _worker_zone "$1"
  VM_STATUS="$(gcloud compute instances describe "$1" --project "$GCP_PROJECT" --zone "$_ZONE" \
    --format='value(status)')" || die "cannot read the status of the VM $1"
  [[ "$VM_STATUS" =~ ^[A-Z_]+$ ]] || die "unexpected status for the VM $1: '$VM_STATUS'"
}

# worker_ssh NAME CMD: run CMD on CPU worker NAME over IAP SSH (OS Login), as cp_ssh does on k3s-cp;
# stdin is forwarded.
worker_ssh() {
  tf_env
  _worker_zone "$1"
  gcloud compute ssh "$1" --project "$GCP_PROJECT" --zone "$_ZONE" --tunnel-through-iap --quiet \
    --ssh-flag=-oConnectTimeout=20 --ssh-flag=-oServerAliveInterval=30 --command "$2"
}

# worker_scp NAME SRC... DST: copy local SRCs (recursively) to DST on CPU worker NAME over IAP.
worker_scp() {
  (( $# >= 3 )) || die "usage: worker_scp NAME SRC... DST"
  local name="$1" dst="${!#}"
  shift
  tf_env
  _worker_zone "$name"
  gcloud compute scp --project "$GCP_PROJECT" --zone "$_ZONE" --tunnel-through-iap --quiet \
    --recurse --scp-flag=-oConnectTimeout=20 "${@:1:$#-1}" "$name:$dst"
}

# gpu_target NAME: ubuntu@IP for GPU node NAME. IP is GPU_SSH_IP (join-node.sh sets it to the IP it
# was given) or else what `scripts/gpuctl ip NAME` reports; either way it must be one IPv4 address.
gpu_target() {
  local ip="${GPU_SSH_IP:-}"
  [[ "$1" =~ $GPU_NODE_RE ]] || die "not a GPU node name: '$1'"
  if [[ -z "$ip" ]]; then
    ip="$("$REPO_ROOT/scripts/gpuctl" ip "$1")" || die "cannot look up the IP of $1 (scripts/gpuctl ip)"
  fi
  is_ipv4 "$ip" || die "the IP of $1 is not an IPv4 address: '$ip'"
  printf 'ubuntu@%s\n' "$ip"
}

# _gpu_ssh_opts NAME: set _GPU_SSH_OPTS, the ssh/scp options for GPU node NAME. One known_hosts file
# per node (.run/known_hosts_NAME, which gpuctl deletes with the node): Lambda hands old IPs to new
# instances, so one shared file would reject a new node's key or vouch for the wrong host.
_gpu_ssh_opts() {
  [[ -n "${SSH_PRIVATE_KEY_PATH:-}" && -f "$SSH_PRIVATE_KEY_PATH" ]] \
    || die "SSH_PRIVATE_KEY_PATH in .env must name the Lambda SSH private key: '${SSH_PRIVATE_KEY_PATH:-}'" 2
  run_dir || die "cannot create $RUN_DIR"
  # BatchMode: never a password or passphrase prompt. IdentitiesOnly: only this key, not every key an
  # ssh-agent holds (those could use up the server's MaxAuthTries first).
  _GPU_SSH_OPTS=(-i "$SSH_PRIVATE_KEY_PATH" -o StrictHostKeyChecking=accept-new
    -o "UserKnownHostsFile=$RUN_DIR/known_hosts_$1" -o ConnectTimeout=20 -o ServerAliveInterval=30
    -o BatchMode=yes -o IdentitiesOnly=yes)
}

# gpu_ssh NAME CMD: run CMD on GPU node NAME as ubuntu over plain SSH (Lambda admits tcp/22 from
# OPERATOR_IP only); stdin is forwarded.
gpu_ssh() {
  local target
  target="$(gpu_target "$1")" || return
  _gpu_ssh_opts "$1"
  # shellcheck disable=SC2029  # CMD is meant to run remotely, as with cp_ssh's --command
  ssh "${_GPU_SSH_OPTS[@]}" "$target" "$2"
}

# gpu_scp NAME SRC... DST: copy local SRCs (recursively) to DST on GPU node NAME.
gpu_scp() {
  (( $# >= 3 )) || die "usage: gpu_scp NAME SRC... DST"
  local name="$1" dst="${!#}" target
  shift
  target="$(gpu_target "$name")" || return
  _gpu_ssh_opts "$name"
  scp -r "${_GPU_SSH_OPTS[@]}" "${@:1:$#-1}" "$target:$dst"
}

# node_ssh NAME CMD / node_scp NAME SRC... DST: cp_* for k3s-cp (the verification checks run on
# every node), worker_* for k3s-cpu-<n>, gpu_* for k3s-gpu-*. join-node.sh never passes k3s-cp: it
# accepts only worker and GPU names (node_kind).
node_ssh() {
  if [[ "$1" == "$CP_NAME" ]]; then
    cp_ssh "$2"
    return
  fi
  case "$(node_kind "$1")" in
    cpu) worker_ssh "$@" ;;
    gpu) gpu_ssh "$@" ;;
    *) die "not k3s-cp, a CPU worker or a GPU node name: '$1'" ;;
  esac
}

node_scp() {
  if [[ "$1" == "$CP_NAME" ]]; then
    shift
    cp_scp "$@"
    return
  fi
  case "$(node_kind "$1")" in
    cpu) worker_scp "$@" ;;
    gpu) gpu_scp "$@" ;;
    *) die "not k3s-cp, a CPU worker or a GPU node name: '$1'" ;;
  esac
}

# KC_REQUEST_TIMEOUT: the bound of one kubectl request (get, apply, label, delete) through the API
# tunnel, so a half-stalled tunnel fails instead of hanging. Waits carry their own --timeout.
KC_REQUEST_TIMEOUT=60s

# node_state NAME: prints Ready, NotReady (registered, not Ready) or Absent (not registered). Any
# other kubectl failure (the tunnel is down, another cluster) dies: it says nothing about the node.
# Only stdout is the jsonpath value: kubectl can succeed and still print a warning or a discovery
# error on stderr, which must not make a Ready node look NotReady.
node_state() {
  local out err errf rc=0
  errf="$(mktemp)" || die "node_state: mktemp failed"
  out="$(kc get node "$1" --request-timeout="$KC_REQUEST_TIMEOUT" \
    -o 'jsonpath={.status.conditions[?(@.type=="Ready")].status}' 2> "$errf")" || rc=$?
  err="$(cat -- "$errf")"
  rm -f -- "$errf"
  if (( rc != 0 )); then
    if [[ "$err" == *"nodes \"$1\" not found"* ]]; then
      echo Absent
      return 0
    fi
    die "cannot read node $1 from the cluster (is the API tunnel up? make tunnel): $err"
  fi
  if [[ "$out" == True ]]; then echo Ready; else echo NotReady; fi
}

# node_ready NAME: 0 when node NAME is Ready, 1 when it is not Ready or not registered; dies as
# node_state does.
node_ready() {
  local state
  state="$(node_state "$1")" || exit
  [[ "$state" == Ready ]]
}

# ---- Cluster lookups for the verification checks (Task 18) ------------------------------------

# A Kubernetes object name (RFC 1123 subdomain, as node and Pod names are). Names read back from the
# cluster are checked before they reach a selector or a command line.
K8S_NAME_RE='^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$'

# node_external_ip NAME: NAME's ExternalIP (status.addresses), which is its public IP (k3s
# node-external-ip). Fails unless the cluster reports exactly one IPv4 address.
node_external_ip() {
  local ip
  [[ "$1" =~ $K8S_NAME_RE ]] || die "not a node name: '$1'"
  ip="$(kc get node "$1" --request-timeout="$KC_REQUEST_TIMEOUT" \
    -o 'jsonpath={.status.addresses[?(@.type=="ExternalIP")].address}')" || return
  is_ipv4 "$ip" || { log "node $1 has no single IPv4 ExternalIP: '$ip'"; return 1; }
  printf '%s\n' "$ip"
}

# _nettest_pod_on APP NODE FIELD: FIELD (a jsonpath under the Pod) of the Running nettest Pod of
# app=APP (the echo and probe DaemonSets, k8s/nettest) on NODE; the first one if there are several.
_nettest_pod_on() {
  local out
  [[ "$2" =~ $K8S_NAME_RE ]] || die "not a node name: '$2'"
  out="$(kc -n nettest get pods -l "app=$1" --field-selector="spec.nodeName=$2,status.phase=Running" \
    --request-timeout="$KC_REQUEST_TIMEOUT" -o "jsonpath={.items[*].$3}")" || return
  out="${out%% *}"
  [[ -n "$out" ]] || { log "no Running $1 Pod on node $2 (namespace nettest)"; return 1; }
  printf '%s\n' "$out"
}

# probe_on NODE: the name of the Running nettest probe Pod (netshoot) on NODE, for kubectl exec.
probe_on() {
  local name
  name="$(_nettest_pod_on probe "$1" metadata.name)" || return
  [[ "$name" =~ $K8S_NAME_RE ]] || { log "unexpected probe Pod name on $1: '$name'"; return 1; }
  printf '%s\n' "$name"
}

# echo_on NODE: the Pod IP of the Running nettest echo Pod (whoami, port 8080) on NODE.
echo_on() {
  local ip
  ip="$(_nettest_pod_on echo "$1" status.podIP)" || return
  is_ipv4 "$ip" || { log "the echo Pod on $1 has no IPv4 Pod IP: '$ip'"; return 1; }
  printf '%s\n' "$ip"
}

# _kc_node_gone_ok NAME ARGS...: kc ARGS, where "node NAME not found" counts as success (Review
# Focus #3). Only that message: a generic NotFound (another server) or another node's name still
# fails, as in gpuctl's Infra.drain_and_delete_node.
_kc_node_gone_ok() {
  local name="$1" out rc=0
  shift
  out="$(kc "$@" 2>&1)" || rc=$?
  (( rc == 0 )) && return 0
  if [[ "$out" == *"nodes \"$name\" not found"* || "$out" == *"node \"$name\" not found"* ]]; then
    log "node $name is not in the cluster: nothing to $1"
    return 0
  fi
  log "$out"
  return "$rc"
}

# drain_and_delete_node NAME: drain NAME (bounded) and delete it from the cluster, with the flags
# gpuctl uses for GPU nodes. A node that is already gone counts as success.
drain_and_delete_node() {
  log "draining $1 and deleting it from the cluster"
  _kc_node_gone_ok "$1" drain "$1" --ignore-daemonsets --delete-emptydir-data --force --timeout=120s \
    || die "cannot drain $1"
  _kc_node_gone_ok "$1" delete node "$1" --request-timeout="$KC_REQUEST_TIMEOUT" || die "cannot delete node $1"
}

# worker_ip_lines: "NAME IP" for every CPU worker in `tf_out_json workers worker_ips`, in numeric
# order (k3s-cpu-9 before k3s-cpu-10). Every name and IP is checked: both reach command lines.
worker_ip_lines() {
  local json lines name ip
  json="$(tf_out_json workers worker_ips)" || die "cannot read worker_ips from the workers stack"
  lines="$(jq -r 'if type == "object" then to_entries[] | "\(.key) \(.value)" else error("not an object") end' \
    <<< "$json")" || die "unexpected worker_ips output from the workers stack"
  while read -r name ip; do
    [[ -n "$name" ]] || continue
    [[ "$name" =~ $CPU_NODE_RE ]] || die "unexpected worker name in worker_ips: '$name'"
    is_ipv4 "$ip" || die "worker_ips: $name has no IPv4 address: '$ip'"
    printf '%s %s\n' "$name" "$ip"
  done <<< "$lines" | sort -t- -k3,3n
}

# join_workers --all | join_workers [NAME...]: join-node.sh --skip-sync for every worker in the
# workers stack's worker_ips (up.sh workers), or only for the NAMEs (cpu-pool.sh scale: the workers
# that have not joined yet; none is fine), in numeric order. Each NAME must be in worker_ips. When
# any worker is to be joined, sync-peers runs once first, so its IP is in k3s-cp's k3s_peers set
# before its agent connects (spec §5.2 step 1); sync-peers also runs last, every time (Task 14
# brief: the up.sh workers stage, scale step 5). join-node.sh leaves a Ready worker alone, so this
# is safe to re-run.
join_workers() {
  local lines name ip line want
  local -a join=()
  lines="$(worker_ip_lines)" || exit
  if [[ "${1:-}" == --all ]]; then
    while read -r line; do
      [[ -n "$line" ]] && join+=("$line")
    done <<< "$lines"
  else
    for want in "$@"; do
      [[ "$want" =~ $CPU_NODE_RE ]] || die "not a CPU worker name: '$want'"
      grep -q "^$want " <<< "$lines" || die "$want is not in the workers stack's worker_ips: it cannot be joined"
    done
    while read -r name ip; do
      for want in "$@"; do
        [[ "$name" == "$want" ]] && join+=("$name $ip")
      done
    done <<< "$lines"
  fi
  if (( ${#join[@]} )); then
    "$REPO_ROOT/scripts/gpuctl" sync-peers || die "sync-peers failed"
    for line in "${join[@]}"; do
      read -r name ip <<< "$line"
      "$REPO_ROOT/scripts/join-node.sh" --node "$name" --external-ip "$ip" --skip-sync < /dev/null \
        || die "join-node.sh failed for $name"
    done
  fi
  "$REPO_ROOT/scripts/gpuctl" sync-peers || die "sync-peers failed"
}
