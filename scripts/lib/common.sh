# shellcheck shell=bash
# Shared helpers for the operator scripts. Source it; do not run it.
# Task 7: REPO_ROOT, log, die, load_env. Task 10: the Terraform, IAP SSH, kubectl and prompt helpers
# below. Kept to bash 3.2 (macOS operators): no bash-4-only features.

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

# tf_dir STACK: the stack's directory, $REPO_ROOT/infra/STACK; dies for anything else.
tf_dir() {
  [[ "$1" =~ ^[a-z][a-z0-9-]*$ && -d "$REPO_ROOT/infra/$1" ]] || die "unknown Terraform stack: $1"
  printf '%s\n' "$REPO_ROOT/infra/$1"
}

# tf_init STACK: init STACK on the gcs backend, bucket $GCP_PROJECT-k3s-tfstate, prefix STACK.
# Skipped when the last init of this stack succeeded for exactly that bucket and prefix. A marker,
# .terraform/k3s-tf-init.ok, is removed before every init and written only after it succeeds.
# .terraform/terraform.tfstate cannot tell: terraform writes it before it installs the providers,
# so a failed or interrupted init would otherwise be skipped on every later run. A stack
# initialised for another bucket (GCP_PROJECT changed) is re-initialised with -reconfigure, which
# switches to the new state without copying the old project's state into it. -lockfile=readonly:
# providers are checked against the committed lock file, never rewritten (make tf-lock does that).
# Terraform's output goes to stderr, so $(tf_out ...) stays clean when it initialises on the way.
# The bootstrap stack keeps local state; `scripts/up.sh init` initialises it, not this.
tf_init() {
  local stack="$1" dir marker want
  [[ "$stack" != bootstrap ]] \
    || die "tf_init: the bootstrap stack keeps local state; scripts/up.sh init initialises it"
  dir="$(tf_dir "$stack")" || exit
  tf_env
  marker="$dir/.terraform/k3s-tf-init.ok"
  want="gcs $TF_VAR_state_bucket $stack"
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

# tf_apply STACK [--no-destroy]: plan STACK into a saved plan, then apply exactly that plan once
# confirm agrees (under AUTO_APPROVE=1 confirm agrees outright, as the brief says). No changes:
# nothing is asked or applied, so re-runs are quiet. A plan that destroys or replaces anything
# lists those addresses first. --no-destroy is opt-in, for a stack that must never be recreated
# unattended (up.sh base: k3s-cp). Under AUTO_APPROVE=1 it refuses such a plan (exit 2);
# interactively it still asks. Declining exits 2. STACK must be initialised already (tf_init;
# `terraform init` for bootstrap).
tf_apply() {
  local stack="$1" opt="${2:-}" dir plan rc=0 doomed
  [[ -z "$opt" || "$opt" == --no-destroy ]] || die "tf_apply: unknown option $opt"
  dir="$(tf_dir "$stack")" || exit
  tf_env
  run_dir || die "cannot create $RUN_DIR"
  plan="$RUN_DIR/$stack.tfplan"
  rm -f -- "$plan"
  terraform -chdir="$dir" plan -input=false -lock-timeout=120s -detailed-exitcode -out="$plan" || rc=$?
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
    if [[ "${AUTO_APPROVE:-}" == 1 && "$opt" == --no-destroy ]]; then
      rm -f -- "$plan"
      die "refusing to destroy or replace resources in $stack under AUTO_APPROVE=1; run it interactively to review" 2
    fi
  fi
  if ! confirm "Apply this plan to the $stack stack?"; then
    rm -f -- "$plan"
    die "not applied: $stack" 2
  fi
  rc=0
  terraform -chdir="$dir" apply -input=false -lock-timeout=120s "$plan" || rc=$?
  rm -f -- "$plan"
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

# wait_for CMD TIMEOUT_S: eval CMD (a string, so shell functions such as kc and cp_ssh work) every
# WAIT_INTERVAL_S seconds (default 5) until it succeeds. Returns 1 after TIMEOUT_S seconds, with
# the last attempt's output on stderr. Each attempt runs in a subshell: a CMD that dies (cp_ssh ->
# tf_out -> die) is a failed attempt whose message is shown, not a silent exit of the whole script;
# variables CMD sets do not outlive the attempt. CMD must bound its own runtime (kubectl --timeout,
# ConnectTimeout), or one hung attempt outlasts TIMEOUT_S.
wait_for() {
  local cmd="$1" timeout="$2" interval="${WAIT_INTERVAL_S:-5}" deadline last
  [[ "$timeout" =~ ^[0-9]+$ ]] || die "wait_for: bad timeout '$timeout'"
  [[ "$interval" =~ ^[0-9]+$ ]] || die "wait_for: bad WAIT_INTERVAL_S '$interval'"
  last="$(mktemp)" || die "wait_for: mktemp failed"
  deadline=$((SECONDS + timeout))
  log "waiting up to ${timeout}s for: $cmd"
  until (eval "$cmd") > "$last" 2>&1; do
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
