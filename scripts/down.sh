#!/usr/bin/env bash
# make down (spec §11): tear the whole environment down, in order. Takes no arguments.
#
#   1. ask once: type the GCP project id (GCP_PROJECT in .env) to go on; anything else, or end of input,
#      exits 2 with nothing changed. AUTO_APPROVE=1 (exactly 1, from the caller's environment) skips the
#      question. Nothing after it asks again. .env is sourced (load_env), but it may not set or unset
#      AUTO_APPROVE, REPO_ROOT, RUN_DIR, KUBECONFIG_FILE, CPU_NODE_RE, CP_NAME, KC_REQUEST_TIMEOUT or
#      DESTROY_STACKS: they are readonly while it loads, and a .env line that touches one, or that fails
#      for any other reason, refuses the run (exit 2, nothing changed). Terraform's own overrides
#      TF_WORKSPACE, TF_DATA_DIR and TF_CLI_ARGS[_*] would point the destroys at another workspace, data
#      directory or -target: set in .env or in the caller's environment, they refuse the run too.
#   2. scripts/gpuctl down --auto-approve: every Lambda GPU node of this cluster (drained, deleted from
#      the cluster, terminated) and the autoscaler overrides.
#   3. drain every CPU worker (k3s-cpu-<n>) the cluster lists and delete its Node. An unreachable
#      cluster (tunnel down, k3s-cp gone, no kubeconfig) skips this step with a warning, and so does a
#      cluster that is not this project's (Node k3s-cp's ExternalIP is not the base stack's cp_ip:
#      ~/.kube/hybrid-gpu.yaml and the tunnel port are one per user, not one per project). A failed
#      drain or delete is only a warning: the destroys below remove the VMs and the cluster anyway.
#      Every kubectl call is killed after DOWN_KC_KILL_S seconds (default 180), so a half-open tunnel
#      cannot hang the teardown.
#   4. terraform destroy -auto-approve -input=false for gpu-access, workers and base, in that order,
#      each after tf_init and with the TF_VAR_* that tf_env sets. A stack with nothing in its state
#      (never applied, or destroyed by an earlier run) destroys nothing and succeeds.
#   5. scripts/tunnel.sh stop; remove .run/known_hosts_* (the GPU nodes' host keys) and the kubeconfig
#      ~/.kube/hybrid-gpu.yaml: it is the admin credential of a cluster that no longer exists, and
#      make up (make kubeconfig) writes a new one.
#   6. the bootstrap stack (the state bucket <GCP_PROJECT>-k3s-tfstate, the enabled APIs) is KEPT.
#   7. ask Lambda once more (scripts/gpuctl list-nodes): a GPU node of this cluster that is still live
#      (started after step 2 by a stretch autoscaler still running, or by a concurrent gpu-apply) keeps
#      billing with nothing left to join, the largest cost risk of spec §13. A listed node, or a list
#      that fails, fails the run (exit 1) and says what to do: terminate it in the Lambda console, since
#      make gpu-down drains each node through the cluster, which is gone by then.
#
# Asking Lambda (steps 2 and 7). Lambda may still list an instance as active for a moment after its
# terminate call returned, so a listed node is asked about again after a pause, until the pauses add up
# to DOWN_LAMBDA_WAIT_S seconds (default 60; 5 s at most each; 0 asks once): N seconds ask 1 + ceil(N/5)
# times, however long each list-nodes call takes. A terminating or terminated instance is gone (it no
# longer bills); an active, booting or unhealthy one still listed after the last ask is live.
#
# Failures, Ctrl-C and re-runs. Every step is safe to repeat, so a teardown that stopped part-way is
# finished by running make down again. The first failing step stops the run (exit 1) and says what is
# left. Ctrl-C stops the run where it is (exit 130): kubectl runs under timeout --foreground, so the
# interrupt reaches it, and the INT trap keeps every later step from running.
#   - gpuctl down failed: the run asks Lambda (scripts/gpuctl list-nodes) whether any GPU node of this
#     cluster is still live (see "Asking Lambda"). If one is, or Lambda cannot be asked, it stops
#     before any destroy: a GPU node left running after gpu-access and base are gone keeps billing,
#     outside the firewall model, with no cluster to join. If none is, the failure was in gpuctl down's firewall and peer step
#     (on a re-run after the workers or base stack is destroyed, sync-peers cannot read their outputs,
#     Ruling R29), which the destroys below make moot: the run goes on.
#   - a terraform init or destroy failed: the stacks after it are not touched (each one reads the base
#     stack's state, so base goes last).
#
# The Lambda API key reaches scripts/gpuctl only (it loads .env itself): it is unset here, as in up.sh.
set -euo pipefail

# shellcheck disable=SC1091  # common.sh is linted on its own
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

usage() {
  log "usage: ${0##*/}   (make down; AUTO_APPROVE=1 skips the question)"
  exit 2
}

# Step 1: the one confirmation. Refusing exits 2 before anything has changed.
confirm_project() {
  local answer=""
  log "make down destroys, in GCP project $GCP_PROJECT:"
  log "  - every Lambda GPU node of this cluster (scripts/gpuctl down)"
  log "  - the CPU workers, the gpu-access, workers and base stacks: k3s-cp, the worker VMs, the VPCs,"
  log "    the static IPs and the firewall rules"
  log "It keeps the bootstrap stack: the Terraform state bucket gs://$GCP_PROJECT-k3s-tfstate and the enabled APIs."
  log "Stop the stretch autoscaler first if it is running (Ctrl-C it): it could start a GPU node again."
  if [[ "${AUTO_APPROVE:-}" == 1 ]]; then
    log "[AUTO_APPROVE=1: going on without asking]"
    return 0
  fi
  printf 'Type the GCP project id (%s) to destroy all of this: ' "$GCP_PROJECT" >&2
  read -r answer || true
  if [[ "$answer" != "$GCP_PROJECT" ]]; then
    log "make down cancelled: the answer was not the project id; nothing was changed"
    exit 2
  fi
}

# interrupted CODE: the INT/TERM trap. bash runs it as soon as the command it was waiting for ends.
interrupted() {
  log "make down interrupted: it stopped here, and no later step runs (what is destroyed stays destroyed); re-run make down to finish"
  exit "$1"
}

# env_refused STATUS: the EXIT trap while .env loads. load_env exits 2 itself, with its reason, for a
# missing or non-private .env; any other non-zero status is a line of .env that failed (bash printed
# why): an assignment to one of the readonly PINNED_NAMES, or a line that fails on its own (a command
# that fails, say). The message names a readonly variable only when .env mentions it.
env_refused() {
  local name named=""
  (( $1 != 0 )) || return 0
  [[ -f "$REPO_ROOT/.env" && "$(_file_mode "$REPO_ROOT/.env" 2>/dev/null)" == 600 ]] || return 0
  for name in $PINNED_NAMES; do
    # Only a line that is not a comment can set the name.
    if grep -Ev '^[[:space:]]*#' "$REPO_ROOT/.env" | grep -Eq "(^|[^A-Za-z0-9_])$name([^A-Za-z0-9_]|\$)"; then
      named="$named${named:+, }$name"
    fi
  done
  if [[ -n "$named" ]]; then
    log "error: a line of $REPO_ROOT/.env failed (see the bash error above), and .env names $named: make down keeps it readonly, since it decides whether the question is asked or what is destroyed, drained or removed; remove that line from .env. make down refused; nothing was changed"
  else
    log "error: a line of $REPO_ROOT/.env failed (see the bash error above); fix or remove it. make down refused; nothing was changed"
  fi
  exit 2
}

# tf_overrides_refused: Terraform's own environment overrides would make every terraform destroy below
# act on another workspace (TF_WORKSPACE: its empty state "destroys" fine), another data directory
# (TF_DATA_DIR) or only -target'ed resources (TF_CLI_ARGS, TF_CLI_ARGS_<command>), while make down
# reports each stack destroyed. Exported by .env (set -a) or by the caller, they refuse the run (S2-1).
tf_overrides_refused() {
  local set_names
  set_names="$(compgen -e | grep -E '^(TF_WORKSPACE|TF_DATA_DIR|TF_CLI_ARGS(_.*)?)$' | paste -sd ' ' -)" || true
  [[ -z "$set_names" ]] && return 0
  die "$set_names set (in $REPO_ROOT/.env or the environment make down runs in): terraform destroy would act on another workspace, data directory or -target, not this project's stacks. Remove it from .env or unset it, then re-run make down. make down refused; nothing was changed" 2
}

# live_gpu_nodes: Lambda's GPU nodes of this cluster that still bill, one line each (list-nodes' third
# field is the status: terminating and terminated ones do not count). Fails when Lambda cannot be asked.
live_gpu_nodes() {
  local listed
  listed="$("$REPO_ROOT/scripts/gpuctl" list-nodes < /dev/null)" || return 1
  printf '%s\n' "$listed" | awk 'NF && $3 != "terminating" && $3 != "terminated"'
}

# lambda_clear: ask Lambda until it lists no live GPU node of this cluster, pausing up to LAMBDA_WAIT_S
# seconds in all (see "Asking Lambda" in the header). The window counts the pauses, not the clock, so a
# slow list-nodes call never shortens it. Returns 0 when none is live, 1 when LIVE_NODES (set here) are
# still live after the last ask, 2 when Lambda cannot be asked.
lambda_clear() {
  local waited=0 pause
  while true; do
    LIVE_NODES="$(live_gpu_nodes)" || return 2
    [[ -z "$LIVE_NODES" ]] && return 0
    (( waited < LAMBDA_WAIT_S )) || return 1
    pause=$(( LAMBDA_WAIT_S - waited ))
    (( pause <= 5 )) || pause=5
    log "Lambda still lists a GPU node of this cluster as live (a node just terminated can be listed as active for a moment): asking again in $pause s, for up to $LAMBDA_WAIT_S s in all"
    sleep "$pause"
    waited=$(( waited + pause ))
  done
}

# Step 2. On failure, Lambda decides whether the teardown may go on (see the header).
gpu_down() {
  local rc=0 clear=0
  log "==> GPU nodes: scripts/gpuctl down"
  "$REPO_ROOT/scripts/gpuctl" down --auto-approve < /dev/null || rc=$?
  (( rc == 0 )) && return 0
  log "scripts/gpuctl down failed (exit $rc); asking Lambda which GPU nodes of this cluster are still live"
  lambda_clear || clear=$?
  if (( clear == 2 )); then
    die "cannot list the GPU nodes on Lambda either: nothing was destroyed. Fix the error above (scripts/gpuctl status shows the GPU nodes) and re-run make down"
  fi
  if (( clear == 1 )); then
    log "GPU nodes still live on Lambda (they keep billing):"
    log "$(printf '%s\n' "$LIVE_NODES" | sed 's/^/  /')"
    die "scripts/gpuctl down did not remove them, so nothing was destroyed: the gpu-access and base stacks stay until they are gone. If the cluster is unreachable, bring the API tunnel up (make tunnel) and re-run make down; if it cannot be reached at all, terminate them in the Lambda console and re-run make down"
  fi
  log "warning: Lambda lists no GPU node of this cluster, so no GPU node is left outside the firewall model; going on (the destroys below remove the gpu-access stack and k3s-cp)"
  # gpuctl down removes the autoscaler overrides only when it succeeds.
  if [[ -e "$REPO_ROOT/lambda/.autoscale-state.json" ]]; then
    log "warning: lambda/.autoscale-state.json (the autoscaler's per-pool GPU targets) is still there, and the next scripts/gpuctl apply would start GPU nodes from it: remove it (rm lambda/.autoscale-state.json)"
  fi
}

# kc_bounded ARGS...: kubectl on the cluster kubeconfig, killed after KC_KILL_S seconds. --foreground
# keeps timeout and kubectl in the terminal's process group, so Ctrl-C reaches kubectl at once.
kc_bounded() {
  timeout --foreground "$KC_KILL_S" kubectl --kubeconfig "$KUBECONFIG_FILE" "$@"
}

# same_cluster: true when the cluster the kubeconfig reaches is this project's, i.e. Node k3s-cp's
# ExternalIP (node-external-ip, the static IP) is the base stack's cp_ip. The internal IP would not
# tell: two projects built from this repo give k3s-cp the same internal address.
same_cluster() {
  local node_ip base_ip
  node_ip="$(kc_bounded get node "$CP_NAME" -o 'jsonpath={.status.addresses[?(@.type=="ExternalIP")].address}' \
    --request-timeout=20s)" || node_ip=""
  base_ip="$(tf_out base cp_ip)" || base_ip=""
  [[ -n "$node_ip" && "$node_ip" == "$base_ip" ]] && return 0
  log "warning: the cluster $KUBECONFIG_FILE reaches may not be GCP project $GCP_PROJECT's: Node $CP_NAME has ExternalIP '${node_ip:-unknown}', the base stack's cp_ip is '${base_ip:-unknown}'; skipping the drains (the destroys below remove this project's workers anyway)"
  return 1
}

# Step 3. Never fails the run: the cluster is destroyed with k3s-cp in step 4.
drain_workers() {
  local listed name
  local -a workers=()
  log "==> CPU workers: drain and delete their Nodes"
  if ! listed="$(kc_bounded get nodes -o name --request-timeout=20s)"; then
    log "warning: cannot reach the cluster (is the API tunnel up? make tunnel): skipping the drains; the destroys below remove the workers and the cluster anyway"
    return 0
  fi
  while read -r name; do
    name="${name#node/}"
    [[ "$name" =~ $CPU_NODE_RE ]] && workers+=("$name")
  done <<< "$listed"
  if (( ${#workers[@]} == 0 )); then
    log "no CPU worker in the cluster"
    return 0
  fi
  same_cluster || return 0
  while read -r name; do
    log "draining $name and deleting it from the cluster"
    kc_bounded drain "$name" --ignore-daemonsets --delete-emptydir-data --force --timeout=120s \
      --request-timeout="$KC_REQUEST_TIMEOUT" \
      || log "warning: cannot drain $name; deleting its Node anyway (its VM is destroyed next)"
    kc_bounded delete node "$name" --request-timeout="$KC_REQUEST_TIMEOUT" \
      || log "warning: cannot delete the Node $name; going on (the cluster is destroyed with k3s-cp)"
  done <<< "$(printf '%s\n' "${workers[@]}" | sort -t- -k3,3n)"
}

# Step 4, for one stack.
destroy_stack() {
  local stack="$1" dir rc=0
  log "==> terraform destroy $stack"
  dir="$(tf_dir "$stack")" || exit
  tf_init "$stack"
  terraform -chdir="$dir" destroy -auto-approve -input=false || rc=$?
  (( rc == 0 )) \
    || die "terraform destroy failed for $stack (exit $rc): the stacks after it ($(stacks_after "$stack")) were not touched; fix the cause and re-run make down (it repeats every step; what is destroyed stays destroyed). 'Error acquiring the state lock' means another gpuctl or terraform run held this stack's state lock: re-run make down once that run has ended"
  log "$stack: destroyed"
}

# stacks_after STACK: the stacks DESTROY_STACKS lists after STACK, or "none".
stacks_after() {
  local after="${DESTROY_STACKS#*"$1"}"
  after="${after# }"
  printf '%s\n' "${after:-none}"
}

# Step 7 (spec §13): no GPU node of this cluster may be left billing on Lambda.
# make gpu-down cannot help here: gpuctl down drains each node through the cluster before it terminates
# it, and k3s-cp, the base stack and the kubeconfig are gone.
check_lambda() {
  local clear=0 why="make gpu-down cannot do it now: it drains each node through the cluster first, and the cluster is destroyed"
  log "==> Lambda: is any GPU node of this cluster still live?"
  lambda_clear || clear=$?
  if (( clear == 2 )); then
    die "everything else is torn down, but Lambda could not be asked whether a GPU node of this cluster is still live (error above): run scripts/gpuctl list-nodes, and terminate any node it lists in the Lambda console ($why)"
  fi
  if (( clear == 1 )); then
    log "GPU nodes still billing on Lambda:"
    log "$(printf '%s\n' "$LIVE_NODES" | sed 's/^/  /')"
    die "GPU nodes still billing on Lambda: terminate them in the Lambda console ($why), then check that scripts/gpuctl list-nodes prints nothing. Everything else is torn down; the bootstrap stack is kept"
  fi
}

(( $# == 0 )) || usage
# Brief step 4: destroyed in this order. gpu-access and workers read the base stack's state.
DESTROY_STACKS="gpu-access workers base"
# Frozen before .env is sourced (load_env exports every line of it, F2-1): the caller's AUTO_APPROVE (set
# or unset) is the only one that skips the question, and .env cannot change the paths, names and stacks
# that decide what is destroyed, drained or removed. A line of .env that sets or unsets one of them
# fails, and env_refused stops the run; tf_overrides_refused then stops one that redirects terraform.
PINNED_NAMES="AUTO_APPROVE REPO_ROOT RUN_DIR KUBECONFIG_FILE CPU_NODE_RE CP_NAME KC_REQUEST_TIMEOUT DESTROY_STACKS"
# shellcheck disable=SC2086  # PINNED_NAMES is a list of variable names
readonly $PINNED_NAMES PINNED_NAMES
trap 'env_refused $?' EXIT
load_env
trap - EXIT
tf_overrides_refused
KC_KILL_S="${DOWN_KC_KILL_S:-180}"
[[ "$KC_KILL_S" =~ ^[1-9][0-9]*$ ]] \
  || die "DOWN_KC_KILL_S='$KC_KILL_S' is not a whole number of seconds (default 180): nothing was changed" 2
LAMBDA_WAIT_S="${DOWN_LAMBDA_WAIT_S:-60}"
[[ "$LAMBDA_WAIT_S" =~ ^(0|[1-9][0-9]{0,3})$ ]] \
  || die "DOWN_LAMBDA_WAIT_S='$LAMBDA_WAIT_S' is not a whole number of seconds from 0 to 9999 (default 60): nothing was changed" 2
# Only scripts/gpuctl needs the Lambda API key, and it loads .env itself: keep the key out of terraform,
# kubectl and the tunnel.
unset LAMBDA_API_KEY
tf_env
confirm_project
trap 'interrupted 130' INT
trap 'interrupted 143' TERM

gpu_down
drain_workers
for stack in $DESTROY_STACKS; do
  destroy_stack "$stack"
done
log "==> API tunnel, GPU host keys, kubeconfig"
"$REPO_ROOT/scripts/tunnel.sh" stop \
  || die "tunnel.sh stop failed; every stack is destroyed already: re-run make down to finish (it also removes .run/known_hosts_* and $KUBECONFIG_FILE)"
rm -f -- "$RUN_DIR"/known_hosts_*
rm -f -- "$KUBECONFIG_FILE"
check_lambda
log "make down: done. Lambda lists no GPU node of this cluster. Kept: the bootstrap stack (gs://$GCP_PROJECT-k3s-tfstate, the enabled APIs)."
