#!/usr/bin/env bash
# make down (spec §11): tear the whole environment down, in order. Takes no arguments.
#
#   1. ask once: type the GCP project id (GCP_PROJECT in .env) to go on; anything else, or end of input,
#      exits 2 with nothing changed. AUTO_APPROVE=1 skips the question. Nothing after it asks again.
#   2. scripts/gpuctl down --auto-approve: every Lambda GPU node of this cluster (drained, deleted from
#      the cluster, terminated) and the autoscaler overrides.
#   3. drain every CPU worker (k3s-cpu-<n>) the cluster lists and delete its Node. An unreachable
#      cluster (tunnel down, k3s-cp gone, no kubeconfig) skips this step with a warning, and a failed
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
#
# Failures and re-runs. Every step is safe to repeat, so a teardown that stopped part-way is finished
# by running make down again. The first failing step stops the run (exit 1) and says what is left:
#   - gpuctl down failed: the run asks Lambda (scripts/gpuctl list-nodes) whether any GPU node of this
#     cluster is still live. If one is, or Lambda cannot be asked, it stops before any destroy: a GPU
#     node left running after gpu-access and base are gone keeps billing, outside the firewall model,
#     with no cluster to join. If none is, the failure was in gpuctl down's firewall and peer step
#     (on a re-run after the workers or base stack is destroyed, sync-peers cannot read their outputs,
#     Ruling R29), which the destroys below make moot: the run goes on.
#   - a terraform init or destroy failed: the stacks after it are not touched (each one reads the base
#     stack's state, so base goes last).
#
# The Lambda API key reaches scripts/gpuctl only (it loads .env itself): it is unset here, as in up.sh.
set -euo pipefail

# shellcheck disable=SC1091  # common.sh is linted on its own
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

# Brief step 4: destroyed in this order. gpu-access and workers read the base stack's state.
DESTROY_STACKS="gpu-access workers base"
KC_KILL_S="${DOWN_KC_KILL_S:-180}"

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
  if [[ "${AUTO_APPROVE:-}" == 1 ]]; then
    log "[AUTO_APPROVE=1: going on without asking]"
    return 0
  fi
  printf 'Type the GCP project id (%s) to destroy all of this: ' "$GCP_PROJECT" >&2
  read -r answer || true
  [[ "$answer" == "$GCP_PROJECT" ]] || die "not confirmed (the answer was not the project id): nothing was changed" 2
}

# Step 2. On failure, Lambda decides whether the teardown may go on (see the header).
gpu_down() {
  local rc=0 nodes
  log "==> GPU nodes: scripts/gpuctl down"
  "$REPO_ROOT/scripts/gpuctl" down --auto-approve < /dev/null || rc=$?
  (( rc == 0 )) && return 0
  log "scripts/gpuctl down failed (exit $rc); asking Lambda which GPU nodes of this cluster are still live"
  if ! nodes="$("$REPO_ROOT/scripts/gpuctl" list-nodes < /dev/null)"; then
    die "cannot list the GPU nodes on Lambda either: nothing was destroyed. Fix the error above (scripts/gpuctl status shows the GPU nodes) and re-run make down"
  fi
  if [[ -n "$nodes" ]]; then
    log "GPU nodes still live on Lambda (they keep billing):"
    log "$(printf '%s\n' "$nodes" | sed 's/^/  /')"
    die "scripts/gpuctl down did not remove them, so nothing was destroyed: the gpu-access and base stacks stay until they are gone. If the cluster is unreachable, bring the API tunnel up (make tunnel) and re-run make down; if it cannot be reached at all, terminate them in the Lambda console and re-run make down"
  fi
  log "warning: Lambda lists no GPU node of this cluster, so no GPU node is left outside the firewall model; going on (the destroys below remove the gpu-access stack and k3s-cp)"
}

# kc_bounded ARGS...: kc, killed after KC_KILL_S seconds.
kc_bounded() {
  timeout "$KC_KILL_S" kubectl --kubeconfig "$KUBECONFIG_FILE" "$@"
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
    || die "terraform destroy failed for $stack (exit $rc): the stacks after it ($(stacks_after "$stack")) were not touched; fix the cause and re-run make down (it repeats every step; what is destroyed stays destroyed)"
  log "$stack: destroyed"
}

# stacks_after STACK: the stacks DESTROY_STACKS lists after STACK, or "none".
stacks_after() {
  local after="${DESTROY_STACKS#*"$1"}"
  after="${after# }"
  printf '%s\n' "${after:-none}"
}

(( $# == 0 )) || usage
load_env
# Only scripts/gpuctl needs the Lambda API key, and it loads .env itself: keep the key out of terraform,
# kubectl and the tunnel.
unset LAMBDA_API_KEY
tf_env
confirm_project

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
log "make down: done. Kept: the bootstrap stack (gs://$GCP_PROJECT-k3s-tfstate, the enabled APIs). scripts/gpuctl list-nodes (Lambda) should print nothing."
