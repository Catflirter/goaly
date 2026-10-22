#!/usr/bin/env bash
# The CPU worker pool (spec §5.3): the inventory infra/workers/workers.auto.tfvars.json, the workers
# stack, and joining or removing the workers.
#
#   cpu-pool.sh render N     print the inventory for N workers (reads the file, changes nothing)
#   cpu-pool.sh scale N      make cpu-scale N=: drain and delete the workers that go, write the
#                            inventory, apply the workers stack, join every worker that has not
#                            joined yet (sync-peers before and after)
#   cpu-pool.sh remove NAME  make cpu-remove NAME=: drain and delete NAME, drop it from the
#                            inventory, apply, sync-peers (k3s-cpu-1 is never removed)
#   cpu-pool.sh stop|start   make cpu-stop / cpu-start: gcloud compute instances stop/start for
#                            every worker; a stopped worker is NotReady and keeps its static IP
#
# render: new workers are k3s-cpu-<n> with the lowest free numbers, in ${CPU_ZONE:-us-central1-a}
# with ${CPU_TYPE:-e2-small}; existing workers keep their settings; scaling down removes the highest
# numbers. Names sort numerically (k3s-cpu-9 before k3s-cpu-10). ${WORKERS_TFVARS} names another
# inventory file. N must be an integer from 1 to MAX_WORKERS (exit 2 otherwise).
#
# Every apply shows its plan and asks (AUTO_APPROVE=1 skips the question). Removing workers asks
# once more, after the plan and before anything is drained: a node is drained and deleted from the
# cluster BEFORE the apply destroys its VM. The workers that go are the ones the workers stack has
# deployed (worker_ips) or the inventory lists, minus the new inventory: so a re-run after a
# declined or failed apply, or after a hand edit of the file, drains them again. The apply may
# destroy or replace only those workers (tf_plan --destroy-only; --no-destroy when none go): under
# AUTO_APPROVE=1 a plan that touches any other worker, k3s-cpu-1 included, is refused (exit 2), and
# interactively it is listed and asked. That plan is made (with the new inventory as a -var-file)
# and checked BEFORE the drain: a refused plan has drained nothing and left the inventory as it was.
# The order is: plan and check, ask, drain and delete, write the inventory, ask, apply that plan.
# Every command is safe to re-run: a second `remove` of the same worker, or a `scale` to the current
# size, changes nothing.
#
# Scale step 4 joins every worker that has not joined: a new one always; one that was in the pool
# before when the cluster has no Node of that name, or when its Node is NotReady and its VM is
# RUNNING (a join that timed out: join-node.sh bootstraps it again). A NotReady worker whose VM is
# stopped (make cpu-stop: TERMINATED, STOPPED; or SUSPENDED, which needs gcloud compute instances
# resume, not cpu-start) is left alone; one whose VM is in any other state (STOPPING, PROVISIONING,
# ...) is left alone too, but the scale then exits 1 naming it.
set -euo pipefail

# shellcheck disable=SC1091  # common.sh is linted on its own
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

TFVARS="${WORKERS_TFVARS:-$REPO_ROOT/infra/workers/workers.auto.tfvars.json}"
# A typo guard (N=30 for N=3 costs real money), not a GCP limit.
MAX_WORKERS=100
# The new inventory a scale-down or remove plans with (tf_plan --var-file), and whether the run is
# between that plan and the drain; the EXIT trap cleans up and says what a failure there left behind.
NEXT_TFVARS=""
PLANNING=0

on_exit() {
  local rc=$?
  [[ -z "$NEXT_TFVARS" ]] || rm -f -- "$NEXT_TFVARS"
  [[ -z "${TF_PLAN:-}" ]] || rm -f -- "$TF_PLAN"
  if (( PLANNING && rc != 0 )); then
    log "nothing was drained or deleted from the cluster, and $TFVARS is unchanged"
  fi
}
trap on_exit EXIT

# plan_removal NAME... [-- JSON]: tf_plan the workers stack, which may destroy only the NAMEs, before
# anything is drained. With JSON (the new inventory) the plan uses it as a -var-file, so the
# inventory file itself is written only after the drain. A refused plan exits 2 here.
plan_removal() {
  local -a names=()
  while (( $# )) && [[ "$1" != -- ]]; do names+=("$1"); shift; done
  tf_init workers
  PLANNING=1
  if (( $# == 2 )); then
    run_dir || die "cannot create $RUN_DIR"
    NEXT_TFVARS="$RUN_DIR/workers.next.tfvars.json"
    printf '%s\n' "$2" > "$NEXT_TFVARS" || die "cannot write $NEXT_TFVARS"
    tf_plan workers --var-file "$NEXT_TFVARS" --destroy-only "${names[@]}"
  else
    tf_plan workers --destroy-only "${names[@]}"
  fi
  PLANNING=0
}

usage() {
  log "usage: ${0##*/} render N | scale N | remove NAME | stop | start"
  [[ $# -eq 0 ]] || log "error: $*"
  exit 2
}

# check_n N: an integer from 1 to MAX_WORKERS (no sign, no leading zero), else a usage error.
check_n() {
  if ! [[ "$1" =~ ^[1-9][0-9]{0,2}$ ]] || (( $1 > MAX_WORKERS )); then
    usage "N must be an integer from 1 to $MAX_WORKERS: '$1'"
  fi
}

# render_pool N: the inventory for N workers on stdout. Dies (1) for a missing or malformed file,
# including any name that is not k3s-cpu-<n>.
render_pool() {
  local n="$1" zone="${CPU_ZONE:-us-central1-a}" type="${CPU_TYPE:-e2-small}"
  check_n "$n"
  [[ "$zone" =~ ^[a-z]+-[a-z]+[0-9]+-[a-z]$ ]] || usage "CPU_ZONE is not a GCP zone: '$zone'"
  [[ "$type" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || usage "CPU_TYPE is not a GCE machine type: '$type'"
  [[ -f "$TFVARS" ]] || die "$TFVARS not found"
  jq --argjson n "$n" --arg zone "$zone" --arg type "$type" '
    def num: ltrimstr("k3s-cpu-") | tonumber;
    if (.cpu_workers | type) != "object" then error("cpu_workers is not an object") else . end
    | (.cpu_workers | keys | map(select(test("^k3s-cpu-[1-9][0-9]*$") | not))) as $bad
    | if ($bad | length) > 0 then error("cpu_workers names must be k3s-cpu-<n>: \($bad | join(", "))") else . end
    | (.cpu_workers | to_entries | sort_by(.key | num) | .[:$n]) as $kept
    | ($kept | map(.key | num)) as $used
    | ([range(1; $n + 1)] | map(select(. as $i | $used | index($i) | not))
       | .[:($n - ($kept | length))]
       | map({key: "k3s-cpu-\(.)", value: {zone: $zone, machine_type: $type}})) as $added
    | .cpu_workers = (($kept + $added) | sort_by(.key | num) | from_entries)
  ' "$TFVARS" || die "cannot render $TFVARS (it must hold cpu_workers: {\"k3s-cpu-<n>\": {zone, machine_type}})"
}

# pool_names [FILE]: the worker names in an inventory (default $TFVARS), one per line, in numeric order.
pool_names() {
  jq -r '.cpu_workers | keys[]' "${1:-$TFVARS}" | sort -t- -k3,3n
}

# write_pool JSON: replace $TFVARS with JSON (atomically: same directory, then rename), unless it
# already holds exactly that, so a no-op scale leaves the file and its mtime alone.
write_pool() {
  local tmp
  if [[ "$(cat -- "$TFVARS")" == "$1" ]]; then
    return 0
  fi
  tmp="$TFVARS.tmp.$$"
  if ! { printf '%s\n' "$1" > "$tmp" && mv -f -- "$tmp" "$TFVARS"; }; then
    rm -f -- "$tmp"
    die "cannot write $TFVARS"
  fi
  log "wrote $TFVARS"
}

# not_in WORDS... -- SET...: the WORDS that are not in SET, one per line.
not_in() {
  local word item found
  local -a words=()
  while [[ "$1" != -- ]]; do words+=("$1"); shift; done
  shift
  for word in ${words[@]+"${words[@]}"}; do
    found=0
    for item in "$@"; do [[ "$item" == "$word" ]] && found=1; done
    (( found )) || printf '%s\n' "$word"
  done
}

cmd_scale() {
  local new old_names new_names deployed current name state
  local -a removed=() join=() unsettled=()
  new="$(render_pool "$1")" || exit
  load_env
  unset LAMBDA_API_KEY
  old_names="$(pool_names)" || die "cannot read $TFVARS"
  new_names="$(jq -r '.cpu_workers | keys[]' <<< "$new")"
  # What the workers stack has deployed now, not only what the file says: a VM still running after
  # an earlier scale-down whose apply was declined or failed is drained (again) before it goes.
  deployed="$(tf_out_json workers worker_ips)" || die "cannot read worker_ips from the workers stack (run make up first)"
  deployed="$(jq -r 'if type == "object" then keys[] else error("not an object") end' <<< "$deployed")" \
    || die "unexpected worker_ips output from the workers stack"
  # shellcheck disable=SC2086  # names are k3s-cpu-<n>: no spaces or globs
  current="$(printf '%s\n' $old_names $deployed | sort -u -t- -k3,3n)"
  # shellcheck disable=SC2086
  while read -r name; do
    [[ -n "$name" ]] && removed+=("$name")
  done <<< "$(not_in $current -- $new_names)"

  if (( ${#removed[@]} )); then
    for name in "${removed[@]}"; do
      [[ "$name" =~ $CPU_NODE_RE ]] || die "unexpected worker name: '$name'"
      # Spec §10.2: k3s-cpu-1 stays (the workers stack refuses an inventory without it as well).
      [[ "$name" != k3s-cpu-1 ]] || die "k3s-cpu-1 is never removed (spec §10.2): $TFVARS must keep it" 2
    done
    log "scaling down to $1: ${removed[*]} will be drained, deleted from the cluster and destroyed"
    # Only the workers that go may be destroyed (never k3s-cpu-1 or another worker, unattended):
    # checked on the plan of the new inventory before anything is drained, and shown before the
    # question.
    plan_removal "${removed[@]}" -- "$new"
    confirm "Drain and delete ${removed[*]}, then apply the workers stack (destroys their VMs)?" \
      || die "not changed: nothing was drained and $TFVARS is unchanged" 2
    for name in "${removed[@]}"; do
      drain_and_delete_node "$name"
    done
    write_pool "$new"
    tf_apply_plan workers
  else
    # Only adding workers (or none): nothing may be destroyed, unattended or not without a review.
    write_pool "$new"
    tf_init workers
    tf_apply workers --no-destroy
  fi
  # Scale step 4: join each worker that has not joined yet. A new worker is always handed to
  # join-node.sh (which leaves it alone once Ready); one that was in the pool before when the cluster
  # has no Node of that name, or when its Node is NotReady while its VM runs (a join that timed out).
  # A NotReady worker whose VM is stopped (make cpu-stop) is not touched: SSH to a stopped VM would
  # fail the whole scale.
  # shellcheck disable=SC2086  # names are k3s-cpu-<n>: no spaces or globs
  while read -r name; do
    [[ -n "$name" ]] || continue
    if grep -qx -- "$name" <<< "$old_names"; then
      state="$(node_state "$name")" || exit
      case "$state" in
        Absent) join+=("$name") ;;
        NotReady)
          vm_status "$name"
          case "$VM_STATUS" in
            RUNNING)
              log "$name is registered but NotReady and its VM is RUNNING: joining it again"
              join+=("$name") ;;
            TERMINATED | STOPPED)
              log "$name is registered but NotReady and its VM is $VM_STATUS: not joined (make cpu-start starts it; a re-run of make cpu-scale joins it if it stays NotReady once RUNNING)" ;;
            SUSPENDED)
              # make cpu-start (instances start) does not wake a suspended VM.
              log "$name is registered but NotReady and its VM is SUSPENDED: not joined (gcloud compute instances resume $name --project $GCP_PROJECT --zone $_ZONE wakes it; a re-run of make cpu-scale joins it if it stays NotReady once RUNNING)" ;;
            *)
              log "$name is registered but NotReady and its VM is $VM_STATUS: not joined now"
              unsettled+=("$name ($VM_STATUS)") ;;
          esac ;;
      esac
    else
      join+=("$name")
    fi
  done <<< "$(printf '%s\n' $new_names | sort -t- -k3,3n)"
  join_workers ${join[@]+"${join[@]}"}
  log "CPU pool: $(pool_names | wc -l | tr -d ' ') worker(s)"
  if (( ${#unsettled[@]} )); then
    die "NotReady, with a VM neither running nor stopped: ${unsettled[*]}; once it settles, a re-run of make cpu-scale joins it if it is RUNNING and still NotReady"
  fi
}

cmd_remove() {
  local name="$1" in_pool new
  [[ "$name" =~ $CPU_NODE_RE ]] || usage "NAME must be a CPU worker, k3s-cpu-<n>: '$name'"
  # Spec §10.2: k3s-cpu-1 stays (the workers stack refuses an inventory without it as well).
  [[ "$name" != k3s-cpu-1 ]] || die "k3s-cpu-1 is never removed (spec §10.2)" 2
  [[ -f "$TFVARS" ]] || die "$TFVARS not found"
  in_pool="$(jq -r --arg n "$name" '.cpu_workers | has($n)' "$TFVARS")" || die "cannot read $TFVARS"
  load_env
  unset LAMBDA_API_KEY
  if [[ "$in_pool" == true ]]; then
    new="$(jq --arg n "$name" '.cpu_workers |= del(.[$n])' "$TFVARS")" || die "cannot update $TFVARS"
    # Only NAME may be destroyed: never k3s-cpu-1 or another worker, unattended. Checked on the plan
    # before the drain, and shown before the question.
    plan_removal "$name" -- "$new"
    confirm "Drain $name, delete it from the cluster, then apply the workers stack (destroys its VM)?" \
      || die "not changed: $name stays" 2
  else
    # Gone from the inventory already (a second run, or an earlier run whose apply was declined):
    # finish whatever is left, which is nothing when that run completed.
    log "$name is not in $TFVARS: making sure it is gone from the cluster and the workers stack"
    plan_removal "$name"
  fi
  drain_and_delete_node "$name"
  if [[ "$in_pool" == true ]]; then
    write_pool "$new"
  fi
  tf_apply_plan workers
  "$REPO_ROOT/scripts/gpuctl" sync-peers || die "sync-peers failed"
  log "$name removed"
}

cmd_power() {
  local action="$1" json zones zone names
  load_env
  unset LAMBDA_API_KEY
  json="$(tf_out_json workers worker_zones)" || die "cannot read worker_zones from the workers stack"
  zones="$(jq -r 'if type == "object" then [.[]] | unique | .[] else error("not an object") end' <<< "$json")" \
    || die "unexpected worker_zones output from the workers stack"
  [[ -n "$zones" ]] || { log "no CPU workers in the workers stack: nothing to $action"; return 0; }
  # Every zone and name is checked before the first gcloud call.
  while read -r zone; do
    [[ "$zone" =~ ^[a-z]+-[a-z]+[0-9]+-[a-z]$ ]] || die "unexpected zone in worker_zones: '$zone'"
  done <<< "$zones"
  while read -r name; do
    [[ "$name" =~ $CPU_NODE_RE ]] || die "unexpected worker name in worker_zones: '$name'"
  done <<< "$(jq -r 'keys[]' <<< "$json")"
  while read -r zone; do
    names="$(jq -r --arg z "$zone" 'to_entries[] | select(.value == $z) | .key' <<< "$json" | sort -t- -k3,3n)"
    log "$action: ${names//$'\n'/ } ($zone)"
    # shellcheck disable=SC2086  # names are checked k3s-cpu-<n>: no spaces or globs
    gcloud compute instances "$action" $names --project "$GCP_PROJECT" --zone "$zone" --quiet
  done <<< "$zones"
}

(( $# >= 1 )) || usage
case "$1" in
  render)
    (( $# == 2 )) || usage
    render_pool "$2" ;;
  scale)
    (( $# == 2 )) || usage
    cmd_scale "$2" ;;
  remove)
    (( $# == 2 )) || usage
    cmd_remove "$2" ;;
  stop | start)
    (( $# == 1 )) || usage
    cmd_power "$1" ;;
  *) usage ;;
esac
