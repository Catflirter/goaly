#!/usr/bin/env bash
# The CPU worker pool (spec §5.3): the inventory infra/workers/workers.auto.tfvars.json, the workers
# stack, and joining or removing the workers.
#
#   cpu-pool.sh render N     print the inventory for N workers (reads the file, changes nothing)
#   cpu-pool.sh scale N      make cpu-scale N=: drain and delete the workers that go, write the
#                            inventory, apply the workers stack, sync-peers, join every new worker
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
# once more first, before anything is drained: a node is drained and deleted from the cluster
# BEFORE the apply destroys its VM. A scale that only adds workers applies with --no-destroy, so an
# unattended run never destroys anything. Every command is safe to re-run: a second `remove` of the
# same worker, or a `scale` to the current size, changes nothing.
set -euo pipefail

# shellcheck disable=SC1091  # common.sh is linted on its own
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

TFVARS="${WORKERS_TFVARS:-$REPO_ROOT/infra/workers/workers.auto.tfvars.json}"
# A typo guard (N=30 for N=3 costs real money), not a GCP limit.
MAX_WORKERS=100

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
  local new old_names new_names name
  local -a removed=()
  new="$(render_pool "$1")" || exit
  load_env
  unset LAMBDA_API_KEY
  old_names="$(pool_names)" || die "cannot read $TFVARS"
  new_names="$(jq -r '.cpu_workers | keys[]' <<< "$new")"
  # shellcheck disable=SC2086  # names are k3s-cpu-<n>: no spaces or globs
  while read -r name; do
    [[ -n "$name" ]] && removed+=("$name")
  done <<< "$(not_in $old_names -- $new_names)"

  if (( ${#removed[@]} )); then
    log "scaling down to $1: ${removed[*]} will be drained, deleted from the cluster and destroyed"
    confirm "Drain and delete ${removed[*]}, then apply the workers stack (destroys their VMs)?" \
      || die "not changed: the pool stays as it is" 2
    for name in "${removed[@]}"; do
      drain_and_delete_node "$name"
    done
  fi
  write_pool "$new"
  tf_init workers
  if (( ${#removed[@]} )); then
    tf_apply workers
  else
    # Only adding workers (or none): nothing may be destroyed, unattended or not without a review.
    tf_apply workers --no-destroy
  fi
  join_workers
  log "CPU pool: $(pool_names | wc -l | tr -d ' ') worker(s)"
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
    confirm "Drain $name, delete it from the cluster, then apply the workers stack (destroys its VM)?" \
      || die "not changed: $name stays" 2
  else
    # Gone from the inventory already (a second run, or an earlier run whose apply was declined):
    # finish whatever is left, which is nothing when that run completed.
    log "$name is not in $TFVARS: making sure it is gone from the cluster and the workers stack"
  fi
  drain_and_delete_node "$name"
  if [[ "$in_pool" == true ]]; then
    new="$(jq --arg n "$name" '.cpu_workers |= del(.[$n])' "$TFVARS")" || die "cannot update $TFVARS"
    write_pool "$new"
  fi
  tf_init workers
  tf_apply workers
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
