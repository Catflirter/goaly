#!/usr/bin/env bash
# Join one node to the cluster (spec §5.2): a GCP CPU worker (k3s-cpu-<n>, over IAP SSH) or a Lambda
# GPU node (k3s-gpu-<pool>-<n>, over plain SSH as ubuntu). Safe to re-run: a Ready node is left alone,
# and a node whose earlier join failed is bootstrapped and joined again (step 4).
#
#   join-node.sh --node NAME --external-ip IP [--gpu-type T] [--skip-sync] [--force]
#
#   1. IP must already be in the cloud firewall: the workers stack's worker_ips[NAME] for a CPU
#      worker, the gpu-access stack's gpu_ips for a GPU node. Then scripts/gpuctl sync-peers puts it
#      into k3s-cp's k3s_peers set (and the Lambda rules), unless --skip-sync: the caller already
#      did (scripts/gpuctl apply, or `make up`/cpu-pool.sh, which sync once for all their nodes).
#   2. NAME already Ready: done (exit 0), unless --force. NAME registered but NotReady: that Node is
#      deleted first (k3s then drops its node-password secret, which would reject a new machine of
#      the same name); never a Ready node or another name.
#   3. Wait (at most 300s) for SSH to the node: a VM or Lambda instance made seconds earlier has no
#      sshd yet. Then node/ into a fresh private /tmp/k3s-node on it, and node-bootstrap.sh as root.
#   4. k3s-agent is stopped (a no-op on a first join), so that it starts from the new token: an agent
#      left running by an earlier join holds that join's token, which is deleted. A join token (TTL
#      15m) is created on k3s-cp and piped straight into the node's token-file; then k3s-agent
#      starts. The token is never in a variable, an argument, a log or a local file.
#   5. Wait (at most 300s) for the node to be Ready.
#   6. Delete the join token on k3s-cp (by its DESCRIPTION, NAME), also when a later step fails.
#   7. GPU node only: the admin-only label node-restriction.kubernetes.io/accelerator=nvidia, then
#      containerd must have picked up the nvidia runtime (spec §6.2 step 2); if not, restart
#      k3s-agent once and check again.
#
# --gpu-type is required for a GPU node and refused for a CPU worker. Exit codes: 2 usage (nothing
# touched), 1 failed.
set -euo pipefail

# shellcheck disable=SC1091  # common.sh is linted on its own
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

STAGING=/tmp/k3s-node
CONTAINERD_GREP='sudo grep -A3 nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml'

usage() {
  log "usage: ${0##*/} --node NAME --external-ip IP [--gpu-type T] [--skip-sync] [--force]"
  [[ $# -eq 0 ]] || log "error: $*"
  exit 2
}

NODE="" IP="" GPU_TYPE="" SKIP_SYNC=0 FORCE=0
while (( $# )); do
  case "$1" in
    --node | --external-ip | --gpu-type)
      (( $# >= 2 )) || usage "$1 needs a value"
      case "$1" in
        --node) NODE="$2" ;;
        --external-ip) IP="$2" ;;
        --gpu-type) GPU_TYPE="$2" ;;
      esac
      shift 2 ;;
    --skip-sync) SKIP_SYNC=1; shift ;;
    --force) FORCE=1; shift ;;
    -h | --help) usage ;;
    *) usage "unknown argument: $1" ;;
  esac
done

# Every value below ends up in a command line run as root on the node: each is checked first.
[[ -n "$NODE" ]] || usage "--node is required"
[[ -n "$IP" ]] || usage "--external-ip is required"
KIND="$(node_kind "$NODE")" || usage "--node must be a CPU worker (k3s-cpu-<n>) or a GPU node (k3s-gpu-<pool>-<n>): '$NODE'"
is_ipv4 "$IP" || usage "--external-ip is not an IPv4 address: '$IP'"
if [[ "$KIND" == gpu ]]; then
  [[ -n "$GPU_TYPE" ]] || usage "a GPU node needs --gpu-type"
  # A Kubernetes label value, as gpuctl's gpu_type_label() makes it (a6000, a100_80gb_sxm4).
  [[ ${#GPU_TYPE} -le 63 && "$GPU_TYPE" =~ ^[a-z0-9]+([_-][a-z0-9]+)*$ ]] || usage "bad --gpu-type: '$GPU_TYPE'"
  GPU_FLAGS="--gpu --gpu-type $GPU_TYPE"
else
  [[ -z "$GPU_TYPE" ]] || usage "--gpu-type is for GPU nodes; $NODE is a CPU worker"
  GPU_FLAGS="--no-gpu"
fi

load_env
# Nothing here needs the Lambda API key (scripts/gpuctl loads .env itself).
unset LAMBDA_API_KEY
MTU="${UNDERLAY_MTU:-1460}"
[[ "$MTU" =~ ^[1-9][0-9]{2,3}$ ]] || die "UNDERLAY_MTU in .env must be an MTU in bytes (100-9999): '$MTU'" 2
READY_TIMEOUT="${JOIN_READY_TIMEOUT_S:-300}"
SSH_TIMEOUT="${JOIN_SSH_TIMEOUT_S:-300}"

# ---- 1. the IP is in the firewall; sync-peers ----------------------------------------------------
if [[ "$KIND" == cpu ]]; then
  json="$(tf_out_json workers worker_ips)" || die "cannot read worker_ips from the workers stack"
  known="$(jq -r --arg n "$NODE" 'if type == "object" then .[$n] // "" else error("not an object") end' <<< "$json")" \
    || die "unexpected worker_ips output from the workers stack"
  [[ "$known" == "$IP" ]] \
    || die "$NODE has no firewall rules for $IP: the workers stack says '${known:-nothing}' (run make up or make cpu-scale)"
else
  json="$(tf_out_json gpu-access gpu_ips)" || die "cannot read gpu_ips from the gpu-access stack"
  jq -e --arg ip "$IP" 'type == "array" and any(.[]; . == $ip)' <<< "$json" > /dev/null \
    || die "$IP is not in the gpu-access stack's gpu_ips: no firewall rules admit $NODE yet (scripts/gpuctl apply adds them)"
  # gpu_ssh's target: the IP just checked, not a Lambda API lookup (scripts/gpuctl ip).
  export GPU_SSH_IP="$IP"
fi
if (( ! SKIP_SYNC )); then
  "$REPO_ROOT/scripts/gpuctl" sync-peers || die "sync-peers failed: $NODE was not joined"
fi

# ---- 2. already Ready? A stale NotReady node of this name? -------------------------------------------
state="$(node_state "$NODE")" || exit
if [[ "$state" == Ready ]] && (( ! FORCE )); then
  log "$NODE is already Ready: nothing to do (--force bootstraps and joins it again)"
  exit 0
fi
# A NotReady Node of exactly this name (left by an instance terminated without a drain, or lost)
# holds k3s's $NODE.node-password.k3s secret: a new machine under the name would be rejected
# ("Node password rejected") after billed minutes. Deleting the Node makes k3s remove that secret
# (review F3-2, Ruling R54). Never a Ready node, never another name. A rejoin of the same machine
# (a Ready timeout re-run) just registers again with the password it already has.
if [[ "$state" == NotReady ]]; then
  log "WARNING: node $NODE is registered but NotReady: deleting it (and so k3s's $NODE.node-password.k3s secret) before the join"
  _kc_node_gone_ok "$NODE" delete node "$NODE" --request-timeout="$KC_REQUEST_TIMEOUT" \
    || die "cannot delete the stale NotReady node $NODE: $NODE was not joined (delete it with kubectl delete node $NODE and re-run)"
fi

# ---- 3. node-bootstrap.sh --------------------------------------------------------------------------
cp_ip="$(tf_out base cp_ip)" || die "cannot read cp_ip from the base stack"
is_ipv4 "$cp_ip" || die "base stack output cp_ip is not an IPv4 address: '$cp_ip'"
# k3s-cp's zone, and a CPU worker's, are looked up here, so the token pipeline below (whose parts
# run in subshells) does not read Terraform outputs.
cp_resolve
[[ "$KIND" == gpu ]] || _worker_zone "$NODE"

log "==> joining $NODE ($IP) to the cluster at $cp_ip"
if [[ "$KIND" == gpu ]]; then
  SSH_HINT="if this is a new instance under an old name, check its host key and remove $RUN_DIR/known_hosts_$NODE"
  # A missing SSH key is a configuration error (exit 2), not something the wait below should retry.
  _gpu_ssh_opts "$NODE"
else
  SSH_HINT="a worker stopped with make cpu-stop needs make cpu-start first"
fi
# A VM the workers apply made, or a Lambda instance that just turned active, can take a minute to
# accept SSH (exit 255 until then): wait for it, as up.sh base waits for k3s-cp (L3-1).
wait_for "node_ssh $NODE true < /dev/null" "$SSH_TIMEOUT" \
  || die "SSH to $NODE not available after ${SSH_TIMEOUT}s ($SSH_HINT)"
# A fresh private staging directory (NODE_STAGING_CMD, as up.sh base does on k3s-cp). Exit 255 is
# ssh's own failure, not the remote check: it gets its own message.
node_ssh "$NODE" "$NODE_STAGING_CMD" < /dev/null || {
  rc=$?
  [[ $rc == 255 ]] && die "SSH to $NODE failed while preparing $STAGING (exit 255); re-run the join ($SSH_HINT)"
  die "cannot create a private $STAGING on $NODE (it exists, is a symlink, or is owned by another user): remove it on the node and re-run the join"
}
node_scp "$NODE" "$REPO_ROOT"/node/* "$STAGING" < /dev/null
node_ssh "$NODE" "sudo bash $STAGING/node-bootstrap.sh --server-ip $cp_ip --node-name $NODE --external-ip $IP --underlay-mtu $MTU $GPU_FLAGS" < /dev/null

# ---- 4.-6. the join token: created, piped, deleted (also on failure) -------------------------------
# On k3s-cp: the id of every token whose DESCRIPTION column is exactly NODE, deleted there, so the
# token list never leaves k3s-cp. Only the id (the part of the TOKEN column before any "."): a full
# id.secret in `sudo k3s token delete`'s argv would land in k3s-cp's sudo log. A list without a
# DESCRIPTION header fails rather than deleting nothing quietly. NODE is a checked node name, so it
# is safe in the command line.
TOKEN_DELETE="list=\$(sudo k3s token list) && ids=\$(printf '%s\\n' \"\$list\" | awk -v d=$NODE"
TOKEN_DELETE+=" '!h { if (!NF) next; h = 1; for (i = 1; i <= NF; i++) if (\$i == \"DESCRIPTION\") c = i; next }"
TOKEN_DELETE+=" c && \$c == d { split(\$1, t, \".\"); print t[1] } END { if (h && !c) exit 1 }') && for id in \$ids;"
TOKEN_DELETE+=" do sudo k3s token delete \"\$id\" || exit 1; done"

TOKEN_PENDING=0
delete_join_token() {
  cp_ssh "$TOKEN_DELETE" < /dev/null
}
cleanup() {
  local rc=$?
  if (( TOKEN_PENDING )); then
    TOKEN_PENDING=0
    delete_join_token \
      || log "WARNING: could not delete the join token of $NODE on $CP_NAME; it expires 15 minutes after it was made"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# k3s-agent reads its token-file only when it starts. An agent still running from an earlier join
# (one that timed out, or a --force re-join) keeps retrying with the token that join's cleanup has
# already deleted, and `systemctl enable --now` below does not restart an active unit. So it is
# stopped first, and starts from the new token (a no-op on a first join: the unit is inactive).
node_ssh "$NODE" 'sudo systemctl stop k3s-agent' < /dev/null \
  || die "cannot stop k3s-agent on $NODE before handing it a new join token"

# Set before the pipeline: k3s-cp may have made a token even if the node never received it.
TOKEN_PENDING=1
cp_ssh "sudo k3s token create --ttl 15m --description $NODE" < /dev/null \
  | node_ssh "$NODE" 'sudo install -m 600 /dev/stdin /etc/rancher/k3s/token && sudo systemctl enable --now k3s-agent' \
  || die "could not install the join token on $NODE and start k3s-agent"

wait_for "kc wait --for=condition=Ready node/$NODE --timeout=10s" "$READY_TIMEOUT" \
  || die "node $NODE not Ready after ${READY_TIMEOUT}s (journalctl -u k3s-agent on the node says why)"

delete_join_token \
  || die "$NODE joined, but its join token could not be deleted on $CP_NAME; it expires 15 minutes after it was made"
TOKEN_PENDING=0

# ---- 7. GPU node: admin label, nvidia runtime in containerd ----------------------------------------
# containerd_has_nvidia: 0 when k3s's containerd config has the nvidia runtime, 1 when not; dies when
# the config cannot be read (ssh failure, a missing file).
containerd_has_nvidia() {
  local out rc=0
  out="$(node_ssh "$NODE" "$CONTAINERD_GREP" < /dev/null)" || rc=$?
  case "$rc" in
    0) [[ -n "$out" ]] ;;
    1) return 1 ;;
    *) die "cannot read the containerd config on $NODE (exit $rc)" ;;
  esac
}

if [[ "$KIND" == gpu ]]; then
  kc label node "$NODE" node-restriction.kubernetes.io/accelerator=nvidia --overwrite \
    --request-timeout="$KC_REQUEST_TIMEOUT"
  if ! containerd_has_nvidia; then
    # k3s detects the nvidia runtime only when it starts (spec §6.2 step 2).
    log "containerd on $NODE has no nvidia runtime: restarting k3s-agent once"
    node_ssh "$NODE" 'sudo systemctl restart k3s-agent' < /dev/null
    wait_for "kc wait --for=condition=Ready node/$NODE --timeout=10s" "$READY_TIMEOUT" \
      || die "node $NODE not Ready after ${READY_TIMEOUT}s after restarting k3s-agent"
    containerd_has_nvidia \
      || die "containerd on $NODE still has no nvidia runtime after a k3s-agent restart (is nvidia-container-runtime on k3s's PATH?)"
  fi
fi

log "$NODE joined and Ready"
