#!/usr/bin/env bash
# Bring the cluster up, one stage at a time; every stage is safe to re-run.
#
#   up.sh init   make init: the Terraform state bucket and the GCP APIs (infra/bootstrap, local state)
#   up.sh base   the base stack, k3s on k3s-cp, kubeconfig, API tunnel; waits for k3s-cp and CoreDNS
#   up.sh all    make up: every stage below in order, init excepted (it is one-time, spec §3.1)
#
# Each terraform apply shows its plan and asks first (AUTO_APPROVE=1 skips the question). The base
# stage applies with --no-destroy: under AUTO_APPROVE=1 a plan that would destroy or replace
# anything there (k3s-cp) is refused. A deliberate replacement of k3s-cp (a ForceNew change such as
# its image, machine type or startup metadata) runs `make up` interactively, where the plan's
# destroys are listed and confirmed. A plan without changes applies nothing.
set -euo pipefail

# shellcheck disable=SC1091  # common.sh is linted on its own
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

# Stages of `all`, in order. Later tasks append workers, system and workloads.
ALL_STAGES="base"

usage() {
  log "usage: ${0##*/} init|base|all"
  exit 2
}

stage_init() {
  local dir="$REPO_ROOT/infra/bootstrap"
  tf_env
  log "==> init: state bucket gs://$GCP_PROJECT-k3s-tfstate and the GCP APIs (infra/bootstrap, local state)"
  terraform -chdir="$dir" init -input=false -lockfile=readonly
  tf_apply bootstrap
}

stage_base() {
  local mtu="${UNDERLAY_MTU:-1460}" cp_ip cp_internal_ip ip rc
  # The MTU and both IPs end up in a command line run as root on k3s-cp: each is checked first.
  [[ "$mtu" =~ ^[1-9][0-9]{2,3}$ ]] || die "UNDERLAY_MTU in .env must be an MTU in bytes (100-9999): '$mtu'" 2
  log "==> base: VPCs, k3s-cp, k3s server, kubeconfig, API tunnel"
  tf_init base
  tf_apply base --no-destroy
  cp_ip="$(tf_out base cp_ip)"
  cp_internal_ip="$(tf_out base cp_internal_ip)"
  for ip in "$cp_ip" "$cp_internal_ip"; do
    [[ "$ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || die "base stack output is not an IPv4 address: '$ip'"
  done
  # Resolved here, not first inside wait_for: a bad cp_zone output dies at once with its message.
  cp_resolve

  wait_for 'cp_ssh true' 300 || die "IAP SSH to $CP_NAME not available after 300s"
  # A fresh staging directory every run, created by the operator's account with mode 0700. mkdir
  # (no -p) fails if anything recreated the path after rm, and the owner check catches a mkdir
  # that is not atomic, so nobody else on k3s-cp can place files there for root to run. node/'s
  # entries go into it, so the script stays at /tmp/k3s-node/server-bootstrap.sh. Exit 255 is ssh's
  # own failure (e.g. IAP dropped the connection), not the remote check: it gets its own message.
  cp_ssh 'rm -rf /tmp/k3s-node && mkdir -m 0700 /tmp/k3s-node && [ ! -L /tmp/k3s-node ] && [ -O /tmp/k3s-node ]' || {
    rc=$?
    [[ $rc == 255 ]] && die "SSH to $CP_NAME failed while preparing /tmp/k3s-node (exit 255); re-run make up"
    die "cannot create a private /tmp/k3s-node on $CP_NAME (it exists, is a symlink, or is owned by another user): remove it on the CP and re-run make up"
  }
  cp_scp "$REPO_ROOT"/node/* /tmp/k3s-node
  cp_ssh "sudo bash /tmp/k3s-node/server-bootstrap.sh --cp-ip $cp_ip --cp-internal-ip $cp_internal_ip --underlay-mtu $mtu"

  "$REPO_ROOT/scripts/kubeconfig.sh"
  "$REPO_ROOT/scripts/tunnel.sh" start
  wait_for 'kc wait --for=condition=Ready node/k3s-cp --timeout=10s' 300 \
    || die "node k3s-cp not Ready after 300s"
  wait_for 'kc -n kube-system wait --for=condition=Ready pod -l k8s-app=kube-dns --timeout=10s' 300 \
    || die "CoreDNS not Ready after 300s (spec §12 #6: the hairpin to the CP's public IP)"
  log "base: k3s-cp and CoreDNS Ready"
}

[[ $# -eq 1 ]] || usage
case "$1" in
  init | base | all) ;;
  *) usage ;;
esac
load_env
# No stage here needs the Lambda API key: keep it out of terraform, gcloud, kubectl and the tunnel
# (scripts/gpuctl, should a later stage run it, loads .env itself).
unset LAMBDA_API_KEY
case "$1" in
  init) stage_init ;;
  base) stage_base ;;
  all)
    for stage in $ALL_STAGES; do
      "stage_$stage"
    done
    ;;
esac
