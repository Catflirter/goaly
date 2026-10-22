#!/usr/bin/env bash
# Bring the cluster up, one stage at a time; every stage is safe to re-run.
#
#   up.sh init       make init: the Terraform state bucket and the GCP APIs (infra/bootstrap, local state)
#   up.sh base       the base stack, k3s on k3s-cp, kubeconfig, API tunnel; waits for k3s-cp and CoreDNS
#   up.sh workers    the workers stack (CPU worker VMs and their firewall rules), sync-peers, then
#                    join-node.sh for every worker (a Ready worker is left alone)
#   up.sh system     metrics-server, the nvidia-gpu RuntimeClass, the NVIDIA device plugin (Helm)
#   up.sh workloads  k8s/nettest, k8s/hello and the gpu-test namespace; waits for their rollouts
#   up.sh all        make up: the stages in ALL_STAGES below, in order. init is not one of them: spec
#                    §11's make up starts at base, and spec §3.1 applies the bootstrap stack once, by hand
#
# Each terraform apply shows its plan and asks first (AUTO_APPROVE=1 skips the question). The base
# stage applies with --no-destroy, since spec §3.1 applies the base stack only after a manual review
# of its plan: under AUTO_APPROVE=1 a plan that would destroy or replace anything there (k3s-cp) is
# refused, so an unattended make up never recreates the control plane. A deliberate replacement of
# k3s-cp (a change Terraform can only make by recreating it, such as a new zone) runs `make up`
# interactively, where the plan's destroys are listed and confirmed. A plan without changes applies
# nothing. The workers stage applies with --no-destroy too: make up only ever adds workers, and a
# worker that left the inventory without `make cpu-remove` was never drained, so an unattended run
# refuses to destroy it (interactively the destroy is listed and asked). Scaling down is
# scripts/cpu-pool.sh's job: it drains and deletes the node before its apply.
set -euo pipefail

# shellcheck disable=SC1091  # common.sh is linted on its own
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

# Stages of `all`, in order (spec §11: base -> workers and their join -> system components -> workloads).
ALL_STAGES="base workers system workloads"
# The bound of every rollout wait in the system and workloads stages.
ROLLOUT_TIMEOUT=300s

usage() {
  log "usage: ${0##*/} init|base|workers|system|workloads|all"
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
  # A fresh private staging directory every run (NODE_STAGING_CMD in common.sh). node/'s entries go
  # into it, so the script stays at /tmp/k3s-node/server-bootstrap.sh. Exit 255 is ssh's own failure
  # (e.g. IAP dropped the connection), not the remote check: it gets its own message.
  cp_ssh "$NODE_STAGING_CMD" || {
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

# Task 14: the CPU workers (spec §3.1, §5.2, §5.3).
stage_workers() {
  log "==> workers: CPU worker VMs and their firewall rules (infra/workers), sync-peers, join each worker"
  tf_init workers
  tf_apply workers --no-destroy
  join_workers
  log "workers: every CPU worker is joined and Ready"
}

# Task 15: the system components (spec §4.4, §6.2): the three manifests, then the device plugin
# chart. Its DaemonSet runs on GPU nodes only; with none yet, --wait returns at once.
stage_system() {
  local sys="$REPO_ROOT/k8s/system"
  log "==> system: metrics-server, RuntimeClass nvidia-gpu, NVIDIA device plugin"
  kc apply -f "$sys/metrics-server.yaml" -f "$sys/nvidia-gpu-runtimeclass.yaml" \
    -f "$sys/nvidia-device-plugin-namespace.yaml"
  helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
  helm repo update nvdp
  helm --kubeconfig "$KUBECONFIG_FILE" upgrade -i nvdp nvdp/nvidia-device-plugin --namespace nvidia-device-plugin \
    --version 0.20.1 -f "$sys/nvdp-values.yaml" --wait --timeout 5m
  kc -n kube-system rollout status deployment/metrics-server --timeout="$ROLLOUT_TIMEOUT"
  log "system: metrics-server and the device plugin are rolled out"
}

# Task 16: the workloads (spec §7). limits-demo and the GPU test Pods belong to make verify.
stage_workloads() {
  local k8s="$REPO_ROOT/k8s"
  log "==> workloads: nettest, hello, the gpu-test namespace"
  kc apply -f "$k8s/nettest/" -f "$k8s/hello/" -f "$k8s/gpu-test/namespace.yaml"
  kc -n nettest rollout status daemonset/echo --timeout="$ROLLOUT_TIMEOUT"
  kc -n nettest rollout status daemonset/probe --timeout="$ROLLOUT_TIMEOUT"
  kc -n hello rollout status deployment/hello --timeout="$ROLLOUT_TIMEOUT"
  log "workloads: nettest and hello are rolled out"
}

[[ $# -eq 1 ]] || usage
case "$1" in
  init | base | workers | system | workloads | all) ;;
  *) usage ;;
esac
load_env
# No stage here needs the Lambda API key: keep it out of terraform, gcloud, kubectl, helm and the
# tunnel. scripts/gpuctl (sync-peers, in the workers stage) loads .env itself.
unset LAMBDA_API_KEY
case "$1" in
  init) stage_init ;;
  base) stage_base ;;
  workers) stage_workers ;;
  system) stage_system ;;
  workloads) stage_workloads ;;
  all)
    for stage in $ALL_STAGES; do
      "stage_$stage"
    done
    ;;
esac
