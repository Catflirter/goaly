#!/usr/bin/env bash
# make verify: run the registered checks into verification/<YYYY-MM-DDTHHMM-0700>/ (spec section 9).
#
#   scripts/verify.sh [GROUP…]   GROUP is A, B, C, D or E (S with STRETCH=1); groups run in the order given
#
# Without arguments it runs A B C D, plus S when STRETCH=1 (the stretch checks: S0 the monitoring stack,
# spec §10.0), plus E when RESILIENCE=1 (E reboots nodes; slow and costly). Without STRETCH=1 group S
# does not exist: no S check is registered and the run is exactly the core one.
# Exit status: 0 when no check failed (SKIP is not a failure) and gitleaks found nothing in the
# run directory, 1 otherwise, 2 for an unknown group or when the run directory already holds files (a
# second run in the same minute: its name has minute resolution; wait and run again).
set -euo pipefail

# checks.sh is linted on its own; pre-commit runs shellcheck without -x, so don't follow it.
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/checks.sh"

VR_GROUPS=(A B C D E)
if [[ "${STRETCH:-0}" == 1 ]]; then
  VR_GROUPS+=(S)
fi
VR_REG_GROUP=()
VR_REG_ID=()
VR_REG_TITLE=()
VR_REG_FUNC=()

# register_check GROUP ID TITLE FUNC: add a check; within a group, checks run in registration order.
register_check() {
  local group="$1" id="$2" title="$3" func="$4"
  [[ " ${VR_GROUPS[*]} " == *" $group "* ]] || die "register_check $id: unknown group '$group'"
  declare -F "$func" >/dev/null || die "register_check $id: function '$func' is not defined"
  VR_REG_GROUP+=("$group")
  VR_REG_ID+=("$id")
  VR_REG_TITLE+=("$title")
  VR_REG_FUNC+=("$func")
}

# Tasks 18 (A–C), 21 (D) and 22 (E) register their checks below this line (the functions are in
# scripts/lib/checks.sh; the PASS criteria are the plan's, Task 18's table and spec §9).
# --- registered checks ---
register_check A A1 'Nodes Ready (at least 2)' check_a1_nodes
register_check A A2 'kubectl describe every node' check_a2_describe_nodes
register_check A A3 'metrics-server and CoreDNS Ready on k3s-cp' check_a3_system_pods
register_check A A4 'flannel annotations: public-ip is the ExternalIP' check_a4_flannel_annotations
register_check B B1 'N x N connectivity matrix (curl, MTU ping, Service DNS)' check_b1_matrix
register_check B B2 'On the wire at k3s-cp: encrypted, WireGuard, expected ports only' check_b2_wire
register_check B B3 'WireGuard peers and recent handshakes' check_b3_wg
register_check B B4 'iperf3 across nodes' check_b4_iperf
register_check B B5 'Negative tests: API ports, metadata server, DNS, guard counters' check_b5_negative
register_check B B6 'Firewall inventory: every source a /32 (or IAP)' check_b6_firewall_inventory
register_check C C1 'hello rollout' check_c1_rollout
register_check C C2 'hello world from another node' check_c2_cross_node_curl
register_check C C3 'hello world through kubectl port-forward' check_c3_port_forward
register_check C C4 'QoS class Guaranteed' check_c4_qos
register_check C C5 'cgroup memory.max and cpu.max' check_c5_cgroup
register_check C C6 'kubectl top pod' check_c6_top
register_check C C7 'limits-demo: OOMKilled and CPU throttling' check_c7_limits_demo
register_check C C8 'NetworkPolicy: allowed from nettest, denied elsewhere' check_c8_netpol
# D1-D6: placeholders (SKIP) until Task 21 replaces check_d_pending with check_d1_gpu_stack … check_d6_overflow.
register_check D D1 'GPU stack: audit, containerd, RuntimeClass' check_d_pending
register_check D D2 'GPU capacity and allocatable' check_d_pending
register_check D D3 'Device plugin registered' check_d_pending
register_check D D4 'gpu-smoke: nvidia-smi' check_d_pending
register_check D D5 'vectoradd: Test PASSED' check_d_pending
register_check D D6 'gpu-overflow stays Pending' check_d_pending
# Stretch checks (group S), registered only with STRETCH=1 (make verify STRETCH=1). S0: stretch S0b;
# S1 (check_s1_hpa) is added by stretch S1.
if [[ "${STRETCH:-0}" == 1 ]]; then
  register_check S S0 'Monitoring: scrape targets up, DCGM data on GPU nodes, Grafana healthy' check_s0_metrics
fi
# --- end of registered checks ---

main() {
  local groups=("$@") group i ran
  if (($# == 0)); then
    groups=(A B C D)
    if [[ "${STRETCH:-0}" == 1 ]]; then
      groups+=(S)
    fi
    if [[ "${RESILIENCE:-0}" == 1 ]]; then
      groups+=(E)
    fi
  fi
  for group in "${groups[@]}"; do
    if [[ "$group" == S && " ${VR_GROUPS[*]} " != *" S "* ]]; then
      die "check group S (the stretch checks) needs STRETCH=1: make verify STRETCH=1" 2
    fi
    [[ " ${VR_GROUPS[*]} " == *" $group "* ]] \
      || die "unknown check group '$group' (groups: ${VR_GROUPS[*]})" 2
  done

  vr_init "$REPO_ROOT/verification/$(TZ="$VR_TZ" date +%Y-%m-%dT%H%M%z)"
  for group in "${groups[@]}"; do
    ran=0
    for ((i = 0; i < ${#VR_REG_ID[@]}; i++)); do
      if [[ "${VR_REG_GROUP[i]}" == "$group" ]]; then
        run_check "${VR_REG_ID[i]}" "${VR_REG_TITLE[i]}" "${VR_REG_FUNC[i]}"
        ran=$((ran + 1))
      fi
    done
    if ((ran == 0)); then
      log "group $group: no checks registered"
    fi
  done
  vr_finish
}

main "$@"
