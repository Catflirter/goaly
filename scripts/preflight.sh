#!/usr/bin/env bash
# make preflight (spec section 9.1: T-60 and T-5 minutes before the demo). One line per item, OK or FAIL; every
# item runs even after a failure, and the exit status is 1 when any item failed, 0 otherwise.
#
#   1. tools: terraform (>= 1.9.0), gcloud, kubectl, jq, gitleaks, curl, the repo's .venv python (>= 3.11),
#      flock (chapter 6's guarded cpu-pool.sh runs need it); each with its version
#   2. gcloud auth list --filter=status:ACTIVE lists an account
#   3. gcloud auth application-default print-access-token succeeds (its token is never printed)
#   4. scripts/gpuctl status exits 0: the Lambda API key works, today's capacity is listed and nothing drifts
#      (a pool below its count, as before make gpu-apply, is only a note, exit 0). Its exit 3 is real drift
#      (a firewall rule with no live instance, a GPU node not in the cluster, a stale node or peer, a pool
#      over its count): a FAIL, which make gpu-apply or make gpu-down reconciles
#   5. scripts/gpuctl sync-peers: the operator's current public IP (the demo's network may not be the usual
#      one) goes into the Lambda firewall and k3s-cp's peer set
#   6. scripts/tunnel.sh status; when the tunnel is not up, scripts/tunnel.sh start
#   7. every node is Ready (and there is at least one)
#   8. screen safety: the operator's public IP (OPERATOR_IP in .env, else checkip.amazonaws.com) and the GCP
#      project id (GCP_PROJECT in .env) are known, so make demo can mask them (it refuses to start without
#      the IP)
#
# Everything it prints goes through redact (scripts/lib/checks.sh), with the operator's public IP masked as
# <operator_ip> and the GCP project id as <GCP_PROJECT> (scripts/lib/screen.sh; spec section 9.1, screen
# safety). The filter ignores SIGINT, so a Ctrl-C lets the running command (sync-peers, tunnel.sh start)
# stop cleanly instead of dying by SIGPIPE; the item it interrupted is the last one (exit 130).
set -euo pipefail

# checks.sh (and common.sh) are linted on their own; pre-commit runs shellcheck without -x.
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/checks.sh"
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/screen.sh"

FAILS=0
INTERRUPTED=0

ok() {
  printf 'OK   %s\n' "$*"
}

fail() {
  printf 'FAIL %s\n' "$*"
  FAILS=$((FAILS + 1))
}

# version_at_least HAVE WANT: dotted version HAVE >= WANT (numeric parts; anything after them is ignored).
version_at_least() {
  local have="$1" want="$2" i h w
  local -a hv wv
  IFS=. read -r -a hv <<< "${have%%[!0-9.]*}"
  IFS=. read -r -a wv <<< "$want"
  for ((i = 0; i < ${#wv[@]}; i++)); do
    h="${hv[i]:-0}"
    w="${wv[i]}"
    [[ "$h" =~ ^[0-9]+$ ]] || return 1
    ((10#$h > 10#$w)) && return 0
    ((10#$h < 10#$w)) && return 1
  done
  return 0
}

# first_line CMD...: the first line of CMD's output (stdout and stderr), or nothing when CMD fails.
first_line() {
  local out
  out="$("$@" 2>&1)" || return 1
  printf '%s\n' "${out%%$'\n'*}"
}

check_tools() {
  local tool v py="$REPO_ROOT/.venv/bin/python"
  if ! command -v terraform >/dev/null 2>&1; then
    fail "terraform: not found (Terraform >= 1.9.0)"
  elif ! v="$(terraform version -json 2>/dev/null | jq -r '.terraform_version // empty')" || [[ -z "$v" ]]; then
    fail "terraform: cannot read its version"
  elif version_at_least "$v" 1.9.0; then
    ok "terraform $v"
  else
    fail "terraform $v: older than 1.9.0"
  fi
  for tool in gcloud kubectl jq gitleaks curl; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      fail "$tool: not found"
      continue
    fi
    case "$tool" in
      gcloud) v="$(first_line gcloud version)" ;;
      kubectl) v="$(first_line kubectl version --client)" ;;
      jq) v="$(first_line jq --version)" ;;
      gitleaks) v="$(first_line gitleaks version)" ;;
      curl) v="$(first_line curl --version)" ;;
    esac
    if [[ -n "${v:-}" ]]; then ok "$tool: $v"; else fail "$tool: cannot read its version"; fi
  done
  if [[ ! -x "$py" ]]; then
    fail "python: $py not found (make venv)"
  elif v="$("$py" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)" \
    && version_at_least "$v" 3.11; then
    ok "python $v (.venv)"
  else
    fail "python ${v:-?} (.venv): Python >= 3.11 is needed"
  fi
  if command -v flock >/dev/null 2>&1; then
    ok "flock (chapter 6's guarded CPU scaling)"
  else
    fail "flock: not found (util-linux); chapter 6's cpu-pool.sh --guarded runs need it"
  fi
}

check_gcloud() {
  local accounts n
  if accounts="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' </dev/null 2>&1)" \
    && [[ -n "${accounts//[[:space:]]/}" ]]; then
    n="$(grep -c . <<< "$accounts" || true)"
    ok "gcloud auth: $n active account(s)"
  else
    fail "gcloud auth: no active account (gcloud auth login)"
  fi
  # The token goes to /dev/null: it is a credential (screen safety). Only gcloud's errors are shown.
  if gcloud auth application-default print-access-token </dev/null >/dev/null; then
    ok "gcloud auth application-default: credentials work (Terraform's)"
  else
    fail "gcloud auth application-default: no usable credentials (gcloud auth application-default login)"
  fi
}

check_gpuctl() {
  local rc=0
  scripts/gpuctl status </dev/null || rc=$?
  case "$rc" in
    0) ok "gpuctl status: the Lambda API answers; capacity above" ;;
    3) fail "gpuctl status (exit 3): drift, see the lines above; make gpu-apply or make gpu-down reconciles it" ;;
    *) fail "gpuctl status (exit $rc): the Lambda API key or the inventory check failed" ;;
  esac
  rc=0
  scripts/gpuctl sync-peers </dev/null || rc=$?
  if ((rc == 0)); then
    ok "sync-peers: the operator's current IP is in the peer allow-lists"
  else
    fail "sync-peers (exit $rc): the allow-lists may not hold the operator's current IP"
  fi
}

check_tunnel() {
  if scripts/tunnel.sh status </dev/null >/dev/null 2>&1; then
    ok "tunnel: up, the API server answers"
  elif scripts/tunnel.sh start </dev/null; then
    ok "tunnel: was down, started"
  else
    fail "tunnel: down and could not be started (make tunnel)"
  fi
}

check_nodes() {
  local out name status total=0 ready=0 not_ready=""
  if ! out="$(kc get nodes "$_VR_RT" </dev/null \
    -o 'jsonpath={range .items[*]}{.metadata.name}{" "}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}')"; then
    fail "nodes: kubectl get nodes failed"
    return 0
  fi
  while read -r name status; do
    [[ -n "$name" ]] || continue
    total=$((total + 1))
    if [[ "$status" == True ]]; then
      ready=$((ready + 1))
    else
      not_ready+=" $name"
    fi
  done <<< "$out"
  if ((total > 0 && ready == total)); then
    ok "nodes Ready: $ready of $total"
  elif ((total == 0)); then
    fail "nodes: the cluster lists none"
  else
    fail "nodes: $ready of $total Ready; not Ready:$not_ready"
  fi
}

check_screen() {
  if ((SCREEN_IP_MASKED)); then
    ok "screen safety: the operator IP is masked as <operator_ip> ($VR_OPERATOR_IP_FROM)"
  else
    fail "screen safety: the operator IP is unknown (set OPERATOR_IP in .env; checkip.amazonaws.com did not answer), so make demo cannot mask it"
  fi
  if ((SCREEN_PROJECT_MASKED)); then
    ok "screen safety: the GCP project id is masked as <GCP_PROJECT>"
  else
    fail "screen safety: GCP_PROJECT in .env is missing or not a project id, so make demo cannot mask it"
  fi
}

main() {
  # Ctrl-C: the running item's command stops (the filter outlives it), then no later item runs.
  trap 'INTERRUPTED=1' INT
  echo "preflight (spec section 9.1)"
  check_tools
  if ((!INTERRUPTED)); then check_gcloud; fi
  if ((!INTERRUPTED)); then check_gpuctl; fi
  if ((!INTERRUPTED)); then check_tunnel; fi
  if ((!INTERRUPTED)); then check_nodes; fi
  if ((!INTERRUPTED)); then check_screen; fi
  if ((INTERRUPTED)); then
    echo "preflight interrupted"
    return 130
  fi
  if ((FAILS > 0)); then
    echo "preflight: $FAILS item(s) FAILED"
    return 1
  fi
  echo "preflight: all OK"
}

cd -- "$REPO_ROOT"
# Before redact starts: from then on the operator's public IP and the project id are masked in everything shown.
screen_masks
set +e
main "$@" 2>&1 | screen_filter
status=("${PIPESTATUS[@]}")
set -e
((status[1] == 0)) || exit 1
exit "${status[0]}"
