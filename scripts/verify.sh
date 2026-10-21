#!/usr/bin/env bash
# make verify: run the registered checks into verification/<YYYY-MM-DDTHHMM-0700>/ (spec section 9).
#
#   scripts/verify.sh [GROUP…]   GROUP is A, B, C, D or E; groups run in the order given
#
# Without arguments it runs A B C D, plus E when RESILIENCE=1 (E reboots nodes; slow and costly).
# Exit status: 0 when no check failed (SKIP is not a failure) and gitleaks found nothing in the
# run directory, 1 otherwise, 2 for an unknown group.
set -euo pipefail

# checks.sh is linted on its own; pre-commit runs shellcheck without -x, so don't follow it.
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/checks.sh"

VR_GROUPS=(A B C D E)
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

# Tasks 18 (A–C), 21 (D) and 22 (E) register their checks below this line.
# --- registered checks ---

main() {
  local groups=("$@") group i ran
  if (($# == 0)); then
    groups=(A B C D)
    if [[ "${RESILIENCE:-0}" == 1 ]]; then
      groups+=(E)
    fi
  fi
  for group in "${groups[@]}"; do
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
