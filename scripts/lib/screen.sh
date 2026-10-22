# shellcheck shell=bash
# Screen safety for make demo and make preflight (spec section 9.1): what they put on the screen, and the
# demo's recordings, go through redact (scripts/lib/checks.sh), which masks secrets and the GCP project id
# (_vr_project_mask: .env's and the caller's GCP_PROJECT when they differ, in any case, anywhere in a line);
# this file adds the operator's public IP to what it masks. Sourced after scripts/lib/checks.sh.
#
#   screen_masks [--local] call once, before the first redact: from then on redact masks the operator's IP as
#                          <operator_ip> (vr_operator_ip) and the GCP project id as <GCP_PROJECT>. Sets
#                          SCREEN_IP_MASKED and SCREEN_PROJECT_MASKED (1 masked, 0 unknown, so not masked).
#                          --local (replay): no network lookup, so the IP is masked only when .env sets it
#   screen_env_project     print GCP_PROJECT as this repo's .env sets it (what make down asks for)
#   screen_filter          redact, ignoring SIGINT: see below

# screen_env_project: GCP_PROJECT from $REPO_ROOT/.env (_vr_project_lookup, a subshell, so nothing else of
# .env reaches the caller); nothing (status 1) without a readable .env or a valid id.
screen_env_project() {
  local id
  [[ -f "$REPO_ROOT/.env" ]] || return 1
  id="$(_vr_project_lookup 2>/dev/null)" || return 1
  GCP_PROJECT="$id" _gcp_project_ok || return 1
  printf '%s\n' "$id"
}

# _screen_env_operator_ip: OPERATOR_IP as .env sets it, trimmed (empty when unset); status 1 without a
# readable .env.
_screen_env_operator_ip() (
  load_env >/dev/null 2>&1 || exit 1
  _vr_trim "${OPERATOR_IP:-}"
)

# shellcheck disable=SC2034 # SCREEN_*_MASKED are read by the callers (demo.sh, preflight.sh)
screen_masks() {
  local ip=""
  SCREEN_IP_MASKED=0
  SCREEN_PROJECT_MASKED=0
  # vr_operator_ip asks checkip.amazonaws.com only when .env has no OPERATOR_IP; --local never lets it.
  if [[ "${1:-}" != --local ]] || { ip="$(_screen_env_operator_ip)" && [[ -n "$ip" ]]; }; then
    if vr_operator_ip; then SCREEN_IP_MASKED=1; fi
  fi
  # .env's id, also when the caller exports another GCP_PROJECT: _vr_project_mask masks both (as after vr_init).
  _VR_PROJECT="$(screen_env_project)" || _VR_PROJECT=""
  _VR_PROJECT_LOOKED=1
  _vr_project_mask
  if [[ -n "$_VR_PROJECT_IDS" ]]; then SCREEN_PROJECT_MASKED=1; fi
}

# screen_filter: redact for a pipeline whose writer may be interrupted. Ctrl-C reaches every process of
# the terminal's foreground process group, the filter (sed) included; a filter that died with it would leave
# the interrupted command (terraform, cpu-pool.sh, down.sh, tunnel.sh start) writing its last words (
# "Interrupt received...") into a closed pipe, and SIGPIPE would kill it before it has stopped cleanly and
# released the Terraform state lock. So the filter ignores SIGINT (sed inherits that) and ends when its
# writer closes the pipe.
screen_filter() {
  (
    trap '' INT
    redact
  )
}
