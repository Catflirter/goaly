# shellcheck shell=bash
# Screen safety for make demo and make preflight (spec section 9.1): what they put on the screen, and the
# demo's recordings, go through redact (scripts/lib/checks.sh), which masks secrets; this file adds the
# operator's public IP and the GCP project id to what it masks. Sourced after scripts/lib/checks.sh.
#
#   screen_masks           call once, before the first redact: from then on redact masks the operator's IP as
#                          <operator_ip> (vr_operator_ip) and the GCP project id as <GCP_PROJECT>. Sets
#                          SCREEN_IP_MASKED and SCREEN_PROJECT_MASKED (1 masked, 0 unknown, so not masked)
#   screen_env_project     print GCP_PROJECT as this repo's .env sets it (what make down asks for)
#   screen_filter          redact, ignoring SIGINT: see below
#
# The project id is masked wherever it appears in a line and in any case: in the state bucket
# (<id>-k3s-tfstate), service accounts (<name>@<id>.iam.gserviceaccount.com), self-links and FQDNs
# (k3s-cp.<zone>.c.<id>.internal), and down.sh's own "in GCP project <id>". Both ids are masked: the one in
# .env (what the scripts use) and the caller's GCP_PROJECT, when they differ. Only an id _gcp_project_ok
# accepts is masked, so it holds no sed metacharacter. (Task 25 fix round 1; minors-2a's _vr_project_mask
# in checks.sh does the same for verify runs.)

# screen_env_project: GCP_PROJECT from $REPO_ROOT/.env, read with load_env in a subshell, so nothing else of
# .env reaches the caller; nothing (status 1) without a readable .env or a valid id.
screen_env_project() (
  [[ -f "$REPO_ROOT/.env" ]] || exit 1
  load_env >/dev/null 2>&1
  _gcp_project_ok || exit 1
  printf '%s\n' "$GCP_PROJECT"
)

# _screen_project_expr ID: an ERE matching ID in any case (portable ERE has no case flag: [xX] per letter).
_screen_project_expr() {
  local id="$1" upper e="" i
  upper="$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')"
  for ((i = 0; i < ${#id}; i++)); do
    if [[ "${id:i:1}" == [a-z] ]]; then e+="[${id:i:1}${upper:i:1}]"; else e+="${id:i:1}"; fi
  done
  printf '%s\n' "$e"
}

# shellcheck disable=SC2034 # SCREEN_*_MASKED are read by the callers (demo.sh, preflight.sh)
screen_masks() {
  local id ids="" env_id
  SCREEN_IP_MASKED=0
  SCREEN_PROJECT_MASKED=0
  # First: vr_operator_ip sets _VR_MASK_EXPRS afresh the first time it runs.
  if vr_operator_ip; then SCREEN_IP_MASKED=1; fi
  env_id="$(screen_env_project)" || env_id=""
  for id in "$env_id" "${GCP_PROJECT:-}"; do
    if GCP_PROJECT="$id" _gcp_project_ok && [[ " $ids " != *" $id "* ]]; then
      ids+=" $id"
      _VR_MASK_EXPRS+=(-e "s/$(_screen_project_expr "$id")/<GCP_PROJECT>/g")
      SCREEN_PROJECT_MASKED=1
    fi
  done
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
