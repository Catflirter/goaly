# shellcheck shell=bash
# Verification harness shared by scripts/verify.sh and the demo (spec section 9). Source it; do not run it.
#
#   vr_init DIR                     start a run: DIR (mode 0700)/SUMMARY.md with an empty results table;
#                                   returns 2 if DIR already holds files; finds the operator's IP
#                                   (vr_operator_ip), which redact then masks as <operator_ip>, and reads
#                                   GCP_PROJECT from .env, which redact masks as <GCP_PROJECT>
#   run_check ID TITLE FUNC [ARG…]  run one check into DIR/<NN>-<ID>.txt; FUNC returns 0 PASS, 77 SKIP, else FAIL
#   vr_cmd CMD…                     print "$ CMD" (redacted), then run CMD; returns CMD's status
#   vr_cap [--no-echo] CMD…         vr_cmd for $(…): command and output to stderr, stdout to the caller
#   vr_bg FILE CMD…                 print "$ CMD &", run CMD in the background into FILE (VR_BG_PID)
#   vr_retry TIMEOUT_S INTERVAL_S CMD…  retry CMD until it succeeds, for at most TIMEOUT_S seconds
#   redact                          stdin -> stdout filter that replaces secret values with <redacted>
#                                   and masks .env's project id and the operator's IP (Ruling R39)
#   vr_finish                       write the totals, scan the run with gitleaks; 1 on any FAIL or leak
#
# The check functions check_<id>_<name> (A–C: Task 18; D: Task 21; E: Task 22) are registered in
# scripts/verify.sh; the demo (Task 25) calls them directly. Each returns 0
# (PASS), 1 (FAIL) or 77 (SKIP); one that cannot read .env exits 2 (a FAIL), so call it in a subshell.
# Every command a check runs goes through vr_cmd, vr_cap or vr_bg, so its evidence shows what ran
# (node_external_ip, probe_on and echo_on appear by name; their kubectl queries are in common.sh;
# so do cp_resolve and _worker_zone, the SSH zone lookups from Terraform outputs).
# The offline tests set VR_NO_WAIT=1 (no fixed sleeps) and VR_PF_PORT (C3's local port).
# Rules for check functions:
#   - bound every wait (kubectl wait --timeout=…, curl --max-time …): run_check has no timeout of its own;
#     kubectl exec runs under kc_within SECONDS and other network calls under bounded SECONDS;
#   - background jobs (kubectl port-forward &, tcpdump &) are killed when the check returns, because
#     run_check waits for every process holding the check's output; so do not set an EXIT trap in a
#     check (it replaces that cleanup), and do not leave grandchildren (setsid, nohup) running;
#     _vr_reap sends them TERM every 0.1 s while any is left and KILL after about 5 s (longer under
#     load), so a job that needs more than about 5 s after TERM to shut down is killed (Ruling R56);
#   - a check that leaves something behind until its end (a pcap, Pods, a namespace, a listening
#     server, GPU instances, a firewall rule) removes it on SIGINT/SIGTERM/SIGHUP too, with
#     _vr_on_signal --left WHAT CMD; the operator sees the cleanup and, if it fails, what may remain and
#     the command that removes it (the GPU checks D4-D6 do this through _vr_gpu_guarded, which also
#     reports a failed cleanup that was not interrupted; E2 and E3 through _vr_e_guarded, whose cleanup
#     also runs when the check dies);
#   - run_check gives a check fd 3 (the operator's stderr) for that; do not use fd 3 otherwise.

# common.sh is linted on its own; pre-commit runs shellcheck without -x, so don't follow it.
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

VR_TZ=America/Los_Angeles

# --- redact -------------------------------------------------------------------------------------

# A secret value: everything up to whitespace, a quote or a comma (the end of a YAML/JSON/shell value).
_vr_value='[^[:space:]"'\'',]+'
# An optional quote around a key or before a value: YAML 'single', JSON/shell "double".
_vr_q="[\"']?"
# Portable ERE (GNU and BSD sed): no \b, no I flag, so case-insensitive words are spelled as [Xx] classes.
# Line-based: a multi-line value is only caught as a PEM private key body (first expression) and a
# YAML block scalar (`token: |`) is not; gitleaks in vr_finish is the backstop for what this misses.
_VR_REDACT_EXPRS=(
  # PEM private key bodies: the base64 lines between -----BEGIN … PRIVATE KEY----- and -----END …-----
  # (first, so the private-key expression below has not rewritten the BEGIN line yet)
  -e "/-----BEGIN [A-Z ]*PRIVATE KEY-----[[:space:]]*\$/,/-----END [A-Z ]*PRIVATE KEY-----/s#^([[:space:]]*)[A-Za-z0-9+/=]+[[:space:]]*\$#\\1<redacted>#"
  # k3s join token K10<ca-hash>::<user>:<password>; the password may hold commas, so stop only at
  # whitespace or a quote
  -e "s/K10[0-9a-fA-F]{64}::[^[:space:]\"']+/<redacted>/g"
  # kubeconfig credentials, service-account / bootstrap tokens, Secret TLS keys and passwords
  # (YAML and JSON forms, either quote)
  -e "s/((client-key-data|client-certificate-data|certificate-authority-data|token|tls\\.key|[Pp]assword)${_vr_q}:[[:space:]]*${_vr_q})${_vr_value}/\\1<redacted>/g"
  # bearer token as a command-line flag (kubectl --token=…, --token …)
  -e "s/(--token[= ]${_vr_q})${_vr_value}/\\1<redacted>/g"
  # Lambda API key assigned or keyed: env (=), YAML (:), JSON ("LAMBDA_API_KEY": "…"), like .gitleaks.toml
  -e "s/(LAMBDA_API_KEY${_vr_q}[[:space:]]*[:=][[:space:]]*${_vr_q})${_vr_value}/\\1<redacted>/g"
  # HTTP credentials (curl -v prints HTTP/2 header names in lower case)
  -e "s/([Aa][Uu][Tt][Hh][Oo][Rr][Ii][Zz][Aa][Tt][Ii][Oo][Nn]:[[:space:]]*([Bb][Ee][Aa][Rr][Ee][Rr]|[Bb][Aa][Ss][Ii][Cc])[[:space:]]+${_vr_q})${_vr_value}/\\1<redacted>/g"
  # WireGuard / TLS private keys: "private key: …", "PrivateKey = …", any case
  -e "s/([Pp][Rr][Ii][Vv][Aa][Tt][Ee][[:space:]_-]?[Kk][Ee][Yy]${_vr_q}[[:space:]]*[:=][[:space:]]*${_vr_q})${_vr_value}/\\1<redacted>/g"
  # Lambda API key by its own prefix (secret_<name>_<hex>[.<random>]), wherever it appears (.gitleaks.toml)
  -e "s/(^|[^A-Za-z0-9_])secret_[A-Za-z0-9_-]+_[0-9a-f]{16,}(\\.[A-Za-z0-9]+)?/\\1<redacted>/g"
)
# Line-buffered, so redacted output reaches a terminal (the demo) as it is produced: GNU sed -u, BSD sed -l.
_VR_SED=(sed)
if sed -u -e '' </dev/null >/dev/null 2>&1; then
  _VR_SED=(sed -u)
elif sed -l -e '' </dev/null >/dev/null 2>&1; then
  _VR_SED=(sed -l)
fi

# The operator's public IP, masked as <operator_ip> once vr_operator_ip has found it (empty until then).
_VR_MASK_EXPRS=()

# The GCP project id, masked as <GCP_PROJECT>: _VR_PROJECT is the one read from .env by vr_init or
# by _vr_project_mask (empty until then, or when it is not a project id); _VR_PROJECT_LOOKED is set
# once either has looked. Kept apart from _VR_MASK_EXPRS, which vr_operator_ip resets.
# _VR_PROJECT_EXPRS masks the ids listed in _VR_PROJECT_IDS (see _vr_project_mask).
_VR_PROJECT=""
_VR_PROJECT_LOOKED=""
_VR_PROJECT_IDS=""
_VR_PROJECT_EXPRS=()

# _vr_project_mask: point _VR_PROJECT_EXPRS at the project ids this shell knows: vr_init's, and
# GCP_PROJECT once vr_env or load_env has read .env. When it knows none, it reads .env itself, once
# (_vr_project_lookup, in a subshell): a redact may start before .env is loaded (the demo's screen
# pipe), and a check's own vr_env runs in the check's subshell, out of reach of the redact it is
# piped into. So a redact masks .env's id whatever the order; the ids are fixed when it starts.
# Only an id _gcp_project_ok accepts is masked, so it holds no sed metacharacter. It is masked
# anywhere in a line and in any case: in A2's FQDNs (k3s-cp.<zone>.c.<id>.internal), the state
# bucket (<id>-k3s-tfstate) and service accounts (<name>@<id>.iam.gserviceaccount.com).
_vr_project_mask() {
  local id ids="" upper e i
  if [[ -z "$_VR_PROJECT_LOOKED" ]] && ! GCP_PROJECT="$_VR_PROJECT" _gcp_project_ok && ! _gcp_project_ok; then
    _VR_PROJECT_LOOKED=1
    _VR_PROJECT="$(_vr_project_lookup 2>/dev/null)" || _VR_PROJECT=""
    GCP_PROJECT="$_VR_PROJECT" _gcp_project_ok || _VR_PROJECT=""
  fi
  for id in "$_VR_PROJECT" "${GCP_PROJECT:-}"; do
    if GCP_PROJECT="$id" _gcp_project_ok && [[ " $ids " != *" $id "* ]]; then ids+="${ids:+ }$id"; fi
  done
  [[ "$ids" != "$_VR_PROJECT_IDS" ]] || return 0
  _VR_PROJECT_IDS="$ids"
  _VR_PROJECT_EXPRS=()
  for id in $ids; do
    # Portable ERE has no case-insensitive flag (see above): each letter becomes an [xX] class.
    upper="$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')"
    e=""
    for ((i = 0; i < ${#id}; i++)); do
      if [[ "${id:i:1}" == [a-z] ]]; then e+="[${id:i:1}${upper:i:1}]"; else e+="${id:i:1}"; fi
    done
    _VR_PROJECT_EXPRS+=(-e "s/$e/<GCP_PROJECT>/g")
  done
}

# LC_ALL=C: match bytes, not characters. All patterns are ASCII, and BSD sed under a UTF-8 locale
# aborts ("illegal byte sequence") on binary or non-UTF-8 output such as tcpdump -A payloads.
redact() {
  _vr_project_mask
  LC_ALL=C "${_VR_SED[@]}" -E "${_VR_REDACT_EXPRS[@]}" ${_VR_PROJECT_EXPRS[@]+"${_VR_PROJECT_EXPRS[@]}"} \
    ${_VR_MASK_EXPRS[@]+"${_VR_MASK_EXPRS[@]}"}
}

# _vr_trim TEXT: TEXT without leading and trailing whitespace (Python's str.strip, as gpuctl uses).
_vr_trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  printf '%s\n' "${s%"${s##*[![:space:]]}"}"
}

# _vr_operator_ip_lookup: print "IP SOURCE" for the operator's public IP, found as gpuctl finds it
# (lambda/gpuctl/infra.py detect_operator_ip): OPERATOR_IP from .env, trimmed; when that is empty,
# the address checkip.amazonaws.com reports (one curl, at most 10 s). Runs in a subshell, so .env's
# values stay out of the caller. Exits 2 without a readable .env (load_env), 1 when OPERATOR_IP is not
# an IPv4 address (gpuctl refuses it) or the lookup gives none.
_vr_operator_ip_lookup() (
  load_env
  ip="$(_vr_trim "${OPERATOR_IP:-}")"
  if [[ -n "$ip" ]]; then
    is_ipv4 "$ip" || exit 1
    printf '%s OPERATOR_IP in .env\n' "$ip"
    exit 0
  fi
  ip="$(bounded 10 curl -fsS --max-time 5 https://checkip.amazonaws.com)" || exit 1
  ip="$(_vr_trim "$ip")"
  is_ipv4 "$ip" || exit 1
  printf '%s checkip.amazonaws.com\n' "$ip"
)

# _vr_project_lookup: print GCP_PROJECT from .env. Runs in a subshell, so .env's values stay out of
# the caller; exits 2 without a readable .env (load_env).
_vr_project_lookup() (
  load_env
  printf '%s\n' "${GCP_PROJECT:-}"
)

# vr_operator_ip: find the operator's public IP once (vr_init does, so every check of the run shares
# it) and from then on mask it as <operator_ip> in everything redact filters: all evidence, not only
# B6. Lambda's "operator ssh" rule holds it, and a run promoted to verification/final is committed
# (Ruling R39 keeps real IPs out of what is delivered). Sets VR_OPERATOR_IP and VR_OPERATOR_IP_FROM
# (where it came from); returns 1 when it is unknown, and then nothing can be masked: B6 fails if a
# Lambda rule shows it. The IP must stand alone to be masked (198.51.100.2 stays in 198.51.100.20 and
# in 10.198.51.100.2). Each match includes the characters around the IP, so one pass misses an IP
# right after another ("IP,IP"); after it, no two unmasked ones are adjacent, so a second pass gets them.
vr_operator_ip() {
  local out e
  if [[ -z "${_VR_OPIP_DONE:-}" ]]; then
    _VR_OPIP_DONE=1
    VR_OPERATOR_IP=""
    VR_OPERATOR_IP_FROM=""
    _VR_OPIP_RC=0
    _VR_MASK_EXPRS=()
    out="$(_vr_operator_ip_lookup 2>/dev/null)" || _VR_OPIP_RC=$?
    if ((_VR_OPIP_RC == 0)); then
      read -r VR_OPERATOR_IP VR_OPERATOR_IP_FROM <<< "$out"
      e="s/(^|[^0-9.])${VR_OPERATOR_IP//./\\.}(\$|[^0-9])/\\1<operator_ip>\\2/g"
      _VR_MASK_EXPRS=(-e "$e" -e "$e")
    fi
  fi
  [[ -n "${VR_OPERATOR_IP:-}" ]]
}

# --- run bookkeeping ----------------------------------------------------------------------------

# ISO 8601 local time in America/Los_Angeles, e.g. 2026-10-21T14:30:05-07:00.
_vr_now() {
  local ts
  ts="$(TZ="$VR_TZ" date +%Y-%m-%dT%H:%M:%S%z)"
  printf '%s:%s\n' "${ts:0:${#ts}-2}" "${ts: -2}"
}

# vr_init DIR: create DIR and its SUMMARY.md. Refuses a DIR that already holds files (returns 2), so
# the evidence of two runs is never mixed. DIR is 0700 and its files 0600: unreviewed evidence (cluster
# topology, IPs, anything redact missed) stays readable by the operator only; promoting a run to
# verification/final/ is a deliberate copy. It also finds the operator's public IP for the whole run
# (vr_operator_ip) and reads GCP_PROJECT from .env, so that redact masks both in all evidence and on
# the operator's screen (Ruling R39), and says in SUMMARY.md whether each is masked; when one is
# unknown, it warns.
vr_init() {
  local dir="$1"
  if [[ -d "$dir" && -n "$(ls -A -- "$dir")" ]]; then
    log "error: $dir already holds files from another run; start again in a minute (new directory name)"
    return 2
  fi
  (umask 077 && mkdir -p -- "$dir")
  chmod 700 -- "$dir" # mkdir -p keeps the mode of a DIR that already exists (empty)
  VR_DIR="$dir"
  VR_SEQ=0
  VR_FAILS=0
  VR_SKIPS=0
  local opip project project_rc=0
  _VR_PROJECT_LOOKED=1
  _VR_PROJECT="$(_vr_project_lookup 2>/dev/null)" || project_rc=$?
  if GCP_PROJECT="$_VR_PROJECT" _gcp_project_ok; then
    project="masked as <GCP_PROJECT>"
  else
    _VR_PROJECT=""
    if ((project_rc == 2)); then
      # A2's describe needs no .env, so its evidence would show the raw id: say so, not only in SUMMARY.md.
      project="not masked (.env is not readable)"
      log "warning: .env is not readable, so the evidence cannot mask the GCP project id"
    else
      project="not masked (GCP_PROJECT in .env is empty or not a project id)"
      log "warning: GCP_PROJECT in .env is empty or not a GCP project id, so the evidence cannot mask it"
    fi
  fi
  _vr_project_mask
  if vr_operator_ip; then
    opip="masked as <operator_ip> ($VR_OPERATOR_IP_FROM)"
  else
    if ((_VR_OPIP_RC == 2)); then
      opip="not determined (.env is not readable)" # the checks that need .env say so themselves
    else
      opip="not determined (OPERATOR_IP in .env is empty or not an address, and checkip.amazonaws.com gave none)"
      log "warning: cannot determine the operator's public IP, so the evidence cannot mask it; B6 fails if a"
      log "Lambda rule shows it. Set OPERATOR_IP in .env to the address gpuctl used, or check the network."
    fi
  fi
  (
    umask 077
    {
      printf '# Verification run\n\n'
      printf 'Started: %s (%s)\n\n' "$(_vr_now)" "$VR_TZ"
      printf 'Operator IP: %s\n\n' "$opip"
      printf 'GCP project id: %s\n\n' "$project"
      printf '| # | ID | Check | Result |\n'
      printf '|---|---|---|---|\n'
    } >"$VR_DIR/SUMMARY.md"
  )
}

# EXIT trap of a check's subshell: kill the background jobs it left running (see the header). TERM
# again until they are gone, KILL what is left after 5s: one TERM is not enough. This trap makes bash
# catch TERM, and a job runs that handler until it has exec'd its program, so a TERM that arrives
# before then can be dropped and the program runs on, holding the check's output open.
_vr_reap() {
  local pids i
  for ((i = 0; i < 50; i++)); do
    pids="$(jobs -pr)"
    [[ -n "$pids" ]] || return 0
    # shellcheck disable=SC2086 # one PID per word
    kill $pids 2>/dev/null || true
    sleep 0.1
  done
  pids="$(jobs -pr)"
  # shellcheck disable=SC2086 # one PID per word
  [[ -z "$pids" ]] || kill -KILL $pids 2>/dev/null || true
}

# run_check ID TITLE FUNC [ARGS…]: run FUNC ARGS in a subshell (so exit and set -e failures end only
# the check) with stdin from /dev/null, stdout+stderr redacted into $VR_DIR/<NN>-<ID>.txt.
# FUNC's status: 0 PASS, 77 SKIP, anything else FAIL; a failed redact filter (lost evidence) is a FAIL
# too. Always returns 0; vr_finish reports the outcome.
# A hangup (SIGHUP: the operator's SSH session dropped) ends this shell at once, but the filter ignores
# it, and the check's subshell exits (129) only after its foreground command, so a check's cleanup
# (_vr_on_signal) runs to its end and is recorded in the evidence file (Task 22b review X0-1, Ruling R62).
run_check() {
  local id="$1" title="$2" func="$3"
  shift 3
  local seq file rc redact_rc result errexit=0
  VR_SEQ=$((VR_SEQ + 1))
  printf -v seq '%02d' "$VR_SEQ"
  file="$VR_DIR/$seq-$id.txt"
  (
    umask 077
    {
      printf '# %s — %s\n' "$id" "$title"
      printf '# started: %s\n' "$(_vr_now)"
    } >"$file"
  )

  # The check keeps the caller's errexit; run_check itself must survive the check failing.
  [[ $- == *e* ]] && errexit=1
  set +e
  # fd 3 is run_check's own stderr (the operator's terminal under make verify), for _vr_tell.
  (
    if ((errexit)); then set -e; fi
    # shellcheck disable=SC2030 # meant for this check's subshell only (_vr_tell)
    _VR_CHECK_ID="$id" _VR_CHECK_FILE="$file" _VR_TELL_FD=3
    trap _vr_reap EXIT
    # A hangup ends this subshell only once its foreground command is over (Ruling R62): dying at once would
    # run _vr_reap while the check still runs, and its SIGTERM (then, after 5 s, its SIGKILL) would hit a
    # check's cleanup or a command that ignores SIGHUP to finish (E's applies). Test:
    # test_run_check_ends_a_hung_up_check_only_after_its_foreground_command (review C1-2).
    trap 'exit 129' HUP
    "$func" "$@"
  ) 3>&2 </dev/null 2>&1 | (
    trap '' HUP
    redact
  ) >>"$file"
  local -a status=("${PIPESTATUS[@]}")
  if ((errexit)); then set -e; fi
  rc=${status[0]}
  redact_rc=${status[1]}

  # Output that ends without a newline would swallow the footer.
  if [[ -n "$(tail -c 1 -- "$file")" ]]; then
    printf '\n' >>"$file"
  fi
  if ((redact_rc != 0)); then
    printf '# redact failed (exit %d); output incomplete\n' "$redact_rc" >>"$file"
  fi
  if ((redact_rc != 0)) || { ((rc != 0)) && ((rc != 77)); }; then
    result=FAIL
    VR_FAILS=$((VR_FAILS + 1))
  elif ((rc == 77)); then
    result=SKIP
    VR_SKIPS=$((VR_SKIPS + 1))
  else
    result=PASS
  fi
  printf '# finished: %s — %s (exit %d)\n' "$(_vr_now)" "$result" "$rc" >>"$file"
  printf '| %d | %s | %s | %s |\n' "$VR_SEQ" "$id" "${title//|/\\|}" "$result" >>"$VR_DIR/SUMMARY.md"
  printf '[%s] %-4s %-4s %s\n' "$seq" "$id" "$result" "$title"
  return 0
}

# _vr_quote CMD…: "$ CMD" (not yet redacted) as vr_cmd, vr_cap and vr_bg record it.
# Arguments are shown shell-quoted with single quotes, not printf %q: %q turns "Bearer X" into
# Bearer\ X, which the redact patterns (key, whitespace, value) would no longer recognise.
_vr_quote() {
  local arg line='$'
  for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Za-z0-9_./:=@%+,-]+$ ]]; then
      line+=" $arg"
    else
      line+=" $(_vr_sq "$arg")"
    fi
  done
  printf '%s\n' "$line"
}

# _vr_sq STRING: STRING in single quotes, each ' in it written as '\''. Not ${s//\'/…} inside double
# quotes: bash <= 4.2 (macOS's /bin/bash 3.2) does no quote removal on that replacement, which would
# keep its backslashes and print a command that cannot be pasted.
_vr_sq() {
  local s="$1" q="'" out=""
  while [[ "$s" == *"$q"* ]]; do
    out+=${s%%"$q"*}$q\\$q$q
    s=${s#*"$q"}
  done
  printf '%s\n' "$q$out$s$q"
}

# vr_cmd CMD…: record the command (redacted) in the check's output, then run it.
vr_cmd() {
  # A failed filter would hide what runs next; do not run it.
  _vr_quote "$@" | redact || return
  "$@"
}

# vr_cap [--no-echo] CMD…: vr_cmd for $(…): the command and its output are recorded on stderr (the
# check's evidence: run_check sends both streams there), and CMD's stdout is printed for the caller
# to parse. --no-echo records the command only, for bulky output (JSON, a packet list) the check
# summarises itself. Returns CMD's status.
vr_cap() {
  local echo=1 out rc=0
  if [[ "${1:-}" == --no-echo ]]; then
    echo=0
    shift
  fi
  _vr_quote "$@" | redact >&2 || return
  out="$("$@")" || rc=$?
  if [[ -n "$out" ]]; then
    if ((echo)); then printf '%s\n' "$out" >&2; fi
    printf '%s\n' "$out"
  fi
  return "$rc"
}

# vr_bg FILE CMD…: record "$ CMD &" and start CMD in the background with its stdout and stderr in
# FILE (the check shows FILE once CMD is done); VR_BG_PID is its PID. The check waits for it or
# stops it with _vr_stop before it returns.
vr_bg() {
  local file="$1" line
  shift
  line="$(_vr_quote "$@")"
  printf '%s &\n' "$line" | redact || return
  "$@" >"$file" 2>&1 </dev/null &
  VR_BG_PID=$!
}

# --- check helpers (Task 18) --------------------------------------------------------------------

# The bound of one kubectl request (get, describe, apply, top); waits carry their own --timeout.
# kubectl exec streams are not bounded by it: exec runs under kc_within (below), and the exec'd
# commands bound themselves too (curl -m, ping -W, timeout N iperf3).
_VR_RT="--request-timeout=$KC_REQUEST_TIMEOUT"

# bounded SECONDS CMD…: run the program CMD for at most SECONDS; exit 124 when it had to be stopped.
# --foreground keeps CMD in the terminal's process group, so Ctrl-C still reaches it. `command` skips
# a shell function named timeout (the offline tests replace it for check B5). In an interrupt cleanup
# (_VR_CLEANING=1, set by _vr_signalled) CMD must ignore a further Ctrl-C as the cleanup's other
# commands do, but timeout sets its own SIGINT handler, so CMD would not inherit the ignore, and
# passes SIGINT on to CMD or ends it (GNU and uutils differ): there the shell bounds CMD instead
# (_vr_within), and CMD inherits the ignore. A SIGTERM and the bound still stop CMD.
bounded() {
  if [[ "${_VR_CLEANING:-}" == 1 ]]; then
    _vr_within "$@"
  else
    command timeout --foreground "$@"
  fi
}

# kc_within SECONDS ARGS…: kc with an outer time bound, for kubectl exec, so a stalled API tunnel
# fails the call instead of hanging the check. A call stopped by the bound exits 124, which no check
# takes for a verdict (B5 and C8 want curl's own 7 or 28).
kc_within() {
  local seconds="$1"
  shift
  bounded "$seconds" kubectl --kubeconfig "$KUBECONFIG_FILE" "$@"
}

# _vr_within SECONDS CMD…: run CMD (a shell function too, such as cp_ssh) in the background and stop
# it after SECONDS. Returns CMD's status, or 124.
_vr_within() {
  local seconds="$1" pid i
  shift
  "$@" &
  pid=$!
  for ((i = 0; i < seconds * 5; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      return
    fi
    sleep 0.2
  done
  _vr_stop "$pid"
  return 124
}

# _vr_on_signal [--left WHAT] [--hint TEXT] [--within SECONDS] [--hup] CMD…: when the check is
# interrupted (SIGINT: Ctrl-C during make verify or the demo; SIGTERM), kill its background jobs, run CMD
# (for at most SECONDS, default 120; a shell function is fine) and end the check by the same signal, so
# verify.sh and the demo stop as they would without the trap. WHAT names what CMD removes; if CMD
# fails, the operator is told it may remain and how to remove it: TEXT, else the command that runs CMD
# by hand (_vr_removal). --hup: a hangup (SIGHUP: the operator's SSH session dropped) runs CMD too,
# once the check's foreground command is over (E2 and E3, Ruling R62); without it SIGHUP keeps
# run_check's trap. _vr_on_signal without CMD removes the trap (a --hup one goes back to run_check's
# `exit 129`). The EXIT trap stays the harness's (_vr_reap).
_vr_on_signal() {
  if (($# == 0)); then
    trap - INT TERM
    if [[ "${_VR_ON_SIGNAL_HUP:-}" == 1 ]]; then trap 'exit 129' HUP; fi
    _VR_ON_SIGNAL_HUP=0
    return 0
  fi
  _VR_ON_SIGNAL_LEFT="what that command removes"
  _VR_ON_SIGNAL_HINT=""
  _VR_ON_SIGNAL_WITHIN=120
  _VR_ON_SIGNAL_HUP=0
  while :; do
    case "$1" in
      --left) _VR_ON_SIGNAL_LEFT="$2" ;;
      --hint) _VR_ON_SIGNAL_HINT="$2" ;;
      --within) _VR_ON_SIGNAL_WITHIN="$2" ;;
      --hup)
        _VR_ON_SIGNAL_HUP=1
        shift
        continue
        ;;
      *) break ;;
    esac
    shift 2
  done
  _VR_ON_SIGNAL=("$@")
  trap '_vr_signalled INT' INT
  trap '_vr_signalled TERM' TERM
  if ((_VR_ON_SIGNAL_HUP)); then trap '_vr_signalled HUP' HUP; fi
}

# _vr_tell [LINE…]: LINEs (else stdin's lines), redacted, to the operator and into the check's
# evidence file. The operator sees them on run_check's own stderr (fd 3; outside run_check, stderr),
# prefixed with the check's id. For the interrupt cleanup: Ctrl-C has also ended the evidence filter
# the check's output normally goes through, so that output would be lost. Outside run_check (a check
# called directly, as the demo does) that is fd 4: the caller's stderr, saved before the loop's own
# stderr goes to /dev/null, where fd 2 would reach no one.
# shellcheck disable=SC2031 # run_check sets the _VR_CHECK_* and _VR_TELL_FD in the check's subshell
_vr_tell() {
  local line fd="${_VR_TELL_FD:-4}"
  if (($#)); then printf '%s\n' "$@"; else cat; fi | redact | while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -n "${_VR_CHECK_FILE:-}" ]]; then printf '%s\n' "$line" >>"$_VR_CHECK_FILE"; fi
    printf '%s: %s\n' "${_VR_CHECK_ID:-check}" "$line" >&"$fd"
  done 4>&2 2>/dev/null
}

# _vr_signalled SIGNAL: the trap set by _vr_on_signal. SIGPIPE is ignored and errexit is off while it
# cleans up (a closed pipe must not end kubectl or gcloud before they have done their work). The
# cleanup's progress, its output and its outcome go to the operator and the evidence (_vr_tell).
# A further Ctrl-C or hangup is ignored until then, by this shell and (inherited) by the cleanup's
# commands, the ones under bounded or kc_within too (_VR_CLEANING: see bounded), so it can neither cut
# the cleanup short nor hide its outcome. A further SIGTERM stops the cleanup's commands (they keep its
# default action: TERM is also how the bound stops them) but not this shell, which still reports the
# outcome; the first signal then ends the check.
_vr_signalled() {
  local line rc
  trap '' INT HUP
  trap ':' TERM
  set +e
  trap '' PIPE
  _VR_CLEANING=1
  _vr_reap
  line="$(_vr_quote "${_VR_ON_SIGNAL[@]}")"
  line="${line#\$ }"
  _vr_tell "interrupted (SIG$1); cleaning up (at most ${_VR_ON_SIGNAL_WITHIN:-120} s): $line"
  _vr_within "${_VR_ON_SIGNAL_WITHIN:-120}" "${_VR_ON_SIGNAL[@]}" </dev/null 2>&1 | _vr_tell
  rc=${PIPESTATUS[0]}
  _vr_cleanup_outcome "$rc"
  trap - INT TERM HUP PIPE
  kill -s "$1" "${BASHPID:-$$}"
  exit 1
}

# _vr_cleanup_outcome RC: tell the operator (_vr_tell) how the cleanup armed by _vr_on_signal ended (RC:
# its status, 124 when its bound stopped it): done, or what may remain and how to remove it.
_vr_cleanup_outcome() {
  local how
  if (($1 == 0)); then
    _vr_tell "cleanup done"
    return 0
  fi
  how="exit $1"
  (($1 != 124)) || how="stopped after ${_VR_ON_SIGNAL_WITHIN:-120} s"
  if [[ -n "${_VR_ON_SIGNAL_HINT:-}" ]]; then
    _vr_tell "cleanup FAILED ($how); may remain: $_VR_ON_SIGNAL_LEFT" "$_VR_ON_SIGNAL_HINT"
  else
    _vr_tell "cleanup FAILED ($how); may remain: $_VR_ON_SIGNAL_LEFT" \
      "to remove it, run from the repository root: $(_vr_removal "${_VR_ON_SIGNAL[@]}")"
  fi
}

# _vr_removal CMD…: the command line that runs a check's cleanup CMD by hand from the repository root.
# In a child shell: pasted into the operator's own bash, a die (exit) ends only that child, and .env's
# values and the harness's functions do not stay behind in the operator's shell.
_vr_removal() {
  local line
  line="$(_vr_quote "$@")"
  line="source scripts/lib/checks.sh && vr_env && ${line#\$ }"
  printf 'bash -c %s\n' "$(_vr_sq "$line")"
}

# vr_env: load .env once per shell (GCP_PROJECT for IAP SSH and gcloud, UNDERLAY_MTU, the Lambda SSH
# key path); a missing or non-private .env ends the check (exit 2, a FAIL). The Lambda API key is
# dropped again: no check needs it, and scripts/gpuctl loads .env itself.
vr_env() {
  [[ "${_VR_ENV_LOADED:-}" == 1 ]] && return 0
  load_env
  unset LAMBDA_API_KEY
  _VR_ENV_LOADED=1
  _vr_project_mask # GCP_PROJECT is known now: mask it (also when it differs from what vr_init read)
}

# _vr_sleep SECONDS: sleep; VR_NO_WAIT=1 (the offline tests) skips it and says so in the evidence.
_vr_sleep() {
  if [[ "${VR_NO_WAIT:-}" == 1 ]]; then
    echo "(VR_NO_WAIT: skipped sleep $1)" >&2
  else
    sleep "$1"
  fi
}

# vr_retry TIMEOUT_S INTERVAL_S CMD…: run CMD (in this shell) until it succeeds, every INTERVAL_S
# seconds, for at most TIMEOUT_S seconds: 1 + TIMEOUT_S/INTERVAL_S attempts, fewer if the attempts
# themselves take long. Returns 1 when it gives up.
vr_retry() {
  local timeout="$1" interval="$2" waited=0 start="$SECONDS"
  shift 2
  while :; do
    "$@" && return 0
    if ((waited + interval > timeout || SECONDS - start + interval > timeout)); then
      printf 'gave up after %ss: %s\n' "$timeout" "$*" >&2
      return 1
    fi
    _vr_sleep "$interval"
    waited=$((waited + interval))
  done
}

# _vr_wait_text FILE TEXT PID SECONDS: wait until FILE (a vr_bg command's output) holds TEXT, i.e. the
# command is ready. Fails when PID has exited without printing it, or after SECONDS.
_vr_wait_text() {
  local i
  for ((i = 0; i < $4 * 5; i++)); do
    grep -qF -- "$2" "$1" 2>/dev/null && return 0
    if ! kill -0 "$3" 2>/dev/null; then
      grep -qF -- "$2" "$1" 2>/dev/null
      return
    fi
    sleep 0.2
  done
  return 1
}

# _vr_stop PID: stop a vr_bg command. A shell function (kc, cp_ssh) runs in a background subshell
# whose child is the real process (kubectl, gcloud), so the children go first; then the job.
_vr_stop() {
  pkill -TERM -P "$1" 2>/dev/null || true
  kill -TERM "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
}

# _vr_show FILE TITLE: print a vr_bg command's output into the evidence, then remove FILE.
_vr_show() {
  printf -- '--- %s ---\n' "$2"
  cat -- "$1"
  rm -f -- "$1"
}

# _vr_load_nodes: VR_NODES = the cluster's node names (kubectl's order), each checked.
_vr_load_nodes() {
  local out n
  VR_NODES=()
  out="$(vr_cap kc get nodes "$_VR_RT" -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}')" || return 1
  while read -r n; do
    [[ -n "$n" ]] || continue
    [[ "$n" =~ $K8S_NAME_RE ]] || { echo "unexpected node name from the cluster: '$n'"; return 1; }
    VR_NODES+=("$n")
  done <<< "$out"
  ((${#VR_NODES[@]})) || { echo "the cluster lists no nodes"; return 1; }
}

# _vr_load_pods NAMESPACE APP: VR_PODS = "NODE POD IP" for each Running Pod labelled app=APP.
_vr_load_pods() {
  local out node name ip
  VR_PODS=()
  out="$(vr_cap kc -n "$1" get pods -l "app=$2" --field-selector=status.phase=Running "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.spec.nodeName}{" "}{.metadata.name}{" "}{.status.podIP}{"\n"}{end}')" || return 1
  while read -r node name ip; do
    [[ -n "$node" ]] || continue
    if [[ ! "$node" =~ $K8S_NAME_RE || ! "$name" =~ $K8S_NAME_RE ]] || ! is_ipv4 "$ip"; then
      echo "unexpected $2 Pod from the cluster: '$node $name $ip'"
      return 1
    fi
    VR_PODS+=("$node $name $ip")
  done <<< "$out"
}

# _vr_pick NODE LINE…: "POD IP" of the first "NODE POD IP" line on NODE; fails if there is none.
_vr_pick() {
  local want="$1" line node name ip
  shift
  for line in "$@"; do
    read -r node name ip <<< "$line"
    if [[ "$node" == "$want" ]]; then
      printf '%s %s\n' "$name" "$ip"
      return 0
    fi
  done
  return 1
}

# _vr_in WORD LIST…: WORD is one of LIST.
_vr_in() {
  local want="$1" w
  shift
  for w in "$@"; do
    [[ "$w" == "$want" ]] && return 0
  done
  return 1
}

# _vr_ssh_prepare NODE…: look up the SSH zones once, in the check's own shell: cp_resolve and
# _worker_zone cache there only, so node_ssh in $(…) would otherwise read Terraform outputs each time.
# Both exit the shell (die) when an output cannot be read, so a check that must clean up runs the
# work that calls this in a subshell.
_vr_ssh_prepare() {
  local n
  for n in "$@"; do
    if [[ "$n" == "$CP_NAME" ]]; then
      vr_cmd cp_resolve
    elif [[ "$n" =~ $CPU_NODE_RE ]]; then
      vr_cmd _worker_zone "$n"
    fi
  done
}

# _vr_container_id ID: a containerd container ID without its scheme (containerd://<64 hex>).
_vr_container_id() {
  local cid="${1#*://}"
  [[ "$cid" =~ ^[0-9a-f]{64}$ ]] || { echo "not a container ID: '$1'" >&2; return 1; }
  printf '%s\n' "$cid"
}

# _vr_cgroup_dir NODE CID: the cgroup directory of container CID on NODE (spec §9 C5).
_vr_cgroup_dir() {
  local out dir
  out="$(vr_cap node_ssh "$1" "sudo find /sys/fs/cgroup -type d -name \"*$2*\"" </dev/null)" || return 1
  dir="${out%%$'\n'*}"
  [[ "$dir" =~ ^/sys/fs/cgroup/[A-Za-z0-9_.:@/-]+$ ]] \
    || { echo "no cgroup directory for container $2 on $1" >&2; return 1; }
  printf '%s\n' "$dir"
}

# --- A: the cluster (spec §9 A) -----------------------------------------------------------------

# A1: every node Ready, and at least 2 of them.
check_a1_nodes() {
  local out name status total=0 ready=0
  vr_cmd kc get nodes -o wide "$_VR_RT" || return 1
  out="$(vr_cap kc get nodes "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{" "}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}')" \
    || return 1
  while read -r name status; do
    [[ -n "$name" ]] || continue
    total=$((total + 1))
    if [[ "$status" == True ]]; then
      ready=$((ready + 1))
    else
      echo "A1: node $name is not Ready (Ready=${status:-unknown})"
    fi
  done <<< "$out"
  echo "A1: $ready of $total nodes Ready (want all of them, and at least 2)"
  ((total >= 2 && ready == total))
}

# A2: kubectl describe node succeeds for every node.
check_a2_describe_nodes() {
  local n fail=0
  _vr_load_nodes || return 1
  for n in "${VR_NODES[@]}"; do
    vr_cmd kc describe node "$n" "$_VR_RT" || { echo "A2: describe node $n failed"; fail=1; }
  done
  return "$fail"
}

# A3: metrics-server and CoreDNS run on k3s-cp and are Ready.
check_a3_system_pods() {
  local out app name node ready ms=0 dns=0 fail=0
  vr_cmd kc -n kube-system get pods -o wide "$_VR_RT" || return 1
  out="$(vr_cap kc -n kube-system get pods -l 'k8s-app in (metrics-server,kube-dns)' "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.labels.k8s-app}{" "}{.metadata.name}{" "}{.spec.nodeName}{" "}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}')" \
    || return 1
  while read -r app name node ready; do
    case "$app" in
      metrics-server) ms=$((ms + 1)) ;;
      kube-dns) dns=$((dns + 1)) ;;
      *) continue ;;
    esac
    if [[ "$node" != "$CP_NAME" || "$ready" != True ]]; then
      echo "A3: $name ($app) is on '$node', Ready=${ready:-unknown}; want $CP_NAME and True"
      fail=1
    fi
  done <<< "$out"
  ((ms > 0)) || { echo "A3: no metrics-server Pod"; fail=1; }
  ((dns > 0)) || { echo "A3: no CoreDNS Pod"; fail=1; }
  ((fail)) || echo "A3: $ms metrics-server and $dns CoreDNS Pod(s), all Ready on $CP_NAME"
  return "$fail"
}

# A4: every node's flannel.alpha.coreos.com/* annotations; the public IP flannel uses
# (public-ip-overwrite when it is set, else public-ip) equals the node's ExternalIP.
check_a4_flannel_annotations() {
  local json report
  json="$(vr_cap --no-echo kc get nodes "$_VR_RT" -o json)" || return 1
  report="$(jq -r '
    .items[] | (.metadata.annotations // {}) as $a
    | ([.status.addresses[]? | select(.type == "ExternalIP") | .address] | first // "") as $ext
    | ($a["flannel.alpha.coreos.com/public-ip-overwrite"] // $a["flannel.alpha.coreos.com/public-ip"] // "") as $pub
    | "node \(.metadata.name) ExternalIP=\($ext)",
      ($a | to_entries[] | select(.key | startswith("flannel.alpha.coreos.com/")) | "  \(.key)=\(.value)"),
      (if $ext != "" and $pub == $ext then "  OK: public IP \($pub) = ExternalIP"
       else "  MISMATCH: public IP '\''\($pub)'\'' != ExternalIP '\''\($ext)'\''" end)' <<< "$json")" \
    || { echo "A4: cannot read the nodes"; return 1; }
  printf '%s\n' "$report"
  [[ "$report" == *"node "* && "$report" != *MISMATCH* ]]
}

# --- B: connectivity (spec §9 B) ----------------------------------------------------------------

# _vr_b1_cell PROBE ECHO_IP SMALL BIG: one matrix cell (spec §2.3 #3: a Pod-to-Pod ICMP payload of
# UNDERLAY_MTU-108 fits, UNDERLAY_MTU-107 does not). Sets VR_CELL to ok or the failed parts. The
# oversized ping counts as blocked only when it fails as an MTU rejection (EMSGSIZE "message too
# long", "frag needed", or every packet lost); a failed exec (kubectl, the bound) is ping-<size>-error.
_vr_b1_cell() {
  local outf rc=0
  VR_CELL=""
  vr_cmd kc_within 60 -n nettest exec "$1" -- curl -fsS -m 5 "http://$2:8080/" || VR_CELL+="curl,"
  vr_cmd kc_within 60 -n nettest exec "$1" -- ping -M 'do' -c 2 -W 2 -s "$3" "$2" || VR_CELL+="ping-$3,"
  outf="$(mktemp)" || return 1
  vr_cmd kc_within 60 -n nettest exec "$1" -- ping -M 'do' -c 2 -W 2 -s "$4" "$2" >"$outf" 2>&1 || rc=$?
  cat -- "$outf"
  if ((rc == 0)); then
    VR_CELL+="ping-$4-passed,"
  elif ! grep -qiE 'message too long|frag(mentation)? needed| 100% packet loss' "$outf"; then
    VR_CELL+="ping-$4-error,"
  fi
  rm -f -- "$outf"
  VR_CELL="${VR_CELL%,}"
  VR_CELL="${VR_CELL:-ok}"
}

# B1: every probe -> every echo Pod IP (curl, ping that fits, ping that must not), and every probe ->
# the echo Service by DNS; printed as an N x N table (rows: probes, columns: echo Pods by node).
check_b1_matrix() {
  local mtu small big pn en p e name ip row fail=0
  local -a probes=() echoes=() rows=()
  vr_env
  mtu="${UNDERLAY_MTU:-1460}"
  [[ "$mtu" =~ ^[1-9][0-9]{2,3}$ ]] || { echo "B1: UNDERLAY_MTU in .env is not an MTU: '$mtu'"; return 1; }
  small=$((mtu - 108))
  big=$((mtu - 107))
  _vr_load_nodes || return 1
  _vr_load_pods nettest probe || return 1
  probes=(${VR_PODS[@]+"${VR_PODS[@]}"})
  _vr_load_pods nettest echo || return 1
  echoes=(${VR_PODS[@]+"${VR_PODS[@]}"})
  for pn in "${VR_NODES[@]}"; do
    _vr_pick "$pn" ${probes[@]+"${probes[@]}"} >/dev/null || { echo "B1: $pn: no Running probe Pod"; fail=1; }
    _vr_pick "$pn" ${echoes[@]+"${echoes[@]}"} >/dev/null || { echo "B1: $pn: no Running echo Pod"; fail=1; }
  done
  # Rows are tab-separated here and aligned by _vr_table at the end.
  row='probe \ echo'
  for en in "${VR_NODES[@]}"; do row+=$'\t'"$en"; done
  rows+=("$row"$'\tsvc-dns')
  for pn in "${VR_NODES[@]}"; do
    row="$pn"
    if ! p="$(_vr_pick "$pn" ${probes[@]+"${probes[@]}"})"; then
      for en in "${VR_NODES[@]}" svc-dns; do row+=$'\tmissing'; done
      rows+=("$row")
      continue
    fi
    read -r name ip <<< "$p"
    for en in "${VR_NODES[@]}"; do
      if e="$(_vr_pick "$en" ${echoes[@]+"${echoes[@]}"})"; then
        _vr_b1_cell "$name" "${e#* }" "$small" "$big"
      else
        VR_CELL=missing
      fi
      [[ "$VR_CELL" == ok ]] || fail=1
      row+=$'\t'"$VR_CELL"
    done
    if vr_cmd kc_within 60 -n nettest exec "$name" -- curl -fsS -m 5 http://echo.nettest.svc.cluster.local:8080/; then
      row+=$'\tok'
    else
      row+=$'\tFAIL'
      fail=1
    fi
    rows+=("$row")
  done
  echo
  echo "B1: N x N matrix, UNDERLAY_MTU=$mtu: each cell is curl to the echo Pod IP, ping -M do -s $small"
  echo "(must pass) and -s $big (must fail as too big for the MTU); svc-dns is curl"
  echo "http://echo.nettest.svc.cluster.local:8080/"
  printf '%s\n' "${rows[@]}" | _vr_table
  return "$fail"
}

# _vr_b2_flows MIN PEER_IP…: read `tcpdump -nn -q -t` lines on stdin and summarise them: the
# udp/51820 packets (any host, and with each peer), and the traffic with the other nodes' public IPs
# by service port. Only 51820/udp (WireGuard), 6443/tcp (the API, mTLS) and 10250/tcp (kubelet, TLS)
# may appear there (spec §2.2: the two TLS ports cross the internet by design); ICMP carries no port
# and is listed. Fails on any other port or protocol, when no udp/51820 packet was seen, or when a
# peer has fewer than MIN udp/51820 packets: each peer sends the marked requests, so its WireGuard
# traffic proves they crossed the wire encrypted while the capture ran (keepalives alone give 1-2
# packets in 25 s), and that the peer's wire address is the one ③ examines.
_vr_b2_flows() {
  local min="$1"
  shift
  awk -v min="$min" -v peers="$*" '
    BEGIN { n = split(peers, p, " "); for (i = 1; i <= n; i++) peer[p[i]] = 1 }
    function host(a,  f) { split(a, f, "."); return f[1] "." f[2] "." f[3] "." f[4] }
    function port(a,  f) { return split(a, f, ".") == 5 ? f[5] : "" }
    $1 == "IP" {
      dst = $4; sub(/:$/, "", dst)
      proto = ($5 == "tcp") ? "tcp" : ($5 ~ /^UDP/) ? "udp" : ($5 == "ICMP") ? "icmp" : "other"
      sh = host($2); dh = host(dst); sp = port($2); dp = port(dst)
      if (proto == "udp" && (sp == 51820 || dp == 51820)) {
        wg++
        if (sh in peer) pwg[sh]++
        if (dh in peer && dh != sh) pwg[dh]++
      }
      if (!(sh in peer) && !(dh in peer)) next
      if (proto == "udp" && (sp == 51820 || dp == 51820)) key = "51820/udp"
      else if (proto == "tcp" && (dp == 6443 || dp == 10250)) key = dp "/tcp"
      else if (proto == "tcp" && (sp == 6443 || sp == 10250)) key = sp "/tcp"
      else if (proto == "icmp") key = "icmp"
      else { bad[proto " " sp " > " dp]++; nbad++; next }
      seen[key]++
    }
    END {
      printf "B2 ②: %d udp/51820 packets in the capture\n", wg
      if (wg < 1) print "B2 ②: no udp/51820 packet: no WireGuard traffic was captured"
      for (i = 1; i <= n; i++) {
        c = pwg[p[i]] + 0
        printf "B2 ②: %d udp/51820 packets with %s (want at least %d: the 20 marked requests and their answers)\n", c, p[i], min
        if (c < min) short = 1
      }
      printf "B2 ③: packets to or from the other nodes'\'' public IPs (%s):\n", peers
      split("51820/udp 6443/tcp 10250/tcp icmp", k, " ")
      for (i = 1; i <= 4; i++) if (seen[k[i]]) printf "  %-10s %d\n", k[i], seen[k[i]]
      for (b in bad) printf "  UNEXPECTED %s: %d packets\n", b, bad[b]
      if (nbad == 0) print "B2 ③: only 51820/udp, 6443/tcp and 10250/tcp"
      exit (wg >= 1 && !short && nbad == 0) ? 0 : 1
    }'
}

# B2's cleanup on interrupt, on k3s-cp: stop the capture, wait until it is gone, then remove its file.
# A Ctrl-C while the check still waits for tcpdump to start could otherwise remove the file before
# tcpdump creates it. The pattern matches the capture's command line ([^ ]+: the interface) but not
# this command's own, nor pkill's or pgrep's ("[.]" in their text is not the "." it matches).
# shellcheck disable=SC2016 # expanded on k3s-cp
_VR_B2_CLEANUP='p='\''tcpdump -ni [^ ]+ -U -w /tmp/b2[.]pcap'\''; sudo pkill -f "$p"; i=0; while sudo pgrep -f "$p" >/dev/null && [ $i -lt 30 ]; do sleep 1; i=$((i + 1)); done; sudo rm -f /tmp/b2.pcap'

# B2: capture on k3s-cp's default interface while the probes on the other nodes send 20 requests
# each, carrying a random marker header, to the echo Pod on k3s-cp. PASS: ① the marker is not in
# the capture's payloads; ② there is udp/51820, and each sending node has at least 40 such packets
# (its marked requests crossed while the capture ran, which must still be running after them);
# ③ only the expected ports cross to the other nodes' public IPs. The capture file is created
# under umask 077 (sudo keeps the stricter umask: never world-readable) and removed afterwards,
# pass, fail or interrupt.
check_b2_wire() {
  local marker route iface n ip probe cp_echo capf pid started=0 rc=0 out matches fail=0
  local -a peers=() probes=()
  vr_env
  vr_cmd cp_resolve
  marker="b2-marker-$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')"
  [[ "$marker" =~ ^b2-marker-[0-9a-f]{16}$ ]] || { echo "B2: cannot make a random marker"; return 1; }
  echo "B2: marker $marker"
  route="$(vr_cap cp_ssh 'ip -o route show default' </dev/null)" || return 1
  iface="$(awk '{ for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit } }' <<< "$route")"
  [[ "$iface" =~ ^[A-Za-z0-9_.-]{1,15}$ ]] || { echo "B2: no default interface on $CP_NAME in: $route"; return 1; }
  _vr_load_nodes || return 1
  for n in "${VR_NODES[@]}"; do
    [[ "$n" != "$CP_NAME" ]] || continue
    ip="$(vr_cap node_external_ip "$n")" || return 1
    probe="$(vr_cap probe_on "$n")" || return 1
    peers+=("$ip")
    probes+=("$probe")
  done
  ((${#probes[@]})) || { echo "B2: no node other than $CP_NAME to send from"; return 1; }
  cp_echo="$(vr_cap echo_on "$CP_NAME")" || return 1

  capf="$(mktemp)" || return 1
  _vr_on_signal --left "/tmp/b2.pcap (the packet capture) on $CP_NAME" cp_ssh "$_VR_B2_CLEANUP"
  vr_bg "$capf" cp_ssh "umask 077 && sudo timeout 25 tcpdump -ni $iface -U -w /tmp/b2.pcap 'not port 22'"
  pid="$VR_BG_PID"
  if _vr_wait_text "$capf" "listening on" "$pid" 60; then
    started=1
    for probe in "${probes[@]}"; do
      vr_cmd kc_within 90 -n nettest exec "$probe" -- sh -c "n=0; for i in \$(seq 20); do curl -fsS -m 2 -o /dev/null -H \"X-Marker: $marker\" http://$cp_echo:8080/ && n=\$((n + 1)); done; echo \"\$n/20 requests answered\"; [ \"\$n\" -eq 20 ]" \
        || { echo "B2: not all 20 requests from $probe were answered"; fail=1; }
    done
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "B2: the capture on $CP_NAME ended before the marked requests finished, so it may not hold them"
      fail=1
    fi
    wait "$pid" || rc=$?
  else
    _vr_stop "$pid"
    echo "B2: the capture on $CP_NAME did not start"
    fail=1
    # Its start may only have been unseen (a slow IAP relay): such a capture writes for up to 25 s.
    echo "B2: waiting 30 s for a capture that may have started late, then removing its file"
    _vr_sleep 30
  fi
  _vr_show "$capf" "capture on $CP_NAME"
  if ((started)); then
    if ((rc != 0 && rc != 124)); then
      echo "B2: the capture on $CP_NAME failed (exit $rc)"
      fail=1
    fi
    rc=0
    # pipefail with grep's "no match" (exit 1) masked: a non-zero status is tcpdump failing to read.
    out="$(vr_cap cp_ssh "bash -o pipefail -c 'sudo tcpdump -A -nn -r /tmp/b2.pcap | { grep -c -F -- $marker || true; }'" </dev/null)" || rc=$?
    matches="${out##*$'\n'}"
    if ((rc == 0)) && [[ "$matches" == 0 ]]; then
      echo "B2 ①: the marker is not in the capture (the cross-node traffic is encrypted)"
    elif ((rc == 0)) && [[ "$matches" =~ ^[1-9][0-9]*$ ]]; then
      echo "B2 ①: MARKER found in the capture $matches times: Pod traffic crossed the wire in plaintext"
      fail=1
    else
      echo "B2 ①: cannot search the capture (exit $rc)"
      fail=1
    fi
    if out="$(vr_cap --no-echo cp_ssh 'sudo tcpdump -nn -q -t -r /tmp/b2.pcap' </dev/null)"; then
      _vr_b2_flows 40 "${peers[@]}" <<< "$out" || fail=1
    else
      echo "B2: cannot read the capture"
      fail=1
    fi
  fi
  vr_cmd cp_ssh 'sudo rm -f /tmp/b2.pcap' </dev/null || { echo "B2: cannot remove /tmp/b2.pcap on $CP_NAME"; fail=1; }
  _vr_on_signal
  return "$fail"
}

# _vr_wg_peers NODE WANT: read `wg show flannel-wg` on stdin; NODE must have WANT peers, each with a
# latest handshake at most 3 minutes old. The keys are not printed again (wg show's own output is).
_vr_wg_peers() {
  awk -v node="$1" -v want="$2" '
    function secs(s,  f, n, i, t, u) {
      if (s ~ /^Now/) return 0
      n = split(s, f, /[ ,]+/)
      t = 0
      for (i = 1; i < n; i++) {
        if (f[i] !~ /^[0-9]+$/) continue
        u = f[i + 1]
        if (u ~ /^year/) t += f[i] * 31536000
        else if (u ~ /^day/) t += f[i] * 86400
        else if (u ~ /^hour/) t += f[i] * 3600
        else if (u ~ /^minute/) t += f[i] * 60
        else if (u ~ /^second/) t += f[i]
        else return -1
      }
      return t
    }
    /^peer:/ { p++; hs[p] = -1; ep[p] = "?" }
    p && /^ *endpoint:/ { ep[p] = $2 }
    p && /^ *latest handshake:/ { s = $0; sub(/^ *latest handshake: */, "", s); hs[p] = secs(s); txt[p] = s }
    END {
      bad = (p != want)
      printf "B3: %s: %d peers (want %d)\n", node, p, want
      for (i = 1; i <= p; i++) {
        if (hs[i] < 0) { printf "  peer %s: no handshake\n", ep[i]; bad = 1 }
        else if (hs[i] > 180) { printf "  peer %s: latest handshake %s: older than 3 minutes\n", ep[i], txt[i]; bad = 1 }
        else printf "  peer %s: latest handshake %d s ago\n", ep[i], hs[i]
      }
      exit bad
    }'
}

# B3: `wg show flannel-wg` on every node (never `dump`, which prints the private key): node count - 1
# peers, each with a handshake in the last 3 minutes.
check_b3_wg() {
  local n out want fail=0
  vr_env
  _vr_load_nodes || return 1
  want=$((${#VR_NODES[@]} - 1))
  _vr_ssh_prepare "${VR_NODES[@]}"
  for n in "${VR_NODES[@]}"; do
    if ! out="$(vr_cap node_ssh "$n" 'sudo wg show flannel-wg' </dev/null)"; then
      echo "B3: $n: wg show failed"
      fail=1
      continue
    fi
    _vr_wg_peers "$n" "$want" <<< "$out" || fail=1
  done
  return "$fail"
}

# Stops the iperf3 servers in a probe (run with sh -c). pkill exits 1 when no process matched: no server
# is left, the outcome wanted (an interrupt before the server started or after it ended is no failed
# cleanup); pkill's own errors (2: syntax, 3: fatal) still fail with their own code, so they do not
# read as that harmless 1, and so does a failed kubectl exec.
# shellcheck disable=SC2016 # expanded by the probe's sh, not here
_VR_B4_STOP='pkill -x iperf3; rc=$?; [ $rc -le 1 ] || exit $rc'

# _vr_b4_pair CLIENT_PROBE CLIENT_NODE SERVER_NODE: iperf3 from CLIENT_PROBE to the probe on
# SERVER_NODE (-n 50M: a bounded transfer, spec §9 B4), and record bits_per_second. The server runs
# with --forceflush: through kubectl exec without a tty its stdout is a pipe, which iperf3 otherwise
# buffers, so "Server listening" would only arrive when it exits. When the measurement fails, the
# server is stopped in its probe too (stopping kubectl leaves it listening on 5201 for up to 150 s);
# so it is on interrupt, and a server left by an earlier run is stopped before this one starts.
_vr_b4_pair() {
  local sprobe sip logf pid json rc=0 sent recv mbit
  sprobe="$(vr_cap probe_on "$3")" || return 1
  sip="$(vr_cap kc -n nettest get pod "$sprobe" "$_VR_RT" -o 'jsonpath={.status.podIP}')" || return 1
  is_ipv4 "$sip" || { echo "B4: $sprobe has no Pod IP: '$sip'"; return 1; }
  logf="$(mktemp)" || return 1
  # A server left by an interrupted run would hold port 5201: stop it.
  vr_cmd kc_within 60 -n nettest exec "$sprobe" -- sh -c "$_VR_B4_STOP" \
    || echo "B4: could not stop an iperf3 server left in $sprobe (going on)"
  _vr_on_signal --left "an iperf3 server on port 5201 in $sprobe (it ends by itself within 150 s)" \
    kc_within 60 -n nettest exec "$sprobe" -- sh -c "$_VR_B4_STOP"
  vr_bg "$logf" kc_within 170 -n nettest exec "$sprobe" -- timeout 150 iperf3 -s -1 --forceflush
  pid="$VR_BG_PID"
  if _vr_wait_text "$logf" "Server listening" "$pid" 60; then
    json="$(vr_cap --no-echo kc_within 140 -n nettest exec "$1" -- timeout 120 iperf3 -c "$sip" -n 50M -J)" \
      || rc=$?
  else
    echo "B4: the iperf3 server on $3 did not start"
    rc=1
  fi
  if ((rc == 0)); then
    wait "$pid" || true
  else
    _vr_stop "$pid"
    vr_cmd kc_within 60 -n nettest exec "$sprobe" -- sh -c "$_VR_B4_STOP" || true
  fi
  _vr_on_signal
  _vr_show "$logf" "iperf3 server on $3 ($sprobe)"
  if ((rc != 0)); then
    echo "B4: iperf3 $2 -> $3 failed (exit $rc)"
    [[ -z "${json:-}" ]] || printf '%s\n' "$json"
    return 1
  fi
  sent="$(jq -r '.end.sum_sent.bits_per_second | floor' <<< "$json" 2>/dev/null)" || sent=""
  recv="$(jq -r '.end.sum_received.bits_per_second | floor' <<< "$json" 2>/dev/null)" || recv=""
  [[ "$sent" =~ ^[0-9]+$ && "$recv" =~ ^[0-9]+$ ]] || { echo "B4: no bits_per_second in iperf3's JSON"; return 1; }
  mbit=$((recv / 1000000))
  echo "B4: $2 -> $3: sum_sent bits_per_second=$sent, sum_received bits_per_second=$recv (about $mbit Mbit/s)"
}

# B4: iperf3 from the probe on k3s-cpu-1 to the probe on k3s-cp, and to each GPU node's probe.
check_b4_iperf() {
  local client=k3s-cpu-1 cprobe n fail=0
  local -a servers=("$CP_NAME")
  _vr_load_nodes || return 1
  for n in "${VR_NODES[@]}"; do
    [[ ! "$n" =~ $GPU_NODE_RE ]] || servers+=("$n")
  done
  cprobe="$(vr_cap probe_on "$client")" || return 1
  for n in "${servers[@]}"; do
    _vr_b4_pair "$cprobe" "$client" "$n" || fail=1
  done
  return "$fail"
}

# B5: what must fail does. ① this machine cannot open tcp 6443 or 10250 on k3s-cp's public IP;
# ② a Pod on a GCP node cannot reach the metadata server (curl itself must fail: exit 28 timeout or
# 7 refused; any other code means kubectl or the probe failed, or the exec's bound ran out (124), so
# nothing was proven); ③ DNS works
# from every probe; ④ the k3s_guard table (with its counters) on every node.
check_b5_negative() {
  local cp_ip port n p name ip rc fail=0
  vr_env
  cp_ip="$(vr_cap node_external_ip "$CP_NAME")" || return 1
  for port in 6443 10250; do
    if vr_cmd timeout 3 bash -c "</dev/tcp/$cp_ip/$port"; then
      echo "B5 ①: tcp/$port on $CP_NAME ($cp_ip) is reachable from this machine"
      fail=1
    else
      echo "B5 ①: tcp/$port on $CP_NAME is not reachable from this machine (expected)"
    fi
  done
  _vr_load_nodes || return 1
  _vr_load_pods nettest probe || return 1
  for n in "${VR_NODES[@]}"; do
    if ! p="$(_vr_pick "$n" ${VR_PODS[@]+"${VR_PODS[@]}"})"; then
      echo "B5: $n: no Running probe Pod"
      fail=1
      continue
    fi
    read -r name ip <<< "$p"
    if [[ "$n" == "$CP_NAME" || "$n" =~ $CPU_NODE_RE ]]; then
      rc=0
      vr_cmd kc_within 60 -n nettest exec "$name" -- curl -m 3 -H 'Metadata-Flavor: Google' \
        http://169.254.169.254/computeMetadata/v1/ || rc=$?
      case "$rc" in
        7 | 28) echo "B5 ②: $n: a Pod cannot reach the metadata server (curl exit $rc, expected)" ;;
        0) echo "B5 ②: $n: a Pod reached the metadata server"; fail=1 ;;
        *) echo "B5 ②: $n: exit $rc: kubectl or curl failed before a verdict (want curl's 28 or 7)"; fail=1 ;;
      esac
    fi
    vr_cmd kc_within 60 -n nettest exec "$name" -- nslookup kubernetes.default.svc.cluster.local \
      || { echo "B5 ③: $n: the DNS lookup failed"; fail=1; }
  done
  _vr_ssh_prepare "${VR_NODES[@]}"
  for n in "${VR_NODES[@]}"; do
    vr_cmd node_ssh "$n" 'sudo nft list table inet k3s_guard' </dev/null \
      || { echo "B5 ④: $n: cannot list the k3s_guard table"; fail=1; }
  done
  return "$fail"
}

# _vr_table: tab-separated lines on stdin as aligned columns.
_vr_table() {
  awk -F'\t' '
    { for (i = 1; i <= NF; i++) { c[NR, i] = $i; if (length($i) > w[i]) w[i] = length($i) }
      if (NF > m) m = NF }
    END { for (r = 1; r <= NR; r++) { line = ""
            for (i = 1; i <= m; i++) line = line sprintf("%-" (w[i] + 2) "s", c[r, i])
            sub(/ +$/, "", line); print line } }'
}

# A line of `gpuctl rules` for Lambda's rule that admits the operator (lambda/gpuctl/peers.py).
_VR_OPERATOR_RULE_RE='^(  [a-z]+ +[^ ]+ +)[0-9.]+(/32 +operator ssh)$'

# _vr_mask_ids: stdin -> stdout with the GCP project id replaced by <GCP_PROJECT> and the operator's
# IP (vr_operator_ip) by <operator_ip>: B6's evidence holds both, and a run promoted to
# verification/final is committed (Ruling R39 keeps real project ids and IPs out of what is
# delivered). redact masks both in all evidence too, with the same expressions (_vr_project_mask
# also finds the id vr_env read); this covers a check called without vr_init or run_check. The
# source of Lambda's "operator ssh" rule is masked even when the operator's IP is unknown.
_vr_mask_ids() {
  _vr_project_mask
  LC_ALL=C sed -E -e '' ${_VR_PROJECT_EXPRS[@]+"${_VR_PROJECT_EXPRS[@]}"} \
    -e "s#${_VR_OPERATOR_RULE_RE}#\\1<operator_ip>\\2#" ${_VR_MASK_EXPRS[@]+"${_VR_MASK_EXPRS[@]}"}
}

# B6: the GCP firewall rules named fw-* as a table, and gpuctl's Lambda rules. Every source must be
# one address (/32), except Google's IAP range 35.235.240.0/20. An ingress rule without source
# ranges admits every address on GCP, so it fails too, as does a tag or service-account source.
# The evidence shows the project id and the operator's IP masked (_vr_mask_ids). When the operator's
# IP is unknown, the rest of the run's evidence cannot mask it, so B6 fails if Lambda's
# "operator ssh" rule shows it.
check_b6_firewall_inventory() {
  local outf rc=0 known=1
  vr_env
  vr_operator_ip || known=0
  outf="$(mktemp)" || return 1
  _vr_b6 >"$outf" 2>&1 || rc=$?
  if ! _vr_mask_ids <"$outf"; then
    echo "B6: cannot mask the project id and the operator IP; the evidence is left out"
    rc=1
  fi
  if ((!known)) && grep -Eq -- "$_VR_OPERATOR_RULE_RE" "$outf"; then
    echo "B6: cannot determine the operator's public IP (OPERATOR_IP in .env is empty or not an address,"
    echo "and checkip.amazonaws.com gave none), so this run's evidence cannot mask it; the source of the"
    echo "'operator ssh' rule is masked above. Set OPERATOR_IP in .env to the address gpuctl used, run again."
    rc=1
  fi
  rm -f -- "$outf"
  return "$rc"
}

_vr_b6() {
  local json table violations rules count fail=0
  json="$(vr_cap --no-echo bounded 120 gcloud compute firewall-rules list --project "$GCP_PROJECT" \
    --filter='name~^fw-' --format=json)" || return 1
  table="$(jq -r '
    (["NAME", "NETWORK", "DIRECTION", "SOURCES", "ALLOWED", "TARGET_SA"] | @tsv),
    (.[] | [.name, ((.network // "-") | split("/") | last), (.direction // "INGRESS"),
            ((.sourceRanges // []) | if length == 0 then "-" else join(",") end),
            ((.allowed // []) | map(.IPProtocol + (if .ports then ":" + (.ports | join(";")) else "" end)) | join(",")),
            ((.targetServiceAccounts // []) | if length == 0 then "-" else join(",") end)] | @tsv)' <<< "$json")" \
    || { echo "B6: cannot read gcloud's answer"; return 1; }
  printf '%s\n' "$table" | _vr_table
  count="$(jq 'length' <<< "$json")"
  ((count > 0)) || { echo "B6: no firewall rule named fw-* in $GCP_PROJECT"; fail=1; }
  violations="$(jq -r '
    .[] | select((.direction // "INGRESS") == "INGRESS") | . as $r
    | (if ((.sourceRanges // []) | length) == 0
       then "VIOLATION \(.name): no source ranges (GCP would admit every address)" else empty end),
      (if (((.sourceTags // []) + (.sourceServiceAccounts // [])) | length) > 0
       then "VIOLATION \(.name): a source that is not an address range" else empty end),
      ((.sourceRanges // [])[] | select((endswith("/32") | not) and . != "35.235.240.0/20")
       | "VIOLATION \($r.name): source \(.) is not a single address (/32)")' <<< "$json")" \
    || { echo "B6: cannot read gcloud's answer"; return 1; }

  rules="$(vr_cap bounded 120 "$REPO_ROOT/scripts/gpuctl" rules </dev/null)" \
    || { echo "B6: scripts/gpuctl rules failed"; fail=1; }
  violations+=$'\n'"$(awk '
    /^[^ ]/ { section = $0; next }
    /^  / && $1 != "(no" && $3 !~ /\/32$/ && $3 != "35.235.240.0/20" {
      print "VIOLATION Lambda " section ": source " $3 " is not a single address (/32)" }' <<< "${rules:-}")"
  if [[ -n "${violations//$'\n'/}" ]]; then
    printf '%s\n' "$violations" | sed '/^$/d'
    fail=1
  elif ((count > 0)); then
    echo "B6: every source is a single address (/32) or the IAP range 35.235.240.0/20"
  fi
  return "$fail"
}

# --- C: hello-world (spec §9 C) -----------------------------------------------------------------

# C1: the hello Deployment is rolled out; its Pods.
check_c1_rollout() {
  vr_cmd kc -n hello rollout status deploy/hello --timeout=120s || return 1
  vr_cmd kc -n hello get pods -o wide "$_VR_RT"
}

# C2: from a probe on another node than the hello replicas (one on a node with no replica if there
# is one, else one on a node other than some replica), the Service answers exactly "hello world".
check_c2_cross_node_curl() {
  local out h line node name ip probe="" pnode="" body rc=0
  local -a hello=()
  out="$(vr_cap kc -n hello get pods -l app=hello --field-selector=status.phase=Running "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.spec.nodeName}{"\n"}{end}')" || return 1
  while read -r h; do
    [[ -z "$h" ]] || hello+=("$h")
  done <<< "$out"
  ((${#hello[@]})) || { echo "C2: no Running hello Pod"; return 1; }
  _vr_load_pods nettest probe || return 1
  for line in ${VR_PODS[@]+"${VR_PODS[@]}"}; do
    read -r node name ip <<< "$line"
    if ! _vr_in "$node" "${hello[@]}"; then probe="$name"; pnode="$node"; break; fi
  done
  if [[ -z "$probe" ]]; then
    for line in ${VR_PODS[@]+"${VR_PODS[@]}"}; do
      read -r node name ip <<< "$line"
      for h in "${hello[@]}"; do
        if [[ "$h" != "$node" ]]; then probe="$name"; pnode="$node"; break 2; fi
      done
    done
  fi
  [[ -n "$probe" ]] || { echo "C2: no probe runs on a node other than a hello replica's"; return 1; }
  echo "C2: hello replicas on: ${hello[*]}; asking from $probe on $pnode"
  body="$(vr_cap kc_within 60 -n nettest exec "$probe" -- curl -fsS -m 5 http://hello.hello.svc.cluster.local)" || rc=$?
  if ((rc == 0)) && [[ "$body" == "hello world" ]]; then
    echo "C2: answered exactly 'hello world'"
    return 0
  fi
  echo "C2: curl exit $rc; want exactly 'hello world'"
  return 1
}

# C3: kubectl port-forward svc/hello (in the background), curl it on localhost, stop the forward.
# VR_PF_PORT (default 18080) is the local port; the offline tests pick a free one. --noproxy '*':
# an operator's http_proxy, HTTPS_PROXY or ALL_PROXY would send the localhost request to the proxy.
check_c3_port_forward() {
  local port="${VR_PF_PORT:-18080}" logf pid body rc=0
  [[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "C3: VR_PF_PORT is not a port: '$port'"; return 1; }
  logf="$(mktemp)" || return 1
  vr_bg "$logf" kc -n hello port-forward svc/hello "$port:80"
  pid="$VR_BG_PID"
  if _vr_wait_text "$logf" "Forwarding from" "$pid" 30; then
    body="$(vr_cap curl --noproxy '*' -fsS -m 5 "localhost:$port")" || rc=$?
  else
    echo "C3: the port-forward did not start"
    rc=1
  fi
  _vr_stop "$pid"
  echo "C3: port-forward stopped"
  _vr_show "$logf" "port-forward"
  if ((rc == 0)) && [[ "$body" == "hello world" ]]; then
    echo "C3: localhost:$port answered exactly 'hello world'"
    return 0
  fi
  echo "C3: curl exit $rc; want exactly 'hello world'"
  return 1
}

# C4: every hello Pod's QoS class is Guaranteed.
check_c4_qos() {
  local out
  out="$(vr_cap kc -n hello get pods -l app=hello "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{" "}{.status.qosClass}{"\n"}{end}')" || return 1
  awk 'NF { n++; if ($2 != "Guaranteed") { print "C4: " $1 " is " ($2 == "" ? "unknown" : $2); bad = 1 } }
       END { if (!n) { print "C4: no hello Pod"; exit 1 }
             if (!bad) print "C4: all " n " hello Pods are Guaranteed"; exit bad }' <<< "$out"
}

# C5: one hello container's cgroup on its node: memory.max = 67108864 (64Mi), cpu.max = 10000 100000
# (100m).
check_c5_cgroup() {
  local out line pod node cid="" dir vals mem cpu fail=0
  vr_env
  out="$(vr_cap kc -n hello get pods -l app=hello --field-selector=status.phase=Running "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{" "}{.spec.nodeName}{" "}{.status.containerStatuses[0].containerID}{"\n"}{end}')" \
    || return 1
  while read -r pod node line; do
    if [[ -n "$line" ]]; then cid="$line"; break; fi
  done <<< "$out"
  [[ -n "$cid" ]] || { echo "C5: no hello container ID"; return 1; }
  cid="$(_vr_container_id "$cid")" || return 1
  [[ "$node" =~ $K8S_NAME_RE ]] || { echo "C5: unexpected node name: '$node'"; return 1; }
  echo "C5: $pod on $node, container $cid"
  _vr_ssh_prepare "$node"
  dir="$(_vr_cgroup_dir "$node" "$cid")" || return 1
  vals="$(vr_cap node_ssh "$node" "cat $dir/memory.max $dir/cpu.max" </dev/null)" || return 1
  mem="$(sed -n 1p <<< "$vals")"
  cpu="$(sed -n 2p <<< "$vals")"
  [[ "$mem" == 67108864 ]] || { echo "C5: memory.max is '$mem'; want 67108864"; fail=1; }
  [[ "$cpu" == "10000 100000" ]] || { echo "C5: cpu.max is '$cpu'; want '10000 100000'"; fail=1; }
  ((fail)) || echo "C5: memory.max 67108864, cpu.max 10000 100000"
  return "$fail"
}

_vr_c6_top() {
  local out
  out="$(vr_cap kc -n hello top pod "$_VR_RT")" || return 1
  [[ -n "$(awk 'NR > 1 && NF' <<< "$out")" ]]
}

# C6: kubectl top pod in hello returns a data row within 90 s (metrics-server needs a scrape first).
check_c6_top() {
  if vr_retry 90 10 _vr_c6_top; then
    echo "C6: kubectl top pod returned data"
    return 0
  fi
  echo "C6: no data from kubectl top pod within 90 s"
  return 1
}

# _vr_millicores CPU: a kubectl top CPU reading (100m, 1) in millicores.
_vr_millicores() {
  if [[ "$1" =~ ^([0-9]+)m$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "$(($1 * 1000))"
  else
    return 1
  fi
}

_vr_c7_oom_done() {
  VR_OOM="$(vr_cap kc -n limits-demo get pod oom "$_VR_RT" \
    -o 'jsonpath={.status.containerStatuses[0].state.terminated.reason}{" "}{.status.containerStatuses[0].state.terminated.exitCode}')" \
    || return 1
  [[ -n "${VR_OOM// /}" ]]
}

_vr_c7_top() {
  local out
  out="$(vr_cap kc -n limits-demo top pod cpu-burn "$_VR_RT")" || return 1
  VR_TOP_CPU="$(awk '$1 == "cpu-burn" { print $2 }' <<< "$out")"
  [[ -n "$VR_TOP_CPU" ]]
}

# _vr_throttled NODE DIR: nr_throttled from DIR/cpu.stat on NODE.
_vr_throttled() {
  local out v
  out="$(vr_cap node_ssh "$1" "cat $2/cpu.stat" </dev/null)" || return 1
  v="$(awk '$1 == "nr_throttled" { print $2 }' <<< "$out")"
  [[ "$v" =~ ^[0-9]+$ ]] || { echo "no nr_throttled in $2/cpu.stat on $1" >&2; return 1; }
  printf '%s\n' "$v"
}

_vr_c7() {
  local reason code milli line node cid dir t1 t2 fail=0
  vr_cmd kc apply -f "$REPO_ROOT/k8s/limits-demo/limits-demo.yaml" "$_VR_RT" || return 1
  if vr_retry 90 5 _vr_c7_oom_done; then
    read -r reason code <<< "$VR_OOM"
    if [[ "$reason" == OOMKilled && "$code" == 137 ]]; then
      echo "C7 ①: oom was OOMKilled, exit code 137"
    else
      echo "C7 ①: oom ended with reason '$reason', exit code '$code'; want OOMKilled and 137"
      fail=1
    fi
  else
    echo "C7 ①: oom did not terminate within 90 s"
    fail=1
  fi

  vr_cmd kc -n limits-demo wait pod/cpu-burn --for=condition=Ready --timeout=90s \
    || { echo "C7 ②: cpu-burn not Ready within 90 s"; return 1; }
  echo "C7 ②: cpu-burn Ready; reading kubectl top after 30 s"
  _vr_sleep 30
  if vr_retry 60 10 _vr_c7_top; then
    if milli="$(_vr_millicores "$VR_TOP_CPU")" && ((milli <= 110)); then
      echo "C7 ②: kubectl top reads $VR_TOP_CPU (at most 110m)"
    else
      echo "C7 ②: kubectl top reads '$VR_TOP_CPU'; want at most 110m"
      fail=1
    fi
  else
    echo "C7 ②: no kubectl top reading for cpu-burn"
    fail=1
  fi
  line="$(vr_cap kc -n limits-demo get pod cpu-burn "$_VR_RT" \
    -o 'jsonpath={.spec.nodeName}{" "}{.status.containerStatuses[0].containerID}')" || return 1
  read -r node cid <<< "$line"
  cid="$(_vr_container_id "${cid:-}")" || return 1
  [[ "$node" =~ $K8S_NAME_RE ]] || { echo "C7: unexpected node name: '$node'"; return 1; }
  _vr_ssh_prepare "$node"
  dir="$(_vr_cgroup_dir "$node" "$cid")" || return 1
  t1="$(_vr_throttled "$node" "$dir")" || return 1
  _vr_sleep 10
  t2="$(_vr_throttled "$node" "$dir")" || return 1
  if ((t2 > t1)); then
    echo "C7 ②: nr_throttled $t1 -> $t2 in 10 s: throttled at its CPU limit"
  else
    echo "C7 ②: nr_throttled $t1 -> $t2 in 10 s: not increasing"
    fail=1
  fi
  return "$fail"
}

# C7: k8s/limits-demo: ① oom is OOMKilled with exit code 137 within 90 s; ② 30 s after cpu-burn is
# Ready, kubectl top reads at most 110m and its cgroup's nr_throttled grows over 10 s. Both Pods are
# deleted before (left over from an earlier run) and after, pass, fail or interrupt. The work runs in
# a subshell, so a die in it (the SSH zone lookup) cannot skip the delete.
check_c7_limits_demo() {
  local rc=0
  vr_env
  vr_cmd kc -n limits-demo delete pod oom cpu-burn --ignore-not-found --timeout=90s \
    || echo "C7: could not delete Pods left from an earlier run (going on)"
  _vr_on_signal --left "the Pods oom and cpu-burn in namespace limits-demo" \
    kc -n limits-demo delete pod oom cpu-burn --ignore-not-found --wait=false "$_VR_RT"
  (_vr_c7) || rc=1
  vr_cmd kc -n limits-demo delete pod oom cpu-burn --ignore-not-found --timeout=90s \
    || { echo "C7: cannot delete the demo Pods"; rc=1; }
  _vr_on_signal
  return "$rc"
}

_vr_c8() {
  local rc=0 code
  vr_cmd kc apply -f "$REPO_ROOT/k8s/verify/np-denied.yaml" "$_VR_RT" || return 1
  vr_cmd kc -n np-denied wait pod/client --for=condition=Ready --timeout=120s || return 1
  # Positive control (Ruling R45): echo.nettest has no NetworkPolicy, so the same client must get
  # HTTP 200 there; a client without a working network or DNS must not count as "denied".
  code="$(vr_cap kc_within 60 -n np-denied exec client -- curl -sS -m 5 -o /dev/null -w '%{http_code}' \
    http://echo.nettest.svc.cluster.local:8080/)" || rc=$?
  if ((rc != 0)) || [[ "$code" != 200 ]]; then
    echo "C8: positive control failed: the np-denied client got curl exit $rc, HTTP '${code:-}' from echo.nettest;"
    echo "C8: want 200, else being unable to reach hello proves nothing about the NetworkPolicy"
    return 1
  fi
  echo "C8: positive control: the np-denied client gets HTTP 200 from echo.nettest"
  rc=0
  vr_cmd kc_within 60 -n np-denied exec client -- curl -sS -m 5 http://hello.hello.svc.cluster.local || rc=$?
  case "$rc" in
    7)
      echo "C8: from np-denied the connection to hello was refused (curl exit 7): the NetworkPolicy rejects it"
      return 0
      ;;
    28)
      echo "C8: from np-denied the request to hello timed out (curl exit 28): the NetworkPolicy drops it"
      return 0
      ;;
  esac
  echo "C8: curl exit $rc from np-denied; want 7 (refused) or 28 (timed out), the NetworkPolicy's verdict (R45)"
  return 1
}

# C8: the allowed path works (C2, from nettest); then a client in the np-denied namespace reaches
# echo.nettest (HTTP 200: the positive control) but not hello: curl exits 7 (kube-router rejects, as
# measured live) or 28 (a policy that drops), per Ruling R45. The namespace is deleted before (left
# over from an earlier run) and after, pass, fail or interrupt; the work runs in a subshell, as C7's.
check_c8_netpol() {
  local rc=0
  echo "C8: first the allowed path from nettest (C2)"
  if ! check_c2_cross_node_curl; then
    echo "C8: C2 failed, so a failure from np-denied would prove nothing"
    return 1
  fi
  vr_cmd kc delete namespace np-denied --ignore-not-found --timeout=120s \
    || { echo "C8: cannot delete np-denied left from an earlier run"; return 1; }
  _vr_on_signal --left "the namespace np-denied (its client Pod)" \
    kc delete namespace np-denied --ignore-not-found --wait=false "$_VR_RT"
  (_vr_c8) || rc=1
  vr_cmd kc delete namespace np-denied --ignore-not-found --timeout=120s \
    || { echo "C8: cannot delete the np-denied namespace"; rc=1; }
  _vr_on_signal
  return "$rc"
}

# --- D: GPU (spec §9 D) -------------------------------------------------------------------------

# Every D check SKIPs (77) when no node carries the admin-only accelerator label join-node.sh sets on a
# GPU node, so make verify passes without a GPU node (Review Focus #5). D1 and D2 check every GPU
# node; D6 holds every allocatable GPU. The Pods D4-D6 create are deleted before (left over from an
# earlier run) and after, pass or fail, and also when the check is interrupted (_vr_gpu_guarded): a
# leftover gpu-hold keeps a billed GPU allocated.

_VR_GPU_LABEL='node-restriction.kubernetes.io/accelerator=nvidia'
# join-node.sh's check that k3s's containerd has the nvidia runtime (spec §6.2 step 2).
_VR_CONTAINERD_NVIDIA='sudo grep -A3 nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml'
# A gpu-type label value as gpuctl's gpu_type_label() makes it (a6000, a10, a100_80gb_sxm4).
_VR_GPU_TYPE_RE='^[a-z0-9]+([_-][a-z0-9]+)*$'

# _vr_gpu_guarded WHAT FUNC CLEANUP DROP…: run FUNC in a subshell (a die in it ends only FUNC), then
# CLEANUP (the Pods' delete, waited for), however FUNC ended. FUNC runs on the left of ||, as C7's and
# C8's work does, so errexit is off in it: it returns its failures.
# Returns FUNC's status; 1 when FUNC passed but CLEANUP failed. An interrupt (Ctrl-C during make verify
# or the demo; SIGTERM) is handled as Task 18's checks handle it (_vr_on_signal): DROP (the same Pods'
# delete without the wait) runs, a further Ctrl-C cannot cut it short, and the check ends by the signal.
# WHAT names the Pods: if CLEANUP or DROP fails, the operator is told at once (_vr_tell: redacted, on
# run_check's fd 3 and in the evidence) that they may remain, holding billed GPUs, and how to run DROP
# by hand in a child shell (_vr_removal).
_vr_gpu_guarded() {
  local what="$1" func="$2" cleanup="$3" rc=0 crc=0
  shift 3
  _vr_on_signal --left "$what" "$@"
  ("$func") || rc=$?
  "$cleanup" || crc=$?
  if ((crc != 0)); then
    _vr_tell "cleanup FAILED (exit $crc); may remain: $what" \
      "to remove it, run from the repository root: $(_vr_removal "$@")"
    ((rc != 0)) || rc=1
  fi
  _vr_on_signal
  return "$rc"
}

# The bounds of the D checks' waits, in seconds: D4/D5's Pod (brief: 300 s), D2's resource (L21-2: 90 s), D6's
# look at gpu-overflow (brief: 30 s). Fixed; only the offline tests shrink them, to run a stalled tunnel on
# the real clock.
_VR_POD_WAIT_S=300
_VR_D2_WAIT_S=90
_VR_D6_WAIT_S=30

# The D waits count the time their kubectl requests take, not only their own sleeps: through a half-stalled
# API tunnel every request waits for its whole --request-timeout, which would stretch D4's 300 s to about an
# hour of a billed GPU session (review F2-1). So they keep a deadline on _vr_clock and cut each request's
# --request-timeout to the time left. Under VR_NO_WAIT=1 (the offline tests skip the sleeps) the clock is
# the sum of the skipped sleeps, so the tests see every poll; otherwise it is bash's SECONDS.
_VR_SLEPT=0

# _vr_clock: the clock's reading, into VR_CLOCK (set here, not printed: no subshell).
_vr_clock() {
  if [[ "${VR_NO_WAIT:-}" == 1 ]]; then
    VR_CLOCK=$_VR_SLEPT
  else
    VR_CLOCK=$SECONDS
  fi
}

# _vr_wait SECONDS: _vr_sleep, counted on the clock.
_vr_wait() {
  _vr_sleep "$1"
  _VR_SLEPT=$((_VR_SLEPT + $1))
}

# _vr_rt_until DEADLINE: VR_RT = the --request-timeout of the next kubectl request: KC_REQUEST_TIMEOUT cut
# to the whole seconds left before DEADLINE. Returns 1 when less than a second is left (kubectl reads
# --request-timeout=0 as no bound at all).
_vr_rt_until() {
  local left max="${KC_REQUEST_TIMEOUT%s}"
  _vr_clock
  left=$(($1 - VR_CLOCK))
  ((left >= 1)) || return 1
  ((left <= max)) || left=$max
  VR_RT="--request-timeout=${left}s"
}

# _vr_poll SECONDS INTERVAL CMD…: run CMD every INTERVAL s until it returns 0 (0), for at most SECONDS on
# _vr_clock, CMD's own time included: CMD passes "$VR_RT" to its kubectl requests, which _vr_poll cuts to
# the time left. CMD returns 2 when waiting longer cannot help (a Pod that Failed): 2. Time up: 1.
_vr_poll() {
  local timeout="$1"
  shift
  _vr_clock
  _vr_poll_until "$((VR_CLOCK + timeout))" "$@"
}

# _vr_poll_until DEADLINE INTERVAL CMD…: _vr_poll up to DEADLINE, a reading of _vr_clock, for a bound that
# several waits share (E1's 120 s and 600 s both count from the reboot command).
_vr_poll_until() {
  local deadline="$1" interval="$2" rc VR_RT
  shift 2
  while _vr_rt_until "$deadline"; do
    rc=0
    "$@" || rc=$?
    ((rc != 0)) || return 0
    ((rc != 2)) || return 2
    _vr_clock
    ((VR_CLOCK + interval < deadline)) || return 1
    _vr_wait "$interval"
  done
  return 1
}

# _vr_gpu_nodes: VR_GPU_NODES = the nodes labelled $_VR_GPU_LABEL. Returns 77 (SKIP) when there are
# none, 1 when the list cannot be read or holds a name that is not a GPU node's (k3s-gpu-<pool>-<n>).
_vr_gpu_nodes() {
  local out n
  VR_GPU_NODES=()
  out="$(vr_cap kc get nodes -l "$_VR_GPU_LABEL" "$_VR_RT" -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}')" \
    || { echo "D: cannot list the GPU nodes"; return 1; }
  while read -r n; do
    [[ -n "$n" ]] || continue
    [[ "$n" =~ $GPU_NODE_RE ]] || { echo "D: unexpected GPU node name from the cluster: '$n'"; return 1; }
    VR_GPU_NODES+=("$n")
  done <<< "$out"
  if ((${#VR_GPU_NODES[@]} == 0)); then
    echo "D: no node has the label $_VR_GPU_LABEL (no GPU node): skipped"
    return 77
  fi
  echo "D: GPU nodes: ${VR_GPU_NODES[*]}"
}

# D1: on every GPU node the bootstrap's audit (/var/log/gpu-audit.txt, spec §6.2 step 1) has content
# and k3s's containerd config has the nvidia runtime; the RuntimeClass nvidia-gpu exists.
check_d1_gpu_stack() {
  local n out fail=0
  _vr_gpu_nodes || return
  vr_env
  for n in "${VR_GPU_NODES[@]}"; do
    if out="$(vr_cap node_ssh "$n" 'sudo cat /var/log/gpu-audit.txt' </dev/null)" && [[ -n "${out//[[:space:]]/}" ]]; then
      echo "D1: $n: /var/log/gpu-audit.txt has content"
    else
      echo "D1: $n: /var/log/gpu-audit.txt is missing, unreadable or empty"
      fail=1
    fi
    if out="$(vr_cap node_ssh "$n" "$_VR_CONTAINERD_NVIDIA" </dev/null)" && [[ "$out" == *nvidia* ]]; then
      echo "D1: $n: containerd has the nvidia runtime"
    else
      echo "D1: $n: no nvidia runtime in k3s's containerd config"
      fail=1
    fi
  done
  vr_cmd kc get runtimeclass nvidia-gpu "$_VR_RT" || { echo "D1: no RuntimeClass nvidia-gpu"; fail=1; }
  return "$fail"
}

# _vr_d2_node NODE: kubectl describe node NODE shows nvidia.com/gpu >= 1 under both Capacity and
# Allocatable. VR_D2_OUT keeps the last answer, VR_D2_CAP/VR_D2_ALLOC the two values. Under _vr_poll, the
# request is bounded by the time left ($VR_RT).
_vr_d2_node() {
  local vals
  VR_D2_OUT="" VR_D2_CAP="" VR_D2_ALLOC=""
  VR_D2_OUT="$(vr_cap --no-echo kc describe node "$1" "${VR_RT:-$_VR_RT}")" || return 1
  vals="$(awk '/^Capacity:/ { s = "cap"; next } /^Allocatable:/ { s = "alloc"; next } /^[^ ]/ { s = "" }
               s && $1 == "nvidia.com/gpu:" { v[s] = $2 } END { print (v["cap"] == "" ? "-" : v["cap"]), (v["alloc"] == "" ? "-" : v["alloc"]) }' \
    <<< "$VR_D2_OUT")"
  read -r VR_D2_CAP VR_D2_ALLOC <<< "$vals"
  [[ "$VR_D2_CAP" =~ ^[0-9]+$ && "$VR_D2_ALLOC" =~ ^[0-9]+$ ]] && ((VR_D2_CAP >= 1 && VR_D2_ALLOC >= 1))
}

# D2: every GPU node reports nvidia.com/gpu >= 1 in Capacity and Allocatable (kubectl describe node).
# The resource appears about 20 s after the device plugin registers (live finding L21-2), so each
# node gets up to 90 s (every 10 s; the requests' own time included).
check_d2_gpu_capacity() {
  local n fail=0
  _vr_gpu_nodes || return
  for n in "${VR_GPU_NODES[@]}"; do
    if _vr_poll "$_VR_D2_WAIT_S" 10 _vr_d2_node "$n"; then
      echo "D2: $n: nvidia.com/gpu capacity $VR_D2_CAP, allocatable $VR_D2_ALLOC"
    else
      echo "D2: $n: nvidia.com/gpu capacity '$VR_D2_CAP', allocatable '$VR_D2_ALLOC' after $_VR_D2_WAIT_S s; want both >= 1"
      fail=1
    fi
    [[ -z "$VR_D2_OUT" ]] || printf -- '--- kubectl describe node %s ---\n%s\n' "$n" "$VR_D2_OUT"
  done
  return "$fail"
}

# D3: the device plugin Pods and their logs; the logs say "Registered device plugin".
check_d3_device_plugin() {
  local out
  _vr_gpu_nodes || return
  vr_cmd kc -n nvidia-device-plugin get pods -o wide "$_VR_RT" || true
  out="$(vr_cap kc -n nvidia-device-plugin logs -l app.kubernetes.io/name=nvidia-device-plugin --tail=100 "$_VR_RT")" \
    || { echo "D3: cannot read the device plugin logs"; return 1; }
  if [[ "$out" == *"Registered device plugin"* ]]; then
    echo "D3: the device plugin registered with the kubelet"
    return 0
  fi
  echo "D3: no 'Registered device plugin' in the last 100 log lines"
  return 1
}

# _vr_gpu_pod_delete POD: delete Pod POD in gpu-test (if it exists), bounded: --timeout the wait for
# the deletion, $_VR_RT each request (a stalled tunnel).
_vr_gpu_pod_delete() {
  vr_cmd kc -n gpu-test delete pod "$1" --ignore-not-found --timeout=120s "$_VR_RT"
}

# _vr_gpu_pod_phase POD: one look at POD's phase (VR_POD_PHASE) under _vr_poll: 0 Succeeded, 2 Failed (it
# never succeeds), else 1.
_vr_gpu_pod_phase() {
  VR_POD_PHASE="$(vr_cap kc -n gpu-test get pod "$1" "$VR_RT" -o 'jsonpath={.status.phase}')" || VR_POD_PHASE=""
  case "$VR_POD_PHASE" in
    Succeeded) return 0 ;;
    Failed) return 2 ;;
  esac
  return 1
}

# _vr_gpu_pod_wait POD: poll POD's phase every 5 s until it is Succeeded (0), for at most 300 s on the
# clock, the requests' own time included (1). A Pod that Failed never succeeds, so that ends the wait at
# once (1) instead of after 300 billed seconds.
_vr_gpu_pod_wait() {
  local rc=0
  VR_POD_PHASE=""
  _vr_poll "$_VR_POD_WAIT_S" 5 _vr_gpu_pod_phase "$1" || rc=$?
  case "$rc" in
    0) return 0 ;;
    2) echo "$1 Failed" ;;
    *) echo "$1 did not succeed within $_VR_POD_WAIT_S s (phase '$VR_POD_PHASE')" ;;
  esac
  return 1
}

# _vr_gpu_pod_run POD: apply k8s/gpu-test/POD.yaml, wait (at most 300 s) until it Succeeded, and set
# VR_POD_LOG to its log. When it does not succeed, its description goes into the evidence.
_vr_gpu_pod_run() {
  local pod="$1"
  VR_POD_LOG=""
  vr_cmd kc apply -f "$REPO_ROOT/k8s/gpu-test/$pod.yaml" "$_VR_RT" || return 1
  if ! _vr_gpu_pod_wait "$pod"; then
    vr_cmd kc -n gpu-test describe pod "$pod" "$_VR_RT" || true
    return 1
  fi
  VR_POD_LOG="$(vr_cap kc -n gpu-test logs "$pod" "$_VR_RT")" || { echo "cannot read the log of $pod"; return 1; }
}

_vr_d4_cleanup() { _vr_gpu_pod_delete gpu-smoke; }

_vr_d4() {
  local node type want
  _vr_d4_cleanup || { echo "D4: cannot delete gpu-smoke left from an earlier run"; return 1; }
  _vr_gpu_pod_run gpu-smoke || return 1
  [[ "$VR_POD_LOG" == *NVIDIA-SMI* ]] || { echo "D4: no NVIDIA-SMI table in the log of gpu-smoke"; return 1; }
  node="$(vr_cap kc -n gpu-test get pod gpu-smoke "$_VR_RT" -o 'jsonpath={.spec.nodeName}')" || return 1
  [[ "$node" =~ $GPU_NODE_RE ]] || { echo "D4: gpu-smoke ran on '$node', not on a GPU node"; return 1; }
  type="$(vr_cap kc get node "$node" "$_VR_RT" -o 'jsonpath={.metadata.labels.gpu-type}')" || return 1
  [[ "$type" =~ $_VR_GPU_TYPE_RE ]] || { echo "D4: $node has no valid gpu-type label: '$type'"; return 1; }
  # The label is the instance type's model, gpu_1x_a10 -> a10; nvidia-smi prints "NVIDIA A10". A longer
  # label (a100_sxm4) is matched by its model part (A100), as nvidia-smi spells the rest differently
  # (A100-SXM4-40GB). A whole word: an A100 must not pass for an A10.
  want="$(printf '%s' "${type%%_*}" | tr '[:lower:]' '[:upper:]')"
  if grep -Eq "(^|[^A-Za-z0-9])${want}([^A-Za-z0-9]|\$)" <<< "$VR_POD_LOG"; then
    echo "D4: gpu-smoke ran nvidia-smi on $node (gpu-type $type): its output names '$want'"
    return 0
  fi
  echo "D4: $node is labelled gpu-type $type, but nvidia-smi's output does not name '$want'"
  return 1
}

# D4: gpu-smoke (nvidia-smi as a non-root user) Succeeded within 300 s; its log has the NVIDIA-SMI
# table and names the GPU of the node it ran on: that node's gpu-type label, uppercased (A10, A6000).
check_d4_gpu_smoke() {
  _vr_gpu_nodes || return
  _vr_gpu_guarded "the Pod gpu-smoke in namespace gpu-test (it holds a billed GPU)" _vr_d4 _vr_d4_cleanup \
    kc -n gpu-test delete pod gpu-smoke --ignore-not-found --wait=false "$_VR_RT"
}

_vr_d5_cleanup() { _vr_gpu_pod_delete vectoradd; }

_vr_d5() {
  _vr_d5_cleanup || { echo "D5: cannot delete vectoradd left from an earlier run"; return 1; }
  _vr_gpu_pod_run vectoradd || return 1
  if [[ "$VR_POD_LOG" == *"Test PASSED"* ]]; then
    echo "D5: vectoradd: Test PASSED"
    return 0
  fi
  echo "D5: no 'Test PASSED' in the log of vectoradd"
  return 1
}

# D5: the CUDA vectorAdd sample Succeeded within 300 s and logged "Test PASSED".
check_d5_vectoradd() {
  _vr_gpu_nodes || return
  _vr_gpu_guarded "the Pod vectoradd in namespace gpu-test (it holds a billed GPU)" _vr_d5 _vr_d5_cleanup \
    kc -n gpu-test delete pod vectoradd --ignore-not-found --wait=false "$_VR_RT"
}

# Holders are deleted by their label: they have generated names (gpu-hold-xxxxx).
_vr_d6_cleanup() {
  local rc=0
  vr_cmd kc -n gpu-test delete pod -l app=gpu-hold --ignore-not-found --timeout=120s "$_VR_RT" || rc=1
  _vr_gpu_pod_delete gpu-overflow || rc=1
  return "$rc"
}

# _vr_d6_drop: D6's interrupt cleanup, and the command printed to run it by hand: the same Pods as
# _vr_d6_cleanup (the holders by their label, gpu-overflow by name), without the wait. Never every Pod in
# gpu-test: the stretch gpu-burn workload runs there too (review F8-1). Both deletes run (each bounded by
# $_VR_RT); it fails if either fails.
_vr_d6_drop() {
  local rc=0
  vr_cmd kc -n gpu-test delete pod -l app=gpu-hold --ignore-not-found --wait=false "$_VR_RT" || rc=1
  vr_cmd kc -n gpu-test delete pod gpu-overflow --ignore-not-found --wait=false "$_VR_RT" || rc=1
  return "$rc"
}

_vr_d6() {
  local out n alloc total=0 i running phase uid events window VR_RT
  out="$(vr_cap kc get nodes -l "$_VR_GPU_LABEL" "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{" "}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}')" || return 1
  while read -r n alloc; do
    [[ -n "$n" ]] || continue
    [[ "${alloc:-0}" =~ ^[0-9]+$ ]] || { echo "D6: $n: allocatable nvidia.com/gpu is not a number: '$alloc'"; return 1; }
    total=$((total + ${alloc:-0}))
  done <<< "$out"
  ((total >= 1)) || { echo "D6: no allocatable GPU to hold"; return 1; }
  echo "D6: $total allocatable GPU(s): one gpu-hold Pod each"
  _vr_d6_cleanup || { echo "D6: cannot delete the gpu-hold and gpu-overflow Pods left from an earlier run"; return 1; }
  for ((i = 0; i < total; i++)); do
    vr_cmd kc -n gpu-test create -f "$REPO_ROOT/k8s/gpu-test/gpu-hold.yaml" "$_VR_RT" || return 1
  done
  vr_cmd kc -n gpu-test wait pod -l app=gpu-hold --for=condition=Ready --timeout=300s \
    || { echo "D6: the gpu-hold Pods are not all Running within 300 s"; return 1; }
  out="$(vr_cap kc -n gpu-test get pods -l app=gpu-hold --field-selector=status.phase=Running "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}')" || return 1
  running="$(grep -c . <<< "$out" || true)"
  ((running == total)) || { echo "D6: $running gpu-hold Pod(s) Running; want $total"; return 1; }
  vr_cmd kc apply -f "$REPO_ROOT/k8s/gpu-test/gpu-overflow.yaml" "$_VR_RT" || return 1
  # The 30 s run from the apply, on the clock: a slow uid read is part of them, not added to them. The two
  # reads after them share one more such window (review F2-1), while every GPU is held.
  _vr_clock
  window=$((VR_CLOCK + _VR_D6_WAIT_S))
  # Events outlive their Pod (about an hour): only this gpu-overflow's, by its uid, count.
  _vr_rt_until "$window" || { echo "D6: no time left to read gpu-overflow's uid"; return 1; }
  uid="$(vr_cap kc -n gpu-test get pod gpu-overflow "$VR_RT" -o 'jsonpath={.metadata.uid}')" || return 1
  [[ "$uid" =~ ^[0-9a-f-]+$ ]] || { echo "D6: gpu-overflow has no valid uid: '$uid'"; return 1; }
  echo "D6: every GPU is held; looking at gpu-overflow $_VR_D6_WAIT_S s after it was applied"
  _vr_clock
  if ((window > VR_CLOCK)); then _vr_wait $((window - VR_CLOCK)); fi
  _vr_clock
  window=$((VR_CLOCK + _VR_D6_WAIT_S))
  _vr_rt_until "$window" || { echo "D6: no time left to read gpu-overflow's phase"; return 1; }
  phase="$(vr_cap kc -n gpu-test get pod gpu-overflow "$VR_RT" -o 'jsonpath={.status.phase}')" || return 1
  _vr_rt_until "$window" || { echo "D6: no time left to read gpu-overflow's events"; return 1; }
  events="$(vr_cap kc -n gpu-test get events \
    "--field-selector=involvedObject.kind=Pod,involvedObject.name=gpu-overflow,involvedObject.uid=$uid" "$VR_RT")" \
    || return 1
  if [[ "$phase" == Pending && "$events" == *"Insufficient nvidia.com/gpu"* ]]; then
    echo "D6: gpu-overflow is Pending with 'Insufficient nvidia.com/gpu'"
    return 0
  fi
  echo "D6: gpu-overflow is '$phase'; want Pending with an 'Insufficient nvidia.com/gpu' event"
  return 1
}

# D6: one gpu-hold Pod per allocatable GPU (on every GPU node), all Running within 300 s; then
# gpu-overflow is still Pending 30 s after it was applied, with an "Insufficient nvidia.com/gpu"
# event of that Pod (its uid, not an earlier run's), both read within 30 s more. The holders (by
# label) and gpu-overflow are deleted before and after.
check_d6_overflow() {
  _vr_gpu_nodes || return
  _vr_gpu_guarded "the gpu-hold Pods (each holds a billed GPU) and gpu-overflow in namespace gpu-test" \
    _vr_d6 _vr_d6_cleanup _vr_d6_drop
}

# --- E: resilience (spec §9 E; make verify RESILIENCE=1) -----------------------------------------

# E1 reboots a CPU worker, E2 replaces every GPU node (a billed launch), E3 puts a deliberate extra rule
# into the gpu-access firewall stack. E2 and E3 SKIP (77) without a GPU node, as D does; E2 also when
# the GPU spec asks for none, or when Lambda has no capacity to launch its replacement and the node does
# not hold a unit gpuctl down would free (that SKIP leaves E2 unverified, and the evidence says so).
# Every wait has a deadline on the D checks' clock, which counts the time its requests take (_vr_poll,
# _vr_poll_until: E1's two waits share one from the reboot command), and gpuctl's read-only calls are
# bounded with timeout; E1 compares the node's timestamps with the node's own clock, never the
# operator's. What E2 and E3 change is put right by a cleanup that runs however the check ends
# (_vr_e_guarded): E2 makes sure no GPU instance is left running when the replacement did not finish (no
# billed orphan), E3 removes the extra rule. On an interrupt (Ctrl-C, SIGTERM, a hangup) it is the
# harness's _vr_signalled that runs it, as for the D checks: the operator sees it and its outcome on the
# one redacted channel (_vr_tell), with what may remain and what to run if it fails. A hangup does not
# kill a terraform apply or gpuctl down/apply half way (they run with SIGHUP ignored, _vr_e_stream
# --nohup, so no Terraform state lock is orphaned); the cleanup runs once it has finished. A second
# Ctrl-C or a hangup does not stop the cleanup: it ignores SIGINT and SIGHUP and runs its commands in a
# process group of their own (_vr_shielded). A restore or sweep blocked by a state lock names the lock
# and its force-unlock (_vr_e_lock_hint). Command output that can carry the project id or the
# operator's IP (gpuctl, terraform, gcloud) goes into the evidence masked (_vr_e_mask); gpuctl masks the
# operator IP it detects itself.

_VR_E1_NODE=k3s-cpu-1
# When the node's Ready condition last changed (kubectl jsonpath).
_VR_E_READY_SINCE='{.status.conditions[?(@.type=="Ready")].lastTransitionTime}'
# E1's look at the nettest Pods on the rebooted node: per Pod, its container's ready flag and when that
# container started (RFC 3339, UTC).
_VR_E1_POD_STARTED='{range .items[*]}{.status.containerStatuses[0].ready}{" "}{.status.containerStatuses[0].state.running.startedAt}{"\n"}{end}'
# E1's reference time is the rebooted node's own clock, not the operator's: its kubelet renews the node's
# lease (kube-node-lease) every 10 s (renewInterval: a quarter of the 40 s lease duration), on the same host
# clock as the containers' startedAt and the Ready condition's lastTransitionTime it writes. A skewed
# operator clock (a laptop or WSL2 after sleep) would otherwise fail a healthy reboot (Task 22b review
# X1-2). The last renewal read just before the reboot command is at most this many seconds before it.
_VR_E1_RENEW_S=10
# How long Lambda may go on listing an instance gpuctl down terminated (gpuctl's Reconciler: it "may still
# list [it] as active for a moment after the terminate call returns"; down.sh waits 60 s).
_VR_E2_GONE_S=120
# How long E2 gives the unit gpuctl down freed to come back as capacity, when the pool showed none but its
# node held it (Ruling R61, review F-CAP): Lambda can show it free only a while after it stops listing the
# instance. Counted on _vr_clock from then; gpuctl status is polled every _VR_E2_FREED_POLL_S, and an apply
# that still finds no capacity is tried again after 15 s, 30 s, 60 s, … within the same bound.
_VR_E2_FREED_S=300
_VR_E2_FREED_POLL_S=15
# TEST-NET-2 (RFC 5737): reserved for documentation, never routed, so the extra rule E3 adds can never
# admit a real host, even if it stayed.
_VR_E3_IP=198.51.100.99
# The bound of one read-only gpuctl call (status, list-nodes): Lambda, Terraform outputs, kubectl and an
# SSH to k3s-cp, each bounded inside gpuctl too. gpuctl down and apply bound their own steps (launch,
# join, drain) and are not cut short from outside: a killed apply could not clean up what it launched.
_VR_E_READ_S=600
# The autoscaler's override file (stretch): gpuctl down removes it (spec section 10.4).
_VR_E_OVERRIDES="$REPO_ROOT/lambda/.autoscale-state.json"

# _vr_e_mask: stdin -> stdout with the GCP project id and the operator's IP (OPERATOR_IP in .env) as
# <GCP_PROJECT> and <operator_ip>: Terraform prints resource ids with the project, and a run promoted
# to verification/final is committed. Each value is checked first, so the only sed metacharacters in
# them are an IP's dots; the IP must stand alone (198.51.100.2 is not masked inside 198.51.100.20), and
# the loop (:m … tm) also catches two occurrences one character apart.
_vr_e_mask() {
  local -a exprs=(-e '')
  if _gcp_project_ok; then
    exprs+=(-e "s/${GCP_PROJECT}/<GCP_PROJECT>/g")
  fi
  if is_ipv4 "${OPERATOR_IP:-}"; then
    exprs+=(-e ':m' -e "s/(^|[^0-9.])${OPERATOR_IP//./\\.}(\$|[^0-9])/\\1<operator_ip>\\2/" -e 'tm')
  fi
  LC_ALL=C "${_VR_SED[@]}" -E "${exprs[@]}"
}

# _vr_e_stream [--nohup] CMD…: vr_cmd CMD with its stdout and stderr masked (_vr_e_mask) into the
# evidence as they come (gpuctl apply and terraform run for minutes). The filter ignores SIGINT/SIGTERM,
# as run_check's does, and SIGHUP, so a Ctrl-C or a hangup ends CMD but not the record of it. Returns
# CMD's status; 1 when CMD succeeded but its output could not be masked (the evidence is incomplete).
# --nohup: CMD runs with SIGHUP ignored, for the commands that change something (terraform apply, gpuctl
# down and apply). Terraform does not handle SIGHUP, so a hangup would kill it half way and orphan the
# GCS state lock, which never expires; gpuctl runs terraform apply too. Both keep an inherited SIG_IGN
# for SIGHUP (Go's os/signal does unless SIGHUP is asked for; Python leaves it alone). A hangup then lets
# CMD finish and release the lock, and the check's HUP trap (_vr_e_guarded) runs the cleanup after it
# (Task 22b review X1-1). A Ctrl-C still reaches CMD: terraform and gpuctl stop cleanly on it.
# VR_E_TEE=FILE: the masked output is also appended to FILE (a cleanup looks for a state-lock error in it).
_vr_e_stream() {
  local nohup=0
  local -a st=()
  if [[ "$1" == --nohup ]]; then
    nohup=1
    shift
  fi
  {
    (
      if ((nohup)); then trap '' HUP; fi
      vr_cmd "$@"
    ) </dev/null 2>&1 | (
      trap '' INT TERM HUP
      set -o pipefail
      if [[ -n "${VR_E_TEE:-}" ]]; then _vr_e_mask | tee -a -- "$VR_E_TEE"; else _vr_e_mask; fi
    )
    st=("${PIPESTATUS[@]}")
  } || :
  if ((st[1] != 0)); then
    echo "the output filter failed (exit ${st[1]}): the evidence of this command is incomplete"
    ((st[0] != 0)) || return 1
  fi
  return "${st[0]}"
}

# _vr_e_cap [--stdout] CMD…: _vr_e_stream for a short command whose output the check reads: the command
# and its output (stdout and stderr, masked) go into the evidence, and VR_E_OUT holds that output.
# --stdout: VR_E_OUT holds CMD's stdout only, for output parsed line by line (gpuctl list-nodes); its
# stderr (a Python warning, the wrapper's terraform init) goes into the evidence, masked, as it comes.
# Returns CMD's status; 1 when the output could not be masked (VR_E_OUT is then empty).
_vr_e_cap() {
  local rc=0 both=1
  if [[ "$1" == --stdout ]]; then
    both=0
    shift
  fi
  _vr_quote "$@" | redact | _vr_e_mask || return
  if ((both)); then
    VR_E_OUT="$("$@" </dev/null 2>&1)" || rc=$?
  else
    VR_E_OUT="$({
      "$@" </dev/null 2>&1 >&3 3>&- | (trap '' INT TERM HUP && _vr_e_mask) >&2 3>&-
      exit "${PIPESTATUS[0]}"
    } 3>&1)" || rc=$?
  fi
  if ! VR_E_OUT="$(_vr_e_mask <<< "$VR_E_OUT")"; then
    VR_E_OUT=""
    echo "cannot mask the output of $1: left out of the evidence"
    return 1
  fi
  [[ -z "$VR_E_OUT" ]] || printf '%s\n' "$VR_E_OUT"
  return "$rc"
}

# _vr_shielded CMD…: run CMD (a shell function too) in a process group of its own and wait for it. A
# terminal sends Ctrl-C to its foreground process group only, which CMD has left, so a second Ctrl-C
# during a cleanup cannot stop CMD half way (terraform and kubectl take SIGINT back even when it is
# ignored). The caller ignores SIGINT and SIGTERM itself meanwhile (trap '' INT TERM). CMD runs in a
# subshell: variables it sets are lost, so it reports through its output or a file.
# wait can return at once (128 + the signal) while CMD still runs: a second trapped signal that came
# with the first (a SIGTERM along with a SIGHUP) stays pending while the cleanup runs, and cuts every
# wait short. Then CMD is waited for by polling, and its status comes from a file it writes (review X0-1).
_vr_shielded() {
  local pid rc=0 st
  st="$(mktemp "${TMPDIR:-/tmp}/k3s-shielded.XXXXXX" 2>/dev/null)" || st=""
  set -m
  {
    rc=0
    "$@" || rc=$?
    if [[ -n "$st" ]]; then printf '%s\n' "$rc" >"$st"; fi
    exit "$rc"
  } &
  pid=$!
  set +m
  wait "$pid" || rc=$?
  if ((rc > 128)) && kill -0 "$pid" 2>/dev/null; then
    while kill -0 "$pid" 2>/dev/null; do
      [[ -z "$st" || ! -s "$st" ]] || break
      sleep 1
    done
    rc=1
    if [[ -n "$st" ]] && read -r rc <"$st" 2>/dev/null && [[ "$rc" =~ ^[0-9]+$ ]]; then :; else rc=1; fi
  fi
  [[ -z "$st" ]] || rm -f -- "$st"
  return "$rc"
}

# _vr_e_lock_hint PREFIX FILE: when FILE (masked terraform or gpuctl output) shows Terraform's "Error
# acquiring the state lock", say which lock and how to release it: an apply killed half way (kill -9, a
# crash, a hangup before review X1-1) leaves its GCS lock behind, it never expires, and make gpu-apply
# would fail on it as well. The lock ID and the stack come from Terraform's "Lock Info" (ID, and Path
# gs://<bucket>/<stack>/default.tflock); gpu-access when the path names no stack.
_vr_e_lock_hint() {
  local prefix="$1" file="$2" id stack
  [[ -n "$file" && -s "$file" ]] && grep -q 'Error acquiring the state lock' "$file" || return 0
  id="$(awk '/Lock Info:/ { t = 1 } t && $1 == "ID:" { print $2; exit }' "$file")"
  stack="$(awk '/Lock Info:/ { t = 1 } t && $1 == "Path:" { print $2; exit }' "$file" \
    | sed -nE 's#^gs://[^/]+/([a-z][a-z0-9-]*)/default\.tflock$#\1#p')"
  [[ -n "$stack" ]] || stack=gpu-access
  if [[ ! "$id" =~ ^[A-Za-z0-9-]+$ ]]; then
    echo "$prefix: the Terraform state of the $stack stack is locked (no lock ID found in the output above)"
    return 0
  fi
  echo "$prefix: the Terraform state of the $stack stack is locked (lock ID $id), most likely by an apply"
  echo "$prefix: that was killed half way. Once no terraform, gpuctl or make target runs anywhere, release it:"
  echo "$prefix:   terraform -chdir=infra/$stack force-unlock $id"
}

# _vr_e_gpuctl ARGS…: a read-only scripts/gpuctl ARGS (status, list-nodes), bounded, through _vr_e_cap;
# for list-nodes VR_E_OUT holds its stdout only (one line per node).
_vr_e_gpuctl() {
  local -a opt=()
  [[ "$1" != list-nodes ]] || opt=(--stdout)
  _vr_e_cap ${opt[@]+"${opt[@]}"} timeout --foreground "$_VR_E_READ_S" "$REPO_ROOT/scripts/gpuctl" "$@"
}

# _vr_e_no_drift ID: gpuctl status exits 0 with every drift source read and every GPU pool at its spec
# count. E2 and E3 change nothing otherwise: drift found later must be theirs, and a pool below its
# count would make their gpuctl apply launch more (billed) nodes than the check took away.
_vr_e_no_drift() {
  local rc=0
  _vr_e_gpuctl status || rc=$?
  if ((rc != 0)); then
    echo "$1: gpuctl status exits $rc before the test (want 0, no drift): changing nothing; make gpu-apply reconciles"
    return 1
  fi
  if grep -q '^ *skipped: ' <<< "$VR_E_OUT"; then
    echo "$1: gpuctl status could not read every drift source: changing nothing"
    return 1
  fi
  if grep -q '^ *note: ' <<< "$VR_E_OUT"; then
    echo "$1: a GPU pool is below its spec count, so gpuctl apply would launch more nodes: changing nothing"
    return 1
  fi
  echo "$1: gpuctl status: no drift"
}

# _vr_e_list_nodes: VR_E_NODES = "NAME IP" for every GPU instance gpuctl list-nodes reports, live in any
# status (IP "-" while it boots), and VR_E_WHERE = "NAME TYPE REGION" for each (list-nodes prints
# "<name> <ip|-> <status> <type> <region>"). Fails when the listing fails or holds a line that is not a
# node's.
_vr_e_list_nodes() {
  local name ip status type region rest
  VR_E_NODES=()
  VR_E_WHERE=()
  _vr_e_gpuctl list-nodes || { echo "E: gpuctl list-nodes failed"; return 1; }
  while read -r name ip status type region rest; do
    [[ -n "$name" ]] || continue
    if [[ ! "$name" =~ $GPU_NODE_RE ]] || { [[ "$ip" != - ]] && ! is_ipv4 "$ip"; }; then
      echo "E: unexpected line from gpuctl list-nodes: '$name $ip $status $type $region $rest'"
      return 1
    fi
    VR_E_NODES+=("$name $ip")
    VR_E_WHERE+=("$name ${type:--} ${region:--}")
  done <<< "$VR_E_OUT"
}

# _vr_e_listed NAME: the IP gpuctl list-nodes gave NAME (VR_E_NODES); fails when it is not listed.
_vr_e_listed() {
  local line
  for line in ${VR_E_NODES[@]+"${VR_E_NODES[@]}"}; do
    if [[ "${line%% *}" == "$1" ]]; then
      printf '%s\n' "${line#* }"
      return 0
    fi
  done
  return 1
}

# The bound of E2's and E3's cleanup (a gpuctl down with its drain, then up to _VR_E2_GONE_S of listing; a
# terraform apply that may wait up to 120 s for its state lock).
_VR_E_CLEANUP_S=1800

# _vr_e_guarded WHAT HINT CLEANUP FUNC: run FUNC, then CLEANUP however FUNC ends: it returns, exits (die,
# set -e) or is interrupted. Returns FUNC's status; 1 when FUNC passed but CLEANUP failed.
# It runs in a subshell of its own, so its traps end with it (the caller's stay as they were). FUNC runs
# in a further subshell (a die in it ends only FUNC) whose hangup trap, as run_check's, ends it only once
# its foreground command is over (Ruling R62); it hands CLEANUP what it changed through a state file
# ($VR_E_STATE, _vr_e_set and _vr_e_get), since its variables end with it.
# An interrupt (SIGINT, SIGTERM, SIGHUP) runs CLEANUP through the harness's _vr_signalled (_vr_on_signal
# --hup: bounded by _VR_E_CLEANUP_S, told to the operator, the check then ends by the signal); WHAT names
# what may remain if CLEANUP fails, and HINT what to run then. Otherwise _vr_e_finally runs it.
_vr_e_guarded() {
  local what="$1" hint="$2" cleanup="$3" func="$4"
  (
    rc=0
    # shellcheck disable=SC2030 # meant for this check's subshell (FUNC, CLEANUP) only
    VR_E_STATE="$(mktemp "${TMPDIR:-/tmp}/k3s-verify-e.XXXXXX")" \
      || { echo "cannot create a temporary file for the cleanup: changing nothing"; exit 1; }
    _vr_on_signal --hup --within "$_VR_E_CLEANUP_S" --left "$what" --hint "$hint" "$cleanup"
    (
      trap 'exit 129' HUP
      "$func"
    ) || rc=$?
    _vr_e_finally "$cleanup" "$rc"
  )
}

# _vr_e_finally CLEANUP RC: _vr_e_guarded's CLEANUP once FUNC has ended with RC (no interrupt so far).
# Exits with RC; 1 when RC is 0 but CLEANUP failed. CLEANUP runs as _vr_signalled runs it (in the
# background, bounded by _VR_E_CLEANUP_S): a Ctrl-C or a hangup now cannot cut it short. Its output goes
# into the evidence once it is over; when a signal came meanwhile (the evidence filter may have died with
# it), it goes to the operator and the evidence through _vr_tell instead, with the outcome, and the check
# then ends by that signal. A failed CLEANUP is told to the operator either way (_vr_cleanup_outcome).
# shellcheck disable=SC2031 # VR_E_STATE is set in _vr_e_guarded's subshell, which runs this
_vr_e_finally() {
  local cleanup="$1" rc="$2" crc=0 out sig
  _VR_E_PENDING=""
  trap '_VR_E_PENDING=${_VR_E_PENDING:-INT}' INT
  trap '_VR_E_PENDING=${_VR_E_PENDING:-TERM}' TERM
  trap '_VR_E_PENDING=${_VR_E_PENDING:-HUP}' HUP
  trap '' PIPE
  set +e
  out="$(mktemp "${TMPDIR:-/tmp}/k3s-verify-e.XXXXXX")" || out=/dev/null
  _vr_within "$_VR_E_CLEANUP_S" "$cleanup" </dev/null >"$out" 2>&1
  crc=$?
  sig="$_VR_E_PENDING"
  if [[ -n "$sig" ]]; then
    trap '' INT TERM HUP
    _vr_tell "interrupted (SIG$sig) during the cleanup ($cleanup); it ran to its end:"
    _vr_tell <"$out"
  else
    cat -- "$out"
  fi
  [[ "$out" == /dev/null ]] || rm -f -- "$out"
  rm -f -- "$VR_E_STATE"
  if ((crc != 0)) || [[ -n "$sig" ]]; then _vr_cleanup_outcome "$crc"; fi
  ((crc == 0 || rc != 0)) || rc=1
  if [[ -n "$sig" ]]; then
    trap - INT TERM HUP PIPE
    kill -s "$sig" "$BASHPID"
  fi
  exit "$rc"
}

# _vr_e_set KEY VALUE: record VALUE (one line) for _vr_e_guarded's CLEANUP. Fails when it cannot.
# shellcheck disable=SC2031 # VR_E_STATE is set in _vr_e_guarded's subshell, which runs this
_vr_e_set() {
  printf '%s=%s\n' "$1" "$2" >>"$VR_E_STATE"
}

# _vr_e_get KEY: the last VALUE recorded for KEY ("" when none).
# shellcheck disable=SC2031 # VR_E_STATE is set in _vr_e_guarded's subshell, which runs this
_vr_e_get() {
  sed -n "s/^$1=//p" "$VR_E_STATE" 2>/dev/null | tail -n 1
}

# _vr_e_epoch TIME: TIME, an RFC 3339 UTC time as the API server prints it (2026-10-22T09:15:42Z, or a
# MicroTime such as a lease's renewTime, 2026-10-22T09:15:42.123456Z), as Unix seconds (the fraction cut
# off); fails on anything else.
_vr_e_epoch() {
  local s
  s="$(jq -rn --arg t "$1" '$t | sub("\\.[0-9]+Z$"; "Z") | fromdateiso8601' 2>/dev/null)" \
    && [[ "$s" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$s"
}

# _vr_e_ready_state NODE: VR_E_READY = NODE's Ready condition (True, False, Unknown; "" when it cannot
# be read).
_vr_e_ready_state() {
  VR_E_READY="$(vr_cap kc get node "$1" "${VR_RT:-$_VR_RT}" \
    -o 'jsonpath={.status.conditions[?(@.type=="Ready")].status}')" || VR_E_READY=""
}

_vr_e_ready() {
  _vr_e_ready_state "$1"
  [[ "$VR_E_READY" == True ]]
}

# NotReady as kubectl shows it: Ready is False, or Unknown once the node controller stops hearing
# from the kubelet.
_vr_e_not_ready() {
  _vr_e_ready_state "$1"
  [[ -n "$VR_E_READY" && "$VR_E_READY" != True ]]
}

# _vr_e_node_field NODE JSONPATH: one field of node NODE (its uid, its boot ID).
_vr_e_node_field() {
  vr_cap kc get node "$1" "$_VR_RT" -o "jsonpath=$2"
}

# _vr_e_nettest_ready: the nettest DaemonSets (probe, echo) are rolled out on every node, so B1 finds a
# Running probe and echo Pod on a node that was just rebooted or replaced.
_vr_e_nettest_ready() {
  local ds fail=0
  for ds in probe echo; do
    vr_cmd kc -n nettest rollout status "ds/$ds" --timeout=300s \
      || { echo "E: the nettest $ds DaemonSet is not rolled out within 300 s"; fail=1; }
  done
  return "$fail"
}

# _vr_e1_pods_restarted NODE SINCE: on NODE, the nettest probe Pod and the echo Pod each have a ready
# container that started at or after SINCE (Unix seconds by NODE's clock: just before the reboot command,
# _vr_e1_node_now). After a reboot quicker
# than the node controller's grace period the Pods' Ready status can still be the stale one from before,
# and B1 would exec into a container that is being recreated (Task 22b review C0-3).
_vr_e1_pods_restarted() {
  local app out ready started t ok
  for app in probe echo; do
    out="$(vr_cap kc -n nettest get pods -l "app=$app" --field-selector="spec.nodeName=$1,status.phase=Running" \
      "${VR_RT:-$_VR_RT}" -o "jsonpath=$_VR_E1_POD_STARTED")" || return 1
    ok=0
    while read -r ready started; do
      [[ "$ready" == true ]] || continue
      t="$(_vr_e_epoch "$started")" || continue
      if ((t >= $2)); then ok=1; fi
    done <<< "$out"
    ((ok)) || return 1
  done
}

# _vr_e_utc UNIX_S: that time as RFC 3339 UTC, like the API server's timestamps.
_vr_e_utc() {
  jq -rn --argjson s "$1" '$s | todateiso8601'
}

# _vr_e1_node_now NODE: VR_E1_REF = NODE's clock now, as Unix seconds, and VR_E1_RENEW its RFC 3339 form:
# the renewTime of its node lease (at most $_VR_E1_RENEW_S s old). Fails when it cannot be read.
_vr_e1_node_now() {
  VR_E1_RENEW="$(vr_cap kc -n kube-node-lease get lease "$1" "$_VR_RT" -o 'jsonpath={.spec.renewTime}')" \
    && VR_E1_REF="$(_vr_e_epoch "$VR_E1_RENEW")"
}

# E1: reboot k3s-cpu-1 (`sudo systemctl reboot`; ssh's exit 255 when the connection drops is expected).
# PASS: it is NotReady within 120 s of the reboot command and Ready again within 600 s of it, with a new
# boot ID (it did reboot); its nettest probe and echo containers started again after the reboot command;
# and then B1 passes (the sysctls, WireGuard and the nettest Pods came back: live finding L18-1, IP
# forwarding must survive a reboot). NotReady is polled every 2 s; a reboot can be quicker than that
# window (the kubelet reports NotReady for a moment at start), so a Ready condition whose
# lastTransitionTime moved since before the reboot, to a time no later than 120 s after the reboot
# command, counts as seen: Ready changes only through NotReady, and the evidence shows both times. Any
# other exit of the reboot command than 0 and 255 is waited out too (an IAP SSH whose tunnel drops under
# the reboot can exit with another code): the boot ID says whether the node rebooted, so the next check
# never starts while it is still rebooting.
# The waits run on the operator's clock (_vr_clock: elapsed time only). The node's timestamps
# (lastTransitionTime, startedAt) are compared with the node's own clock read just before the reboot
# command (its lease's renewTime, at most $_VR_E1_RENEW_S s earlier; _vr_e1_node_now): "no later than 120 s
# after the reboot command" is no later than that reading + 120 + $_VR_E1_RENEW_S s, and "started after the
# reboot" is started at or after that reading. The operator's clock is never compared with them.
check_e1_worker_reboot() {
  local node="$_VR_E1_NODE" boot1 boot2 since1 since2 s2 t0 w0 ref rc=0
  vr_env
  _vr_e_ready_state "$node"
  if [[ "$VR_E_READY" != True ]]; then
    echo "E1: $node is not Ready before the reboot (Ready='$VR_E_READY'): not rebooting it"
    return 1
  fi
  boot1="$(_vr_e_node_field "$node" '{.status.nodeInfo.bootID}')" || return 1
  [[ -n "$boot1" ]] || { echo "E1: $node reports no boot ID"; return 1; }
  since1="$(_vr_e_node_field "$node" "$_VR_E_READY_SINCE")" || return 1
  [[ -n "$since1" ]] || { echo "E1: $node reports no lastTransitionTime for Ready"; return 1; }
  if ! _vr_e1_node_now "$node"; then
    echo "E1: cannot read $node's clock (the renewTime of its lease in kube-node-lease: '${VR_E1_RENEW:-}'): not rebooting it"
    return 1
  fi
  ref="$VR_E1_REF"
  _vr_clock
  t0="$VR_CLOCK"
  w0="$(date +%s)"
  _vr_e_stream worker_ssh "$node" 'sudo systemctl reboot' || rc=$?
  echo "E1: $node's clock just before the reboot command: $VR_E1_RENEW (its lease's last renewal; the operator's clock read $(_vr_e_utc "$w0"))"
  case "$rc" in
    0 | 255) echo "E1: reboot sent to $node (ssh exit $rc; 255 is the connection closing)" ;;
    *) echo "E1: the reboot command ended with exit $rc: waiting to see whether $node reboots anyway (its boot ID decides)" ;;
  esac
  if _vr_poll_until "$((t0 + 120))" 2 _vr_e_not_ready "$node"; then
    echo "E1: $node is NotReady (Ready=$VR_E_READY)"
  elif since2="$(_vr_e_node_field "$node" "$_VR_E_READY_SINCE")" && [[ -n "$since2" && "$since2" != "$since1" ]]; then
    s2="$(_vr_e_epoch "$since2")" || { echo "E1: unexpected Ready lastTransitionTime '$since2'"; return 1; }
    if ((s2 > ref + 120 + _VR_E1_RENEW_S)); then
      echo "E1: $node was never seen NotReady; Ready last changed at $since2, more than 120 s after the reboot command"
      echo "E1: (by $node's clock: more than $((120 + _VR_E1_RENEW_S)) s after its lease's renewal at $VR_E1_RENEW)"
      return 1
    fi
    echo "E1: $node went NotReady and back between two polls (Ready lastTransitionTime $since1 -> $since2)"
  else
    echo "E1: $node was never NotReady within 120 s of the reboot command"
    if boot2="$(_vr_e_node_field "$node" '{.status.nodeInfo.bootID}')" && [[ -n "$boot2" && "$boot2" != "$boot1" ]]; then
      echo "E1: its boot ID changed ($boot1 -> $boot2): it rebooted faster than the node controller marks a node NotReady"
    elif [[ "$rc" != 0 && "$rc" != 255 ]]; then
      echo "E1: the reboot command failed (exit $rc): its boot ID is unchanged, $node did not reboot"
    else
      echo "E1: its boot ID is unchanged: $node did not reboot (an ssh exit 255 can also be a connection that failed)"
    fi
    return 1
  fi
  if _vr_poll_until "$((t0 + 600))" 10 _vr_e_ready "$node"; then
    _vr_clock
    echo "E1: $node is Ready again ($((VR_CLOCK - t0)) s after the reboot command)"
  else
    echo "E1: $node is not Ready again within 600 s of the reboot command (Ready='$VR_E_READY')"
    echo "E1: to bring it back: look at its console (gcloud compute instances get-serial-port-output $node --zone ZONE),"
    echo "E1: then reset it (gcloud compute instances reset $node --zone ZONE; ZONE: the workers stack's worker_zones"
    echo "E1: output, terraform -chdir=infra/workers output worker_zones) and wait for kubectl get nodes to show it Ready"
    return 1
  fi
  boot2="$(_vr_e_node_field "$node" '{.status.nodeInfo.bootID}')" || return 1
  if [[ -z "$boot2" || "$boot2" == "$boot1" ]]; then
    echo "E1: the boot ID did not change ('$boot1' -> '$boot2'): $node did not reboot"
    return 1
  fi
  echo "E1: boot ID $boot1 -> $boot2"
  _vr_e_nettest_ready || return 1
  if ! _vr_poll 300 5 _vr_e1_pods_restarted "$node" "$ref"; then
    echo "E1: the nettest probe and echo Pods on $node did not restart after the reboot within 300 s"
    return 1
  fi
  echo "E1: the nettest probe and echo Pods on $node restarted after the reboot"
  echo "E1: B1 after the reboot"
  check_b1_matrix
}

# E2's cleanup (_vr_e_guarded): when the replacement did not finish (VR_E2_CHANGED 1 in the state file:
# gpuctl down or apply failed, the check died or was interrupted between them), no GPU instance may be
# left running: an instance gpuctl did not clean up, or one apply launched but did not join, is billed for
# nothing. Lists them; if any (or the listing fails), runs gpuctl down and lists again until Lambda stops
# listing them (_VR_E2_GONE_S). It ignores SIGINT, SIGHUP and SIGPIPE (it runs in a subshell of its own).
# shellcheck disable=SC2031 # VR_E_STATE is set in _vr_e_guarded's subshell, which runs this
_vr_e2_cleanup() {
  local rc=0
  trap '' INT HUP PIPE
  if [[ "$(_vr_e_get VR_E2_CHANGED)" == 1 ]]; then
    echo "E2: the replacement did not finish: making sure no GPU instance is left running (Ctrl-C waits for this)"
    _vr_e2_sweep || rc=$?
  fi
  rm -f -- "$VR_E_STATE"
  return "$rc"
}

# _vr_e2_listing FILE: gpuctl list-nodes into the evidence and its (masked) output into FILE.
_vr_e2_listing() {
  local rc=0
  _vr_e_gpuctl list-nodes || rc=$?
  printf '%s\n' "${VR_E_OUT:-}" >"$1"
  return "$rc"
}

# _vr_e2_none FILE: gpuctl list-nodes succeeds and lists nothing (FILE "": no way to tell, so no).
_vr_e2_none() {
  [[ -n "$1" ]] && _vr_shielded _vr_e2_listing "$1" && [[ -z "$(tr -d '[:space:]' <"$1")" ]]
}

_vr_e2_sweep() {
  local f rc=0
  f="$(mktemp)" || f=""
  if _vr_e2_none "$f"; then
    echo "E2: no GPU instance is left running; make gpu-apply brings the GPU node back"
  else
    echo "E2: removing what is left with gpuctl down"
    if ! VR_E_TEE="${f:+$f.out}" _vr_shielded _vr_e_stream --nohup "$REPO_ROOT/scripts/gpuctl" down --auto-approve; then
      echo "E2: gpuctl down failed"
      _vr_e_lock_hint E2 "${f:+$f.out}"
    fi
    # Lambda can list an instance it is terminating as active for a while: no false alarm (review S0-1).
    if _vr_poll "$_VR_E2_GONE_S" 5 _vr_e2_none "$f"; then
      echo "E2: no GPU instance is left running; make gpu-apply brings the GPU node back"
    else
      echo "E2: GPU instance(s) may still be running and billing: ${f:+$(tr '\n' ' ' <"$f")}"
      echo "E2: start make tunnel if it is not running (gpuctl down does not terminate a node it cannot drain while"
      echo "E2: the cluster exists), then run make gpu-down, then make gpu-status (want 0 instances)"
      rc=1
    fi
  fi
  [[ -z "$f" ]] || rm -f -- "$f" "$f.out"
  return "$rc"
}

# _vr_e2_gone NAME…: gpuctl list-nodes succeeds and lists none of NAME… (in any live status).
_vr_e2_gone() {
  local name
  _vr_e_list_nodes || return 1
  for name in "$@"; do
    if _vr_e_listed "$name" >/dev/null; then return 1; fi
  done
}

# _vr_e_pool NAME: the pool of GPU node NAME (k3s-gpu-<pool>-<n>, GPU_NODE_RE).
_vr_e_pool() {
  local n="${1#k3s-gpu-}"
  printf '%s\n' "${n%-*}"
}

# _vr_e2_pools NAME…: the pool of each NAME, one per line, sorted (the same pool once per node).
_vr_e2_pools() {
  local name
  for name in "$@"; do _vr_e_pool "$name"; done | LC_ALL=C sort
}

# _vr_e2_capacity POOL [TYPE REGION]: gpuctl status's capacity table (VR_E2_STATUS) has a row for POOL (one
# per candidate type and region) whose AVAILABLE column is yes: gpuctl apply has somewhere to launch it.
# With TYPE and REGION: the table has the row POOL TYPE REGION, available or not (a candidate of POOL).
_vr_e2_capacity() {
  awk -v pool="$1" -v type="${2:-}" -v region="${3:-}" '
    /^Capacity / { t = 1; next }
    t && NF == 0 { exit }
    t && $1 == pool && (type == "" ? $4 == "yes" : ($2 == type && $3 == region)) { f = 1 }
    END { exit !f }' <<< "$VR_E2_STATUS"
}

# _vr_e2_holds POOL: the first node of POOL (VR_E_WHERE) that runs one of POOL's candidate type/region pairs,
# as "NAME TYPE REGION"; fails when none does.
_vr_e2_holds() {
  local line name type region
  for line in ${VR_E_WHERE[@]+"${VR_E_WHERE[@]}"}; do
    read -r name type region <<< "$line"
    if [[ "$(_vr_e_pool "$name")" == "$1" ]] && _vr_e2_capacity "$1" "$type" "$region"; then
      printf '%s\n' "$line"
      return 0
    fi
  done
  return 1
}

# _vr_e2_freed POOL…: gpuctl status (any exit: the pools are below their counts now) shows capacity for each
# POOL (VR_E2_STATUS: its output).
_vr_e2_freed() {
  local pool
  _vr_e_gpuctl status || :
  VR_E2_STATUS="$VR_E_OUT"
  for pool in "$@"; do
    _vr_e2_capacity "$pool" || return 1
  done
}

# _vr_e2_apply DEADLINE HELD: gpuctl apply --auto-approve (SIGHUP ignored). HELD > 0 (a pool goes ahead on
# the unit its node held): an apply that fails with gpuctl's "no capacity for any candidate type" (nothing
# was launched; Lambda does not offer the freed unit to the launch call yet) is tried again after 15 s,
# 30 s, 60 s, … while the next try starts before DEADLINE (_vr_clock). Returns the last apply's status.
_vr_e2_apply() {
  local deadline="$1" held="$2" f rc backoff=15
  f="$(mktemp)" || f=""
  while :; do
    rc=0
    [[ -z "$f" ]] || : >"$f"
    VR_E_TEE="$f" _vr_e_stream --nohup "$REPO_ROOT/scripts/gpuctl" apply --auto-approve || rc=$?
    if ((rc == 0 || held == 0)) || [[ -z "$f" ]] || ! grep -q 'no capacity for any candidate type' "$f"; then
      break
    fi
    _vr_clock
    if ((VR_CLOCK + backoff >= deadline)); then
      echo "E2: gpuctl apply still finds no capacity until $_VR_E2_FREED_S s after gpuctl down: giving up"
      break
    fi
    echo "E2: gpuctl apply finds no capacity (Lambda does not offer the freed unit yet): trying again in $backoff s"
    _vr_wait "$backoff"
    backoff=$((backoff * 2))
  done
  [[ -z "$f" ]] || rm -f -- "$f"
  return "$rc"
}

_vr_e2() {
  local line name ip new_ip old_ip uid new_uid pool extra missing held type region deadline fail=0
  local -a before=() names=() uids=() now=() held_pools=()
  if [[ -e "$_VR_E_OVERRIDES" ]]; then
    echo "E2: lambda/.autoscale-state.json (the autoscaler's overrides) exists: gpuctl down removes it, so the"
    echo "E2: gpuctl apply after it would launch the spec's counts instead of what runs now: changing nothing"
    return 1
  fi
  _vr_e_no_drift E2 || return 1
  VR_E2_STATUS="$VR_E_OUT"
  _vr_e_list_nodes || return 1
  ((${#VR_E_NODES[@]})) || { echo "E2: gpuctl list-nodes lists no GPU instance"; return 1; }
  for line in "${VR_E_NODES[@]}"; do
    read -r name ip <<< "$line"
    uid="$(_vr_e_node_field "$name" '{.metadata.uid}')" || return 1
    [[ "$uid" =~ ^[0-9a-f-]+$ ]] || { echo "E2: node $name has no valid uid: '$uid'"; return 1; }
    before+=("$name $ip $uid")
    names+=("$name")
    uids+=("$uid")
  done
  # A node taken down that gpuctl apply cannot launch again is lost for the session (review X0-3). Lambda's
  # capacity says whether a NEW instance can launch now: the node under test may hold the last unit of its
  # type in its region, which gpuctl down frees for the apply (review C1-1). Then E2 goes ahead, with a
  # warning; only a pool with no capacity that none of its nodes holds is skipped.
  for pool in $(_vr_e2_pools "${names[@]}" | uniq); do
    _vr_e2_capacity "$pool" && continue
    if held="$(_vr_e2_holds "$pool")"; then
      read -r name type region <<< "$held"
      echo "E2: warning: gpuctl status shows no free capacity for GPU pool $pool,"
      echo "E2: but $name holds a $type in $region, one of the pool's candidates: gpuctl down frees that unit"
      echo "E2: for the apply, so it is replaced anyway (E2 waits up to $_VR_E2_FREED_S s for Lambda to show it free)."
      echo "E2: If Lambda hands the unit to someone else first, E2 FAILs, no GPU instance is left running, and"
      echo "E2: make gpu-apply brings the node back once there is capacity"
      held_pools+=("$pool")
      continue
    fi
    echo "E2: gpuctl status shows no capacity for GPU pool $pool (no candidate type available in any of its"
    echo "E2: regions), and none of its nodes holds a unit of a candidate that gpuctl down would free: gpuctl apply"
    echo "E2: could not bring the node back, so it is kept: skipped. E2 is NOT verified by this run (a SKIP counts"
    echo "E2: as success in the run's result): re-run scripts/verify.sh E once make gpu-status shows capacity, or"
    echo "E2: replace the node by hand (make gpu-down, then make gpu-apply, then make gpu-status)"
    return 77
  done
  echo "E2: replacing ${#before[@]} GPU node(s): gpuctl down, then gpuctl apply"

  # From here until the apply has brought the node(s) back, the cleanup makes sure no GPU instance runs.
  _vr_e_set VR_E2_CHANGED 1 || { echo "E2: cannot record the state for the cleanup: changing nothing"; return 1; }
  _vr_e_stream --nohup "$REPO_ROOT/scripts/gpuctl" down --auto-approve || { echo "E2: gpuctl down failed"; return 1; }
  # Lambda can list a terminated instance as active for a while (gpuctl's Reconciler; review S0-1).
  if ! _vr_poll "$_VR_E2_GONE_S" 5 _vr_e2_gone "${names[@]}"; then
    for name in "${names[@]}"; do
      if _vr_e_listed "$name" >/dev/null; then echo "E2: $name is still listed $_VR_E2_GONE_S s after gpuctl down"; fi
    done
    echo "E2: gpuctl down did not remove every GPU node within $_VR_E2_GONE_S s (or Lambda could not be listed)"
    return 1
  fi
  _vr_clock
  deadline=$((VR_CLOCK + _VR_E2_FREED_S))
  if ((${#held_pools[@]})); then
    echo "E2: waiting up to $_VR_E2_FREED_S s for gpuctl status to show the freed unit as capacity for GPU pool(s) ${held_pools[*]}"
    if ! _vr_poll_until "$deadline" "$_VR_E2_FREED_POLL_S" _vr_e2_freed "${held_pools[@]}"; then
      echo "E2: gpuctl status shows no capacity for GPU pool ${held_pools[*]} within $_VR_E2_FREED_S s after gpuctl down"
      echo "E2: freed its unit (Lambda may have handed it to someone else): not running gpuctl apply. Nothing was"
      echo "E2: launched; the cleanup below makes sure no GPU instance is left running. make gpu-apply brings the"
      echo "E2: node back once make gpu-status shows capacity"
      return 1
    fi
  fi
  _vr_e2_apply "$deadline" "${#held_pools[@]}" || { echo "E2: gpuctl apply failed"; return 1; }
  # What runs now is what the spec wants: nothing for the cleanup to remove, unless it is more (below). A
  # failed record leaves 1: the cleanup then removes the replacement too (nothing billed unattended).
  _vr_e_set VR_E2_CHANGED 0 || echo "E2: cannot record the state for the cleanup: it removes every GPU instance"

  _vr_e_list_nodes || return 1
  for line in ${VR_E_NODES[@]+"${VR_E_NODES[@]}"}; do now+=("${line%% *}"); done
  # Compared per pool: gpuctl apply names a node by the lowest free index of its pool, so the
  # replacement of k3s-gpu-a6000-2 can come back as k3s-gpu-a6000-1 (review S0-5).
  extra="$(LC_ALL=C comm -13 <(_vr_e2_pools "${names[@]}") <(_vr_e2_pools ${now[@]+"${now[@]}"}) | uniq | tr '\n' ' ')"
  missing="$(LC_ALL=C comm -23 <(_vr_e2_pools "${names[@]}") <(_vr_e2_pools ${now[@]+"${now[@]}"}) | uniq | tr '\n' ' ')"
  if [[ -n "$extra" ]]; then
    # No extra GPU instance is left billing (review X0-4): the cleanup removes every one.
    _vr_e_set VR_E2_CHANGED 1 || echo "E2: cannot record the state for the cleanup: remove them with make gpu-down"
    echo "E2: gpuctl apply runs more GPU instances than the check replaced (pool(s) ${extra% }), billed: ${now[*]}"
    echo "E2: the cleanup removes every GPU instance (gpuctl down); make gpu-apply then launches what"
    echo "E2: lambda/gpu-nodes.yaml asks for, and make gpu-status shows it"
    return 1
  fi
  if [[ -n "$missing" ]]; then
    echo "E2: gpuctl apply did not bring back every replaced GPU node: pool(s) ${missing% } short (listed now: ${now[*]:-none})"
    return 1
  fi
  for line in "${VR_E_NODES[@]}"; do
    read -r name new_ip <<< "$line"
    old_ip=""
    for ip in "${before[@]}"; do
      if [[ "${ip%% *}" == "$name" ]]; then
        old_ip="${ip#* }"
        old_ip="${old_ip%% *}"
      fi
    done
    if [[ -n "$old_ip" ]]; then
      echo "E2: $name: $old_ip -> $new_ip"
    else
      echo "E2: $name: a new name in pool $(_vr_e_pool "$name"), $new_ip"
    fi
    if ! _vr_poll 300 10 _vr_e_ready "$name"; then
      echo "E2: $name is not Ready within 300 s after gpuctl apply (Ready='$VR_E_READY')"
      return 1
    fi
    new_uid="$(_vr_e_node_field "$name" '{.metadata.uid}')" || return 1
    if _vr_in "$new_uid" "${uids[@]}"; then
      echo "E2: $name is the same Node object as before (uid $new_uid): it was not replaced"
      return 1
    fi
    echo "E2: $name is Ready as a new Node (uid $new_uid; before: ${uids[*]})"
  done
  _vr_e_nettest_ready || return 1
  # Spec section 9 E2: the node rejoins, and the matrix and all of D pass on it (review S0-2).
  echo "E2: D1-D6 and B1 on the new GPU node(s)"
  check_d1_gpu_stack || fail=1
  check_d2_gpu_capacity || fail=1
  check_d3_device_plugin || fail=1
  check_d4_gpu_smoke || fail=1
  check_d5_vectoradd || fail=1
  check_d6_overflow || fail=1
  check_b1_matrix || fail=1
  return "$fail"
}

# _vr_e_spec_count: the number of GPU nodes lambda/gpu-nodes.yaml asks for (every pool's count), read with
# gpuctl's own spec loader (the repo's venv, as scripts/gpuctl runs it); 0 when there is no spec file.
# Fails when the spec cannot be read.
_vr_e_spec_count() {
  local spec="$REPO_ROOT/lambda/gpu-nodes.yaml"
  if [[ ! -e "$spec" ]]; then
    echo "E2: no GPU spec ($spec)" >&2
    echo 0
    return 0
  fi
  vr_cap env PYTHONPATH="$REPO_ROOT/lambda" "$REPO_ROOT/.venv/bin/python" -P -c \
    'import sys; from gpuctl.spec import load_spec; print(sum(p.count for p in load_spec(sys.argv[1]).pools.values()))' \
    "$spec" </dev/null
}

# E2: replace the GPU node(s): gpuctl down --auto-approve, then gpuctl apply --auto-approve. PASS: before,
# gpuctl status shows no drift; after down, Lambda stops listing the node within 120 s; after apply,
# each pool runs as many GPU nodes as before, no more (a new instance, maybe a new IP, host key and
# name: the lowest free index), each a new Node object that is Ready within 300 s, and D1-D6 and B1
# pass on them (spec section 9 E2: "the matrix and all of D"). SKIP without a GPU node, when the spec
# asks for none (no GPU pool, or every count 0: gpuctl apply would not bring the node back), and when
# gpuctl status shows no capacity to launch a node of its pool again and none of its nodes holds a unit
# that gpuctl down would free (the node is kept; the evidence says E2 is not verified). When it goes
# ahead on such a held unit, it first waits (up to _VR_E2_FREED_S) for gpuctl status to show the freed
# unit as capacity, and tries an apply that finds none again within that bound (Ruling R61). Refused (FAIL,
# nothing changed) while the autoscaler's override file exists. A replacement that does not finish, or
# that ends with more instances than before, leaves no GPU instance running (_vr_e2_cleanup).
check_e2_gpu_replace() {
  local want
  _vr_gpu_nodes || return
  want="$(_vr_e_spec_count)" || { echo "E2: cannot read the GPU spec lambda/gpu-nodes.yaml"; return 1; }
  [[ "$want" =~ ^[0-9]+$ ]] || { echo "E2: unexpected node count from the GPU spec: '$want'"; return 1; }
  if ((want == 0)); then
    echo "E2: the GPU spec asks for no GPU node, so gpuctl apply would not bring one back: skipped"
    return 77
  fi
  vr_env
  _vr_e_guarded "GPU instance(s) billed for nothing (k3s-gpu-*)" \
    "start make tunnel if it is not running, then run make gpu-down, then make gpu-status (want 0 instances)" \
    _vr_e2_cleanup _vr_e2
}

# _vr_e3_ips: the gpu-access stack's gpu_ips output (JSON) on stdout, as vr_cap prints it; the command,
# the output and Terraform's messages (they name the state bucket, so the project) go into the
# evidence masked.
_vr_e3_ips() {
  local -a st=()
  { vr_cap tf_out_json gpu-access gpu_ips 2>&1 >&3 | (trap '' INT TERM HUP && _vr_e_mask) >&2; st=("${PIPESTATUS[@]}"); } 3>&1 || :
  return "${st[0]}"
}

# _vr_e3_addresses: the resource addresses in the gpu-access state (terraform state list) on stdout; the
# command, the list and Terraform's messages go into the evidence masked, as for _vr_e3_ips.
_vr_e3_addresses() {
  local -a st=()
  { vr_cap terraform -chdir="$VR_E3_DIR" state list 2>&1 >&3 | (trap '' INT TERM HUP && _vr_e_mask) >&2; st=("${PIPESTATUS[@]}"); } 3>&1 || :
  return "${st[0]}"
}

# _vr_e3_state FILE: _vr_e3_ips into FILE and _vr_e3_addresses into FILE.addr; fails when either fails.
_vr_e3_state() {
  _vr_e3_ips >"$1" || return
  _vr_e3_addresses >"$1.addr"
}

# _vr_e3_gone FILE: the gpu-access state can be read (into FILE, FILE.addr) and holds no rule for
# $_VR_E3_IP (FILE "": no way to tell, so no). Both the gpu_ips output and the firewall resources
# themselves: gpu_ips echoes the stack's input, so an apply that failed half way can leave it clean
# while the rule (google_compute_firewall.gpu_to_*["<ip>"]) is still in the state (review X0-2).
_vr_e3_gone() {
  [[ -n "$1" ]] && _vr_shielded _vr_e3_state "$1" || return 1
  jq -e --arg ip "$_VR_E3_IP" 'type == "array" and ([.[] | select(. == $ip)] | length == 0)' "$1" >/dev/null 2>&1 \
    && ! grep -qF "[\"$_VR_E3_IP\"]" "$1.addr"
}

# _vr_e3_tf_apply GPU_IPS_JSON [VAR…]: terraform apply of the gpu-access stack for exactly these GPU IPs,
# with SIGHUP ignored (_vr_e_stream --nohup).
_vr_e3_tf_apply() {
  local ips="$1"
  shift
  _vr_e_stream --nohup terraform -chdir="$VR_E3_DIR" apply -auto-approve -input=false -lock-timeout=120s "$@" \
    "-var=gpu_ips=$ips"
}

# E3's cleanup (_vr_e_guarded), once the extra rule may exist (VR_E3_ORIG in the state file: the gpu_ips
# from before the check): if the gpu-access state still holds $_VR_E3_IP (or cannot be read), apply those
# GPU IPs again. It ignores SIGINT, SIGHUP and SIGPIPE (it runs in a subshell of its own).
# shellcheck disable=SC2031 # VR_E_STATE is set in _vr_e_guarded's subshell, which runs this
_vr_e3_cleanup() {
  local rc=0
  trap '' INT HUP PIPE
  VR_E3_ORIG="$(_vr_e_get VR_E3_ORIG)"
  if [[ -n "$VR_E3_ORIG" ]]; then
    echo "E3: cleanup: the rule for $_VR_E3_IP must not stay (Ctrl-C waits for this)"
    _vr_e3_restore || rc=$?
  fi
  rm -f -- "$VR_E_STATE"
  return "$rc"
}

# The restore counts only when its own terraform apply succeeded and the state then holds no rule for
# $_VR_E3_IP (review X0-2).
_vr_e3_restore() {
  local f rc=0
  f="$(mktemp)" || f=""
  if _vr_e3_gone "$f"; then
    echo "E3: cleanup: the gpu-access state has no rule for $_VR_E3_IP"
  else
    echo "E3: cleanup: restoring gpu_ips=$VR_E3_ORIG"
    if ! VR_E_TEE="${f:+$f.out}" _vr_shielded _vr_e3_tf_apply "$VR_E3_ORIG"; then
      echo "E3: cleanup: terraform apply failed"
      _vr_e_lock_hint "E3: cleanup" "${f:+$f.out}"
      rc=1
    elif _vr_e3_gone "$f"; then
      echo "E3: cleanup: restored: the gpu-access state has no rule for $_VR_E3_IP"
    else
      rc=1
    fi
    if ((rc)); then
      echo "E3: cleanup: the gpu-access stack may STILL allow $_VR_E3_IP (TEST-NET-2, never routed):"
      echo "E3: cleanup: run make gpu-apply, then make gpu-status (want no drift)"
    fi
  fi
  [[ -z "$f" ]] || rm -f -- "$f" "$f.addr" "$f.out"
  return "$rc"
}

_vr_e3() {
  local ips n ip inject rc fail=0
  _vr_e_no_drift E3 || return 1
  _vr_e_stream tf_init gpu-access || return 1
  ips="$(_vr_e3_ips)" || { echo "E3: cannot read gpu_ips from the gpu-access stack"; return 1; }
  if ! jq -e 'type == "array" and all(.[]; type == "string")' <<< "$ips" >/dev/null 2>&1; then
    echo "E3: unexpected gpu_ips output from the gpu-access stack"
    return 1
  fi
  while read -r ip; do
    [[ -z "$ip" ]] || is_ipv4 "$ip" || { echo "E3: gpu_ips holds '$ip', not an IPv4 address"; return 1; }
  done <<< "$(jq -r '.[]' <<< "$ips")"
  VR_E3_ORIG="$(jq -c . <<< "$ips")"
  n="$(jq length <<< "$ips")"
  inject="$(jq -c --arg ip "$_VR_E3_IP" '. + [$ip]' <<< "$ips")"

  _vr_e_set VR_E3_ORIG "$VR_E3_ORIG" || { echo "E3: cannot record the state for the cleanup: changing nothing"; return 1; }
  echo "E3 ①: adding a rule for $_VR_E3_IP next to the $n real GPU IP(s)"
  # max_nodes: the stack refuses more IPs than its copy of the spec's max_nodes; the extra IP is one more.
  _vr_e3_tf_apply "$inject" "-var=max_nodes=$((n + 1))" || { echo "E3 ①: terraform apply failed"; return 1; }

  rc=0
  _vr_e_gpuctl status || rc=$?
  if ((rc != 3)); then
    echo "E3 ②: gpuctl status exits $rc; want 3 (drift)"
    fail=1
  elif ! grep -Eq "(^|[^0-9.])${_VR_E3_IP//./\\.}([^0-9]|\$)" <<< "$VR_E_OUT"; then
    echo "E3 ②: gpuctl status exits 3 but does not name $_VR_E3_IP"
    fail=1
  else
    echo "E3 ②: gpuctl status exits 3 and names $_VR_E3_IP"
  fi

  if _vr_e_stream --nohup "$REPO_ROOT/scripts/gpuctl" apply --auto-approve; then
    echo "E3 ③: gpuctl apply repaired the drift"
  else
    echo "E3 ③: gpuctl apply failed"
    fail=1
  fi

  rc=0
  _vr_e_gpuctl status || rc=$?
  if ((rc == 0)); then
    echo "E3 ④: gpuctl status exits 0: no drift"
  else
    echo "E3 ④: after the repair gpuctl status exits $rc; want 0"
    fail=1
  fi
  return "$fail"
}

# E3: drift in the GPU firewall is found and repaired. ① terraform apply of the gpu-access stack with
# gpu_ips = the real GPU IPs plus $_VR_E3_IP (one extra rule, via TF_VAR_* and tf_init as the other
# scripts); ② gpuctl status exits 3 and names that IP; ③ gpuctl apply --auto-approve repairs it; ④
# gpuctl status exits 0. Nothing is injected unless gpuctl status shows no drift first, and the extra
# rule is removed however the check ends (_vr_e3_cleanup). SKIP without a GPU node (gpuctl apply would
# launch one).
check_e3_firewall_drift() {
  _vr_gpu_nodes || return
  vr_env
  # The cleanup's terraform apply needs the stack's variables too (tf_env, which exits on a bad .env).
  tf_env
  VR_E3_DIR="$(tf_dir gpu-access)" || return 1
  _vr_e_guarded "a gpu-access firewall rule for $_VR_E3_IP (TEST-NET-2, never routed)" \
    "run make gpu-apply, then make gpu-status (want no drift)" _vr_e3_cleanup _vr_e3
}

# --- run summary --------------------------------------------------------------------------------

# vr_finish: append the totals to SUMMARY.md and scan the run directory with gitleaks (when it is
# installed). Returns 1 if any check failed or gitleaks found a secret, else 0 (SKIPs are not failures).
vr_finish() {
  local rc=0 scan
  printf '\n**Result:** %d passed, %d failed, %d skipped\n' \
    "$((VR_SEQ - VR_FAILS - VR_SKIPS))" "$VR_FAILS" "$VR_SKIPS" >>"$VR_DIR/SUMMARY.md"
  if command -v gitleaks >/dev/null 2>&1; then
    if gitleaks dir "$VR_DIR" --config "$REPO_ROOT/.gitleaks.toml" --no-banner --redact --exit-code 1; then
      scan="gitleaks found no leaks"
    else
      scan="gitleaks reported leaks (or failed); do not share this run"
      rc=1
    fi
  else
    scan="not run (gitleaks is not installed)"
    log "warning: gitleaks is not installed; $VR_DIR was not scanned for secrets"
  fi
  printf '\n**Secret scan:** %s\n' "$scan" >>"$VR_DIR/SUMMARY.md"
  if ((VR_FAILS > 0)); then rc=1; fi
  printf 'verification: %d passed, %d failed, %d skipped; %s; summary: %s\n' \
    "$((VR_SEQ - VR_FAILS - VR_SKIPS))" "$VR_FAILS" "$VR_SKIPS" "$scan" "$VR_DIR/SUMMARY.md"
  return "$rc"
}
