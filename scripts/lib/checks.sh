# shellcheck shell=bash
# Verification harness shared by scripts/verify.sh and the demo (spec section 9). Source it; do not run it.
#
#   vr_init DIR                     start a run: DIR (mode 0700)/SUMMARY.md with an empty results table;
#                                   returns 2 if DIR already holds files; finds the operator's IP
#                                   (vr_operator_ip), which redact then masks as <operator_ip>
#   run_check ID TITLE FUNC [ARG…]  run one check into DIR/<NN>-<ID>.txt; FUNC returns 0 PASS, 77 SKIP, else FAIL
#   vr_cmd CMD…                     print "$ CMD" (redacted), then run CMD; returns CMD's status
#   vr_cap [--no-echo] CMD…         vr_cmd for $(…): command and output to stderr, stdout to the caller
#   vr_bg FILE CMD…                 print "$ CMD &", run CMD in the background into FILE (VR_BG_PID)
#   vr_retry TIMEOUT_S INTERVAL_S CMD…  retry CMD until it succeeds, for at most TIMEOUT_S seconds
#   redact                          stdin -> stdout filter that replaces secret values with <redacted>
#   vr_finish                       write the totals, scan the run with gitleaks; 1 on any FAIL or leak
#
# The check functions check_<id>_<name> (A–C: Task 18; D: Task 21, a SKIP placeholder until then; E:
# Task 22; S, the stretch checks, only with STRETCH=1: S0 stretch S0b) are registered in
# scripts/verify.sh; the demo (Task 25) calls them directly. Each returns 0
# (PASS), 1 (FAIL) or 77 (SKIP); one that cannot read .env exits 2 (a FAIL), so call it in a subshell.
# Every command a check runs goes through vr_cmd, vr_cap or vr_bg, so its evidence shows what ran
# (node_external_ip, probe_on and echo_on appear by name; their kubectl queries are in common.sh;
# so do cp_resolve and _worker_zone, the SSH zone lookups from Terraform outputs).
# The offline tests set VR_NO_WAIT=1 (no fixed sleeps), VR_PF_PORT (C3's local port) and
# VR_S0_PROM_PORT / VR_S0_GRAFANA_PORT (S0's).
# Rules for check functions:
#   - bound every wait (kubectl wait --timeout=…, curl --max-time …): run_check has no timeout of its own;
#     kubectl exec runs under kc_within SECONDS and other network calls under bounded SECONDS;
#   - background jobs (kubectl port-forward &, tcpdump &) are killed when the check returns, because
#     run_check waits for every process holding the check's output; so do not set an EXIT trap in a
#     check (it replaces that cleanup), and do not leave grandchildren (setsid, nohup) running;
#   - a check that leaves something behind until its end (a pcap, Pods, a namespace, a listening
#     server) removes it on SIGINT/SIGTERM too, with _vr_on_signal --left WHAT CMD; the operator sees
#     the cleanup and, if it fails, what may remain and the command that removes it;
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

# LC_ALL=C: match bytes, not characters. All patterns are ASCII, and BSD sed under a UTF-8 locale
# aborts ("illegal byte sequence") on binary or non-UTF-8 output such as tcpdump -A payloads.
redact() {
  LC_ALL=C "${_VR_SED[@]}" -E "${_VR_REDACT_EXPRS[@]}" ${_VR_MASK_EXPRS[@]+"${_VR_MASK_EXPRS[@]}"}
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
# (vr_operator_ip) and says in SUMMARY.md whether it is masked; when it is unknown, it warns.
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
  local opip
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
      printf '| # | ID | Check | Result |\n'
      printf '|---|---|---|---|\n'
    } >"$VR_DIR/SUMMARY.md"
  )
}

# EXIT trap of a check's subshell: kill the background jobs it left running (see the header). TERM
# again until they are gone, KILL what is left after 5s: one TERM is not enough. This trap makes bash
# catch TERM, and a job runs that handler until it has exec'd its program, so a TERM that arrives
# before then can be dropped and the program runs on, holding the check's output open.
# A job that runs a shell function (vr_bg kc …) is a subshell whose child is the real program (kubectl
# port-forward), so the children go first, as in _vr_stop: else a TERM to the check's shell alone (kill
# PID, not the process group) would leave the program running, still listening and holding the output.
_vr_reap() {
  local pids pid i
  for ((i = 0; i < 50; i++)); do
    pids="$(jobs -pr)"
    [[ -n "$pids" ]] || return 0
    for pid in $pids; do
      pkill -TERM -P "$pid" 2>/dev/null || true
    done
    # shellcheck disable=SC2086 # one PID per word
    kill $pids 2>/dev/null || true
    sleep 0.1
  done
  pids="$(jobs -pr)"
  for pid in $pids; do
    pkill -KILL -P "$pid" 2>/dev/null || true
  done
  # shellcheck disable=SC2086 # one PID per word
  [[ -z "$pids" ]] || kill -KILL $pids 2>/dev/null || true
}

# run_check ID TITLE FUNC [ARGS…]: run FUNC ARGS in a subshell (so exit and set -e failures end only
# the check) with stdin from /dev/null, stdout+stderr redacted into $VR_DIR/<NN>-<ID>.txt.
# FUNC's status: 0 PASS, 77 SKIP, anything else FAIL; a failed redact filter (lost evidence) is a FAIL
# too. Always returns 0; vr_finish reports the outcome.
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
    "$func" "$@"
  ) 3>&2 </dev/null 2>&1 | redact >>"$file"
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

# _vr_on_signal [--left WHAT] CMD…: when the check is interrupted (SIGINT: Ctrl-C during make verify
# or the demo; SIGTERM), kill its background jobs, run CMD (for at most 120 s; a shell function is
# fine) and end the check by the same signal, so verify.sh and the demo stop as they would without
# the trap. WHAT names what CMD removes; if CMD fails, the operator is told it may remain and how to
# remove it. _vr_on_signal without CMD removes the trap. The EXIT trap stays the harness's (_vr_reap).
_vr_on_signal() {
  if (($# == 0)); then
    trap - INT TERM
    return 0
  fi
  _VR_ON_SIGNAL_LEFT="what that command removes"
  if [[ "$1" == --left ]]; then
    _VR_ON_SIGNAL_LEFT="$2"
    shift 2
  fi
  _VR_ON_SIGNAL=("$@")
  trap '_vr_signalled INT' INT
  trap '_vr_signalled TERM' TERM
}

# _vr_tell [LINE…]: LINEs (else stdin's lines), redacted, to the operator and into the check's
# evidence file. The operator sees them on run_check's own stderr (fd 3; outside run_check, stderr),
# prefixed with the check's id. For the interrupt cleanup: Ctrl-C has also ended the evidence filter
# the check's output normally goes through, so that output would be lost.
# shellcheck disable=SC2031 # run_check sets the _VR_CHECK_* and _VR_TELL_FD in the check's subshell
_vr_tell() {
  local line fd="${_VR_TELL_FD:-2}"
  if (($#)); then printf '%s\n' "$@"; else cat; fi | redact | while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -n "${_VR_CHECK_FILE:-}" ]]; then printf '%s\n' "$line" >>"$_VR_CHECK_FILE"; fi
    printf '%s: %s\n' "${_VR_CHECK_ID:-check}" "$line" >&"$fd"
  done 2>/dev/null
}

# _vr_signalled SIGNAL: the trap set by _vr_on_signal. SIGPIPE is ignored and errexit is off while it
# cleans up (a closed pipe must not end kubectl or gcloud before they have done their work). The
# cleanup's progress, its output and its outcome go to the operator and the evidence (_vr_tell).
# A further Ctrl-C is ignored until then, by this shell and (inherited) by the cleanup's commands, the
# ones under bounded or kc_within too (_VR_CLEANING: see bounded), so it can neither cut the cleanup
# short nor hide its outcome. A further SIGTERM stops the cleanup's commands (they keep its default
# action: TERM is also how the 120 s bound stops them) but not this shell, which still reports the
# outcome; the first signal then ends the check.
_vr_signalled() {
  local line rc how
  trap '' INT
  trap ':' TERM
  set +e
  trap '' PIPE
  _VR_CLEANING=1
  _vr_reap
  line="$(_vr_quote "${_VR_ON_SIGNAL[@]}")"
  line="${line#\$ }"
  _vr_tell "interrupted (SIG$1); cleaning up (at most 120 s): $line"
  _vr_within 120 "${_VR_ON_SIGNAL[@]}" </dev/null 2>&1 | _vr_tell
  rc=${PIPESTATUS[0]}
  if ((rc == 0)); then
    _vr_tell "cleanup done"
  else
    how="exit $rc"
    ((rc != 124)) || how="stopped after 120 s"
    # In a child shell: pasted into the operator's own bash, a die (exit) ends only that child, and
    # .env's values and the harness's functions do not stay behind in the operator's shell.
    line="source scripts/lib/checks.sh && vr_env && $line"
    _vr_tell "cleanup FAILED ($how); may remain: $_VR_ON_SIGNAL_LEFT" \
      "to remove it, run from the repository root: bash -c $(_vr_sq "$line")"
  fi
  trap - INT TERM PIPE
  kill -s "$1" "${BASHPID:-$$}"
  exit 1
}

# vr_env: load .env once per shell (GCP_PROJECT for IAP SSH and gcloud, UNDERLAY_MTU, the Lambda SSH
# key path); a missing or non-private .env ends the check (exit 2, a FAIL). The Lambda API key is
# dropped again: no check needs it, and scripts/gpuctl loads .env itself.
vr_env() {
  [[ "${_VR_ENV_LOADED:-}" == 1 ]] && return 0
  load_env
  unset LAMBDA_API_KEY
  _VR_ENV_LOADED=1
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

# _vr_mask_ids: stdin -> stdout with the GCP project id from .env replaced by <GCP_PROJECT> and the
# operator's IP (vr_operator_ip) by <operator_ip>: B6's evidence holds both, and a run promoted to
# verification/final is committed (Ruling R39 keeps real project ids and IPs out of what is
# delivered). redact masks the IP in all evidence too; this covers a check called without vr_init.
# The project id is checked first, so it holds no sed metacharacter. The source of Lambda's
# "operator ssh" rule is masked even when the operator's IP is unknown.
_vr_mask_ids() {
  local -a exprs=(-e '')
  if _gcp_project_ok; then
    exprs+=(-e "s/${GCP_PROJECT}/<GCP_PROJECT>/g")
  fi
  exprs+=(-e "s#${_VR_OPERATOR_RULE_RE}#\\1<operator_ip>\\2#")
  LC_ALL=C sed -E "${exprs[@]}" ${_VR_MASK_EXPRS[@]+"${_VR_MASK_EXPRS[@]}"}
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
# VR_PF_PORT (default 18080) is the local port; the offline tests pick a free one.
check_c3_port_forward() {
  local port="${VR_PF_PORT:-18080}" logf pid body rc=0
  [[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "C3: VR_PF_PORT is not a port: '$port'"; return 1; }
  logf="$(mktemp)" || return 1
  vr_bg "$logf" kc -n hello port-forward svc/hello "$port:80"
  pid="$VR_BG_PID"
  if _vr_wait_text "$logf" "Forwarding from" "$pid" 30; then
    body="$(vr_cap curl -fsS -m 5 "localhost:$port")" || rc=$?
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

# D1-D6 are implemented by Task 21; until then they are registered with this placeholder (SKIP).
check_d_pending() {
  echo "D: the GPU checks D1-D6 are added by Task 21; skipped until then"
  return 77
}

# --- S: stretch (make verify STRETCH=1; spec §10.0) ---------------------------------------------

# The admin-only label of a GPU node (join-node.sh sets it); the DCGM exporter runs on these nodes only.
VR_GPU_LABEL='node-restriction.kubernetes.io/accelerator=nvidia'
# How long S0 waits (wall clock, vr_retry) for the scrape targets, for DCGM data (its Pod is Ready about
# 45 s after it starts, then one 15 s scrape) and for Grafana's health.
VR_S0_TARGETS_WAIT_S=120
VR_S0_DCGM_WAIT_S=180
VR_S0_GRAFANA_WAIT_S=60
VR_S0_UP_QUERY='up{job=~"node-exporter|kube-state-metrics"}'

# _vr_s0_forward LOG SVC LOCAL REMOTE: kubectl port-forward svc/SVC in namespace monitoring on
# 127.0.0.1:LOCAL only (C3's pattern: vr_bg, wait for "Forwarding from", at most 30 s); VR_S0_PF is
# its PID, which _vr_s0_unforward stops. Prometheus and Grafana have no other way in (spec §10.0; the
# API server's service proxy is blocked by the namespace's default-deny NetworkPolicy, S0a).
_vr_s0_forward() {
  vr_bg "$1" kc -n monitoring port-forward --address 127.0.0.1 "svc/$2" "$3:$4"
  VR_S0_PF="$VR_BG_PID"
  _vr_wait_text "$1" "Forwarding from 127.0.0.1:$3" "$VR_S0_PF" 30
}

# _vr_s0_unforward LOG SVC: stop the port-forward and show its output.
_vr_s0_unforward() {
  _vr_stop "$VR_S0_PF"
  echo "S0: $2 port-forward stopped"
  _vr_show "$1" "port-forward svc/$2"
}

# _vr_s0_query PORT QUERY: the answer to a Prometheus instant query (through the port-forward). No -f:
# Prometheus answers a bad query with HTTP 400/422 and a JSON body that says why, which _vr_s0_result
# reports; an answer that is not JSON (an empty 503) fails there too. --noproxy: the port-forward is on
# 127.0.0.1, so an http_proxy in the operator's shell must not get the request.
_vr_s0_query() {
  vr_cap --no-echo curl -sS -m 5 --noproxy '*' --get --data-urlencode "query=$2" \
    "http://127.0.0.1:$1/api/v1/query"
}

# _vr_s0_result ANSWER JQ: the lines JQ makes of each series of a successful query. Fails with the
# reason: Prometheus' errorType and error, or (on stderr, the evidence) the start of an answer that is
# not a query result at all.
_vr_s0_result() {
  local rc=0 start
  jq -nr 'input | if .status == "success" and (.data.result | type) == "array" then .data.result[] | '"$2"'
          else error("Prometheus answered status \(.status // "?"): \(.errorType // "-"): \(.error // "no result")")
          end' <<< "$1" || rc=$?
  if ((rc != 0)); then
    start="$(head -c 200 <<< "$1" | tr -s '\r\n\t' '   ')"
    start="${start% }"
    echo "S0: the answer (at most 200 bytes): ${start:-(empty)}" >&2
  fi
  return "$rc"
}

# _vr_s0_targets PORT: S0 ①, one attempt: every node-exporter and kube-state-metrics target is up (1),
# kube-state-metrics has one, and every node in VR_NODES has an up node-exporter target. A node whose
# node-exporter Pod is not Running has no target at all (Prometheus discovers Running Pods only), which
# is why each node is looked for by name.
_vr_s0_targets() {
  local json lines job node inst val ksm=0 bad=0 n
  local -a ne=()
  json="$(_vr_s0_query "$1" "$VR_S0_UP_QUERY")" || return 1
  lines="$(_vr_s0_result "$json" '"\(.metric.job) \(.metric.node // "-") \(.metric.instance // "-") \(.value[1])"')" \
    || { echo "S0 ①: not a usable Prometheus answer"; return 1; }
  echo "S0 ①: targets (job node instance up):"
  [[ -n "$lines" ]] || echo "  (none)"
  while read -r job node inst val; do
    [[ -n "$job" ]] || continue
    echo "  $job $node $inst $val"
    if [[ "$val" != 1 ]]; then
      echo "S0 ①: $job target $inst (node $node) is down (up=$val)"
      bad=1
    elif [[ "$job" == kube-state-metrics ]]; then
      ksm=$((ksm + 1))
    elif [[ "$job" == node-exporter ]]; then
      ne+=("$node")
    fi
  done <<< "$lines"
  ((ksm > 0)) || { echo "S0 ①: no kube-state-metrics target is up"; bad=1; }
  for n in "${VR_NODES[@]}"; do
    _vr_in "$n" ${ne[@]+"${ne[@]}"} || { echo "S0 ①: no node-exporter target up on node $n"; bad=1; }
  done
  return "$bad"
}

# _vr_s0_dcgm PORT GPU_NODE…: S0 ②, one attempt: DCGM_FI_DEV_GPU_UTIL has a series from each GPU node.
_vr_s0_dcgm() {
  local port="$1" json lines node gpu val bad=0 g
  local -a seen=()
  shift
  json="$(_vr_s0_query "$port" DCGM_FI_DEV_GPU_UTIL)" || return 1
  lines="$(_vr_s0_result "$json" '"\(.metric.node // "-") \(.metric.gpu // "-") \(.value[1])"')" \
    || { echo "S0 ②: not a usable Prometheus answer"; return 1; }
  while read -r node gpu val; do
    [[ -n "$node" ]] || continue
    echo "S0 ②: DCGM_FI_DEV_GPU_UTIL node $node gpu $gpu: $val%"
    seen+=("$node")
  done <<< "$lines"
  for g in "$@"; do
    _vr_in "$g" ${seen[@]+"${seen[@]}"} || { echo "S0 ②: no DCGM_FI_DEV_GPU_UTIL from $g"; bad=1; }
  done
  return "$bad"
}

# _vr_s0_grafana PORT: S0 ③, one attempt: Grafana's /api/health (no login needed) says database ok.
_vr_s0_grafana() {
  local body db
  body="$(vr_cap curl -fsS -m 5 --noproxy '*' "http://127.0.0.1:$1/api/health")" || return 1
  db="$(jq -r '.database // empty' <<< "$body" 2>/dev/null)" || db=""
  [[ "$db" == ok ]] && return 0
  echo "S0 ③: Grafana /api/health says database '${db:-?}'; want ok"
  return 1
}

# _vr_s0 PROM_PORT GRAFANA_PORT DIR: S0's work (check_s0_metrics), the port-forward logs in DIR.
# Called in an || list, so errexit is off here: every failure is handled and the caller cleans up.
_vr_s0() {
  local pport="$1" gport="$2" dir="$3" out g n fail=0
  local -a gpus=() unlabelled=()
  vr_cmd kc -n monitoring get pods -o wide "$_VR_RT" || true
  vr_cmd kc -n monitoring-host get pods -o wide "$_VR_RT" || true
  _vr_load_nodes || return 1
  out="$(vr_cap kc get nodes -l "$VR_GPU_LABEL" "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}')" || return 1
  while read -r g; do
    [[ -n "$g" ]] || continue
    [[ "$g" =~ $K8S_NAME_RE ]] || { echo "S0: unexpected GPU node name from the cluster: '$g'"; return 1; }
    gpus+=("$g")
  done <<< "$out"
  # A joined GPU node (k3s-gpu-<pool>-<n>) without the admin label runs no DCGM exporter (the DaemonSet
  # selects the label): ② cannot pass for it, and it must not pass as "no GPU node".
  for n in "${VR_NODES[@]}"; do
    if [[ "$n" =~ $GPU_NODE_RE ]] && ! _vr_in "$n" ${gpus[@]+"${gpus[@]}"}; then
      unlabelled+=("$n")
      echo "S0 ②: FAILED: node $n looks like a GPU node but lacks the label $VR_GPU_LABEL, so the"
      echo "S0 ②: DCGM exporter does not run there (scripts/join-node.sh sets the label when the node joins)"
      fail=1
    fi
  done

  if _vr_s0_forward "$dir/prometheus.log" prometheus "$pport" 9090; then
    if vr_retry "$VR_S0_TARGETS_WAIT_S" 10 _vr_s0_targets "$pport"; then
      echo "S0 ①: every node-exporter (${#VR_NODES[@]} nodes) and kube-state-metrics target is up"
    else
      echo "S0 ①: FAILED: not every node-exporter and kube-state-metrics target was up within ${VR_S0_TARGETS_WAIT_S} s"
      fail=1
    fi
    if ((${#gpus[@]} == 0)); then
      ((${#unlabelled[@]})) || echo "S0 ②: no GPU node (label $VR_GPU_LABEL): the DCGM part is skipped"
    elif vr_retry "$VR_S0_DCGM_WAIT_S" 10 _vr_s0_dcgm "$pport" "${gpus[@]}"; then
      echo "S0 ②: DCGM_FI_DEV_GPU_UTIL has data from every GPU node (${gpus[*]})"
    else
      echo "S0 ②: FAILED: no DCGM_FI_DEV_GPU_UTIL from every GPU node (${gpus[*]}) within ${VR_S0_DCGM_WAIT_S} s"
      fail=1
    fi
  else
    echo "S0 ①②: FAILED: the prometheus port-forward did not start"
    fail=1
  fi
  _vr_s0_unforward "$dir/prometheus.log" prometheus

  if _vr_s0_forward "$dir/grafana.log" grafana "$gport" 3000; then
    if vr_retry "$VR_S0_GRAFANA_WAIT_S" 5 _vr_s0_grafana "$gport"; then
      echo "S0 ③: Grafana /api/health: database ok"
    else
      echo "S0 ③: FAILED: Grafana was not healthy within ${VR_S0_GRAFANA_WAIT_S} s"
      fail=1
    fi
  else
    echo "S0 ③: FAILED: the grafana port-forward did not start"
    fail=1
  fi
  _vr_s0_unforward "$dir/grafana.log" grafana
  return "$fail"
}

# S0 (stretch, STRETCH=1): the monitoring stack (k8s/monitoring/, make monitoring) works. ① Prometheus'
# `up` is 1 for every node-exporter target (one on every node) and every kube-state-metrics target;
# ② with a GPU node (label VR_GPU_LABEL), DCGM_FI_DEV_GPU_UTIL has data from each GPU node (without one
# this part is skipped and said so; a node named as a GPU node, k3s-gpu-*, without the label fails it);
# ③ Grafana's /api/health says database ok. Prometheus and Grafana are reached through a kubectl
# port-forward on 127.0.0.1 (never another address; curl ignores any proxy), one at a time, each stopped
# when its part is done; their logs are in a private temporary directory that is removed after, pass,
# fail or interrupt. SKIP when the stack is not installed (no namespace monitoring).
# VR_S0_PROM_PORT (default 19090) and VR_S0_GRAFANA_PORT (default 13000) are the local ports; they
# differ from make prometheus / make grafana's 9090 and 3000, so an open dashboard is not in the way.
check_s0_metrics() {
  local pport="${VR_S0_PROM_PORT:-19090}" gport="${VR_S0_GRAFANA_PORT:-13000}" p out dir rc=0
  for p in "$pport" "$gport"; do
    if [[ ! "$p" =~ ^[1-9][0-9]{0,4}$ ]] || ((p > 65535)); then
      echo "S0: VR_S0_PROM_PORT / VR_S0_GRAFANA_PORT: not a port: '$p'"
      return 1
    fi
  done
  out="$(vr_cap kc get namespace monitoring --ignore-not-found -o name "$_VR_RT")" \
    || { echo "S0: cannot tell whether the monitoring stack is installed (kubectl failed)"; return 1; }
  if [[ -z "$out" ]]; then
    echo "S0: SKIP: the monitoring stack is not installed (no namespace monitoring); install it with"
    echo "S0: make monitoring (or make up STRETCH=1), then run make verify STRETCH=1 again"
    return 77
  fi
  dir="$(umask 077 && mktemp -d)" || return 1
  _vr_on_signal --left "the port-forward logs in $dir" rm -rf -- "$dir"
  _vr_s0 "$pport" "$gport" "$dir" || rc=$?
  rm -rf -- "$dir"
  _vr_on_signal
  return "$rc"
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
