# shellcheck shell=bash
# Verification harness shared by scripts/verify.sh and the demo (spec section 9). Source it; do not run it.
#
#   vr_init DIR                     start a run: DIR (mode 0700)/SUMMARY.md with an empty results table;
#                                   returns 2 if DIR already holds files
#   run_check ID TITLE FUNC [ARG…]  run one check into DIR/<NN>-<ID>.txt; FUNC returns 0 PASS, 77 SKIP, else FAIL
#   vr_cmd CMD…                     print "$ CMD" (redacted), then run CMD; returns CMD's status
#   redact                          stdin -> stdout filter that replaces secret values with <redacted>
#   vr_finish                       write the totals, scan the run with gitleaks; 1 on any FAIL or leak
#
# The check functions themselves (A–E) are added by later tasks and registered in scripts/verify.sh.
# Rules for check functions:
#   - bound every wait (kubectl wait --timeout=…, curl --max-time …): run_check has no timeout of its own;
#   - background jobs (kubectl port-forward &, tcpdump &) are killed when the check returns, because
#     run_check waits for every process holding the check's output; so do not set an EXIT trap in a
#     check (it replaces that cleanup), and do not leave grandchildren (setsid, nohup) running.

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

# LC_ALL=C: match bytes, not characters. All patterns are ASCII, and BSD sed under a UTF-8 locale
# aborts ("illegal byte sequence") on binary or non-UTF-8 output such as tcpdump -A payloads.
redact() {
  LC_ALL=C "${_VR_SED[@]}" -E "${_VR_REDACT_EXPRS[@]}"
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
# verification/final/ is a deliberate copy.
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
  (
    umask 077
    {
      printf '# Verification run\n\n'
      printf 'Started: %s (%s)\n\n' "$(_vr_now)" "$VR_TZ"
      printf '| # | ID | Check | Result |\n'
      printf '|---|---|---|---|\n'
    } >"$VR_DIR/SUMMARY.md"
  )
}

# EXIT trap of a check's subshell: kill the background jobs it left running (see the header).
_vr_reap() {
  local pids
  pids="$(jobs -p)"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086 # one PID per word
    kill $pids 2>/dev/null || true
  fi
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
  (
    if ((errexit)); then set -e; fi
    trap _vr_reap EXIT
    "$func" "$@"
  ) </dev/null 2>&1 | redact >>"$file"
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

# vr_cmd CMD…: record the command (redacted) in the check's output, then run it.
# Arguments are shown shell-quoted with single quotes, not printf %q: %q turns "Bearer X" into
# Bearer\ X, which the redact patterns (key, whitespace, value) would no longer recognise.
vr_cmd() {
  local arg line='$'
  for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Za-z0-9_./:=@%+,-]+$ ]]; then
      line+=" $arg"
    else
      line+=" '${arg//\'/\'\\\'\'}'"
    fi
  done
  # A failed filter would hide what runs next; do not run it.
  printf '%s\n' "$line" | redact || return
  "$@"
}

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
