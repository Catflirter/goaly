# shellcheck shell=bash
# Verification harness shared by scripts/verify.sh and the demo (spec section 9). Source it; do not run it.
#
#   vr_init DIR                     start a run: DIR/SUMMARY.md with an empty results table
#   run_check ID TITLE FUNC [ARG…]  run one check into DIR/<NN>-<ID>.txt; FUNC returns 0 PASS, 77 SKIP, else FAIL
#   vr_cmd CMD…                     print "$ CMD" (redacted), then run CMD; returns CMD's status
#   redact                          stdin -> stdout filter that replaces secret values with <redacted>
#   vr_finish                       write the totals, scan the run with gitleaks; 1 on any FAIL or leak
#
# The check functions themselves (A–E) are added by later tasks and registered in scripts/verify.sh.

# common.sh is linted on its own; pre-commit runs shellcheck without -x, so don't follow it.
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

VR_TZ=America/Los_Angeles

# --- redact -------------------------------------------------------------------------------------

# A secret value: everything up to whitespace, a quote or a comma (the end of a YAML/JSON/shell value).
_vr_value='[^[:space:]"'\'',]+'
# Portable ERE (GNU and BSD sed): no \b, no I flag, so case-insensitive words are spelled as [Xx] classes.
_VR_REDACT_EXPRS=(
  # k3s join token K10<ca-hash>::<user>:<password>
  -e "s/K10[0-9a-fA-F]{64}::${_vr_value}/<redacted>/g"
  # kubeconfig credentials and service-account / bootstrap tokens (YAML and JSON forms)
  -e "s/((client-key-data|client-certificate-data|certificate-authority-data|token)\"?:[[:space:]]*\"?)${_vr_value}/\\1<redacted>/g"
  # Lambda API key in env form
  -e "s/(LAMBDA_API_KEY=[\"']?)${_vr_value}/\\1<redacted>/g"
  # HTTP credentials (curl -v prints HTTP/2 header names in lower case)
  -e "s/([Aa][Uu][Tt][Hh][Oo][Rr][Ii][Zz][Aa][Tt][Ii][Oo][Nn]:[[:space:]]*([Bb][Ee][Aa][Rr][Ee][Rr]|[Bb][Aa][Ss][Ii][Cc])[[:space:]]+)${_vr_value}/\\1<redacted>/g"
  # WireGuard / TLS private keys: "private key: …", "PrivateKey = …", any case
  -e "s/([Pp][Rr][Ii][Vv][Aa][Tt][Ee][[:space:]_-]?[Kk][Ee][Yy]\"?[[:space:]]*[:=][[:space:]]*\"?)${_vr_value}/\\1<redacted>/g"
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

redact() {
  "${_VR_SED[@]}" -E "${_VR_REDACT_EXPRS[@]}"
}

# --- run bookkeeping ----------------------------------------------------------------------------

# ISO 8601 local time in America/Los_Angeles, e.g. 2026-10-21T14:30:05-07:00.
_vr_now() {
  local ts
  ts="$(TZ="$VR_TZ" date +%Y-%m-%dT%H:%M:%S%z)"
  printf '%s:%s\n' "${ts:0:${#ts}-2}" "${ts: -2}"
}

# vr_init DIR: create DIR and its SUMMARY.md. Refuses a DIR that already holds files, so the
# evidence of two runs is never mixed.
vr_init() {
  local dir="$1"
  if [[ -d "$dir" && -n "$(ls -A -- "$dir")" ]]; then
    log "error: $dir already holds files from another run; start again in a minute (new directory name)"
    return 2
  fi
  mkdir -p -- "$dir"
  VR_DIR="$dir"
  VR_SEQ=0
  VR_FAILS=0
  VR_SKIPS=0
  {
    printf '# Verification run\n\n'
    printf 'Started: %s (%s)\n\n' "$(_vr_now)" "$VR_TZ"
    printf '| # | ID | Check | Result |\n'
    printf '|---|---|---|---|\n'
  } >"$VR_DIR/SUMMARY.md"
}

# run_check ID TITLE FUNC [ARGS…]: run FUNC ARGS in a subshell (so exit and set -e failures end only
# the check) with stdin from /dev/null, stdout+stderr redacted into $VR_DIR/<NN>-<ID>.txt.
# FUNC's status: 0 PASS, 77 SKIP, anything else FAIL. Always returns 0; vr_finish reports the outcome.
run_check() {
  local id="$1" title="$2" func="$3"
  shift 3
  local seq file rc result errexit=0
  VR_SEQ=$((VR_SEQ + 1))
  printf -v seq '%02d' "$VR_SEQ"
  file="$VR_DIR/$seq-$id.txt"
  {
    printf '# %s — %s\n' "$id" "$title"
    printf '# started: %s\n' "$(_vr_now)"
  } >"$file"

  # The check keeps the caller's errexit; run_check itself must survive the check failing.
  [[ $- == *e* ]] && errexit=1
  set +e
  (
    if ((errexit)); then set -e; fi
    "$func" "$@"
  ) </dev/null 2>&1 | redact >>"$file"
  rc=${PIPESTATUS[0]}
  if ((errexit)); then set -e; fi

  case "$rc" in
    0) result=PASS ;;
    77) result=SKIP; VR_SKIPS=$((VR_SKIPS + 1)) ;;
    *) result=FAIL; VR_FAILS=$((VR_FAILS + 1)) ;;
  esac
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
  printf '%s\n' "$line" | redact
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
