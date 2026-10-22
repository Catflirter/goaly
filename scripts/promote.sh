#!/usr/bin/env bash
# Promote recorded evidence into verification/final/, the committed deliverable (spec section 9).
#
#   scripts/promote.sh verify RUN_DIR   (make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>)
#
# verify: RUN_DIR must be a finished scripts/verify.sh run, verification/<YYYY-MM-DDTHHMM±zzzz>/ directly
# under this repo's verification/ (after resolving symlinks and ..), never verification/final itself.
#   - Its SUMMARY.md is one verify.sh finished (its **Result:** line) with no failed check: final is the
#     run where every check passed (spec section 9, Task 24 Step 3). PROMOTE_ALLOW_FAIL=1 promotes a
#     run with failed checks anyway, with a warning; skipped checks only warn.
#   - It holds only what verify.sh writes: regular files directly in it named SUMMARY.md, NN-<ID>.txt or
#     <name>.log (the autoscaler's log, spec section 10.5). No subdirectories, dotfiles, symlinks or other
#     names: gitleaks' default path allowlist (.gitleaks.toml extends the defaults) skips images, .git,
#     vendor/, node_modules/ and more, so such a file would reach final/ unscanned.
# It is copied to a staging directory inside verification/ and every file of the copy is scanned on its
# own with gitleaks and the repo's .gitleaks.toml, through stdin so that no path (the file's or the
# repo's location) can match an allowlist, and with inline gitleaks:allow markers ignored. When
# OPERATOR_IP is set, the copy must not contain it either (Task 24 Step 4). Only a clean copy replaces
# verification/final/: the old final is moved aside, the copy renamed into its place, and the old one
# removed (not merged into). Interrupts are ignored during that swap, and on any failure the old final is
# moved back, so verification/final is always either the previous final or the whole scanned copy.
# A lock (verification/.promote.lock) keeps two promotions from interleaving.
#
# Exit status: 0 promoted, 1 failed (gitleaks found a secret or failed, the operator IP was found, the
# copy or swap failed, another promotion holds the lock), 2 refused (usage, a RUN_DIR that is not a
# finished, passing verify run).
set -euo pipefail

# common.sh is linted on its own; pre-commit runs shellcheck without -x, so don't follow it.
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

# A relative RUN_DIR is resolved from the current directory; CDPATH would make `cd` look elsewhere.
unset CDPATH
# "$dir"/* lists every entry of a run, dotfiles included, and nothing when it is empty.
shopt -s nullglob dotglob

VERIFICATION_DIR="$REPO_ROOT/verification"
FINAL_DIR="$VERIFICATION_DIR/final"
LOCK_FILE="$VERIFICATION_DIR/.promote.lock"
# The directory name scripts/verify.sh gives a run: TZ=America/Los_Angeles date +%Y-%m-%dT%H%M%z.
VERIFY_RUN_NAME='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{4}[+-][0-9]{4}$'
# The files a run may hold: SUMMARY.md, run_check's <NN>-<ID>.txt, the autoscaler's <name>.log. No dots
# inside a name, so none can match the default allowlist's unanchored gitleaks\.toml-style patterns.
EVIDENCE_NAME='^(SUMMARY\.md|[0-9]{2,}-[A-Za-z0-9]+\.txt|[a-z0-9-]+\.log)$'
RESULT_LINE='^\*\*Result:\*\* ([0-9]+) passed, ([0-9]+) failed, ([0-9]+) skipped$'

usage() {
  cat <<'EOF'
usage: scripts/promote.sh verify RUN_DIR
       (make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>)

  verify RUN_DIR   copy a finished verify run to verification/final/ (replacing the old one) after
                   gitleaks finds no secrets in the copy

  PROMOTE_ALLOW_FAIL=1  promote a run with failed checks (refused otherwise)
  OPERATOR_IP=<ip>      also fail when the copy contains this address (Task 24 Step 4)
EOF
}

# Staging directory of the copy being scanned (and of the old final during the swap); removed on exit.
STAGE=""
LOCKED=0
_cleanup() {
  if [[ -n "$STAGE" ]]; then
    # A swap that stopped half way (final gone, the old one aside): put the previous final back first.
    if [[ ! -e "$FINAL_DIR" && ! -L "$FINAL_DIR" ]] && [[ -e "$STAGE/old" || -L "$STAGE/old" ]] \
      && ! mv -- "$STAGE/old" "$FINAL_DIR"; then
      log "error: the previous final is in $STAGE/old; move it back to $FINAL_DIR by hand"
    else
      rm -rf -- "$STAGE"
    fi
  fi
  if ((LOCKED)); then
    rm -f -- "$LOCK_FILE"
  fi
}

# resolve_run ARG: print the physical path of the verify run ARG names, or die (2) saying why not.
resolve_run() {
  local arg="$1" run vdir name entry bad=""
  [[ -d "$VERIFICATION_DIR" && ! -L "$VERIFICATION_DIR" ]] \
    || die "$VERIFICATION_DIR is not a directory of this repo (missing or a symlink)" 2
  vdir="$(cd -P -- "$VERIFICATION_DIR" && pwd -P)" || die "cannot enter $VERIFICATION_DIR" 2
  [[ -d "$arg" ]] || die "$arg is not a directory (expected verification/<YYYY-MM-DDTHHMM-0700>)" 2
  run="$(cd -P -- "$arg" && pwd -P)" || die "cannot enter $arg" 2
  [[ "${run%/*}" == "$vdir" ]] \
    || die "$arg is not a run directly under $vdir/ (resolves to $run)" 2
  name="${run##*/}"
  [[ "$name" != final ]] || die "$arg is verification/final itself; name the verify run to promote" 2
  [[ "$name" =~ $VERIFY_RUN_NAME ]] \
    || die "$arg ($name) is not named like a verify run (verification/<YYYY-MM-DDTHHMM-0700>)" 2
  [[ -f "$run/SUMMARY.md" && ! -L "$run/SUMMARY.md" ]] \
    || die "$arg has no SUMMARY.md (a regular file); is it a scripts/verify.sh run?" 2
  [[ "$(head -n 1 -- "$run/SUMMARY.md")" == "# Verification run" ]] \
    || die "$arg/SUMMARY.md is not a verify run summary (first line is not '# Verification run')" 2
  grep -q '^\*\*Result:\*\* ' -- "$run/SUMMARY.md" \
    || die "$arg/SUMMARY.md has no **Result:** line: the run did not finish; run make verify again" 2
  for entry in "$run"/*; do
    if [[ -L "$entry" || ! -f "$entry" || ! "${entry##*/}" =~ $EVIDENCE_NAME ]]; then
      bad+=" ${entry##*/}"
    fi
  done
  [[ -z "$bad" ]] \
    || die "$arg holds entries verify.sh does not write (only regular files named SUMMARY.md, NN-<ID>.txt or <name>.log):$bad" 2
  printf '%s\n' "$run"
}

# check_result RUN ARG: refuse (2) a run with failed checks unless PROMOTE_ALLOW_FAIL=1; warn on skips.
check_result() {
  local run="$1" arg="$2" line failed skipped
  line="$(grep -m 1 '^\*\*Result:\*\* ' -- "$run/SUMMARY.md")"
  [[ "$line" =~ $RESULT_LINE ]] \
    || die "$arg/SUMMARY.md: cannot read its Result line ('$line'); expected '**Result:** N passed, N failed, N skipped'" 2
  failed="${BASH_REMATCH[2]}"
  skipped="${BASH_REMATCH[3]}"
  if ((failed > 0)); then
    [[ "${PROMOTE_ALLOW_FAIL:-0}" == 1 ]] \
      || die "$arg has $failed failed check(s): verification/final is the run where every check passed (spec section 9); fix them and run make verify again, or set PROMOTE_ALLOW_FAIL=1 to promote it anyway" 2
    log "warning: promoting $arg with $failed failed check(s) (PROMOTE_ALLOW_FAIL=1)"
  fi
  if ((skipped > 0)); then
    log "warning: $arg has $skipped skipped check(s); see its SUMMARY.md"
  fi
}

# take_lock: create verification/.promote.lock (noclobber: O_EXCL, atomic) holding this PID, or fail.
take_lock() {
  local holder
  if ! (set -C && printf '%s\n' "$$" >"$LOCK_FILE") 2>/dev/null; then
    holder="$(head -n 1 -- "$LOCK_FILE" 2>/dev/null || true)"
    die "another promotion holds $LOCK_FILE (pid ${holder:-unknown}); if none is running, remove that file and try again"
  fi
  LOCKED=1
}

# scan_copy DIR: scan every file of DIR with gitleaks; print the names that leaked (or failed to scan).
scan_copy() {
  local file found=""
  for file in "$1"/*; do
    if ! gitleaks stdin --config "$REPO_ROOT/.gitleaks.toml" --ignore-gitleaks-allow \
      --no-banner --no-color --redact --verbose --exit-code 1 <"$file" >&2; then
      log "gitleaks: ${file##*/}: leak (or scanner failure)"
      found+=" ${file##*/}"
    fi
  done
  printf '%s' "$found"
}

# swap_in COPY: replace verification/final with COPY. Interrupts are ignored (also by mv, which
# inherits it) so the two renames finish together; a failure moves the previous final back.
swap_in() {
  local copy="$1" rc=0
  trap '' INT TERM HUP
  if [[ -e "$FINAL_DIR" || -L "$FINAL_DIR" ]]; then
    mv -- "$FINAL_DIR" "$STAGE/old" || rc=1
  fi
  # Without the lock a concurrent promotion could have recreated final; mv would then nest the copy in it.
  if ((rc == 0)) && [[ ! -e "$FINAL_DIR" && ! -L "$FINAL_DIR" ]]; then
    mv -- "$copy" "$FINAL_DIR" || rc=1
  else
    rc=1
  fi
  if ((rc != 0)) && [[ ! -e "$FINAL_DIR" && ! -L "$FINAL_DIR" ]] && [[ -e "$STAGE/old" || -L "$STAGE/old" ]]; then
    mv -- "$STAGE/old" "$FINAL_DIR" || log "error: the previous final is in $STAGE/old; restore it by hand"
  fi
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP
  return "$rc"
}

promote_verify() {
  (($# == 1)) || { usage >&2; die "usage: verify takes exactly one RUN_DIR" 2; }
  local run copy result leaks ip_files rc
  run="$(resolve_run "$1")" || exit
  check_result "$run" "$1"
  command -v gitleaks >/dev/null 2>&1 \
    || die "gitleaks is not installed; a run is promoted only after a secret scan (Task 1 tooling)"
  [[ -f "$REPO_ROOT/.gitleaks.toml" ]] || die "$REPO_ROOT/.gitleaks.toml not found"

  trap _cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP
  take_lock
  # Stage inside verification/ (gitignored, same filesystem), so the clean copy is renamed into place.
  STAGE="$(mktemp -d "$VERIFICATION_DIR/.promote-XXXXXX")" || die "cannot create a staging directory"
  copy="$STAGE/final"
  cp -R -- "$run" "$copy" || die "copying $run to $copy failed"
  [[ -d "$copy" ]] || die "copying $run to $copy failed"

  log "scanning each file of the copy of ${run#"$REPO_ROOT"/} for secrets (gitleaks, $REPO_ROOT/.gitleaks.toml)"
  leaks="$(scan_copy "$copy")"
  if [[ -n "$leaks" ]]; then
    log "error: gitleaks reported leaks (or failed) in:$leaks"
    log "  these are the same files in $run/; redact them and promote again"
    log "  verification/final was not changed"
    exit 1
  fi
  if [[ -n "${OPERATOR_IP:-}" ]]; then
    # The run is flat (resolve_run), so the names grep prints are the run's file names.
    ip_files="$(cd -- "$copy" && grep -lF -e "$OPERATOR_IP" ./*)" && rc=0 || rc=$?
    ip_files="${ip_files//.\//}"
    if ((rc != 1)); then
      log "error: the operator IP (OPERATOR_IP) appears in (or grep failed on): ${ip_files//$'\n'/ }"
      log "  redact it in $run/ and promote again; verification/final was not changed"
      exit 1
    fi
  fi

  swap_in "$copy" || die "replacing $FINAL_DIR with the scanned copy failed; verification/final was not changed"

  result="$(grep -m 1 '^\*\*Result:\*\* ' -- "$FINAL_DIR/SUMMARY.md")"
  log "promoted ${run#"$REPO_ROOT"/} to verification/final ($result; gitleaks found no leaks)"
  log "before git add verification/final, check that the operator IP is not in it (Task 24 Step 4):"
  # shellcheck disable=SC2016 # the command is printed for the operator to run, not expanded here
  log '  grep -r "$(curl -s https://checkip.amazonaws.com)" verification/final   # must print nothing'
}

main() {
  case "${1:-}" in
    verify)
      shift
      promote_verify "$@"
      ;;
    -h | --help | help)
      usage
      ;;
    *)
      usage >&2
      die "usage: unknown or missing subcommand '${1:-}'" 2
      ;;
  esac
}

main "$@"
