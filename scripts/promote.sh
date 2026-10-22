#!/usr/bin/env bash
# Promote recorded evidence into verification/final/, the committed deliverable (spec section 9).
#
#   scripts/promote.sh verify RUN_DIR   (make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>)
#
# verify: RUN_DIR must be a finished scripts/verify.sh run, verification/<YYYY-MM-DDTHHMM±zzzz>/ directly
# under this repo's verification/ (after resolving symlinks and ..), never verification/final itself, with
# a SUMMARY.md that verify.sh finished (its **Result:** line), and holding only regular files and
# directories. It is copied to a staging directory inside verification/ and the copy is scanned with
# gitleaks and the repo's .gitleaks.toml (a directory scan, no git history). Only a clean copy replaces
# verification/final/: the old final is removed (not merged into) and the copy renamed into its place.
# On a finding, or when gitleaks fails or is missing, the copy is deleted and the previous final is
# left as it was, so verification/final never holds an unscanned or leaking run.
#
# Exit status: 0 promoted, 1 failed (gitleaks found a secret or failed, the copy failed), 2 refused
# (usage, a RUN_DIR that is not a verify run).
set -euo pipefail

# common.sh is linted on its own; pre-commit runs shellcheck without -x, so don't follow it.
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

# A relative RUN_DIR is resolved from the current directory; CDPATH would make `cd` look elsewhere.
unset CDPATH

VERIFICATION_DIR="$REPO_ROOT/verification"
FINAL_DIR="$VERIFICATION_DIR/final"
# The directory name scripts/verify.sh gives a run: TZ=America/Los_Angeles date +%Y-%m-%dT%H%M%z.
VERIFY_RUN_NAME='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{4}[+-][0-9]{4}$'

usage() {
  cat <<'EOF'
usage: scripts/promote.sh verify RUN_DIR
       (make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>)

  verify RUN_DIR   copy a finished verify run to verification/final/ (replacing the old one) after
                   gitleaks finds no secrets in the copy
EOF
}

# Staging directory of the copy being scanned; removed on every exit.
STAGE=""
_cleanup() {
  if [[ -n "$STAGE" ]]; then
    rm -rf -- "$STAGE"
  fi
}

# resolve_run ARG: print the physical path of the verify run ARG names, or die (2) saying why not.
resolve_run() {
  local arg="$1" run vdir name special
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
  # gitleaks dir does not follow symlinks, so a linked file would reach final/ unscanned; verify.sh
  # writes only regular files.
  special="$(find "$run" ! -type f ! -type d -print)" || die "cannot list $arg" 2
  [[ -z "$special" ]] \
    || die "$arg holds entries that are not regular files or directories (symlinks?): ${special//$'\n'/ }" 2
  printf '%s\n' "$run"
}

promote_verify() {
  (($# == 1)) || { usage >&2; die "usage: verify takes exactly one RUN_DIR" 2; }
  local run copy result
  run="$(resolve_run "$1")" || exit
  command -v gitleaks >/dev/null 2>&1 \
    || die "gitleaks is not installed; a run is promoted only after a secret scan (Task 1 tooling)"
  [[ -f "$REPO_ROOT/.gitleaks.toml" ]] || die "$REPO_ROOT/.gitleaks.toml not found"

  # Stage inside verification/ (gitignored, same filesystem), so the clean copy is renamed into place.
  trap _cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP
  STAGE="$(mktemp -d "$VERIFICATION_DIR/.promote-XXXXXX")" || die "cannot create a staging directory"
  copy="$STAGE/final"
  cp -R -- "$run" "$copy" || die "copying $run to $copy failed"
  [[ -d "$copy" ]] || die "copying $run to $copy failed" # gitleaks reports no leaks for a missing path

  log "scanning the copy of ${run#"$REPO_ROOT"/} for secrets (gitleaks, $REPO_ROOT/.gitleaks.toml)"
  if ! gitleaks dir "$copy" --config "$REPO_ROOT/.gitleaks.toml" \
    --no-banner --no-color --redact --verbose --exit-code 1 >&2; then
    log "error: gitleaks reported leaks (or failed) in the copy of $run"
    log "  the files named above under $copy/ are the same files in $run/; redact them and promote again"
    log "  verification/final was not changed"
    exit 1
  fi

  rm -rf -- "$FINAL_DIR"
  [[ ! -e "$FINAL_DIR" && ! -L "$FINAL_DIR" ]] || die "cannot remove the old $FINAL_DIR"
  mv -- "$copy" "$FINAL_DIR" || die "moving the scanned copy to $FINAL_DIR failed"

  result="$(grep -m 1 '^\*\*Result:\*\* ' -- "$FINAL_DIR/SUMMARY.md")"
  log "promoted ${run#"$REPO_ROOT"/} to verification/final ($result; gitleaks found no leaks)"
  log "review verification/final, then: git add verification/final"
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
