#!/usr/bin/env bash
# Promote recorded evidence into verification/final/, the committed deliverable (spec section 9).
#
#   scripts/promote.sh verify RUN_DIR   (make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>)
#
# verify: RUN_DIR must be a finished scripts/verify.sh run, verification/<YYYY-MM-DDTHHMM±zzzz>/ directly
# under this repo's verification/ (after resolving symlinks and ..), never verification/final itself.
#   - Its SUMMARY.md is one verify.sh finished (its **Result:** line) in which every check passed: final
#     is the run where every check passed (spec section 9, Task 24 Step 3: A-E all PASS). At least one
#     check ran, none failed, none was skipped, and the table rows agree with the totals. A failed check
#     is always refused. PROMOTE_ALLOW_SKIP=1 promotes a run with skipped checks (and no failed one),
#     with a warning and a note appended to final's SUMMARY.md that records how many were skipped.
#   - It holds only what verify.sh writes: regular files directly in it named SUMMARY.md, NN-<ID>.txt or
#     <name>.log (the autoscaler's log, spec section 10.5). No subdirectories, dotfiles, symlinks or other
#     names: gitleaks' default path allowlist (.gitleaks.toml extends the defaults) skips images, .git,
#     vendor/, node_modules/ and more, so such a file would reach final/ unscanned.
# It is copied to a staging directory inside verification/, and the copy (what is scanned and promoted)
# is checked again by the same rules, so an entry that reached the run after its check fails the
# promotion. Every file of the copy is scanned on its own with gitleaks and the repo's .gitleaks.toml,
# through stdin so that no path (the file's or the repo's location) can match an allowlist, and with
# inline gitleaks:allow markers ignored; a file passes only when gitleaks exits 0, says "no leaks found"
# and logs no error or warning (gitleaks stdin exits 0 when it cannot read its input). When OPERATOR_IP
# is set (surrounding whitespace dropped, as gpuctl reads it; it must be one IPv4 address), the copy
# must not contain it either (Task 24 Step 4). Only a clean copy replaces
# verification/final/: the old final is moved aside, the copy renamed into its place, and the old one
# removed (not merged into). Interrupts are ignored during that swap, and on any failure the old final is
# moved back, so verification/final is always either the previous final or the whole scanned copy.
# A lock (verification/.promote.lock) keeps two promotions from interleaving.
#
# Exit status: 0 promoted, 1 failed (gitleaks found a secret or failed, the operator IP was found, the
# copy was not a passing verify run, the copy or swap failed, another promotion holds the lock),
# 2 refused (usage, an OPERATOR_IP that is not one IPv4 address, a RUN_DIR that is not a finished verify
# run in which every check passed).
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
# One table row of SUMMARY.md as run_check writes it: | <seq> | <ID> | <title> | PASS/FAIL/SKIP |
TABLE_ROW='^\| [0-9]+ \| '
# One dotted-quad IPv4 address, no leading zeros (gpuctl's ipaddress module rejects them too).
OCTET='(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
IPV4="^$OCTET\\.$OCTET\\.$OCTET\\.$OCTET\$"

usage() {
  cat <<'EOF'
usage: scripts/promote.sh verify RUN_DIR
       (make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>)

  verify RUN_DIR   copy a finished verify run to verification/final/ (replacing the old one) after
                   gitleaks finds no secrets in the copy

  PROMOTE_ALLOW_SKIP=1  promote a run with skipped (never failed) checks, noted in final's SUMMARY.md
  OPERATOR_IP=<ip>      also fail when the copy contains this IPv4 address (Task 24 Step 4)
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
  local arg="$1" run vdir name
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
  printf '%s\n' "$run"
}

# bad_entries DIR: print (each after a space) the entries of DIR a verify run never holds: anything
# but a regular file directly in DIR named SUMMARY.md, NN-<ID>.txt or <name>.log.
bad_entries() {
  local entry bad=""
  for entry in "$1"/*; do
    if [[ -L "$entry" || ! -f "$entry" || ! "${entry##*/}" =~ $EVIDENCE_NAME ]]; then
      bad+=" ${entry##*/}"
    fi
  done
  printf '%s' "$bad"
}

# count_rows FILE REGEX: how many lines of FILE match REGEX (grep -E), 0 included.
count_rows() {
  grep -cE -- "$2" "$1" || true
}

# check_run DIR WHAT CODE: die (CODE) unless DIR is a finished verify run in which every check passed:
# flat (bad_entries), SUMMARY.md written by verify.sh with a Result line of at least one passed check,
# no failed one and no skipped one (PROMOTE_ALLOW_SKIP=1 allows skips), and table rows that match the
# totals. WHAT names DIR in messages. Sets RUN_SKIPPED to the number of skipped checks.
RUN_SKIPPED=0
check_run() {
  local dir="$1" what="$2" code="$3" bad line passed failed skipped rows pass_rows fail_rows skip_rows
  [[ -f "$dir/SUMMARY.md" && ! -L "$dir/SUMMARY.md" ]] \
    || die "$what has no SUMMARY.md (a regular file); is it a scripts/verify.sh run?" "$code"
  [[ "$(head -n 1 -- "$dir/SUMMARY.md")" == "# Verification run" ]] \
    || die "$what/SUMMARY.md is not a verify run summary (first line is not '# Verification run')" "$code"
  line="$(grep -m 1 '^\*\*Result:\*\* ' -- "$dir/SUMMARY.md")" \
    || die "$what/SUMMARY.md has no **Result:** line: the run did not finish; run make verify again" "$code"
  bad="$(bad_entries "$dir")"
  [[ -z "$bad" ]] \
    || die "$what holds entries verify.sh does not write (only regular files named SUMMARY.md, NN-<ID>.txt or <name>.log):$bad" "$code"
  [[ "$line" =~ $RESULT_LINE ]] \
    || die "$what/SUMMARY.md: cannot read its Result line ('$line'); expected '**Result:** N passed, N failed, N skipped'" "$code"
  passed="${BASH_REMATCH[1]}"
  failed="${BASH_REMATCH[2]}"
  skipped="${BASH_REMATCH[3]}"
  ((10#$failed == 0)) \
    || die "$what has $failed failed check(s): verification/final is the run where every check passed (spec section 9, Task 24 Step 3); fix them and run make verify again" "$code"
  ((10#$passed > 0)) || die "$what/SUMMARY.md: no check passed (no check ran); nothing to promote" "$code"
  rows="$(count_rows "$dir/SUMMARY.md" "$TABLE_ROW")"
  pass_rows="$(count_rows "$dir/SUMMARY.md" "$TABLE_ROW.* \\| PASS \\|\$")"
  fail_rows="$(count_rows "$dir/SUMMARY.md" "$TABLE_ROW.* \\| FAIL \\|\$")"
  skip_rows="$(count_rows "$dir/SUMMARY.md" "$TABLE_ROW.* \\| SKIP \\|\$")"
  ((pass_rows == 10#$passed && fail_rows == 0 && skip_rows == 10#$skipped && rows == pass_rows + skip_rows)) \
    || die "$what/SUMMARY.md: its table ($pass_rows PASS, $fail_rows FAIL, $skip_rows SKIP of $rows rows) does not match its Result line ('$line')" "$code"
  if ((10#$skipped > 0)) && [[ "${PROMOTE_ALLOW_SKIP:-}" != 1 ]]; then
    die "$what has $skipped skipped check(s): verification/final is the run where every check passed (spec section 9, Task 24 Step 3: A-E all PASS); run make verify RESILIENCE=1 again, or set PROMOTE_ALLOW_SKIP=1 to promote it anyway (noted in final's SUMMARY.md)" "$code"
  fi
  RUN_SKIPPED=$((10#$skipped))
}

# operator_ip: print OPERATOR_IP without surrounding whitespace (nothing when unset or blank, as gpuctl
# reads it), or die (2) when it is not one IPv4 address: grep would then match nothing and pass.
operator_ip() {
  local ip="${OPERATOR_IP:-}"
  ip="${ip#"${ip%%[![:space:]]*}"}"
  ip="${ip%"${ip##*[![:space:]]}"}"
  if [[ -n "$ip" && ! "$ip" =~ $IPV4 ]]; then
    die "OPERATOR_IP is not one IPv4 address (like 203.0.113.7; the value is not printed); set it to the operator's public IP or leave it unset" 2
  fi
  printf '%s' "$ip"
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

# scan_file FILE: succeed only when FILE is a regular file and gitleaks read it, reported "no leaks
# found" and logged no error or warning. gitleaks stdin exits 0 when it cannot read its input (a
# directory: 'ERR error="could not read file: ... is a directory"'), so the exit status alone is not enough.
scan_file() {
  local file="$1" out rc=0
  [[ -f "$file" && ! -L "$file" ]] || { log "gitleaks: ${file##*/}: not a regular file; not scanned"; return 1; }
  out="$(gitleaks stdin --config "$REPO_ROOT/.gitleaks.toml" --ignore-gitleaks-allow \
    --no-banner --no-color --redact --verbose --exit-code 1 <"$file" 2>&1)" || rc=$?
  printf '%s\n' "$out" >&2
  ((rc == 0)) || return 1
  if grep -qE '(^|[[:space:]])(ERR|WRN|FTL|PNC)([[:space:]]|$)|could not read' <<<"$out"; then
    return 1
  fi
  grep -qE '(^|[[:space:]])INF no leaks found$' <<<"$out"
}

# scan_copy DIR: scan every entry of DIR (scan_file); print the names that leaked or did not scan cleanly.
scan_copy() {
  local file found=""
  for file in "$1"/*; do
    if ! scan_file "$file"; then
      log "gitleaks: ${file##*/}: leak, scanner error or unreadable file"
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
  local run copy result leaks ip ip_files rc skipped
  run="$(resolve_run "$1")" || exit
  check_run "$run" "$1" 2
  skipped="$RUN_SKIPPED"
  ip="$(operator_ip)" || exit
  if ((skipped > 0)); then
    log "WARNING: promoting $1 with $skipped skipped check(s) (PROMOTE_ALLOW_SKIP=1); final's SUMMARY.md notes it"
  fi
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
  [[ -d "$copy" && ! -L "$copy" ]] || die "copying $run to $copy failed"
  # What is scanned and promoted is the copy: check it again, so nothing that reached the run after its
  # check above (a subdirectory, symlink, fifo, other name, or a changed SUMMARY.md) is promoted.
  check_run "$copy" "the copy of $1" 1
  ((RUN_SKIPPED == skipped)) || die "the copy of $1 has $RUN_SKIPPED skipped check(s), the run had $skipped"
  if ((skipped > 0)); then
    printf '\n**Promoted with skipped checks:** %d skipped check(s) accepted with PROMOTE_ALLOW_SKIP=1 by scripts/promote.sh at %s\n' \
      "$skipped" "$(TZ=America/Los_Angeles date +%Y-%m-%dT%H:%M:%S%z)" >>"$copy/SUMMARY.md"
  fi

  log "scanning each file of the copy of ${run#"$REPO_ROOT"/} for secrets (gitleaks, $REPO_ROOT/.gitleaks.toml)"
  leaks="$(scan_copy "$copy")"
  if [[ -n "$leaks" ]]; then
    log "error: gitleaks reported leaks (or failed) in:$leaks"
    log "  these are the same files in $run/; redact them and promote again"
    log "  verification/final was not changed"
    exit 1
  fi
  if [[ -n "$ip" ]]; then
    # The copy is flat (check_run), so the names grep prints are the run's file names.
    ip_files="$(cd -- "$copy" && grep -lF -e "$ip" ./*)" && rc=0 || rc=$?
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
