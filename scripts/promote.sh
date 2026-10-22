#!/usr/bin/env bash
# Promote recorded evidence into verification/final/, the committed deliverable (spec section 9).
#
#   scripts/promote.sh verify RUN_DIR   (make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>)
#   scripts/promote.sh demo DEMO_DIR    (make promote-demo RUN=verification/demo-<YYYY-MM-DDTHHMM-0700>)
#
# demo (Task 25): DEMO_DIR must be a scripts/demo.sh run, verification/demo-<YYYY-MM-DDTHHMM±zzzz>/ directly
# under verification/, holding only regular files named chN.txt (N 1-8; at least one) and timings.txt. A run
# whose timings.txt records a chapter with a failed step, or an interrupted chapter, is refused (exit 2): its
# recording is a fallback that would replay the failure; PROMOTE_ALLOW_FAIL=1 promotes it with a warning
# naming those chapters. It is copied, checked again, scanned and IP-checked exactly as a verify run is
# (below), and project-checked (see check_project), and the clean copy replaces verification/final/demo/
# (the recordings make demo-replay plays), the rest of final/ untouched; a missing final/ is created. A
# verify promotion keeps an existing final/demo/: it is copied into the staged copy, checked by the demo
# rules, scanned, IP-checked and project-checked with it, and swapped in together with it (a final/demo/
# that fails any of that fails the promotion, exit 1). So promoting a demo before or after the final verify
# run gives the same final/ (progress note for Task 25: verify replaces all of final/ otherwise). Only
# final/demo/ is exempt from the flat-run rule, and only by these demo rules.
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
# must not contain it (as a whole address, between non-digits) either (Task 24 Step 4). Nor may it
# contain the GCP project id (spec section 6: no project id in committed evidence): neither GCP_PROJECT
# from the environment nor the one this repo's .env gives (read with load_env in a subshell, so nothing
# else in .env reaches this script; a .env that load_env refuses is refused here too). Both are looked
# for: verify.sh's load_env sources .env over the environment, so the evidence holds .env's id even when
# the shell exports another one (X0-1). Each must be a GCP project id (_gcp_project_ok). An id is looked
# for as a fixed string, in any case, anywhere in a file: it also sits inside <id>-k3s-tfstate and in GCE
# hostnames such as k3s-cp.<zone>.c.<id>.internal. The ids are never printed, nor put on a command line
# (grep reads them from a pipe; argv shows in ps and in audit logs); the files that hold one are named.
# Without an id (neither sets GCP_PROJECT) that check is skipped with a warning, as the operator-IP check
# is without OPERATOR_IP. Only a clean copy replaces
# verification/final/: the old final is moved aside, the copy renamed into its place, and the old one
# removed (not merged into). Interrupts are ignored during that swap, and on any failure the old final is
# moved back, so verification/final is always either the previous final or the whole scanned copy.
# A lock (verification/.promote.lock) keeps two promotions from interleaving.
#
# Exit status: 0 promoted, 1 failed (gitleaks found a secret or failed, the operator IP or the GCP
# project id was found, the copy was not a passing verify run, the copy or swap failed, another
# promotion holds the lock), 2 refused (usage, an OPERATOR_IP that is not one IPv4 address, a
# GCP_PROJECT in the environment or in .env that is not a GCP project id, a .env that load_env refuses,
# a RUN_DIR that is not a finished verify run in which every check passed).
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
# The directory name scripts/demo.sh gives a run, and the files it writes there (Task 25).
DEMO_RUN_NAME='^demo-[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{4}[+-][0-9]{4}$'
DEMO_NAME='^(ch[1-8]\.txt|timings\.txt)$'
# The subdirectories of verification/final a verify promotion keeps (each checked by check_demo).
KEPT_IN_FINAL=(demo)
# A chapter's line of timings.txt as scripts/demo.sh writes it; BASH_REMATCH[2] is its failed steps.
DEMO_TIMING_LINE='^ch([1-8]) [0-9]+s \(commands [0-9]+s; [0-9]+ step\(s\), ([0-9]+) failed, '
DEMO_INTERRUPTED_LINE='^ch([1-8]) interrupted after '
# One dotted-quad IPv4 address, no leading zeros (gpuctl's ipaddress module rejects them too).
OCTET='(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
IPV4="^$OCTET\\.$OCTET\\.$OCTET\\.$OCTET\$"

usage() {
  cat <<'EOF'
usage: scripts/promote.sh verify RUN_DIR
       (make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>)
       scripts/promote.sh demo DEMO_DIR
       (make promote-demo RUN=verification/demo-<YYYY-MM-DDTHHMM-0700>)

  verify RUN_DIR   copy a finished verify run to verification/final/ (replacing the old one, but keeping
                   final/demo/) after gitleaks finds no secrets in the copy
  demo DEMO_DIR    copy a demo run (scripts/demo.sh) to verification/final/demo/ (replacing the old one)
                   after gitleaks finds no secrets in the copy, nor the GCP project id (GCP_PROJECT from
                   the environment, else .env; any case); PROMOTE_ALLOW_FAIL=1 promotes a run with a
                   failed or interrupted chapter (warned)

  PROMOTE_ALLOW_SKIP=1  promote a run with skipped (never failed) checks, noted in final's SUMMARY.md
  OPERATOR_IP=<ip>      also fail when the copy contains this IPv4 address (Task 24 Step 4)
  GCP_PROJECT=<id>      also fail when the copy contains this project id, in any case; GCP_PROJECT in .env
                        is looked for too (with neither, that check is skipped with a warning)
EOF
}

# Staging directory of the copy being scanned (and of the old final during the swap); removed on exit.
STAGE=""
LOCKED=0
# What swap_in replaces: verification/final (verify) or verification/final/demo (demo).
SWAP_TARGET="$FINAL_DIR"
_cleanup() {
  if [[ -n "$STAGE" ]]; then
    # A swap that stopped half way (target gone, the old one aside): put the previous one back first.
    if [[ ! -e "$SWAP_TARGET" && ! -L "$SWAP_TARGET" ]] && [[ -e "$STAGE/old" || -L "$STAGE/old" ]] \
      && ! mv -- "$STAGE/old" "$SWAP_TARGET"; then
      log "error: the previous ${SWAP_TARGET#"$REPO_ROOT"/} is in $STAGE/old; move it back to $SWAP_TARGET by hand"
    else
      rm -rf -- "$STAGE"
    fi
  fi
  if ((LOCKED)); then
    rm -f -- "$LOCK_FILE"
  fi
}

# resolve_run ARG [NAME_RE PATTERN KIND]: print the physical path of the run ARG names, or die (2) saying why
# not. A verify run by default; a demo run with $DEMO_RUN_NAME demo-<YYYY-MM-DDTHHMM-0700> demo.
resolve_run() {
  local arg="$1" name_re="${2:-$VERIFY_RUN_NAME}" pattern="${3:-<YYYY-MM-DDTHHMM-0700>}" kind="${4:-verify}"
  local run vdir name
  [[ -d "$VERIFICATION_DIR" && ! -L "$VERIFICATION_DIR" ]] \
    || die "$VERIFICATION_DIR is not a directory of this repo (missing or a symlink)" 2
  vdir="$(cd -P -- "$VERIFICATION_DIR" && pwd -P)" || die "cannot enter $VERIFICATION_DIR" 2
  [[ -d "$arg" ]] || die "$arg is not a directory (expected verification/$pattern)" 2
  run="$(cd -P -- "$arg" && pwd -P)" || die "cannot enter $arg" 2
  [[ "${run%/*}" == "$vdir" ]] \
    || die "$arg is not a run directly under $vdir/ (resolves to $run)" 2
  name="${run##*/}"
  [[ "$name" != final ]] || die "$arg is verification/final itself; name the $kind run to promote" 2
  [[ "$name" =~ $name_re ]] \
    || die "$arg ($name) is not named like a $kind run (verification/$pattern)" 2
  printf '%s\n' "$run"
}

# check_demo DIR WHAT CODE: die (CODE) unless DIR is a demo run as scripts/demo.sh writes it: only regular
# files (no subdirectory, symlink or dotfile) named chN.txt (N 1-8) or timings.txt, at least one chapter and
# the timings. WHAT names DIR in messages.
check_demo() {
  local dir="$1" what="$2" code="$3" entry bad="" chapters=0
  [[ -d "$dir" && ! -L "$dir" ]] || die "$what is not a directory (a symlink or missing)" "$code"
  for entry in "$dir"/*; do
    if [[ -L "$entry" || ! -f "$entry" || ! "${entry##*/}" =~ $DEMO_NAME ]]; then
      bad+=" ${entry##*/}"
    elif [[ "${entry##*/}" == ch* ]]; then
      chapters=$((chapters + 1))
    fi
  done
  [[ -z "$bad" ]] \
    || die "$what holds entries scripts/demo.sh does not write (only regular files named chN.txt, N 1-8, or timings.txt):$bad" "$code"
  ((chapters > 0)) || die "$what holds no chapter recording (chN.txt); is it a scripts/demo.sh run?" "$code"
  [[ -f "$dir/timings.txt" ]] || die "$what has no timings.txt; is it a finished scripts/demo.sh run?" "$code"
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

# project_ids: print, one per line, the GCP project ids the copy must not contain: GCP_PROJECT from the
# environment, and the one load_env gives when $REPO_ROOT/.env exists (what verify.sh sees: .env's
# GCP_PROJECT, sourced over the environment's); each once, nothing when neither sets one. Callers run it
# in $(...), so .env's other values never reach promote.sh's commands. load_env's refusals (a .env that
# is not 600) exit 2. An id that is not a GCP project id dies (2) without printing it: an id the check
# cannot match must not pass.
project_ids() {
  local env_id="${GCP_PROJECT:-}" dotenv_id=""
  if [[ -n "$env_id" ]] && ! _gcp_project_ok; then
    die "GCP_PROJECT in the environment is not a GCP project id (the value is not printed); fix it, or unset it" 2
  fi
  if [[ -f "$REPO_ROOT/.env" ]]; then
    load_env
    dotenv_id="${GCP_PROJECT:-}"
    if [[ -n "$dotenv_id" ]] && ! _gcp_project_ok; then
      die "GCP_PROJECT in $REPO_ROOT/.env is not a GCP project id (the value is not printed); fix it there, or leave it unset" 2
    fi
  fi
  [[ -z "$env_id" ]] || printf '%s\n' "$env_id"
  [[ -z "$dotenv_id" || "$dotenv_id" == "$env_id" ]] || printf '%s\n' "$dotenv_id"
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

# swap_in COPY [TARGET]: replace TARGET (default verification/final) with COPY. Interrupts are ignored
# (also by mv, which inherits it) so the two renames finish together; a failure moves the previous one back.
swap_in() {
  local copy="$1" target="${2:-$FINAL_DIR}" rc=0
  SWAP_TARGET="$target"
  trap '' INT TERM HUP
  if [[ -e "$target" || -L "$target" ]]; then
    mv -- "$target" "$STAGE/old" || rc=1
  fi
  # Without the lock a concurrent promotion could have recreated the target; mv would then nest the copy in it.
  if ((rc == 0)) && [[ ! -e "$target" && ! -L "$target" ]]; then
    mv -- "$copy" "$target" || rc=1
  else
    rc=1
  fi
  if ((rc != 0)) && [[ ! -e "$target" && ! -L "$target" ]] && [[ -e "$STAGE/old" || -L "$STAGE/old" ]]; then
    mv -- "$STAGE/old" "$target" || log "error: the previous ${target#"$REPO_ROOT"/} is in $STAGE/old; restore it by hand"
  fi
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP
  return "$rc"
}

# check_ip DIR IP WHERE: exit 1 when the operator IP (non-empty IP) is in a file of DIR, a flat copy (so the
# names grep prints are its file names); WHERE says where to redact it. The address is matched whole, between
# non-digits (73.15.40.4 is not in 73.15.40.40 or 173.15.40.4); ip is a validated dotted quad, so escaping
# its dots makes it a literal. LC_ALL=C: [^0-9] matches any byte.
check_ip() {
  local dir="$1" ip="$2" where="$3" ip_files rc
  [[ -n "$ip" ]] || return 0
  ip_files="$(cd -- "$dir" && LC_ALL=C grep -lE -e "(^|[^0-9])${ip//./\\.}([^0-9]|\$)" ./*)" && rc=0 || rc=$?
  ip_files="${ip_files//.\//}"
  if ((rc != 1)); then
    log "error: the operator IP (OPERATOR_IP) appears in (or grep failed on): ${ip_files//$'\n'/ }"
    log "  redact it in $where and promote again; verification/final was not changed"
    exit 1
  fi
}

# project_for_check: print the GCP project id a demo copy (and a kept final/demo/) must not contain:
# GCP_PROJECT from the environment, else from $REPO_ROOT/.env (load_env; callers run it in $(...), so .env's
# other values never reach promote.sh's commands); nothing when neither sets one. load_env's refusals (a .env
# that is not 600) exit 2, and so does an id that is not a GCP project id (not printed): an id the check
# cannot match must not pass. (The same rule as open PR minors-1's project_id for the verify copy; whichever
# merges second keeps one of the two.)
project_for_check() {
  local id="${GCP_PROJECT:-}"
  if [[ -z "$id" && -f "$REPO_ROOT/.env" ]]; then
    load_env
    id="${GCP_PROJECT:-}"
  fi
  if [[ -n "$id" ]] && ! GCP_PROJECT="$id" _gcp_project_ok; then
    die "GCP_PROJECT is not a GCP project id (the value is not printed); fix it in the environment or in $REPO_ROOT/.env, or leave it unset" 2
  fi
  printf '%s' "$id"
}

# check_project DIR ID WHERE: exit 1 when the GCP project id (non-empty ID) is in a file of DIR, a flat copy;
# WHERE says where to redact it. A fixed string (-F), in any case (-i), anywhere in a line: the id also sits
# inside <id>-k3s-tfstate, service-account emails and FQDNs. LC_ALL=C: bytes that are not UTF-8 cannot hide it,
# and -i folds ASCII only, which is all a project id holds. grep prints file names only, never the id; the
# demo masks it as <GCP_PROJECT> (scripts/lib/screen.sh), which this does not match.
check_project() {
  local dir="$1" id="$2" where="$3" files rc
  [[ -n "$id" ]] || return 0
  files="$(cd -- "$dir" && LC_ALL=C grep -liF -e "$id" ./*)" && rc=0 || rc=$?
  files="${files//.\//}"
  if ((rc != 1)); then
    log "error: the GCP project id (GCP_PROJECT) appears in (or grep failed on): ${files//$'\n'/ }"
    log "  redact it in $where and promote again; verification/final was not changed"
    exit 1
  fi
}

# failed_chapters TIMINGS: the chapters timings.txt records with a failed step or as interrupted, each after a
# space ("" when none).
failed_chapters() {
  local line out=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ $DEMO_TIMING_LINE ]] && ((10#${BASH_REMATCH[2]} > 0)); then
      out+=" ch${BASH_REMATCH[1]} (${BASH_REMATCH[2]} failed)"
    elif [[ "$line" =~ $DEMO_INTERRUPTED_LINE ]]; then
      out+=" ch${BASH_REMATCH[1]} (interrupted)"
    fi
  done <"$1"
  printf '%s' "$out"
}

# carry_kept COPY IP: put a copy of each existing verification/final/<sub> of KEPT_IN_FINAL (final/demo/)
# into COPY, the staged verify copy that replaces final, so a verify promotion does not delete it. The copy is
# checked by the demo rules, scanned, IP-checked and project-checked as a demo run is; anything wrong exits 1
# (final unchanged). A final that is a symlink is not the repo's: nothing is carried from it.
carry_kept() {
  local copy="$1" ip="$2" sub src leaks project
  [[ -d "$FINAL_DIR" && ! -L "$FINAL_DIR" ]] || return 0
  for sub in "${KEPT_IN_FINAL[@]}"; do
    src="$FINAL_DIR/$sub"
    [[ -e "$src" || -L "$src" ]] || continue
    project="$(project_for_check)" || exit
    check_demo "$src" "verification/final/$sub" 1
    cp -R -- "$src" "$copy/$sub" || die "copying verification/final/$sub into the staged copy failed"
    check_demo "$copy/$sub" "the copy of verification/final/$sub" 1
    log "keeping verification/final/$sub: scanning its copy for secrets"
    leaks="$(scan_copy "$copy/$sub")"
    if [[ -n "$leaks" ]]; then
      log "error: gitleaks reported leaks (or failed) in verification/final/$sub:$leaks"
      log "  redact them (or promote the demo again) and promote again; verification/final was not changed"
      exit 1
    fi
    check_ip "$copy/$sub" "$ip" "verification/final/$sub/"
    check_project "$copy/$sub" "$project" "verification/final/$sub/"
  done
}

promote_verify() {
  (($# == 1)) || { usage >&2; die "usage: verify takes exactly one RUN_DIR" 2; }
  local run copy result leaks ip ip_files projects project_files rc skipped
  run="$(resolve_run "$1")" || exit
  check_run "$run" "$1" 2
  skipped="$RUN_SKIPPED"
  ip="$(operator_ip)" || exit
  projects="$(project_ids)" || exit
  if ((skipped > 0)); then
    log "WARNING: promoting $1 with $skipped skipped check(s) (PROMOTE_ALLOW_SKIP=1); final's SUMMARY.md notes it"
  fi
  if [[ -z "$projects" ]]; then
    log "WARNING: no GCP project id (GCP_PROJECT in the environment or .env): the project-id check was skipped"
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
    # The copy is flat (check_run), so the names grep prints are the run's file names. The address is
    # matched whole, between non-digits (73.15.40.4 is not in 73.15.40.40 or 173.15.40.4); ip is a
    # validated dotted quad, so escaping its dots makes it a literal. LC_ALL=C: [^0-9] matches any byte.
    ip_files="$(cd -- "$copy" && LC_ALL=C grep -lE -e "(^|[^0-9])${ip//./\\.}([^0-9]|\$)" ./*)" && rc=0 || rc=$?
    ip_files="${ip_files//.\//}"
    if ((rc != 1)); then
      log "error: the operator IP (OPERATOR_IP) appears in (or grep failed on): ${ip_files//$'\n'/ }"
      log "  redact it in $run/ and promote again; verification/final was not changed"
      exit 1
    fi
  fi
  if [[ -n "$projects" ]]; then
    # Fixed strings (-F), in any case (-i), anywhere in a line: an id is 6-30 characters and also sits
    # inside <id>-k3s-tfstate and FQDNs. LC_ALL=C: bytes that are not UTF-8 cannot hide it, and -i folds
    # ASCII only, which is all a project id holds. The ids reach grep as a pattern file on a pipe (-f;
    # printf is a builtin), never as arguments, and none is empty (an empty pattern matches every file).
    # grep prints file names only, never an id.
    project_files="$(cd -- "$copy" && LC_ALL=C grep -liF -f <(printf '%s\n' "$projects") ./*)" && rc=0 || rc=$?
    project_files="${project_files//.\//}"
    if ((rc != 1)); then
      log "error: the GCP project id (GCP_PROJECT, environment or .env) appears in (or grep failed on): ${project_files//$'\n'/ }"
      log "  redact it in $run/ and promote again; verification/final was not changed"
      exit 1
    fi
  fi

  # A promoted demo (final/demo/) stays: it goes into the copy, checked, scanned, IP- and project-checked.
  carry_kept "$copy" "$ip"
  swap_in "$copy" || die "replacing $FINAL_DIR with the scanned copy failed; verification/final was not changed"

  result="$(grep -m 1 '^\*\*Result:\*\* ' -- "$FINAL_DIR/SUMMARY.md")"
  log "promoted ${run#"$REPO_ROOT"/} to verification/final ($result; gitleaks found no leaks)"
  log "before git add verification/final, check that the operator IP is not in it (Task 24 Step 4):"
  # shellcheck disable=SC2016 # the command is printed for the operator to run, not expanded here
  log '  grep -r "$(curl -s https://checkip.amazonaws.com)" verification/final   # must print nothing'
}

# promote_demo DEMO_DIR: verification/final/demo/ = a scanned copy of the demo run (see the header).
promote_demo() {
  (($# == 1)) || { usage >&2; die "usage: demo takes exactly one DEMO_DIR" 2; }
  local run copy leaks ip project failed
  run="$(resolve_run "$1" "$DEMO_RUN_NAME" "demo-<YYYY-MM-DDTHHMM-0700>" demo)" || exit
  check_demo "$run" "$1" 2
  # A failed or interrupted chapter's recording would replay the failure: the fallback would not work.
  failed="$(failed_chapters "$run/timings.txt")"
  if [[ -n "$failed" ]]; then
    if [[ "${PROMOTE_ALLOW_FAIL:-}" != 1 ]]; then
      log "error: $1 has chapters that failed or were interrupted:$failed"
      log "  their recordings would replay the failure; promote a run where they passed, or PROMOTE_ALLOW_FAIL=1"
      exit 2
    fi
    log "WARNING: promoting $1 with chapters that failed or were interrupted (PROMOTE_ALLOW_FAIL=1):$failed"
  fi
  ip="$(operator_ip)" || exit
  project="$(project_for_check)" || exit
  if [[ -z "$project" ]]; then
    log "WARNING: no GCP project id (GCP_PROJECT in the environment or .env): the project-id check was skipped"
  fi
  if [[ -L "$FINAL_DIR" || (-e "$FINAL_DIR" && ! -d "$FINAL_DIR") ]]; then
    die "$FINAL_DIR is not a directory of this repo (a symlink or a file); nothing was promoted" 2
  fi
  command -v gitleaks >/dev/null 2>&1 \
    || die "gitleaks is not installed; a demo run is promoted only after a secret scan (Task 1 tooling)"
  [[ -f "$REPO_ROOT/.gitleaks.toml" ]] || die "$REPO_ROOT/.gitleaks.toml not found"

  trap _cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP
  take_lock
  STAGE="$(mktemp -d "$VERIFICATION_DIR/.promote-XXXXXX")" || die "cannot create a staging directory"
  copy="$STAGE/demo"
  cp -R -- "$run" "$copy" || die "copying $run to $copy failed"
  check_demo "$copy" "the copy of $1" 1

  log "scanning each file of the copy of ${run#"$REPO_ROOT"/} for secrets (gitleaks, $REPO_ROOT/.gitleaks.toml)"
  leaks="$(scan_copy "$copy")"
  if [[ -n "$leaks" ]]; then
    log "error: gitleaks reported leaks (or failed) in:$leaks"
    log "  these are the same files in $run/; redact them and promote again"
    log "  verification/final was not changed"
    exit 1
  fi
  check_ip "$copy" "$ip" "$run/"
  check_project "$copy" "$project" "$run/"

  if [[ ! -d "$FINAL_DIR" ]]; then
    mkdir -- "$FINAL_DIR" || die "cannot create $FINAL_DIR"
  fi
  swap_in "$copy" "$FINAL_DIR/demo" \
    || die "replacing $FINAL_DIR/demo with the scanned copy failed; verification/final/demo was not changed"
  log "promoted ${run#"$REPO_ROOT"/} to verification/final/demo (gitleaks found no leaks); make demo-replay CHAPTER=n plays it"
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
    demo)
      shift
      promote_demo "$@"
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
