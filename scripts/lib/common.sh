# shellcheck shell=bash
# Shared helpers for the operator scripts. Source it; do not run it.
# Minimal version (Task 7): REPO_ROOT, log, die, load_env. Task 10 adds the Terraform/SSH helpers.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"

log() {
  printf '%s\n' "$*" >&2
}

# die MESSAGE [CODE]: log MESSAGE and exit CODE (default 1). Codes follow gpuctl's: 1 failed, 2 refused.
die() {
  log "error: $1"
  exit "${2:-1}"
}

# Mode of a file (following symlinks) as octal digits; GNU stat first, then BSD/macOS stat.
_file_mode() {
  stat -L -c '%a' -- "$1" 2>/dev/null || stat -L -f '%Lp' -- "$1"
}

# Export every variable in $REPO_ROOT/.env. The file holds the Lambda API key, so it must be private (600).
# A missing or non-private .env exits 2 (refused), as gpuctl does when LAMBDA_API_KEY is missing.
load_env() {
  local env_file="$REPO_ROOT/.env" mode
  [[ -f "$env_file" ]] \
    || die "$env_file not found: cp .env.example .env && chmod 600 .env, then fill it in" 2
  mode="$(_file_mode "$env_file")" || die "cannot read the permissions of $env_file" 2
  [[ "$mode" == 600 ]] \
    || die "$env_file has mode $mode; it holds secrets and must be 600: chmod 600 $env_file" 2
  set -a
  # shellcheck source=/dev/null
  source "$env_file"
  set +a
}
