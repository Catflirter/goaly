#!/usr/bin/env bash
# k3s-guard-peers: replace k3s-cp's k3s_peers set and persist it for k3s-guard.service (spec §3.3).
# server-bootstrap.sh installs it as /usr/local/sbin/k3s-guard-peers (root, 0755). gpuctl
# sync-peers (Task 13) runs it over IAP SSH with gpuctl.peers.nft_replace_script() on stdin:
#
#   sudo /usr/local/sbin/k3s-guard-peers <<'EOF'
#   flush set inet k3s_guard k3s_peers
#   add element inet k3s_guard k3s_peers { 10.10.0.2, 34.82.10.10, ... }
#   EOF
#
# Nothing but that exact full replacement (flush, then one non-empty element list) is accepted, so
# the persisted list is always the complete current list: a node removed by gpu-down/cpu-remove
# does not come back after a CP reboot or a guard reload. The script is applied to the live set
# first (nft runs it as one transaction); only if that succeeds is it written to
# /etc/k3s-guard/peers.nft (0644, temp file in the same directory, fsync, rename), which
# k3s-guard.service loads after every start and reload. Concurrent runs are serialised with flock.
#
# Exit codes: 1 nft rejected the script or persisting failed (peers.nft is then unchanged),
# 2 stdin is not an nft_replace_script() output (nothing applied or written).
# Test seam: K3S_BOOTSTRAP_TEST_ROOT (see node/common.sh) prefixes /etc and /run; tests only.
set -euo pipefail

log() { printf '[k3s-guard-peers] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

HOST=${K3S_BOOTSTRAP_TEST_ROOT:-}
DIR="$HOST/etc/k3s-guard"
PEERS="$DIR/peers.nft"
LOCK="$HOST/run/lock/k3s-guard-peers.lock"
if [[ -z $HOST ]]; then
  (( EUID == 0 )) || die "run as root (sudo)"
fi

# Canonical dotted quads (no leading zeros, which nft would read as octal); nft range-checks them.
octet='(0|[1-9][0-9]{0,2})'
ip="${octet}\\.${octet}\\.${octet}\\.${octet}"
nl=$'\n'
shape="^flush set inet k3s_guard k3s_peers${nl}add element inet k3s_guard k3s_peers \\{ ${ip}(, ${ip})* \\}${nl}?\$"

# The trailing x keeps trailing newlines through $(...).
input=$(cat && printf x) || die "cannot read stdin"
input=${input%x}
if ! [[ $input =~ $shape ]]; then
  log "ERROR: stdin is not an nft_replace_script() output: \"flush set inet k3s_guard k3s_peers\"" \
      "then one \"add element inet k3s_guard k3s_peers { IP, ... }\" line; nothing applied"
  exit 2
fi

mkdir -p -- "$DIR" "${LOCK%/*}"
exec 9> "$LOCK"
flock 9

tmp=$(mktemp -- "$DIR/.peers.nft.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
printf '%s\n' "${input%"$nl"}" > "$tmp"
chmod 0644 -- "$tmp"
nft -f "$tmp" || die "nft rejected the new k3s_peers list; live set and $PEERS left unchanged"
sync -- "$tmp"
mv -f -- "$tmp" "$PEERS"
sync -- "$DIR"
count=$(grep -o ', ' <<< "$input" | wc -l)
log "k3s_peers replaced ($((count + 1)) addresses) and persisted to $PEERS"
