#!/usr/bin/env bash
# Control-plane bootstrap for k3s-cp (spec §2.1, §2.3 #3, §4.2, §4.3, §5.1). Idempotent; run as
# root over IAP SSH with node/ copied alongside:
#
#   server-bootstrap.sh --cp-ip IP --cp-internal-ip IP [--underlay-mtu 1460]
#                       [--k3s-version v1.36.5+k3s1] [--iface IF] [--render-only DIR]
#
# --render-only DIR runs the argument checks and the MTU preflight, renders every file under DIR
# instead of /, and changes nothing on the host.
# Exit codes: 2 usage, 3 MTU below --underlay-mtu, 4 unresolved template placeholder.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# common.sh is linted on its own; pre-commit runs shellcheck without -x, so don't follow it.
# shellcheck disable=SC1091
source "$HERE/common.sh"

usage() {
  log "usage: ${0##*/} --cp-ip IP --cp-internal-ip IP [--underlay-mtu 1460]" \
      "[--k3s-version v1.36.5+k3s1] [--iface IF] [--render-only DIR]"
  [[ $# -eq 0 ]] || log "ERROR: $*"
  exit 2
}

CP_IP="" CP_INTERNAL_IP="" UNDERLAY_MTU=1460 K3S_VERSION=v1.36.5+k3s1 IFACE=""
RENDER_ONLY=0 RENDER_DIR=""
while (( $# )); do
  case $1 in
    --cp-ip | --cp-internal-ip | --underlay-mtu | --k3s-version | --iface | --render-only)
      (( $# >= 2 )) || usage "$1 needs a value"
      case $1 in
        --cp-ip) CP_IP=$2 ;;
        --cp-internal-ip) CP_INTERNAL_IP=$2 ;;
        --underlay-mtu) UNDERLAY_MTU=$2 ;;
        --k3s-version) K3S_VERSION=$2 ;;
        --iface) IFACE=$2 ;;
        --render-only) RENDER_ONLY=1 RENDER_DIR=$2 ;;
      esac
      shift 2 ;;
    -h | --help) usage ;;
    *) usage "unknown argument: $1" ;;
  esac
done

[[ -n $CP_IP ]] || usage "--cp-ip is required"
[[ -n $CP_INTERNAL_IP ]] || usage "--cp-internal-ip is required"
valid_ipv4 "$CP_IP" || usage "--cp-ip is not an IPv4 address: $CP_IP"
valid_ipv4 "$CP_INTERNAL_IP" || usage "--cp-internal-ip is not an IPv4 address: $CP_INTERNAL_IP"
[[ $UNDERLAY_MTU =~ ^[1-9][0-9]{2,3}$ ]] || usage "--underlay-mtu must be an MTU in bytes (100-9999): $UNDERLAY_MTU"
[[ $K3S_VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+\+k3s[0-9]+$ ]] || usage "bad --k3s-version: $K3S_VERSION"
[[ -z $IFACE || $IFACE =~ ^[A-Za-z0-9_.:-]+$ ]] || usage "bad --iface: $IFACE"
(( ! RENDER_ONLY )) || [[ -n $RENDER_DIR ]] || usage "--render-only needs a directory"

if [[ -z $IFACE ]]; then IFACE=$(default_iface); fi
mtu_preflight "$IFACE" "$UNDERLAY_MTU"

# k3s_peers seed: the CP's own internal and public addresses (hairpin, spec §2.3 #1), sorted
# numerically. sync-peers adds the workers at runtime.
PEERS=$(printf '%s\n' "$CP_INTERNAL_IP" "$CP_IP" | sort -u -t. -k1,1n -k2,2n -k3,3n -k4,4n | paste -sd' ' -)
PEERS=${PEERS// /, }

K3S_FILES=(etc/rancher/k3s/config.yaml etc/rancher/k3s/psa.yaml etc/rancher/k3s/audit-policy.yaml)
PEERS_HELPER=usr/local/sbin/k3s-guard-peers

render_all() {
  local root=$1
  render_template "$HERE/k3s/server-config.yaml.tmpl" "$root/etc/rancher/k3s/config.yaml" CP_IP
  render_template "$HERE/k3s/psa.yaml" "$root/etc/rancher/k3s/psa.yaml"
  render_template "$HERE/k3s/audit-policy.yaml" "$root/etc/rancher/k3s/audit-policy.yaml"
  render_template "$HERE/k3s/90-kubelet.conf" "$root/etc/sysctl.d/90-kubelet.conf"
  render_template "$HERE/guard/k3s-guard-server.nft.tmpl" "$root/etc/k3s-guard/k3s-guard.nft" PEERS
  render_template "$HERE/guard/k3s-guard.service" "$root/etc/systemd/system/k3s-guard.service"
  render_template "$HERE/k3s/10-guard.conf" "$root/etc/systemd/system/k3s.service.d/10-guard.conf"
  # sync-peers' entry point on this host (Ruling R14): replaces k3s_peers and persists it in
  # peers.nft.
  render_template "$HERE/guard/k3s-guard-peers.sh" "$root/$PEERS_HELPER"
}

if (( RENDER_ONLY )); then
  render_all "$RENDER_DIR"
  log "rendered into $RENDER_DIR (render-only: nothing on this host was changed)"
  exit 0
fi

# ---- Everything below modifies the host. ------------------------------------------------------
# HOST is "" (i.e. /) on a node; the offline tests point K3S_BOOTSTRAP_TEST_ROOT at a scratch tree
# and put stub binaries first on PATH (common.sh).
HOST=${K3S_BOOTSTRAP_TEST_ROOT:-}
# How long to wait for the apiserver's /readyz (seconds). Test mode may shorten it with
# READYZ_TIMEOUT, so the offline tests reach the deadline; on a node it is always 180.
READYZ_WAIT=180
if [[ -n $HOST ]]; then
  log "WARNING: K3S_BOOTSTRAP_TEST_ROOT=$HOST: test mode, host paths are prefixed with it"
  READYZ_WAIT=${READYZ_TIMEOUT:-180}
  # No leading zero: $(( )) reads 010 as octal 8 and fails on 08.
  [[ $READYZ_WAIT =~ ^(0|[1-9][0-9]*)$ ]] || die "READYZ_TIMEOUT must be a number of seconds: $READYZ_WAIT"
else
  (( EUID == 0 )) || die "run as root, or use --render-only DIR"
fi

STAGE=$(mktemp -d)
trap 'rm -rf -- "$STAGE"' EXIT
render_all "$STAGE"

# An apiserver config change needs a restart, unless install.sh is about to (re)start k3s anyway.
# The marker is written before the new files are installed and removed only once k3s has been
# (re)started with them, so a run that fails in between still restarts k3s on the re-run (the
# files no longer differ by then).
RESTART_PENDING="$HOST/var/lib/rancher/k3s/bootstrap-restart-pending"
for rel in "${K3S_FILES[@]}"; do
  if file_differs "$STAGE/$rel" "$HOST/$rel"; then
    mkdir -p -- "${RESTART_PENDING%/*}"
    : > "$RESTART_PENDING"
  fi
done
k3s_already_current=0
if [[ $(k3s_installed_version) == "$K3S_VERSION" ]]; then k3s_already_current=1; fi

for rel in "${K3S_FILES[@]}" etc/sysctl.d/90-kubelet.conf; do
  install_file "$STAGE/$rel" "$HOST/$rel"
done
install_file "$STAGE/$PEERS_HELPER" "$HOST/$PEERS_HELPER" 0755

sysctl --system
set_mtu "$IFACE" "$UNDERLAY_MTU"
install_packages
install_guard "$STAGE"
mkdir -p "$HOST/var/lib/rancher/k3s/server/logs"
install_k3s server "$K3S_VERSION" false
if [[ -e $RESTART_PENDING ]] && (( k3s_already_current )); then
  log "k3s config changed: restarting k3s"
  systemctl restart k3s
elif (( k3s_already_current )) && ! systemctl is-active --quiet k3s; then
  # Ruling R24: installed at the target version, config unchanged, but stopped. install.sh is
  # skipped, so nothing else would start it and the /readyz wait below could only time out.
  log "k3s $K3S_VERSION installed but not running: starting k3s"
  systemctl enable --now k3s
fi
# Either restarted above or (re)started by install.sh with the new files.
rm -f -- "$RESTART_PENDING"

log "waiting up to ${READYZ_WAIT}s for the apiserver to report /readyz"
deadline=$((SECONDS + READYZ_WAIT))
# --request-timeout bounds each probe, so a hung connection cannot stretch the wait.
until k3s kubectl --request-timeout=5s get --raw /readyz > /dev/null 2>&1; do
  (( SECONDS < deadline )) || die "apiserver not ready after ${READYZ_WAIT}s; see journalctl -u k3s"
  # Not a foreground sleep, so a Ctrl-C as it ends still stops the bootstrap (see gpu_audit in common.sh).
  sleep 5 &
  wait "$!"
done
log "k3s server $K3S_VERSION ready"
