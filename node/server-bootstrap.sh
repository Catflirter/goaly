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

render_all() {
  local root=$1
  render_template "$HERE/k3s/server-config.yaml.tmpl" "$root/etc/rancher/k3s/config.yaml" CP_IP
  render_template "$HERE/k3s/psa.yaml" "$root/etc/rancher/k3s/psa.yaml"
  render_template "$HERE/k3s/audit-policy.yaml" "$root/etc/rancher/k3s/audit-policy.yaml"
  render_template "$HERE/k3s/90-kubelet.conf" "$root/etc/sysctl.d/90-kubelet.conf"
  render_template "$HERE/guard/k3s-guard-server.nft.tmpl" "$root/etc/k3s-guard/k3s-guard.nft" PEERS
  render_template "$HERE/guard/k3s-guard.service" "$root/etc/systemd/system/k3s-guard.service"
  render_template "$HERE/k3s/10-guard.conf" "$root/etc/systemd/system/k3s.service.d/10-guard.conf"
}

if (( RENDER_ONLY )); then
  render_all "$RENDER_DIR"
  log "rendered into $RENDER_DIR (render-only: nothing on this host was changed)"
  exit 0
fi

# ---- Everything below modifies the host. ------------------------------------------------------
(( EUID == 0 )) || die "run as root, or use --render-only DIR"

STAGE=$(mktemp -d)
trap 'rm -rf -- "$STAGE"' EXIT
render_all "$STAGE"

# An apiserver config change needs a restart, unless install.sh is about to (re)start k3s anyway.
k3s_config_changed=0
for rel in "${K3S_FILES[@]}"; do
  if file_differs "$STAGE/$rel" "/$rel"; then k3s_config_changed=1; fi
done
k3s_already_current=0
if [[ $(k3s_installed_version) == "$K3S_VERSION" ]]; then k3s_already_current=1; fi

for rel in "${K3S_FILES[@]}" etc/sysctl.d/90-kubelet.conf; do
  install_file "$STAGE/$rel" "/$rel"
done

sysctl --system
set_mtu "$IFACE" "$UNDERLAY_MTU"
install_packages
install_guard "$STAGE"
mkdir -p /var/lib/rancher/k3s/server/logs
install_k3s server "$K3S_VERSION" false
if (( k3s_config_changed && k3s_already_current )); then
  log "k3s config changed: restarting k3s"
  systemctl restart k3s
fi

log "waiting up to 180s for the apiserver to report /readyz"
deadline=$((SECONDS + 180))
until k3s kubectl get --raw /readyz > /dev/null 2>&1; do
  (( SECONDS < deadline )) || die "apiserver not ready after 180s; see journalctl -u k3s"
  sleep 5
done
log "k3s server $K3S_VERSION ready"
