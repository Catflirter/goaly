#!/usr/bin/env bash
# Agent bootstrap for every k3s worker: GCP CPU workers and Lambda GPU nodes (spec §2.1, §2.3 #3,
# §4.2, §4.5, §5.1, §6.2). Idempotent; scripts/join-node.sh copies node/ to the node and runs it as
# root over SSH:
#
#   node-bootstrap.sh --server-ip IP --node-name NAME --external-ip IP [--underlay-mtu 1460]
#                     [--k3s-version v1.36.5+k3s1] [--gpu --gpu-type T | --no-gpu] [--iface IF]
#                     [--render-only DIR]
#
# Without --gpu/--no-gpu the node is a GPU node when nvidia-smi is on PATH; a GPU node needs
# --gpu-type (exit 2 without it). k3s-agent is installed but not started: join-node.sh pipes the
# join token into /etc/rancher/k3s/token and then starts it. The token never passes through an
# environment variable, because install.sh would write it into the service's env file.
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
  log "usage: ${0##*/} --server-ip IP --node-name NAME --external-ip IP [--underlay-mtu 1460]" \
      "[--k3s-version v1.36.5+k3s1] [--gpu --gpu-type T | --no-gpu] [--iface IF] [--render-only DIR]"
  [[ $# -eq 0 ]] || log "ERROR: $*"
  exit 2
}

SERVER_IP="" NODE_NAME="" EXTERNAL_IP="" UNDERLAY_MTU=1460 K3S_VERSION=v1.36.5+k3s1 IFACE=""
GPU_FLAG="" GPU_TYPE="" RENDER_ONLY=0 RENDER_DIR=""
while (( $# )); do
  case $1 in
    --server-ip | --node-name | --external-ip | --underlay-mtu | --k3s-version | --gpu-type | --iface \
    | --render-only)
      (( $# >= 2 )) || usage "$1 needs a value"
      case $1 in
        --server-ip) SERVER_IP=$2 ;;
        --node-name) NODE_NAME=$2 ;;
        --external-ip) EXTERNAL_IP=$2 ;;
        --underlay-mtu) UNDERLAY_MTU=$2 ;;
        --k3s-version) K3S_VERSION=$2 ;;
        --gpu-type) GPU_TYPE=$2 ;;
        --iface) IFACE=$2 ;;
        --render-only) RENDER_ONLY=1 RENDER_DIR=$2 ;;
      esac
      shift 2 ;;
    --gpu | --no-gpu)
      [[ -z $GPU_FLAG || $GPU_FLAG == "$1" ]] || usage "--gpu and --no-gpu exclude each other"
      GPU_FLAG=$1
      shift ;;
    -h | --help) usage ;;
    *) usage "unknown argument: $1" ;;
  esac
done

[[ -n $SERVER_IP ]] || usage "--server-ip is required"
[[ -n $NODE_NAME ]] || usage "--node-name is required"
[[ -n $EXTERNAL_IP ]] || usage "--external-ip is required"
valid_ipv4 "$SERVER_IP" || usage "--server-ip is not an IPv4 address: $SERVER_IP"
valid_ipv4 "$EXTERNAL_IP" || usage "--external-ip is not an IPv4 address: $EXTERNAL_IP"
# An RFC 1123 label: valid as a K8s node name and as a hostname, and safe to write into YAML bare.
[[ $NODE_NAME =~ ^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$ ]] \
  || usage "--node-name must be a lowercase RFC 1123 label (at most 63 characters): $NODE_NAME"
[[ $UNDERLAY_MTU =~ ^[1-9][0-9]{2,3}$ ]] || usage "--underlay-mtu must be an MTU in bytes (100-9999): $UNDERLAY_MTU"
[[ $K3S_VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+\+k3s[0-9]+$ ]] || usage "bad --k3s-version: $K3S_VERSION"
[[ -z $IFACE || $IFACE =~ ^[A-Za-z0-9_.:-]+$ ]] || usage "bad --iface: $IFACE"
(( ! RENDER_ONLY )) || [[ -n $RENDER_DIR ]] || usage "--render-only needs a directory"

case $GPU_FLAG in
  --gpu) GPU=1 ;;
  --no-gpu)
    [[ -z $GPU_TYPE ]] || usage "--gpu-type contradicts --no-gpu"
    GPU=0 ;;
  *)
    # --gpu-type alone states a GPU node; otherwise nvidia-smi on PATH makes it one.
    if [[ -n $GPU_TYPE ]] || command -v nvidia-smi > /dev/null 2>&1; then GPU=1; else GPU=0; fi
    if (( GPU )) && [[ -z $GPU_TYPE ]]; then
      usage "nvidia-smi found: this is a GPU node, pass --gpu --gpu-type T (or --no-gpu)"
    fi ;;
esac
if (( GPU )); then
  [[ -n $GPU_TYPE ]] || usage "--gpu needs --gpu-type T"
  # A K8s label value (it becomes the gpu-type=T node label).
  [[ $GPU_TYPE =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?$ ]] \
    || usage "--gpu-type must be a Kubernetes label value (at most 63 characters): $GPU_TYPE"
fi

if [[ -z $IFACE ]]; then IFACE=$(default_iface); fi
mtu_preflight "$IFACE" "$UNDERLAY_MTU"

# The GPU label and taint (spec §2.1); a CPU node gets neither. GPU_CONFIG and CP_IP are read by
# render_template, by name.
# shellcheck disable=SC2034
if (( GPU )); then
  GPU_CONFIG="node-label: [\"nvidia.com/gpu.present=true\", \"gpu-type=$GPU_TYPE\"]
node-taint: [\"nvidia.com/gpu=present:NoSchedule\"]"
else
  GPU_CONFIG="# CPU node: no GPU label or taint."
fi
# shellcheck disable=SC2034
CP_IP=$SERVER_IP
CONFIG=etc/rancher/k3s/config.yaml
SYSCTL=etc/sysctl.d/90-kubelet.conf

render_all() {
  local root=$1
  render_template "$HERE/k3s/agent-config.yaml.tmpl" "$root/$CONFIG" SERVER_IP NODE_NAME EXTERNAL_IP GPU_CONFIG
  render_template "$HERE/k3s/90-kubelet.conf" "$root/$SYSCTL"
  render_template "$HERE/guard/k3s-guard-agent.nft.tmpl" "$root/etc/k3s-guard/k3s-guard.nft" CP_IP
  render_template "$HERE/guard/k3s-guard.service" "$root/etc/systemd/system/k3s-guard.service"
  render_template "$HERE/k3s/10-guard.conf" "$root/etc/systemd/system/k3s-agent.service.d/10-guard.conf"
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
if [[ -n $HOST ]]; then
  log "WARNING: K3S_BOOTSTRAP_TEST_ROOT=$HOST: test mode, host paths are prefixed with it"
else
  (( EUID == 0 )) || die "run as root, or use --render-only DIR"
fi
if (( GPU )) && ! command -v nvidia-smi > /dev/null 2>&1; then
  die "GPU node requested but nvidia-smi is not on PATH; nothing changed (use --no-gpu for a CPU node)"
fi

# disable_units UNIT...: systemctl disable --now the UNITs that exist; a missing one only warns.
disable_units() {
  local unit
  local -a present=()
  for unit in "$@"; do
    if systemctl cat "$unit" > /dev/null 2>&1; then
      present+=("$unit")
    else
      log "WARNING: $unit not found; nothing to disable"
    fi
  done
  if (( ${#present[@]} )); then systemctl disable --now "${present[@]}"; fi
}

STAGE=$(mktemp -d)
trap 'rm -rf -- "$STAGE"' EXIT
render_all "$STAGE"

AUDIT="$HOST/var/log/gpu-audit.txt"
if (( GPU )); then
  # Before anything changes, so it records the host as Lambda delivered it, NIC MTU included
  # (live check #2).
  log "GPU node: recording the host state in $AUDIT"
  gpu_audit "$AUDIT" "$IFACE" before
fi

install_file "$STAGE/$SYSCTL" "$HOST/$SYSCTL"
sysctl --system
set_mtu "$IFACE" "$UNDERLAY_MTU"
install_packages
install_guard "$STAGE"

if (( GPU )); then
  # spec §4.5: lambda-jupyter can bypass the firewall; docker is not needed by k3s and rewrites
  # FORWARD. The FORWARD policy docker set is left as it is (k3s/flannel add their own rules).
  disable_units lambda-jupyter
  disable_units docker.socket docker.service
  gpu_audit "$AUDIT" "$IFACE" after
fi

# A running k3s-agent only picks up a new config or binary when restarted. The marker is written
# before the new config is installed and removed only once that restart has happened (or was not
# needed because k3s-agent is not running; join-node.sh starts it), so a run that fails in
# between still restarts on the re-run.
RESTART_PENDING="$HOST/var/lib/rancher/k3s/bootstrap-restart-pending"
if file_differs "$STAGE/$CONFIG" "$HOST/$CONFIG" || [[ $(k3s_installed_version) != "$K3S_VERSION" ]]; then
  mkdir -p -- "${RESTART_PENDING%/*}"
  : > "$RESTART_PENDING"
fi
install_file "$STAGE/$CONFIG" "$HOST/$CONFIG"
install_k3s agent "$K3S_VERSION" true
if [[ -e $RESTART_PENDING ]] && systemctl is-active --quiet k3s-agent; then
  log "k3s-agent config or version changed: restarting k3s-agent"
  systemctl restart k3s-agent
fi
rm -f -- "$RESTART_PENDING"

log "k3s agent $K3S_VERSION installed as $NODE_NAME; join-node.sh now installs the join token" \
    "and starts k3s-agent"
