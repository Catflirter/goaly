# shellcheck shell=bash
# Helpers shared by node/server-bootstrap.sh and node/node-bootstrap.sh. Source this file; it sets
# no shell options (callers run with `set -euo pipefail`).
#
# Exit codes: 3 = MTU preflight failed, 4 = unresolved @PLACEHOLDER@ in a rendered template.
# Test seams: SYSFS_NET (default /sys/class/net) is where interface MTU/MAC are read;
# K3S_BOOTSTRAP_TEST_ROOT (default empty, i.e. /) prefixes every host path these helpers and the
# bootstrap scripts read or write. It exists only for the offline tests; never set it on a node.

log() { printf '[%s] %s\n' "${0##*/}" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# valid_ipv4 ADDR: a unicast host address in canonical dotted-quad form. Octets are 0-255 without
# leading zeros (nft/inet_aton would read "034" as octal 28; Go's net.ParseIP rejects it), and
# 0.0.0.0/8, 127.0.0.0/8 and 224.0.0.0 and up (multicast, reserved, broadcast) are refused.
valid_ipv4() {
  local octet='(0|[1-9][0-9]{0,2})'
  [[ $1 =~ ^${octet}\.${octet}\.${octet}\.${octet}$ ]] || return 1
  local -a o=("${BASH_REMATCH[@]:1}")
  (( o[0] <= 255 && o[1] <= 255 && o[2] <= 255 && o[3] <= 255 )) || return 1
  (( o[0] != 0 && o[0] != 127 && o[0] < 224 ))
}

# render_template SRC DST VAR...: copy SRC to DST, replacing each @VAR@ with the value of $VAR.
# Exits 4 if any @[A-Z_]+@ placeholder is left afterwards.
render_template() {
  local src=$1 dst=$2 content="" var
  shift 2
  [[ -r $src ]] || die "render_template: cannot read $src"
  # The trailing x keeps trailing newlines through $(...); && makes a failed cat fail the assignment.
  content=$(cat -- "$src" && printf x) || die "render_template: cannot read $src"
  content=${content%x}
  for var in "$@"; do
    [[ -v $var ]] || die "render_template: \$$var is not set"
    content=${content//"@${var}@"/"${!var}"}
  done
  if [[ $content =~ @[A-Z_]+@ ]]; then
    log "ERROR: $src: unresolved placeholder ${BASH_REMATCH[0]}"
    exit 4
  fi
  mkdir -p -- "$(dirname -- "$dst")"
  printf '%s' "$content" > "$dst"
}

# file_differs A B: true unless both files exist with identical bytes.
file_differs() { ! cmp -s -- "$1" "$2"; }

# install_file SRC DST [MODE]: install SRC as DST (MODE, default 0644; parent dirs created) only if
# the bytes differ.
install_file() {
  if file_differs "$1" "$2"; then
    install -D -m "${3:-0644}" -- "$1" "$2"
  fi
}

# k3s_install_url VERSION: install.sh pinned to the release tag ("+" URL-encoded), never get.k3s.io.
k3s_install_url() {
  printf 'https://raw.githubusercontent.com/k3s-io/k3s/%s/install.sh\n' "${1//+/%2B}"
}

# default_iface: the interface carrying the default route. Its name must pass the same check as an
# explicit --iface (Linux allows e.g. "$" and "(" in interface names); anything else dies.
default_iface() {
  local iface
  iface=$(ip -o route get 1.1.1.1 | awk '{for (i = 1; i < NF; i++) if ($i == "dev") {print $(i + 1); exit}}')
  [[ -n $iface ]] || die "cannot determine the default-route interface"
  [[ $iface =~ ^[A-Za-z0-9_.:-]+$ ]] || die "unexpected default-route interface name: $iface (pass --iface)"
  printf '%s\n' "$iface"
}

# mtu_preflight IFACE TARGET: exit 3 if the interface MTU is below TARGET (UNDERLAY_MTU).
mtu_preflight() {
  local iface=$1 target=$2 file cur
  file="${SYSFS_NET:-/sys/class/net}/$iface/mtu"
  [[ -r $file ]] || die "interface $iface not found ($file)"
  cur=$(< "$file")
  [[ $cur =~ ^[0-9]+$ ]] || die "unexpected MTU '$cur' in $file"
  if (( cur < target )); then
    log "ERROR: MTU $cur < UNDERLAY_MTU $target on $iface; lower UNDERLAY_MTU and re-run every node"
    exit 3
  fi
}

# set_mtu IFACE TARGET: apply TARGET now (ip link) and persist it with `netplan set` on the netplan
# id whose generated networkd file matches IFACE (by Name= or MAC). Never runs `netplan apply`.
set_mtu() {
  local iface=$1 target=$2 dir mac="" f id
  local -a ids=()
  dir="${K3S_BOOTSTRAP_TEST_ROOT:-}/run/systemd/network"
  if [[ -r ${SYSFS_NET:-/sys/class/net}/$iface/address ]]; then
    mac=$(< "${SYSFS_NET:-/sys/class/net}/$iface/address")
  fi
  for f in "$dir"/10-netplan-*.network; do
    [[ -e $f ]] || continue
    if grep -qxF "Name=$iface" "$f" \
       || { [[ -n $mac ]] && grep -qixE "(Permanent)?MACAddress=${mac}" "$f"; }; then
      id=${f##*/10-netplan-}
      ids+=("${id%.network}")
    fi
  done
  (( ${#ids[@]} == 1 )) \
    || die "need exactly one netplan id for $iface under $dir, found ${#ids[@]}: ${ids[*]:-none}"
  ip link set dev "$iface" mtu "$target"
  netplan set "ethernets.${ids[0]}.mtu=${target}"
}

# install_packages: nftables (guard) and wireguard-tools (tunnel evidence). Never upgrades.
# `apt-get update` only refreshes the package lists: fresh cloud images can ship with them empty,
# and `install` then fails with "Unable to locate package". It upgrades nothing.
install_packages() {
  DEBIAN_FRONTEND=noninteractive apt-get update -q
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nftables wireguard-tools
}

# install_guard STAGE: install the guard files freshly rendered under STAGE into
# ${K3S_BOOTSTRAP_TEST_ROOT}/ and keep Ubuntu's nftables.service masked (it flushes the whole
# ruleset, k3s's rules included). STAGE is the brief's "ROOT": the directory the caller rendered
# into. It must be a staging directory, never the install destination (/) itself, otherwise the
# rendered rules could not be compared with the installed ones; that case dies.
#
# Both bootstraps call it the same way (Ruling R16): `install_guard "$STAGE"`. The unit the guard
# drop-in belongs to follows from what STAGE holds: server-bootstrap.sh renders
# etc/systemd/system/k3s.service.d/10-guard.conf, node-bootstrap.sh renders
# etc/systemd/system/k3s-agent.service.d/10-guard.conf. Exactly one of the two must be present;
# none or both dies before anything is checked, installed or started.
#
# - The rendered nft file is syntax-checked (nft -c) before it replaces the installed copy.
# - The table is reloaded only when the rendered rules changed: a reload resets k3s_peers to the
#   rendered CP addresses (k3s-guard.service then restores the peers sync-peers persisted, if any).
# - An inactive unit is always enabled and started, changed or not: its table is absent (ExecStop
#   deleted it, or it never ran), so there is no k3s_peers to lose, and leaving it down would leave
#   6443/10250 open.
# - A pending reload is remembered in /run until it succeeds, so a run that fails between installing
#   the new rules and reloading them still reloads on the re-run. /run is the right place: after a
#   reboot the unit loads the installed file anyway.
install_guard() {
  local stage=$1 dest="${K3S_BOOTSTRAP_TEST_ROOT:-}" nft=etc/k3s-guard/k3s-guard.nft rel pending unit
  local -a units=()
  [[ -d $stage ]] || die "install_guard: staging directory $stage not found"
  if [[ $(cd -- "$stage" && pwd -P) == "$(cd -- "${dest:-/}" && pwd -P)" ]]; then
    die "install_guard: $stage is the install destination; pass the directory the files were rendered into"
  fi
  for unit in k3s k3s-agent; do
    if [[ -f $stage/etc/systemd/system/$unit.service.d/10-guard.conf ]]; then units+=("$unit"); fi
  done
  case ${#units[@]} in
    1) ;;
    0) die "install_guard: no guard drop-in under $stage (expected etc/systemd/system/k3s.service.d/10-guard.conf or k3s-agent.service.d/10-guard.conf)" ;;
    *) die "install_guard: $stage holds guard drop-ins for both k3s and k3s-agent; render exactly one" ;;
  esac
  pending="$dest/run/k3s-guard.reload-pending"
  if file_differs "$stage/$nft" "$dest/$nft"; then
    nft -c -f "$stage/$nft" || die "install_guard: rendered $nft fails nft -c; installed rules left unchanged"
    mkdir -p -- "${pending%/*}"
    : > "$pending"
  fi
  for rel in etc/systemd/system/k3s-guard.service "etc/systemd/system/${units[0]}.service.d/10-guard.conf" "$nft"; do
    install_file "$stage/$rel" "$dest/$rel"
  done
  systemctl mask nftables.service
  systemctl daemon-reload
  if ! systemctl is-active --quiet k3s-guard; then
    systemctl enable --now k3s-guard
  elif [[ -e $pending ]]; then
    log "k3s-guard rules changed: reloading (k3s_peers is reset to the CP addresses plus the peers" \
        "sync-peers persisted; run make sync-peers if a node is missing)"
    systemctl reload k3s-guard
  else
    log "k3s-guard rules unchanged: not reloading (that would reset k3s_peers)"
  fi
  rm -f -- "$pending"
}

# gpu_audit FILE IFACE LABEL: append a "===== gpu-audit LABEL" section to FILE recording the GPU
# host's state (spec §6.2 step 1, §12 live checks #2 and #5): each probe as "$ CMD", then its
# output. A failing probe (tool or file missing) is recorded with its exit status and does not stop
# the audit. Every probe only reads; in particular the FORWARD policy is recorded, never changed.
# FILE is appended to, never truncated, so a re-run keeps the first run's evidence (the NIC MTU
# before set_mtu changed it). IFACE and the paths are shell-quoted (%q) into the probe text, so
# any value is safe. Each probe gets GPU_AUDIT_TIMEOUT seconds (default 60; nvidia-smi can hang on
# a GPU in a bad state); one that runs out is recorded as "[exit 124]" and the audit goes on.
gpu_audit() {
  local file=$1 iface=$2 label=$3 limit=${GPU_AUDIT_TIMEOUT:-60} cmd rc
  local nvcfg sysfs_mtu iface_q
  [[ $limit =~ ^[1-9][0-9]*$ ]] || die "gpu_audit: GPU_AUDIT_TIMEOUT must be a number of seconds: $limit"
  iface_q=$(printf '%q' "$iface")
  nvcfg=$(printf '%q' "${K3S_BOOTSTRAP_TEST_ROOT:-}/etc/nvidia-container-runtime/config.toml")
  sysfs_mtu=$(printf '%q' "${SYSFS_NET:-/sys/class/net}/$iface/mtu")
  local -a probes=(
    "date -Is"
    "nvidia-smi"
    "nvidia-smi --query-gpu=name,driver_version --format=csv"
    "lsb_release -a"
    "uname -r"
    "command -v nvidia-container-runtime"
    "grep -i mode $nvcfg"
    "dpkg -l | grep -E 'nvidia-container|libnvidia-container'"
    "iptables -V"
    "iptables -S FORWARD | head -3"
    "ip -4 addr show $iface_q"
    "cat $sysfs_mtu"
    "systemctl is-enabled lambda-jupyter docker"
  )
  mkdir -p -- "$(dirname -- "$file")"
  {
    printf '===== gpu-audit %s\n' "$label"
    for cmd in "${probes[@]}"; do
      printf '$ %s\n' "$cmd"
      # bash -c runs without pipefail: a pipeline reports its last command (grep: 1 = no match;
      # head never fails), not a SIGPIPE in iptables. timeout signals the probe's whole process
      # group (TERM, then KILL 10 s later) and exits 124 when the limit is hit.
      rc=0
      timeout -k 10 "$limit" bash -c "$cmd" 2>&1 || rc=$?
      if (( rc != 0 )); then printf '[exit %d]\n' "$rc"; fi
    done
  } >> "$file"
}

# k3s_installed_version: e.g. v1.36.5+k3s1, or empty when k3s is not installed.
k3s_installed_version() {
  if command -v k3s > /dev/null 2>&1; then
    { k3s --version || true; } | awk 'NR == 1 {print $3}'
  fi
}

# install_k3s EXEC VERSION SKIP_START: run the tag-pinned install.sh unless k3s VERSION is already
# installed. SKIP_START=true installs without starting the service. install.sh writes every K3S_*
# variable in its environment into the service's env file (and K3S_URL/K3S_TOKEN change its mode),
# so join credentials inherited from the caller are removed: the token only ever reaches
# token-file (spec §5.1).
install_k3s() {
  local exec=$1 version=$2 skip_start=$3
  local -a vars=(INSTALL_K3S_VERSION="$version" INSTALL_K3S_EXEC="$exec")
  if [[ $(k3s_installed_version) == "$version" ]]; then
    log "k3s $version already installed"
    return 0
  fi
  if [[ $skip_start == true ]]; then vars+=(INSTALL_K3S_SKIP_START=true); fi
  curl -fsSL "$(k3s_install_url "$version")" \
    | env -u K3S_TOKEN -u K3S_AGENT_TOKEN -u K3S_TOKEN_FILE -u K3S_URL "${vars[@]}" sh -
}
