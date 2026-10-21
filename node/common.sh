# shellcheck shell=bash
# Helpers shared by node/server-bootstrap.sh and node/node-bootstrap.sh. Source this file; it sets
# no shell options (callers run with `set -euo pipefail`).
#
# Exit codes: 3 = MTU preflight failed, 4 = unresolved @PLACEHOLDER@ in a rendered template.
# Test seams: SYSFS_NET (default /sys/class/net) is where interface MTU/MAC are read;
# SYSROOT (default empty, i.e. /) prefixes what set_mtu reads and install_guard writes.

log() { printf '[%s] %s\n' "${0##*/}" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# valid_ipv4 ADDR: dotted quad, each octet 0-255.
valid_ipv4() {
  [[ $1 =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] || return 1
  local -a o=("${BASH_REMATCH[@]:1}")
  (( 10#${o[0]} <= 255 && 10#${o[1]} <= 255 && 10#${o[2]} <= 255 && 10#${o[3]} <= 255 ))
}

# render_template SRC DST VAR...: copy SRC to DST, replacing each @VAR@ with the value of $VAR.
# Exits 4 if any @[A-Z_]+@ placeholder is left afterwards.
render_template() {
  local src=$1 dst=$2 content="" var
  shift 2
  [[ -r $src ]] || die "render_template: cannot read $src"
  IFS= read -r -d '' content < "$src" || true
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

# install_file SRC DST: install SRC as DST (0644, parent dirs created) only if the bytes differ.
install_file() {
  if file_differs "$1" "$2"; then
    install -D -m 0644 -- "$1" "$2"
  fi
}

# k3s_install_url VERSION: install.sh pinned to the release tag ("+" URL-encoded), never get.k3s.io.
k3s_install_url() {
  printf 'https://raw.githubusercontent.com/k3s-io/k3s/%s/install.sh\n' "${1//+/%2B}"
}

# default_iface: the interface carrying the default route.
default_iface() {
  local iface
  iface=$(ip -o route get 1.1.1.1 | awk '{for (i = 1; i < NF; i++) if ($i == "dev") {print $(i + 1); exit}}')
  [[ -n $iface ]] || die "cannot determine the default-route interface"
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
  dir="${SYSROOT:-}/run/systemd/network"
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
install_packages() {
  DEBIAN_FRONTEND=noninteractive apt-get update -q
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nftables wireguard-tools
}

# install_guard STAGE: install the guard files rendered under STAGE into ${SYSROOT}/ and keep
# Ubuntu's nftables.service masked (it flushes the whole ruleset, k3s's rules included).
# The table is (re)loaded only when it is not loaded yet or the rendered rules changed: a reload
# resets k3s_peers to the rendered CP addresses and drops every node sync-peers had added.
install_guard() {
  local stage=$1 dest="${SYSROOT:-}" nft=etc/k3s-guard/k3s-guard.nft changed=0 rel
  if file_differs "$stage/$nft" "$dest/$nft"; then changed=1; fi
  for rel in etc/systemd/system/k3s-guard.service etc/systemd/system/k3s.service.d/10-guard.conf "$nft"; do
    install_file "$stage/$rel" "$dest/$rel"
  done
  systemctl mask nftables.service
  systemctl daemon-reload
  if ! systemctl is-active --quiet k3s-guard; then
    systemctl enable --now k3s-guard
  elif (( changed )); then
    log "k3s-guard rules changed: reloading; k3s_peers now holds only the CP addresses, run make sync-peers"
    systemctl reload k3s-guard
  else
    log "k3s-guard rules unchanged: not reloading (that would reset k3s_peers)"
  fi
}

# k3s_installed_version: e.g. v1.36.5+k3s1, or empty when k3s is not installed.
k3s_installed_version() {
  if command -v k3s > /dev/null 2>&1; then
    { k3s --version || true; } | awk 'NR == 1 {print $3}'
  fi
}

# install_k3s EXEC VERSION SKIP_START: run the tag-pinned install.sh unless k3s VERSION is already
# installed. SKIP_START=true installs without starting the service.
install_k3s() {
  local exec=$1 version=$2 skip_start=$3
  local -a vars=(INSTALL_K3S_VERSION="$version" INSTALL_K3S_EXEC="$exec")
  if [[ $(k3s_installed_version) == "$version" ]]; then
    log "k3s $version already installed"
    return 0
  fi
  if [[ $skip_start == true ]]; then vars+=(INSTALL_K3S_SKIP_START=true); fi
  curl -fsSL "$(k3s_install_url "$version")" | env "${vars[@]}" sh -
}
