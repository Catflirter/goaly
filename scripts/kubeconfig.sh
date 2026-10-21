#!/usr/bin/env bash
# make kubeconfig (spec §8): fetch k3s-cp's /etc/rancher/k3s/k3s.yaml over IAP SSH into
# ~/.kube/hybrid-gpu.yaml (0600). k3s writes the server as https://127.0.0.1:6443, which is where
# `make tunnel` forwards the API to; any other server line is refused and the previous file kept.
# The file holds the cluster-admin client key: it is never printed.
set -euo pipefail

# shellcheck disable=SC1091  # common.sh is linted on its own
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"
load_env

umask 077
mkdir -p -- "${KUBECONFIG_FILE%/*}"
# Written next to the target and renamed over it, so a failed fetch never leaves a half file.
tmp="$(mktemp "$KUBECONFIG_FILE.XXXXXX")"
trap 'rm -f -- "$tmp"' EXIT

cp_ssh 'sudo cat /etc/rancher/k3s/k3s.yaml' > "$tmp" || die "could not read k3s.yaml from $CP_NAME"
server="$(awk '$1 == "server:" {print $2}' "$tmp")"
[[ "$server" == "https://127.0.0.1:6443" ]] \
  || die "k3s.yaml on $CP_NAME must have exactly one server line, https://127.0.0.1:6443; found '${server//$'\n'/ }'"
chmod 600 -- "$tmp"
mv -f -- "$tmp" "$KUBECONFIG_FILE"
log "wrote $KUBECONFIG_FILE (0600); make tunnel forwards https://127.0.0.1:6443 to the API"
