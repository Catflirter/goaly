#!/usr/bin/env bash
# make monitoring / monitoring-down / grafana / prometheus (spec §10.0, stretch S0): the lightweight
# monitoring stack in k8s/monitoring/ and the operator's only way in, kubectl port-forward on 127.0.0.1.
#
#   monitoring.sh install              install or upgrade: the namespaces, the Grafana admin Secret (made
#                                      once, see below), the dashboard ConfigMap, the manifests; then
#                                      wait for every rollout. Safe to re-run.
#   monitoring.sh uninstall            ask, then delete the stack: both namespaces (with Prometheus' data
#                                      and the Secret) and kube-state-metrics' ClusterRole and binding
#   monitoring.sh forward grafana|prometheus
#                                      kubectl port-forward in the background on 127.0.0.1:GRAFANA_PORT
#                                      (3000) or 127.0.0.1:PROMETHEUS_PORT (9090); PID and log in .run/
#   monitoring.sh stop grafana|prometheus
#                                      stop that port-forward; idempotent
#
# The Grafana admin password: install generates it with openssl rand when the Secret
# monitoring/grafana-admin does not exist yet, sends the Secret to `kubectl create -f -` on stdin (never
# on a command line, never in a file, never through apply's last-applied annotation) and shows the
# password once, on the terminal (/dev/tty), not on stdout or stderr, which make, tee or a CI job may
# keep. Without a terminal it is not shown at all; the command to read it from the Secret is.
# A later install keeps the Secret. Grafana keeps its database in an emptyDir, so a new password
# (the Secret was deleted) takes effect with the Grafana restart install then does.
# Everything goes through the cluster kubeconfig (IAP tunnel, `make tunnel`); nothing here needs .env.
# Test seam: MONITORING_TTY replaces /dev/tty; only the offline tests set it.
set -euo pipefail

# shellcheck disable=SC1091  # common.sh is linted on its own
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

# Not needed here; keep it out of kubectl's environment if the caller exported it.
unset LAMBDA_API_KEY

MON_DIR="$REPO_ROOT/k8s/monitoring"
# Applied in this order after namespaces.yaml; tests/k8s/test_monitoring_manifests.py checks that this
# list and the directory agree.
MANIFESTS=(prometheus.yaml kube-state-metrics.yaml node-exporter.yaml dcgm-exporter.yaml grafana.yaml
  network-policies.yaml)
NS=monitoring
HOST_NS=monitoring-host
SECRET=grafana-admin
ROLLOUT_TIMEOUT=5m
TTY="${MONITORING_TTY:-/dev/tty}"
FORWARD_TIMEOUT="${FORWARD_TIMEOUT_S:-30}"

usage() {
  log "usage: ${0##*/} install|uninstall|forward grafana|forward prometheus|stop grafana|stop prometheus"
  exit 2
}

need_kubeconfig() {
  [[ -f "$KUBECONFIG_FILE" ]] || die "$KUBECONFIG_FILE not found: run make kubeconfig and make tunnel first"
}

# b64 TEXT: TEXT base64-encoded on one line. printf is a builtin, so TEXT never appears in an argv.
b64() {
  printf '%s' "$1" | base64 | tr -d '\n'
}

# show_password PASSWORD: write it to the terminal only (see the header).
show_password() {
  if (exec 3>> "$TTY") 2> /dev/null; then
    printf '\nGrafana admin password (shown once; it is kept only in the Secret %s/%s):\n\n    %s\n\n' \
      "$NS" "$SECRET" "$1" >> "$TTY"
    log "the new Grafana admin password was shown on the terminal (user: admin)"
  else
    log "no terminal: the new Grafana admin password is not printed. Read it later with:"
    log "  kubectl --kubeconfig $KUBECONFIG_FILE -n $NS get secret $SECRET -o jsonpath='{.data.admin-password}' | base64 -d"
  fi
}

# ensure_admin_secret: create the Secret if it does not exist; NEW_SECRET=1 when this run created it.
ensure_admin_secret() {
  local out rc=0 password secret_key
  NEW_SECRET=0
  out="$(kc -n "$NS" get secret "$SECRET" -o name --request-timeout="$KC_REQUEST_TIMEOUT" 2>&1)" || rc=$?
  if (( rc == 0 )); then
    log "Grafana admin password: unchanged (Secret $NS/$SECRET exists)"
    return 0
  fi
  [[ "$out" == *NotFound* ]] || die "cannot read the Secret $NS/$SECRET: $out"
  # 32 letters and digits (about 190 bits), and the key Grafana encrypts its stored secrets with.
  password="$(openssl rand -base64 48 | LC_ALL=C tr -dc 'A-Za-z0-9' | cut -c1-32)" || die "openssl rand failed"
  secret_key="$(openssl rand -hex 32)" || die "openssl rand failed"
  [[ "$password" =~ ^[A-Za-z0-9]{32}$ && "$secret_key" =~ ^[0-9a-f]{64}$ ]] \
    || die "could not generate the Grafana admin password"
  rc=0
  printf 'apiVersion: v1\nkind: Secret\nmetadata:\n  name: %s\n  namespace: %s\n  labels:\n    app.kubernetes.io/name: grafana\n    app.kubernetes.io/part-of: monitoring\ntype: Opaque\ndata:\n  admin-user: %s\n  admin-password: %s\n  secret-key: %s\n' \
    "$SECRET" "$NS" "$(b64 admin)" "$(b64 "$password")" "$(b64 "$secret_key")" \
    | kc create -f - --request-timeout="$KC_REQUEST_TIMEOUT" >&2 || rc=$?
  if (( rc != 0 )); then
    # Another install created it in the meantime: that one shows its password; this one keeps quiet.
    if kc -n "$NS" get secret "$SECRET" -o name --request-timeout="$KC_REQUEST_TIMEOUT" > /dev/null 2>&1; then
      log "Grafana admin password: the Secret $NS/$SECRET was created by another run meanwhile; kept"
      return 0
    fi
    die "cannot create the Secret $NS/$SECRET"
  fi
  NEW_SECRET=1
  show_password "$password"
}

install() {
  local grafana_existed=0 f
  local -a files=()
  need_kubeconfig
  log "==> monitoring: namespaces $NS (restricted) and $HOST_NS (privileged, for hostPath)"
  kc apply -f "$MON_DIR/namespaces.yaml" --request-timeout="$KC_REQUEST_TIMEOUT" >&2
  if kc -n "$NS" get deployment grafana -o name --request-timeout="$KC_REQUEST_TIMEOUT" > /dev/null 2>&1; then
    grafana_existed=1
  fi
  ensure_admin_secret
  log "==> monitoring: the dashboard ConfigMap from grafana-dashboard.json"
  kc -n "$NS" create configmap grafana-dashboards \
    --from-file="hybrid-gpu.json=$MON_DIR/grafana-dashboard.json" --dry-run=client -o yaml \
    | kc apply -f - --request-timeout="$KC_REQUEST_TIMEOUT" >&2
  log "==> monitoring: Prometheus, kube-state-metrics, node-exporter, DCGM exporter, Grafana"
  for f in "${MANIFESTS[@]}"; do
    files+=(-f "$MON_DIR/$f")
  done
  kc apply "${files[@]}" --request-timeout="$KC_REQUEST_TIMEOUT" >&2
  if (( NEW_SECRET && grafana_existed )); then
    log "the admin password is new: restarting Grafana so it starts from it"
    kc -n "$NS" rollout restart deployment/grafana >&2
  fi
  kc -n "$NS" rollout status deployment/prometheus --timeout="$ROLLOUT_TIMEOUT" >&2
  kc -n "$NS" rollout status deployment/kube-state-metrics --timeout="$ROLLOUT_TIMEOUT" >&2
  kc -n "$NS" rollout status deployment/grafana --timeout="$ROLLOUT_TIMEOUT" >&2
  kc -n "$HOST_NS" rollout status daemonset/node-exporter --timeout="$ROLLOUT_TIMEOUT" >&2
  kc -n "$HOST_NS" rollout status daemonset/dcgm-exporter --timeout="$ROLLOUT_TIMEOUT" >&2
  log "monitoring: rolled out. Open Grafana with: make grafana"
}

uninstall() {
  local f name
  local -a files=()
  need_kubeconfig
  confirm "Delete the monitoring stack (namespaces $NS and $HOST_NS with Prometheus' data and the Grafana admin Secret)?" \
    || die "not deleted" 2
  for name in grafana prometheus; do
    stop_forward "$name"
  done
  for f in "${MANIFESTS[@]}" namespaces.yaml; do
    files+=(-f "$MON_DIR/$f")
  done
  kc delete "${files[@]}" --ignore-not-found --request-timeout="$KC_REQUEST_TIMEOUT" --timeout=5m >&2
  log "monitoring: deleted"
}

# ---- port-forwards ----------------------------------------------------------------------------

# forward_target NAME: set FWD_PORT (local), FWD_REMOTE (the Service port), FWD_PIDFILE and FWD_LOG.
forward_target() {
  case "$1" in
    grafana) FWD_PORT="${GRAFANA_PORT:-3000}" FWD_REMOTE=3000 ;;
    prometheus) FWD_PORT="${PROMETHEUS_PORT:-9090}" FWD_REMOTE=9090 ;;
    *) usage ;;
  esac
  if ! [[ "$FWD_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || (( FWD_PORT > 65535 )); then
    die "bad local port for $1: '$FWD_PORT'"
  fi
  FWD_PIDFILE="$RUN_DIR/port-forward-$1.pid"
  FWD_LOG="$RUN_DIR/port-forward-$1.log"
}

# forward_pid NAME: the PID in its PID file if that process is alive and is our port-forward of svc/NAME
# (a recycled PID that now belongs to something else is not); 1 otherwise. Call forward_target first.
forward_pid() {
  local pid
  [[ -f "$FWD_PIDFILE" ]] || return 1
  pid="$(cat -- "$FWD_PIDFILE")" || return 1
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2> /dev/null || return 1
  [[ " $(ps -ww -p "$pid" -o command= 2> /dev/null) " == *" port-forward "*" svc/$1 "* ]] || return 1
  printf '%s\n' "$pid"
}

port_in_use() {
  (exec 3<> "/dev/tcp/127.0.0.1/$1") 2> /dev/null
}

stop_forward() {
  local pid
  forward_target "$1"
  if pid="$(forward_pid "$1")"; then
    # Started with job control: its process group id is its PID.
    kill -TERM -- "-$pid" 2> /dev/null || kill -TERM "$pid" 2> /dev/null || true
    log "$1 port-forward stopped (pid $pid)"
  else
    log "$1 port-forward not running"
  fi
  rm -f -- "$FWD_PIDFILE"
}

forward() {
  local name="$1" pid deadline url
  forward_target "$name"
  need_kubeconfig
  url="http://127.0.0.1:$FWD_PORT"
  if pid="$(forward_pid "$name")"; then
    if [[ " $(ps -ww -p "$pid" -o command= 2> /dev/null) " == *" $FWD_PORT:$FWD_REMOTE "* ]]; then
      log "$name port-forward already running (pid $pid): $url"
      return 0
    fi
    stop_forward "$name"
  fi
  rm -f -- "$FWD_PIDFILE"
  ! port_in_use "$FWD_PORT" || die "127.0.0.1:$FWD_PORT is already in use; stop that or set a free port"
  [[ "$FORWARD_TIMEOUT" =~ ^[0-9]+$ ]] || die "bad FORWARD_TIMEOUT_S '$FORWARD_TIMEOUT'"
  run_dir || die "cannot create $RUN_DIR"
  umask 077
  # --address 127.0.0.1: only this machine, never the LAN (kubectl's default "localhost" adds ::1).
  # Job control gives it its own process group; nohup keeps it alive when the terminal closes.
  set -m
  nohup kubectl --kubeconfig "$KUBECONFIG_FILE" port-forward --address 127.0.0.1 -n "$NS" "svc/$name" \
    "$FWD_PORT:$FWD_REMOTE" < /dev/null > "$FWD_LOG" 2>&1 &
  pid=$!
  set +m
  printf '%s\n' "$pid" > "$FWD_PIDFILE"
  deadline=$((SECONDS + FORWARD_TIMEOUT))
  until grep -q "Forwarding from 127.0.0.1:$FWD_PORT" -- "$FWD_LOG" 2> /dev/null; do
    if ! kill -0 "$pid" 2> /dev/null || (( SECONDS >= deadline )); then
      kill -TERM -- "-$pid" 2> /dev/null || true
      rm -f -- "$FWD_PIDFILE"
      tail -n 20 -- "$FWD_LOG" >&2 || true
      die "the $name port-forward did not start (is the API tunnel up? make tunnel); log: $FWD_LOG"
    fi
    sleep 0.2
  done
  log "$name: $url (port-forward pid $pid; stop it with make $name-stop)"
  if [[ "$name" == grafana ]]; then
    log "log in as admin with the password make monitoring showed once, or read it with:"
    log "  kubectl --kubeconfig $KUBECONFIG_FILE -n $NS get secret $SECRET -o jsonpath='{.data.admin-password}' | base64 -d"
  fi
}

case "${1:-}" in
  install) [[ $# -eq 1 ]] || usage; install ;;
  uninstall) [[ $# -eq 1 ]] || usage; uninstall ;;
  forward) [[ $# -eq 2 ]] || usage; forward "$2" ;;
  stop) [[ $# -eq 2 ]] || usage; stop_forward "$2" ;;
  *) usage ;;
esac
