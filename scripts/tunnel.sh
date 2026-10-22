#!/usr/bin/env bash
# make tunnel / tunnel-stop (spec §8): forward 127.0.0.1:6443 to the API server on k3s-cp through
# IAP SSH, so 6443 is never opened to the internet.
#
#   tunnel.sh start    start it in the background (PID in .run/tunnel.pid, log in .run/tunnel.log)
#                      and wait up to 30s (TUNNEL_TIMEOUT_S) for /readyz; already up: no-op;
#                      running but the API does not answer: replaced
#   tunnel.sh stop     stop it (its whole process group: gcloud, ssh, the IAP helper); idempotent
#   tunnel.sh status   0 if it runs and the API answers /readyz, 1 otherwise
#
# start and stop hold a lock, flock(1) on .run/tunnel.lock (C-2), so `make tunnel` during `make up`'s
# tunnel step waits for it instead of launching a second tunnel that would orphan the first. The
# kernel releases the lock when the run exits, however it ends: a killed run leaves no stale lock, and
# a PID in the file (reused after a reboot, say) means nothing. A live holder is waited for, at most
# TUNNEL_TIMEOUT_S + 30 s, then the run gives up. The tunnel itself never holds it. Without flock(1)
# (util-linux), or with a bash older than 4.1, start and stop are refused (exit 2).
#
# The forward is spec §8's -L 6443:127.0.0.1:6443. It names no bind address, so ssh listens on the
# loopback addresses only unless GatewayPorts is on. start passes -o GatewayPorts=no, and ssh takes
# a command-line option over any ssh config file, so a `GatewayPorts yes` in the operator's ssh
# config cannot expose the cluster-admin API on the LAN. Test seam: K3S_TUNNEL_TEST_PORT replaces
# the local port 6443; only the offline tests set it (the kubeconfig always points at 6443).
set -euo pipefail

# shellcheck disable=SC1091  # common.sh is linted on its own
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"

PIDFILE="$RUN_DIR/tunnel.pid"
LOGFILE="$RUN_DIR/tunnel.log"
LOCAL_PORT="${K3S_TUNNEL_TEST_PORT:-6443}"
# "-L $FORWARD" also identifies our tunnel among running processes (tunnel_pid).
FORWARD="$LOCAL_PORT:127.0.0.1:6443"
TIMEOUT="${TUNNEL_TIMEOUT_S:-30}"
LOCKFILE="$RUN_DIR/tunnel.lock"

usage() {
  log "usage: ${0##*/} start|stop|status"
  exit 2
}

# lock: hold flock(1) on $LOCKFILE until this run exits (the descriptor stays open; the background
# tunnel does not inherit it: close_fds_above_2). A live holder is waited for, at most as long as a
# start can take, then this run gives up. The holder writes its PID into the file, for the waiting
# message only. The file is never removed (removing it would let a waiting run lock an unlinked file).
lock() {
  local fd holder
  [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || die "bad TUNNEL_TIMEOUT_S '$TIMEOUT'"
  _bash_version_ok \
    || die "tunnel.sh needs bash >= 4.1 (this is bash ${BASH_VERSION}); run it with a newer bash (macOS: brew install bash): nothing was changed" 2
  command -v flock > /dev/null 2>&1 \
    || die "tunnel.sh needs flock(1) (util-linux) on PATH to keep start and stop apart ($FLOCK_INSTALL): nothing was changed" 2
  run_dir || die "cannot create $RUN_DIR"
  exec {fd}>> "$LOCKFILE" || die "cannot open $LOCKFILE"
  if ! flock -n "$fd"; then
    holder="$(head -n 1 -- "$LOCKFILE" 2> /dev/null || true)"
    [[ "$holder" =~ ^[1-9][0-9]*$ ]] || holder=unknown
    log "waiting for another tunnel.sh (pid $holder) to finish"
    flock -w "$((TIMEOUT + 30))" "$fd" \
      || die "another tunnel.sh (pid $holder) still holds $LOCKFILE after $((TIMEOUT + 30))s; it is released when that run exits"
  fi
  printf '%s\n' "$$" > "$LOCKFILE" || true
}

# tunnel_pid: print the PID in $PIDFILE if that process is alive and is this tunnel (its command
# line holds the arguments -L $FORWARD); 1 otherwise. A recycled PID that now belongs to something
# else is not ours, nor is another forward that ends in ours, such as -L 16443:127.0.0.1:6443.
tunnel_pid() {
  local pid
  [[ -f "$PIDFILE" ]] || return 1
  pid="$(cat -- "$PIDFILE")" || return 1
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2> /dev/null || return 1
  [[ " $(ps -ww -p "$pid" -o command= 2> /dev/null) " == *" -L $FORWARD "* ]] || return 1
  printf '%s\n' "$pid"
}

# close_fds_above_2: close every descriptor of this shell above 2 (the ones /dev/fd lists). Run in the
# background job right before it execs the tunnel.
close_fds_above_2() {
  local f n
  for f in /dev/fd/*; do
    n="${f##*/}"
    if [[ "$n" =~ ^[0-9]+$ ]] && (( n > 2 )); then
      eval "exec $n>&-"
    fi
  done
}

api_ready() {
  [[ "$(kc --request-timeout=5s get --raw /readyz 2> /dev/null)" == ok ]]
}

port_in_use() {
  (exec 3<> "/dev/tcp/127.0.0.1/$LOCAL_PORT") 2> /dev/null
}

# kill_tunnel PID: TERM its process group (started with job control, so the group id is PID), then
# KILL whatever is left after 5s.
kill_tunnel() {
  local pid="$1" _
  kill -TERM -- "-$pid" 2> /dev/null || kill -TERM "$pid" 2> /dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 -- "-$pid" 2> /dev/null || return 0
    sleep 0.5
  done
  log "tunnel (pid $pid) ignored TERM: killing it"
  kill -KILL -- "-$pid" 2> /dev/null || true
}

stop() {
  local pid
  lock
  if pid="$(tunnel_pid)"; then
    kill_tunnel "$pid"
    log "tunnel stopped (pid $pid)"
  else
    log "tunnel not running"
  fi
  rm -f -- "$PIDFILE"
}

status() {
  local pid
  if ! pid="$(tunnel_pid)"; then
    log "tunnel not running"
    return 1
  fi
  if ! api_ready; then
    log "tunnel running (pid $pid), but the API does not answer /readyz"
    return 1
  fi
  log "tunnel running (pid $pid); API ready at https://127.0.0.1:$LOCAL_PORT"
}

start() {
  local pid deadline
  [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || die "bad TUNNEL_TIMEOUT_S '$TIMEOUT'"
  load_env
  # Not needed here, and the tunnel runs for days: keep the key out of its environment.
  unset LAMBDA_API_KEY
  [[ -f "$KUBECONFIG_FILE" ]] || die "$KUBECONFIG_FILE not found: run make kubeconfig first"
  lock
  if pid="$(tunnel_pid)"; then
    if api_ready; then
      log "tunnel already running (pid $pid); API ready"
      return 0
    fi
    log "tunnel (pid $pid) runs but the API does not answer: replacing it"
    kill_tunnel "$pid"
  fi
  rm -f -- "$PIDFILE"
  if port_in_use; then
    die "127.0.0.1:$LOCAL_PORT is already in use by another process (not this repo's tunnel); stop it first"
  fi
  cp_resolve
  run_dir || die "cannot create $RUN_DIR"
  umask 077
  # Job control puts the background job in its own process group, so stop can end gcloud, the ssh
  # it runs and ssh's IAP ProxyCommand together; nohup keeps it alive when the terminal closes. The
  # job execs gcloud (its PID stays $!) with every descriptor above 2 closed: the tunnel runs for days
  # and must not keep a lock its starter holds (a stack's lock when make up starts it, stack_lock).
  set -m
  {
    close_fds_above_2
    exec nohup gcloud compute ssh "$CP_NAME" --project "$GCP_PROJECT" --zone "$CP_ZONE" --tunnel-through-iap \
      --quiet -- -N -L "$FORWARD" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o GatewayPorts=no \
      < /dev/null > "$LOGFILE" 2>&1
  } &
  pid=$!
  set +m
  printf '%s\n' "$pid" > "$PIDFILE"
  log "tunnel starting (pid $pid); waiting up to ${TIMEOUT}s for the API's /readyz"
  deadline=$((SECONDS + TIMEOUT))
  while :; do
    if ! kill -0 "$pid" 2> /dev/null; then
      rm -f -- "$PIDFILE"
      tail -n 20 -- "$LOGFILE" >&2 || true
      die "the tunnel exited; full log in $LOGFILE"
    fi
    if api_ready; then
      log "tunnel up (pid $pid); API ready at https://127.0.0.1:$LOCAL_PORT"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      kill_tunnel "$pid"
      rm -f -- "$PIDFILE"
      tail -n 20 -- "$LOGFILE" >&2 || true
      die "the API did not answer /readyz through the tunnel within ${TIMEOUT}s; tunnel stopped (log: $LOGFILE)"
    fi
    sleep 1
  done
}

[[ $# -eq 1 ]] || usage
case "$1" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) usage ;;
esac
