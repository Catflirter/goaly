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
# start and stop hold a lock (.run/tunnel.lock), so `make tunnel` during `make up`'s tunnel step
# waits for it instead of launching a second tunnel that would orphan the first.
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
BREAKFILE="$LOCKFILE.break"

usage() {
  log "usage: ${0##*/} start|stop|status"
  exit 2
}

# lock: take $LOCKFILE (it holds this run's PID), released on exit. Created with bash's noclobber,
# which opens with O_EXCL inside the shell: atomic everywhere, unlike flock (absent on macOS) or
# mkdir (uutils coreutils' mkdir, Ubuntu's default from 25.10, is not atomic under contention). A
# lock whose PID is gone (a killed run) is taken over (break_stale_lock); a live holder is waited
# for, at most as long as a start can take, then this run gives up.
lock() {
  local deadline holder said=""
  [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || die "bad TUNNEL_TIMEOUT_S '$TIMEOUT'"
  deadline=$((SECONDS + TIMEOUT + 30))
  run_dir || die "cannot create $RUN_DIR"
  trap unlock EXIT
  until (set -o noclobber; printf '%s\n' "$$" > "$LOCKFILE") 2> /dev/null; do
    holder="$(cat -- "$LOCKFILE" 2> /dev/null || true)"
    if [[ "$holder" =~ ^[1-9][0-9]*$ ]] && ! kill -0 "$holder" 2> /dev/null \
      && break_stale_lock "$holder"; then
      continue
    fi
    if [[ -z "$said" ]]; then
      log "waiting for another tunnel.sh (pid ${holder:-unknown}) to finish"
      said=1
    fi
    (( SECONDS < deadline )) \
      || die "another tunnel.sh still holds $LOCKFILE (pid ${holder:-unknown}); if none is running, remove it and $BREAKFILE"
    sleep 0.2
  done
}

# break_stale_lock DEAD_PID: remove $LOCKFILE if it still holds DEAD_PID; 0 if this run removed it.
# Serialised by a second noclobber lock, $BREAKFILE. Without it, two waiters that read the same dead
# holder could both remove "the stale lock": the later rm deletes the lock the other has just
# taken, and both run start or stop at once (C1-1). Under $BREAKFILE the lock is read again; a dead
# PID never takes a new lock, so a file that still holds it is the stale one. A run that cannot
# take $BREAKFILE (another run is breaking the lock) waits and retries like any other waiter.
break_stale_lock() {
  local dead="$1" removed=1
  (set -o noclobber; printf '%s\n' "$$" > "$BREAKFILE") 2> /dev/null || return 1
  if [[ "$(cat -- "$LOCKFILE" 2> /dev/null || true)" == "$dead" ]]; then
    log "removing a stale tunnel lock (pid $dead is gone)"
    rm -f -- "$LOCKFILE"
    removed=0
  fi
  rm -f -- "$BREAKFILE"
  return "$removed"
}

# unlock (EXIT trap): remove $LOCKFILE and $BREAKFILE, each only while it holds this run's PID, so a
# run never deletes a lock that another run holds.
unlock() {
  local f
  for f in "$LOCKFILE" "$BREAKFILE"; do
    if [[ "$(cat -- "$f" 2> /dev/null || true)" == "$$" ]]; then
      rm -f -- "$f"
    fi
  done
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
  # it runs and ssh's IAP ProxyCommand together; nohup keeps it alive when the terminal closes.
  set -m
  nohup gcloud compute ssh "$CP_NAME" --project "$GCP_PROJECT" --zone "$CP_ZONE" --tunnel-through-iap --quiet \
    -- -N -L "$FORWARD" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o GatewayPorts=no \
    < /dev/null > "$LOGFILE" 2>&1 &
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
