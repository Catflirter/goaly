# shellcheck shell=bash
# Verification harness shared by scripts/verify.sh and the demo (spec section 9). Source it; do not run it.
#
#   vr_init DIR                     start a run: DIR (mode 0700)/SUMMARY.md with an empty results table;
#                                   returns 2 if DIR already holds files
#   run_check ID TITLE FUNC [ARG…]  run one check into DIR/<NN>-<ID>.txt; FUNC returns 0 PASS, 77 SKIP, else FAIL
#   vr_cmd CMD…                     print "$ CMD" (redacted), then run CMD; returns CMD's status
#   vr_cap [--no-echo] CMD…         vr_cmd for $(…): command and output to stderr, stdout to the caller
#   vr_bg FILE CMD…                 print "$ CMD &", run CMD in the background into FILE (VR_BG_PID)
#   vr_retry TIMEOUT_S INTERVAL_S CMD…  retry CMD until it succeeds, for at most TIMEOUT_S seconds
#   redact                          stdin -> stdout filter that replaces secret values with <redacted>
#   vr_finish                       write the totals, scan the run with gitleaks; 1 on any FAIL or leak
#
# The check functions check_<id>_<name> (A–C: Task 18; D: Task 21; E: Task 22) are registered in scripts/verify.sh; the demo (Task 25) calls them directly. Each returns 0
# (PASS), 1 (FAIL) or 77 (SKIP); one that cannot read .env exits 2 (a FAIL), so call it in a subshell.
# Every command a check runs goes through vr_cmd, vr_cap or vr_bg, so its evidence shows what ran
# (node_external_ip, probe_on and echo_on appear by name; their kubectl queries are in common.sh).
# The offline tests set VR_NO_WAIT=1 (no fixed sleeps) and VR_PF_PORT (C3's local port).
# Rules for check functions:
#   - bound every wait (kubectl wait --timeout=…, curl --max-time …): run_check has no timeout of its own;
#   - background jobs (kubectl port-forward &, tcpdump &) are killed when the check returns, because
#     run_check waits for every process holding the check's output; so do not set an EXIT trap in a
#     check (it replaces that cleanup), and do not leave grandchildren (setsid, nohup) running;
#   - cleanup that must run however the check ends (the GPU Pods of D4-D6, E2's leftover GPU
#     instances, E3's extra firewall rule) goes through _vr_guarded, whose EXIT trap lives in a
#     subshell of its own; a cleanup that a second Ctrl-C must not stop ignores SIGINT/SIGTERM and
#     runs its commands through _vr_shielded (a process group of their own, waited for).

# common.sh is linted on its own; pre-commit runs shellcheck without -x, so don't follow it.
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

VR_TZ=America/Los_Angeles

# --- redact -------------------------------------------------------------------------------------

# A secret value: everything up to whitespace, a quote or a comma (the end of a YAML/JSON/shell value).
_vr_value='[^[:space:]"'\'',]+'
# An optional quote around a key or before a value: YAML 'single', JSON/shell "double".
_vr_q="[\"']?"
# Portable ERE (GNU and BSD sed): no \b, no I flag, so case-insensitive words are spelled as [Xx] classes.
# Line-based: a multi-line value is only caught as a PEM private key body (first expression) and a
# YAML block scalar (`token: |`) is not; gitleaks in vr_finish is the backstop for what this misses.
_VR_REDACT_EXPRS=(
  # PEM private key bodies: the base64 lines between -----BEGIN … PRIVATE KEY----- and -----END …-----
  # (first, so the private-key expression below has not rewritten the BEGIN line yet)
  -e "/-----BEGIN [A-Z ]*PRIVATE KEY-----[[:space:]]*\$/,/-----END [A-Z ]*PRIVATE KEY-----/s#^([[:space:]]*)[A-Za-z0-9+/=]+[[:space:]]*\$#\\1<redacted>#"
  # k3s join token K10<ca-hash>::<user>:<password>; the password may hold commas, so stop only at
  # whitespace or a quote
  -e "s/K10[0-9a-fA-F]{64}::[^[:space:]\"']+/<redacted>/g"
  # kubeconfig credentials, service-account / bootstrap tokens, Secret TLS keys and passwords
  # (YAML and JSON forms, either quote)
  -e "s/((client-key-data|client-certificate-data|certificate-authority-data|token|tls\\.key|[Pp]assword)${_vr_q}:[[:space:]]*${_vr_q})${_vr_value}/\\1<redacted>/g"
  # bearer token as a command-line flag (kubectl --token=…, --token …)
  -e "s/(--token[= ]${_vr_q})${_vr_value}/\\1<redacted>/g"
  # Lambda API key assigned or keyed: env (=), YAML (:), JSON ("LAMBDA_API_KEY": "…"), like .gitleaks.toml
  -e "s/(LAMBDA_API_KEY${_vr_q}[[:space:]]*[:=][[:space:]]*${_vr_q})${_vr_value}/\\1<redacted>/g"
  # HTTP credentials (curl -v prints HTTP/2 header names in lower case)
  -e "s/([Aa][Uu][Tt][Hh][Oo][Rr][Ii][Zz][Aa][Tt][Ii][Oo][Nn]:[[:space:]]*([Bb][Ee][Aa][Rr][Ee][Rr]|[Bb][Aa][Ss][Ii][Cc])[[:space:]]+${_vr_q})${_vr_value}/\\1<redacted>/g"
  # WireGuard / TLS private keys: "private key: …", "PrivateKey = …", any case
  -e "s/([Pp][Rr][Ii][Vv][Aa][Tt][Ee][[:space:]_-]?[Kk][Ee][Yy]${_vr_q}[[:space:]]*[:=][[:space:]]*${_vr_q})${_vr_value}/\\1<redacted>/g"
  # Lambda API key by its own prefix (secret_<name>_<hex>[.<random>]), wherever it appears (.gitleaks.toml)
  -e "s/(^|[^A-Za-z0-9_])secret_[A-Za-z0-9_-]+_[0-9a-f]{16,}(\\.[A-Za-z0-9]+)?/\\1<redacted>/g"
)
# Line-buffered, so redacted output reaches a terminal (the demo) as it is produced: GNU sed -u, BSD sed -l.
_VR_SED=(sed)
if sed -u -e '' </dev/null >/dev/null 2>&1; then
  _VR_SED=(sed -u)
elif sed -l -e '' </dev/null >/dev/null 2>&1; then
  _VR_SED=(sed -l)
fi

# LC_ALL=C: match bytes, not characters. All patterns are ASCII, and BSD sed under a UTF-8 locale
# aborts ("illegal byte sequence") on binary or non-UTF-8 output such as tcpdump -A payloads.
redact() {
  LC_ALL=C "${_VR_SED[@]}" -E "${_VR_REDACT_EXPRS[@]}"
}

# --- run bookkeeping ----------------------------------------------------------------------------

# ISO 8601 local time in America/Los_Angeles, e.g. 2026-10-21T14:30:05-07:00.
_vr_now() {
  local ts
  ts="$(TZ="$VR_TZ" date +%Y-%m-%dT%H:%M:%S%z)"
  printf '%s:%s\n' "${ts:0:${#ts}-2}" "${ts: -2}"
}

# vr_init DIR: create DIR and its SUMMARY.md. Refuses a DIR that already holds files (returns 2), so
# the evidence of two runs is never mixed. DIR is 0700 and its files 0600: unreviewed evidence (cluster
# topology, IPs, anything redact missed) stays readable by the operator only; promoting a run to
# verification/final/ is a deliberate copy.
vr_init() {
  local dir="$1"
  if [[ -d "$dir" && -n "$(ls -A -- "$dir")" ]]; then
    log "error: $dir already holds files from another run; start again in a minute (new directory name)"
    return 2
  fi
  (umask 077 && mkdir -p -- "$dir")
  chmod 700 -- "$dir" # mkdir -p keeps the mode of a DIR that already exists (empty)
  VR_DIR="$dir"
  VR_SEQ=0
  VR_FAILS=0
  VR_SKIPS=0
  (
    umask 077
    {
      printf '# Verification run\n\n'
      printf 'Started: %s (%s)\n\n' "$(_vr_now)" "$VR_TZ"
      printf '| # | ID | Check | Result |\n'
      printf '|---|---|---|---|\n'
    } >"$VR_DIR/SUMMARY.md"
  )
}

# EXIT trap of a check's subshell: kill the background jobs it left running (see the header).
_vr_reap() {
  local pids
  pids="$(jobs -p)"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086 # one PID per word
    kill $pids 2>/dev/null || true
  fi
}

# run_check ID TITLE FUNC [ARGS…]: run FUNC ARGS in a subshell (so exit and set -e failures end only
# the check) with stdin from /dev/null, stdout+stderr redacted into $VR_DIR/<NN>-<ID>.txt.
# FUNC's status: 0 PASS, 77 SKIP, anything else FAIL; a failed redact filter (lost evidence) is a FAIL
# too. Returns 0; vr_finish reports the outcome.
# Interrupted: a Ctrl-C reaches the whole process group. The filter ignores SIGINT and SIGTERM (sed
# inherits that) and ends when the check's output does, so a check cleaning up after the signal
# (_vr_guarded) can still record, and so run, its cleanup commands (Task 21a review C0-3/X0-2). This
# shell notes the signal instead of dying of it at once; once the check has ended and its result is
# recorded, it ends itself by that signal (the caller's own trap, if any, runs), so the run stops.
run_check() {
  local id="$1" title="$2" func="$3"
  shift 3
  local seq file rc redact_rc result errexit=0 traps signal=""
  VR_SEQ=$((VR_SEQ + 1))
  printf -v seq '%02d' "$VR_SEQ"
  file="$VR_DIR/$seq-$id.txt"
  (
    umask 077
    {
      printf '# %s — %s\n' "$id" "$title"
      printf '# started: %s\n' "$(_vr_now)"
    } >"$file"
  )

  # The check keeps the caller's errexit; run_check itself must survive the check failing.
  [[ $- == *e* ]] && errexit=1
  set +e
  traps="$(trap -p INT TERM)"
  trap 'signal=${signal:-INT}' INT
  trap 'signal=${signal:-TERM}' TERM
  (
    if ((errexit)); then set -e; fi
    trap _vr_reap EXIT
    "$func" "$@"
  ) </dev/null 2>&1 | (
    trap '' INT TERM
    redact
  ) >>"$file"
  local -a status=("${PIPESTATUS[@]}")
  trap - INT TERM
  eval "$traps"
  if ((errexit)); then set -e; fi
  rc=${status[0]}
  redact_rc=${status[1]}

  # Output that ends without a newline would swallow the footer.
  if [[ -n "$(tail -c 1 -- "$file")" ]]; then
    printf '\n' >>"$file"
  fi
  if ((redact_rc != 0)); then
    printf '# redact failed (exit %d); output incomplete\n' "$redact_rc" >>"$file"
  fi
  if ((redact_rc != 0)) || { ((rc != 0)) && ((rc != 77)); }; then
    result=FAIL
    VR_FAILS=$((VR_FAILS + 1))
  elif ((rc == 77)); then
    result=SKIP
    VR_SKIPS=$((VR_SKIPS + 1))
  else
    result=PASS
  fi
  printf '# finished: %s — %s (exit %d)\n' "$(_vr_now)" "$result" "$rc" >>"$file"
  printf '| %d | %s | %s | %s |\n' "$VR_SEQ" "$id" "${title//|/\\|}" "$result" >>"$VR_DIR/SUMMARY.md"
  printf '[%s] %-4s %-4s %s\n' "$seq" "$id" "$result" "$title"
  if [[ -n "$signal" ]]; then
    printf 'interrupted (SIG%s) during %s: stopping\n' "$signal" "$id" >&2
    kill -s "$signal" "${BASHPID:-$$}"
  fi
  return 0
}

# _vr_quote CMD…: "$ CMD" (not yet redacted) as vr_cmd, vr_cap and vr_bg record it.
# Arguments are shown shell-quoted with single quotes, not printf %q: %q turns "Bearer X" into
# Bearer\ X, which the redact patterns (key, whitespace, value) would no longer recognise.
_vr_quote() {
  local arg line='$'
  for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Za-z0-9_./:=@%+,-]+$ ]]; then
      line+=" $arg"
    else
      line+=" '${arg//\'/\'\\\'\'}'"
    fi
  done
  printf '%s\n' "$line"
}

# vr_cmd CMD…: record the command (redacted) in the check's output, then run it.
vr_cmd() {
  # A failed filter would hide what runs next; do not run it.
  _vr_quote "$@" | redact || return
  "$@"
}

# vr_cap [--no-echo] CMD…: vr_cmd for $(…): the command and its output are recorded on stderr (the
# check's evidence: run_check sends both streams there), and CMD's stdout is printed for the caller
# to parse. --no-echo records the command only, for bulky output (JSON, a packet list) the check
# summarises itself. Returns CMD's status.
vr_cap() {
  local echo=1 out rc=0
  if [[ "${1:-}" == --no-echo ]]; then
    echo=0
    shift
  fi
  _vr_quote "$@" | redact >&2 || return
  out="$("$@")" || rc=$?
  if [[ -n "$out" ]]; then
    if ((echo)); then printf '%s\n' "$out" >&2; fi
    printf '%s\n' "$out"
  fi
  return "$rc"
}

# vr_bg FILE CMD…: record "$ CMD &" and start CMD in the background with its stdout and stderr in
# FILE (the check shows FILE once CMD is done); VR_BG_PID is its PID. The check waits for it or
# stops it with _vr_stop before it returns.
vr_bg() {
  local file="$1" line
  shift
  line="$(_vr_quote "$@")"
  printf '%s &\n' "$line" | redact || return
  "$@" >"$file" 2>&1 </dev/null &
  VR_BG_PID=$!
}

# --- check helpers (Task 18) --------------------------------------------------------------------

# The bound of one kubectl request (get, describe, apply, top); waits carry their own --timeout,
# and exec'd commands bound themselves (curl -m, ping -W, timeout N iperf3).
_VR_RT="--request-timeout=$KC_REQUEST_TIMEOUT"

# vr_env: load .env once per shell (GCP_PROJECT for IAP SSH and gcloud, UNDERLAY_MTU, the Lambda SSH
# key path); a missing or non-private .env ends the check (exit 2, a FAIL). The Lambda API key is
# dropped again: no check needs it, and scripts/gpuctl loads .env itself.
vr_env() {
  [[ "${_VR_ENV_LOADED:-}" == 1 ]] && return 0
  load_env
  unset LAMBDA_API_KEY
  _VR_ENV_LOADED=1
}

# _vr_sleep SECONDS: sleep; VR_NO_WAIT=1 (the offline tests) skips it and says so in the evidence, so
# the tests can see the wait (D6's 30 s).
_vr_sleep() {
  if [[ "${VR_NO_WAIT:-}" == 1 ]]; then
    echo "(wait ${1}s skipped)" >&2
  else
    sleep "$1"
  fi
}

# vr_retry TIMEOUT_S INTERVAL_S CMD…: run CMD (in this shell) until it succeeds, every INTERVAL_S
# seconds, for at most TIMEOUT_S seconds: 1 + TIMEOUT_S/INTERVAL_S attempts, fewer if the attempts
# themselves take long. Returns 1 when it gives up.
vr_retry() {
  local timeout="$1" interval="$2" waited=0 start="$SECONDS"
  shift 2
  while :; do
    "$@" && return 0
    if ((waited + interval > timeout || SECONDS - start + interval > timeout)); then
      printf 'gave up after %ss: %s\n' "$timeout" "$*" >&2
      return 1
    fi
    _vr_sleep "$interval"
    waited=$((waited + interval))
  done
}

# _vr_wait_text FILE TEXT PID SECONDS: wait until FILE (a vr_bg command's output) holds TEXT, i.e. the
# command is ready. Fails when PID has exited without printing it, or after SECONDS.
_vr_wait_text() {
  local i
  for ((i = 0; i < $4 * 5; i++)); do
    grep -qF -- "$2" "$1" 2>/dev/null && return 0
    if ! kill -0 "$3" 2>/dev/null; then
      grep -qF -- "$2" "$1" 2>/dev/null
      return
    fi
    sleep 0.2
  done
  return 1
}

# _vr_stop PID: stop a vr_bg command. A shell function (kc, cp_ssh) runs in a background subshell
# whose child is the real process (kubectl, gcloud), so the children go first; then the job.
_vr_stop() {
  pkill -TERM -P "$1" 2>/dev/null || true
  kill -TERM "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
}

# _vr_show FILE TITLE: print a vr_bg command's output into the evidence, then remove FILE.
_vr_show() {
  printf -- '--- %s ---\n' "$2"
  cat -- "$1"
  rm -f -- "$1"
}

# _vr_load_nodes: VR_NODES = the cluster's node names (kubectl's order), each checked.
_vr_load_nodes() {
  local out n
  VR_NODES=()
  out="$(vr_cap kc get nodes "$_VR_RT" -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}')" || return 1
  while read -r n; do
    [[ -n "$n" ]] || continue
    [[ "$n" =~ $K8S_NAME_RE ]] || { echo "unexpected node name from the cluster: '$n'"; return 1; }
    VR_NODES+=("$n")
  done <<< "$out"
  ((${#VR_NODES[@]})) || { echo "the cluster lists no nodes"; return 1; }
}

# _vr_load_pods NAMESPACE APP: VR_PODS = "NODE POD IP" for each Running Pod labelled app=APP.
_vr_load_pods() {
  local out node name ip
  VR_PODS=()
  out="$(vr_cap kc -n "$1" get pods -l "app=$2" --field-selector=status.phase=Running "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.spec.nodeName}{" "}{.metadata.name}{" "}{.status.podIP}{"\n"}{end}')" || return 1
  while read -r node name ip; do
    [[ -n "$node" ]] || continue
    if [[ ! "$node" =~ $K8S_NAME_RE || ! "$name" =~ $K8S_NAME_RE ]] || ! is_ipv4 "$ip"; then
      echo "unexpected $2 Pod from the cluster: '$node $name $ip'"
      return 1
    fi
    VR_PODS+=("$node $name $ip")
  done <<< "$out"
}

# _vr_pick NODE LINE…: "POD IP" of the first "NODE POD IP" line on NODE; fails if there is none.
_vr_pick() {
  local want="$1" node name ip
  shift
  for line in "$@"; do
    read -r node name ip <<< "$line"
    if [[ "$node" == "$want" ]]; then
      printf '%s %s\n' "$name" "$ip"
      return 0
    fi
  done
  return 1
}

# _vr_in WORD LIST…: WORD is one of LIST.
_vr_in() {
  local want="$1" w
  shift
  for w in "$@"; do
    [[ "$w" == "$want" ]] && return 0
  done
  return 1
}

# _vr_ssh_prepare NODE…: look up the SSH zones once, in the check's own shell: cp_resolve and
# _worker_zone cache there only, so node_ssh in $(…) would otherwise read Terraform outputs each time.
_vr_ssh_prepare() {
  local n
  for n in "$@"; do
    if [[ "$n" == "$CP_NAME" ]]; then
      cp_resolve
    elif [[ "$n" =~ $CPU_NODE_RE ]]; then
      _worker_zone "$n"
    fi
  done
}

# _vr_container_id ID: a containerd container ID without its scheme (containerd://<64 hex>).
_vr_container_id() {
  local cid="${1#*://}"
  [[ "$cid" =~ ^[0-9a-f]{64}$ ]] || { echo "not a container ID: '$1'" >&2; return 1; }
  printf '%s\n' "$cid"
}

# _vr_cgroup_dir NODE CID: the cgroup directory of container CID on NODE (spec §9 C5).
_vr_cgroup_dir() {
  local out dir
  out="$(vr_cap node_ssh "$1" "sudo find /sys/fs/cgroup -type d -name \"*$2*\"" </dev/null)" || return 1
  dir="${out%%$'\n'*}"
  [[ "$dir" =~ ^/sys/fs/cgroup/[A-Za-z0-9_.:@/-]+$ ]] \
    || { echo "no cgroup directory for container $2 on $1" >&2; return 1; }
  printf '%s\n' "$dir"
}

# --- A: the cluster (spec §9 A) -----------------------------------------------------------------

# A1: every node Ready, and at least 2 of them.
check_a1_nodes() {
  local out name status total=0 ready=0
  vr_cmd kc get nodes -o wide "$_VR_RT" || return 1
  out="$(vr_cap kc get nodes "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{" "}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}')" \
    || return 1
  while read -r name status; do
    [[ -n "$name" ]] || continue
    total=$((total + 1))
    if [[ "$status" == True ]]; then
      ready=$((ready + 1))
    else
      echo "A1: node $name is not Ready (Ready=${status:-unknown})"
    fi
  done <<< "$out"
  echo "A1: $ready of $total nodes Ready (want all of them, and at least 2)"
  ((total >= 2 && ready == total))
}

# A2: kubectl describe node succeeds for every node.
check_a2_describe_nodes() {
  local n fail=0
  _vr_load_nodes || return 1
  for n in "${VR_NODES[@]}"; do
    vr_cmd kc describe node "$n" "$_VR_RT" || { echo "A2: describe node $n failed"; fail=1; }
  done
  return "$fail"
}

# A3: metrics-server and CoreDNS run on k3s-cp and are Ready.
check_a3_system_pods() {
  local out app name node ready ms=0 dns=0 fail=0
  vr_cmd kc -n kube-system get pods -o wide "$_VR_RT" || return 1
  out="$(vr_cap kc -n kube-system get pods -l 'k8s-app in (metrics-server,kube-dns)' "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.labels.k8s-app}{" "}{.metadata.name}{" "}{.spec.nodeName}{" "}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}')" \
    || return 1
  while read -r app name node ready; do
    case "$app" in
      metrics-server) ms=$((ms + 1)) ;;
      kube-dns) dns=$((dns + 1)) ;;
      *) continue ;;
    esac
    if [[ "$node" != "$CP_NAME" || "$ready" != True ]]; then
      echo "A3: $name ($app) is on '$node', Ready=${ready:-unknown}; want $CP_NAME and True"
      fail=1
    fi
  done <<< "$out"
  ((ms > 0)) || { echo "A3: no metrics-server Pod"; fail=1; }
  ((dns > 0)) || { echo "A3: no CoreDNS Pod"; fail=1; }
  ((fail)) || echo "A3: $ms metrics-server and $dns CoreDNS Pod(s), all Ready on $CP_NAME"
  return "$fail"
}

# A4: every node's flannel.alpha.coreos.com/* annotations; the public IP flannel uses
# (public-ip-overwrite when it is set, else public-ip) equals the node's ExternalIP.
check_a4_flannel_annotations() {
  local json report
  json="$(vr_cap --no-echo kc get nodes "$_VR_RT" -o json)" || return 1
  report="$(jq -r '
    .items[] | (.metadata.annotations // {}) as $a
    | ([.status.addresses[]? | select(.type == "ExternalIP") | .address] | first // "") as $ext
    | ($a["flannel.alpha.coreos.com/public-ip-overwrite"] // $a["flannel.alpha.coreos.com/public-ip"] // "") as $pub
    | "node \(.metadata.name) ExternalIP=\($ext)",
      ($a | to_entries[] | select(.key | startswith("flannel.alpha.coreos.com/")) | "  \(.key)=\(.value)"),
      (if $ext != "" and $pub == $ext then "  OK: public IP \($pub) = ExternalIP"
       else "  MISMATCH: public IP '\''\($pub)'\'' != ExternalIP '\''\($ext)'\''" end)' <<< "$json")" \
    || { echo "A4: cannot read the nodes"; return 1; }
  printf '%s\n' "$report"
  [[ "$report" == *"node "* && "$report" != *MISMATCH* ]]
}

# --- B: connectivity (spec §9 B) ----------------------------------------------------------------

# _vr_b1_cell PROBE ECHO_IP SMALL BIG: one matrix cell (spec §2.3 #3: a Pod-to-Pod ICMP payload of
# UNDERLAY_MTU-108 fits, UNDERLAY_MTU-107 does not). Sets VR_CELL to ok or the failed parts.
_vr_b1_cell() {
  VR_CELL=""
  vr_cmd kc -n nettest exec "$1" -- curl -fsS -m 5 "http://$2:8080/" || VR_CELL+="curl,"
  vr_cmd kc -n nettest exec "$1" -- ping -M 'do' -c 2 -W 2 -s "$3" "$2" || VR_CELL+="ping-$3,"
  if vr_cmd kc -n nettest exec "$1" -- ping -M 'do' -c 2 -W 2 -s "$4" "$2"; then
    VR_CELL+="ping-$4-passed,"
  fi
  VR_CELL="${VR_CELL%,}"
  VR_CELL="${VR_CELL:-ok}"
}

# B1: every probe -> every echo Pod IP (curl, ping that fits, ping that must not), and every probe ->
# the echo Service by DNS; printed as an N x N table (rows: probes, columns: echo Pods by node).
check_b1_matrix() {
  local mtu small big pn en p e name ip row fail=0
  local -a probes=() echoes=() rows=()
  vr_env
  mtu="${UNDERLAY_MTU:-1460}"
  [[ "$mtu" =~ ^[1-9][0-9]{2,3}$ ]] || { echo "B1: UNDERLAY_MTU in .env is not an MTU: '$mtu'"; return 1; }
  small=$((mtu - 108))
  big=$((mtu - 107))
  _vr_load_nodes || return 1
  _vr_load_pods nettest probe || return 1
  probes=(${VR_PODS[@]+"${VR_PODS[@]}"})
  _vr_load_pods nettest echo || return 1
  echoes=(${VR_PODS[@]+"${VR_PODS[@]}"})
  for pn in "${VR_NODES[@]}"; do
    _vr_pick "$pn" ${probes[@]+"${probes[@]}"} >/dev/null || { echo "B1: $pn: no Running probe Pod"; fail=1; }
    _vr_pick "$pn" ${echoes[@]+"${echoes[@]}"} >/dev/null || { echo "B1: $pn: no Running echo Pod"; fail=1; }
  done
  # Rows are tab-separated here and aligned by _vr_table at the end.
  row='probe \ echo'
  for en in "${VR_NODES[@]}"; do row+=$'\t'"$en"; done
  rows+=("$row"$'\tsvc-dns')
  for pn in "${VR_NODES[@]}"; do
    row="$pn"
    if ! p="$(_vr_pick "$pn" ${probes[@]+"${probes[@]}"})"; then
      for en in "${VR_NODES[@]}" svc-dns; do row+=$'\tmissing'; done
      rows+=("$row")
      continue
    fi
    read -r name ip <<< "$p"
    for en in "${VR_NODES[@]}"; do
      if e="$(_vr_pick "$en" ${echoes[@]+"${echoes[@]}"})"; then
        _vr_b1_cell "$name" "${e#* }" "$small" "$big"
      else
        VR_CELL=missing
      fi
      [[ "$VR_CELL" == ok ]] || fail=1
      row+=$'\t'"$VR_CELL"
    done
    if vr_cmd kc -n nettest exec "$name" -- curl -fsS -m 5 http://echo.nettest.svc.cluster.local:8080/; then
      row+=$'\tok'
    else
      row+=$'\tFAIL'
      fail=1
    fi
    rows+=("$row")
  done
  echo
  echo "B1: N x N matrix, UNDERLAY_MTU=$mtu: each cell is curl to the echo Pod IP, ping -M do -s $small"
  echo "(must pass) and -s $big (must fail); svc-dns is curl http://echo.nettest.svc.cluster.local:8080/"
  printf '%s\n' "${rows[@]}" | _vr_table
  return "$fail"
}

# _vr_b2_flows PEER_IP…: read `tcpdump -nn -q -t` lines on stdin and summarise them: the udp/51820
# packets (any host), and the traffic with the other nodes' public IPs by service port. Only
# 51820/udp (WireGuard), 6443/tcp (the API, mTLS) and 10250/tcp (kubelet, TLS) may appear there
# (spec §2.2: the two TLS ports cross the internet by design); ICMP carries no port and is listed.
# Fails on any other port or protocol, or when no udp/51820 packet was seen.
_vr_b2_flows() {
  awk -v peers="$*" '
    BEGIN { n = split(peers, p, " "); for (i = 1; i <= n; i++) peer[p[i]] = 1 }
    function host(a,  f) { split(a, f, "."); return f[1] "." f[2] "." f[3] "." f[4] }
    function port(a,  f) { return split(a, f, ".") == 5 ? f[5] : "" }
    $1 == "IP" {
      dst = $4; sub(/:$/, "", dst)
      proto = ($5 == "tcp") ? "tcp" : ($5 ~ /^UDP/) ? "udp" : ($5 == "ICMP") ? "icmp" : "other"
      sh = host($2); dh = host(dst); sp = port($2); dp = port(dst)
      if (proto == "udp" && (sp == 51820 || dp == 51820)) wg++
      if (!(sh in peer) && !(dh in peer)) next
      if (proto == "udp" && (sp == 51820 || dp == 51820)) key = "51820/udp"
      else if (proto == "tcp" && (dp == 6443 || dp == 10250)) key = dp "/tcp"
      else if (proto == "tcp" && (sp == 6443 || sp == 10250)) key = sp "/tcp"
      else if (proto == "icmp") key = "icmp"
      else { bad[proto " " sp " > " dp]++; nbad++; next }
      seen[key]++
    }
    END {
      printf "B2 ②: %d udp/51820 packets in the capture\n", wg
      if (wg < 1) print "B2 ②: no udp/51820 packet: no WireGuard traffic was captured"
      printf "B2 ③: packets to or from the other nodes'\'' public IPs (%s):\n", peers
      split("51820/udp 6443/tcp 10250/tcp icmp", k, " ")
      for (i = 1; i <= 4; i++) if (seen[k[i]]) printf "  %-10s %d\n", k[i], seen[k[i]]
      for (b in bad) printf "  UNEXPECTED %s: %d packets\n", b, bad[b]
      if (nbad == 0) print "B2 ③: only 51820/udp, 6443/tcp and 10250/tcp"
      exit (wg >= 1 && nbad == 0) ? 0 : 1
    }'
}

# B2: capture on k3s-cp's default interface while the probes on the other nodes send 20 requests
# each, carrying a random marker header, to the echo Pod on k3s-cp. PASS: ① the marker is not in
# the capture's payloads; ② there is udp/51820; ③ only the expected ports cross to the other nodes'
# public IPs. The capture file is removed afterwards, pass or fail.
check_b2_wire() {
  local marker route iface n ip probe cp_echo capf pid started=0 rc=0 out matches fail=0
  local -a peers=() probes=()
  vr_env
  cp_resolve
  marker="b2-marker-$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')"
  [[ "$marker" =~ ^b2-marker-[0-9a-f]{16}$ ]] || { echo "B2: cannot make a random marker"; return 1; }
  echo "B2: marker $marker"
  route="$(vr_cap cp_ssh 'ip -o route show default' </dev/null)" || return 1
  iface="$(awk '{ for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit } }' <<< "$route")"
  [[ "$iface" =~ ^[A-Za-z0-9_.-]{1,15}$ ]] || { echo "B2: no default interface on $CP_NAME in: $route"; return 1; }
  _vr_load_nodes || return 1
  for n in "${VR_NODES[@]}"; do
    [[ "$n" != "$CP_NAME" ]] || continue
    ip="$(vr_cap node_external_ip "$n")" || return 1
    probe="$(vr_cap probe_on "$n")" || return 1
    peers+=("$ip")
    probes+=("$probe")
  done
  ((${#probes[@]})) || { echo "B2: no node other than $CP_NAME to send from"; return 1; }
  cp_echo="$(vr_cap echo_on "$CP_NAME")" || return 1

  capf="$(mktemp)" || return 1
  vr_bg "$capf" cp_ssh "sudo timeout 25 tcpdump -ni $iface -U -w /tmp/b2.pcap 'not port 22'"
  pid="$VR_BG_PID"
  if _vr_wait_text "$capf" "listening on" "$pid" 60; then
    started=1
    for probe in "${probes[@]}"; do
      vr_cmd kc -n nettest exec "$probe" -- sh -c "n=0; for i in \$(seq 20); do curl -fsS -m 2 -o /dev/null -H \"X-Marker: $marker\" http://$cp_echo:8080/ && n=\$((n + 1)); done; echo \"\$n/20 requests answered\"; [ \"\$n\" -eq 20 ]" \
        || { echo "B2: not all 20 requests from $probe were answered"; fail=1; }
    done
    wait "$pid" || rc=$?
  else
    _vr_stop "$pid"
    echo "B2: the capture on $CP_NAME did not start"
    fail=1
  fi
  _vr_show "$capf" "capture on $CP_NAME"
  if ((started)); then
    if ((rc != 0 && rc != 124)); then
      echo "B2: the capture on $CP_NAME failed (exit $rc)"
      fail=1
    fi
    rc=0
    out="$(vr_cap cp_ssh "sudo tcpdump -A -nn -r /tmp/b2.pcap | grep -c -F -- '$marker'" </dev/null)" || rc=$?
    matches="${out##*$'\n'}"
    if ((rc <= 1)) && [[ "$matches" == 0 ]]; then
      echo "B2 ①: the marker is not in the capture (the cross-node traffic is encrypted)"
    elif ((rc <= 1)) && [[ "$matches" =~ ^[0-9]+$ ]]; then
      echo "B2 ①: MARKER found in the capture $matches times: Pod traffic crossed the wire in plaintext"
      fail=1
    else
      echo "B2 ①: cannot search the capture (exit $rc)"
      fail=1
    fi
    if out="$(vr_cap --no-echo cp_ssh 'sudo tcpdump -nn -q -t -r /tmp/b2.pcap' </dev/null)"; then
      _vr_b2_flows "${peers[@]}" <<< "$out" || fail=1
    else
      echo "B2: cannot read the capture"
      fail=1
    fi
  fi
  vr_cmd cp_ssh 'sudo rm -f /tmp/b2.pcap' </dev/null || { echo "B2: cannot remove /tmp/b2.pcap on $CP_NAME"; fail=1; }
  return "$fail"
}

# _vr_wg_peers NODE WANT: read `wg show flannel-wg` on stdin; NODE must have WANT peers, each with a
# latest handshake at most 3 minutes old. The keys are not printed again (wg show's own output is).
_vr_wg_peers() {
  awk -v node="$1" -v want="$2" '
    function secs(s,  f, n, i, t, u) {
      if (s ~ /^Now/) return 0
      n = split(s, f, /[ ,]+/)
      t = 0
      for (i = 1; i < n; i++) {
        if (f[i] !~ /^[0-9]+$/) continue
        u = f[i + 1]
        if (u ~ /^year/) t += f[i] * 31536000
        else if (u ~ /^day/) t += f[i] * 86400
        else if (u ~ /^hour/) t += f[i] * 3600
        else if (u ~ /^minute/) t += f[i] * 60
        else if (u ~ /^second/) t += f[i]
        else return -1
      }
      return t
    }
    /^peer:/ { p++; hs[p] = -1; ep[p] = "?" }
    p && /^ *endpoint:/ { ep[p] = $2 }
    p && /^ *latest handshake:/ { s = $0; sub(/^ *latest handshake: */, "", s); hs[p] = secs(s); txt[p] = s }
    END {
      bad = (p != want)
      printf "B3: %s: %d peers (want %d)\n", node, p, want
      for (i = 1; i <= p; i++) {
        if (hs[i] < 0) { printf "  peer %s: no handshake\n", ep[i]; bad = 1 }
        else if (hs[i] > 180) { printf "  peer %s: latest handshake %s: older than 3 minutes\n", ep[i], txt[i]; bad = 1 }
        else printf "  peer %s: latest handshake %d s ago\n", ep[i], hs[i]
      }
      exit bad
    }'
}

# B3: `wg show flannel-wg` on every node (never `dump`, which prints the private key): node count - 1
# peers, each with a handshake in the last 3 minutes.
check_b3_wg() {
  local n out want fail=0
  vr_env
  _vr_load_nodes || return 1
  want=$((${#VR_NODES[@]} - 1))
  _vr_ssh_prepare "${VR_NODES[@]}"
  for n in "${VR_NODES[@]}"; do
    if ! out="$(vr_cap node_ssh "$n" 'sudo wg show flannel-wg' </dev/null)"; then
      echo "B3: $n: wg show failed"
      fail=1
      continue
    fi
    _vr_wg_peers "$n" "$want" <<< "$out" || fail=1
  done
  return "$fail"
}

# _vr_b4_pair CLIENT_PROBE CLIENT_NODE SERVER_NODE: iperf3 from CLIENT_PROBE to the probe on
# SERVER_NODE (-n 50M: a bounded transfer, spec §9 B4), and record bits_per_second.
_vr_b4_pair() {
  local sprobe sip logf pid json rc=0 sent recv mbit
  sprobe="$(vr_cap probe_on "$3")" || return 1
  sip="$(vr_cap kc -n nettest get pod "$sprobe" "$_VR_RT" -o 'jsonpath={.status.podIP}')" || return 1
  is_ipv4 "$sip" || { echo "B4: $sprobe has no Pod IP: '$sip'"; return 1; }
  logf="$(mktemp)" || return 1
  vr_bg "$logf" kc -n nettest exec "$sprobe" -- timeout 150 iperf3 -s -1
  pid="$VR_BG_PID"
  if _vr_wait_text "$logf" "Server listening" "$pid" 60; then
    json="$(vr_cap --no-echo kc -n nettest exec "$1" -- timeout 120 iperf3 -c "$sip" -n 50M -J)" || rc=$?
  else
    echo "B4: the iperf3 server on $3 did not start"
    rc=1
  fi
  if ((rc == 0)); then wait "$pid" || true; else _vr_stop "$pid"; fi
  _vr_show "$logf" "iperf3 server on $3 ($sprobe)"
  if ((rc != 0)); then
    echo "B4: iperf3 $2 -> $3 failed (exit $rc)"
    [[ -z "${json:-}" ]] || printf '%s\n' "$json"
    return 1
  fi
  sent="$(jq -r '.end.sum_sent.bits_per_second | floor' <<< "$json" 2>/dev/null)" || sent=""
  recv="$(jq -r '.end.sum_received.bits_per_second | floor' <<< "$json" 2>/dev/null)" || recv=""
  [[ "$sent" =~ ^[0-9]+$ && "$recv" =~ ^[0-9]+$ ]] || { echo "B4: no bits_per_second in iperf3's JSON"; return 1; }
  mbit=$((recv / 1000000))
  echo "B4: $2 -> $3: sum_sent bits_per_second=$sent, sum_received bits_per_second=$recv (about $mbit Mbit/s)"
}

# B4: iperf3 from the probe on k3s-cpu-1 to the probe on k3s-cp, and to each GPU node's probe.
check_b4_iperf() {
  local client=k3s-cpu-1 cprobe n fail=0
  local -a servers=("$CP_NAME")
  _vr_load_nodes || return 1
  for n in "${VR_NODES[@]}"; do
    [[ ! "$n" =~ $GPU_NODE_RE ]] || servers+=("$n")
  done
  cprobe="$(vr_cap probe_on "$client")" || return 1
  for n in "${servers[@]}"; do
    _vr_b4_pair "$cprobe" "$client" "$n" || fail=1
  done
  return "$fail"
}

# B5: what must fail does. ① this machine cannot open tcp 6443 or 10250 on k3s-cp's public IP;
# ② a Pod on a GCP node cannot reach the metadata server (curl itself must fail: exit 28 timeout or
# 7 refused; any other code means kubectl or the probe failed, so nothing was proven); ③ DNS works
# from every probe; ④ the k3s_guard table (with its counters) on every node.
check_b5_negative() {
  local cp_ip port n p name ip rc fail=0
  vr_env
  cp_ip="$(vr_cap node_external_ip "$CP_NAME")" || return 1
  for port in 6443 10250; do
    if vr_cmd timeout 3 bash -c "</dev/tcp/$cp_ip/$port"; then
      echo "B5 ①: tcp/$port on $CP_NAME ($cp_ip) is reachable from this machine"
      fail=1
    else
      echo "B5 ①: tcp/$port on $CP_NAME is not reachable from this machine (expected)"
    fi
  done
  _vr_load_nodes || return 1
  _vr_load_pods nettest probe || return 1
  for n in "${VR_NODES[@]}"; do
    if ! p="$(_vr_pick "$n" ${VR_PODS[@]+"${VR_PODS[@]}"})"; then
      echo "B5: $n: no Running probe Pod"
      fail=1
      continue
    fi
    read -r name ip <<< "$p"
    if [[ "$n" == "$CP_NAME" || "$n" =~ $CPU_NODE_RE ]]; then
      rc=0
      vr_cmd kc -n nettest exec "$name" -- curl -m 3 -H 'Metadata-Flavor: Google' \
        http://169.254.169.254/computeMetadata/v1/ || rc=$?
      case "$rc" in
        7 | 28) echo "B5 ②: $n: a Pod cannot reach the metadata server (curl exit $rc, expected)" ;;
        0) echo "B5 ②: $n: a Pod reached the metadata server"; fail=1 ;;
        *) echo "B5 ②: $n: exit $rc: kubectl or curl failed before a verdict (want curl's 28 or 7)"; fail=1 ;;
      esac
    fi
    vr_cmd kc -n nettest exec "$name" -- nslookup kubernetes.default.svc.cluster.local \
      || { echo "B5 ③: $n: the DNS lookup failed"; fail=1; }
  done
  _vr_ssh_prepare "${VR_NODES[@]}"
  for n in "${VR_NODES[@]}"; do
    vr_cmd node_ssh "$n" 'sudo nft list table inet k3s_guard' </dev/null \
      || { echo "B5 ④: $n: cannot list the k3s_guard table"; fail=1; }
  done
  return "$fail"
}

# _vr_table: tab-separated lines on stdin as aligned columns.
_vr_table() {
  awk -F'\t' '
    { for (i = 1; i <= NF; i++) { c[NR, i] = $i; if (length($i) > w[i]) w[i] = length($i) }
      if (NF > m) m = NF }
    END { for (r = 1; r <= NR; r++) { line = ""
            for (i = 1; i <= m; i++) line = line sprintf("%-" (w[i] + 2) "s", c[r, i])
            sub(/ +$/, "", line); print line } }'
}

# B6: the GCP firewall rules named fw-* as a table, and gpuctl's Lambda rules. Every source must be
# one address (/32), except Google's IAP range 35.235.240.0/20. An ingress rule without source
# ranges admits every address on GCP, so it fails too, as does a tag or service-account source.
check_b6_firewall_inventory() {
  local json table violations rules count fail=0
  vr_env
  json="$(vr_cap --no-echo gcloud compute firewall-rules list --project "$GCP_PROJECT" --filter='name~^fw-' \
    --format=json)" || return 1
  table="$(jq -r '
    (["NAME", "NETWORK", "DIRECTION", "SOURCES", "ALLOWED", "TARGET_SA"] | @tsv),
    (.[] | [.name, ((.network // "-") | split("/") | last), (.direction // "INGRESS"),
            ((.sourceRanges // []) | if length == 0 then "-" else join(",") end),
            ((.allowed // []) | map(.IPProtocol + (if .ports then ":" + (.ports | join(";")) else "" end)) | join(",")),
            ((.targetServiceAccounts // []) | if length == 0 then "-" else join(",") end)] | @tsv)' <<< "$json")" \
    || { echo "B6: cannot read gcloud's answer"; return 1; }
  printf '%s\n' "$table" | _vr_table
  count="$(jq 'length' <<< "$json")"
  ((count > 0)) || { echo "B6: no firewall rule named fw-* in $GCP_PROJECT"; fail=1; }
  violations="$(jq -r '
    .[] | select((.direction // "INGRESS") == "INGRESS") | . as $r
    | (if ((.sourceRanges // []) | length) == 0
       then "VIOLATION \(.name): no source ranges (GCP would admit every address)" else empty end),
      (if (((.sourceTags // []) + (.sourceServiceAccounts // [])) | length) > 0
       then "VIOLATION \(.name): a source that is not an address range" else empty end),
      ((.sourceRanges // [])[] | select((endswith("/32") | not) and . != "35.235.240.0/20")
       | "VIOLATION \($r.name): source \(.) is not a single address (/32)")' <<< "$json")" \
    || { echo "B6: cannot read gcloud's answer"; return 1; }

  rules="$(vr_cap "$REPO_ROOT/scripts/gpuctl" rules </dev/null)" \
    || { echo "B6: scripts/gpuctl rules failed"; fail=1; }
  violations+=$'\n'"$(awk '
    /^[^ ]/ { section = $0; next }
    /^  / && $1 != "(no" && $3 !~ /\/32$/ && $3 != "35.235.240.0/20" {
      print "VIOLATION Lambda " section ": source " $3 " is not a single address (/32)" }' <<< "${rules:-}")"
  if [[ -n "${violations//$'\n'/}" ]]; then
    printf '%s\n' "$violations" | sed '/^$/d'
    fail=1
  elif ((count > 0)); then
    echo "B6: every source is a single address (/32) or the IAP range 35.235.240.0/20"
  fi
  return "$fail"
}

# --- C: hello-world (spec §9 C) -----------------------------------------------------------------

# C1: the hello Deployment is rolled out; its Pods.
check_c1_rollout() {
  vr_cmd kc -n hello rollout status deploy/hello --timeout=120s || return 1
  vr_cmd kc -n hello get pods -o wide "$_VR_RT"
}

# C2: from a probe on another node than the hello replicas (one on a node with no replica if there
# is one, else one on a node other than some replica), the Service answers exactly "hello world".
check_c2_cross_node_curl() {
  local out h line node name ip probe="" pnode="" body rc=0
  local -a hello=()
  out="$(vr_cap kc -n hello get pods -l app=hello --field-selector=status.phase=Running "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.spec.nodeName}{"\n"}{end}')" || return 1
  while read -r h; do
    [[ -z "$h" ]] || hello+=("$h")
  done <<< "$out"
  ((${#hello[@]})) || { echo "C2: no Running hello Pod"; return 1; }
  _vr_load_pods nettest probe || return 1
  for line in ${VR_PODS[@]+"${VR_PODS[@]}"}; do
    read -r node name ip <<< "$line"
    if ! _vr_in "$node" "${hello[@]}"; then probe="$name"; pnode="$node"; break; fi
  done
  if [[ -z "$probe" ]]; then
    for line in ${VR_PODS[@]+"${VR_PODS[@]}"}; do
      read -r node name ip <<< "$line"
      for h in "${hello[@]}"; do
        if [[ "$h" != "$node" ]]; then probe="$name"; pnode="$node"; break 2; fi
      done
    done
  fi
  [[ -n "$probe" ]] || { echo "C2: no probe runs on a node other than a hello replica's"; return 1; }
  echo "C2: hello replicas on: ${hello[*]}; asking from $probe on $pnode"
  body="$(vr_cap kc -n nettest exec "$probe" -- curl -fsS -m 5 http://hello.hello.svc.cluster.local)" || rc=$?
  if ((rc == 0)) && [[ "$body" == "hello world" ]]; then
    echo "C2: answered exactly 'hello world'"
    return 0
  fi
  echo "C2: curl exit $rc; want exactly 'hello world'"
  return 1
}

# C3: kubectl port-forward svc/hello (in the background), curl it on localhost, stop the forward.
# VR_PF_PORT (default 18080) is the local port; the offline tests pick a free one.
check_c3_port_forward() {
  local port="${VR_PF_PORT:-18080}" logf pid body rc=0
  [[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "C3: VR_PF_PORT is not a port: '$port'"; return 1; }
  logf="$(mktemp)" || return 1
  vr_bg "$logf" kc -n hello port-forward svc/hello "$port:80"
  pid="$VR_BG_PID"
  if _vr_wait_text "$logf" "Forwarding from" "$pid" 30; then
    body="$(vr_cap curl -fsS -m 5 "localhost:$port")" || rc=$?
  else
    echo "C3: the port-forward did not start"
    rc=1
  fi
  _vr_stop "$pid"
  echo "C3: port-forward stopped"
  _vr_show "$logf" "port-forward"
  if ((rc == 0)) && [[ "$body" == "hello world" ]]; then
    echo "C3: localhost:$port answered exactly 'hello world'"
    return 0
  fi
  echo "C3: curl exit $rc; want exactly 'hello world'"
  return 1
}

# C4: every hello Pod's QoS class is Guaranteed.
check_c4_qos() {
  local out
  out="$(vr_cap kc -n hello get pods -l app=hello "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{" "}{.status.qosClass}{"\n"}{end}')" || return 1
  awk 'NF { n++; if ($2 != "Guaranteed") { print "C4: " $1 " is " ($2 == "" ? "unknown" : $2); bad = 1 } }
       END { if (!n) { print "C4: no hello Pod"; exit 1 }
             if (!bad) print "C4: all " n " hello Pods are Guaranteed"; exit bad }' <<< "$out"
}

# C5: one hello container's cgroup on its node: memory.max = 67108864 (64Mi), cpu.max = 10000 100000
# (100m).
check_c5_cgroup() {
  local out line pod node cid="" dir vals mem cpu fail=0
  vr_env
  out="$(vr_cap kc -n hello get pods -l app=hello --field-selector=status.phase=Running "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{" "}{.spec.nodeName}{" "}{.status.containerStatuses[0].containerID}{"\n"}{end}')" \
    || return 1
  while read -r pod node line; do
    if [[ -n "$line" ]]; then cid="$line"; break; fi
  done <<< "$out"
  [[ -n "$cid" ]] || { echo "C5: no hello container ID"; return 1; }
  cid="$(_vr_container_id "$cid")" || return 1
  [[ "$node" =~ $K8S_NAME_RE ]] || { echo "C5: unexpected node name: '$node'"; return 1; }
  echo "C5: $pod on $node, container $cid"
  _vr_ssh_prepare "$node"
  dir="$(_vr_cgroup_dir "$node" "$cid")" || return 1
  vals="$(vr_cap node_ssh "$node" "cat $dir/memory.max $dir/cpu.max" </dev/null)" || return 1
  mem="$(sed -n 1p <<< "$vals")"
  cpu="$(sed -n 2p <<< "$vals")"
  [[ "$mem" == 67108864 ]] || { echo "C5: memory.max is '$mem'; want 67108864"; fail=1; }
  [[ "$cpu" == "10000 100000" ]] || { echo "C5: cpu.max is '$cpu'; want '10000 100000'"; fail=1; }
  ((fail)) || echo "C5: memory.max 67108864, cpu.max 10000 100000"
  return "$fail"
}

_vr_c6_top() {
  local out
  out="$(vr_cap kc -n hello top pod "$_VR_RT")" || return 1
  [[ -n "$(awk 'NR > 1 && NF' <<< "$out")" ]]
}

# C6: kubectl top pod in hello returns a data row within 90 s (metrics-server needs a scrape first).
check_c6_top() {
  if vr_retry 90 10 _vr_c6_top; then
    echo "C6: kubectl top pod returned data"
    return 0
  fi
  echo "C6: no data from kubectl top pod within 90 s"
  return 1
}

# _vr_millicores CPU: a kubectl top CPU reading (100m, 1) in millicores.
_vr_millicores() {
  if [[ "$1" =~ ^([0-9]+)m$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "$(($1 * 1000))"
  else
    return 1
  fi
}

_vr_c7_oom_done() {
  VR_OOM="$(vr_cap kc -n limits-demo get pod oom "$_VR_RT" \
    -o 'jsonpath={.status.containerStatuses[0].state.terminated.reason}{" "}{.status.containerStatuses[0].state.terminated.exitCode}')" \
    || return 1
  [[ -n "${VR_OOM// /}" ]]
}

_vr_c7_top() {
  local out
  out="$(vr_cap kc -n limits-demo top pod cpu-burn "$_VR_RT")" || return 1
  VR_TOP_CPU="$(awk '$1 == "cpu-burn" { print $2 }' <<< "$out")"
  [[ -n "$VR_TOP_CPU" ]]
}

# _vr_throttled NODE DIR: nr_throttled from DIR/cpu.stat on NODE.
_vr_throttled() {
  local out v
  out="$(vr_cap node_ssh "$1" "cat $2/cpu.stat" </dev/null)" || return 1
  v="$(awk '$1 == "nr_throttled" { print $2 }' <<< "$out")"
  [[ "$v" =~ ^[0-9]+$ ]] || { echo "no nr_throttled in $2/cpu.stat on $1" >&2; return 1; }
  printf '%s\n' "$v"
}

_vr_c7() {
  local reason code milli line node cid dir t1 t2 fail=0
  vr_cmd kc apply -f "$REPO_ROOT/k8s/limits-demo/limits-demo.yaml" "$_VR_RT" || return 1
  if vr_retry 90 5 _vr_c7_oom_done; then
    read -r reason code <<< "$VR_OOM"
    if [[ "$reason" == OOMKilled && "$code" == 137 ]]; then
      echo "C7 ①: oom was OOMKilled, exit code 137"
    else
      echo "C7 ①: oom ended with reason '$reason', exit code '$code'; want OOMKilled and 137"
      fail=1
    fi
  else
    echo "C7 ①: oom did not terminate within 90 s"
    fail=1
  fi

  vr_cmd kc -n limits-demo wait pod/cpu-burn --for=condition=Ready --timeout=90s \
    || { echo "C7 ②: cpu-burn not Ready within 90 s"; return 1; }
  echo "C7 ②: cpu-burn Ready; reading kubectl top after 30 s"
  _vr_sleep 30
  if vr_retry 60 10 _vr_c7_top; then
    if milli="$(_vr_millicores "$VR_TOP_CPU")" && ((milli <= 110)); then
      echo "C7 ②: kubectl top reads $VR_TOP_CPU (at most 110m)"
    else
      echo "C7 ②: kubectl top reads '$VR_TOP_CPU'; want at most 110m"
      fail=1
    fi
  else
    echo "C7 ②: no kubectl top reading for cpu-burn"
    fail=1
  fi
  line="$(vr_cap kc -n limits-demo get pod cpu-burn "$_VR_RT" \
    -o 'jsonpath={.spec.nodeName}{" "}{.status.containerStatuses[0].containerID}')" || return 1
  read -r node cid <<< "$line"
  cid="$(_vr_container_id "${cid:-}")" || return 1
  [[ "$node" =~ $K8S_NAME_RE ]] || { echo "C7: unexpected node name: '$node'"; return 1; }
  _vr_ssh_prepare "$node"
  dir="$(_vr_cgroup_dir "$node" "$cid")" || return 1
  t1="$(_vr_throttled "$node" "$dir")" || return 1
  _vr_sleep 10
  t2="$(_vr_throttled "$node" "$dir")" || return 1
  if ((t2 > t1)); then
    echo "C7 ②: nr_throttled $t1 -> $t2 in 10 s: throttled at its CPU limit"
  else
    echo "C7 ②: nr_throttled $t1 -> $t2 in 10 s: not increasing"
    fail=1
  fi
  return "$fail"
}

# C7: k8s/limits-demo: ① oom is OOMKilled with exit code 137 within 90 s; ② 30 s after cpu-burn is
# Ready, kubectl top reads at most 110m and its cgroup's nr_throttled grows over 10 s. Both Pods are
# deleted before (left over from an earlier run) and after, pass or fail.
check_c7_limits_demo() {
  local rc=0
  vr_env
  vr_cmd kc -n limits-demo delete pod oom cpu-burn --ignore-not-found --timeout=90s \
    || echo "C7: could not delete Pods left from an earlier run (going on)"
  _vr_c7 || rc=1
  vr_cmd kc -n limits-demo delete pod oom cpu-burn --ignore-not-found --timeout=90s \
    || { echo "C7: cannot delete the demo Pods"; rc=1; }
  return "$rc"
}

_vr_c8() {
  local rc=0
  vr_cmd kc apply -f "$REPO_ROOT/k8s/verify/np-denied.yaml" "$_VR_RT" || return 1
  vr_cmd kc -n np-denied wait pod/client --for=condition=Ready --timeout=120s || return 1
  vr_cmd kc -n np-denied exec client -- curl -sS -m 5 http://hello.hello.svc.cluster.local || rc=$?
  if ((rc == 28)); then
    echo "C8: from np-denied the request timed out (curl exit 28): the NetworkPolicy drops it"
    return 0
  fi
  echo "C8: curl exit $rc from np-denied; want 28 (a timeout: the NetworkPolicy drops the connection)"
  return 1
}

# C8: the allowed path works (C2, from nettest), then a client in the np-denied namespace times out
# (curl exit 28). The namespace is deleted before (left over from an earlier run) and after.
check_c8_netpol() {
  local rc=0
  echo "C8: first the allowed path from nettest (C2)"
  if ! check_c2_cross_node_curl; then
    echo "C8: C2 failed, so a timeout from np-denied would prove nothing"
    return 1
  fi
  vr_cmd kc delete namespace np-denied --ignore-not-found --timeout=120s \
    || { echo "C8: cannot delete np-denied left from an earlier run"; return 1; }
  _vr_c8 || rc=1
  vr_cmd kc delete namespace np-denied --ignore-not-found --timeout=120s \
    || { echo "C8: cannot delete the np-denied namespace"; rc=1; }
  return "$rc"
}

# --- D: GPU (spec §9 D) -------------------------------------------------------------------------

# Every D check SKIPs (77) when no node carries the admin-only accelerator label join-node.sh sets on a
# GPU node, so make verify passes without a GPU node (Review Focus #5). D1 and D2 check every GPU
# node; D6 holds every allocatable GPU. The Pods D4-D6 create are deleted before (left over from an
# earlier run) and after, pass or fail, and also when the check is cut short (_vr_guarded): a
# leftover gpu-hold keeps a billed GPU allocated.

_VR_GPU_LABEL='node-restriction.kubernetes.io/accelerator=nvidia'
# join-node.sh's check that k3s's containerd has the nvidia runtime (spec §6.2 step 2).
_VR_CONTAINERD_NVIDIA='sudo grep -A3 nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml'
# A gpu-type label value as gpuctl's gpu_type_label() makes it (a6000, a10, a100_80gb_sxm4).
_VR_GPU_TYPE_RE='^[a-z0-9]+([_-][a-z0-9]+)*$'

# _vr_guarded CLEANUP FUNC: run FUNC in a subshell whose EXIT trap runs CLEANUP however FUNC ends: it
# returns or exits (die, set -e). The trap lives in the subshell, so the caller's own EXIT trap
# (run_check's _vr_reap, the demo's) is kept. Returns FUNC's status; 1 when FUNC passed but CLEANUP
# failed. When SIGINT or SIGTERM stops FUNC (Ctrl-C during make verify or the demo), _vr_guard_signal
# runs CLEANUP and ends the subshell by that signal, so verify.sh and the demo stop as they would
# without the trap.
_vr_guarded() {
  (
    # shellcheck disable=SC2064,SC2154 # CLEANUP ($1) is expanded now, on purpose; _vr_g and $? when it runs
    trap "_vr_g=\$?; trap - EXIT; if ! $1; then echo 'cleanup failed'; ((_vr_g)) || _vr_g=1; fi; exit \$_vr_g" EXIT
    # shellcheck disable=SC2064 # CLEANUP ($1) is expanded now, on purpose
    trap "_vr_guard_signal INT $1" INT
    # shellcheck disable=SC2064
    trap "_vr_guard_signal TERM $1" TERM
    "$2"
  )
}

# _vr_guard_signal SIGNAL CLEANUP: the INT/TERM trap of _vr_guarded. Runs CLEANUP, then kills its own
# subshell with SIGNAL. If the evidence can no longer be written (run_check's filter keeps it open,
# but a caller's pipe may have died with the Ctrl-C), output goes to /dev/null: a closed pipe must
# not stop CLEANUP (vr_cmd runs nothing it cannot record) and leave a billed GPU allocated.
_vr_guard_signal() {
  set +e
  trap - EXIT INT TERM
  trap '' PIPE
  _vr_reap
  if ! printf 'interrupted (SIG%s); cleaning up\n' "$1" 2>/dev/null; then
    exec >/dev/null 2>&1
  fi
  "$2" </dev/null 2>&1 || echo 'cleanup failed'
  trap - PIPE
  kill -s "$1" "$BASHPID"
  exit 1
}

# _vr_gpu_nodes: VR_GPU_NODES = the nodes labelled $_VR_GPU_LABEL. Returns 77 (SKIP) when there are
# none, 1 when the list cannot be read or holds a name that is not a GPU node's (k3s-gpu-<pool>-<n>).
_vr_gpu_nodes() {
  local out n
  VR_GPU_NODES=()
  out="$(vr_cap kc get nodes -l "$_VR_GPU_LABEL" "$_VR_RT" -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}')" \
    || { echo "D: cannot list the GPU nodes"; return 1; }
  while read -r n; do
    [[ -n "$n" ]] || continue
    [[ "$n" =~ $GPU_NODE_RE ]] || { echo "D: unexpected GPU node name from the cluster: '$n'"; return 1; }
    VR_GPU_NODES+=("$n")
  done <<< "$out"
  if ((${#VR_GPU_NODES[@]} == 0)); then
    echo "D: no node has the label $_VR_GPU_LABEL (no GPU node): skipped"
    return 77
  fi
  echo "D: GPU nodes: ${VR_GPU_NODES[*]}"
}

# D1: on every GPU node the bootstrap's audit (/var/log/gpu-audit.txt, spec §6.2 step 1) has content
# and k3s's containerd config has the nvidia runtime; the RuntimeClass nvidia-gpu exists.
check_d1_gpu_stack() {
  local n out fail=0
  _vr_gpu_nodes || return
  vr_env
  for n in "${VR_GPU_NODES[@]}"; do
    if out="$(vr_cap node_ssh "$n" 'sudo cat /var/log/gpu-audit.txt' </dev/null)" && [[ -n "${out//[[:space:]]/}" ]]; then
      echo "D1: $n: /var/log/gpu-audit.txt has content"
    else
      echo "D1: $n: /var/log/gpu-audit.txt is missing, unreadable or empty"
      fail=1
    fi
    if out="$(vr_cap node_ssh "$n" "$_VR_CONTAINERD_NVIDIA" </dev/null)" && [[ "$out" == *nvidia* ]]; then
      echo "D1: $n: containerd has the nvidia runtime"
    else
      echo "D1: $n: no nvidia runtime in k3s's containerd config"
      fail=1
    fi
  done
  vr_cmd kc get runtimeclass nvidia-gpu "$_VR_RT" || { echo "D1: no RuntimeClass nvidia-gpu"; fail=1; }
  return "$fail"
}

# _vr_d2_node NODE: kubectl describe node NODE shows nvidia.com/gpu >= 1 under both Capacity and
# Allocatable. VR_D2_OUT keeps the last answer, VR_D2_CAP/VR_D2_ALLOC the two values.
_vr_d2_node() {
  local vals
  VR_D2_OUT="" VR_D2_CAP="" VR_D2_ALLOC=""
  VR_D2_OUT="$(vr_cap --no-echo kc describe node "$1" "$_VR_RT")" || return 1
  vals="$(awk '/^Capacity:/ { s = "cap"; next } /^Allocatable:/ { s = "alloc"; next } /^[^ ]/ { s = "" }
               s && $1 == "nvidia.com/gpu:" { v[s] = $2 } END { print (v["cap"] == "" ? "-" : v["cap"]), (v["alloc"] == "" ? "-" : v["alloc"]) }' \
    <<< "$VR_D2_OUT")"
  read -r VR_D2_CAP VR_D2_ALLOC <<< "$vals"
  [[ "$VR_D2_CAP" =~ ^[0-9]+$ && "$VR_D2_ALLOC" =~ ^[0-9]+$ ]] && ((VR_D2_CAP >= 1 && VR_D2_ALLOC >= 1))
}

# D2: every GPU node reports nvidia.com/gpu >= 1 in Capacity and Allocatable (kubectl describe node).
# The resource appears about 20 s after the device plugin registers (live finding L21-2), so each
# node gets up to 90 s.
check_d2_gpu_capacity() {
  local n fail=0
  _vr_gpu_nodes || return
  for n in "${VR_GPU_NODES[@]}"; do
    if vr_retry 90 10 _vr_d2_node "$n"; then
      echo "D2: $n: nvidia.com/gpu capacity $VR_D2_CAP, allocatable $VR_D2_ALLOC"
    else
      echo "D2: $n: nvidia.com/gpu capacity '$VR_D2_CAP', allocatable '$VR_D2_ALLOC' after 90 s; want both >= 1"
      fail=1
    fi
    [[ -z "$VR_D2_OUT" ]] || printf -- '--- kubectl describe node %s ---\n%s\n' "$n" "$VR_D2_OUT"
  done
  return "$fail"
}

# D3: the device plugin Pods and their logs; the logs say "Registered device plugin".
check_d3_device_plugin() {
  local out
  _vr_gpu_nodes || return
  vr_cmd kc -n nvidia-device-plugin get pods -o wide "$_VR_RT" || true
  out="$(vr_cap kc -n nvidia-device-plugin logs -l app.kubernetes.io/name=nvidia-device-plugin --tail=100 "$_VR_RT")" \
    || { echo "D3: cannot read the device plugin logs"; return 1; }
  if [[ "$out" == *"Registered device plugin"* ]]; then
    echo "D3: the device plugin registered with the kubelet"
    return 0
  fi
  echo "D3: no 'Registered device plugin' in the last 100 log lines"
  return 1
}

# _vr_gpu_pod_delete POD: delete Pod POD in gpu-test (if it exists), bounded: --timeout the wait for
# the deletion, $_VR_RT each request (a stalled tunnel).
_vr_gpu_pod_delete() {
  vr_cmd kc -n gpu-test delete pod "$1" --ignore-not-found --timeout=120s "$_VR_RT"
}

# _vr_gpu_pod_wait POD: poll POD's phase every 5 s until it is Succeeded (0), for at most 300 s (1). A
# Pod that Failed never succeeds, so that ends the wait at once (1) instead of after 300 billed seconds.
_vr_gpu_pod_wait() {
  local phase="" waited=0
  while :; do
    phase="$(vr_cap kc -n gpu-test get pod "$1" "$_VR_RT" -o 'jsonpath={.status.phase}')" || phase=""
    case "$phase" in
      Succeeded) return 0 ;;
      Failed)
        echo "$1 Failed"
        return 1
        ;;
    esac
    if ((waited + 5 > 300)); then
      echo "$1 did not succeed within 300 s (phase '$phase')"
      return 1
    fi
    _vr_sleep 5
    waited=$((waited + 5))
  done
}

# _vr_gpu_pod_run POD: apply k8s/gpu-test/POD.yaml, wait (at most 300 s) until it Succeeded, and set
# VR_POD_LOG to its log. When it does not succeed, its description goes into the evidence.
_vr_gpu_pod_run() {
  local pod="$1"
  VR_POD_LOG=""
  vr_cmd kc apply -f "$REPO_ROOT/k8s/gpu-test/$pod.yaml" "$_VR_RT" || return 1
  if ! _vr_gpu_pod_wait "$pod"; then
    vr_cmd kc -n gpu-test describe pod "$pod" "$_VR_RT" || true
    return 1
  fi
  VR_POD_LOG="$(vr_cap kc -n gpu-test logs "$pod" "$_VR_RT")" || { echo "cannot read the log of $pod"; return 1; }
}

_vr_d4_cleanup() { _vr_gpu_pod_delete gpu-smoke; }

_vr_d4() {
  local node type want
  _vr_d4_cleanup || { echo "D4: cannot delete gpu-smoke left from an earlier run"; return 1; }
  _vr_gpu_pod_run gpu-smoke || return 1
  [[ "$VR_POD_LOG" == *NVIDIA-SMI* ]] || { echo "D4: no NVIDIA-SMI table in the log of gpu-smoke"; return 1; }
  node="$(vr_cap kc -n gpu-test get pod gpu-smoke "$_VR_RT" -o 'jsonpath={.spec.nodeName}')" || return 1
  [[ "$node" =~ $GPU_NODE_RE ]] || { echo "D4: gpu-smoke ran on '$node', not on a GPU node"; return 1; }
  type="$(vr_cap kc get node "$node" "$_VR_RT" -o 'jsonpath={.metadata.labels.gpu-type}')" || return 1
  [[ "$type" =~ $_VR_GPU_TYPE_RE ]] || { echo "D4: $node has no valid gpu-type label: '$type'"; return 1; }
  # The label is the instance type's model, gpu_1x_a10 -> a10; nvidia-smi prints "NVIDIA A10". A longer
  # label (a100_sxm4) is matched by its model part (A100), as nvidia-smi spells the rest differently
  # (A100-SXM4-40GB). A whole word: an A100 must not pass for an A10.
  want="$(printf '%s' "${type%%_*}" | tr '[:lower:]' '[:upper:]')"
  if grep -Eq "(^|[^A-Za-z0-9])${want}([^A-Za-z0-9]|\$)" <<< "$VR_POD_LOG"; then
    echo "D4: gpu-smoke ran nvidia-smi on $node (gpu-type $type): its output names '$want'"
    return 0
  fi
  echo "D4: $node is labelled gpu-type $type, but nvidia-smi's output does not name '$want'"
  return 1
}

# D4: gpu-smoke (nvidia-smi as a non-root user) Succeeded within 300 s; its log has the NVIDIA-SMI
# table and names the GPU of the node it ran on: that node's gpu-type label, uppercased (A10, A6000).
check_d4_gpu_smoke() {
  _vr_gpu_nodes || return
  _vr_guarded _vr_d4_cleanup _vr_d4
}

_vr_d5_cleanup() { _vr_gpu_pod_delete vectoradd; }

_vr_d5() {
  _vr_d5_cleanup || { echo "D5: cannot delete vectoradd left from an earlier run"; return 1; }
  _vr_gpu_pod_run vectoradd || return 1
  if [[ "$VR_POD_LOG" == *"Test PASSED"* ]]; then
    echo "D5: vectoradd: Test PASSED"
    return 0
  fi
  echo "D5: no 'Test PASSED' in the log of vectoradd"
  return 1
}

# D5: the CUDA vectorAdd sample Succeeded within 300 s and logged "Test PASSED".
check_d5_vectoradd() {
  _vr_gpu_nodes || return
  _vr_guarded _vr_d5_cleanup _vr_d5
}

# Holders are deleted by their label: they have generated names (gpu-hold-xxxxx).
_vr_d6_cleanup() {
  local rc=0
  vr_cmd kc -n gpu-test delete pod -l app=gpu-hold --ignore-not-found --timeout=120s "$_VR_RT" || rc=1
  _vr_gpu_pod_delete gpu-overflow || rc=1
  return "$rc"
}

_vr_d6() {
  local out n alloc total=0 i running phase uid events
  out="$(vr_cap kc get nodes -l "$_VR_GPU_LABEL" "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{" "}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}')" || return 1
  while read -r n alloc; do
    [[ -n "$n" ]] || continue
    [[ "${alloc:-0}" =~ ^[0-9]+$ ]] || { echo "D6: $n: allocatable nvidia.com/gpu is not a number: '$alloc'"; return 1; }
    total=$((total + ${alloc:-0}))
  done <<< "$out"
  ((total >= 1)) || { echo "D6: no allocatable GPU to hold"; return 1; }
  echo "D6: $total allocatable GPU(s): one gpu-hold Pod each"
  _vr_d6_cleanup || { echo "D6: cannot delete the gpu-hold and gpu-overflow Pods left from an earlier run"; return 1; }
  for ((i = 0; i < total; i++)); do
    vr_cmd kc -n gpu-test create -f "$REPO_ROOT/k8s/gpu-test/gpu-hold.yaml" "$_VR_RT" || return 1
  done
  vr_cmd kc -n gpu-test wait pod -l app=gpu-hold --for=condition=Ready --timeout=300s \
    || { echo "D6: the gpu-hold Pods are not all Running within 300 s"; return 1; }
  out="$(vr_cap kc -n gpu-test get pods -l app=gpu-hold --field-selector=status.phase=Running "$_VR_RT" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}')" || return 1
  running="$(grep -c . <<< "$out" || true)"
  ((running == total)) || { echo "D6: $running gpu-hold Pod(s) Running; want $total"; return 1; }
  vr_cmd kc apply -f "$REPO_ROOT/k8s/gpu-test/gpu-overflow.yaml" "$_VR_RT" || return 1
  # Events outlive their Pod (about an hour): only this gpu-overflow's, by its uid, count.
  uid="$(vr_cap kc -n gpu-test get pod gpu-overflow "$_VR_RT" -o 'jsonpath={.metadata.uid}')" || return 1
  [[ "$uid" =~ ^[0-9a-f-]+$ ]] || { echo "D6: gpu-overflow has no valid uid: '$uid'"; return 1; }
  echo "D6: every GPU is held; looking at gpu-overflow after 30 s"
  _vr_sleep 30
  phase="$(vr_cap kc -n gpu-test get pod gpu-overflow "$_VR_RT" -o 'jsonpath={.status.phase}')" || return 1
  events="$(vr_cap kc -n gpu-test get events \
    "--field-selector=involvedObject.kind=Pod,involvedObject.name=gpu-overflow,involvedObject.uid=$uid" "$_VR_RT")" \
    || return 1
  if [[ "$phase" == Pending && "$events" == *"Insufficient nvidia.com/gpu"* ]]; then
    echo "D6: gpu-overflow is Pending with 'Insufficient nvidia.com/gpu'"
    return 0
  fi
  echo "D6: gpu-overflow is '$phase'; want Pending with an 'Insufficient nvidia.com/gpu' event"
  return 1
}

# D6: one gpu-hold Pod per allocatable GPU (on every GPU node), all Running within 300 s; then
# gpu-overflow is still Pending 30 s after it was applied, with an "Insufficient nvidia.com/gpu"
# event of that Pod (its uid, not an earlier run's). The holders (by label) and gpu-overflow are
# deleted before and after.
check_d6_overflow() {
  _vr_gpu_nodes || return
  _vr_guarded _vr_d6_cleanup _vr_d6
}

# --- E: resilience (spec §9 E; make verify RESILIENCE=1) -----------------------------------------

# E1 reboots a CPU worker, E2 replaces every GPU node (a billed launch), E3 puts a deliberate extra rule
# into the gpu-access firewall stack. E2 and E3 SKIP (77) without a GPU node, as D does; E2 also when
# the GPU spec asks for none. Every wait has a deadline in real elapsed time as well as in counted
# sleeps (vr_retry; E1's two waits share one clock from the reboot command, _vr_e_until), and gpuctl's
# read-only calls are bounded with timeout. What E2 and E3 change is put right by a cleanup (_vr_guarded) that
# runs however the check ends: E2 makes sure no GPU instance is left running when the replacement did
# not finish (no billed orphan), E3 removes the extra rule. A second Ctrl-C does not stop either: the
# cleanup ignores SIGINT/SIGTERM and runs its commands in a process group of their own (_vr_shielded).
# Command output that can carry the project id or the operator's IP (gpuctl, terraform, gcloud) goes
# into the evidence masked (_vr_e_mask); gpuctl masks the operator IP it detects itself.

_VR_E1_NODE=k3s-cpu-1
# When the node's Ready condition last changed (kubectl jsonpath).
_VR_E_READY_SINCE='{.status.conditions[?(@.type=="Ready")].lastTransitionTime}'
# TEST-NET-2 (RFC 5737): reserved for documentation, never routed, so the extra rule E3 adds can never
# admit a real host, even if it stayed.
_VR_E3_IP=198.51.100.99
# The bound of one read-only gpuctl call (status, list-nodes): Lambda, Terraform outputs, kubectl and an
# SSH to k3s-cp, each bounded inside gpuctl too. gpuctl down and apply bound their own steps (launch,
# join, drain) and are not cut short from outside: a killed apply could not clean up what it launched.
_VR_E_READ_S=600
# The autoscaler's override file (stretch): gpuctl down removes it (spec section 10.4).
_VR_E_OVERRIDES="$REPO_ROOT/lambda/.autoscale-state.json"

# _vr_e_mask: stdin -> stdout with the GCP project id and the operator's IP (OPERATOR_IP in .env) as
# <GCP_PROJECT> and <operator_ip>: Terraform prints resource ids with the project, and a run promoted
# to verification/final is committed. Each value is checked first, so the only sed metacharacters in
# them are an IP's dots; the IP must stand alone (198.51.100.2 is not masked inside 198.51.100.20), and
# the loop (:m … tm) also catches two occurrences one character apart.
_vr_e_mask() {
  local -a exprs=(-e '')
  if _gcp_project_ok; then
    exprs+=(-e "s/${GCP_PROJECT}/<GCP_PROJECT>/g")
  fi
  if is_ipv4 "${OPERATOR_IP:-}"; then
    exprs+=(-e ':m' -e "s/(^|[^0-9.])${OPERATOR_IP//./\\.}(\$|[^0-9])/\\1<operator_ip>\\2/" -e 'tm')
  fi
  LC_ALL=C "${_VR_SED[@]}" -E "${exprs[@]}"
}

# _vr_e_stream CMD…: vr_cmd CMD with its stdout and stderr masked (_vr_e_mask) into the evidence as
# they come (gpuctl apply and terraform run for minutes). The filter ignores SIGINT/SIGTERM, as
# run_check's does, so a Ctrl-C ends CMD but not the record of it. Returns CMD's status; 1 when CMD
# succeeded but its output could not be masked (the evidence is incomplete).
_vr_e_stream() {
  local -a st=()
  { vr_cmd "$@" </dev/null 2>&1 | (trap '' INT TERM && _vr_e_mask); st=("${PIPESTATUS[@]}"); } || :
  if ((st[1] != 0)); then
    echo "the output filter failed (exit ${st[1]}): the evidence of this command is incomplete"
    ((st[0] != 0)) || return 1
  fi
  return "${st[0]}"
}

# _vr_e_cap CMD…: _vr_e_stream for a short command whose output the check reads: the command and
# its output (stdout and stderr, masked) go into the evidence, and VR_E_OUT holds that output.
# Returns CMD's status; 1 when the output could not be masked (VR_E_OUT is then empty).
_vr_e_cap() {
  local rc=0
  _vr_quote "$@" | redact | _vr_e_mask || return
  VR_E_OUT="$("$@" </dev/null 2>&1)" || rc=$?
  if ! VR_E_OUT="$(_vr_e_mask <<< "$VR_E_OUT")"; then
    VR_E_OUT=""
    echo "cannot mask the output of $1: left out of the evidence"
    return 1
  fi
  [[ -z "$VR_E_OUT" ]] || printf '%s\n' "$VR_E_OUT"
  return "$rc"
}

# _vr_shielded CMD…: run CMD (a shell function too) in a process group of its own and wait for it. A
# terminal sends Ctrl-C to its foreground process group only, which CMD has left, so a second Ctrl-C
# during a cleanup cannot stop CMD half way (terraform and kubectl take SIGINT back even when it is
# ignored). The caller ignores SIGINT and SIGTERM itself meanwhile (trap '' INT TERM). CMD runs in a
# subshell: variables it sets are lost, so it reports through its output or a file.
_vr_shielded() {
  local pid rc=0
  set -m
  "$@" &
  pid=$!
  set +m
  wait "$pid" || rc=$?
  return "$rc"
}

# _vr_e_gpuctl ARGS…: a read-only scripts/gpuctl ARGS (status, list-nodes), bounded, through _vr_e_cap.
_vr_e_gpuctl() {
  _vr_e_cap timeout --foreground "$_VR_E_READ_S" "$REPO_ROOT/scripts/gpuctl" "$@"
}

# _vr_e_no_drift ID: gpuctl status exits 0 with every drift source read and every GPU pool at its spec
# count. E2 and E3 change nothing otherwise: drift found later must be theirs, and a pool below its
# count would make their gpuctl apply launch more (billed) nodes than the check took away.
_vr_e_no_drift() {
  local rc=0
  _vr_e_gpuctl status || rc=$?
  if ((rc != 0)); then
    echo "$1: gpuctl status exits $rc before the test (want 0, no drift): changing nothing; make gpu-apply reconciles"
    return 1
  fi
  if grep -q '^ *skipped: ' <<< "$VR_E_OUT"; then
    echo "$1: gpuctl status could not read every drift source: changing nothing"
    return 1
  fi
  if grep -q '^ *note: ' <<< "$VR_E_OUT"; then
    echo "$1: a GPU pool is below its spec count, so gpuctl apply would launch more nodes: changing nothing"
    return 1
  fi
  echo "$1: gpuctl status: no drift"
}

# _vr_e_list_nodes: VR_E_NODES = "NAME IP" for every GPU instance gpuctl list-nodes reports, live in any
# status (IP "-" while it boots). Fails when the listing fails or holds a line that is not a node's.
_vr_e_list_nodes() {
  local name ip rest
  VR_E_NODES=()
  _vr_e_gpuctl list-nodes || { echo "E: gpuctl list-nodes failed"; return 1; }
  while read -r name ip rest; do
    [[ -n "$name" ]] || continue
    if [[ ! "$name" =~ $GPU_NODE_RE ]] || { [[ "$ip" != - ]] && ! is_ipv4 "$ip"; }; then
      echo "E: unexpected line from gpuctl list-nodes: '$name $ip $rest'"
      return 1
    fi
    VR_E_NODES+=("$name $ip")
  done <<< "$VR_E_OUT"
}

# _vr_e_listed NAME: the IP gpuctl list-nodes gave NAME (VR_E_NODES); fails when it is not listed.
_vr_e_listed() {
  local line
  for line in ${VR_E_NODES[@]+"${VR_E_NODES[@]}"}; do
    if [[ "${line%% *}" == "$1" ]]; then
      printf '%s\n' "${line#* }"
      return 0
    fi
  done
  return 1
}

# _vr_e_until DEADLINE INTERVAL_S CMD…: vr_retry CMD every INTERVAL_S seconds until DEADLINE, a point in
# this shell's $SECONDS: a wall-clock bound shared by several waits (E1's 120 s and 600 s both count from
# the reboot command). VR_E_DEADLINE holds DEADLINE meanwhile, so a request made by CMD is cut to the
# time left (_vr_e_rt). Fails without trying when DEADLINE has passed.
_vr_e_until() {
  local deadline="$1" interval="$2" rc=0
  shift 2
  if ((SECONDS >= deadline)); then
    printf 'gave up: no time left for %s\n' "$*" >&2
    return 1
  fi
  VR_E_DEADLINE="$deadline"
  vr_retry "$((deadline - SECONDS))" "$interval" "$@" || rc=$?
  VR_E_DEADLINE=""
  return "$rc"
}

# _vr_e_rt: a kubectl --request-timeout of 15 s, or the time left before VR_E_DEADLINE when that is
# shorter (at least 1 s), so one stalled request cannot carry a wait past its deadline.
_vr_e_rt() {
  local s=15
  if [[ -n "${VR_E_DEADLINE:-}" ]] && ((VR_E_DEADLINE - SECONDS < s)); then
    s=$((VR_E_DEADLINE - SECONDS))
    ((s >= 1)) || s=1
  fi
  printf -- '--request-timeout=%ss\n' "$s"
}

# _vr_e_ready_state NODE: VR_E_READY = NODE's Ready condition (True, False, Unknown; "" when it cannot
# be read).
_vr_e_ready_state() {
  VR_E_READY="$(vr_cap kc get node "$1" "$(_vr_e_rt)" \
    -o 'jsonpath={.status.conditions[?(@.type=="Ready")].status}')" || VR_E_READY=""
}

_vr_e_ready() {
  _vr_e_ready_state "$1"
  [[ "$VR_E_READY" == True ]]
}

# NotReady as kubectl shows it: Ready is False, or Unknown once the node controller stops hearing
# from the kubelet.
_vr_e_not_ready() {
  _vr_e_ready_state "$1"
  [[ -n "$VR_E_READY" && "$VR_E_READY" != True ]]
}

# _vr_e_node_field NODE JSONPATH: one field of node NODE (its uid, its boot ID).
_vr_e_node_field() {
  vr_cap kc get node "$1" "$_VR_RT" -o "jsonpath=$2"
}

# _vr_e_nettest_ready: the nettest DaemonSets (probe, echo) are rolled out on every node, so B1 finds a
# Running probe and echo Pod on a node that was just rebooted or replaced.
_vr_e_nettest_ready() {
  local ds fail=0
  for ds in probe echo; do
    vr_cmd kc -n nettest rollout status "ds/$ds" --timeout=300s \
      || { echo "E: the nettest $ds DaemonSet is not rolled out within 300 s"; fail=1; }
  done
  return "$fail"
}

# E1: reboot k3s-cpu-1 (`sudo systemctl reboot`; ssh's exit 255 when the connection drops is expected).
# PASS: it is NotReady within 120 s of the reboot command and Ready again within 600 s of it, with a new
# boot ID (it did reboot), and then B1 passes (the sysctls, WireGuard and the nettest Pods came back:
# live finding L18-1, IP forwarding must survive a reboot). NotReady is polled every 2 s; a reboot can
# be quicker than that window (the kubelet reports NotReady for a moment at start), so a Ready
# condition whose lastTransitionTime moved since before the reboot counts as seen: Ready changes only
# through NotReady, and the evidence shows both times.
check_e1_worker_reboot() {
  local node="$_VR_E1_NODE" boot1 boot2 since1 since2 t0 rc=0
  vr_env
  _vr_e_ready_state "$node"
  if [[ "$VR_E_READY" != True ]]; then
    echo "E1: $node is not Ready before the reboot (Ready='$VR_E_READY'): not rebooting it"
    return 1
  fi
  boot1="$(_vr_e_node_field "$node" '{.status.nodeInfo.bootID}')" || return 1
  [[ -n "$boot1" ]] || { echo "E1: $node reports no boot ID"; return 1; }
  since1="$(_vr_e_node_field "$node" "$_VR_E_READY_SINCE")" || return 1
  [[ -n "$since1" ]] || { echo "E1: $node reports no lastTransitionTime for Ready"; return 1; }
  t0="$SECONDS"
  _vr_e_stream worker_ssh "$node" 'sudo systemctl reboot' || rc=$?
  case "$rc" in
    0 | 255) echo "E1: reboot sent to $node (ssh exit $rc; 255 is the connection closing)" ;;
    *)
      echo "E1: the reboot command failed (exit $rc)"
      return 1
      ;;
  esac
  if _vr_e_until "$((t0 + 120))" 2 _vr_e_not_ready "$node"; then
    echo "E1: $node is NotReady (Ready=$VR_E_READY)"
  elif since2="$(_vr_e_node_field "$node" "$_VR_E_READY_SINCE")" && [[ -n "$since2" && "$since2" != "$since1" ]]; then
    echo "E1: $node went NotReady and back between two polls (Ready lastTransitionTime $since1 -> $since2)"
  else
    echo "E1: $node was never NotReady within 120 s of the reboot command"
    if boot2="$(_vr_e_node_field "$node" '{.status.nodeInfo.bootID}')" && [[ -n "$boot2" && "$boot2" != "$boot1" ]]; then
      echo "E1: its boot ID changed ($boot1 -> $boot2): it rebooted faster than the node controller marks a node NotReady"
    else
      echo "E1: its boot ID is unchanged: $node did not reboot (an ssh exit 255 can also be a connection that failed)"
    fi
    return 1
  fi
  if _vr_e_until "$((t0 + 600))" 10 _vr_e_ready "$node"; then
    echo "E1: $node is Ready again ($((SECONDS - t0)) s after the reboot command)"
  else
    echo "E1: $node is not Ready again within 600 s of the reboot command (Ready='$VR_E_READY')"
    return 1
  fi
  boot2="$(_vr_e_node_field "$node" '{.status.nodeInfo.bootID}')" || return 1
  if [[ -z "$boot2" || "$boot2" == "$boot1" ]]; then
    echo "E1: the boot ID did not change ('$boot1' -> '$boot2'): $node did not reboot"
    return 1
  fi
  echo "E1: boot ID $boot1 -> $boot2"
  _vr_e_nettest_ready || return 1
  echo "E1: B1 after the reboot"
  check_b1_matrix
}

# E2's cleanup (_vr_guarded): when the replacement did not finish (VR_E2_CHANGED: gpuctl down or apply
# failed, the check died or was interrupted between them), no GPU instance may be left running: an
# instance gpuctl did not clean up, or one apply launched but did not join, is billed for nothing.
# Lists them; if any (or the listing fails), runs gpuctl down and lists again.
_vr_e2_cleanup() {
  local rc=0
  [[ "${VR_E2_CHANGED:-0}" == 1 ]] || return 0
  trap '' INT TERM
  _vr_e2_sweep || rc=$?
  trap - INT TERM
  return "$rc"
}

# _vr_e2_listing FILE: gpuctl list-nodes into the evidence and its (masked) output into FILE.
_vr_e2_listing() {
  local rc=0
  _vr_e_gpuctl list-nodes || rc=$?
  printf '%s\n' "${VR_E_OUT:-}" >"$1"
  return "$rc"
}

# _vr_e2_none FILE: gpuctl list-nodes succeeds and lists nothing (FILE "": no way to tell, so no).
_vr_e2_none() {
  [[ -n "$1" ]] && _vr_shielded _vr_e2_listing "$1" && [[ -z "$(tr -d '[:space:]' <"$1")" ]]
}

_vr_e2_sweep() {
  local f rc=0
  echo "E2: the replacement did not finish: making sure no GPU instance is left running (Ctrl-C waits for this)"
  f="$(mktemp)" || f=""
  if _vr_e2_none "$f"; then
    echo "E2: no GPU instance is left running; make gpu-apply brings the GPU node back"
  else
    echo "E2: removing what is left with gpuctl down"
    _vr_shielded _vr_e_stream "$REPO_ROOT/scripts/gpuctl" down --auto-approve || echo "E2: gpuctl down failed"
    if _vr_e2_none "$f"; then
      echo "E2: no GPU instance is left running; make gpu-apply brings the GPU node back"
    else
      echo "E2: GPU instance(s) may still be running and billing: ${f:+$(tr '\n' ' ' <"$f")}"
      echo "E2: run make gpu-down, then make gpu-status (want 0 instances)"
      rc=1
    fi
  fi
  [[ -z "$f" ]] || rm -f -- "$f"
  return "$rc"
}

_vr_e2() {
  local line name ip new_ip uid new_uid extra="" fail=0
  local -a before=()
  VR_E2_CHANGED=0
  if [[ -e "$_VR_E_OVERRIDES" ]]; then
    echo "E2: lambda/.autoscale-state.json (the autoscaler's overrides) exists: gpuctl down removes it, so the"
    echo "E2: gpuctl apply after it would launch the spec's counts instead of what runs now: changing nothing"
    return 1
  fi
  _vr_e_no_drift E2 || return 1
  _vr_e_list_nodes || return 1
  ((${#VR_E_NODES[@]})) || { echo "E2: gpuctl list-nodes lists no GPU instance"; return 1; }
  for line in "${VR_E_NODES[@]}"; do
    read -r name ip <<< "$line"
    uid="$(_vr_e_node_field "$name" '{.metadata.uid}')" || return 1
    [[ "$uid" =~ ^[0-9a-f-]+$ ]] || { echo "E2: node $name has no valid uid: '$uid'"; return 1; }
    before+=("$name $ip $uid")
  done
  echo "E2: replacing ${#before[@]} GPU node(s): gpuctl down, then gpuctl apply"

  VR_E2_CHANGED=1
  _vr_e_stream "$REPO_ROOT/scripts/gpuctl" down --auto-approve || { echo "E2: gpuctl down failed"; return 1; }
  _vr_e_list_nodes || return 1
  for line in "${before[@]}"; do
    name="${line%% *}"
    if _vr_e_listed "$name" >/dev/null; then
      echo "E2: $name is still listed after gpuctl down"
      return 1
    fi
  done
  _vr_e_stream "$REPO_ROOT/scripts/gpuctl" apply --auto-approve || { echo "E2: gpuctl apply failed"; return 1; }
  # What runs now is what the spec wants: nothing for the cleanup to remove, whatever fails below.
  VR_E2_CHANGED=0

  _vr_e_list_nodes || return 1
  for line in ${VR_E_NODES[@]+"${VR_E_NODES[@]}"}; do
    name="${line%% *}"
    _vr_in "$name" "${before[@]%% *}" || extra+=" $name"
  done
  if [[ -n "$extra" ]]; then
    echo "E2: gpuctl apply runs more GPU instances than the check replaced:$extra (billed)"
    echo "E2: check make gpu-status; make gpu-down removes every GPU node"
    return 1
  fi
  for line in "${before[@]}"; do
    read -r name ip uid <<< "$line"
    if ! new_ip="$(_vr_e_listed "$name")"; then
      echo "E2: $name is not listed after gpuctl apply"
      return 1
    fi
    echo "E2: $name: $ip -> $new_ip"
    if ! _vr_e_until "$((SECONDS + 300))" 10 _vr_e_ready "$name"; then
      echo "E2: $name is not Ready within 300 s after gpuctl apply (Ready='$VR_E_READY')"
      return 1
    fi
    new_uid="$(_vr_e_node_field "$name" '{.metadata.uid}')" || return 1
    if [[ "$new_uid" == "$uid" ]]; then
      echo "E2: $name is the same Node object as before (uid $uid): it was not replaced"
      return 1
    fi
    echo "E2: $name is Ready as a new Node (uid $uid -> $new_uid)"
  done
  _vr_e_nettest_ready || return 1
  echo "E2: D2, D4 and B1 on the new GPU node(s)"
  check_d2_gpu_capacity || fail=1
  check_d4_gpu_smoke || fail=1
  check_b1_matrix || fail=1
  return "$fail"
}

# _vr_e_spec_count: the number of GPU nodes lambda/gpu-nodes.yaml asks for (every pool's count), read with
# gpuctl's own spec loader (the repo's venv, as scripts/gpuctl runs it); 0 when there is no spec file.
# Fails when the spec cannot be read.
_vr_e_spec_count() {
  local spec="$REPO_ROOT/lambda/gpu-nodes.yaml"
  if [[ ! -e "$spec" ]]; then
    echo "E2: no GPU spec ($spec)" >&2
    echo 0
    return 0
  fi
  vr_cap env PYTHONPATH="$REPO_ROOT/lambda" "$REPO_ROOT/.venv/bin/python" -P -c \
    'import sys; from gpuctl.spec import load_spec; print(sum(p.count for p in load_spec(sys.argv[1]).pools.values()))' \
    "$spec" </dev/null
}

# E2: replace the GPU node(s): gpuctl down --auto-approve, then gpuctl apply --auto-approve. PASS: before,
# gpuctl status shows no drift; after down, the node is gone from Lambda; after apply, the same node
# name is back (a new instance, maybe a new IP and host key) as a new Node object that is Ready within
# 300 s, no other GPU instance runs, and D2, D4 and B1 pass. SKIP without a GPU node, and when the spec
# asks for none (no GPU pool, or every count 0: gpuctl apply would not bring the node back). Refused
# (FAIL, nothing changed) while the autoscaler's override file exists. A replacement that does not
# finish leaves no GPU instance running (_vr_e2_cleanup).
check_e2_gpu_replace() {
  local want
  _vr_gpu_nodes || return
  want="$(_vr_e_spec_count)" || { echo "E2: cannot read the GPU spec lambda/gpu-nodes.yaml"; return 1; }
  [[ "$want" =~ ^[0-9]+$ ]] || { echo "E2: unexpected node count from the GPU spec: '$want'"; return 1; }
  if ((want == 0)); then
    echo "E2: the GPU spec asks for no GPU node, so gpuctl apply would not bring one back: skipped"
    return 77
  fi
  vr_env
  _vr_guarded _vr_e2_cleanup _vr_e2
}

# _vr_e3_ips: the gpu-access stack's gpu_ips output (JSON) on stdout, as vr_cap prints it; the command,
# the output and Terraform's messages (they name the state bucket, so the project) go into the
# evidence masked.
_vr_e3_ips() {
  local -a st=()
  { vr_cap tf_out_json gpu-access gpu_ips 2>&1 >&3 | (trap '' INT TERM && _vr_e_mask) >&2; st=("${PIPESTATUS[@]}"); } 3>&1 || :
  return "${st[0]}"
}

# _vr_e3_state FILE: _vr_e3_ips into FILE.
_vr_e3_state() {
  _vr_e3_ips >"$1"
}

# _vr_e3_gone FILE: the gpu-access state can be read (into FILE) and holds no rule for $_VR_E3_IP
# (FILE "": no way to tell, so no).
_vr_e3_gone() {
  [[ -n "$1" ]] && _vr_shielded _vr_e3_state "$1" || return 1
  jq -e --arg ip "$_VR_E3_IP" 'type == "array" and ([.[] | select(. == $ip)] | length == 0)' "$1" >/dev/null 2>&1
}

# _vr_e3_tf_apply GPU_IPS_JSON [VAR…]: terraform apply of the gpu-access stack for exactly these GPU IPs.
_vr_e3_tf_apply() {
  local ips="$1"
  shift
  _vr_e_stream terraform -chdir="$VR_E3_DIR" apply -auto-approve -input=false -lock-timeout=120s "$@" \
    "-var=gpu_ips=$ips"
}

# E3's cleanup (_vr_guarded), once the extra rule may exist (VR_E3_ARMED): if the gpu-access state still
# holds $_VR_E3_IP (or cannot be read), apply the state's own GPU IPs as they were before the check.
_vr_e3_cleanup() {
  local rc=0
  [[ "${VR_E3_ARMED:-0}" == 1 ]] || return 0
  trap '' INT TERM
  _vr_e3_restore || rc=$?
  trap - INT TERM
  return "$rc"
}

_vr_e3_restore() {
  local f rc=0
  echo "E3: cleanup: the rule for $_VR_E3_IP must not stay (Ctrl-C waits for this)"
  f="$(mktemp)" || f=""
  if _vr_e3_gone "$f"; then
    echo "E3: cleanup: the gpu-access state has no rule for $_VR_E3_IP"
  else
    echo "E3: cleanup: restoring gpu_ips=$VR_E3_ORIG"
    _vr_shielded _vr_e3_tf_apply "$VR_E3_ORIG" || echo "E3: cleanup: terraform apply failed"
    if _vr_e3_gone "$f"; then
      echo "E3: cleanup: restored: the gpu-access state has no rule for $_VR_E3_IP"
    else
      echo "E3: cleanup: the gpu-access stack may STILL allow $_VR_E3_IP (TEST-NET-2, never routed):"
      echo "E3: cleanup: run make gpu-apply, then make gpu-status (want no drift)"
      rc=1
    fi
  fi
  [[ -z "$f" ]] || rm -f -- "$f"
  return "$rc"
}

_vr_e3() {
  local ips n ip inject rc fail=0
  VR_E3_ARMED=0
  tf_env
  VR_E3_DIR="$(tf_dir gpu-access)"
  _vr_e_no_drift E3 || return 1
  _vr_e_stream tf_init gpu-access || return 1
  ips="$(_vr_e3_ips)" || { echo "E3: cannot read gpu_ips from the gpu-access stack"; return 1; }
  if ! jq -e 'type == "array" and all(.[]; type == "string")' <<< "$ips" >/dev/null 2>&1; then
    echo "E3: unexpected gpu_ips output from the gpu-access stack"
    return 1
  fi
  while read -r ip; do
    [[ -z "$ip" ]] || is_ipv4 "$ip" || { echo "E3: gpu_ips holds '$ip', not an IPv4 address"; return 1; }
  done <<< "$(jq -r '.[]' <<< "$ips")"
  VR_E3_ORIG="$(jq -c . <<< "$ips")"
  n="$(jq length <<< "$ips")"
  inject="$(jq -c --arg ip "$_VR_E3_IP" '. + [$ip]' <<< "$ips")"

  echo "E3 ①: adding a rule for $_VR_E3_IP next to the $n real GPU IP(s)"
  VR_E3_ARMED=1
  # max_nodes: the stack refuses more IPs than its copy of the spec's max_nodes; the extra IP is one more.
  _vr_e3_tf_apply "$inject" "-var=max_nodes=$((n + 1))" || { echo "E3 ①: terraform apply failed"; return 1; }

  rc=0
  _vr_e_gpuctl status || rc=$?
  if ((rc != 3)); then
    echo "E3 ②: gpuctl status exits $rc; want 3 (drift)"
    fail=1
  elif ! grep -Eq "(^|[^0-9.])${_VR_E3_IP//./\\.}([^0-9]|\$)" <<< "$VR_E_OUT"; then
    echo "E3 ②: gpuctl status exits 3 but does not name $_VR_E3_IP"
    fail=1
  else
    echo "E3 ②: gpuctl status exits 3 and names $_VR_E3_IP"
  fi

  if _vr_e_stream "$REPO_ROOT/scripts/gpuctl" apply --auto-approve; then
    echo "E3 ③: gpuctl apply repaired the drift"
  else
    echo "E3 ③: gpuctl apply failed"
    fail=1
  fi

  rc=0
  _vr_e_gpuctl status || rc=$?
  if ((rc == 0)); then
    echo "E3 ④: gpuctl status exits 0: no drift"
  else
    echo "E3 ④: after the repair gpuctl status exits $rc; want 0"
    fail=1
  fi
  return "$fail"
}

# E3: drift in the GPU firewall is found and repaired. ① terraform apply of the gpu-access stack with
# gpu_ips = the real GPU IPs plus $_VR_E3_IP (one extra rule, via TF_VAR_* and tf_init as the other
# scripts); ② gpuctl status exits 3 and names that IP; ③ gpuctl apply --auto-approve repairs it; ④
# gpuctl status exits 0. Nothing is injected unless gpuctl status shows no drift first, and the extra
# rule is removed however the check ends (_vr_e3_cleanup). SKIP without a GPU node (gpuctl apply would
# launch one).
check_e3_firewall_drift() {
  _vr_gpu_nodes || return
  vr_env
  _vr_guarded _vr_e3_cleanup _vr_e3
}

# --- run summary --------------------------------------------------------------------------------

# vr_finish: append the totals to SUMMARY.md and scan the run directory with gitleaks (when it is
# installed). Returns 1 if any check failed or gitleaks found a secret, else 0 (SKIPs are not failures).
vr_finish() {
  local rc=0 scan
  printf '\n**Result:** %d passed, %d failed, %d skipped\n' \
    "$((VR_SEQ - VR_FAILS - VR_SKIPS))" "$VR_FAILS" "$VR_SKIPS" >>"$VR_DIR/SUMMARY.md"
  if command -v gitleaks >/dev/null 2>&1; then
    if gitleaks dir "$VR_DIR" --config "$REPO_ROOT/.gitleaks.toml" --no-banner --redact --exit-code 1; then
      scan="gitleaks found no leaks"
    else
      scan="gitleaks reported leaks (or failed); do not share this run"
      rc=1
    fi
  else
    scan="not run (gitleaks is not installed)"
    log "warning: gitleaks is not installed; $VR_DIR was not scanned for secrets"
  fi
  printf '\n**Secret scan:** %s\n' "$scan" >>"$VR_DIR/SUMMARY.md"
  if ((VR_FAILS > 0)); then rc=1; fi
  printf 'verification: %d passed, %d failed, %d skipped; %s; summary: %s\n' \
    "$((VR_SEQ - VR_FAILS - VR_SKIPS))" "$VR_FAILS" "$VR_SKIPS" "$scan" "$VR_DIR/SUMMARY.md"
  return "$rc"
}
