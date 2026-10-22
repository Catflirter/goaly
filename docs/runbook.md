# Runbook

Day-to-day operation of the hybrid cluster: bring-up, scaling, GPU sessions, verification,
break-glass access, teardown and the demo day. Every `make` target here is safe to re-run. Every
Terraform apply of the base and workers stacks, `gpu-apply` and `gpu-down` show what they will do and
ask first; `make down` asks once for the project id. `AUTO_APPROVE=1` skips each of those questions,
so never export it in a shell. `cpu-stop`, `cpu-start` and `sync-peers` act at once without asking.
Architecture and rationale: [README](../README.md). Times are Pacific (PDT).

## Prerequisites

**Accounts**

- A GCP project with billing enabled. The operator account needs `roles/compute.osAdminLogin`,
  `roles/iap.tunnelResourceAccessor`, `roles/iam.serviceAccountUser` on the `k3s-cp` and `k3s-worker`
  service accounts, and the rights to create the resources (project Owner or Editor for the first
  `make init`, which enables six APIs).
- A Lambda Cloud account, an API key, and an SSH public key uploaded to Lambda.

**Tools on the operator machine** (Linux or macOS)

| Tool | Version | Used by |
|---|---|---|
| terraform | >= 1.9.0 | every stack |
| gcloud | current | IAP SSH, the API tunnel, `cpu-stop`/`cpu-start` |
| kubectl, helm | current | cluster objects, the device plugin chart |
| python3 | >= 3.11 | `gpuctl` (in `.venv`, from `make venv`) |
| jq, ssh, coreutils `timeout` | any | scripts |
| gitleaks | >= 8.19 | `make lint`, `make verify`, `make promote-verify` |
| bash | >= 4.1 (macOS ships 3.2: `brew install bash`) | `make up`, `make down`, `make tunnel`, `make cpu-scale`, `make cpu-remove`: each takes a lock with `exec {fd}>` and refuses an older bash (exit 2) |
| flock(1) (util-linux; macOS: `brew install bash util-linux`, or `brew install flock`) | any | install it, with bash >= 4.1, before `make up` or `make down`: neither refuses up front without it. `make down` warns and runs its stacks unlocked, then `tunnel.sh` exits 2 at the tunnel stop, before the final Lambda check. `tunnel.sh` needs it for `make tunnel`, `make tunnel-stop` and the tunnel stage of `make up`; `cpu-pool.sh --guarded` (and so `make demo` chapter 6) refuses without it; `cpu-scale` and `cpu-remove` run without the workers lock, with a warning |
| shellcheck | installed by pre-commit (`shellcheck-py`) | `make lint` |
| kubeconform | any, on `PATH` (a `language: system` hook, like `terraform fmt`) | `make lint` |

**One-time setup**

```bash
gcloud auth login && gcloud auth application-default login
cp .env.example .env && chmod 600 .env
# .env: GCP_PROJECT, LAMBDA_API_KEY, LAMBDA_SSH_KEY_NAME (the key's name in Lambda),
#       SSH_PRIVATE_KEY_PATH; OPERATOR_IP is optional (detected through checkip.amazonaws.com);
#       UNDERLAY_MTU defaults to 1460
make venv && make lint
```

`.env` is sourced as shell, so treat it as code: it must stay mode 600 (the scripts refuse
anything else) and must never be committed (it is git-ignored).

## Bring-up

```bash
make init     # once per project: state bucket <GCP_PROJECT>-k3s-tfstate + six APIs (local state)
make up       # base -> workers -> system -> workloads
```

What `make up` does ([scripts/up.sh](../scripts/up.sh)):

| Stage | Steps | Done when |
|---|---|---|
| base | apply `infra/base` (10 resources); copy `node/` to `k3s-cp` over IAP and run `server-bootstrap.sh`; `make kubeconfig`; `make tunnel` | `node/k3s-cp` and CoreDNS Ready |
| workers | apply `infra/workers` (6 resources for one worker); `sync-peers`; `join-node.sh` for every worker not yet Ready; `sync-peers` | every worker Ready |
| system | metrics-server, RuntimeClass `nvidia-gpu`, namespace `nvidia-device-plugin`, Helm chart `nvdp` 0.20.1 | rollouts complete |
| workloads | `k8s/nettest`, `k8s/hello`, namespace `gpu-test` | rollouts complete |

A single stage can be re-run with `scripts/up.sh <stage>`. Under `AUTO_APPROVE=1`, the base and
workers applies refuse any plan that destroys or replaces something; a deliberate replacement of
`k3s-cp` is done interactively, where the plan's destroys are listed and confirmed.

Access after bring-up:

```bash
export KUBECONFIG=~/.kube/hybrid-gpu.yaml   # server https://127.0.0.1:6443 through make tunnel
make tunnel                                 # starts it if it is not running; make tunnel-stop stops it
kubectl get nodes -o wide
```

## Scaling

The CPU pool is the inventory `infra/workers/workers.auto.tfvars.json`
([scripts/cpu-pool.sh](../scripts/cpu-pool.sh)).

| Command | Effect |
|---|---|
| `make cpu-scale N=2` | adds `k3s-cpu-2` (`us-central1-a`, `e2-small` unless `CPU_TYPE` is set): plan, ask, apply, `sync-peers`, join, `sync-peers` |
| `make cpu-scale N=1` | plans first, asks, drains and deletes the workers that go, writes the inventory, asks, applies a plan that may destroy only those workers |
| `make cpu-remove NAME=k3s-cpu-2` | drain, delete the Node, drop it from the inventory, apply, `sync-peers` |
| `make cpu-stop` / `make cpu-start` | stop or start every worker VM; stopped workers are NotReady and keep their static IPs |

`k3s-cpu-1` is never removed. It is an `e2-standard-2`, set in the inventory; workers that
`cpu-scale` adds default to `e2-small` (`CPU_TYPE` overrides). For unattended use,
`scripts/cpu-pool.sh scale N --guarded` also requires the saved plan to pass `gpuctl tfguard` (only
the four per-worker resources of the added or removed workers may change).

The scripts that plan or apply the base and workers stacks hold that stack's lock,
`.run/<stack>.lock` (flock), until they exit: `make up` (each stack from the stage that plans it),
every `cpu-scale` and `cpu-remove` (`workers.lock`) and `make down` (all three stacks). A second run
on the same stack refuses at once (exit 2) and says how to find the holder (`fuser -v` or `lsof` on
the lock file). The kernel drops the lock when the holder exits, however it ends, so there is no
stale lock to clean up. `gpuctl`'s own applies of `gpu-access` (`gpu-apply`, `gpu-down`,
`sync-peers`) take no `.run` lock; Terraform's lock on the GCS state serialises them, and a second
apply fails with a state-lock error instead of waiting.

## GPU nodes

The spec is [lambda/gpu-nodes.yaml](../lambda/gpu-nodes.yaml); `gpuctl` reconciles Lambda to it
([ADR-0004](adr/0004-gpu-nodes-as-cattle-spec-reconciler.md)).

```bash
make gpu-status   # capacity per pool/type/region, this cluster's instances, drift (exit 3)
make gpu-plan     # creates/deletes and the hourly cost at live prices
make gpu-apply    # asks, launches, opens the firewalls, joins, labels
make verify       # D1-D6 now run instead of SKIP
make gpu-down     # asks, drains, deletes the Node, terminates, closes the firewalls
make gpu-status   # want: "GPU instances (k3s-gpu-*): 0" and no drift
```

Read the instance count, not only the exit code: a pool below its target is a note, never drift, so
a leftover instance in a pool whose target is 1 still exits 0.

- **Every GPU session ends with `make gpu-down` and a `make gpu-status` that shows no instance.**
  Lambda bills until an instance is terminated; shutting it down from inside does not stop billing.
- To change the count, edit `count` in the spec; `scripts/gpuctl plan --set-count a6000=2` previews a
  count without editing the file. With the spec as committed, that preview prints the plan followed
  by `plan refused: budget_usd_per_hour ...` (see the budget below).
- Capacity: `gpu-status` shows which (type, region) pairs have stock. On both measured days only
  `gpu_1x_a10` had any; `gpu-apply` falls back to it automatically.
- The budget check prices kept nodes at their own type and every new node at the pool's most
  expensive candidate (the A100 at $1.99/h). With `gpu_1x_a100` listed as a fallback, the 3.00
  budget admits only one GPU node in total: a second node next to a running A10 is refused
  ($1.29 + $1.99 = $3.28), and so are two new nodes ($3.98). To run two, raise
  `budget_usd_per_hour` to at least 3.98, or drop the A100 fallback (then 2 × $1.29 = $2.58 for two
  A10s).
- `scripts/gpuctl apply --keep-on-failure` keeps a node whose join failed, for debugging. It bills
  until `make gpu-down`.
- `make sync-peers` after the operator's network changes: it replaces the operator's old IP with the
  new one in Lambda's SSH rules.

## Verification

```bash
make verify                       # A-D into verification/<YYYY-MM-DDTHHMM-0700>/
make verify RESILIENCE=1          # adds E1-E3: reboots k3s-cpu-1, replaces the GPU node, forges drift
scripts/verify.sh B               # one group
OPERATOR_IP=$(curl -s https://checkip.amazonaws.com) \
  make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>  # scan, then replace verification/final/
grep -r "$(curl -s https://checkip.amazonaws.com)" verification/final   # must print nothing
```

A run exits 0 when no check failed (SKIP is not a failure) and gitleaks found nothing in it. D-group
checks SKIP without a GPU node. E2 replaces the GPU node only when its pool shows capacity or the node
itself holds a unit of a candidate type; otherwise it SKIPs, which means "not verified": replace the
node by hand (`make gpu-down`, `make gpu-apply`) and re-run `scripts/verify.sh E`.
`make promote-verify` refuses a run with a failed check, and a run with a skipped check unless
`PROMOTE_ALLOW_SKIP=1`. It copies the run to a staging directory and fails if gitleaks finds
anything, or if the copy contains the GCP project id (from the environment and from `.env`, in any
case; [scripts/promote.sh](../scripts/promote.sh)) or `OPERATOR_IP`. The run itself already shows the
project id as `<GCP_PROJECT>` (`redact` in [scripts/lib/checks.sh](../scripts/lib/checks.sh)), so this
check is a backstop. The operator-IP check runs only when `OPERATOR_IP` is set in the caller's
environment (promote.sh does not look the IP up itself), hence the command above. The `grep` after
it is a required step before the promoted copy is committed. The final from-scratch run is committed
as [verification/final/SUMMARY.md](../verification/final/SUMMARY.md): 2026-10-22 13:20 PDT, after
Lambda's write API recovered (about 13:10 PDT), 24/24 PASS (A-D), 0 failed, 0 skipped, 6m52s.

## Break-glass

**The API tunnel is down or hangs.** `make tunnel-stop && make tunnel`. If IAP itself works, the API
is also reachable on the control plane:

```bash
gcloud compute ssh k3s-cp --zone us-west1-b --tunnel-through-iap -- sudo k3s kubectl get nodes
```

**A CPU worker needs a shell.** `gcloud compute ssh k3s-cpu-1 --zone us-central1-a --tunnel-through-iap`.

**A GPU node needs a shell.** SSH is open only to the operator's current `/32`:

```bash
make sync-peers                           # if the operator's network changed
ip=$(scripts/gpuctl ip k3s-gpu-a6000-1)
ssh -i ~/.ssh/<your-lambda-key> -o UserKnownHostsFile=.run/known_hosts_k3s-gpu-a6000-1 ubuntu@"$ip"
```

The key is the private key that `SSH_PRIVATE_KEY_PATH` names in `.env`; write its path out. Do not
`source .env` into your shell: it would export the Lambda API key there.

**A GPU node must go and the cluster is unreachable.** `scripts/gpuctl down --no-drain` terminates it
without the drain. Risk: the node's Pods are killed without eviction, and its Node object stays in
the cluster; delete it with `kubectl delete node <name>` once the API is back. When the cluster is
known to be gone (no base outputs, no kubeconfig), `make gpu-down` skips the drain by itself. Last
resort: terminate it in the Lambda console, then run `make gpu-down` so the firewalls and
`k3s_peers` follow (`make gpu-apply` only if you want a replacement: it launches and bills a new
node).

**The host firewall on `k3s-cp` blocks something it should not** (for example a node whose IP is
missing from `k3s_peers`). First try `make sync-peers`. In an emergency, on `k3s-cp`:

```bash
sudo nft delete table inet k3s_guard      # removes the host layer only
sudo systemctl reload k3s-guard           # restores it, with the last persisted peer list
```

Risks while the table is gone: 6443 and 10250 are filtered only by the cloud firewall, and pods on
`k3s-cp` can reach the metadata server. Do not `systemctl stop k3s-guard`: `k3s.service` requires it,
so stopping it stops k3s.

**A Terraform state lock is left behind** (an interrupted apply). The interrupted command prints the
lock ID; `terraform -chdir=infra/<stack> force-unlock <id>` after confirming nothing else runs.

## Teardown

```bash
make gpu-down && make gpu-status          # optional: make down does this too
make down                                 # asks for the GCP project id, then destroys everything
```

`AUTO_APPROVE=1` in the caller's environment skips that one question, and nothing after it asks: the
whole teardown, `k3s-cp` included, then runs unattended. `.env` cannot set it.

`make down` ([scripts/down.sh](../scripts/down.sh)), in order: `gpuctl down`; drain and delete every
CPU worker's Node (skipped with a warning when the cluster is unreachable); `terraform destroy` of
`gpu-access`, `workers`, `base`; stop the tunnel; remove the GPU host keys and
`~/.kube/hybrid-gpu.yaml`; finally ask Lambda whether a GPU node of this cluster is still live, and
fail with "run make gpu-down" if one is. The bootstrap stack (state bucket, APIs) is kept. A teardown
that stopped part-way is finished by running `make down` again.

To confirm nothing is left (with `gcloud config set project <GCP_PROJECT>`):

```bash
gcloud compute instances list --filter='name~^k3s-'
gcloud compute addresses list --filter='name~k3s'
gcloud compute firewall-rules list --filter='name~^fw-'
gcloud compute networks list --filter='name~^k3s-'
gcloud iam service-accounts list --filter='email~^k3s-'
scripts/gpuctl list-nodes
```

All six print nothing.

## Demo-day checklist

The demo is run live from the operator's machine (`scripts/demo.sh`,
`scripts/preflight.sh`); `scripts/demo.sh list` prints the chapters and their
estimated length.

| When | Action |
|---|---|
| Day before | `make up`; `make verify`; check `SUMMARY.md` is all PASS (D included only with a GPU node); GCP stays up overnight (see [Cost](../README.md#cost)) |
| T-60 min | `make preflight`, every line OK: tool versions (flock included); an active `gcloud auth list` account; ADC works (as `gcloud auth application-default print-access-token > /dev/null`: the token is never shown); `scripts/gpuctl status` exits 0 (capacity listed, no drift); `make sync-peers` (the operator IP of this network); the tunnel, started if down; every node Ready; the operator IP and project id known, so the demo can mask them |
| T-30 min | `make gpu-apply` (5-10 min measured). If no candidate type has capacity in any region, the GPU chapter uses its recorded output |
| T-5 min | `make demo-reset` (deletes the limits-demo Pods, the `np-denied` namespace and the GPU test Pods; scales the CPU pool back to 1); `make preflight` again |
| During | `make demo` (pauses after every step; `DEMO_NO_PAUSE=1` does not); `make demo CHAPTERS=1,2,4,5` for a shorter slot; a failed step names its fallback, `make demo-replay CHAPTER=n`, and the demo goes on |
| After the demo | review `verification/demo-<ts>/`; `make promote-demo RUN=verification/demo-<ts>` keeps it as the replay source in `verification/final/demo/` |
| Wrap-up | `make gpu-down`, then `make gpu-status` shows no instance; decide whether GCP stays up or `make down` |

Screen safety: never `cat .env` or the kubeconfig on screen, and never print an access token; demo
output goes through the same redaction filter as `make verify`.

## Troubleshooting

| Symptom | Cause (measured) | Remedy |
|---|---|---|
| `make down` warns that it skipped the drains | the IAP tunnel had died while idle (2026-10-22) | run `make tunnel` before `make down`, so the drains run; the destroys remove the VMs either way |
| C7 reports the oom Pod's reason as `Error` (exit 137), not `OOMKilled` | containerd reported the wrong reason (1 of 6 runs); the node's kernel log showed `Memory cgroup out of memory: Killed process (tail) anon-rss:64792kB` (`CONSTRAINT_MEMCG`), so the 64Mi limit was enforced | re-run `scripts/verify.sh C`; planned fix: C7 also accepts exit 137 plus a kernel memcg OOM record |
| B3 reports 3 WireGuard peers instead of 2 | flannel's wireguard-native backend on `k3s-cp` keeps a stale peer (it carries no traffic): seen after a GPU node was deleted (`make gpu-down`, or E2's replacement) and, intermittently, after `k3s-cpu-1` re-keyed on reboot | on `k3s-cp`: `sudo wg show flannel-wg latest-handshakes`, then `sudo wg set flannel-wg peer <stale-public-key> remove` for each key that matches no Node's `flannel.alpha.coreos.com/backend-data` PublicKey (or restart k3s); planned fix: `gpuctl down` removes the peer |
| `make gpu-apply` or `gpuctl down` fails with a timeout or HTTP 504 from Lambda (Lambda write API timeouts/504) | Lambda's write API: on 2026-10-22, instance launches and firewall updates timed out even at 120 s, with HTTP 504, for about 30 minutes, while reads took about 0.4 s (resolved about 13:10 PDT). `gpuctl`'s Lambda API timeout is a fixed 30 s (`TIMEOUT_S` in `lambda/gpuctl/lambda_api.py`) | `make gpu-status` to confirm what exists (every failure that day left no instance: 0 instances, no drift); retry later, and use the recorded GPU output meanwhile; planned fix: a configurable timeout |
| `gpuctl plan --set-count a6000=2` prints `plan refused: budget_usd_per_hour ...` | by design: every new GPU node is priced at the pool's most expensive candidate ($1.99/h, the A100), so the 3.0 budget fits one new node per plan | raise `budget_usd_per_hour` or drop the A100 fallback ([GPU nodes](#gpu-nodes)) |
| `gpu-apply`'s join step takes about 4 minutes | `apt-get update` over the Lambda image's package sources, then the installs (1m03s and 4m06s in two runs) | wait; the join is bounded (SSH 300 s, Ready 300 s) |
| `gpu-apply` launches an A10 instead of an A6000 | no `gpu_1x_a6000` capacity in any listed region | expected fallback; `make gpu-status` shows capacity. E2 waits a bounded time for a freed unit before it gives up |
| `make up` waits for `node/k3s-cp` that never appears | the node registered under the GCE FQDN hostname (first live run, before `node-name: k3s-cp`) | current code pins the name; on a cluster built before that, `kubectl delete node <fqdn-name>` |
| Pod traffic dead on a node after a re-run of the bootstrap | `sysctl --system` applied GCE's `60-gce-network-security.conf` (`ip_forward=0`) before `90-kubelet.conf` existed | current code writes `90-kubelet.conf`; on an old node `sudo sysctl -w net.ipv4.ip_forward=1`, then `make up` |
| `nvidia.com/gpu` missing right after a GPU join | the device plugin registers about 20 s after the node is Ready | wait; D2 retries for 90 s |
| `curl` from a non-allowed namespace to `hello` exits 7, not 28 | kube-router rejects denied traffic instead of dropping it | expected; C8 accepts 7 or 28 with a positive control |

## Measured timings

From live runs on 2026-10-21 and 2026-10-22. "Final run" numbers come from the Task 24 final
from-scratch build, 2026-10-22 (make output; its `make gpu-apply` ran at about 10:25 PDT, before the
verify runs, when Lambda still launched instances); "final relaunch" is the `make gpu-apply` that
relaunched the GPU node on the same build after Lambda's write API recovered, just before the final 24/24 run;
"rehearsal" is the rebuild rehearsal earlier the same day. Each chapter's measured time is in the
`timings.txt` of a demo run.

**Bootstrap and bring-up**

| Operation | Time |
|---|---|
| `make init`, first run (bucket, APIs) | about 1m15s |
| `make init`, re-run without changes | 7.5 s |
| Base stack apply | about 1 min |
| `make up` from scratch, final run | 6m19s |
| `make up` from scratch, rehearsal | 6m59s |
| `make up` re-run with nothing to change | 92 s (no restart, guard rules unchanged) |

**CPU pool** (each measured twice)

| Operation | Time |
|---|---|
| Join of one CPU worker | 107-112 s |
| `make cpu-scale N=2` from 1 | 156 s |
| `make cpu-scale N=1` from 2 | 179-182 s |

**GPU nodes** (`gpu_1x_a10`)

| Phase of `make gpu-apply` | Final run | Final relaunch | Task 21 run | Rehearsal run |
|---|---|---|---|---|
| Total | 5m59s | 6m03s | 5m43s | 9m41s |
| Launch | 8 s | 8 s | 9 s | 10 s |
| Wait until active with an IP | 4m26s | 4m25s | 3m55s | 4m46s |
| `terraform apply` gpu-access | 15 s | 15 s | 15 s | 15 s |
| `sync-peers` | 15 s | 19 s | 14 s | 18 s |
| Join | 50 s | 49 s | 1m03s | 4m06s |

`make gpu-down`: 36-39 s. `make down`, whole teardown, final run: 6m10s (`k3s-cpu-1` drained and
deleted, then the 6 workers and 10 base resources destroyed).

**Verification**

| Run | Result | Time |
|---|---|---|
| A-C, no GPU node | 18 PASS, D SKIP | 258-263 s |
| A-D, one GPU node | 24/24 PASS | 412 s |
| `RESILIENCE=1` on a freshly rebuilt cluster (rehearsal) | 27/27 PASS, 0 skipped, gitleaks clean | 21m54s |
| A-C, 12:50 run on the final build (no GPU node: Lambda write API timeouts) | 18 PASS, 0 failed, 6 skipped (D) | 4m20s |
| A-D, final run, 13:20 (after the Lambda write API recovered) | 24/24 PASS, 0 failed, 0 skipped | 6m52s |

**Demo rehearsal** ([verification/final/demo/timings.txt](../verification/final/demo/timings.txt),
`verification/demo-2026-10-22T1328-0700`)

| Chapter | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| Time | 3 s | 73 s | 49 s | 148 s | 111 s | 343 s | 5 s |

Chapters 1-6: 12m07s against a 25-minute target. Chapter 6's last step
(`gpuctl plan --set-count a6000=2`) is refused by the budget guard as designed ($3.28/h over the
$3.00/h cap: a new node is priced at the pool's most expensive candidate, a100 at $1.99/h);
`demo.sh` records it as a failed step, so the recording was promoted with `PROMOTE_ALLOW_FAIL=1`
(planned fix: chapter 6 treats the refusal as the expected outcome). `make demo-replay CHAPTER=n`
plays the recordings.

**Network:** iperf3 across nodes over WireGuard about 205 Mbit/s; pod-to-pod ICMP payload 1352 bytes
passes with DF set, 1353 does not.
