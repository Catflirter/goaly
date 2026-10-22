# Runbook

Day-to-day operation of the hybrid cluster: bring-up, scaling, GPU sessions, verification,
break-glass access, teardown and the demo day. Every `make` target here is safe to re-run; each one
that changes cloud resources shows a plan and asks first unless `AUTO_APPROVE=1` is set. Architecture
and rationale: [README](../README.md). Times are Pacific (PDT).

## Prerequisites

**Accounts**

- A GCP project with billing enabled. The operator account needs `roles/compute.osAdminLogin`,
  `roles/iap.tunnelResourceAccessor`, `roles/iam.serviceAccountUser` on the `k3s-cp` and `k3s-worker`
  service accounts, and the rights to create the resources (project Owner or Editor for the first
  `make init`, which enables six APIs).
- A Lambda Cloud account, an API key, and an SSH public key uploaded to Lambda.

**Tools on the operator machine** (Linux or macOS; the scripts keep to bash 3.2)

| Tool | Version | Used by |
|---|---|---|
| terraform | >= 1.9.0 | every stack |
| gcloud | current | IAP SSH, the API tunnel, `cpu-stop`/`cpu-start` |
| kubectl, helm | current | cluster objects, the device plugin chart |
| python3 | >= 3.11 | `gpuctl` (in `.venv`, from `make venv`) |
| jq, ssh, coreutils `timeout` | any | scripts |
| gitleaks | >= 8.19 | `make lint`, `make verify`, `make promote-verify` |
| kubeconform, shellcheck (via pre-commit) | any | `make lint` |
| flock (util-linux) | any | only for `cpu-pool.sh --guarded` |

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
the four per-worker resources of the added or removed workers may change) and holds a lock so two
guarded runs never overlap.

## GPU nodes

The spec is [lambda/gpu-nodes.yaml](../lambda/gpu-nodes.yaml); `gpuctl` reconciles Lambda to it
([ADR-0004](adr/0004-gpu-nodes-as-cattle-spec-reconciler.md)).

```bash
make gpu-status   # capacity per pool/type/region, this cluster's instances, drift (exit 3)
make gpu-plan     # creates/deletes and the hourly cost at live prices
make gpu-apply    # asks, launches, opens the firewalls, joins, labels
make verify       # D1-D6 now run instead of SKIP
make gpu-down     # asks, drains, deletes the Node, terminates, closes the firewalls
make gpu-status   # want: no instance of this cluster, no drift
```

- **Every GPU session ends with `make gpu-down` and a `make gpu-status` that shows no instance.**
  Lambda bills until an instance is terminated; shutting it down from inside does not stop billing.
- To change the count, edit `count` in the spec; `scripts/gpuctl plan --set-count a6000=2` previews a
  count without editing the file.
- Capacity: `gpu-status` shows which (type, region) pairs have stock. On both measured days only
  `gpu_1x_a10` had any; `gpu-apply` falls back to it automatically.
- The budget check prices new nodes at the pool's most expensive candidate (the A100 at $1.99/h): with
  `budget_usd_per_hour: 3.0`, one node can be added at a time. Raise the budget or drop the A100
  fallback to launch two at once.
- `scripts/gpuctl apply --keep-on-failure` keeps a node whose join failed, for debugging. It bills
  until `make gpu-down`.
- `make sync-peers` after the operator's network changes: it replaces the operator's old IP with the
  new one in Lambda's SSH rules.

## Verification

```bash
make verify                       # A-D into verification/<YYYY-MM-DDTHHMM-0700>/
make verify RESILIENCE=1          # adds E1-E3: reboots k3s-cpu-1, replaces the GPU node, forges drift
scripts/verify.sh B               # one group
make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>   # scan and copy to verification/final/
```

A run exits 0 when no check failed (SKIP is not a failure) and gitleaks found nothing in it. D-group
checks SKIP without a GPU node. E2 replaces the GPU node only when its pool shows capacity or the node
itself holds a unit of a candidate type; otherwise it SKIPs, which means "not verified": replace the
node by hand (`make gpu-down`, `make gpu-apply`) and re-run `scripts/verify.sh E`.
`make promote-verify` refuses a run with a failed check, and a run with a skipped check unless
`PROMOTE_ALLOW_SKIP=1`.

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
ssh -i "$SSH_PRIVATE_KEY_PATH" -o UserKnownHostsFile=.run/known_hosts_k3s-gpu-a6000-1 ubuntu@"$ip"
```

**A GPU node must go and the cluster is unreachable.** `scripts/gpuctl down --no-drain` terminates it
without the drain. When the cluster is known to be gone (no base outputs, no kubeconfig), `make
gpu-down` does that by itself. Last resort: terminate it in the Lambda console, then run
`make gpu-apply` or `make gpu-down` so the firewalls and `k3s_peers` follow.

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
make down                                 # asks for the GCP project id; AUTO_APPROVE=1 skips the question
```

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
scripts/gpuctl list-nodes
```

All four print nothing.

## Demo-day checklist

The demo is run live from the operator's machine. The demo tooling (`make preflight`, `make demo`,
`make demo-replay`, `make demo-reset`) is added by Task 25, which also records the rehearsal timings
below; until then each step lists the commands it automates.

| When | Action |
|---|---|
| Day before | `make up`; `make verify`; check `SUMMARY.md` is all PASS (D included only with a GPU node); GCP stays up overnight (see [Cost](../README.md#cost)) |
| T-60 min | `make preflight`: tool versions; `gcloud auth list` and `gcloud auth application-default print-access-token`; `scripts/gpuctl status` (also checks Lambda capacity); `make sync-peers` (refreshes the operator IP for this network); `scripts/tunnel.sh status` or `make tunnel`; every node Ready |
| T-30 min | `make gpu-apply` (5-10 min measured). If no candidate type has capacity in any region, the GPU chapter uses its recorded output |
| T-5 min | `make demo-reset` (removes demo pods, the `np-denied` namespace, GPU test pods; scales the CPU pool back to 1); `make preflight` again |
| During | `make demo` (pauses after every step); `make demo CHAPTERS=1,2,4,5` for a shorter slot; `make demo-replay CHAPTER=n` if a live step fails |
| After | `make gpu-down`, then `make gpu-status` shows no instance; decide whether GCP stays up or `make down` |

Screen safety: never `cat .env` or the kubeconfig on screen; demo output goes through the same
redaction filter as `make verify`.

## Troubleshooting

| Symptom | Cause (measured) | Remedy |
|---|---|---|
| `make down` warns that it skipped the drains | the API tunnel had died while idle (2026-10-22) | `make tunnel`, then re-run; the destroys remove the VMs either way |
| `gpu-apply`'s join step takes about 4 minutes | `apt-get update` over the Lambda image's package sources, then the installs (1m03s and 4m06s in two runs) | wait; the join is bounded (SSH 300 s, Ready 300 s) |
| `gpu-apply` launches an A10 instead of an A6000 | no `gpu_1x_a6000` capacity in any listed region | expected fallback; `make gpu-status` shows capacity. E2 waits a bounded time for a freed unit before it gives up |
| `make up` waits for `node/k3s-cp` that never appears | the node registered under the GCE FQDN hostname (first live run, before `node-name: k3s-cp`) | current code pins the name; on a cluster built before that, `kubectl delete node <fqdn-name>` |
| Pod traffic dead on a node after a re-run of the bootstrap | `sysctl --system` applied GCE's `60-gce-network-security.conf` (`ip_forward=0`) before `90-kubelet.conf` existed | current code writes `90-kubelet.conf`; on an old node `sudo sysctl -w net.ipv4.ip_forward=1`, then `make up` |
| `nvidia.com/gpu` missing right after a GPU join | the device plugin registers about 20 s after the node is Ready | wait; D2 retries for 90 s |
| `curl` from a non-allowed namespace to `hello` exits 7, not 28 | kube-router rejects denied traffic instead of dropping it | expected; C8 accepts 7 or 28 with a positive control |

## Measured timings

From live runs on 2026-10-21 and 2026-10-22. Task 24 re-measures every row from a final from-scratch
run, and Task 25 adds the per-chapter demo timings after the rehearsals.

**Bootstrap and bring-up**

| Operation | Time |
|---|---|
| `make init`, first run (bucket, APIs) | about 1m15s |
| `make init`, re-run without changes | 7.5 s |
| Base stack apply | about 1 min |
| `make up` from scratch (rebuild rehearsal, 2026-10-22) | 6m59s |
| `make up` re-run with nothing to change | 92 s (no restart, guard rules unchanged) |

**CPU pool** (each measured twice)

| Operation | Time |
|---|---|
| Join of one CPU worker | 107-112 s |
| `make cpu-scale N=2` from 1 | 156 s |
| `make cpu-scale N=1` from 2 | 179-182 s |

**GPU nodes** (`gpu_1x_a10`)

| Phase of `make gpu-apply` | Task 21 run | Rehearsal run |
|---|---|---|
| Total | 5m43s | 9m41s |
| Launch | 9 s | 10 s |
| Wait until active with an IP | 3m55s | 4m46s |
| `terraform apply` gpu-access | 15 s | 15 s |
| `sync-peers` | 14 s | 18 s |
| Join | 1m03s | 4m06s |

`make gpu-down`: 36-39 s. `make down`, whole teardown: 6m10s.

**Verification**

| Run | Result | Time |
|---|---|---|
| A-C, no GPU node | 18 PASS, D SKIP | 258-263 s |
| A-D, one GPU node | 24/24 PASS | 412 s |
| `RESILIENCE=1` on a freshly rebuilt cluster | 27/27 PASS, 0 skipped, gitleaks clean | 21m54s |

**Network:** iperf3 across nodes over WireGuard about 205 Mbit/s; pod-to-pod ICMP payload 1352 bytes
passes with DF set, 1353 does not.
