---
status: accepted
date: 2026-10-22
---

# GPU nodes are cattle, reconciled from a spec by `gpuctl`

## Context and Problem Statement

GPU capacity is rented by the hour from Lambda Cloud. Lambda's model shapes everything:

- **No stop state.** An instance bills until it is terminated; shutting it down from inside keeps
  billing. Terminating loses the local disk and every pulled image.
- **A new public IP at every launch**, behind NAT (measured), so every launch changes three
  allow-lists: the GCP `gpu-access` rules, `k3s-cp`'s `k3s_peers` set and the Lambda rulesets.
- **Firewall rulesets attach only at launch** and are per region; their rules can be PATCHed later.
- **Capacity varies by type and region.** In both live sessions (2026-10-21 and 2026-10-22) the only
  1-GPU type with capacity was `gpu_1x_a10` ($1.29/h, us-west-1 or us-east-1); `gpu_1x_a6000`
  ($1.09/h) and `gpu_1x_a100` ($1.99/h) showed none.
- **No usable Terraform provider** ([ADR-0003](0003-terraform-three-stacks.md)).
- **The cost of a mistake is real:** a forgotten A10 costs about $31 a day.

How should GPU nodes be created, joined, replaced and removed so that the cluster, the firewalls and
the bill always agree, and a failure never leaves an orphan billing?

## Decision Drivers

- The source of truth for "what exists" must be Lambda itself, not a file that can go stale.
- Any command can fail halfway (capacity, network, Ctrl-C); a re-run must converge.
- Hard limits on count and cost, checked against live prices before anything is launched.
- No instance left billing after a failed join, and no allow-list entry for an instance that is gone.
- Readable by a reviewer: one spec file, one command per intent.

## Considered Options

- Long-lived ("pet") GPU nodes created by hand and kept
- Terraform with the community Lambda provider
- A YAML spec plus a reconciler without local state (`gpuctl`)
- The Kubernetes cluster-autoscaler

## Decision Outcome

Chosen option: **a YAML spec plus a stateless reconciler**, because Lambda's lifecycle is create and
destroy only, so a node is cheapest and safest to treat as disposable, and a reconciler that reads
Lambda on every run cannot disagree with it.

The spec is [lambda/gpu-nodes.yaml](../../lambda/gpu-nodes.yaml):

```yaml
budget_usd_per_hour: 3.0
max_nodes: 3
pools:
  a6000:
    count: 1
    instance_type: gpu_1x_a6000
    fallback_types: [gpu_1x_a10, gpu_1x_a100]
    regions: [us-west-1, us-west-2, us-east-1]
    image_family: lambda-stack-22-04
```

`gpuctl` ([lambda/gpuctl/](../../lambda/gpuctl/), run as `scripts/gpuctl`) owns every instance whose
name starts with `k3s-gpu-`; nodes are named `k3s-gpu-<pool>-<n>`, and the Lambda instance name, the
hostname and the Kubernetes node name are the same.

| Command | Does |
|---|---|
| `make gpu-plan` | diffs the spec against Lambda's live instances; prints creates, deletes and the hourly cost at live prices. Read-only |
| `make gpu-apply` | shows the plan, asks, then converges: launches what is missing, deletes what is surplus |
| `make gpu-down` | removes the autoscaler override file and sets every pool to 0, then converges |
| `make gpu-status` | capacity per pool, type and region; the cluster's GPU instances; a drift check (exit 3 on drift) |
| `make sync-peers` | rewrites the three allow-lists from the current node IPs and the operator's IP |

**Launching a node** (each step prints its duration):

1. Pick the first (type, region) with capacity, in the order `[instance_type] + fallback_types` ×
   `regions`. `insufficient-capacity` moves to the next pair.
2. Ensure the region's ruleset `k3s-gpu-<region>` exists with the current peers, then launch with it
   attached, one SSH key and the explicit image family.
3. Poll until the instance is `active` with an IP; a `null` IP is never passed to Terraform.
4. Apply `gpu-access` with every live GPU IP, then `sync-peers`.
5. `scripts/join-node.sh` with the GPU branch ([ADR-0005](0005-gpu-device-plugin-and-runtimeclass.md)).

**Deleting a node:** drain, `kubectl delete node`, terminate, apply `gpu-access` without its IP,
`sync-peers`, and remove its `known_hosts` entry. A drain is skipped only when the cluster is known
to be gone (no base stack outputs, no kubeconfig) or with `--no-drain`. When base outputs exist but
the API is unreachable (for example `make tunnel` is not running), `gpuctl` refuses that node's
undrained terminate and says "start make tunnel and re-run, or pass --no-drain".

### Guardrails

- **Budget at live prices.** The plan sums `price_cents_per_hour` from Lambda's API: kept instances
  at their own type's price, new nodes at the pool's most expensive candidate type (today the A100,
  $1.99/h), and refuses a plan above `budget_usd_per_hour` (`gpuctl plan` prints the plan, then
  `plan refused: budget_usd_per_hour ...`). With today's prices and the A100 listed as a fallback, the
  3.00 budget admits only one GPU node in total: a second node next to a running A10 is refused
  ($1.29 + $1.99 = $3.28), and so are two new nodes ($3.98). Running two needs a budget of at least
  3.98, or the A100 dropped from the fallbacks (then 2 × $1.29 = $2.58).
- **Count.** `max_nodes` caps the total; the `gpu-access` stack refuses more IPs than its own copy of
  the cap, and `gpuctl` passes the spec's value on every apply.
- **Architecture.** arm64 types (GH200) are refused when the spec is loaded.
- **Rate limits.** At least 1 s between API calls and 12 s between launches.
- **Clean-up on failure.** Instances launched in this run that fail to join are terminated, and their
  IPs leave `gpu-access` and the peer lists, unless `--keep-on-failure` keeps them for debugging (they
  bill until `make gpu-down`).
- **Confirmation.** `gpu-apply` and `gpu-down` ask unless `AUTO_APPROVE=1`.
- **Duplicates.** If two live instances share a node name, the one with the best status (`active` >
  `booting` > `unhealthy`), then the smallest id, is kept; the other is terminated by id only, without
  touching the Node or host key that belong to the kept one.
- **Statuses.** `booting` (no IP yet) and `unhealthy` count as existing, so a re-run does not launch a
  second copy; `terminating` and `terminated` do not.

### Drift check

`gpuctl status` compares five sources: the target counts, Lambda's instances, the cluster's GPU
nodes, the IPs in the `gpu-access` state and the members of `k3s_peers`. Any problem exits 3. An
allow-list entry (`gpu-access` rule or `k3s_peers` member) for an IP with no live instance is
reported first, as a security problem; a live instance missing from the cluster or an allow-list,
and a stale Node, are drift. `gpuctl` keeps no local state, so it cannot tell a pool that
`gpu-down` emptied from one never applied: a pool with fewer live instances than its target (the spec
count, or the autoscaler's override) is only a note, never drift, and a pool with more is drift.

So after `make gpu-down` the exit code alone does not prove the session is over: one instance left
in a pool whose target is 1 is neither drift nor a problem. The operator confirms by reading the
instance list in `make gpu-status`, which must read `GPU instances (k3s-gpu-*): 0`.

### Consequences

- Good, because there is no state file to lose or corrupt: deleting the checkout loses nothing about
  GPU nodes, and an instance terminated in the Lambda console is simply gone on the next run.
- Good, because every command is idempotent: an interrupted `gpu-apply` is finished by running it
  again, and a second `gpu-down` is a no-op.
- Good, because a GPU node is replaceable in minutes: check E2 terminates and relaunches it, and the
  GPU and connectivity checks pass again on the new instance.
- Bad, because each launch starts from a fresh image: the join pays for `apt-get update` and package
  installs on the Lambda image, and every container image is pulled again.
- Bad, because availability is Lambda's: on the measured days only the A10 fallback launched. The
  fallbacks make that a planned path, not an outage.
- Bad, because `gpuctl` runs on the operator's machine with an account-wide API key
  ([threat model](../threat-model.md)). There is no in-cluster controller.

### Confirmation

Measured `make gpu-apply`, one `gpu_1x_a10`:

| Phase | 2026-10-21 run | 2026-10-22 rehearsal |
|---|---|---|
| Total | 5m43s | 9m41s |
| Launch call | 9 s | 10 s |
| Wait until `active` with an IP | 3m55s | 4m46s |
| `terraform apply` gpu-access | 15 s | 15 s |
| `sync-peers` | 14 s | 18 s |
| Join | 1m03s | 4m06s |

The rehearsal's join was slower because of `apt` on the Lambda image; Lambda's own provisioning time
dominates both runs. `make gpu-down` took 36-39 s. With `make verify RESILIENCE=1`, E2 (replace the GPU
node, then D1-D6 and B1) and E3 (a forged gpu-access rule found by `gpuctl status` and removed by
`gpuctl apply`) passed on 2026-10-22. Unit tests in [lambda/tests/](../../lambda/tests/) cover the
planner, the reconciler and the drift check against a fake Lambda API.

## Pros and Cons of the Options

### Pet nodes created by hand

- Good, because there is nothing to build.
- Bad, because Lambda cannot stop an instance, so "keeping" a node means paying for it around the
  clock, and the three allow-lists must be edited by hand on every change.

### Terraform with the community provider

- Good, because it would share a workflow with the GCP stacks.
- Bad, because it cannot set the image family or the firewall rulesets, and its state would go stale
  whenever Lambda terminates an instance on its own or capacity changes the type.

### Spec plus stateless reconciler (chosen)

- Good, because Lambda's API is the state, and every guard runs before any launch.
- Bad, because it is our code: about 2,700 lines of Python (without the stretch autoscaler), with its
  own tests.

### cluster-autoscaler

- Good, because it is the standard way to add nodes on demand.
- Bad, because it has no Lambda provider, and on a non-GKE cluster it needs cloud credentials inside
  the cluster and a long-lived join token; see [ADR-0006](0006-ip-allowlist-vs-identity-network.md).

## More Information

- Join procedure: [scripts/join-node.sh](../../scripts/join-node.sh); allow-list construction:
  [lambda/gpuctl/peers.py](../../lambda/gpuctl/peers.py); planning and limits:
  [lambda/gpuctl/planner.py](../../lambda/gpuctl/planner.py).
