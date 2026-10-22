---
status: accepted
date: 2026-10-22
---

# Peers authenticated by IP allow-lists now; when to move to an identity-based network

## Context and Problem Statement

With option A ([ADR-0002](0002-network-k3s-wireguard-native.md)) the network admits a peer by its
public IP. Every node's `/32` is kept in three places:

1. the GCP firewall: the `workers` and `gpu-access` stacks, from the worker inventory and the live
   GPU IPs ([ADR-0003](0003-terraform-three-stacks.md));
2. the `k3s_peers` set in `k3s-cp`'s host table `k3s_guard`, which guards 6443 and 10250;
3. the Lambda firewall: the account-wide rule list (operator SSH only) and one `k3s-gpu-<region>`
   ruleset per region.

`gpuctl sync-peers` rebuilds all three from scratch from the Terraform outputs, Lambda's API and the
operator's current public IP, and every scaling command calls it. Kubernetes itself authenticates
nodes with certificates and WireGuard authenticates the pod tunnel with keys, but the outer gate is
the IP. Is that the right gate for this cluster, and at what point does it stop being right?

## Decision Drivers

- Nothing on the internet reaches a cluster port unless the operator put its address in a list.
- No third-party control plane in the trust base.
- No long-lived credential on a node or in the cluster that can add nodes or change firewalls.
- Simple to verify: a reviewer can list every rule and read every source.

## Considered Options

- IP allow-lists in three places, rebuilt from one source by `sync-peers`
- An identity-based overlay: host WireGuard with per-node keys (option B of ADR-0002)
- An identity-based overlay: Tailscale with Tailnet Lock (option C of ADR-0002)
- Automatic node scaling with the Kubernetes cluster-autoscaler

## Decision Outcome

Chosen option: **IP allow-lists rebuilt by `sync-peers`**, because at this size (one control plane,
one to a few CPU workers, up to `max_nodes` 3 GPU nodes) it gives a closed network without a new
trust anchor, every rule is a reviewable `/32`, and the credentials that change the lists stay on the
operator's machine.

How the lists stay correct:

- **Rebuilt, never patched.** `sync-peers` computes each list from the current `PeerSet` and replaces
  it whole: the `k3s_peers` set is flushed and refilled in one nft transaction and persisted for the
  next boot; each Lambda ruleset and the global rule list are sent as a complete
  `rules` list (PATCH replaces the list). When the operator's IP
  changes, the old IP disappears everywhere (unit tests `test_operator_ip_change_leaves_no_trace_of_old_ip`
  and `test_sync_replaces_old_operator_ip`).
- **Order.** A new node's IP enters the lists before its agent connects, and leaves them after it is
  drained and deleted ([ADR-0004](0004-gpu-nodes-as-cattle-spec-reconciler.md)).
- **Checked.** `gpuctl status` reports an allow-list entry with no live instance as a security
  problem (exit 3); verification check B6 requires every live GCP rule source to be a `/32` or IAP's
  range; check E3 forges a stale rule and requires `status` to find it and `apply` to remove it.
- **Operator IP.** Detected through `checkip.amazonaws.com` or set with `OPERATOR_IP`; it only ever
  opens tcp/22 on the Lambda nodes. GCP shells go through IAP and OS Login, and `kubectl` through an
  IAP tunnel, so a change of the operator's network never opens a cluster port.

### When to switch

Move to an identity-based network (B, or C with Tailnet Lock) when any of these holds:

| Trigger | Why the allow-list design breaks |
|---|---|
| Node churn is automatic or frequent (autoscaling, spot capacity) | every change needs an apply and a `sync-peers` from a machine holding GCP and Lambda credentials; a measured `sync-peers` takes 14-18 s and touches every region's ruleset |
| More than a few dozen nodes | three list types grow with N, and each Lambda region's ruleset carries two rules per node; IP churn makes the GCP rule set large |
| A node has no stable public IPv4 or no inbound UDP (carrier-grade NAT, an on-premises site) | there is no address to allow-list; an overlay that dials out solves it |
| 6443 and 10250 must not be reachable even from peer addresses | only an overlay that carries all node traffic (B or C) hides them |
| A provider reuses IPs quickly between tenants | a stale entry briefly admits a stranger's machine to the TLS endpoints (they still need credentials) |

### Why not the cluster-autoscaler

- It has no Lambda provider.
- On a non-GKE cluster, its GCE provider expects GCE-format provider IDs and managed instance groups.
- It runs inside the cluster and needs a GCP credential that can resize instance groups, which breaks
  the "service accounts have no IAM roles" rule.
- New nodes must join unattended, so a long-lived join token would live in an instance template.
- It cannot update the allow-lists: a node it adds is not in `k3s_peers` or the Lambda rulesets, so
  it could not join at all. Autoscaling and IP allow-lists conflict by design.

The optional stretch autoscaling demo therefore runs a small loop on the operator's machine that calls
the same reviewed paths (`cpu-pool.sh --guarded`, `gpuctl apply`), so no credential enters the
cluster. The cost is that scaling stops when the operator's machine is off.

### Consequences

- Good, because there is no extra daemon, coordination service or key distribution.
- Good, because the attack surface is enumerable: B6 lists it in every verification run.
- Good, because the credentials that change firewalls (gcloud ADC, the Lambda API key) never leave the
  operator's machine.
- Bad, because the operator's machine is on the critical path of every node change.
- Bad, because IP is a weak identity: anyone who can send from an allowed IP reaches the TLS
  endpoints. That is acceptable only because the endpoints themselves require certificates or kubelet
  authorisation ([threat model](../threat-model.md)).
- Bad, because three lists can disagree after a partial failure; `gpuctl status` detects that and a
  re-run of the failed command repairs it.

### Confirmation

- Offline: [lambda/tests/test_peers.py](../../lambda/tests/test_peers.py) and
  [lambda/tests/test_sync_peers.py](../../lambda/tests/test_sync_peers.py).
- Live: B6 and E3 passed in the 2026-10-22 `make verify RESILIENCE=1` run. On 2026-10-21, from the
  operator's machine, the GPU node's tcp/10250, 6443, 443 and 8888 timed out and only tcp/22
  answered; `gpuctl rules` showed the global list reduced to the operator's SSH `/32`.

## Pros and Cons of the Options

### IP allow-lists (chosen)

- Good, because it needs nothing beyond the clouds' firewalls and nft.
- Bad, because of the operator-in-the-loop and churn limits above.

### Host WireGuard with per-node keys

- Good, because peers are identified by key, and only udp/51820 is public.
- Bad, because the keys and peer entries are distributed by our own tooling, which is the same
  operator-in-the-loop problem for the key lists; and see ADR-0002 option B.

### Tailscale with Tailnet Lock

- Good, because nodes join by key with NAT traversal, and Tailnet Lock keeps the coordination service
  from adding nodes.
- Bad, because it adds a third-party control plane and auth-key handling, and the k3s integration is
  experimental (ADR-0002 option C).

### cluster-autoscaler

- Good, because it is the standard autoscaler.
- Bad, for the reasons listed above.

## More Information

- `sync-peers`: [lambda/gpuctl/reconcile.py](../../lambda/gpuctl/reconcile.py); the persisted peer
  set: [node/guard/k3s-guard-peers.sh](../../node/guard/k3s-guard-peers.sh).
- The identity-network and in-cluster autoscaler items are expanded in the [P2 roadmap](../p2-roadmap.md).
