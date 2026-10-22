---
status: accepted
date: 2026-10-22
---

# Cross-cloud network: k3s `wireguard-native` with external IPs

## Context and Problem Statement

The control plane runs on GCP (VPC `k3s-core`, us-west1), the CPU workers in a second GCP VPC with
no peering (`k3s-edge`, us-central1), and the GPU workers on Lambda Cloud. The three networks have
no private connection; Lambda on-demand instances offer no VPC peering or interconnect. Every node
therefore reaches the others over public IPv4 addresses, and both providers NAT them: GCP's external
IP is a 1:1 NAT that never appears on the guest NIC, and a Lambda instance's NIC carries only a
private `/20` address (measured). The kubelet refuses a `--node-ip` that is not on a local interface,
so a public IP cannot simply be the node IP.

How should nodes in three networks form one cluster so that pod traffic is encrypted, no cluster port
is open to the internet, and the design still fits a one-to-two-day build?

## Decision Drivers

- Every cross-provider packet of pod traffic is encrypted.
- No cluster port is reachable by everyone on the internet; every inbound rule names a peer.
- A documented, supported path rather than a home-made one, so failures can be looked up.
- Few moving parts: each extra daemon, interface or boot-ordering dependency is another failure mode.
- Lambda's constraints: a new public IP at every launch; firewall rulesets attach only at launch.
- A written fallback for the unknowns of Lambda's NAT and NIC MTU.

## Considered Options

- **A.** k3s multicloud mode: flannel `wireguard-native` with `--node-external-ip` and
  `--flannel-external-ip`
- **B.** Host WireGuard `wg0`, hub and spoke on the GCP static IP, with k3s on the overlay addresses
  (`node-ip` = wg IP, flannel VXLAN inside `wg0`)
- **C.** Tailscale: k3s `--vpn-auth`, or tailscaled managed separately with `flannel-iface: tailscale0`
- **D.** kubeadm with Cilium or Calico WireGuard encryption

## Decision Outcome

Chosen option: **A, k3s `wireguard-native` with external IPs**, because it is the documented k3s
path for this topology, it encrypts all pod traffic with a full WireGuard mesh using three server
settings and one agent setting, and it adds no interface, daemon or boot-ordering dependency. The
price is that tcp/6443 and tcp/10250 travel over the internet protected by TLS, not by the tunnel. We
accept it because both ports admit only peer `/32` addresses at two independent layers (cloud
firewall and the host nft table `k3s_guard`), both are authenticated TLS endpoints, and the option
that hides them (B) adds a hand-built layer whose failure takes nodes down.

Server configuration ([node/k3s/server-config.yaml.tmpl](../../node/k3s/server-config.yaml.tmpl)):

```yaml
node-name: k3s-cp
node-external-ip: <cp_ip>
flannel-backend: wireguard-native
flannel-external-ip: true
```

Agents set `node-external-ip: <own public IP>` and no flannel options at all
([node/k3s/agent-config.yaml.tmpl](../../node/k3s/agent-config.yaml.tmpl)).

### Comparison

| | A: `wireguard-native` (chosen) | B: host `wg0` hub-spoke | C: Tailscale | D: kubeadm + Cilium WireGuard |
|---|---|---|---|---|
| Encrypted by WireGuard | all cross-node pod traffic | all node-to-node traffic, 6443 and 10250 included | all node-to-node traffic through `tailscale0` | pod-to-pod; node encryption is beta and disabled to and from control-plane nodes |
| Outside the tunnel, TLS only | agent to 6443; `k3s-cp` pods to the `kubernetes` Service (hairpin); metrics-server to 10250 | nothing | nothing | API server to kubelet 10250, kubelet certificate unverified by default |
| Public inbound ports | GCP: udp/51820, tcp/6443 and ICMP to `k3s-cp` from peer `/32`s; udp/51820 and ICMP between agents; tcp/10250 on agents from `<cp_ip>` only. Lambda: udp/51820 and ICMP from every node's `/32`, tcp/10250 from `<cp_ip>`, tcp/22 from the operator's `/32` | udp/51820 and ICMP from peer `/32`s | none (outbound 443, udp/3478, udp/41641) | as B: it needs a B-style underlay to cross NAT |
| Complexity | lowest: four documented settings | medium: keys per node, `wg-quick` ordered before k3s, double encapsulation (pod MTU 1330), not a documented k3s pattern | low to set up; the k3s integration is experimental; adds a third-party control plane, ACLs, route approval, auth keys | highest: kubeadm, CRI on Lambda's Docker containerd, CNI, kubelet certificate approval, plus the B underlay |
| Failure modes | a Lambda IP change needs firewall and peer-list updates; hairpin; metrics-server address choice; MTU differences | `wg0` down means NotReady; a missing `flannel-iface` silently binds flannel to the NIC; spoke-to-spoke traffic transits the hub | coordination outage blocks joins; relay paths; fixed MTU 1280 (pod MTU 1230); on k3s `v1.36.5` a tailscaled restart deletes flannel's VXLAN device and restarts k3s | NAT and node-ip handling; CRI misconfiguration; kubelet TLS; time overrun |

### Consequences

- Good, because pod traffic between all three networks is encrypted over a full mesh; there is no hub
  that spoke-to-spoke traffic must cross.
- Good, because there is no extra network layer to fail: no `wg0`, no tailscaled, no ordering between
  a VPN and k3s.
- Good, because the pod MTU stays at 1380 (one encapsulation) instead of 1330 (B) or 1230 (C).
- Bad, because **tcp/6443 and tcp/10250 cross the internet outside WireGuard.** Both require client
  certificates or kubelet authorisation and accept only peer `/32`s at the cloud firewall and in
  `k3s_guard`, but an on-path observer sees TLS metadata (endpoints, timing, sizes), and a TLS or
  authentication flaw in either service is reachable from any allowed peer address. The
  [threat model](../threat-model.md) carries this as a residual risk.
- Bad, because the `kubernetes` Service endpoint is the control plane's public IP, so pods on `k3s-cp`
  (CoreDNS, metrics-server) reach the API through a hairpin. **Measured 2026-10-21:** the
  EndpointSlice held `<cp_ip>` while the NIC carries only `10.10.0.2`; `ss` on `k3s-cp` showed those
  6443 connections arriving with `<cp_ip>` itself as the source (GCP's 1:1 NAT rewrites it on the way
  back in). The rule `fw-cp-allow-self` (`<cp_ip>/32`, tcp 6443 and 10250) and both control-plane
  addresses in `k3s_peers` were sufficient: CoreDNS was Ready about 3 s after it was recreated, and
  the peer accept rule counted 632 packets with 0 drops. The prepared fallback (a hairpin unit) was not
  needed.
- Bad, because with `--flannel-external-ip` metrics-server prefers each node's ExternalIP. We disable
  k3s's bundled metrics-server and deploy our own copy on the control plane with
  `--kubelet-preferred-address-types=ExternalIP,InternalIP,Hostname`
  ([k8s/system/metrics-server.yaml](../../k8s/system/metrics-server.yaml)), so an agent's 10250 only
  ever needs to accept `<cp_ip>`.
- Bad, because WireGuard subtracts 80 bytes from the NIC MTU, and the NICs differ: GCP 1460, Lambda
  1500 (measured). Mismatched tunnel MTUs drop large non-TCP packets. Every node's NIC is set to
  `UNDERLAY_MTU` (1460) and persisted through netplan before k3s starts, so every tunnel is 1380; the
  bootstrap exits 3 if a NIC is smaller. Check B1 sweeps it: a 1352-byte ICMP payload passes and 1353
  does not (`UNDERLAY_MTU` minus 108 and minus 107). **Measured** on the three-node matrix (control
  plane, CPU worker, Lambda A10): every pair passes 1352 and fails 1353.
- Bad, because a Lambda node's IP changes at every launch. `gpuctl` applies the `gpu-access` stack and
  runs `sync-peers` as soon as the IP is known and before the node joins
  ([ADR-0004](0004-gpu-nodes-as-cattle-spec-reconciler.md)).
- Bad, because the peers are IP allow-lists kept in three places. That is fine at this scale and is
  the main reason to change designs later ([ADR-0006](0006-ip-allowlist-vs-identity-network.md)).
- Measured, Lambda side (2026-10-21 and 2026-10-22, `gpu_1x_a10`): the public IP is NAT (the NIC
  holds a private `/20`), and the instance's egress IP equals the IP the API reports, so allow-lists
  keyed on the API IP are correct. WireGuard handshakes between the GPU node and both GCP nodes
  were 37-43 s old when read, through NAT on both sides; the 3×3 matrix passed; iperf3 across nodes gave
  about 205 Mbit/s. Option A holds on Lambda; no fallback was needed.

### When to fall back

- **To B** if WireGuard through Lambda's NAT fails or becomes unstable (for example a provider change
  to symmetric NAT). In B the Lambda node is a spoke that dials the GCP static IP and needs no inbound
  reachability, so Lambda's NAT behaviour stops mattering. B keeps k3s and changes only the transport:
  `wg0` on every node, `node-ip` and `flannel-iface` on the overlay, and no 6443 or 10250 rule on any
  public interface. It is also the choice if the requirement becomes "no cluster port reachable even
  from peer addresses".
- **To C** if a node has no usable public IPv4 or no inbound UDP (carrier-grade NAT, a restrictive
  site), or if nodes must join without the operator updating allow-lists, because Tailscale
  authenticates nodes by key and traverses NAT. C must come with Tailnet Lock or a self-hosted
  coordination server; otherwise the coordination service can add nodes to the network.
- **Not to D**: it combines kubeadm's costs ([ADR-0001](0001-k3s-not-kubeadm.md)) with B's underlay
  and gains nothing for this topology.

### Confirmation

- Group A: each node's flannel `backend-type` is the WireGuard backend and `public-ip` equals its
  ExternalIP.
- Group B: the N×N matrix (pod IP, Service DNS, MTU sweep); a capture on `k3s-cp`'s NIC during marked
  cross-node requests in which the marker never appears in clear text, every peer shows at least 40
  udp/51820 packets, and the only protocols to other nodes' public IPs are udp/51820, tcp/6443,
  tcp/10250 and ICMP; `wg show flannel-wg` (never the `dump` form) with a handshake under 3 minutes
  old for every peer; negative tests (6443 and 10250 time out from the operator's machine).
- Measured 2026-10-21: the capture saw 201 udp/51820 packets with the CPU worker, only udp/51820, and
  no marker; the 2026-10-22 run passed the same checks with the GPU node in the mesh.

## Pros and Cons of the Options

### A. k3s `wireguard-native` with external IPs

- Good, because it is documented by k3s and not marked experimental.
- Good, because it is a full mesh: any two nodes talk directly.
- Good, because WireGuard does not answer unauthenticated packets, so udp/51820 reveals nothing to a
  scanner even from an allowed source.
- Bad, because control-plane traffic is TLS on the internet rather than inside the tunnel.
- Bad, because of the hairpin, metrics-server and MTU issues above, each handled explicitly.

### B. Host `wg0` hub and spoke, k3s on overlay addresses

- Good, because the only public cluster port is udp/51820; 6443 and 10250 are never public. On attack
  surface alone this was the research recommendation.
- Good, because Lambda's NAT does not matter: the spoke dials the hub's static IP.
- Bad, because it is a community pattern, not a documented k3s mode, and every node gets an extra
  interface that k3s depends on (systemd `Requires=`/`After=` on `wg-quick@wg0`).
- Bad, because pod traffic is double-encapsulated (VXLAN inside WireGuard, pod MTU 1330), and in hub
  and spoke CPU-to-GPU pod traffic crosses the control plane; a full mesh instead means managing
  N×(N-1) peer entries by hand.
- Bad, because a broken `wg0` or a forgotten `flannel-iface` fails in ways that look like k3s problems.

### C. Tailscale

- Good, because no inbound port is needed and nodes are identified by key, not by IP.
- Good, because NAT traversal and relays are built in.
- Bad, because the k3s integration (`--vpn-auth`) is experimental and puts the auth key in k3s's
  configuration (and briefly in a child process's argv); the manual mode avoids that, but on k3s
  `v1.36.5` a tailscaled restart removes flannel's VXLAN device and restarts k3s.
- Bad, because the coordination service joins the trust base: without Tailnet Lock it can add a node
  that then reaches the unauthenticated VXLAN pod network.
- Bad, because the tailnet policy, tags and auth keys are another system to build and secure. It
  outsources the networking problem the exercise is about.

### D. kubeadm with Cilium or Calico WireGuard

- Good, because Cilium brings eBPF NetworkPolicy and observability.
- Bad, because neither CNI traverses NAT, so it still needs B's underlay.
- Bad, because Cilium's node-to-node encryption is beta and disabled to and from control-plane nodes,
  and kubeadm's API server to kubelet path is unverified by default.
- Bad, because it is the most work and the largest surface for a cluster of a few nodes.

## More Information

- Cloud firewall rules: [infra/base/firewall.tf](../../infra/base/firewall.tf),
  [infra/workers/firewall.tf](../../infra/workers/firewall.tf), [infra/gpu-access/main.tf](../../infra/gpu-access/main.tf);
  the Lambda rules are built in [lambda/gpuctl/peers.py](../../lambda/gpuctl/peers.py).
- Host firewall: [node/guard/](../../node/guard/).
