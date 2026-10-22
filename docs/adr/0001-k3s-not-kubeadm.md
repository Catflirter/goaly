---
status: accepted
date: 2026-10-22
---

# Use k3s, not kubeadm or RKE2

## Context and Problem Statement

We need a lightweight Kubernetes control plane on one small GCP VM, and agents on two kinds of
remote machines: GCP CPU VMs in a second VPC and Lambda Cloud GPU instances. There is no private
network between the providers, so the nodes join over public IPs, and both clouds NAT those IPs
(measured: GCP's external IP is a 1:1 NAT, and a Lambda instance's NIC carries only a private
address). The GPU nodes run Lambda's image, which ships the driver, the NVIDIA Container Toolkit and
Docker. The work is a one-to-two-day build graded on architecture, trade-offs and security. Which
Kubernetes distribution gives a secure cross-cloud cluster with the fewest parts we build ourselves?

## Decision Drivers

- A documented way to run nodes that reach each other only over public, NATed IPs.
- Encryption of cross-node pod traffic without a separate VPN layer.
- Safe defaults for the API server to kubelet path, which crosses the internet in this topology.
- GPU support that uses the driver and toolkit the Lambda image already has.
- A small footprint: the control plane is an e2-medium (2 vCPU, 4 GB).
- Few moving parts to install, upgrade and explain.

## Considered Options

- k3s `v1.36.5+k3s1` (stable channel)
- kubeadm with a separately installed CNI and container runtime
- RKE2
- A managed control plane (GKE) with external nodes

## Decision Outcome

Chosen option: **k3s**, because it is the only option with a documented cross-cloud mode
(`--node-external-ip`, `--flannel-backend=wireguard-native`, `--flannel-external-ip`) that encrypts
pod traffic with WireGuard and needs no extra component, because its defaults keep the API server to
kubelet path inside an authenticated tunnel, and because it registers the NVIDIA runtime by itself.

What k3s provides that this design relies on:

- **One binary** with containerd, flannel, CoreDNS, the kube-router NetworkPolicy controller and
  local-path storage. One version pin (`v1.36.5+k3s1`) pins all of them. `install.sh` is fetched
  from the release tag URL, never from `get.k3s.io`, so the installer is pinned too
  ([node/common.sh](../../node/common.sh)).
- **Multicloud mode**, documented and not marked experimental. Agents dial the server's public IP on
  6443; pod traffic uses flannel's `wireguard-native` backend on udp/51820 with a 25 s keepalive.
- **The supervisor tunnel.** With the default `--egress-selector-mode=agent`, API server to kubelet
  traffic (`logs`, `exec`, `port-forward`) rides a websocket the agent opened to 6443.
  metrics-server, on `k3s-cp`, scrapes the kubelets on 10250 directly ([ADR-0002](0002-network-k3s-wireguard-native.md)).
- **NVIDIA runtime detection.** At start, k3s looks for `nvidia-container-runtime` on its PATH and
  registers a containerd handler `nvidia`. Measured on the Lambda image: the binary is
  `/usr/bin/nvidia-container-runtime` (toolkit 1.18.1), so no host runtime configuration is needed
  ([ADR-0005](0005-gpu-device-plugin-and-runtimeclass.md)).
- **CA-pinned join tokens.** `k3s token create --ttl 15m` issues a `K10<CA hash>::…` token, so a
  joining agent verifies the server's CA.
- **Hardening switches** from the k3s CIS guide in one config file: `secrets-encryption`,
  `protect-kernel-defaults`, a Pod Security admission configuration and audit logging
  ([node/k3s/server-config.yaml.tmpl](../../node/k3s/server-config.yaml.tmpl)).

### Consequences

- Good, because the whole cluster network is three server settings and one agent setting, all
  documented upstream.
- Good, because the API server to kubelet path for `logs`/`exec` stays inside the agent's
  authenticated tunnel. With kubeadm that path is a direct connection to every kubelet over the
  internet, with the kubelet's serving certificate unverified by default.
- Good, because one version pin covers runtime, CNI and add-ons, and the server fits an e2-medium.
- Bad, because multicloud mode does not support embedded etcd, so the cluster has one server. High
  availability needs an external datastore or a different network design ([P2 roadmap](../p2-roadmap.md)).
- Bad, because flannel options are server-only: setting `flannel-backend` on an agent left workers
  on VXLAN and NotReady (k3s issue 12942). The agent template carries no flannel setting
  ([node/k3s/agent-config.yaml.tmpl](../../node/k3s/agent-config.yaml.tmpl)).
- Bad, because containerd comes with the k3s release. `v1.36.5+k3s1` bundles containerd 2.3.4,
  which is likely affected by two medium-severity image-pull DoS advisories published 2026-10-13
  (GHSA-vwhp-8p66-mq4f, GHSA-8647-2jxj-p2rc; fixed upstream in 2.3.7). We wait for a k3s release
  rather than replace the runtime ([threat model](../threat-model.md)).
- Bad, because k3s creates its own RuntimeClass `nvidia` without a node selector. We leave it and
  add a scoped one ([ADR-0005](0005-gpu-device-plugin-and-runtimeclass.md)).
- Neutral: the hostname of GCE's Ubuntu 24.04 image is an FQDN (`k3s-cp.<zone>.c.<GCP_PROJECT>.internal`).
  Without `node-name`, the first live run registered the server under that name and `make up` waited
  for `node/k3s-cp` in vain. The server config now pins `node-name: k3s-cp`; agents always pinned it.

### Confirmation

Verification group A records `kubectl get nodes -o wide`, `describe node` for every node, that
metrics-server and CoreDNS run on `k3s-cp`, and each node's flannel annotations. On 2026-10-22 a
from-scratch `make up` followed by `make verify RESILIENCE=1` passed 27 of 27 checks, including the
reboot of a CPU worker (E1) and the replacement of the GPU node (E2).

## Pros and Cons of the Options

### k3s

- Good, because the cross-cloud recipe is documented and has an upstream end-to-end test with the
  same flags (on private networks with IPs bound to the NICs, so NAT on both sides was ours to
  verify; it worked, see [ADR-0002](0002-network-k3s-wireguard-native.md)).
- Good, because the NVIDIA handler is registered only where the binary exists, so CPU nodes never
  get it.
- Bad, because it is opinionated: bundled add-ons must be disabled explicitly (we disable Traefik,
  ServiceLB and the bundled metrics-server), and some upstream settings are reached only through
  `kube-apiserver-arg`.

### kubeadm

- Good, because it is plain upstream Kubernetes and every component can be chosen.
- Bad, because the API server dials each kubelet on 10250 and does not verify its serving
  certificate by default; upstream calls this unsafe over untrusted or public networks. Fixing it
  needs `serverTLSBootstrap` and approval of every kubelet certificate request.
- Bad, because the CNI is ours to install, and the CNIs that encrypt with WireGuard (Cilium,
  Calico) do not traverse NAT, so they would still need a hand-built WireGuard underlay
  ([ADR-0002](0002-network-k3s-wireguard-native.md), option D).
- Bad, because on Lambda the runtime would be Docker's containerd, which then needs CRI and NVIDIA
  configuration on the host (`nvidia-ctk runtime configure`).

### RKE2

- Good, because it is hardened toward the CIS benchmark by default and shares k3s's supervisor and
  token design.
- Bad, because it is heavier (embedded etcd by default) on a 4 GB control plane.
- Bad, because we found no documented recipe equivalent to k3s's multicloud mode; the cross-cloud
  network would be assembled from CNI options.
- Bad, because its documented NVIDIA path is the GPU Operator, which this design avoids
  ([ADR-0005](0005-gpu-device-plugin-and-runtimeclass.md)).

### Managed control plane (GKE) with external nodes

- Good, because Google would operate the control plane and its availability.
- Bad, because GKE does not accept self-managed nodes from another provider; that needs a fleet
  product and a different architecture.
- Bad, because the exercise asks for a control plane we provision on a VM.

## More Information

- Server and agent configuration: [node/k3s/](../../node/k3s/); bootstraps:
  [node/server-bootstrap.sh](../../node/server-bootstrap.sh), [node/node-bootstrap.sh](../../node/node-bootstrap.sh).
- The network decision built on this one: [ADR-0002](0002-network-k3s-wireguard-native.md).
