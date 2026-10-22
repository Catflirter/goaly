# Guide: Hybrid GPU Kubernetes Cluster

A short walkthrough of the problem, the design and its trade-offs, and how to verify the running
cluster with plain `kubectl`. The full architecture brief is [README.md](README.md). The decision
records are in [docs/adr/](docs/adr/), and the operator guide is [docs/runbook.md](docs/runbook.md).

## 1. Understanding the problem

**Core problem:** run one Kubernetes cluster across clouds, with CPU compute on GCP and GPU compute
on Lambda Cloud.

### Key considerations

**Pod connectivity across networks**

- GCP and Lambda both use private `10.x` address space, including `10.19.x` ranges, so the private
  networks cannot simply be routed together.
- Three kinds of nodes sit in three different networks and must be bridged:
  - the control plane, in a GCP VPC;
  - the CPU workers, in a second GCP VPC;
  - the GPU nodes, on Lambda.

**Security**

Security is built on cloud firewalls plus host and network policies.

- **Addresses change.** CPU and GPU instances get new IP addresses whenever they are recreated.
  Every allow-list must then be updated: GCP firewall rules, Lambda firewall rules, and the
  control plane's host firewall.
- **Lambda hosts are not hardened by default.** Lambda instances come with a broad default attack
  surface (for example Jupyter, open SSH, IP forwarding), and the provider has root on them.
- **Defence in depth.** No single layer is trusted on its own; several defences must be in place
  at once.

**Limited resources**

- Every step must be safe to run more than once: at least once, and idempotent.
- GPUs are expensive and are frequently sold out, so GPU nodes are created on demand, use
  fallback types and regions, and are torn down right after use.

## 2. Design and trade-offs

### High-level design

A k3s cluster with:
- the control plane on GCP;
- the CPU worker nodes in a separate VPC on GCP;
- the GPU nodes on Lambda Cloud.

How it is operated:
- **Make** on the operator's workstation drives every operation (`make up`, `make gpu-apply`,
  `make verify`, `make down`, ...).
- **Terraform** manages everything on GCP. **gpuctl** (Python) manages Lambda: instances,
  firewall rules and the GPU part of the peer allow-lists.
- **Shell scripts** bootstrap the nodes and join them to the cluster.

### Networking

**Option 1 (taken): k3s / flannel `wireguard-native`**

Every node, the k3s API server included, declares its public IP with `--node-external-ip`. Flannel
builds a WireGuard mesh between those public endpoints.

- Pros:
  - easy to set up;
  - well supported upstream.
- Cons:
  - works through NAT only because each node declares its public address;
  - the WireGuard port and the k3s API are reachable from the public internet, limited by IP
    allow-lists;
  - an embedded etcd multi-server control plane is not supported in this setup.

**Option 2: self-managed WireGuard**

- Pros:
  - fully customisable;
  - minimal public exposure;
  - not sensitive to NAT.
- Cons:
  - hard to maintain;
  - the hub becomes a single point of traffic and of failure.

**Option 3: Tailscale**

- Pros:
  - easy to use;
  - access is defined by ACLs, with no network configuration needed.
- Cons:
  - a third party (Tailscale) has to be trusted;
  - traffic between nodes can be slow when it is relayed.

### Security

**Threat model: assets**

- the Kubernetes API;
- the kubelet API;
- Pod traffic;
- the credentials (token and certificates) used to join the cluster;
- the GPU instances.

**Threat model: attack surfaces**

- WireGuard, UDP 51820;
- the Kubernetes API, TCP 6443;
- the kubelet, TCP 10250;
- SSH to Lambda instances, TCP 22.

**Defences on GCP**

- **Custom-mode VPCs.** No default firewall rules exist. Every rule is managed by Terraform.
- **Host firewall on the control plane (`k3s-guard`).** An nftables guard on k3s-cp admits the
  k3s ports only from the known peer addresses.

**Defences on Lambda**

- **gpuctl manages the firewall rules.** They allow only:
  - the operator on TCP 22;
  - the control plane on TCP 10250;
  - the cluster nodes on UDP 51820, plus ICMP.
- **Two layers of rules.** Global rules carry only the operator's SSH. Per-region rulesets carry
  the cluster rules and are attached at launch.

## 3. Verification with plain `kubectl`

Each command below sets `KUBECONFIG`, so it runs on its own. The only dependency is the IAP API
tunnel (`make tunnel`), which is the cluster's single access path: kubeconfig →
`127.0.0.1:6443`.

### 3.1 Control plane on GCP, CPU workers in a separate VPC, GPU nodes on Lambda

```bash
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && kubectl get nodes -o wide
```

Expect:
- k3s-cp with InternalIP `10.10.0.x` (core VPC);
- k3s-cpu-1 and k3s-cpu-2 with InternalIP `10.20.0.x` (edge VPC, no peering with core);
- GPU nodes with Lambda private IPs;
- a `-gcp` kernel on the GCP nodes.

```bash
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && kubectl describe node k3s-cp
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && kubectl describe node k3s-cpu-1
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && kubectl describe node k3s-gpu-a6000-1
```

Check:
- the control-plane role label on k3s-cp;
- InternalIP and ExternalIP under Addresses;
- on GPU nodes:
  - `nvidia.com/gpu: 1` in Capacity and Allocatable;
  - the label `node-restriction.kubernetes.io/accelerator=nvidia`;
  - the taint `nvidia.com/gpu=present:NoSchedule`.

### 3.2 Nodes connected over WireGuard

```bash
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && kubectl get nodes -o custom-columns='NAME:.metadata.name,BACKEND:.metadata.annotations.flannel\.alpha\.coreos\.com/backend-type,PUBLIC-IP:.metadata.annotations.flannel\.alpha\.coreos\.com/public-ip'
```

Expect `BACKEND=wireguard` on every node. PUBLIC-IP is each node's tunnel endpoint.

### 3.3 NVIDIA device plugin and an `nvidia-smi` Pod

```bash
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && kubectl -n nvidia-device-plugin get ds,pods -o wide
```

Expect device-plugin Pods only on the GPU nodes.

```bash
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && kubectl -n gpu-test delete pod gpu-smoke --ignore-not-found && kubectl apply -f k8s/gpu-test/gpu-smoke.yaml && kubectl -n gpu-test wait --for=jsonpath='{.status.phase}'=Succeeded pod/gpu-smoke --timeout=180s && kubectl -n gpu-test logs gpu-smoke && kubectl -n gpu-test get pod gpu-smoke -o wide
```

`k8s/gpu-test/gpu-smoke.yaml` is a plain Pod:
- one `nvidia.com/gpu`;
- `runtimeClassName: nvidia-gpu`;
- the command `nvidia-smi`.

Expect the `nvidia-smi` table for the GPU, with the Pod placed on a GPU node.

### 3.4 The two CPU nodes communicate

From a Pod on k3s-cpu-1 to a Pod IP on k3s-cpu-2:

```bash
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && P=$(kubectl -n nettest get pod -l app=probe --field-selector spec.nodeName=k3s-cpu-1 -o name | head -1) && E=$(kubectl -n nettest get pod -l app=echo --field-selector spec.nodeName=k3s-cpu-2 -o jsonpath='{.items[0].status.podIP}') && echo "$P -> $E" && kubectl -n nettest exec $P -- curl -sS -m 5 http://$E:8080/
```

Expect an echo response. Replace `k3s-cpu-2` with `k3s-gpu-a6000-1` to show that GCP ↔ Lambda
traffic works as well.

### 3.5 hello-world Deployment with requests and limits

```bash
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && kubectl -n hello get deploy,pods,svc -o wide
```

```bash
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && kubectl -n hello get deploy hello -o jsonpath='{.spec.template.spec.containers[0].resources}{"\n"}' && kubectl -n hello get pods -o custom-columns='NAME:.metadata.name,NODE:.spec.nodeName,QOS:.status.qosClass'
```

Expect requests and limits of `cpu: 100m, memory: 64Mi`, which gives the QoS class `Guaranteed`.

```bash
export KUBECONFIG=$HOME/.kube/hybrid-gpu.yaml && P=$(kubectl -n nettest get pod -l app=probe --field-selector spec.nodeName=k3s-cpu-2 -o name | head -1) && kubectl -n nettest exec $P -- curl -sS -m 5 http://hello.hello.svc.cluster.local
```

Expect `hello world`. The request must come from the `nettest` namespace, because a NetworkPolicy
allows ingress to hello only from there.

### 3.6 Architecture brief and verification logs

- Architecture brief: [README.md](README.md) and [docs/](docs/).
- Verification logs: [verification/final/SUMMARY.md](verification/final/SUMMARY.md), with one
  evidence file per check next to it.
