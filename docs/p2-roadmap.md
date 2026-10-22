# P2 roadmap

Work deliberately left out of this build, each with why it matters and how it would be done. The
order inside each section is the order we would do it in. Residual risks that motivate an item are
in the [threat model](threat-model.md).

Optional stretch, outside this list: the repository also carries an operator-side autoscaling demo
and a port-forward-only monitoring stack (`k8s/monitoring/`, `make monitoring`); they are documented
separately and change nothing in the core described here.

## Networking and scale

| Item | Why | How |
|---|---|---|
| Identity-based network (host WireGuard hub-spoke, or Tailscale with Tailnet Lock) | IP allow-lists put the operator's machine on the path of every node change and leave 6443/10250 reachable from peer addresses ([ADR-0006](adr/0006-ip-allowlist-vs-identity-network.md)) | B: `wg0` on every node with keys issued at join, k3s `node-ip` and `flannel-iface` on the overlay, cloud rules reduced to udp/51820. C: Tailscale with Tailnet Lock or a self-hosted coordination server, tags per node class, short-lived pre-auth keys from the join flow |
| A real node autoscaler | The demo autoscaler runs on the operator's machine and stops with it | After the identity network: cluster-autoscaler with GCE managed instance groups for the CPU pool (a scoped service account with `compute.instanceGroupManager` rights only) and an in-cluster controller for Lambda, with the same budget and `max_nodes` guards as `gpuctl`. Trade-off: that controller makes the cluster hold an account-wide Lambda key in a Secret, which today's rule "no long-lived credential in the cluster can add nodes or change firewalls" ([ADR-0006](adr/0006-ip-allowlist-vs-identity-network.md)) forbids; it therefore requires the dedicated Lambda account and key rotation (Security, below) first |
| An in-cluster autoscaler controller with scoped credentials | Keeps scaling running without the operator, without a cluster-wide credential | A Deployment on `k3s-cp` (tolerating `CriticalAddonsOnly`) using a Kubernetes ServiceAccount limited to reading Pods and Nodes and draining. "Scoped" applies to GCP only (a service account limited to the k3s-* resources): Lambda keys cannot be scoped, so the Lambda side carries the trade-off of the row above |
| GPU HPA on utilisation | Scale GPU workloads on what the GPU does, not on CPU | DCGM exporter, Prometheus and prometheus-adapter exposing `DCGM_FI_DEV_GPU_UTIL` as a custom metric for an `autoscaling/v2` HPA |
| Forward-chain filter on Lambda hosts (T-1) | With `ip_forward=1` a Lambda host routes between all interfaces and has no NIC-level anti-spoofing like GCE's `can_ip_forward=false`, so a host on the same L2 segment can reach any pod or ClusterIP Service through it, past both allow-list layers ([threat model](threat-model.md#residual-risks)) | Meanwhile, a default-deny NetworkPolicy in every namespace narrows it. The fix, in `k3s_guard`'s forward chain: drop forwarded packets when neither the input nor the output interface is `cni0` or `flannel-wg`. Rolled out with the guard reload on every node and covered by B1 |

## High availability

| Item | Why | How |
|---|---|---|
| Three control-plane servers | One server is a single point of failure for the API and the datastore | k3s's multicloud mode does not support embedded etcd, so either an external datastore (Cloud SQL for PostgreSQL through `--datastore-endpoint`, reached over private networking from the servers) or the identity network first, then embedded etcd across three servers in different zones |
| Backup and restore of the datastore and the server token | Losing `k3s-cp` today means rebuilding the cluster and re-joining every node | `k3s etcd-snapshot` (or SQL backups with an external datastore) to a versioned GCS bucket with a retention policy; the server token and the encryption config kept in Secret Manager; a restore drill in the runbook |

## Security

| Item | Why | How |
|---|---|---|
| Per-person kubectl credentials or OIDC | The kubeconfig is one shared cluster-admin credential | `kube-apiserver` OIDC flags against Google identities, RBAC roles per team; the admin kubeconfig kept for break-glass only |
| Egress control | Pods and nodes reach the internet freely | Default-deny egress NetworkPolicies per namespace with DNS allowed, plus an egress proxy for image pulls and apt |
| EventRateLimit admission | Limits API floods from a misbehaving client or node | `--enable-admission-plugins=EventRateLimit` with a config file in `node/k3s/` |
| kube-bench CIS baseline | An independent check of the hardening | Run the k3s CIS profile as a Job and keep the report with the verification evidence |
| Image signature verification | Digest pinning stops tag swaps but not a malicious first push | A ValidatingAdmissionPolicy or policy controller that requires signatures (Sigstore) for the namespaces we own |
| Restrict the `nvidia-gpu` RuntimeClass and use CDI | Any pod under the NVIDIA runtime whose image sets `NVIDIA_VISIBLE_DEVICES=all` sees all GPUs | Device plugin in CDI mode; a ValidatingAdmissionPolicy that allows `runtimeClassName: nvidia-gpu` (and k3s's `nvidia`) only in GPU namespaces |
| Lambda API key rotation and a dedicated account | The key is account-wide and long-lived | A Lambda account used only by this cluster; rotate the key on a schedule and after every operator change |
| Parse `.env` instead of sourcing it | Sourcing makes a writable `.env` code execution as the operator | A small `KEY=VALUE` parser in `scripts/lib/common.sh` that rejects anything else |

## GPU

| Item | Why | How |
|---|---|---|
| Time-slicing or MIG | One pod per GPU wastes small inference jobs | Device plugin `config.sharing.timeSlicing` for small GPUs; MIG profiles on A100-class nodes |
| GPU Feature Discovery | Schedule on GPU model, memory and driver instead of a hand-set `gpu-type` label | Enable GFD in the device plugin chart |
| DCGM exporter | GPU utilisation, memory, temperature and XID errors as metrics | DaemonSet on GPU nodes only, scraped by Prometheus |
| GPU Operator | Worth it for a fleet with mixed images and drivers | Operator with `driver.enabled=false` and `toolkit.enabled=false` while Lambda's image provides them |
| Lambda persistent filesystem for model caches | Every launch starts with an empty disk and pulls everything again | A Lambda filesystem in the node's region mounted on the GPU nodes, holding model weights and a registry mirror |
| `ttl_hours` in the spec | A forgotten GPU node costs about $31 a day | `gpuctl` records launch time in the instance name or a tag and `gpu-status` (or a scheduled `gpu-apply`) terminates nodes past their TTL |

## Operations

| Item | Why | How |
|---|---|---|
| Long-term metrics | Today's observability is metrics-server's live values, the audit log, firewall logs, guard counters and `gpuctl status`; nothing is kept as a time series | In-cluster Prometheus with remote write to a long-term store, dashboards reached by port-forward only; no new firewall rule or public port |
| Central log collection | Node and pod logs live and die with each node; the audit log is only on `k3s-cp` | A log shipper DaemonSet sending to a store we run (Loki) rather than Cloud Logging, which would need IAM roles and credentials on the Lambda hosts |
| Alerting, including a GPU runtime alert | Nobody is told about a NotReady node or a GPU left running | Alertmanager rules for node readiness, certificate expiry and GPU instance age, delivered to chat |
| GitHub Actions CI | `make lint` runs only on the operator's machine today | A workflow that runs `make lint` (pre-commit, gitleaks over history, `terraform test` with the mock provider, pytest) on every pull request |
| Public Ingress with TLS for hello world | The service is reachable only inside the cluster and through port-forward | An ingress controller on a CPU worker with a cert-manager certificate and a GCP firewall rule for 443 on that worker only. Trade-off: a public rule needs an any-source range, a deliberate exception to the `/32`-only rule that `tests/test_terraform_static.py` and check B6 enforce; it needs a named allow-list exception in both, or a front such as Cloud Armor or IAP that keeps the VM's own rules `/32` |
| GitOps (Flux or Argo CD) | `kubectl apply` from the operator's machine leaves no audit trail of who applied what | Flux reconciling `k8s/` from the repository, with the device plugin as a HelmRelease |
