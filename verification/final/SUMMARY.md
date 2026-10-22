# Verification run

Started: 2026-10-22T12:50:46-07:00 (America/Los_Angeles)

Operator IP: masked as <operator_ip> (checkip.amazonaws.com)

GCP project id: masked as <GCP_PROJECT>

| # | ID | Check | Result |
|---|---|---|---|
| 1 | A1 | Nodes Ready (at least 2) | PASS |
| 2 | A2 | kubectl describe every node | PASS |
| 3 | A3 | metrics-server and CoreDNS Ready on k3s-cp | PASS |
| 4 | A4 | flannel annotations: public-ip is the ExternalIP | PASS |
| 5 | B1 | N x N connectivity matrix (curl, MTU ping, Service DNS) | PASS |
| 6 | B2 | On the wire at k3s-cp: encrypted, WireGuard, expected ports only | PASS |
| 7 | B3 | WireGuard peers and recent handshakes | PASS |
| 8 | B4 | iperf3 across nodes | PASS |
| 9 | B5 | Negative tests: API ports, metadata server, DNS, guard counters | PASS |
| 10 | B6 | Firewall inventory: every source a /32 (or IAP) | PASS |
| 11 | C1 | hello rollout | PASS |
| 12 | C2 | hello world from another node | PASS |
| 13 | C3 | hello world through kubectl port-forward | PASS |
| 14 | C4 | QoS class Guaranteed | PASS |
| 15 | C5 | cgroup memory.max and cpu.max | PASS |
| 16 | C6 | kubectl top pod | PASS |
| 17 | C7 | limits-demo: OOMKilled and CPU throttling | PASS |
| 18 | C8 | NetworkPolicy: allowed from nettest, denied elsewhere | PASS |
| 19 | D1 | GPU stack: audit, containerd, RuntimeClass | SKIP |
| 20 | D2 | GPU capacity and allocatable | SKIP |
| 21 | D3 | Device plugin registered | SKIP |
| 22 | D4 | gpu-smoke: nvidia-smi | SKIP |
| 23 | D5 | vectoradd: Test PASSED | SKIP |
| 24 | D6 | gpu-overflow stays Pending | SKIP |

**Result:** 18 passed, 0 failed, 6 skipped

**Secret scan:** gitleaks found no leaks

**Promoted with skipped checks:** 6 skipped check(s) accepted with PROMOTE_ALLOW_SKIP=1 by scripts/promote.sh at 2026-10-22T12:55:14-0700
