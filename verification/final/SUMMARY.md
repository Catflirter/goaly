# Verification run

Started: 2026-10-22T13:20:22-07:00 (America/Los_Angeles)

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
| 19 | D1 | GPU stack: audit, containerd, RuntimeClass | PASS |
| 20 | D2 | GPU capacity and allocatable | PASS |
| 21 | D3 | Device plugin registered | PASS |
| 22 | D4 | gpu-smoke: nvidia-smi | PASS |
| 23 | D5 | vectoradd: Test PASSED | PASS |
| 24 | D6 | gpu-overflow stays Pending | PASS |

**Result:** 24 passed, 0 failed, 0 skipped

**Secret scan:** gitleaks found no leaks
