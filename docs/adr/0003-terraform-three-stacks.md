---
status: accepted
date: 2026-10-22
---

# Terraform in three stacks (plus bootstrap); Lambda resources owned by Python

## Context and Problem Statement

The GCP side has resources with very different lifecycles and very different tolerance for
unattended change:

- the state bucket and the enabled APIs, created once per project;
- the networks, the service accounts and `k3s-cp`, which hold the cluster's identity (CA, datastore,
  server token) and must never be recreated by accident;
- the CPU workers, scaled by hand from an inventory;
- the firewall rules that admit the GPU nodes, which change every time a Lambda instance is launched
  or terminated and are applied by `gpuctl` without a human reading the plan.

Lambda Cloud has instances, a global firewall rule list and per-region firewall rulesets, but no
official Terraform provider: the community provider (`elct9620/lambdalabs` v0.8.2) cannot set an
instance's image, `user_data` or firewall rulesets, and its firewall data source is read-only. Lambda
also has no stop state and no static IPs. How should infrastructure as code be cut so that scaling
and GPU churn can never touch the control plane, and every resource has exactly one owner?

## Decision Drivers

- An unattended apply (by `gpuctl`, or a future autoscaler) must not be able to reach `k3s-cp`.
- A plan a human reviews should be small enough to read.
- GCP opens an ingress rule with no source to everyone; that must be impossible by construction.
- One owner per resource, so no tool "fixes" what another one manages.
- Destroy must work in a known order and keep the state bucket.

## Considered Options

- One Terraform stack for all of GCP
- Three stacks (`base`, `workers`, `gpu-access`) plus a one-time `bootstrap` stack; Lambda in Python
- Terraform for Lambda as well, through the community provider
- No Terraform: `gcloud` and shell scripts

## Decision Outcome

Chosen option: **three stacks plus bootstrap, with Lambda owned by `gpuctl`**, because the split
puts the control plane structurally out of reach of every automated apply, keeps each reviewed plan
small, and lets the GPU firewall follow Lambda's IPs without a human in the loop.

| Stack | Holds | Applied by | Unattended (`AUTO_APPROVE=1` / `-auto-approve`) |
|---|---|---|---|
| [infra/bootstrap](../../infra/bootstrap/) (local state) | GCS state bucket `<GCP_PROJECT>-k3s-tfstate` (versioned, uniform access, public access prevented); six APIs, never disabled on destroy | `make init`, once | not used |
| [infra/base](../../infra/base/) | VPCs `k3s-core` and `k3s-edge` with their subnets, `k3s-cp`, its static IP, SAs `k3s-cp` and `k3s-worker` with no IAM bindings, the fixed rules `fw-cp-allow-iap-ssh` and `fw-cp-allow-self` | `make up` (base stage) | applies only plans that destroy or replace nothing (`tf_apply --no-destroy`); a replacement of `k3s-cp` must be confirmed interactively |
| [infra/workers](../../infra/workers/) | `var.cpu_workers` from `workers.auto.tfvars.json`: per worker a static IP, the VM, `fw-<name>-to-cp`, `fw-<name>-mesh`; shared `fw-cp-to-workers`, `fw-workers-allow-iap-ssh` | `make up` (workers stage), `make cpu-scale`, `make cpu-remove` | add-only plans; a scale-down may destroy only the workers it drained (`--destroy-only NAME`), never `k3s-cpu-1` |
| [infra/gpu-access](../../infra/gpu-access/) | `var.gpu_ips`: per GPU IP `fw-gpu-<ip>-to-cp` (6443, 51820/udp, ICMP) and `fw-gpu-<ip>-to-workers` (51820/udp, ICMP) | `gpuctl` on every launch and terminate | always: it holds nothing else, so that is the whole blast radius |

The workers and gpu-access stacks read the base stack's outputs (`cp_ip`, network names, SA emails)
through `terraform_remote_state`; nothing reads them the other way. Each stack's state is a prefix
of the bucket, with GCS's native locking and object versioning.

Lambda instances, the global rule list and the `k3s-gpu-<region>` rulesets belong to `gpuctl`
([lambda/gpuctl/](../../lambda/gpuctl/), [ADR-0004](0004-gpu-nodes-as-cattle-spec-reconciler.md)).
Node OS settings belong to [node/](../../node/), cluster objects to [k8s/](../../k8s/).

### Guardrails in the stacks

1. **No rule without a source.** GCP documents that an ingress rule without `source_ranges` admits
   everyone. Every dynamic rule is a `for_each` over the inventory, so an empty list yields no rule
   at all. `infra/gpu-access/tests` pins this both ways (`empty_list_creates_no_rules`,
   `back_to_empty_removes_all_rules`): when the last GPU node goes, every GPU rule goes with it.
2. **Single addresses only.** Each `gpu_ips` entry must be one IPv4 address with octets 0-255 and no
   leading zeros; CIDRs, private, CGNAT and loopback addresses are rejected, and the list may not be
   longer than `max_nodes`. The rules add the `/32` themselves.
3. **No open-internet literal anywhere.** A repository test fails if the any-IPv4 CIDR appears in any
   tracked file ([tests/test_terraform_static.py](../../tests/test_terraform_static.py)), and
   verification check B6 lists every live rule and requires each source to be a `/32` or IAP's
   `35.235.240.0/20`.
4. Every rule has `log_config` and targets a service account, not a network tag.
5. Shielded VM (secure boot, vTPM, integrity monitoring) set explicitly, because secure boot defaults
   to off in the provider; OS Login with `block-project-ssh-keys`; `can_ip_forward = false`.
6. `ignore_changes` on the boot image, so a new image in the Ubuntu family never rebuilds a VM.
7. Provider pinned (`google = 8.7.0`, Terraform `>= 1.9.0`) with committed lock files used
   `-lockfile=readonly`; the `tf-test` target validates and tests every stack offline.

### Consequences

- Good, because the stack `gpuctl` applies with `-auto-approve` contains only per-GPU-IP rules. The
  worst a bug there can do is add or drop rules for GPU IPs; it cannot destroy or reconfigure a VM.
- Good, because a human-reviewed plan is small: a CPU scale-up from 1 to 2 workers is the four
  resources of the new worker.
- Good, because `make down` destroys `gpu-access`, `workers`, `base` in that order (each reads
  `base`) and keeps the bootstrap stack, so the next `make up` needs no `make init`.
- Bad, because there are four state prefixes and four `terraform init`s; scripts initialise lazily
  (`tf_init` with a success marker) to keep re-runs fast.
- Bad, because `max_nodes` exists twice: in [lambda/gpu-nodes.yaml](../../lambda/gpu-nodes.yaml)
  (the source of truth) and as the default of `var.max_nodes`. `gpuctl` passes the spec's value on
  every apply, and a static test fails while the two defaults differ.
- Bad, because the bootstrap stack's state is local to the operator's checkout (the bucket cannot
  hold the state that creates it). Losing it means importing two kinds of resources before the next
  change to that stack; nothing else depends on that file.
- Neutral: a rule name such as `fw-gpu-203-0-113-7-to-cp` embeds the IP, so a new Lambda IP is a new
  rule, never an in-place edit of an old one.

### Confirmation

- Offline: `make tf-test` runs `terraform validate` and `terraform test` for all four stacks with a
  mock provider; `make lint` includes it.
- Measured (2026-10-21/22): `make init` about 1m15s the first time, 7.5 s on a re-run; the base stack
  (10 resources) about 1 min; the workers stack 6 resources for one worker; a second `make up` with
  nothing to change took 92 s and changed nothing (no Terraform change, no k3s restart, guard rules
  unchanged). Check E3 adds a gpu-access rule for an address with no instance; `gpuctl status`
  reports it as drift and `gpuctl apply` removes it.

## Pros and Cons of the Options

### One stack

- Good, because there is one state and one plan.
- Bad, because every GPU launch would re-plan `k3s-cp`, and an `-auto-approve` for the GPU rules
  would also approve any drift on the control plane.
- Bad, because a reviewed plan mixes unrelated changes.

### Three stacks plus bootstrap (chosen)

- Good, because the blast radius of each apply matches who approves it.
- Bad, because of cross-stack outputs and an ordered destroy.

### Terraform for Lambda too

- Good, because one tool would describe everything.
- Bad, because the community provider cannot set the image family or attach the per-region firewall
  rulesets at launch, which the design depends on, and Terraform state would duplicate what Lambda's API already knows (an instance
  that dies or is terminated in the console leaves stale state).
- Bad, because Lambda has no stop state and no static IP, so every "apply" is a create or a destroy;
  Terraform adds nothing over a small reconciler.

### gcloud and scripts

- Good, because there is nothing to learn.
- Bad, because there is no plan to review, no drift detection and no declarative "empty list means no
  rule".

## More Information

- Plan and apply helpers with the `--no-destroy` and `--destroy-only` guards: `tf_plan` and
  `tf_apply` in [scripts/lib/common.sh](../../scripts/lib/common.sh).
- Teardown order: [scripts/down.sh](../../scripts/down.sh).
