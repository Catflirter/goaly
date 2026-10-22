SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

# Absolute repo root, so targets also work as `make -f <repo>/Makefile` from another directory.
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
GITLEAKS := gitleaks git --config $(ROOT)/.gitleaks.toml --redact --no-banner --exit-code 1

.PHONY: help venv test lint secrets

help: ## List available targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-16s %s\n", $$1, $$2}'

venv: ## Create .venv (Python >= 3.11) and install dev requirements
	@$(PYTHON) -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
	  || { echo "Python >= 3.11 required ($(PYTHON) is $$($(PYTHON) -V 2>&1)); set PYTHON=..." >&2; exit 1; }
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

test: ## Run the pytest suite
	$(PY) -m pytest

# The upstream gitleaks pre-commit hook only scans staged changes, even under --all-files,
# so lint scans the repo's tracked content explicitly: every commit on HEAD's history,
# then staged and unstaged changes to tracked files. Untracked/ignored files (.env) are not read.
secrets: ## Scan committed history and uncommitted tracked changes for secrets
	$(GITLEAKS) --log-opts=HEAD .
	$(GITLEAKS) --pre-commit --staged .
	$(GITLEAKS) --pre-commit .

# Offline only: -backend=false and the tests' mock provider need no cloud credentials.
# -lockfile=readonly: init verifies providers against the committed .terraform.lock.hcl and
# never rewrites it. To change providers, run tf-lock deliberately and commit the result.
# TF_DATA_DIR keeps this offline init in .run/tf-test/<stack>, never in infra/<stack>/.terraform: a
# -backend=false init there leaves a .terraform with providers but no backend, which the real
# (GCS) init used by gpuctl and scripts/up.sh would otherwise have to undo.
.PHONY: tf-test tf-lock
tf-test: ## terraform init + validate every infra/ stack, then terraform test where it has tests/
	@for d in $(ROOT)/infra/*/; do \
	  d=$${d%/}; echo "==> $$d"; export TF_DATA_DIR="$(ROOT)/.run/tf-test/$${d##*/}"; \
	  terraform -chdir="$$d" init -backend=false -input=false -lockfile=readonly; \
	  terraform -chdir="$$d" validate; \
	  if [ -d "$$d/tests" ]; then terraform -chdir="$$d" test; fi; \
	done

# Cloud entry points (spec §8, §11). Every terraform apply shows its plan and asks first;
# AUTO_APPROVE=1 skips the question, but `make up` never lets it destroy or replace anything in
# the base stack (k3s-cp) or the workers stack.
.PHONY: init up kubeconfig tunnel tunnel-stop
init: ## Create the Terraform state bucket and enable the GCP APIs (infra/bootstrap; one-time, billable)
	$(ROOT)/scripts/up.sh init

up: ## Bring the cluster up: base, k3s-cp, kubeconfig, tunnel, CPU workers, system components, workloads (safe to re-run)
	$(ROOT)/scripts/up.sh all

# Ordered teardown (spec §11): asks once for the GCP project id; the bootstrap stack (state bucket)
# is kept. Only AUTO_APPROVE=1 skips the question (unlike gpu-apply/gpu-down, yes and true do not),
# and only on the command line or in the environment: a .env that sets it refuses the run. Declining
# changes nothing and exits 2, so make reports "Error 2". Ends by asking Lambda that no GPU node is
# left (DOWN_LAMBDA_WAIT_S, default 60 s, for a node just terminated). Safe to re-run after a partial
# teardown.
.PHONY: down
down: ## Tear everything down: GPU nodes, CPU workers, gpu-access/workers/base stacks, tunnel (keeps bootstrap)
	$(ROOT)/scripts/down.sh

# The CPU worker pool (spec §5.3). Removing a worker drains it and deletes it from the cluster
# before the apply destroys its VM; k3s-cpu-1 is never removed.
.PHONY: cpu-scale cpu-remove cpu-stop cpu-start
cpu-scale: ## Scale the CPU pool to N workers (make cpu-scale N=3): drain removed ones, plan, confirm, apply, join new ones
	$(ROOT)/scripts/cpu-pool.sh scale "$(N)"

cpu-remove: ## Drain and delete CPU worker NAME, then destroy its VM (make cpu-remove NAME=k3s-cpu-2)
	$(ROOT)/scripts/cpu-pool.sh remove "$(NAME)"

cpu-stop: ## Stop every CPU worker VM (nodes go NotReady, static IPs are kept)
	$(ROOT)/scripts/cpu-pool.sh stop

cpu-start: ## Start every CPU worker VM again
	$(ROOT)/scripts/cpu-pool.sh start

kubeconfig: ## Fetch the kubeconfig over IAP SSH into ~/.kube/hybrid-gpu.yaml (0600)
	$(ROOT)/scripts/kubeconfig.sh

tunnel: ## Start the IAP SSH tunnel to the API server at https://127.0.0.1:6443
	$(ROOT)/scripts/tunnel.sh start

tunnel-stop: ## Stop the API tunnel
	$(ROOT)/scripts/tunnel.sh stop

TF_LOCK_PLATFORMS := linux_amd64 linux_arm64 darwin_amd64 darwin_arm64
tf-lock: ## Regenerate every infra/ stack's .terraform.lock.hcl with hashes for all supported platforms
	@for d in $(ROOT)/infra/*/; do \
	  d=$${d%/}; echo "==> $$d"; \
	  terraform -chdir="$$d" providers lock $(addprefix -platform=,$(TF_LOCK_PLATFORMS)); \
	done

lint: ## Run all pre-commit hooks, the repo secret scan, the Terraform tests, then the tests
	$(PY) -m pre_commit run --all-files
	$(MAKE) secrets
	$(MAKE) tf-test
	$(MAKE) test

# RESILIENCE=1 (make verify RESILIENCE=1) reaches verify.sh through the environment and adds group E.
.PHONY: verify
verify: ## Run verification checks A-D (RESILIENCE=1 adds E) into verification/<timestamp>/
	$(ROOT)/scripts/verify.sh

# RUN is a path to the run, as the shell sees it from where make runs (make -C <repo> for a repo-relative one).
# PROMOTE_ALLOW_SKIP=1 (a run with skipped, never failed, checks), OPERATOR_IP=<ip> and GCP_PROJECT=<id> (default: .env)
# reach promote.sh through the environment.
.PHONY: promote-verify
promote-verify: ## Copy a verify run to verification/final/ if gitleaks finds no secrets (make promote-verify RUN=verification/<ts>)
	@[ -n "$(RUN)" ] || { echo "usage: make promote-verify RUN=verification/<YYYY-MM-DDTHHMM-0700>" >&2; exit 2; }
	$(ROOT)/scripts/promote.sh verify "$(RUN)"

# AUTO_APPROVE=1 (or yes/true) skips the confirmation prompt of gpu-apply and gpu-down.
GPU_AUTO_APPROVE := $(if $(filter 1 yes true,$(AUTO_APPROVE)), --auto-approve)

.PHONY: gpu-plan gpu-apply gpu-down gpu-status
gpu-plan: ## Show what gpu-apply would create/delete on Lambda and the hourly cost (read-only)
	$(ROOT)/scripts/gpuctl plan

gpu-apply: ## Create/delete Lambda GPU nodes to match lambda/gpu-nodes.yaml, then firewall and join (AUTO_APPROVE=1)
	$(ROOT)/scripts/gpuctl apply$(GPU_AUTO_APPROVE)

gpu-down: ## Delete every Lambda GPU node and the autoscaler overrides (AUTO_APPROVE=1 skips the prompt)
	$(ROOT)/scripts/gpuctl down$(GPU_AUTO_APPROVE)

gpu-status: ## Show GPU capacity, GPU instances and drift (changes no cloud state; may terraform init locally; exit 3 on drift)
	$(ROOT)/scripts/gpuctl status

.PHONY: sync-peers
sync-peers: ## Rewrite the peer allow-lists (cp k3s_peers set, Lambda firewall) from the current node IPs
	$(ROOT)/scripts/gpuctl sync-peers

# Stretch S0 (spec §10.0): the monitoring stack in k8s/monitoring/. Grafana and Prometheus have no
# exposed Service: they are reached only through kubectl port-forward on 127.0.0.1 (via make tunnel).
.PHONY: monitoring monitoring-down grafana grafana-stop prometheus prometheus-stop
monitoring: ## Install or upgrade the monitoring stack (expects k3s-cpu-1 as e2-standard-2; warns otherwise); a new Grafana password is shown once
	$(ROOT)/scripts/monitoring.sh install

monitoring-down: ## Delete the monitoring stack and its data (asks first; AUTO_APPROVE=1 skips)
	$(ROOT)/scripts/monitoring.sh uninstall

grafana: ## Port-forward Grafana to http://127.0.0.1:3000 in the background (GRAFANA_PORT=... to change)
	$(ROOT)/scripts/monitoring.sh forward grafana

grafana-stop: ## Stop the Grafana port-forward
	$(ROOT)/scripts/monitoring.sh stop grafana

prometheus: ## Port-forward Prometheus to http://127.0.0.1:9090 in the background (PROMETHEUS_PORT=...)
	$(ROOT)/scripts/monitoring.sh forward prometheus

prometheus-stop: ## Stop the Prometheus port-forward
	$(ROOT)/scripts/monitoring.sh stop prometheus
