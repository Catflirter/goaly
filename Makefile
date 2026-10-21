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
.PHONY: tf-test tf-lock
tf-test: ## terraform init + validate every infra/ stack, then terraform test where it has tests/
	@for d in $(ROOT)/infra/*/; do \
	  d=$${d%/}; echo "==> $$d"; \
	  terraform -chdir="$$d" init -backend=false -input=false -lockfile=readonly; \
	  terraform -chdir="$$d" validate; \
	  if [ -d "$$d/tests" ]; then terraform -chdir="$$d" test; fi; \
	done

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

.PHONY: gpu-plan gpu-status
gpu-plan: ## Show what gpu-apply would create/delete on Lambda and the hourly cost (read-only)
	$(ROOT)/scripts/gpuctl plan

gpu-status: ## Show Lambda GPU capacity and this cluster's GPU instances (read-only)
	$(ROOT)/scripts/gpuctl status

.PHONY: sync-peers
sync-peers: ## Rewrite the peer allow-lists (cp k3s_peers set, Lambda firewall) from the current node IPs
	$(ROOT)/scripts/gpuctl sync-peers
