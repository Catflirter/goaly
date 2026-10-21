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
.PHONY: tf-test
tf-test: ## terraform init + validate every infra/ stack, then terraform test where it has tests/
	@for d in $(ROOT)/infra/*/; do \
	  d=$${d%/}; echo "==> $$d"; \
	  terraform -chdir="$$d" init -backend=false -input=false; \
	  terraform -chdir="$$d" validate; \
	  if [ -d "$$d/tests" ]; then terraform -chdir="$$d" test; fi; \
	done

lint: ## Run all pre-commit hooks, the repo secret scan, the Terraform tests, then the tests
	$(PY) -m pre_commit run --all-files
	$(MAKE) secrets
	$(MAKE) tf-test
	$(MAKE) test
