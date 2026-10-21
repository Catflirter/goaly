SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV := .venv
PY := $(VENV)/bin/python

.PHONY: help venv test lint

help: ## List available targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-16s %s\n", $$1, $$2}'

venv: ## Create .venv and install dev requirements
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

test: ## Run the pytest suite
	$(PY) -m pytest

lint: ## Run all pre-commit hooks, then the tests
	$(PY) -m pre_commit run --all-files
	$(MAKE) test
