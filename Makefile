# Developer and CI entry points. Requires GNU make and uv.
# Every tool runs through `uv run --locked`, so a stale uv.lock fails instead of being rewritten.

UV_RUN := uv run --locked

.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck imports proto proto-check test test-integration check

help: ## List the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z][a-z-]*:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create .venv from uv.lock and install the git hooks
	uv sync --locked
	$(UV_RUN) pre-commit install

lint: ## Check linting and formatting
	$(UV_RUN) ruff check
	$(UV_RUN) ruff format --check

format: ## Format, then apply lint fixes
	$(UV_RUN) ruff format
	$(UV_RUN) ruff check --fix

typecheck: ## Type-check src, tests and scripts (mypy --strict)
	$(UV_RUN) mypy

imports: ## Check the import-layer contracts
	$(UV_RUN) lint-imports --no-logo

proto: ## Regenerate src/goaly/contracts/gen/ from proto/
	$(UV_RUN) python scripts/protogen.py

proto-check: ## Fail if src/goaly/contracts/gen/ is out of date with proto/
	$(UV_RUN) python scripts/protogen.py --check

test: ## Run the unit tests with branch coverage
	$(UV_RUN) pytest --cov

# pytest exits with 5 when no test is selected: not a failure while no integration test exists.
test-integration: ## Run only the integration tests (Docker, databases, the real model API)
	$(UV_RUN) pytest --integration -m integration || test $$? -eq 5

check: lint typecheck imports proto-check test ## Run everything CI runs
