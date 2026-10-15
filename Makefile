# Developer and CI entry points. Requires GNU make and uv.
# Every tool runs through `uv run --locked`, so a stale uv.lock fails instead of being rewritten.

UV_RUN := uv run --locked

.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck imports proto proto-check test test-integration check \
	freeze prepare final final-plan baseline report

help: ## List the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z][a-z-]*:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Every dependency group, rsi included; the organizers' plain `uv sync --locked` leaves rsi out.
setup: ## Create .venv from uv.lock, with every dependency group, and install the git hooks
	uv sync --locked --all-groups
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

# The final evaluation (spec §10). Every run is the isolated pipeline: proxy, solve container, host
# grading. These need Docker, PostgreSQL and the executor API's key; `make final-plan` prints the
# final's plan without running anything.
FINAL_RUNS ?= runs/final
# Where v0's runs are; point it at T12's archived runs if they live elsewhere.
BASELINE_RUNS ?= runs/baseline
HARNESS ?= v0
V0_REF ?= v0
REPORT ?= docs/results/final-report.md
# A run is passed to the report only once its directory exists; any other role shows as pending.
report_run = $(if $(wildcard $(2)/manifest.json),--$(1) $(2))

freeze: ## Print the freeze plan for tagging main as final; FREEZE_ARGS=--yes tags and pushes the tag
	$(UV_RUN) python scripts/freeze.py $(FREEZE_ARGS)

prepare: ## Prepare the TPC-H database, the PostgreSQL instances and the solve sandbox image
	$(UV_RUN) python scripts/effisql/pgenv.py setup
	$(UV_RUN) python -c 'from pathlib import Path; from goaly.bench.sandbox import ensure_image; print(ensure_image(Path.cwd()))'

final: prepare ## Reproduce the final result at the tag final: held-out x3, val x3, then the report
	$(UV_RUN) python scripts/effisql/final_runs.py final --root $(FINAL_RUNS) --harness $(HARNESS) $(RUNS_ARGS)
	$(MAKE) report

final-plan: ## Print what make final would run, and check the checkout; runs nothing
	$(UV_RUN) python scripts/effisql/final_runs.py final --root $(FINAL_RUNS) --harness $(HARNESS) --dry-run

baseline: prepare ## Rerun v0 at V0_REF: train x1 and val x2 in dev mode, then held-out x1 official
	$(UV_RUN) python scripts/effisql/final_runs.py baseline --root $(BASELINE_RUNS) --ref $(V0_REF) --harness $(HARNESS) $(RUNS_ARGS)

report: ## Write the final report from the runs and the ledger; REPORT_ARGS=--allow-partial drafts it
	$(UV_RUN) python scripts/effisql/final_report.py --template docs/results/final-report-template.md \
		--out $(REPORT) --verdicts docs/results/audit-verdicts.json \
		$(call report_run,v0-heldout,$(BASELINE_RUNS)/heldout) \
		$(call report_run,v0-val,$(BASELINE_RUNS)/val) $(call report_run,final-heldout,$(FINAL_RUNS)/heldout) \
		$(call report_run,final-val,$(FINAL_RUNS)/val) $(REPORT_ARGS)
