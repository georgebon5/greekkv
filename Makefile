.PHONY: help install check lint fmt test clean reproduce verify-reproduce

PY ?= python3
# Frozen epoch used by verify-reproduce. Any valid Unix timestamp works;
# it only has to be stable between two consecutive runs. Default:
# 2026-11-15T00:00:00Z — the day the project was bootstrapped.
SOURCE_DATE_EPOCH ?= 1794700800

help:
	@echo "greekkv dev targets:"
	@echo "  install            install package + dev deps in current env"
	@echo "  check              run everything CI runs (lint + format + tests)"
	@echo "  lint               ruff check"
	@echo "  fmt                ruff format (writes files)"
	@echo "  test               pytest"
	@echo "  reproduce          regenerate figures + docs/RESULTS.md from committed results"
	@echo "  verify-reproduce   run reproduce twice and fail if output drifts"
	@echo "  clean              remove caches and build artifacts"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

check: lint test
	ruff format --check .

lint:
	ruff check .

fmt:
	ruff format .
	ruff check --fix .

test:
	pytest

reproduce:
	$(PY) experiments/plot_kv_cost.py
	$(PY) experiments/plot_perplexity.py
	$(PY) experiments/plot_benchmarks.py
	$(PY) scripts/build_report.py

# Idempotency check: run reproduce twice with a fixed SOURCE_DATE_EPOCH
# so generated_at does not drift, then diff the output files tracked by
# git. A clean exit means `make reproduce` is deterministic and the
# committed RESULTS.md / data.json agree with the JSONL sources.
verify-reproduce:
	@echo "→ first pass (SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH))"
	@SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(MAKE) --no-print-directory reproduce >/dev/null
	@cp docs/RESULTS.md /tmp/greekkv-results.md.first
	@cp planner/public/data.json /tmp/greekkv-data.json.first
	@echo "→ second pass"
	@SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(MAKE) --no-print-directory reproduce >/dev/null
	@diff -u /tmp/greekkv-results.md.first docs/RESULTS.md \
		|| (echo "✗ docs/RESULTS.md drifted between runs"; exit 1)
	@diff -u /tmp/greekkv-data.json.first planner/public/data.json \
		|| (echo "✗ planner/public/data.json drifted between runs"; exit 1)
	@rm -f /tmp/greekkv-results.md.first /tmp/greekkv-data.json.first
	@echo "✓ reproduce is idempotent"

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
