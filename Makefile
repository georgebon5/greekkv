.PHONY: help install check lint fmt test clean

PY ?= python3

help:
	@echo "greekkv dev targets:"
	@echo "  install   install package + dev deps in current env"
	@echo "  check     run everything CI runs (lint + format + tests)"
	@echo "  lint      ruff check"
	@echo "  fmt       ruff format (writes files)"
	@echo "  test      pytest"
	@echo "  clean     remove caches and build artifacts"

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

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
