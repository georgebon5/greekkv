# Architecture Decision Records (ADR)

Short, dated, decision-focused notes. Each entry: context → decision → consequences.

---

## ADR-0001 — Merge T02 (CI) without a green CI run

**Date:** 2026-11-15
**Status:** Accepted (bootstrap exception)

### Context
The T02 PR introduces `.github/workflows/ci.yml` with a Python 3.10/3.11 matrix,
ruff lint, ruff format check, and pytest. The workflow file is correct and all
checks pass locally.

GitHub Actions on this account is blocked with:
> "The job was not started because your account is locked due to a billing issue."

The repository is **public** (Actions is free for public repos) and the billing
dashboard shows `$0` metered usage with `$18.00` of included compute available.
The blocker is an **account-level** lock, not a repo-level or usage-level issue.
Removing the expired MasterCard that was on file did not auto-unlock the account.

### Decision
Merge PR #36 (T02) without a green CI run, under the same bootstrap exception
already used for T00 and T01 (which merged before CI existed). From T03 onward,
`make check` is treated as the authoritative substitute for CI and every PR
description must include its output until Actions is unblocked.

### Consequences
- Every subsequent PR must paste `make check` output into the PR body.
- The CI badge in the README will remain red/unknown until the account is
  unlocked; the README explicitly says so to avoid misleading readers.
- `workflow_dispatch` is added to the workflow so CI can be triggered manually
  with a single `gh workflow run` call the moment the account unlocks, without
  needing a dummy commit.
- If the account is unlocked and the workflow still has issues, that is treated
  as a real CI bug and fixed in its own PR (not folded into T02).

---

## ADR-0002 — `src/` layout + hatchling + ruff + pytest

**Date:** 2026-11-15
**Status:** Accepted

### Context
First tooling choice of the project. Options considered:
- Flat layout vs `src/` layout
- `setuptools` vs `hatchling` vs `poetry` vs `uv` as build backend
- `ruff` vs `black + isort + flake8` for lint/format
- `pytest` vs `unittest`

### Decision
- **`src/` layout.** Forces imports to go through the installed package. If the
  package isn't installed, tests fail — which is the signal we want in CI. A
  flat layout lets stale local imports sneak in and silently pass.
- **`hatchling`** as build backend. PEP 517, zero legacy baggage, cleanest
  config for `src/` layout, no `setup.py` / `MANIFEST.in` required.
- **`ruff` (+ `ruff-format`).** One tool for linting, formatting, and import
  order. Faster than `black + isort + flake8` combined and configured in a
  single section of `pyproject.toml`. Enabled rule sets: E, F, I, B, UP, N, SIM.
- **`pytest`.** Industry standard; `pythonpath = ["src"]` wired in
  `pyproject.toml`.
- **100-char line length.** Compromise between legacy 80 and pragmatic 120 for
  evaluation code with many kwargs. `ruff-format` handles wrapping.

### Consequences
- Contributors must `pip install -e ".[dev]"` once before tests pass.
- All configuration lives in `pyproject.toml`; no separate `setup.cfg`,
  `.ruff.toml`, `.flake8`, `tox.ini`.
- Future tool additions (mypy, coverage, hypothesis) extend `pyproject.toml`
  and the `dev` optional-dependency group — no new config files.
- `make check` is the single local entry point that mirrors CI.
