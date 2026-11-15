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
