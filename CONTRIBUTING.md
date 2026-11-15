# Contributing to GreekKV

Thanks for taking the time to improve this project. The goal of GreekKV is to
produce **measured, reproducible** numbers about KV-cache quantization on Greek
LLMs — every contribution is judged by whether it moves that bar.

## Workflow

One task = one issue = one branch = one PR. No bundling.

1. **Pick (or open) an issue.** Check which milestone it belongs to on the
   [roadmap](https://github.com/georgebon5/greekkv/milestones).
2. **Branch off latest `main`:**
   ```bash
   git checkout main && git pull
   git checkout -b t07-kv-memory-calculator
   ```
   Branch name: `t<NN>-short-slug` matching the issue number and title.
3. **Code in small, meaningful commits.** Each commit must build and pass tests
   on its own. No padding, no "wip" commits — if you need to iterate, rebase
   before opening the PR.
4. **Run `make check` locally** before opening the PR:
   ```bash
   make check   # ruff lint + pytest + ruff format check
   ```
   Everything must be green. Paste the output into the PR description — it is
   the CI substitute until GitHub Actions is unblocked (see
   [`docs/DECISIONS.md`](docs/DECISIONS.md) ADR-0001).
5. **Open the PR** with the template filled in and `Closes #<issue>`.
6. **Squash-merge** into `main`. Delete the branch.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) — `type(scope): subject`:

| type       | use for                                      |
|------------|----------------------------------------------|
| `feat`     | new library capability                       |
| `fix`      | bug fix                                      |
| `test`     | tests only                                   |
| `docs`     | README, docs, outreach drafts                |
| `refactor` | no behavior change                           |
| `ci`       | workflows, hooks                             |
| `chore`    | tooling, build, scaffolding                  |

Subject is a short imperative. Greek or English is fine — pick the one that
reads more naturally for that change. Body (optional) explains the *why* when
it's non-obvious.

## Code style

- Python 3.10+, type hints on public functions.
- `ruff check` and `ruff format` are the source of truth (configured in
  `pyproject.toml`).
- `pre-commit install` runs them on every commit.
- **Default to no comments.** Only add a comment when the *why* is non-obvious
  (hidden invariant, subtle workaround). Well-named identifiers already explain
  the *what*.

## Data & reproducibility

Every number that lands in `README.md`, `docs/`, or `results/` must be
traceable to:

1. A file in `results/` committed to git, **and**
2. The code + config that produced it (`git_sha` + `config_hash` are embedded
   in each `ResultRecord`).

If you cannot provide both, the number is an **estimate**, not a measurement —
label it as such everywhere it appears.

## Scope discipline

- Don't add features, refactor, or introduce abstractions beyond what the task
  requires. A bug fix doesn't need surrounding cleanup.
- Don't add error handling for scenarios that can't happen. Validate at system
  boundaries (user input, external APIs), trust internal code.
- Three similar lines is better than a premature abstraction.

## Reporting bugs

Use the **Bug** issue template. Include a minimal reproducer (config + command)
and the exact environment. Bug fixes land as their own PRs with a regression
test.

## Code of Conduct

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).
