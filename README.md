# GreekKV

[![CI](https://github.com/georgebon5/greekkv/actions/workflows/ci.yml/badge.svg)](https://github.com/georgebon5/greekkv/actions/workflows/ci.yml)
[![Pages](https://github.com/georgebon5/greekkv/actions/workflows/pages.yml/badge.svg)](https://github.com/georgebon5/greekkv/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Planner](https://img.shields.io/badge/planner-live-brightgreen)](https://georgebon5.github.io/greekkv/)
[![Reproducible](https://img.shields.io/badge/reproducible-make%20verify--reproduce-blue)](Makefile)

**At fixed GPU memory, `int8-kivi` serves ~1.9× more concurrent Greek users than FP16 at the same perplexity (5.16, Qwen2.5-0.5B, n_ctx=1024)** and at marginally lower decode latency (5.28 vs 6.03 ms/tok). INT4 pushes that to ~3.8× users on paper but blows up decode latency to ~140 ms/tok under the vanilla fake-quant cache; KIVI's residual window avoids that 23× slowdown while keeping the extra capacity. Every number in that sentence is measured on `Qwen/Qwen2.5-0.5B` and committed as JSONL under `results/` — `greekkv-capacity --measured` prints the full grid, and the live dashboard at [`georgebon5.github.io/greekkv/`](https://georgebon5.github.io/greekkv/) plots it.

> **Headline question.** At fixed GPU memory, how many more concurrent users can a Greek LLM serve if the KV cache is quantized to INT8 / INT4, and what does that cost in Greek-language quality?

Status: **v0.1.0-rc** — M1–M5 complete and measured end-to-end on a single model. M6 (release + report) is in flight. The capacity model, planner UI, and dashboard are usable today.

## Planned deliverables

| Milestone | Scope |
|-----------|-------|
| **M0 — Foundation** | Repo, CI, configs, contribution scaffolding |
| **M1 — Greek tokenization & KV cost** | Tokenizer fertility, KV calculator, Greek-vs-English figure |
| **M2 — Quantization library** | Primitives, fake-quant cache, KIVI-style cache, baselines |
| **M3 — Evaluation** | Perplexity, Greek benchmarks, needle, fidelity metrics |
| **M4 — Systems** | Memory/latency profiler, Triton kernel, C++ cross-contribution |
| **M5 — Product** | Capacity CLI, planner web app, results dashboard |
| **M6 — Packaging & outreach** | README, technical report, release, outreach drafts |

Full task list lives in [GitHub Issues](https://github.com/georgebon5/greekkv/issues) (one issue per task, grouped by milestone).

## Reproduce the whole report in one command

```bash
git clone https://github.com/georgebon5/greekkv.git && cd greekkv
python -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[dev]"  — CPU wheels, ~2 GB
make reproduce      # regenerates figures + docs/RESULTS.md + planner/public/data.json
make check          # ruff + format + pytest (291 tests, ~35s on CPU)
```

Everything under `results/` is already committed, so `make reproduce` never calls a model — it only re-plots the figures and regenerates the aggregated report + planner data file from the JSONL. No GPU, no internet after `pip install`, no accounts. Python 3.10 or 3.11 supported.

**Determinism check.** `make verify-reproduce` runs the pipeline twice with a frozen `SOURCE_DATE_EPOCH` and fails if `docs/RESULTS.md` or `planner/public/data.json` diverge between the two passes. Verified locally in a fresh venv as of v0.1.0-rc:

```
→ first pass (SOURCE_DATE_EPOCH=1794700800)
→ second pass
✓ reproduce is idempotent
```

> **Note on CI status.** The `CI` and `Pages` badges above may show red/unknown while this account's GitHub Actions is temporarily locked (see [`docs/DECISIONS.md`](docs/DECISIONS.md) ADR-0001). Every PR pastes the `make check` output into its description as a CI substitute until the lock lifts; the workflows themselves are green when triggered locally with `act`.

## Try it

**Web planner + dashboard:** [`georgebon5.github.io/greekkv/`](https://georgebon5.github.io/greekkv/) — one static TypeScript app, 52 KB gzip, offline-friendly, two tabs:

- **Planner.** Pick a model, dial VRAM / GPU count / context length, and watch max concurrent users update live per scheme. Perplexity and decode-latency columns are populated from the committed `results/` JSONL via `scripts/build_report.py → planner/public/data.json`.
- **Dashboard.** Three charts over the committed JSONL: perplexity × n_ctx (log-y), decode latency × n_ctx (linear-y), and fidelity top-1 vs KL scatter. Every dot is one JSONL row — no interpolation, no smoothing. See [`docs/dashboard.md`](docs/dashboard.md).

**CLI:** after `pip install -e .`, the `greekkv-capacity` command is on your PATH:

```bash
greekkv-capacity --model Qwen/Qwen2.5-0.5B --vram-gb 16 --n-ctx 1024 --all
```

Rows tagged `measured` come from the T17 / T23 JSONL; `estimated` rows have no corresponding measurement row and blank the PPL / ms columns. See [`docs/capacity.md`](docs/capacity.md) for the arithmetic, scheme breakdown, and honesty contract, and [`planner/README.md`](planner/README.md) for the web app's build + deploy story.

> **Note on Pages deployment:** The Pages workflow may not have run yet due to the same account-level Actions lock documented in [`docs/DECISIONS.md`](docs/DECISIONS.md) (ADR-0001). `cd planner && npm ci && npm run build && npm run preview` reproduces the deployed app locally.

## Reproducibility contract

Every number in this repo must be traceable to:
1. A file under `results/` committed to git, and
2. The code + config that produced it.

Any result from larger hardware than free tiers (Kaggle T4/P100, Colab T4) is labelled **estimate**, never **measurement**.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for workflow, commit conventions, and
the reproducibility contract. Architectural decisions are logged as ADRs in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

All participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).
