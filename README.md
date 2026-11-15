# GreekKV

[![CI](https://github.com/georgebon5/greekkv/actions/workflows/ci.yml/badge.svg)](https://github.com/georgebon5/greekkv/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Open-source study + toolkit measuring what KV-cache quantization costs in quality and gains in serving capacity for Greek-language LLM workloads.**

> Headline question: *At fixed GPU memory, how many more concurrent users can a Greek LLM serve if the KV cache is quantized to INT8/INT4, and what does that cost in Greek-language quality?*

Status: **pre-alpha** — scaffolding in place; first measured results land in M1–M3.

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

## Development setup

```bash
git clone https://github.com/georgebon5/greekkv.git
cd greekkv
python -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[dev]"
pre-commit install
make check          # ruff + format + pytest (same as CI)
```

Python 3.10 or 3.11 supported.

> **Note on CI status:** The CI badge above may show red/unknown while this
> account's GitHub Actions is temporarily unavailable (see
> [`docs/DECISIONS.md`](docs/DECISIONS.md), ADR-0001). Every PR runs `make check`
> locally and pastes the output into the PR description as a substitute.

## Reproducibility contract

Every number in this repo must be traceable to:
1. A file under `results/` committed to git, and
2. The code + config that produced it.

Any result from larger hardware than free tiers (Kaggle T4/P100, Colab T4) is labelled **estimate**, never **measurement**.

## License

MIT — see [LICENSE](LICENSE).
