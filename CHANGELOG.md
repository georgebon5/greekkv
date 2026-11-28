# Changelog

All notable changes to GreekKV live here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each entry carries a GitHub issue reference (`#NN`) so the full code +
discussion path stays one click away.

## [Unreleased]

Tracking M4 Kaggle T4 re-run (Krikri / Meltemi quality cells),
`triton_fused` kernel timings, and the follow-ups listed in
[`README.md`](README.md) **Roadmap after v0.1.0**.

## [0.1.0] — 2026-11-28

First tagged release. Measures what KV-cache quantization costs in
Greek-language quality and gains in serving capacity on
`Qwen/Qwen2.5-0.5B`, with a static planner + dashboard, a Python +
TypeScript capacity model, a hand-written INT8 + NEON KV-cache
reference in `minillm.cpp`, and a reproducibility contract that `make
verify-reproduce` enforces mechanically.

### Added

- **M1 — Greek tokenization & KV cost** (#5 – #8)
  - Parallel corpus loader, tokenizer-fertility sweep across Krikri /
    Meltemi / Qwen, KV-byte calculator, Greek-vs-English figure.
- **M2 — Quantization library** (#9 – #14)
  - Scalar primitives (symmetric per-row INT8 / INT4), fake-quant
    cache, KIVI-style cache with residual window, eviction baseline,
    HF + vLLM-FP8 cache adapters.
- **M3 — Evaluation** (#15 – #20)
  - Model loader with VRAM guard, context-then-continuation scorer,
    perplexity sweep on Greek Wikipedia, Greek multiple-choice
    benchmark runner (arc_el / hellaswag_el / mmlu_el), needle-in-a-
    haystack retrieval, fidelity metrics (top-1 / top-5 / KL) vs FP16.
- **M4 — Systems** (#21 – #26)
  - Resumable sweep runner (`scripts/run_sweep.py`), statistical
    report generator (`scripts/build_report.py` → `docs/RESULTS.md`),
    memory + latency profiler, fused dequant-attention Triton kernel,
    kernel benchmark (fp16 baseline vs unfused PyTorch vs Triton
    fused), **cross-contribution** INT8 KV-cache + NEON dequant path
    in `minillm.cpp` (see
    [`github.com/georgebon5/minillm-cpp`](https://github.com/georgebon5/minillm-cpp#int8-kv-cache-opt-in-utility)).
- **M5 — Product** (#28 – #30)
  - `greekkv-capacity` CLI with `--measured | --estimate | --all`
    modes and JSON output; static Vite + React + TypeScript planner
    app with live controls, memory-breakdown stacked bar, and three
    dashboard charts (perplexity × n_ctx, decode latency × n_ctx,
    fidelity top-1 vs KL scatter); 25 TypeScript parity tests pin the
    TS capacity port byte-for-byte to the Python reference.
- **M6 — Packaging** (#31, #33)
  - README headline + measured-vs-estimated section + limitations +
    roadmap; `make verify-reproduce` honours `SOURCE_DATE_EPOCH` and
    diffs two consecutive runs; this CHANGELOG.

### Measured — headline numbers at `Qwen/Qwen2.5-0.5B`, 16 GiB VRAM, n_ctx = 1024

- `fp16`: PPL **4.15**, top-1 agree 1.00, decode **15.4 ms/tok**, 1,232 users.
- `int8-kivi`: PPL **5.16**, top-1 agree 0.80 (0.77, 0.83), decode **5.3 ms/tok**, **2,314 users (1.88× vs FP16)**.
- `int4-fakequant`: PPL 22.2, top-1 agree 0.22 (0.19, 0.25), decode **139.6 ms/tok** (⚠ 23× hit — see `docs/systems_profile.md`), 4,640 users.
- `eviction-sink` is a floor, not a competitor: 35,061 users at PPL = 12,323.

### Known limitations in v0.1.0

- One measured model (`Qwen/Qwen2.5-0.5B`). Krikri-8B and Meltemi-7B
  configs are usable in the planner UI with `estimated` tags; quality
  columns stay blank until the M4 Kaggle T4 re-run commits rows.
- Context grid stops at `n_ctx = 1024`; capacity extrapolates
  closed-form, but quality past that is unknown.
- All measurements on CPU host (Apple Silicon). The T24 Triton kernel
  and T26 NEON path are unit-tested but not timed vs FP16 on a T4 yet.
- Technical report (#32) ships in 0.1.1 — this release attaches
  `docs/RESULTS.md` and `docs/kernel_bench.md` as asset references.

### Determinism

`make verify-reproduce` runs the full report pipeline twice with a
frozen `SOURCE_DATE_EPOCH` and fails if `docs/RESULTS.md` or
`planner/public/data.json` diverge. Verified green in a fresh venv as
of this tag.

[Unreleased]: https://github.com/georgebon5/greekkv/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/georgebon5/greekkv/releases/tag/v0.1.0
