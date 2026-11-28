# GreekKV v0.1.0 — first measured release

> **TL;DR.** At fixed GPU memory, `int8-kivi` serves **1.88× more concurrent Greek users than FP16** at the same perplexity (5.16 on `Qwen/Qwen2.5-0.5B`, n_ctx=1024) and lower decode latency (5.3 vs 15.4 ms/tok). INT4 pushes the capacity ratio to 3.4–3.8× but comes with either a 23× decode-latency hit (`int4-fakequant`, T23 architectural finding) or a steep quality drop (`int4-kivi`: top-1 agreement vs FP16 collapses to 0.18). All numbers are measured, committed under `results/`, and reproducible with one `make reproduce`.

## What's in this release

Everything planned for M1 – M5 landed, plus the M6 release scaffolding. The repo is self-contained: clone, `make install`, `make reproduce`, and every chart and table in the docs regenerates from the committed JSONL. No model call runs at reproduce time — the measurements are fixed artefacts.

- **Capacity model + CLI** — `greekkv-capacity --measured | --estimate | --all` prints the full scheme grid with measured annotations wherever the T17 / T20 / T23 JSONL has a matching cell.
- **Static planner + dashboard** — [`georgebon5.github.io/greekkv/`](https://georgebon5.github.io/greekkv/). Two tabs: live capacity table + memory-breakdown stacked bar (Planner), and three measured-data charts (Dashboard: perplexity × n_ctx, decode latency × n_ctx, fidelity top-1 vs KL scatter). 52 KB gzip, zero chart libs.
- **TS ↔ Python parity** — 25 vitest fixtures pin the browser's capacity math byte-for-byte to the Python reference.
- **Determinism gate** — `make verify-reproduce` honours `SOURCE_DATE_EPOCH` and diffs two consecutive runs; failing CI on a drift catches silent JSONL / markdown mismatches.
- **minillm.cpp cross-contribution** — INT8 KV-cache + NEON dequant path lives in the companion repo ([`georgebon5/minillm-cpp`](https://github.com/georgebon5/minillm-cpp#int8-kv-cache-opt-in-utility)) with 132/132 green tests and a documented bench.

## Headline measured results

`Qwen/Qwen2.5-0.5B`, 16 GiB VRAM, n_ctx = 1024. 95% bootstrap CIs on top-1 agreement (1000 resamples, seed = 0). Full grid in `docs/RESULTS.md`.

| Scheme | PPL | top-1 vs FP16 | decode ms/tok | users | vs FP16 |
|---|---:|---:|---:|---:|---:|
| `fp16` (baseline) | 4.15 | 1.00 | 15.4 | 1,232 | 1.00× |
| `int8-fakequant` | 5.16 | 0.81 (0.78, 0.84) | 6.0 | 2,390 | 1.94× |
| **`int8-kivi`** | **5.16** | 0.80 (0.77, 0.83) | **5.3** | 2,314 | **1.88×** |
| `int4-fakequant` | 22.2 | 0.22 (0.19, 0.25) | ⚠ 139.6 | 4,640 | 3.77× |
| `int4-kivi` | 35.5 | 0.18 (0.15, 0.21) | 5.9 | 4,229 | 3.43× |
| `eviction-sink` | 12 323 | 0.02 (0.01, 0.04) | 5.0 | 35,061 | 28.5× |

## How to reproduce

```bash
git clone https://github.com/georgebon5/greekkv.git && cd greekkv
python -m venv .venv && source .venv/bin/activate
make install
make reproduce               # re-plots + regenerates RESULTS.md + data.json
make verify-reproduce        # idempotency gate (frozen SOURCE_DATE_EPOCH)
make check                   # ruff + format + pytest (293 tests)
greekkv-capacity --model Qwen/Qwen2.5-0.5B --vram-gb 16 --n-ctx 1024 --all
```

No GPU, no internet after `pip install`, no accounts.

## Known limitations

- Measured model coverage is **`Qwen/Qwen2.5-0.5B` only**. Krikri-8B and Meltemi-7B configs are usable in the planner UI with `estimated` tags; quality columns stay blank until the M4 Kaggle T4 re-run commits rows.
- Context grid stops at `n_ctx = 1024`. Capacity extrapolates closed-form, but PPL / latency past that point are unknown and the dashboard will not draw them.
- All measurements on an Apple-Silicon CPU host. The T24 Triton kernel and T26 NEON path are unit-tested but not timed vs FP16 on a T4 yet.
- **Technical report (#32)** is in flight — this release attaches `docs/RESULTS.md` and `docs/kernel_bench.md` as the canonical result references. A standalone PDF follows in 0.1.1.

## Assets

- [`docs/RESULTS.md`](../docs/RESULTS.md) — full aggregated report (`make reproduce` regenerates).
- [`docs/capacity.md`](../docs/capacity.md) — arithmetic + scheme bytes + honesty contract.
- [`docs/systems_profile.md`](../docs/systems_profile.md) — T23 profiler findings including the `int4-fakequant` dispatch-overhead diagnosis.
- [`docs/kernel_bench.md`](../docs/kernel_bench.md) — T25 fp16 vs unfused vs Triton kernel timings.
- [`docs/dashboard.md`](../docs/dashboard.md) — three dashboard charts + schema v2 contract.

## Thanks

Cross-contribution to [`georgebon5/minillm-cpp`](https://github.com/georgebon5/minillm-cpp) sharpened the C++ side of the KV story; the two repos cross-reference each other in their READMEs.

## What's next

Roadmap after 0.1.0 is in [`README.md` → Roadmap](../README.md#roadmap-after-v010):

1. **M4 Kaggle T4 re-run** — Krikri-8B / Meltemi-7B rows + Triton kernel timings.
2. **T32 technical report** — PDF walking through the claims and the pitfalls.
3. **T27 throughput sweep (optional)** — batch=1 vs batch=16 on T4.
4. **T34 outreach drafts** — short-form copy so the reader lands on this README with context.
