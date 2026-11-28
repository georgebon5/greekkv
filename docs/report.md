---
title: "GreekKV — what KV-cache quantization costs and gains on Greek LLM workloads"
author: "Giorgos Mponobas"
date: "2026-11-28"
version: "v0.1.0"
---

# GreekKV — what KV-cache quantization costs and gains on Greek LLM workloads

> **Version.** v0.1.0 · 2026-11-28 · companion to
> [github.com/georgebon5/greekkv](https://github.com/georgebon5/greekkv)

## 1. Question

Modern Greek LLM deployments (Krikri, Meltemi, multilingual Qwen) spend most of their serving budget on the **KV cache**, not the weights. A 7–8B model at FP16 fits in a single consumer GPU, but at `n_ctx = 1024` the KV cache for a hundred concurrent users already dwarfs the weights. Shrinking that cache by a factor of 2 or 4 — if the quality holds — reshapes the economics.

That turns into one measurable question:

> **At fixed GPU memory, how many more concurrent users can a Greek LLM serve if the KV cache is quantized to INT8 / INT4, and what does that cost in Greek-language quality?**

Three constraints inform the rest of this report:

1. **Greek, not English.** Byte-level BPE tokenizers tuned on English waste 4–5× more tokens on Greek than on their native language. That inflates the cache cost of a document in Greek before any quantization enters the picture — see §3.1.
2. **Measured, not estimated.** Every quality or latency number in this report comes from JSONL files committed under [`results/`](https://github.com/georgebon5/greekkv/tree/main/results); every hardware number comes from a closed-form formula (not a benchmark). The planner UI and the CLI both tag every row `measured` or `estimated` so a reader never has to guess.
3. **Free-tier hardware.** All measurements in v0.1.0 were produced on a CPU host (Apple Silicon). This bounds two things: which models fit (Qwen2.5-0.5B is the only measured one) and which numbers are decisive (quality trends are, absolute latency on a T4 is not — §6).

## 2. Related work in one page

Three strands of prior work fix the shape of the measurement.

- **KV-cache quantization for serving.** KIVI ([Liu et al., 2024](https://arxiv.org/abs/2402.02750)) proposed the residual-window scheme used throughout this project: fresh tokens stay in FP16 inside a `n_sink + n_recent` window, older tokens are symmetric-per-group quantized to INT8 / INT4. The scheme matters for v0.1.0 because it is the only quantized variant whose decode latency does not blow up on CPU (§4.3) — the residual window moves the quantize-dequantize round-trip off the hot path for the new token.
- **Streaming attention baselines.** StreamingLLM ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)) introduced attention sinks + a recent-window cache that trades quality for a bounded memory footprint. We reproduce it under the `eviction-sink` label as a *floor* rather than a competitor: its capacity numbers are huge (§4.4) but its quality collapses (top-1 agreement with FP16 drops to 2%).
- **Greek LLMs.** The two public Greek-centric 7B-class checkpoints — Krikri ([ILSP, 2025](https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct)) and Meltemi ([ILSP, 2024](https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1.5)) — ship their own tokenizers. The Greek-vs-English fertility difference across those tokenizers (§3.1) is why the headline question is interesting at all: an English-only BPE tokenizer inflates the Greek cache cost 4× before quantization gets a say.

Everything else — Triton kernels, NEON intrinsics, dispatch-overhead analyses — is referenced inline next to the specific finding it informs.

## 3. Method

### 3.1 Data

| Signal | Source | Sampling | Committed under |
|---|---|---|---|
| Tokenizer fertility (T06) | FLORES-200 devtest, EN + EL | 500 sentences, seed = 0 | `results/fertility.csv` |
| KV cost per doc (T08) | Greek Wikipedia (1000-word articles) | 10 docs per model | `results/figures/kv_greek_vs_english.csv` |
| Perplexity (T17) | Greek Wikipedia (sliding window) | 15 chunks per `(scheme, n_ctx)` cell | `results/perplexity/qwen2_5_0_5b.jsonl` |
| Greek multi-choice (T18) | ILSP Greek translations of ARC / HellaSwag / MMLU | 50 / 50 / 20 questions, seed = 42 | `results/benchmarks/qwen2_5_0_5b.jsonl` |
| Needle retrieval (T19) | Synthetic Greek haystack, 3 needle categories × grid | 18 trials per scheme | `results/needle/qwen2_5_0_5b.jsonl` |
| Systems profile (T23) | Random-token prefill + 16-token decode | 3 `n_ctx` × 6 schemes | `results/profile/qwen2_5_0_5b.jsonl` |
| Kernel benchmark (T25) | Synthetic `[H, D]` Q + `[H_kv, N, D]` KV | 10 iters per cell | `results/kernel_bench/qwen2_5_0_5b_shape.jsonl` |

### 3.2 Model

`Qwen/Qwen2.5-0.5B` (494 M parameters, 24 transformer layers, 14 Q heads, 2 KV heads, head dim = 64, vocab = 151 936). GQA ratio of 7:1 means the raw KV cache is ~7× smaller than the equivalent MHA layout before any quantization; that frames every absolute MB number in the report.

### 3.3 Schemes

Six cache behaviours, every one of them measured end-to-end:

| Scheme | Bits (KV) | Group size | Residual FP16 | Cache size vs `n_ctx` |
|---|---:|---:|---:|---|
| `fp16` | 16 | — | — | grows linearly |
| `int8-fakequant` | 8 | 64 | 0 | grows linearly (1/2 of `fp16`) |
| `int8-kivi` | 8 | 64 | 36 tokens | grows linearly after window |
| `int4-fakequant` | 4 | 64 | 0 | grows linearly (1/4 of `fp16`) |
| `int4-kivi` | 4 | 64 | 36 tokens | grows linearly after window |
| `eviction-sink` | 16 | — | 36 tokens | **bounded at 36 tokens** regardless of `n_ctx` |

The residual window of 36 tokens = 4 sink + 32 recent is the StreamingLLM default; we keep it so downstream comparisons stay apples-to-apples with the public baseline. §6 discusses what happens at wider windows (not yet measured).

### 3.4 Metrics and CIs

Quality:

- **Perplexity** on a Greek Wikipedia sliding window. Reported as a mean over 15 chunks per cell.
- **Top-1 / top-5 agreement vs FP16** — fraction of positions where the quantized argmax matches the FP16 reference. 95% bootstrap CIs with 1000 resamples and `seed = 0` are the ones quoted in every table; `scripts/build_report.py` regenerates them deterministically.
- **KL(P_ref ‖ P_q)** — mean and p95. The p95 is the tail-behaviour signal that quality means hide.

Latency and capacity:

- **Decode latency (ms/tok)** — one `n_ctx`-token prefill followed by 16 one-token decodes against the already-quantized cache. Peak RSS delta over the post-load baseline is logged alongside.
- **Capacity (max concurrent users)** — closed-form arithmetic over committed byte formulas: `users = (vram − weights_bytes − overhead_bytes) ⁄ kv_bytes_per_user`. Default overhead is 650 MiB (median post-load allocator state from T23). The TypeScript and Python implementations are pinned byte-for-byte against each other (`planner/src/capacity.test.ts` × 25 fixtures) so the UI and the CLI cannot drift.

### 3.5 Honesty gates

Three mechanisms keep the numbers trustworthy without relying on discipline:

- `greekkv-capacity --measured` only annotates `ppl` and `ms/tok` on rows whose exact `(model, scheme, n_ctx)` triple exists in the JSONL; everything else prints `—`.
- `make verify-reproduce` runs the full report pipeline twice with a frozen `SOURCE_DATE_EPOCH` and fails if `docs/RESULTS.md` or `planner/public/data.json` diverge.
- `test_build_report_runs_against_committed_rows` parses the real `results/` tree in CI, so a broken JSONL cannot ship inside a passing PR.
