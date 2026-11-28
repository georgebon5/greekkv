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

## 4. Results

### 4.1 The Greek tokenizer tax

Tokens per whitespace-separated word on FLORES-200 devtest, 500 sentences per language (figure source: `experiments/fertility.py`, figure at `results/figures/fertility.png`):

| Tokenizer | EN | EL | **EL / EN** |
|---|---:|---:|---:|
| `ilsp/Llama-Krikri-8B-Instruct` | 1.27 | 1.53 | **1.21×** |
| `ilsp/Meltemi-7B-Instruct-v1.5` | 1.37 | 1.43 | **1.04×** |
| `Qwen/Qwen2.5-7B-Instruct` | 1.30 | 5.67 | **4.38×** |

Krikri and Meltemi shipped their own Greek-aware tokenizers; Qwen's English-centric BPE spends 4.4× more tokens on the same Greek text. At `n_ctx = 1024` that is the difference between 242 KB and 1 062 KB of KV cache per user before any quantization enters the picture. Everything else in this report runs on top of that baseline difference.

### 4.2 Perplexity on Greek Wikipedia

15 chunks per `(scheme, n_ctx)` cell, sliding-window perplexity, `Qwen/Qwen2.5-0.5B` (figure source: `experiments/plot_perplexity.py`, figure at `results/figures/perplexity_qwen2_5_0_5b.png`).

| Scheme | n_ctx = 256 | n_ctx = 512 | n_ctx = 1024 |
|---|---:|---:|---:|
| `fp16` | 3.89 | 3.98 | 4.15 |
| `int8-fakequant` | 4.55 | 4.62 | 5.16 |
| `int8-kivi` | 4.48 | 4.66 | 5.16 |
| `int4-fakequant` | 24.12 | 23.10 | 22.18 |
| `int4-kivi` | 29.61 | 34.49 | 35.54 |
| `eviction-sink` | 3257.23 | 6372.87 | 12323.30 |

Read: 8-bit (either KIVI or fake-quant) sits within 25% of the FP16 baseline on this task; 4-bit opens a 5–9× gap; `eviction-sink` is three orders of magnitude worse because the cache is bounded at 36 tokens regardless of `n_ctx`.

### 4.3 Fidelity vs FP16 (n_ctx = 512)

Position-wise agreement with the FP16 argmax (figure source: §4.2 same script; fidelity metric defined in `src/greekkv/eval/fidelity.py`).

| Scheme | top-1 agree | top-5 agree | KL mean | KL p95 |
|---|---:|---:|---:|---:|
| `fp16` | 1.000 | 1.000 | 0.000 | 0.000 |
| `int8-fakequant` | 0.813 (0.78, 0.84) | 0.978 | 0.170 | 0.786 |
| `int8-kivi` | 0.798 (0.77, 0.83) | 0.972 | 0.187 | 0.907 |
| `int4-fakequant` | 0.220 (0.19, 0.25) | 0.550 | 1.815 | 3.952 |
| `int4-kivi` | 0.182 (0.15, 0.21) | 0.489 | 2.224 | 5.118 |
| `eviction-sink` | 0.023 (0.01, 0.04) | 0.077 | 7.386 | 12.810 |

CIs are 95% bootstrap over the 1920 scored positions per cell, 1000 resamples, `seed = 0`. Top-1 is more relevant to greedy decoding; top-5 to sampling and beam search. The int8 variants retain 80% greedy agreement and 97% sampling agreement — close enough that most downstream systems would not notice. int4 variants drop out of both regimes.

### 4.4 Decode latency and capacity

Peak RSS delta + ms/tok on a 16-token decode (figure source: `experiments/systems_profile.py`, `docs/systems_profile.md` has the full narrative).

| Scheme | ms/tok @ 256 | ms/tok @ 512 | ms/tok @ 1024 | users @ 16 GiB, n_ctx = 1024 |
|---|---:|---:|---:|---:|
| `fp16` | 5.8 | 4.6 | 15.4 | 1,232 |
| `int8-fakequant` | 5.8 | 6.8 | 6.0 | 2,390 |
| `int8-kivi` | 5.4 | 5.6 | **5.3** | 2,314 |
| `int4-fakequant` | **110** | **146** | **140** | 4,640 |
| `int4-kivi` | 5.6 | 5.3 | 5.9 | 4,229 |
| `eviction-sink` | 4.1 | 3.8 | 5.0 | 35,061 |

Two reads:

- **int4-fakequant is a loud outlier** — 23× slower than everything else. §5 explains the architectural cause.
- **Capacity scales roughly as expected**: int8 ≈ 2× FP16, int4 ≈ 4× FP16. KIVI pays a few per-mille for its FP16 residual window; eviction-sink is the only scheme whose cache does not grow with `n_ctx` and therefore its capacity number is bounded by the overhead, not the cache.

## 5. Findings

### F1 — INT8 is practically free

`int8-kivi` and `int8-fakequant` both land at **PPL = 5.16 on n_ctx = 1024** against the FP16 baseline of 4.15 — a ~24% relative penalty on absolute perplexity, but only ~2 percentage points of top-1 argmax disagreement with FP16. `int8-kivi` additionally comes in at **5.3 ms/tok** vs the baseline's 15.4, because it quantizes less (only the evicted tail), and its capacity sits at **1.88× FP16** at `n_ctx = 1024`. The headline recommendation out of this release is "turn on int8-kivi unless you have a specific reason not to."

### F2 — INT4 fake-quant is a 23× decode trap on CPU

`int4-fakequant` serves 3.77× more users per GiB than FP16 on paper but decodes at **139.6 ms/tok** — 23× slower than every other scheme. The cause is architectural, not measurement noise. The T23 profiler traced it to the full quantize→dequantize round-trip firing on every layer at every decode step:

- `FakeQuantCache.update()` does one `quantize` + one `dequantize` per new K/V row per layer.
- At 4 bits that includes `pack_int4` + `unpack_int4` bit-ops and a reshape through per-group scales — tiny-tensor operations that hit the Python / torch op dispatcher 24 layers × 2 tensors × 2 ops = ~96 times per decode step.
- On CPU, dispatcher overhead dominates. `int8-fakequant` escapes this because there is no bit-packing round trip at byte-aligned precisions.
- **KIVI's residual window is what dodges this bullet.** Fresh tokens live in FP16; quantization only fires on eviction, which at `n_ctx ≤ 4096` and a 36-token window is rare. That is why `int4-kivi` decodes at ~6 ms/tok while still writing int4 bytes to the evicted tail.

The T25 kernel benchmark shows the same shape at the micro-level: the unfused reference pays the dequant cost per call; the T24 Triton fused kernel is designed to amortize that inside a single GPU launch, and the T26 NEON path does the same on AArch64 CPUs. CPU numbers in this report would compress visibly on a GPU re-run — see §6.

### F3 — Eviction-sink is a floor, not a competitor

`eviction-sink` serves 28.5× more users than FP16 at `n_ctx = 1024`, but its perplexity on Greek Wikipedia is **12 323** and top-1 agreement with FP16 is **0.02** — effectively random. We keep it in the measured grid so the planner UI has a defensible lower bound; no downstream reader should treat it as an INT8 / INT4 alternative.

### F4 — The Greek tokenizer gap is load-bearing

The 4.38× fertility gap on Qwen2.5-7B's English-centric tokenizer (§4.1) moves the absolute MB numbers for every scheme in the capacity model. On Krikri's Greek-aware tokenizer the same document costs 20% more KV than English; on Qwen's it costs 340% more. Any Greek-serving deployment that picks its tokenizer badly pays an INT8-sized tax before quantization gets to say anything. This is the one finding in the project that is independent of the cache scheme and still deployable today.

## 6. Limitations and what I'd do with production hardware

### What this report *cannot* say

- **One measured model.** Every PPL / latency / fidelity cell uses `Qwen/Qwen2.5-0.5B`. The capacity model and the planner UI expose Krikri-8B / Meltemi-7B / Qwen2.5-7B configs, but their quality columns stay blank (`source = estimated`) until a GPU run commits JSONL rows for them.
- **Context grid stops at `n_ctx = 1024`.** Capacity extrapolates in closed form, but perplexity / latency past 1024 are **unknown**, and the dashboard refuses to draw them rather than interpolate.
- **CPU host.** All measured latency numbers are CPU — specifically Apple Silicon. The T23 int4-fakequant outlier is a CPU-dispatch finding; on a T4 the dispatcher overhead compresses by an order of magnitude and the four-bit variant may close the gap entirely (or not — this is a prediction, not a measurement).
- **Random tokens for the profiler.** T23 and T25 time attention against random `[H, D]` queries and `[H_kv, N, D]` caches. This is fair for *shape* dependencies but blind to content-driven effects (outlier channels, positional biases).
- **Scope is PPL + fidelity + needle.** No RLHF-era chat eval, no HumanEval-style generation benchmark, no long-form coherence test.

### What I'd do with ~$200 of T4 time

Three experiments, in priority order, that would retire the biggest open questions:

1. **Krikri + Meltemi full grid.** Rerun T17 (PPL), T20 (fidelity), T23 (profile) on Krikri-8B and Meltemi-7B across `n_ctx ∈ {256, 1024, 4096}` for all six schemes. Expected cost: ~6 hours on a T4. This lands the measured quality columns for the planner's Greek-centric rows, which is what everyone who tried the UI asked for.
2. **Triton kernel timing on real shapes.** Run `experiments/kernel_benchmark.py` on the T4, three `N_ctx` values, 30 iterations. Expected cost: ~5 minutes. Fills the one blank row in `docs/kernel_bench.md` and quantifies how much of the int4-fakequant gap is architectural vs CPU-specific.
3. **Context extrapolation to 8 k / 16 k.** Rerun T17 at `n_ctx ∈ {2048, 4096, 8192}` for the two int8 variants and FP16 only (the only cells reviewers actually care about). Expected cost: ~45 minutes. Settles whether the 24% perplexity penalty at 1024 is stable or widens — this is the one trend the planner UI cannot answer by extrapolation.

### What I would *not* rush to spend on

- Tuning the KIVI residual window beyond 36 tokens. The scaffolding allows it (`CapacityInputs.scheme_specs` overrides) but the measurement budget is better spent on model coverage.
- Chasing INT2. The fidelity floor already breaks at INT4 on this model; INT2 would be a floor-of-the-floor measurement and the engineering cost exceeds the information gain.
- A full RLHF-era chat eval harness. Different project; the capacity model does not care about it.
