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

## Contents

1. Question
2. Related work in one page
3. Method
4. Results with CIs
5. Findings
6. Limitations and what I'd do with production hardware
