# X / Twitter thread — 5 tweets

**Target:** 5 tweets, each ≤ 280 chars (X free tier), one visual per tweet except the ask. Posting order matters: the first tweet must stand on its own because the reply chain is rendered differently depending on client.

**Images folder:** `results/figures/` holds every figure the project generates. Suggested picks are inline below.

---

## Thread

### Tweet 1/5 — hook

> New: **GreekKV v0.1.0** — measured how much KV-cache quantization costs in Greek-language quality and gains in serving capacity.
>
> On `Qwen/Qwen2.5-0.5B` at n_ctx=1024, `int8-kivi` serves **1.88× more concurrent users than FP16** at identical PPL (5.16).
>
> Numbers + repo ↓
>
> 📷 [perplexity sweep figure](../../results/figures/perplexity_qwen2_5_0_5b.png)

Character count target: ≤ 275 (leave room for URL expansion).

### Tweet 2/5 — the trap

> The `int4-fakequant` trap: 3.8× more users on paper, but decode latency jumps **23× on CPU** (139 vs 5 ms/tok).
>
> Cause is architectural: per-layer quantize/dequantize round-trip firing on every decode step. KIVI's residual window dodges it by keeping fresh tokens in FP16.
>
> 📷 [decode latency figure (T23 profile)](../../results/figures/profile/profile_summary.png)

### Tweet 3/5 — the tokenizer finding

> Scheme-independent: **Qwen's English-centric BPE spends 4.38× more tokens per Greek word than Krikri's Greek-aware tokenizer.**
>
> That one number swings the KV-cache budget on a 1000-word Greek doc more than any quantization choice downstream.
>
> Pick the tokenizer first.
>
> 📷 [tokenizer fertility figure](../../results/figures/fertility.png)

### Tweet 4/5 — what's shipped

> Everything measured, committed as JSONL, reproducible with one `make reproduce`.
>
> 📦 Python capacity CLI + static TS planner (52 KB gzip)
> 🧪 25 Py↔TS parity tests
> 📄 6-page technical report with CIs
> 🔧 NEON INT8 KV cache contributed to minillm.cpp
>
> 🔗 github.com/georgebon5/greekkv

### Tweet 5/5 — the ask

> Honest limits: one measured model (Qwen2.5-0.5B), n_ctx stops at 1024, CPU host for every latency number.
>
> Next T4 re-run is Krikri-8B + Meltemi-7B quality cells. If you want to help or want a specific cell prioritised, reply or DM.
>
> 🔗 [v0.1.0 release](https://github.com/georgebon5/greekkv/releases/tag/v0.1.0)

---

## Character-count notes

X's current count treats every URL as 23 chars regardless of real length — no need to shorten links. Emoji are variable-width; the picks above use 1-char glyphs (📷 📦 🧪 📄 🔧 🔗). If a tweet runs over, the easiest cut is the second sentence in tweet 2 (the architectural-cause line) — the first sentence is the content people retweet.

## Pre-post checklist

- [ ] Each tweet copy-pasted into [x.com's composer](https://x.com/compose/post) individually, character count ≤ 280.
- [ ] Image attached before posting the next tweet in the chain — once posted, images cannot be appended retroactively.
- [ ] Links resolve. If Pages is still waiting on Actions billing, swap any `georgebon5.github.io/greekkv/` URL for the release page.
- [ ] Final tweet names an ask, not a vague "thoughts?".
- [ ] Alt-text set on every image (X supports up to 1 000 chars per image; use it for the key number in that figure).
