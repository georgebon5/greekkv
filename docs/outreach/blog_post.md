# Blog post — long-form rewrite of the v0.1.0 technical report

**Target length:** ~1500 words. The audience is someone who found the project through the HN post or a LinkedIn share and wants the full story but in a conversational voice, not academic.

**Suggested platforms:** personal site (static Markdown), Dev.to (tag `#llm #quantization #opensource`), or Medium (paywall-friendly if you want eyeballs beyond the ML crowd). The technical report lives alongside this post as the formal reference; this version keeps the structure but drops the hedging and the tables.

**Lead image:** [`results/figures/perplexity_qwen2_5_0_5b.png`](../../results/figures/perplexity_qwen2_5_0_5b.png) — same figure as the LinkedIn post, same reason.

---

## Title picks

- "What I learned measuring KV-cache quantization on a Greek LLM" *(primary)*
- "KV-cache quantization is nearly free on Greek — until it isn't"
- "Three numbers that reshape Greek LLM serving economics"

## Draft

### Opening (the hook)

I set out to answer one question: *at fixed GPU memory, how many more concurrent users can a Greek LLM serve if the KV cache is quantized to INT8 or INT4, and what does that cost in Greek-language quality?* Three weeks later I had an open-source toolkit, a one-page technical report, a static planner web app, and three numbers I could not have guessed from first principles.

Here they are, up front:

1. **`int8-kivi` serves 1.88× more concurrent Greek users than FP16** at identical perplexity (5.16, Qwen2.5-0.5B, n_ctx=1024) and *lower* decode latency (5.3 vs 15.4 ms/tok). Top-1 argmax agreement with FP16 stays at 0.80 — most downstream systems wouldn't notice.
2. **`int4-fakequant` is a 23× decode-latency trap on CPU.** It doubles capacity again on paper but decodes at 139 ms/tok against 5 ms for every other scheme. KIVI dodges this — but the quality at 4 bits collapses on top of that.
3. **Qwen's English-centric tokenizer spends 4.38× more tokens per Greek word than Krikri's Greek-aware tokenizer.** On a 1000-word Greek document, that single choice swings the KV budget more than any quantization scheme downstream.

Whichever role you have reading this — serving infra, research, or product — at least one of those numbers is probably load-bearing for a decision you're making. The rest of the post explains how I got them and where the method stops being trustworthy.

### Why Greek matters here

Most published KV-cache quantization work is tested on English. English BPE tokenizers were trained on English; the few hundred million people who speak a Greek-script language inherit whatever byte-level fallback the tokenizer had for non-Latin ranges. The practical effect: if you run Qwen2.5-7B on Greek text, every word costs 4.4× more tokens than it would on English text. The KV cache is proportional to tokens, so your capacity budget is already 4.4× smaller before you've even picked a quantization scheme.

Krikri (an 8B Llama-family model fine-tuned on Greek) and Meltemi (a 7B Greek model from the same team at ILSP) ship their own Greek-aware tokenizers. On the same FLORES-200 Greek text, Krikri's BPE spends 1.21× English rate and Meltemi's spends 1.04×. That's the single largest deployable finding in the project: **pick the tokenizer first**, then worry about bits.

### What the measurement looks like

Every number in this project is produced by running a committed script against committed data and dropping the result into a committed JSONL file. `make reproduce` re-plays the plots and the aggregated report from those JSONL files; no model call runs at reproduce time. The honest-numbers contract is enforced by code, not by discipline:

- The `greekkv-capacity` CLI tags every row `measured` or `estimated`. If a `(model, scheme, n_ctx)` triple isn't in the JSONL, the PPL and ms/tok columns print `—`, never an inferred value.
- `make verify-reproduce` runs the pipeline twice with a frozen `SOURCE_DATE_EPOCH` and fails if `docs/RESULTS.md` or `planner/public/data.json` diverge between the two passes.
- The static planner app and the Python CLI share one capacity formula, pinned byte-for-byte by a 18-fixture parity test.

That last one is the piece I'm proudest of. A capacity planner that invents numbers in the browser is worth nothing; a capacity planner whose browser code is a *mirror* of the Python code I can run from a terminal is a different kind of artifact.

### The one architectural finding

Of the six cache schemes I measured, `int4-fakequant` is a loud outlier. It decodes at ~140 ms/tok on CPU — 23× slower than every other scheme. The capacity model says it should give you 3.8× more users than FP16, so on paper it's the sweet spot. In practice the throughput-per-user numbers invert.

Why? The T23 profiler traced it to a single architectural detail. `FakeQuantCache.update()` runs a full `quantize → dequantize` pass on every layer's new K and V tensors at every decode step. At 4 bits that includes `pack_int4` + `unpack_int4` bit-ops and a reshape through per-group scales — tiny-tensor operations that hit the Python / torch op dispatcher 24 layers × 2 tensors × 2 ops = ~96 times per decode step. On CPU, the dispatcher overhead dominates. `int8-fakequant` escapes this because there is no bit-packing round-trip at byte-aligned precisions.

KIVI's residual window — a FlashAttention-adjacent trick — avoids this entirely: fresh tokens stay in FP16 for the first `n_sink + n_recent` positions, and quantization only fires on eviction. At `n_ctx ≤ 4096` and a 36-token window, eviction is rare, so `int4-kivi` decodes at ~6 ms/tok while still writing int4 bytes to the evicted tail. The quality at 4 bits still collapses for other reasons (fidelity top-1 agreement vs FP16 is 0.18, which isn't usable for most downstream work), but the latency problem is solved cleanly.

The T24 Triton fused kernel and the T26 NEON dequant path (contributed to a companion C++ inference engine called `minillm.cpp`) are both designed to amortize this cost inside a single kernel launch. On a T4 the gap should compress visibly — that's one of the three experiments I'd run first with production hardware.

### What I'd do with ~$200 of T4 time

The technical report has the priority-ordered list; here it is in plain English:

1. **Krikri + Meltemi full grid.** Rerun perplexity, fidelity, and the systems profile on Krikri-8B and Meltemi-7B across `n_ctx ∈ {256, 1024, 4096}` for all six schemes. About 6 hours of T4 time. This lands the measured quality columns for the planner's Greek-centric rows, which is what everyone who tried the UI asked for first.
2. **Triton kernel timing on real shapes.** Run the kernel benchmark on the T4 — three `N_ctx` values, 30 iterations. About 5 minutes. Fills the one blank row in the kernel-bench doc and quantifies how much of the int4-fakequant gap is architectural vs CPU-specific.
3. **Context extrapolation to 8 k / 16 k.** Rerun the perplexity sweep at longer context for the two int8 variants and FP16 only. About 45 minutes. Settles whether the 24% perplexity penalty at 1024 is stable or widens.

I would *not* rush to tune the KIVI residual window beyond 36 tokens, chase INT2, or build a full chat-eval harness. Different projects.

### What this does *not* measure

I want this part to be loud, because reviewers skim and skimmers misquote:

- One measured model. Qwen2.5-0.5B. Everything else in the planner is hardware-arithmetic only, with the quality columns tagged `estimated` and blank.
- Context grid stops at 1024 tokens. Capacity extrapolates in closed form, but perplexity past 1024 is unknown and the dashboard refuses to draw it.
- CPU host for every latency number. On a GPU the dispatcher overhead compresses by an order of magnitude; the int4-fakequant finding may compress with it.
- Random tokens in the profiler. Fair for shape dependencies, blind to content-driven effects (outlier channels, positional biases).

If any of those limits make the finding you cared about look different, that's exactly the signal the next release should land.

### Where to find everything

- Repo and release: [github.com/georgebon5/greekkv](https://github.com/georgebon5/greekkv) (v0.1.0 tagged, Keep-a-Changelog format).
- Live planner + dashboard: [georgebon5.github.io/greekkv/](https://georgebon5.github.io/greekkv/) — pick a model, dial VRAM / context, read off the measured + estimated grid.
- Technical report: `docs/report.pdf` on the v0.1.0 release page, alongside the wheel and the raw markdown.
- Companion C++ contribution: [github.com/georgebon5/minillm-cpp](https://github.com/georgebon5/minillm-cpp#int8-kv-cache-opt-in-utility) — INT8 KV cache + NEON dequant path, 132/132 green tests.

If you're thinking about serving a Greek LLM, Krikri-first-then-int8-kivi is the config I'd run at the moment. If you're thinking about contributing, the top of my list is Krikri and Meltemi quality rows on a T4 — the CLI, the planner, and the dashboard are already there waiting for those cells.

---

## Pre-publish checklist

- [ ] Lead image embedded and width looks right at 1200 px.
- [ ] Every number matches `docs/report.md` §4 verbatim.
- [ ] All four outbound links resolve.
- [ ] Byline + affiliation accurate.
- [ ] Cross-post plan: LinkedIn (link to the blog post), Twitter thread (link to the blog post), HN (link to the repo).
