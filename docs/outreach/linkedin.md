# LinkedIn post — v0.1.0 announcement

**Target length:** ~250 words (LinkedIn truncates at ~210 chars in feed; the rest unfolds on click).
**Figure:** attach one image. Default pick is [`results/figures/perplexity_qwen2_5_0_5b.png`](../../results/figures/perplexity_qwen2_5_0_5b.png) — the perplexity sweep clearly shows the FP16/INT8 cluster at the bottom and the INT4/eviction-sink gap above, which is the headline visual the rest of the post refers to. If LinkedIn's image aspect ratio is a problem, the memory-breakdown shot from the planner (`screenshots` would need to be added — note for the author) works too.

---

## Post body

> **The Greek KV-cache quantization trade-off is visible in three numbers.**
>
> I just tagged **v0.1.0** of GreekKV — an open-source study + toolkit measuring exactly what KV-cache quantization costs in Greek-language quality and gains in serving capacity.
>
> Headline, measured on `Qwen/Qwen2.5-0.5B` at `n_ctx = 1024`:
>
> → `int8-kivi` serves **1.88× more concurrent users** than FP16 at **identical perplexity** (5.16) and lower decode latency (5.3 vs 15.4 ms/tok). 95% bootstrap CI on top-1 agreement vs FP16: (0.77, 0.83).
>
> → `int4-fakequant` serves 3.8× more users on paper but decodes **23× slower** on CPU (139 vs 5 ms/tok). The cause is architectural: a full quantize/dequantize round-trip firing on every layer, every decode step — Python dispatcher overhead dominates. KIVI's residual window dodges this.
>
> → The one scheme-independent finding: Qwen's English-centric tokenizer spends **4.38× more tokens per Greek word** than Krikri's Greek-aware tokenizer. On a 1000-word document, that is 342% larger KV cache before quantization enters the picture. Whatever Greek LLM you deploy, pick the tokenizer first.
>
> Everything is measured, reproducible with one `make reproduce`, and tagged `measured` or `estimated` per row — no hidden interpolation, no fake-data fallbacks.
>
> 🔗 Repo + release: github.com/georgebon5/greekkv
> 🔗 Live planner + dashboard: georgebon5.github.io/greekkv/
> 📄 Technical report + CHANGELOG + wheel on the v0.1.0 release page.
>
> Happy to hear what cell reviewers want the next T4 re-run to fill in first — Krikri and Meltemi quality rows are at the top of my list.
>
> #LLM #Greek #NLP #KVCache #Quantization #OpenSource

---

## Hashtag note

LinkedIn only surfaces 3–5 hashtags reliably; the trailing set above is a sampler. Pick the three that best match your network:

- **Greek audience:** `#GreekNLP #LLM #KVCache`
- **ML-infra audience:** `#LLM #Quantization #Inference`
- **Academic audience:** `#NLP #Reproducibility #OpenSource`

## Pre-post checklist

- [ ] Figure attached and preview looks right at 1200×627 (LinkedIn's preferred OG size).
- [ ] Both links resolve.
- [ ] The three numbers in the post exactly match [`docs/report.md` §4](../report.md). Copy-paste, do not retype.
- [ ] If Pages is still waiting on Actions billing, swap the planner link with the release page.
