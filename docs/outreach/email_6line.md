# 6-line email template

The brief says *result → links → role → ask* in six lines. Below is the structural skeleton, then two variants tuned for different recipient types. Use the variant whose first line's number matches what the recipient cares about most.

Keep the subject line to one measured number, not an adjective.

---

## Skeleton (fill in `[brackets]`)

> **Subject:** [one measured number from the headline table, e.g. "Greek KV cache — 1.9× users at same perplexity"]
>
> Hi [First name],
>
> [One sentence with the measured headline number + CI, named model, and context length. No adjectives.]
> [One sentence with the single concrete finding that matches their job — infra, capacity, quality, or tokenizer.]
> Repo and release: https://github.com/georgebon5/greekkv (v0.1.0, measured on Qwen2.5-0.5B).
> Live planner + dashboard: https://georgebon5.github.io/greekkv/ · technical report: docs/report.pdf in the release.
> [One sentence role-framing — who you are + why you're reaching out specifically to this person.]
> [One sentence with the explicit ask — review, call, hiring conversation, redistribution. Signed, Giorgos.]

Six content lines. The subject is not counted.

---

## Variant A — serving infra lead

> **Subject:** Greek LLM serving — 1.88× users at the same perplexity, measured
>
> Hi [First name],
>
> On `Qwen/Qwen2.5-0.5B` at `n_ctx = 1024`, `int8-kivi` serves 1.88× more concurrent users than FP16 at identical perplexity (5.16) and lower decode latency (5.3 vs 15.4 ms/tok). 95% bootstrap CI on top-1 agreement vs FP16: (0.77, 0.83).
> The useful bit for an infra team is the `int4-fakequant` trap: 23× decode-latency hit on CPU from the per-layer quantize/dequantize round-trip — `int4-kivi` dodges it by keeping fresh tokens in an FP16 residual window.
> Repo and release: https://github.com/georgebon5/greekkv (v0.1.0, all results committed as JSONL).
> Live planner + dashboard: https://georgebon5.github.io/greekkv/ · technical report: docs/report.pdf in the release.
> I'm a [grad student / junior ML engineer / etc.] working on Greek-language LLM serving; reaching out because your team ships [specific product / paper].
> If any of this is useful to you, I'd love 20 minutes to compare notes — otherwise, feedback on the method would already be a big deal.

---

## Variant B — Greek NLP researcher

> **Subject:** KV-cache quantization on Greek — Krikri tokenizer is doing most of the work
>
> Hi [First name],
>
> The headline tokenizer-fertility number is 4.38× tokens/word for Qwen's BPE on Greek vs 1.21× for Krikri's — Krikri alone saves ~75% of the KV cache on a 1000-word Greek document before any quantization enters the picture.
> On top of that baseline, `int8-kivi` is practically free (top-1 agreement vs FP16 = 0.80, CI (0.77, 0.83)) and `int4-kivi` keeps the decode path fast but drops to 0.18 top-1 agreement — the quality floor on 4-bit is lower than published Qwen results on English would suggest.
> Repo and release: https://github.com/georgebon5/greekkv (v0.1.0, measured on Qwen2.5-0.5B — Krikri / Meltemi quality cells are the next re-run).
> Live planner + dashboard: https://georgebon5.github.io/greekkv/ · technical report: docs/report.pdf in the release.
> I'm a [student / independent researcher] running this as a thesis-adjacent project; reaching out because of your work on [paper / dataset].
> Would you be open to a pointer on priors to watch out for before the Krikri re-run, or a reader-check on the report? Signed, Giorgos.

---

## Checklist before send

- [ ] Subject line has one number, not an adjective.
- [ ] First content line has the measured number for this recipient's job.
- [ ] Both links resolve (planner URL may be down if Actions is locked — fall back to `cd planner && npm run preview`).
- [ ] Recipient's name is their name, not a role placeholder.
- [ ] The ask is one specific action, not "thoughts".
