# Show HN draft

HN's bar is "something you built that works today, with numbers". Title has a hard 80-char limit (visible length). First paragraph of the self-text is what decides whether the submission gets upvoted; everything after is read only by commenters.

**Submission URL:** `https://github.com/georgebon5/greekkv` (prefer the repo over the planner — HN audiences want to see the code first).

---

## Title (≤ 80 chars)

Primary pick (76 chars including spaces):

> Show HN: GreekKV – 1.9× users at same perplexity, measured on a Greek LLM

Alternatives (pick one that matches the mood of the day):

- `Show HN: Measuring what KV-cache quantization costs on Greek LLMs` (62 chars)
- `Show HN: KV-cache quantization on a Greek LLM, all numbers committed` (69 chars)

Avoid: adjectives in the title ("blazing fast", "efficient"). Avoid the word "simple" — HN rewards specific numbers, not vibes.

---

## Self-text

> Hi HN,
>
> I spent ~3 weeks measuring what KV-cache quantization actually costs in Greek-language quality and gains in serving capacity. The project is open-source and every number is committed as a JSONL under `results/`, with a one-command reproduce (`make reproduce`).
>
> The headline, measured on `Qwen/Qwen2.5-0.5B` at `n_ctx = 1024`:
>
> - `int8-kivi` serves 1.88× more concurrent users than FP16 at identical perplexity (5.16) and lower decode latency (5.3 vs 15.4 ms/tok). 95% bootstrap CI on top-1 agreement vs FP16: (0.77, 0.83).
> - `int4-fakequant` serves 3.8× more users on paper but decodes 23× slower on CPU (139 vs 5 ms/tok). The cause is architectural: a full quantize/dequantize round-trip firing on every layer at every decode step hits the Python dispatcher ~96 times per step; KIVI dodges this by keeping fresh tokens in an FP16 residual window.
> - Qwen's English-centric tokenizer spends 4.38× more tokens per Greek word than Krikri's (ILSP's Greek Llama fork). That one number swings the KV-cache budget more than any quantization scheme.
>
> What's in the repo:
>
> - Python capacity model + a `greekkv-capacity` CLI that tags every row `measured` or `estimated` depending on whether the committed JSONL has a matching cell.
> - A static TypeScript planner + dashboard (52 KB gzip, no backend). TS port of the capacity math is byte-for-byte parity-tested against the Python reference (25 vitest fixtures).
> - An INT8 KV-cache + hand-written NEON dequant path contributed to a companion C++ inference engine (`minillm.cpp`).
> - A 6-page technical report (`docs/report.pdf`) with CIs, limitations, and a priority-ordered plan for what to do with ~$200 of T4 time.
>
> Known limits: one measured model (Qwen2.5-0.5B), context grid stops at 1024, CPU host for every measurement. The honesty contract is enforced mechanically by `make verify-reproduce` (idempotency gate) and by the `measured|estimated` source tag on every row.
>
> Happy to answer questions, especially on the KIVI residual-window trade-off or on the tokenizer finding.

---

## First-comment self-reply (optional, HN-friendly)

Within the first hour, post one top-level comment on your own thread with the thing you actually want feedback on. Example:

> One specific thing I'd love a sanity check on: my Python / TypeScript capacity models produce byte-identical output on a 18-row parity suite (planner/src/capacity.test.ts). That's the gate I trust most. If anyone sees a case the fixtures miss — a weight-only quantization row, a mixed KV-weight scheme, an over-provisioned KIVI window — I'd love a PR or just a comment pointing at the corner.

## Pre-submit checklist

- [ ] Title under 80 chars, one measured number in it.
- [ ] Submission URL is the repo (not the planner, not the release page).
- [ ] Self-text first paragraph names the headline number.
- [ ] Every number matches `docs/report.md` §4 verbatim.
- [ ] Open the submission URL in an incognito window before posting — HN fetches the README synchronously.
- [ ] Be online for the next 2 hours after posting; the half-life of the ranking signal is that short.
