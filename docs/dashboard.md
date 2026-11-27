# Results dashboard (T30)

A second tab in the static planner app that visualises every row in
the committed `results/*.jsonl` without re-running anything. Nothing
in the dashboard is interpolated or smoothed — if a cell was not
measured, it is not drawn. The acceptance bar from issue #30 is:
"charts are reproducible from the committed results file; no hidden
data." Every chart here is one `fetch("data.json")` away from the
same user's browser.

## Where to find it

- Deployed: [`georgebon5.github.io/greekkv/`](https://georgebon5.github.io/greekkv/) → **Dashboard** tab
- Local: `cd planner && npm ci && npm run build && npm run preview`

## The three charts

### 1. Perplexity × n_ctx (log-y)

- **Source:** `results/perplexity/qwen2_5_0_5b.jsonl` (T17).
- **Lower is better.** Log-y because the schemes span three decades
  (fp16 ≈ 4; int4-fakequant ≈ 23; eviction-sink ≈ 12k).
- **What to read out:** `int8-fakequant` and `int8-kivi` sit right on
  top of `fp16` — the quality tax at 8-bit is in the second decimal.
  `int4-fakequant` opens a visible gap. `eviction-sink` is the loud
  floor (unreadable text, but you can still measure it).

### 2. Decode latency × n_ctx (linear-y)

- **Source:** `results/profile/qwen2_5_0_5b.jsonl` (T23).
- **Lower is better.** Linear-y specifically so the outlier reads as
  an outlier.
- **What to read out:** `int4-fakequant` ≈ 140 ms/token vs ≈ 5 ms for
  everything else. The gap is architectural: `FakeQuantCache.update()`
  runs a full quantize → dequantize round trip on every layer at
  every decode step, and at 4 bits the pack/unpack bit-ops hit the
  dispatcher 96 times per step. See `docs/systems_profile.md` for the
  diagnosis. KIVI escapes this because fresh tokens live in the FP16
  residual window and quantization only fires on eviction.

### 3. Fidelity scatter — top-1 vs KL

- **Source:** `results/perplexity/qwen2_5_0_5b.jsonl` (T17, fidelity
  fields added in T20).
- **Top-left is the goal:** high argmax agreement with FP16 + low
  KL(P_ref ∥ P_q).
- **What to read out:** `fp16` sits on (0, 1) by construction. int8
  variants cluster around (0.18, 0.80) — ~80% of argmax calls still
  match FP16. int4 variants fall off to (~2.0, 0.20). `eviction-sink`
  is around (7, 0.03) — effectively guessing.

## How to reproduce the data

```bash
# Rebuild RESULTS.md and the dashboard snapshot from the committed JSONL.
python scripts/build_report.py

# This writes:
#   docs/RESULTS.md           — human-readable report
#   planner/public/data.json  — schema v2 snapshot the UI reads
```

The two outputs share the same JSONL inputs (`results/*.jsonl`) so a
reviewer can grep a number in the markdown report and find the same
number in the dashboard without any middle step.

## Schema v2

`planner/public/data.json` carries both the planner's joined grid and
the dashboard's raw rows:

| Key | Shape | Produced by |
|---|---|---|
| `measurements` | `[{model, scheme, n_ctx, perplexity?, kl_mean?, top1_agree?, decode_ms_per_token?, peak_mem_mb?}]` | T28 planner lookup |
| `perplexity_rows` | `[{model, scheme, metrics: {…}}]` | T17 + T20 |
| `profile_rows` | `[{model, scheme, metrics: {…}}]` | T23 |
| `kernel_bench_rows` | `[{model, scheme, metrics: {…}}]` | T25 |

If the schema bumps again, the planner code guards `schema_version >=
2` and shows a "stale data" placeholder rather than rendering garbage.

## Honesty contract

- **No interpolation.** Each dot is one JSONL row, period.
- **No cross-model averaging.** The dashboard picks the primary
  model (currently Qwen2.5-0.5B — the only one with a committed grid).
  Krikri / Meltemi rows will appear automatically once the M4 Kaggle
  T4 re-run commits them.
- **No fake-data fallback.** If the JSONL has no row for a
  (scheme, n_ctx) cell, the chart just does not draw that dot. Nothing
  is backed into a formula.
