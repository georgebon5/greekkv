# Capacity model + CLI (T28)

**Measured, not assumed — wherever possible.** `greekkv-capacity`
translates a `(model, VRAM, n_ctx, scheme)` tuple into the maximum
number of concurrent users that fit alongside the weights and a fixed
runtime overhead. The hardware number is always arithmetic; the
quality (perplexity) and latency (ms/token) columns come from the
committed T17 / T23 JSONL files when they match, and are blanked
otherwise. Every row carries a `measured` / `estimated` tag so the
reader never has to guess which half of the number is load-bearing.

## The arithmetic

For each scheme:

```
free_bytes  = vram_bytes − weights_bytes(cfg, bits_weights)
            − overhead_bytes

kv_per_user = f(scheme, n_ctx)             # see table below

users       = free_bytes // kv_per_user    # 0 if either term is ≤ 0
```

`weights_bytes` and `kv_bytes` are the T07 formulas from
`src/greekkv/memory/kv.py`. `overhead_bytes` is a single knob (default
650 MB — the T23 median post-load allocator state on free-tier Kaggle
T4 / Colab T4). Override it with `--overhead-mb N` if you have a local
profile.

## Scheme bytes

| Scheme | KV bytes per user, `n_ctx = N` |
|---|---|
| `fp16`              | `kv_bytes(cfg, N, bits=16, group_size=0)` |
| `int8-fakequant`    | `kv_bytes(cfg, N, bits=8,  group_size=64)` |
| `int4-fakequant`    | `kv_bytes(cfg, N, bits=4,  group_size=64)` |
| `int8-kivi`         | `kv_bytes(cfg, 36, bits=16)` + `kv_bytes(cfg, N−36, bits=8,  group_size=64)` |
| `int4-kivi`         | `kv_bytes(cfg, 36, bits=16)` + `kv_bytes(cfg, N−36, bits=4,  group_size=64)` |
| `eviction-sink`     | `kv_bytes(cfg, min(N, 36), bits=16, group_size=0)` |

The residual window `36 = n_sink + n_recent` matches the T17 / T23
configuration. Override it with `CapacityInputs.scheme_specs` in code,
or by extending `DEFAULT_SCHEME_SPECS`.

## CLI

```bash
# Default: --measured — annotate only rows with a measured JSONL cell.
greekkv-capacity --model Qwen/Qwen2.5-0.5B --vram-gb 16 --n-ctx 1024

# Formula-only: ppl and ms/tok columns always blank (no JSONL lookup).
greekkv-capacity --model Qwen/Qwen2.5-0.5B --vram-gb 16 --n-ctx 4096 --estimate

# Mix measurements with the estimate fallback — same hardware math,
# richer annotations where measurements exist.
greekkv-capacity --model Qwen/Qwen2.5-0.5B --vram-gb 16 --n-ctx 1024 --all

# JSON output for downstream scripts / the T30 dashboard.
greekkv-capacity --model Qwen/Qwen2.5-0.5B --vram-gb 16 --n-ctx 1024 --json
```

### Known models (as of 2026-11)

- `Qwen/Qwen2.5-0.5B` — the only one with a committed measurement grid
  (T17 / T23). Works with `--measured`.
- `Qwen/Qwen2.5-7B`
- `ilsp/Llama-Krikri-8B-Instruct`
- `ilsp/Meltemi-7B-Instruct-v1.5`

To add a model, extend `_KNOWN_MODELS` in
`src/greekkv/cli/capacity_cli.py` (an `ArchConfig` with
`num_parameters` is enough; the rest is for the KV formula).

## Sample output

```text
model=Qwen/Qwen2.5-0.5B  vram=16.0 GiB  n_ctx=1024  overhead=650 MiB  mode=measured

scheme         | users  | kv/user   | weights   | ppl      | ms/tok | source
---------------+--------+-----------+-----------+----------+--------+---------
fp16           | 1,232  | 12.0 MiB  | 942.2 MiB | 4.15     | 15.37  | measured
int8-fakequant | 2,390  | 6.2 MiB   | 942.2 MiB | 5.16     | 6.03   | measured
int8-kivi      | 2,314  | 6.4 MiB   | 942.2 MiB | 5.16     | 5.28   | measured
int4-fakequant | 4,640  | 3.2 MiB   | 942.2 MiB | 22.18    | 139.58 | measured
int4-kivi      | 4,229  | 3.5 MiB   | 942.2 MiB | 35.54    | 5.90   | measured
eviction-sink  | 35,061 | 432.0 KiB | 942.2 MiB | 12323.30 | 4.96   | measured
```

Reading this row by row:

- **`fp16` sets the floor**: 1,232 users at 12 MiB of KV cache each
  before the budget is spent. Any quantization scheme that keeps
  quality has to buy more users than this to be worth the quality risk.
- **`int8-kivi` is the practical winner** at this `n_ctx`: 1.9× more
  users than FP16 at identical PPL (5.16) and slightly lower latency
  (5.28 vs 6.03 ms/tok). No surprise — it's also what T17 and T23
  independently pointed at.
- **`int4-fakequant` is the loud fail**: 3.8× more users than FP16 at
  the cost of 23× worse decode latency (139 ms/tok) and 5× worse PPL.
  The slowdown is architectural, not measurement noise — see
  `docs/systems_profile.md` for the diagnosis.
- **`eviction-sink` is a floor, not a competitor**: enormous capacity
  (35k users!) but PPL 12,323 — effectively broken on this task. It
  stays in the output as the "what if we threw the cache away?" lower
  bound.

## Honesty contract

- **`measured`** means every number on that row came from a committed
  JSONL file. The source paths are configurable
  (`--perplexity-jsonl`, `--profile-jsonl`) and default to the T17
  and T23 artefacts.
- **`estimated`** means the hardware number (`users`, `kv/user`,
  `weights`) is still the arithmetic formula, but the `ppl` and
  `ms/tok` columns had no row in the JSONL for this `(model, scheme,
  n_ctx)` triple and are printed as `—`.
- There is **no interpolation** across `n_ctx` or across models. If you
  need a measurement at `n_ctx = 2048`, run the T17 / T23 sweep there
  and commit the row — do not ask the CLI to invent it. `--all` is a
  convenience for reading, not a promise about accuracy.

## Where this lives

- Core: `src/greekkv/memory/capacity.py`
- Measurement resolver: `src/greekkv/memory/measured.py`
- CLI: `src/greekkv/cli/capacity_cli.py`
- Tests: `tests/test_capacity.py`, `tests/test_measured.py`,
  `tests/test_capacity_cli.py` (32 cases total)
- Related docs: `docs/kv_cost.md` (T07/T08), `docs/perplexity.md` (T17),
  `docs/systems_profile.md` (T23)
