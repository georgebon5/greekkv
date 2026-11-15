# VRAM / RAM table for `greekkv.eval.load_model`

Predicted totals from :func:`greekkv.eval.estimate_memory` at `n_ctx = 2048`,
batch = 1, with the default 20 % activation overhead. **"Predicted"** means
these are formula-based — see T07 (`src/greekkv/memory/kv.py`) for the
weights + KV math. "Measured" entries are populated as we run each cell on
free-tier hardware.

Shorthand:

- **Weights**: `num_parameters · bits / 8  +  group-scale overhead` (group
  size = 64 for int4/int8, matching bitsandbytes NF4).
- **KV**: `2 · L · H_kv · d_head · n_ctx · 2` (FP16; the cache always starts
  fresh in FP16 before a cache scheme kicks in).
- **Activations**: 20 % of weights, rough budget for intermediate tensors
  during a forward pass.

## Kaggle T4 (16 GB VRAM) — predicted at n_ctx = 2048

| Model | dtype | Weights (GB) | KV (GB) | Act (GB) | **Total (GB)** | Fits T4? |
|---|---|---:|---:|---:|---:|:---:|
| `Qwen/Qwen2.5-0.5B`           | fp16 | 1.2  | 0.03 | 0.24 | **1.47** | ✅ |
| `Qwen/Qwen2.5-1.5B`           | fp16 | 3.1  | 0.11 | 0.62 | **3.83** | ✅ |
| `Qwen/Qwen2.5-3B`             | fp16 | 7.1  | 0.22 | 1.41 | **8.73** | ✅ |
| `Qwen/Qwen2.5-7B`             | fp16 | 15.3 | 0.44 | 3.06 | **18.80** | ❌ (**over**) |
| `Qwen/Qwen2.5-7B`             | int4 | 4.3  | 0.44 | 0.86 | **5.60**  | ✅ |
| `ilsp/Meltemi-7B-Instruct-v1.5` | fp16 | 14.4 | 1.00 | 2.88 | **18.28** | ❌ (**over**) |
| `ilsp/Meltemi-7B-Instruct-v1.5` | int4 | 4.1  | 1.00 | 0.82 | **5.92**  | ✅ |
| `ilsp/Llama-Krikri-8B-Instruct` | fp16 | 16.3 | 1.00 | 3.26 | **20.56** | ❌ (**over**) |
| `ilsp/Llama-Krikri-8B-Instruct` | int4 | 4.6  | 1.00 | 0.92 | **6.52**  | ✅ |

> Reproduce any row with:
> ```python
> from greekkv.eval import LoadSpec, estimate_memory
> from greekkv.memory import arch_from_hf, estimate_num_parameters
> from transformers import AutoConfig
>
> spec = LoadSpec(name="ilsp/Meltemi-7B-Instruct-v1.5", dtype="int4",
>                 memory_budget_gb=16.0, n_ctx_estimate=2048)
> cfg = arch_from_hf(AutoConfig.from_pretrained(spec.name))
> if cfg.num_parameters is None:
>     cfg = type(cfg)(**{**cfg.__dict__, "num_parameters": estimate_num_parameters(cfg)})
> print(estimate_memory(cfg, spec))
> ```

## Google Colab free T4 (16 GB VRAM)

Same ceiling as Kaggle T4. The headline project experiments on the 7B / 8B
Greek models **must** run in int4 to fit. fp16 runs are used only on the
≤3B models.

## Local CPU (Mac M-series, 32 GB RAM)

| Model | dtype | Total (GB) | Runs? |
|---|---|---:|:---:|
| `Qwen/Qwen2.5-0.5B`            | fp32 | ~2.9 | ✅ (used for T13 outlier analysis) |
| `Qwen/Qwen2.5-1.5B`            | fp16 | ~3.8 | ✅ (slow, ok for one-off) |
| `Qwen/Qwen2.5-7B`              | int4 | — | ❌ (bitsandbytes needs CUDA) |

CPU paths serve unit tests only — the headline numbers live on Kaggle / Colab.

## Measured entries

| Model | dtype | Hardware | Measured peak (GB) | Notes |
|---|---|---|---:|---|
| `hf-internal-testing/tiny-random-MistralForCausalLM` | fp16 | CPU | < 0.05 | T07 / T10 / T11 integration tests |
| `Qwen/Qwen2.5-0.5B`            | fp32 | CPU (Mac) | ~2.1 (resident) | T13 outlier analysis |

Future runs update this table from `results/vram_measured.jsonl` (planned in
T21).
