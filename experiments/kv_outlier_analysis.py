"""T13 — Per-layer, per-channel K/V magnitude stats on Greek text.

Goal
----
Answer, with measurements: *why* does per-channel quantization help for K
and per-token for V (the KIVI layout)? The test is empirical — we run a
real model over Greek text, capture the K and V tensors at every layer,
and compute two things per layer:

- **Per-channel max magnitude** across sequence positions. If a handful of
  channels dominate, per-channel quantization confines their inflated
  scale to those channels and keeps the rest of the row tight.
- **Concentration ratio** ``max / median`` across channels. High ratios
  mean "a few channels carry outsized magnitude" — i.e. per-channel wins.
  Low ratios mean magnitude is diffuse across channels; per-token or
  per-tensor may suffice.

Usage
-----
::

    python experiments/kv_outlier_analysis.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --n-samples 4 \\
        --max-tokens 512 \\
        --out-json results/kv_outliers.json \\
        --out-fig results/figures/kv_outliers.png

The JSON schema lists per-layer channel/token stats. The figure is the
2-panel summary consumed by ``docs/kv_outliers.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from greekkv.memory import arch_from_hf, head_dim_of
from greekkv.tokenization import load_el_wikipedia_sample
from greekkv.utils import seed_all


class CaptureCache(DynamicCache):
    """DynamicCache that snapshots the NEW K/V chunk arriving at each layer.

    We keep a CPU + FP32 clone so analysis is numerically stable even when the
    model runs in fp16. Only the new chunk is kept — the accumulated cache is
    still delegated to the parent.
    """

    def __init__(self) -> None:
        super().__init__()
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.k[layer_idx] = key_states.detach().to(torch.float32).cpu().clone()
        self.v[layer_idx] = value_states.detach().to(torch.float32).cpu().clone()
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


def _per_layer_stats(k: torch.Tensor, v: torch.Tensor) -> dict[str, Any]:
    """Compute reduction statistics for one layer's captured K/V tensors.

    K and V have shape ``[B, H_kv, N, D]`` (B=1 here, H_kv is KV-head count,
    N is sequence, D is head_dim).
    """
    k_abs = k.abs()
    v_abs = v.abs()

    # Per-channel: reduce over (B, H_kv, N) → vectors of length D.
    k_ch_max = k_abs.amax(dim=(0, 1, 2)).tolist()
    k_ch_mean = k_abs.mean(dim=(0, 1, 2)).tolist()
    v_ch_max = v_abs.amax(dim=(0, 1, 2)).tolist()
    v_ch_mean = v_abs.mean(dim=(0, 1, 2)).tolist()

    # Per-token: reduce over (B, H_kv, D) → vectors of length N.
    k_tok_max = k_abs.amax(dim=(0, 1, 3)).tolist()
    v_tok_max = v_abs.amax(dim=(0, 1, 3)).tolist()

    def _concentration(xs: list[float]) -> float:
        a = np.asarray(xs)
        med = float(np.median(a))
        return float(a.max() / (med + 1e-8))

    return {
        "k_channel_max": k_ch_max,
        "k_channel_mean": k_ch_mean,
        "v_channel_max": v_ch_max,
        "v_channel_mean": v_ch_mean,
        "k_token_max": k_tok_max,
        "v_token_max": v_tok_max,
        "k_concentration": _concentration(k_ch_max),
        "v_concentration": _concentration(v_ch_max),
    }


def _run(model_name: str, text: str, max_tokens: int) -> tuple[CaptureCache, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=True,
    )
    cache = CaptureCache()
    with torch.no_grad():
        model(**inputs, past_key_values=cache, use_cache=True)

    cfg = arch_from_hf(AutoConfig.from_pretrained(model_name))
    meta = {
        "model": model_name,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": head_dim_of(cfg),
        "n_tokens": int(inputs.input_ids.shape[-1]),
    }
    return cache, meta


def _plot(payload: dict[str, Any], out_fig: Path) -> None:
    layers = payload["per_layer"]
    d_head = payload["meta"]["head_dim"]

    # Pick a middle layer for the per-channel comparison.
    mid = layers[len(layers) // 2]
    channels = np.arange(d_head)
    k_ch = np.asarray(mid["k_channel_max"])
    v_ch = np.asarray(mid["v_channel_max"])

    # Concentration vs layer.
    layer_ids = [row["layer_idx"] for row in layers]
    k_conc = [row["k_concentration"] for row in layers]
    v_conc = [row["v_concentration"] for row in layers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.bar(channels - 0.2, k_ch, width=0.4, label="K", color="#4C72B0")
    ax.bar(channels + 0.2, v_ch, width=0.4, label="V", color="#DD8452")
    ax.set_yscale("log")
    ax.set_xlabel("Channel index (head_dim)")
    ax.set_ylabel("max |value|  (log)")
    ax.set_title(f"Per-channel magnitude — layer {mid['layer_idx']} (middle)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(layer_ids, k_conc, marker="o", label="K", color="#4C72B0")
    ax.plot(layer_ids, v_conc, marker="s", label="V", color="#DD8452")
    ax.set_yscale("log")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("max / median per channel  (log)")
    ax.set_title("Outlier concentration per layer")
    ax.axhline(y=10, color="gray", ls="--", lw=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")

    fig.suptitle(
        f"K vs V outlier structure — {payload['meta']['model']}  on Greek text",
        fontsize=12,
    )
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-samples", type=int, default=4, help="Wikipedia articles")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, default=Path("results/kv_outliers.json"))
    ap.add_argument("--out-fig", type=Path, default=Path("results/figures/kv_outliers.png"))
    args = ap.parse_args()

    seed_all(args.seed)

    articles = load_el_wikipedia_sample(n=args.n_samples, seed=args.seed)
    text = "\n\n".join(articles)
    print(f"Loaded {len(articles)} Greek Wikipedia articles ({len(text)} chars).")

    print(f"Running {args.model} on {args.max_tokens}-token slice...")
    cache, meta = _run(args.model, text, args.max_tokens)
    print(f"  → captured {len(cache.k)} layers")

    per_layer = [
        {"layer_idx": i, **_per_layer_stats(cache.k[i], cache.v[i])} for i in sorted(cache.k)
    ]
    payload = {"meta": meta, "per_layer": per_layer}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"Wrote {args.out_json}")

    _plot(payload, args.out_fig)
    print(f"Wrote {args.out_fig}")

    print("\nTop 5 layers by K concentration:")
    for row in sorted(per_layer, key=lambda r: r["k_concentration"], reverse=True)[:5]:
        print(
            f"  layer {row['layer_idx']:>3}: K max/med = {row['k_concentration']:6.2f}  "
            f"V max/med = {row['v_concentration']:6.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
