"""T17 — Sliding-window perplexity on Greek text (CLI entry point).

Thin wrapper around :mod:`greekkv.eval.perplexity`. The sweep runner
(`scripts/run_sweep.py`, T21) reuses the same library helpers so results
produced here can be resumed from there and vice versa.

Usage
-----
::

    python experiments/perplexity.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --n-ctx 256 512 1024 \\
        --n-chunks 20 \\
        --out results/perplexity/qwen2_5_0_5b.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from greekkv.eval import LoadSpec, load_model
from greekkv.eval.perplexity import (
    DEFAULT_SCHEMES,
    chunks_from_ids,
    collect_fp16_logits,
    run_perplexity_cell,
)
from greekkv.tokenization import load_el_wikipedia_sample
from greekkv.utils import seed_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-ctx", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--n-samples", type=int, default=6, help="Wikipedia articles")
    ap.add_argument("--n-chunks", type=int, default=20)
    ap.add_argument("--ctx-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/perplexity/qwen2_5_0_5b.jsonl"),
    )
    args = ap.parse_args()

    seed_all(args.seed)

    print(f"Loading {args.model}...")
    spec = LoadSpec(name=args.model, dtype="fp32", device="auto")
    model, tok, mem = load_model(spec)
    model.eval()
    print(f"  loaded (≈{mem.total_gb:.2f} GB est.)")

    articles = load_el_wikipedia_sample(n=args.n_samples, seed=args.seed)
    text = "\n\n".join(articles)
    all_ids = tok.encode(text, add_special_tokens=False)
    print(f"Loaded {len(articles)} Greek Wikipedia articles → {len(all_ids)} tokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for n_ctx in args.n_ctx:
        chunks = chunks_from_ids(all_ids, n_ctx, args.n_chunks)
        if not chunks:
            print(f"SKIP n_ctx={n_ctx}: not enough tokens for even one chunk")
            continue
        print(f"\nn_ctx={n_ctx}  → {len(chunks)} chunks × {len(DEFAULT_SCHEMES)} schemes")

        print(f"  (collecting fp16 reference logits over {len(chunks)} chunks)")
        ctx_len_for_ref = int(n_ctx * args.ctx_frac)
        fp16_logits = collect_fp16_logits(model, chunks, ctx_len_for_ref)

        for cache_spec in DEFAULT_SCHEMES:
            fp16_ref = None if cache_spec.scheme == "fp16" else fp16_logits
            try:
                row = run_perplexity_cell(
                    model=model,
                    chunks=chunks,
                    cache_spec=cache_spec,
                    n_ctx=n_ctx,
                    ctx_frac=args.ctx_frac,
                    model_name=args.model,
                    out=args.out,
                    fp16_logits_by_chunk=fp16_ref,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {cache_spec.scheme:20s} n_ctx={n_ctx:5d}  FAILED: {e}")
                continue
            summaries.append(row)
            extra = f"  top1={row['top1']:.3f}  KL={row['kl_mean']:.3f}" if "top1" in row else ""
            print(
                f"  {row['scheme']:20s} n_ctx={row['n_ctx']:5d}  "
                f"PPL={row['ppl']:8.2f}  wall={row['wall_s']:5.1f}s{extra}"
            )

    print(f"\nWrote {len(summaries)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
