"""T17 — Sliding-window perplexity on Greek text, swept across cache schemes.

Measures per-token NLL (and the derived perplexity) when the context is
prefilled into a **quantized** cache before scoring the continuation. For
every ``(model, scheme, n_ctx)`` triple we run ``n_chunks`` independent
windows carved out of Greek Wikipedia text, score the second half of each
window given the first half, and aggregate.

Usage
-----
::

    python experiments/perplexity.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --n-ctx 256 512 1024 \\
        --n-chunks 20 \\
        --out results/perplexity/qwen2_5_0_5b.jsonl

Output
------
One ``ResultRecord`` (T03) per ``(scheme, n_ctx)`` combination. Fields
captured in the ``metrics`` dict let the T22 report regenerate the table
without a re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

import torch

from greekkv.eval import LoadSpec, load_model
from greekkv.eval.scorer import _score_against_cache
from greekkv.quant import CacheSpec, build_cache
from greekkv.tokenization import load_el_wikipedia_sample
from greekkv.utils import ResultRecord, append_result, git_sha, seed_all

DEFAULT_SCHEMES: list[CacheSpec] = [
    CacheSpec(scheme="fp16"),
    CacheSpec(scheme="int8-fakequant", bits=8, group_size=64),
    CacheSpec(scheme="int4-fakequant", bits=4, group_size=64),
    CacheSpec(scheme="int8-kivi", bits=8, group_size=64, residual=16),
    CacheSpec(scheme="int4-kivi", bits=4, group_size=64, residual=16),
    CacheSpec(scheme="eviction-sink", n_sink=4, n_recent=32),
]


def _spec_hash(spec: CacheSpec, n_ctx: int) -> str:
    """Stable 12-char hash of the (spec, n_ctx) tuple for result provenance."""
    key = (
        f"{spec.scheme}|bits={spec.bits}|g={spec.group_size}|r={spec.residual}|"
        f"sym={int(spec.symmetric)}|sink={spec.n_sink}|recent={spec.n_recent}|ctx={n_ctx}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _chunks_from_ids(all_ids: list[int], n_ctx: int, n_chunks: int) -> list[list[int]]:
    """Carve ``n_chunks`` non-overlapping windows of length ``n_ctx`` from ``all_ids``."""
    chunks: list[list[int]] = []
    pos = 0
    for _ in range(n_chunks):
        if pos + n_ctx > len(all_ids):
            break
        chunks.append(all_ids[pos : pos + n_ctx])
        pos += n_ctx
    return chunks


def _score_chunks(
    model,
    chunks: list[list[int]],
    ctx_len: int,
    cache_spec: CacheSpec,
) -> tuple[list[float], float]:
    """Return ``(per-token NLLs across all chunks, wall-clock seconds)``."""
    device = next(model.parameters()).device
    start = time.perf_counter()
    all_nll: list[float] = []
    for chunk in chunks:
        ids = torch.tensor([chunk], dtype=torch.long, device=device)
        ctx_ids = ids[:, :ctx_len]
        cont_ids = ids[:, ctx_len:]
        cache = build_cache(cache_spec, model_config=model.config)
        res = _score_against_cache(model, ctx_ids, cont_ids, cache)
        all_nll.extend(res.nll_per_token)
    return all_nll, time.perf_counter() - start


def _run_one(
    model,
    chunks: list[list[int]],
    cache_spec: CacheSpec,
    n_ctx: int,
    ctx_frac: float,
    model_name: str,
    out: Path,
) -> dict:
    """Score one (scheme, n_ctx) combination, append the row, return a summary."""
    ctx_len = int(n_ctx * ctx_frac)
    all_nll, wall = _score_chunks(model, chunks, ctx_len, cache_spec)
    if not all_nll:
        raise RuntimeError(f"no scored tokens for scheme={cache_spec.scheme}")

    nll_mean = sum(all_nll) / len(all_nll)
    ppl = math.exp(nll_mean)

    record = ResultRecord(
        experiment="perplexity",
        model=model_name,
        scheme=cache_spec.scheme,
        git_sha=git_sha(),
        config_hash=_spec_hash(cache_spec, n_ctx),
        metrics={
            "bits": cache_spec.bits,
            "group_size": cache_spec.group_size,
            "residual": cache_spec.residual,
            "symmetric": cache_spec.symmetric,
            "n_sink": cache_spec.n_sink,
            "n_recent": cache_spec.n_recent,
            "n_ctx": n_ctx,
            "ctx_len": ctx_len,
            "cont_len": n_ctx - ctx_len,
            "n_chunks": len(chunks),
            "nll_mean": nll_mean,
            "perplexity": ppl,
            "wall_time_s": wall,
        },
        tokens=len(all_nll),
    )
    append_result(out, record)
    return {
        "scheme": cache_spec.scheme,
        "n_ctx": n_ctx,
        "ppl": ppl,
        "wall_s": wall,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-ctx", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--n-samples", type=int, default=6, help="Wikipedia articles")
    ap.add_argument("--n-chunks", type=int, default=20)
    ap.add_argument(
        "--ctx-frac",
        type=float,
        default=0.5,
        help="Fraction of each window used as context (rest is scored).",
    )
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
    print(f"  loaded (≈{mem.total_gb:.2f} GB total est.)")

    articles = load_el_wikipedia_sample(n=args.n_samples, seed=args.seed)
    text = "\n\n".join(articles)
    all_ids = tok.encode(text, add_special_tokens=False)
    print(f"Loaded {len(articles)} Greek Wikipedia articles → {len(all_ids)} tokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for n_ctx in args.n_ctx:
        chunks = _chunks_from_ids(all_ids, n_ctx, args.n_chunks)
        if not chunks:
            print(f"SKIP n_ctx={n_ctx}: not enough tokens for even one chunk")
            continue
        print(f"\nn_ctx={n_ctx}  → {len(chunks)} chunks × {len(DEFAULT_SCHEMES)} schemes")
        for cache_spec in DEFAULT_SCHEMES:
            try:
                row = _run_one(
                    model=model,
                    chunks=chunks,
                    cache_spec=cache_spec,
                    n_ctx=n_ctx,
                    ctx_frac=args.ctx_frac,
                    model_name=args.model,
                    out=args.out,
                )
            except Exception as e:  # noqa: BLE001 — record+continue on per-scheme failure
                print(f"  {cache_spec.scheme:20s} n_ctx={n_ctx:5d}  FAILED: {e}")
                continue
            summaries.append(row)
            print(
                f"  {row['scheme']:20s} n_ctx={row['n_ctx']:5d}  "
                f"PPL={row['ppl']:8.2f}  wall={row['wall_s']:5.1f}s"
            )

    print(f"\nWrote {len(summaries)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
