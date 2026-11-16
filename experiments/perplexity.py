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

from greekkv.eval import LoadSpec, compute_fidelity, load_model
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
    fp16_logits_by_chunk: dict[int, torch.Tensor] | None = None,
) -> tuple[list[float], float, list[dict] | None]:
    """Return ``(per-token NLLs across all chunks, wall-clock seconds, per-chunk fidelity or None)``.

    If ``fp16_logits_by_chunk`` is provided, each chunk is scored with
    ``return_logits=True`` and a fidelity record is computed per chunk against
    the stored FP16 logits. ``None`` means "this is the FP16 pass — collect
    the logits for later schemes to compare against".
    """
    device = next(model.parameters()).device
    start = time.perf_counter()
    all_nll: list[float] = []
    fidelities: list[dict] | None = [] if fp16_logits_by_chunk else None
    want_logits = fp16_logits_by_chunk is not None
    for i, chunk in enumerate(chunks):
        ids = torch.tensor([chunk], dtype=torch.long, device=device)
        ctx_ids = ids[:, :ctx_len]
        cont_ids = ids[:, ctx_len:]
        cache = build_cache(cache_spec, model_config=model.config)
        if want_logits:
            res, logits_q = _score_against_cache(
                model, ctx_ids, cont_ids, cache, return_logits=True
            )
            fid = compute_fidelity(fp16_logits_by_chunk[i], logits_q)
            fidelities.append(
                {
                    "kl_mean": fid.kl_mean,
                    "kl_p95": fid.kl_p95,
                    "top1_agree": fid.top1_agree,
                    "top5_agree": fid.top5_agree,
                    "n_positions": fid.n_positions,
                }
            )
        else:
            res = _score_against_cache(model, ctx_ids, cont_ids, cache)
        all_nll.extend(res.nll_per_token)
    return all_nll, time.perf_counter() - start, fidelities


def _collect_fp16_logits(
    model,
    chunks: list[list[int]],
    ctx_len: int,
) -> dict[int, torch.Tensor]:
    """Score every chunk once with a vanilla ``DynamicCache`` and keep the per-position logits.

    These are reused as the fidelity reference for every subsequent scheme,
    so the only overhead vs. the T17 original run is the storage of the
    fp16 logits — the fp16 pass was already a required scheme.
    """
    from transformers.cache_utils import DynamicCache  # noqa: PLC0415

    device = next(model.parameters()).device
    out: dict[int, torch.Tensor] = {}
    for i, chunk in enumerate(chunks):
        ids = torch.tensor([chunk], dtype=torch.long, device=device)
        ctx_ids = ids[:, :ctx_len]
        cont_ids = ids[:, ctx_len:]
        _, logits = _score_against_cache(
            model, ctx_ids, cont_ids, DynamicCache(), return_logits=True
        )
        out[i] = logits
    return out


def _run_one(
    model,
    chunks: list[list[int]],
    cache_spec: CacheSpec,
    n_ctx: int,
    ctx_frac: float,
    model_name: str,
    out: Path,
    fp16_logits_by_chunk: dict[int, torch.Tensor] | None = None,
) -> dict:
    """Score one (scheme, n_ctx) combination, append the row, return a summary.

    When ``fp16_logits_by_chunk`` is supplied, per-chunk fidelity is computed
    against the stored FP16 logits and averaged into the row.
    """
    ctx_len = int(n_ctx * ctx_frac)
    all_nll, wall, fidelities = _score_chunks(
        model, chunks, ctx_len, cache_spec, fp16_logits_by_chunk
    )
    if not all_nll:
        raise RuntimeError(f"no scored tokens for scheme={cache_spec.scheme}")

    nll_mean = sum(all_nll) / len(all_nll)
    ppl = math.exp(nll_mean)

    metrics = {
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
    }
    if fidelities:
        # Equal-weight average across chunks (same cont_len per chunk).
        avg = {
            k: sum(f[k] for f in fidelities) / len(fidelities)
            for k in ("kl_mean", "kl_p95", "top1_agree", "top5_agree")
        }
        metrics.update(avg)
    elif fp16_logits_by_chunk is None and cache_spec.scheme == "fp16":
        # FP16 vs itself — fidelity is trivially perfect.
        metrics.update(kl_mean=0.0, kl_p95=0.0, top1_agree=1.0, top5_agree=1.0)

    record = ResultRecord(
        experiment="perplexity",
        model=model_name,
        scheme=cache_spec.scheme,
        git_sha=git_sha(),
        config_hash=_spec_hash(cache_spec, n_ctx),
        metrics=metrics,
        tokens=len(all_nll),
    )
    append_result(out, record)
    summary = {"scheme": cache_spec.scheme, "n_ctx": n_ctx, "ppl": ppl, "wall_s": wall}
    if "top1_agree" in metrics:
        summary["top1"] = metrics["top1_agree"]
        summary["kl_mean"] = metrics["kl_mean"]
    return summary


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

        # Collect FP16 logits once per n_ctx — reused by every non-fp16 scheme
        # as the fidelity reference. Only extra cost vs the original T17 run
        # is the storage of these logits.
        print(f"  (collecting fp16 reference logits over {len(chunks)} chunks for fidelity)")
        ctx_len_for_ref = int(n_ctx * args.ctx_frac)
        fp16_logits = _collect_fp16_logits(model, chunks, ctx_len_for_ref)

        for cache_spec in DEFAULT_SCHEMES:
            fp16_ref = None if cache_spec.scheme == "fp16" else fp16_logits
            try:
                row = _run_one(
                    model=model,
                    chunks=chunks,
                    cache_spec=cache_spec,
                    n_ctx=n_ctx,
                    ctx_frac=args.ctx_frac,
                    model_name=args.model,
                    out=args.out,
                    fp16_logits_by_chunk=fp16_ref,
                )
            except Exception as e:  # noqa: BLE001 — record+continue on per-scheme failure
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
