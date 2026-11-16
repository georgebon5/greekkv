"""T17 perplexity + T20 fidelity as reusable library helpers.

The CLI entry point (`experiments/perplexity.py`) and the T21 sweep runner
(`scripts/run_sweep.py`) both build on these functions. Keeping them here
means the sweep runner can **resume an interrupted experiment** by hashing
the same ``(scheme, n_ctx)`` key as the original run.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from greekkv.eval.fidelity import compute_fidelity
from greekkv.eval.scorer import _score_against_cache
from greekkv.quant import CacheSpec, build_cache
from greekkv.utils import ResultRecord, append_result, git_sha

DEFAULT_SCHEMES: list[CacheSpec] = [
    CacheSpec(scheme="fp16"),
    CacheSpec(scheme="int8-fakequant", bits=8, group_size=64),
    CacheSpec(scheme="int4-fakequant", bits=4, group_size=64),
    CacheSpec(scheme="int8-kivi", bits=8, group_size=64, residual=16),
    CacheSpec(scheme="int4-kivi", bits=4, group_size=64, residual=16),
    CacheSpec(scheme="eviction-sink", n_sink=4, n_recent=32),
]


def spec_hash(spec: CacheSpec, n_ctx: int) -> str:
    """Stable 12-char hash of the ``(spec, n_ctx)`` tuple for result provenance.

    The sweep runner uses this to decide whether a cell has already been
    scored — the JSONL is deduped by this hash.
    """
    key = (
        f"{spec.scheme}|bits={spec.bits}|g={spec.group_size}|r={spec.residual}|"
        f"sym={int(spec.symmetric)}|sink={spec.n_sink}|recent={spec.n_recent}|ctx={n_ctx}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def chunks_from_ids(all_ids: list[int], n_ctx: int, n_chunks: int) -> list[list[int]]:
    """Carve ``n_chunks`` non-overlapping windows of length ``n_ctx`` from ``all_ids``."""
    chunks: list[list[int]] = []
    pos = 0
    for _ in range(n_chunks):
        if pos + n_ctx > len(all_ids):
            break
        chunks.append(all_ids[pos : pos + n_ctx])
        pos += n_ctx
    return chunks


def collect_fp16_logits(model, chunks: list[list[int]], ctx_len: int) -> dict[int, torch.Tensor]:
    """Score every chunk with a vanilla ``DynamicCache`` and keep per-position logits."""
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


def _score_chunks(
    model,
    chunks: list[list[int]],
    ctx_len: int,
    cache_spec: CacheSpec,
    fp16_logits_by_chunk: dict[int, torch.Tensor] | None,
) -> tuple[list[float], float, list[dict] | None]:
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


def run_perplexity_cell(
    model,
    chunks: list[list[int]],
    cache_spec: CacheSpec,
    n_ctx: int,
    ctx_frac: float,
    model_name: str,
    out: Path,
    fp16_logits_by_chunk: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Score one ``(scheme, n_ctx)`` cell, append the row, return a summary dict.

    When ``fp16_logits_by_chunk`` is supplied, per-chunk fidelity is computed
    and averaged into the row. ``fp16_logits_by_chunk=None`` + scheme ``fp16``
    emits the trivial-fidelity row.
    """
    ctx_len = int(n_ctx * ctx_frac)
    all_nll, wall, fidelities = _score_chunks(
        model, chunks, ctx_len, cache_spec, fp16_logits_by_chunk
    )
    if not all_nll:
        raise RuntimeError(f"no scored tokens for scheme={cache_spec.scheme}")

    nll_mean = sum(all_nll) / len(all_nll)
    ppl = math.exp(nll_mean)

    metrics: dict[str, Any] = {
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
        avg = {
            k: sum(f[k] for f in fidelities) / len(fidelities)
            for k in ("kl_mean", "kl_p95", "top1_agree", "top5_agree")
        }
        metrics.update(avg)
    elif fp16_logits_by_chunk is None and cache_spec.scheme == "fp16":
        metrics.update(kl_mean=0.0, kl_p95=0.0, top1_agree=1.0, top5_agree=1.0)

    record = ResultRecord(
        experiment="perplexity",
        model=model_name,
        scheme=cache_spec.scheme,
        git_sha=git_sha(),
        config_hash=spec_hash(cache_spec, n_ctx),
        metrics=metrics,
        tokens=len(all_nll),
    )
    append_result(out, record)
    summary = {
        "scheme": cache_spec.scheme,
        "n_ctx": n_ctx,
        "ppl": ppl,
        "wall_s": wall,
    }
    if "top1_agree" in metrics:
        summary["top1"] = metrics["top1_agree"]
        summary["kl_mean"] = metrics["kl_mean"]
    return summary
