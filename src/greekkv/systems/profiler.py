"""Peak memory + prefill / decode latency profiler.

Measures three numbers per ``(cache scheme, n_ctx)`` pair:

- **Peak memory** during a prefill + ``n_decode`` decoding steps. On CUDA
  this is ``torch.cuda.max_memory_allocated()`` (precise, allocator-level).
  On CPU this is the max resident-set-size observed via ``psutil`` across
  three checkpoints (baseline / post-prefill / post-decode); the number
  includes the Python interpreter + torch runtime + model weights, so we
  report both the absolute peak and the delta above the baseline.
- **Prefill wall time** for one ``n_ctx``-token forward pass.
- **Per-token decode latency** as ``n_decode`` consecutive one-token
  forward passes against the (already-quantized) cache.

The measurement routine is deliberately tiny. The point is to hand the
same ``(scheme, n_ctx)`` grid used by T17/T19 a *systems* number to pair
with each quality number — this is what T28 will multiply through to
turn "quality vs bits" into "concurrent users vs bits".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import psutil
import torch

from greekkv.quant import CacheSpec, build_cache


@dataclass(frozen=True)
class ProfileResult:
    """One (scheme, n_ctx) data-point produced by :func:`profile_scheme`."""

    scheme: str
    n_ctx: int
    n_decode: int
    device: str
    peak_mem_mb: float  # absolute: cuda.max_memory_allocated OR CPU RSS
    peak_mem_delta_mb: float  # above baseline (same number on CUDA)
    prefill_wall_s: float
    decode_wall_s: float
    decode_ms_per_token: float
    decode_tokens_per_s: float


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_scheme(
    model: Any,
    cache_spec: CacheSpec,
    n_ctx: int,
    n_decode: int = 16,
    seed: int = 0,
) -> ProfileResult:
    """Prefill ``n_ctx`` random tokens, decode ``n_decode`` more, return the
    memory + latency breakdown.

    The model must already be on its target device; ``device`` is reported
    verbatim. Random tokens are fine because we're measuring *system*
    behaviour of the cache, not quality.
    """
    if n_ctx <= 0 or n_decode <= 0:
        raise ValueError(f"n_ctx={n_ctx} and n_decode={n_decode} must be positive")

    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, 100, (1, n_ctx), generator=gen, dtype=torch.long).to(device)

    cache = build_cache(cache_spec, model_config=model.config)

    # Memory baseline.
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        baseline_mb = torch.cuda.memory_allocated() / (1024**2)
    else:
        baseline_mb = _rss_mb()

    # Prefill.
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(tokens, past_key_values=cache, use_cache=True)
    _sync_if_cuda(device)
    prefill_wall = time.perf_counter() - t0
    post_prefill_rss = _rss_mb() if not is_cuda else None

    # Decode n_decode single-token steps.
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_decode):
            out = model(next_tok, past_key_values=cache, use_cache=True)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    _sync_if_cuda(device)
    decode_wall = time.perf_counter() - t0
    post_decode_rss = _rss_mb() if not is_cuda else None

    # Peak memory.
    if is_cuda:
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        peak_delta_mb = peak_mem_mb - baseline_mb
    else:
        # psutil-based peak sampled at the two checkpoints after baseline.
        assert post_prefill_rss is not None and post_decode_rss is not None
        peak_mem_mb = max(post_prefill_rss, post_decode_rss)
        peak_delta_mb = max(0.0, peak_mem_mb - baseline_mb)

    decode_ms_per_token = (decode_wall / n_decode) * 1000.0 if n_decode > 0 else 0.0
    decode_tokens_per_s = n_decode / decode_wall if decode_wall > 0 else 0.0

    return ProfileResult(
        scheme=cache_spec.scheme,
        n_ctx=n_ctx,
        n_decode=n_decode,
        device=str(device),
        peak_mem_mb=peak_mem_mb,
        peak_mem_delta_mb=peak_delta_mb,
        prefill_wall_s=prefill_wall,
        decode_wall_s=decode_wall,
        decode_ms_per_token=decode_ms_per_token,
        decode_tokens_per_s=decode_tokens_per_s,
    )
