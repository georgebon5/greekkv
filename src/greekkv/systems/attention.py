"""Fused dequantize-attention kernel for single-query decoding.

Two implementations live side-by-side:

- :func:`decode_attention_reference` — pure PyTorch. Always available, runs
  on CPU, used both at inference time on non-CUDA hardware and as the
  numerical reference for the Triton kernel.
- :func:`decode_attention_triton` — Triton kernel that fuses K/V
  dequantization, the ``Q @ K^T`` matmul, the softmax, and the
  ``softmax @ V`` accumulation into a single GPU launch. Requires CUDA +
  Triton; raises :class:`RuntimeError` on CPU.

:func:`decode_attention` is the public entry point that dispatches to the
Triton kernel on CUDA and to the reference on CPU.

Layout (matches T11 KIVI)
-------------------------
- K is quantized **per-channel** along ``head_dim`` with group size ``G``.
  ``k_data`` has shape ``[H_kv, N, D]`` and lives in ``int8``.
  ``k_scales`` has shape ``[H_kv, D // G]`` and is shared across the
  sequence axis ``N``.
- V is quantized **per-token** along ``seq`` with group size ``G``.
  ``v_data`` has shape ``[H_kv, N, D]`` and lives in ``int8``.
  ``v_scales`` has shape ``[H_kv, N // G]`` and is shared across
  ``head_dim``.

Numerical contract
------------------
The reference computes everything in FP32. Triton accumulates in FP32 too,
so the only expected gap vs. the reference is FP arithmetic noise. The T25
benchmark asserts ``max_abs_error < 1e-3`` on ``float16`` inputs; see the
docstring of :func:`decode_attention_triton`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


def _check_shapes(
    q: torch.Tensor,
    k_data: torch.Tensor,
    k_scales: torch.Tensor,
    v_data: torch.Tensor,
    v_scales: torch.Tensor,
    group_size: int,
) -> tuple[int, int, int, int, int]:
    """Validate the KIVI-layout shapes and return (H, H_kv, N, D, G).

    Batch dim is implicitly 1 — the single-query decode path never batches
    beyond that on free-tier hardware.
    """
    if q.ndim != 2:
        raise ValueError(f"q must be 2D [H, D], got shape {tuple(q.shape)}")
    if k_data.ndim != 3 or v_data.ndim != 3:
        raise ValueError(
            f"k/v_data must be 3D [H_kv, N, D]; got k={tuple(k_data.shape)} v={tuple(v_data.shape)}"
        )
    h, d = q.shape
    h_kv, n, d_kv = k_data.shape
    if d_kv != d:
        raise ValueError(f"head_dim mismatch: q D={d} k D={d_kv}")
    if v_data.shape != k_data.shape:
        raise ValueError(
            f"v_data shape {tuple(v_data.shape)} != k_data shape {tuple(k_data.shape)}"
        )
    if h % h_kv != 0:
        raise ValueError(f"H={h} must be divisible by H_kv={h_kv} for GQA mapping")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if d % group_size != 0:
        raise ValueError(f"D={d} must be divisible by group_size={group_size}")
    if n % group_size != 0:
        raise ValueError(f"N={n} must be divisible by group_size={group_size}")
    if k_scales.shape != (h_kv, d // group_size):
        raise ValueError(
            f"k_scales shape {tuple(k_scales.shape)} != expected {(h_kv, d // group_size)}"
        )
    if v_scales.shape != (h_kv, n // group_size):
        raise ValueError(
            f"v_scales shape {tuple(v_scales.shape)} != expected {(h_kv, n // group_size)}"
        )
    return h, h_kv, n, d, group_size


def _dequantize_k(k_data: torch.Tensor, k_scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """K dequant: per-channel scales shared across the sequence axis.

    ``k_data[h, n, d]  →  k_data[h, n, d] * k_scales[h, d // G]`` in FP32.
    """
    h_kv, n, d = k_data.shape
    # Expand scales from [H_kv, D/G] → [H_kv, 1, D] by repeat-interleave.
    s = k_scales.to(torch.float32).repeat_interleave(group_size, dim=-1)  # [H_kv, D]
    return k_data.to(torch.float32) * s.unsqueeze(1)


def _dequantize_v(v_data: torch.Tensor, v_scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """V dequant: per-token scales shared across head_dim.

    ``v_data[h, n, d]  →  v_data[h, n, d] * v_scales[h, n // G]`` in FP32.
    """
    h_kv, n, d = v_data.shape
    s = v_scales.to(torch.float32).repeat_interleave(group_size, dim=-1)  # [H_kv, N]
    return v_data.to(torch.float32) * s.unsqueeze(-1)


def decode_attention_reference(
    q: torch.Tensor,
    k_data: torch.Tensor,
    k_scales: torch.Tensor,
    v_data: torch.Tensor,
    v_scales: torch.Tensor,
    group_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Pure-PyTorch reference: dequant → ``Q @ K^T`` → softmax → ``softmax @ V``.

    Parameters
    ----------
    q
        ``[H, D]`` single query (batch = 1 collapsed).
    k_data, k_scales
        ``[H_kv, N, D]`` int8 and ``[H_kv, D // G]`` scales (KIVI per-channel).
    v_data, v_scales
        ``[H_kv, N, D]`` int8 and ``[H_kv, N // G]`` scales (KIVI per-token).
    group_size
        Quantization group size (must divide both ``D`` and ``N``).
    scale
        Pre-softmax multiplier. Defaults to ``1 / sqrt(D)``.
    """
    h, h_kv, n, d, _ = _check_shapes(q, k_data, k_scales, v_data, v_scales, group_size)
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    k = _dequantize_k(k_data, k_scales, group_size)  # [H_kv, N, D] fp32
    v = _dequantize_v(v_data, v_scales, group_size)  # [H_kv, N, D] fp32

    # GQA mapping: Q head h attends to K/V head h // (H / H_kv).
    kv_ratio = h // h_kv
    out = torch.empty((h, d), dtype=torch.float32, device=q.device)
    q_f = q.to(torch.float32)
    for hi in range(h):
        hk = hi // kv_ratio
        scores = (q_f[hi] @ k[hk].T) * scale  # [N]
        weights = F.softmax(scores, dim=-1)
        out[hi] = weights @ v[hk]  # [D]
    return out


def decode_attention_triton(
    q: torch.Tensor,
    k_data: torch.Tensor,
    k_scales: torch.Tensor,
    v_data: torch.Tensor,
    v_scales: torch.Tensor,
    group_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Triton kernel path. Requires a CUDA device + ``triton`` installed.

    The kernel fuses:
    1. **K dequantization** by multiplying each int8 column with the FP16
       per-channel scale that lives in ``k_scales``.
    2. The ``Q @ K^T`` dot product, accumulated in FP32.
    3. Running-max + running-sum online softmax (FlashAttention-style) so
       the full ``[N]`` attention vector never has to be materialised.
    4. **V dequantization** on the fly while accumulating the weighted sum.

    Numerical tolerance against :func:`decode_attention_reference`:
    ``max_abs_error < 1e-3`` for FP16 inputs with normalised magnitudes.
    The T25 benchmark asserts this in CI on T4.
    """
    if q.device.type != "cuda":
        raise RuntimeError(
            f"decode_attention_triton requires a CUDA device, got {q.device!r}. "
            "Use decode_attention_reference() or the public decode_attention() wrapper."
        )
    try:
        import triton  # noqa: F401, PLC0415
        import triton.language as tl  # noqa: F401, PLC0415
    except ImportError as e:
        raise ModuleNotFoundError(
            "decode_attention_triton requires `triton` (bundled with torch on "
            "most Linux + CUDA builds). Install via `pip install triton`."
        ) from e

    h, h_kv, n, d, g = _check_shapes(q, k_data, k_scales, v_data, v_scales, group_size)
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    out = torch.empty((h, d), dtype=torch.float32, device=q.device)
    kv_ratio = h // h_kv
    block_n = min(64, n)  # keep per-block tile small for free-tier T4

    # Launch one program instance per query head.
    grid = (h,)
    _fused_decode_attention_kernel[grid](  # type: ignore[index]
        q,
        k_data,
        k_scales,
        v_data,
        v_scales,
        out,
        scale,
        n,
        d,
        g,
        kv_ratio,
        block_n,
    )
    return out


def decode_attention(
    q: torch.Tensor,
    k_data: torch.Tensor,
    k_scales: torch.Tensor,
    v_data: torch.Tensor,
    v_scales: torch.Tensor,
    group_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Dispatch: Triton on CUDA, pure-PyTorch reference elsewhere."""
    if q.device.type == "cuda":
        return decode_attention_triton(q, k_data, k_scales, v_data, v_scales, group_size, scale)
    return decode_attention_reference(q, k_data, k_scales, v_data, v_scales, group_size, scale)


# ---------------------------------------------------------------------------
# The Triton kernel. Only imported when CUDA is available — guarded so
# `import greekkv.systems.attention` stays safe on CPU-only hosts.
# ---------------------------------------------------------------------------


_fused_decode_attention_kernel: Any = None


def _install_triton_kernel() -> None:
    """Build the Triton kernel at first use. No-op on CPU-only hosts."""
    global _fused_decode_attention_kernel
    if _fused_decode_attention_kernel is not None:
        return
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return

    @triton.jit
    def _kernel(
        q_ptr,
        k_ptr,
        k_scales_ptr,
        v_ptr,
        v_scales_ptr,
        out_ptr,
        scale,
        N: tl.constexpr,  # noqa: N803
        D: tl.constexpr,  # noqa: N803
        G: tl.constexpr,  # noqa: N803
        KV_RATIO: tl.constexpr,  # noqa: N803
        BLOCK_N: tl.constexpr,  # noqa: N803
    ):
        hi = tl.program_id(0)
        hk = hi // KV_RATIO
        # Load Q head (shape [D]).
        d_offs = tl.arange(0, D)
        q = tl.load(q_ptr + hi * D + d_offs).to(tl.float32)

        # Expanded per-channel K scale for this head: scales[hk, d // G].
        k_scale_offs = hk * (D // G) + d_offs // G
        k_scale = tl.load(k_scales_ptr + k_scale_offs).to(tl.float32)  # [D]

        # Online softmax accumulators.
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((D,), dtype=tl.float32)

        # Walk over N in BLOCK_N tiles.
        for n_start in range(0, N, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            mask = n_offs < N

            # Load K tile: [BLOCK_N, D] int8, dequantize to fp32.
            k_tile = tl.load(
                k_ptr + hk * N * D + n_offs[:, None] * D + d_offs[None, :],
                mask=mask[:, None],
                other=0,
            ).to(tl.float32)
            k_tile = k_tile * k_scale[None, :]

            # Scores: q @ k_tile.T → [BLOCK_N]
            scores = tl.sum(q[None, :] * k_tile, axis=1) * scale
            scores = tl.where(mask, scores, -float("inf"))

            # V tile + per-token scale.
            v_scale_offs = hk * (N // G) + n_offs // G
            v_scale = tl.load(v_scales_ptr + v_scale_offs, mask=mask, other=0).to(tl.float32)
            v_tile = tl.load(
                v_ptr + hk * N * D + n_offs[:, None] * D + d_offs[None, :],
                mask=mask[:, None],
                other=0,
            ).to(tl.float32)
            v_tile = v_tile * v_scale[:, None]

            # Online softmax: new_max, rescale accumulator, add contribution.
            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_new = alpha * l_i + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
            m_i = m_new
            l_i = l_new

        out_offs = hi * D + d_offs
        tl.store(out_ptr + out_offs, acc / l_i)

    _fused_decode_attention_kernel = _kernel


# Install on import — becomes a no-op on CPU-only hosts.
_install_triton_kernel()
