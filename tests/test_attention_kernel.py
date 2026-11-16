"""Tests for greekkv.systems.attention.

The reference path runs on CPU and is tested comprehensively against a
naive dequant-then-standard-attention baseline. The Triton path requires
CUDA + Triton; those tests are skipped on hosts without both.
"""

from __future__ import annotations

import math

import pytest
import torch

from greekkv.systems.attention import (
    decode_attention,
    decode_attention_reference,
    decode_attention_triton,
)


def _rand_fp32(shape, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32)


def _build_quant_kv(
    h_kv: int, n: int, d: int, group_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate an int8-quantized K, V pair with KIVI-layout scales."""
    k = _rand_fp32((h_kv, n, d), seed=seed)
    v = _rand_fp32((h_kv, n, d), seed=seed + 1)

    # K per-channel (over D, shared across N): compute max over N + within-group.
    k_grouped = k.abs().reshape(h_kv, n, d // group_size, group_size)
    k_max = k_grouped.amax(dim=(1, 3))  # [H_kv, D/G]
    k_scales = (k_max / 127.0).clamp_min(1e-6)  # [H_kv, D/G]
    k_scales_expanded = k_scales.repeat_interleave(group_size, dim=-1)  # [H_kv, D]
    k_data = (k / k_scales_expanded.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)

    # V per-token (over N, shared across D): max over D + within-group.
    v_grouped = v.abs().reshape(h_kv, n // group_size, group_size, d)
    v_max = v_grouped.amax(dim=(2, 3))  # [H_kv, N/G]
    v_scales = (v_max / 127.0).clamp_min(1e-6)  # [H_kv, N/G]
    v_scales_expanded = v_scales.repeat_interleave(group_size, dim=-1)  # [H_kv, N]
    v_data = (v / v_scales_expanded.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)

    return k_data, k_scales, v_data, v_scales


def _naive_attention(
    q: torch.Tensor,
    k_data: torch.Tensor,
    k_scales: torch.Tensor,
    v_data: torch.Tensor,
    v_scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Independent dequant-then-attention computed with PyTorch SDPA semantics."""
    h, d = q.shape
    h_kv, n, _ = k_data.shape
    k_s = k_scales.to(torch.float32).repeat_interleave(group_size, dim=-1)
    v_s = v_scales.to(torch.float32).repeat_interleave(group_size, dim=-1)
    k = k_data.to(torch.float32) * k_s.unsqueeze(1)
    v = v_data.to(torch.float32) * v_s.unsqueeze(-1)
    out = torch.empty((h, d), dtype=torch.float32)
    kv_ratio = h // h_kv
    scale = 1.0 / math.sqrt(d)
    for hi in range(h):
        hk = hi // kv_ratio
        scores = (q[hi] @ k[hk].T) * scale
        w = torch.softmax(scores.to(torch.float32), dim=-1)
        out[hi] = w @ v[hk]
    return out


# ---------------------------------------------------------------------------
# Reference matches independent naive implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "h, h_kv, n, d, group_size",
    [
        (2, 2, 32, 16, 8),  # non-GQA
        (4, 2, 64, 32, 16),  # GQA 2:1
        (8, 1, 32, 16, 8),  # heavy GQA 8:1
        (4, 4, 128, 64, 64),  # full-group head_dim
    ],
)
def test_reference_matches_naive_baseline(h, h_kv, n, d, group_size):
    q = _rand_fp32((h, d), seed=0)
    k_data, k_scales, v_data, v_scales = _build_quant_kv(h_kv, n, d, group_size, seed=1)
    ref = decode_attention_reference(q, k_data, k_scales, v_data, v_scales, group_size)
    naive = _naive_attention(q, k_data, k_scales, v_data, v_scales, group_size)
    torch.testing.assert_close(ref, naive, atol=1e-5, rtol=1e-5)


def test_reference_output_shape_matches_q():
    q = _rand_fp32((4, 32), seed=0)
    k_data, k_scales, v_data, v_scales = _build_quant_kv(2, 32, 32, 8, seed=1)
    out = decode_attention_reference(q, k_data, k_scales, v_data, v_scales, 8)
    assert out.shape == q.shape


def test_reference_is_deterministic():
    q = _rand_fp32((2, 16), seed=0)
    k_data, k_scales, v_data, v_scales = _build_quant_kv(2, 16, 16, 8, seed=1)
    a = decode_attention_reference(q, k_data, k_scales, v_data, v_scales, 8)
    b = decode_attention_reference(q, k_data, k_scales, v_data, v_scales, 8)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


def test_reference_rejects_mismatched_head_dim():
    q = _rand_fp32((2, 16), seed=0)
    k_data = torch.zeros(2, 32, 32, dtype=torch.int8)
    k_scales = torch.ones(2, 32 // 8, dtype=torch.float32)
    v_data = torch.zeros(2, 32, 32, dtype=torch.int8)
    v_scales = torch.ones(2, 32 // 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="head_dim"):
        decode_attention_reference(q, k_data, k_scales, v_data, v_scales, 8)


def test_reference_rejects_bad_gqa_ratio():
    q = _rand_fp32((3, 16), seed=0)  # 3 heads
    k_data = torch.zeros(2, 16, 16, dtype=torch.int8)  # 2 kv heads, 3 not divisible by 2
    k_scales = torch.ones(2, 16 // 8, dtype=torch.float32)
    v_data = torch.zeros(2, 16, 16, dtype=torch.int8)
    v_scales = torch.ones(2, 16 // 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="divisible"):
        decode_attention_reference(q, k_data, k_scales, v_data, v_scales, 8)


def test_reference_rejects_bad_group_size():
    q = _rand_fp32((2, 10), seed=0)  # D=10 not divisible by G=8
    k_data = torch.zeros(2, 16, 10, dtype=torch.int8)
    k_scales = torch.ones(2, 10 // 8, dtype=torch.float32)  # intentionally wrong
    v_data = torch.zeros(2, 16, 10, dtype=torch.int8)
    v_scales = torch.ones(2, 16 // 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="divisible"):
        decode_attention_reference(q, k_data, k_scales, v_data, v_scales, 8)


# ---------------------------------------------------------------------------
# Public wrapper dispatches to reference on CPU
# ---------------------------------------------------------------------------


def test_decode_attention_uses_reference_on_cpu():
    q = _rand_fp32((2, 16), seed=0)
    k_data, k_scales, v_data, v_scales = _build_quant_kv(2, 32, 16, 8, seed=1)
    a = decode_attention(q, k_data, k_scales, v_data, v_scales, 8)
    b = decode_attention_reference(q, k_data, k_scales, v_data, v_scales, 8)
    assert torch.equal(a, b)


def test_triton_path_raises_on_cpu():
    q = _rand_fp32((2, 16), seed=0)
    k_data, k_scales, v_data, v_scales = _build_quant_kv(2, 32, 16, 8, seed=1)
    with pytest.raises(RuntimeError, match="CUDA"):
        decode_attention_triton(q, k_data, k_scales, v_data, v_scales, 8)


# ---------------------------------------------------------------------------
# Triton kernel — runs only on CUDA
# ---------------------------------------------------------------------------


_cuda_available = torch.cuda.is_available()
try:  # noqa: SIM105 — pytest.mark.skipif argument has to be a module-level name
    import triton  # noqa: F401

    _triton_available = True
except ImportError:
    _triton_available = False


@pytest.mark.skipif(
    not (_cuda_available and _triton_available),
    reason="CUDA + Triton required for kernel comparison",
)
def test_triton_matches_reference_fp16():  # pragma: no cover — CPU CI skips
    q = _rand_fp32((4, 64), seed=0).to(torch.float16).cuda()
    k_data, k_scales, v_data, v_scales = _build_quant_kv(2, 128, 64, 64, seed=1)
    k_data = k_data.cuda()
    v_data = v_data.cuda()
    k_scales = k_scales.to(torch.float16).cuda()
    v_scales = v_scales.to(torch.float16).cuda()

    ref = decode_attention_reference(
        q.cpu().to(torch.float32),
        k_data.cpu(),
        k_scales.cpu(),
        v_data.cpu(),
        v_scales.cpu(),
        64,
    )
    triton_out = decode_attention_triton(q, k_data, k_scales, v_data, v_scales, 64).cpu()
    torch.testing.assert_close(triton_out, ref, atol=1e-3, rtol=1e-3)
