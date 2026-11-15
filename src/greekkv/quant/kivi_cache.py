"""KIVI-style KV cache: per-channel K, per-token V, FP16 residual window.

Design sketch (from the KIVI paper)
-----------------------------------
- **Keys** quantized **per-channel** with a group size ``G`` along the
  head-dimension. The scale is *shared across sequence positions* within a
  channel group. Rationale: K tends to have the same small set of outlier
  channels regardless of position; sharing the scale across positions keeps
  those outliers from poisoning other channels' scales.
- **Values** quantized **per-token** with a group size ``G`` along the
  sequence dim. The scale is *shared across head_dim* within a token group.
  Rationale: V distributions vary strongly by position; per-token scales
  track that variation.
- **Residual FP16 window.** The most-recent ``R`` tokens are kept in FP16
  without quantization — attention weights are highest on recent tokens, so
  lossless storage preserves accuracy there. When new tokens push the window
  past ``R``, the *oldest* tokens in the window are evicted into the
  quantized store one chunk at a time.

Scope
-----
This is a **fake-quant** implementation: K/V are stored as FP16 tensors with
a quantize→dequantize pass applied at eviction time. Real packed storage is
left for the T24 kernel; measuring the *quality* impact of KIVI needs only
the fake-quant path, and keeps the code simple + CPU-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from transformers.cache_utils import DynamicCache

_EPS = 1e-8


@dataclass(frozen=True)
class KiviCacheConfig:
    """Hyper-parameters for :class:`KiviCache`.

    - ``bits=16`` is a no-op that makes :class:`KiviCache` behave bit-identically
      to the vanilla ``DynamicCache``.
    - ``residual`` is the FP16 window size in tokens.
    """

    bits: int = 4
    group_size: int = 128
    residual: int = 32
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.bits not in (4, 8, 16):
            raise ValueError(f"bits must be 4, 8, or 16, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.residual < 0:
            raise ValueError(f"residual must be non-negative, got {self.residual}")


class KiviCache(DynamicCache):
    """DynamicCache that applies KIVI-style fake-quant at eviction time."""

    def __init__(self, cfg: KiviCacheConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # layer_idx → number of leading tokens already fake-quantized.
        self._q_len: dict[int, int] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        if self.cfg.bits >= 16:
            return k, v

        seq_len = k.shape[-2]
        new_q_end = max(0, seq_len - self.cfg.residual)
        q_end = self._q_len.get(layer_idx, 0)
        if new_q_end <= q_end:
            # Residual still covers everything new; nothing to evict yet.
            return k, v

        # The returned tensor *is* the layer-cache tensor (verified in tests),
        # so in-place slice assignment propagates to the stored cache.
        k_slice = k[..., q_end:new_q_end, :]
        v_slice = v[..., q_end:new_q_end, :]
        k[..., q_end:new_q_end, :] = _fake_quant_per_channel(
            k_slice, self.cfg.bits, self.cfg.group_size, self.cfg.symmetric
        )
        v[..., q_end:new_q_end, :] = _fake_quant_per_token(
            v_slice, self.cfg.bits, self.cfg.group_size, self.cfg.symmetric
        )
        self._q_len[layer_idx] = new_q_end
        return k, v


# ---------------------------------------------------------------------------
# Shared-scale fake-quant helpers (KIVI layout).
#
# Both take a tensor shaped ``[..., N, D]`` where ``N`` is the sequence and
# ``D`` is the head dimension. They return a tensor of the same shape and
# dtype, with values transformed through a lossy quantize→dequantize step
# using the KIVI sharing pattern.
# ---------------------------------------------------------------------------


def _fake_quant_per_channel(
    x: torch.Tensor, bits: int, group_size: int, symmetric: bool
) -> torch.Tensor:
    """Group along D (head_dim); share the scale across N (sequence)."""
    orig_dtype = x.dtype
    d = x.shape[-1]
    pad = (-d) % group_size
    if pad:
        x = F.pad(x, (0, pad))
    d_pad = x.shape[-1]

    xf = x.to(torch.float32)
    # Shape → [..., N, n_groups, G]
    xg = xf.reshape(*xf.shape[:-1], d_pad // group_size, group_size)

    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        max_abs = xg.abs().amax(dim=-3, keepdim=True).amax(dim=-1, keepdim=True).clamp_min(_EPS)
        scale = max_abs / qmax
        q = (xg / scale).round().clamp_(-qmax, qmax)
        xq = q * scale
    else:
        qmax = (1 << bits) - 1
        x_min = xg.amin(dim=-3, keepdim=True).amin(dim=-1, keepdim=True)
        x_max = xg.amax(dim=-3, keepdim=True).amax(dim=-1, keepdim=True)
        scale = ((x_max - x_min) / qmax).clamp_min(_EPS)
        zp = (-x_min / scale).round().clamp_(0, qmax)
        q = (xg / scale + zp).round().clamp_(0, qmax)
        xq = (q - zp) * scale

    y = xq.reshape(*xf.shape[:-1], d_pad)
    y = y[..., :d]
    return y.to(orig_dtype)


def _fake_quant_per_token(
    x: torch.Tensor, bits: int, group_size: int, symmetric: bool
) -> torch.Tensor:
    """Group along N (sequence); share the scale across D (head_dim)."""
    orig_dtype = x.dtype
    n = x.shape[-2]
    # F.pad last two dims: (left_D, right_D, left_N, right_N)
    pad = (-n) % group_size
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    n_pad = x.shape[-2]
    d = x.shape[-1]

    xf = x.to(torch.float32)
    # Shape → [..., n_groups, G, D]
    xg = xf.reshape(*xf.shape[:-2], n_pad // group_size, group_size, d)

    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        max_abs = xg.abs().amax(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).clamp_min(_EPS)
        scale = max_abs / qmax
        q = (xg / scale).round().clamp_(-qmax, qmax)
        xq = q * scale
    else:
        qmax = (1 << bits) - 1
        x_min = xg.amin(dim=-1, keepdim=True).amin(dim=-2, keepdim=True)
        x_max = xg.amax(dim=-1, keepdim=True).amax(dim=-2, keepdim=True)
        scale = ((x_max - x_min) / qmax).clamp_min(_EPS)
        zp = (-x_min / scale).round().clamp_(0, qmax)
        q = (xg / scale + zp).round().clamp_(0, qmax)
        xq = (q - zp) * scale

    y = xq.reshape(*xf.shape[:-2], n_pad, d)
    y = y[..., :n, :]
    return y.to(orig_dtype)
