"""Fake-quantized KV cache that plugs into any HF model that accepts a ``Cache``.

The cache stores K/V as **dequantized FP16 tensors** — memory-wise it looks
like a vanilla cache, but each stored tensor has been through a lossy
quantize→dequantize step. That is enough to measure the *quality* impact of a
quantization scheme in isolation, without writing a custom kernel for packed
storage. Real packed storage (and its memory savings) is the job of T11+ and
of the T24 kernel.

Contract
--------
- ``FakeQuantCache(bits=16, ...)`` is **bit-identical** to the vanilla
  ``DynamicCache`` on every operation. This is the sanity switch used by
  every downstream eval to prove the baseline is clean.
- ``bits in {4, 8}`` applies :func:`greekkv.quant.quantize` → dequantize on
  every new K and V chunk that arrives in ``update()``. Previously-stored
  chunks are kept as they were (they were already fake-quantized when they
  were inserted), matching how a real production cache would behave.

Design notes
------------
- Subclasses ``transformers.cache_utils.DynamicCache`` so the concatenation /
  ``get_seq_length`` / beam-reorder logic is inherited for free.
- Quantization runs in FP32 to keep the per-group scale computation stable,
  then returns to the input dtype so the model receives the familiar type.
- One ``(bits, group_size, axis, symmetric)`` scheme applies to both K and V.
  KIVI-style per-K / per-V differentiation is T11's job.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from greekkv.quant.primitives import dequantize, quantize


class FakeQuantCache(DynamicCache):
    """DynamicCache that fake-quantizes each K/V chunk on ``update()``.

    Parameters
    ----------
    bits
        16 (no-op, bit-identical to vanilla), 8, or 4. Anything else raises.
    group_size
        Elements per shared scale along ``axis``. See
        :func:`greekkv.quant.quantize`.
    axis
        Which dim the group window slides along. ``-1`` means the last axis
        (head dimension for the standard ``[B, H, N, D]`` KV layout).
    symmetric
        Symmetric (zero-point implicit) vs asymmetric quantization.
    """

    def __init__(
        self,
        bits: int = 16,
        group_size: int = 128,
        axis: int = -1,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        if bits not in (4, 8, 16):
            raise ValueError(f"bits must be 4, 8, or 16, got {bits}")
        self.bits = bits
        self.group_size = group_size
        self.axis = axis
        self.symmetric = symmetric

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bits >= 16:
            # FP16/bf16 mode — no-op. Must round-trip bit-identically so the
            # baseline evaluation has no hidden quality drift.
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        key_states = self._fake_quant(key_states)
        value_states = self._fake_quant(value_states)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    def _fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize → dequantize ``x``. Returns a tensor with the original dtype."""
        orig_dtype = x.dtype
        qt = quantize(
            x.to(torch.float32),
            bits=self.bits,
            group_size=self.group_size,
            axis=self.axis,
            symmetric=self.symmetric,
        )
        y = dequantize(qt)
        return y.to(orig_dtype)
