"""Simplified token-eviction cache (attention-sink / StreamingLLM style).

This is a **deliberate simplification** of H2O / SnapKV / StreamingLLM. It
keeps the first ``n_sink`` positions and the last ``n_recent`` positions,
and **zeros out** the K/V at positions in between. It does **not** shrink
the underlying tensor and therefore does **not** save memory — the goal is
to isolate the *quality* impact of "discard middle tokens" so it can be
compared head-to-head with the precision-reduction schemes of T09–T12.

What it skips vs the real papers
--------------------------------
- No importance scoring from attention weights (H2O uses heavy-hitters).
- No per-head / per-layer budgets (SnapKV picks a budget per layer).
- No actual tensor shrinking (which would require re-rotating RoPE embeddings
  for the surviving positions).

These are **real quality knobs** in the original papers and leaving them out
likely over-penalises this baseline. That is precisely why its scheme name
in the eval harness is ``eviction-sink`` — the "sink" prefix tells the
reader this is the simplest shape of the family, not a faithful
reimplementation.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache


class StreamingEvictionCache(DynamicCache):
    """Zero K/V at positions outside ``[0:n_sink] ∪ [-n_recent:]`` after update.

    Shapes and tensor identities are preserved (we mutate the layer-cache in
    place via the returned tensor, same trick as :class:`KiviCache`), so the
    model's attention still iterates over all positions — the zeroed ones
    simply contribute zero to the weighted sum.
    """

    def __init__(self, n_sink: int = 4, n_recent: int = 32) -> None:
        super().__init__()
        if n_sink < 0:
            raise ValueError(f"n_sink must be non-negative, got {n_sink}")
        if n_recent < 0:
            raise ValueError(f"n_recent must be non-negative, got {n_recent}")
        self.n_sink = n_sink
        self.n_recent = n_recent

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        total = k.shape[-2]
        budget = self.n_sink + self.n_recent
        if total <= budget:
            # Everything fits inside sink + recent → nothing to evict.
            return k, v

        # In-place zero of the middle slice. The returned tensor is the layer
        # cache itself (same identity trick as KiviCache), so the write
        # propagates to storage.
        middle_end = total - self.n_recent
        k[..., self.n_sink : middle_end, :] = 0
        v[..., self.n_sink : middle_end, :] = 0
        return k, v
