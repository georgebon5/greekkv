"""Distribution-level fidelity metrics vs an FP16 reference.

Perplexity measures the absolute quality of the quantized cache; fidelity
measures *how much the model's output distribution has moved* relative to
the FP16 baseline. Two quantized schemes can have very similar PPL yet
disagree with FP16 on different sets of tokens — an important signal for
downstream decoding (sampling, beam search, speculative decoding).

Four metrics are reported per position-pair and then aggregated:

- ``KL(P_ref || P_q)`` — how much the quantized distribution has moved
  *where the reference puts mass*. The mean and 95th percentile are both
  logged so we can see the typical move and the tail.
- Top-1 agreement — fraction of positions where the argmax matches. Most
  relevant for greedy decoding.
- Top-5 agreement — fraction of positions where the quantized argmax lies
  in the FP16 top-5. Most relevant for sampling / beam search.

The implementation is torch-pure and runs on whatever device the logits
live on; typical cost is a handful of milliseconds per hundred positions
on CPU (negligible next to the forward pass that produced the logits).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812


@dataclass(frozen=True)
class FidelityResult:
    """Aggregate fidelity of a quantized distribution against the FP16 reference."""

    kl_mean: float
    kl_p95: float
    top1_agree: float
    top5_agree: float
    n_positions: int


def compute_fidelity(
    logits_ref: torch.Tensor,
    logits_q: torch.Tensor,
    top_k: int = 5,
) -> FidelityResult:
    """Return per-position aggregated fidelity between the two logit tensors.

    Parameters
    ----------
    logits_ref, logits_q
        Tensors of shape ``[N, V]`` where ``N`` is the number of positions
        and ``V`` is the vocabulary size. Both must have the same shape.
    top_k
        How many of the reference's top predictions count as "agreement"
        for :attr:`FidelityResult.top5_agree`.
    """
    if logits_ref.shape != logits_q.shape:
        raise ValueError(f"shape mismatch: ref={tuple(logits_ref.shape)} q={tuple(logits_q.shape)}")
    if logits_ref.ndim != 2:
        raise ValueError(f"expected 2D logits, got ndim={logits_ref.ndim}")
    if logits_ref.shape[0] == 0:
        raise ValueError("cannot compute fidelity on zero positions")

    # Compute in FP32 for numerical stability regardless of input dtype.
    lr = logits_ref.to(torch.float32)
    lq = logits_q.to(torch.float32)

    log_p_ref = F.log_softmax(lr, dim=-1)
    log_p_q = F.log_softmax(lq, dim=-1)
    p_ref = log_p_ref.exp()
    # KL(ref || q) = sum_v p_ref * (log p_ref - log p_q).
    kl_per_pos = (p_ref * (log_p_ref - log_p_q)).sum(dim=-1)
    # Guard against the tiny negatives that can appear from FP rounding.
    kl_per_pos = kl_per_pos.clamp_min(0.0)

    top1_ref = lr.argmax(dim=-1)
    top1_q = lq.argmax(dim=-1)
    agree1 = (top1_ref == top1_q).float().mean()

    topk_ref = lr.topk(min(top_k, lr.shape[-1]), dim=-1).indices  # [N, K]
    in_topk = (topk_ref == top1_q.unsqueeze(-1)).any(dim=-1)
    agree_k = in_topk.float().mean()

    return FidelityResult(
        kl_mean=float(kl_per_pos.mean().item()),
        kl_p95=float(kl_per_pos.quantile(0.95).item()),
        top1_agree=float(agree1.item()),
        top5_agree=float(agree_k.item()),
        n_positions=int(lr.shape[0]),
    )
