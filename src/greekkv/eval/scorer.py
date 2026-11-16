"""Context-then-continuation scorer — the eval backbone for M3.

The project's hard rule
-----------------------
Log-likelihood scoring done in a single full forward pass never reads a
quantized cache — the ``past_key_values`` is built fresh from FP16 K/V. For
honest evaluation we must:

1. **Prefill the context** into a chosen cache. The cache's own ``update``
   applies the quantization (fake-quant, KIVI, eviction, ...). From this
   point the stored K/V are no longer identical to the FP16 reference.
2. **Score the continuation** against that already-quantized cache. Each
   continuation token's log-probability depends on attention over the
   modified ctx K/V — which is exactly what a production decoder would see.

This module exposes :func:`score_continuation` which does that in one call.

Sanity contract (see ``tests/test_scorer.py``)
----------------------------------------------
- ``FakeQuantCache(bits=16)`` through the scorer is **bit-identical** to a
  vanilla ``DynamicCache`` through the same scorer.
- An aggressive destroyer (``eviction-sink`` with ``n_sink=0``,
  ``n_recent=0``) produces a dramatically higher NLL than FP16 — proof the
  cache actually participates in the computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from greekkv.quant import CacheSpec, KVCacheBackend, build_cache


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of a single prefill-then-score call."""

    nll_sum: float
    nll_per_token: list[float]
    n_tokens: int

    @property
    def nll_mean(self) -> float:
        if self.n_tokens == 0:
            return float("nan")
        return self.nll_sum / self.n_tokens


def _tokenize_pair(
    tokenizer: Any, context: str, continuation: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ctx_ids, cont_ids)`` with sensible special-token handling.

    Context gets BOS (if the tokenizer uses one); continuation does not, so
    we can concatenate at the token level without duplicating BOS.
    """
    ctx_ids = tokenizer.encode(context, add_special_tokens=True, return_tensors="pt").to(device)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False, return_tensors="pt").to(
        device
    )
    if ctx_ids.shape[-1] == 0:
        raise ValueError("context tokenizes to zero tokens")
    if cont_ids.shape[-1] == 0:
        raise ValueError("continuation tokenizes to zero tokens")
    return ctx_ids, cont_ids


def _score_against_cache(
    model: Any,
    ctx_ids: torch.Tensor,
    cont_ids: torch.Tensor,
    cache: KVCacheBackend,
    return_logits: bool = False,
) -> ScoreResult | tuple[ScoreResult, torch.Tensor]:
    """The inner workhorse — prefills ``ctx_ids[:-1]`` then scores
    ``ctx_ids[-1:] + cont_ids`` using the (possibly quantized) cache.

    This split is important: we need the logits that predict the FIRST
    continuation token, which are produced by the forward of the last
    context token. If we prefilled the full context, we'd throw those
    logits away.

    When ``return_logits`` is set, the function additionally returns the
    per-position logits tensor ``[n_cont, V]`` — used by the T20 fidelity
    pipeline so a single forward pass serves both PPL and distribution
    comparison.
    """
    n_cont = cont_ids.shape[-1]

    with torch.no_grad():
        # 1) Prefill ctx[:-1] into the cache. Quantization lives in cache.update.
        if ctx_ids.shape[-1] > 1:
            model(ctx_ids[:, :-1], past_key_values=cache, use_cache=True)

        # 2) Score pass: last ctx token + all continuation tokens.
        scoring_input = torch.cat([ctx_ids[:, -1:], cont_ids], dim=-1)
        out = model(scoring_input, past_key_values=cache, use_cache=True)

        # logits[i] predicts scoring_input[i+1]. We only need the first n_cont
        # positions — those predict the n_cont continuation tokens.
        logits = out.logits[0, :n_cont, :].to(torch.float32)
        targets = cont_ids[0]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        nll_per_token = (-token_log_probs).cpu().tolist()

    result = ScoreResult(
        nll_sum=float(sum(nll_per_token)),
        nll_per_token=nll_per_token,
        n_tokens=n_cont,
    )
    if return_logits:
        return result, logits.detach().cpu()
    return result


def score_continuation(
    model: Any,
    tokenizer: Any,
    context: str,
    continuation: str,
    cache_spec: CacheSpec,
) -> ScoreResult:
    """Score ``continuation`` conditioned on ``context`` using a cache built
    per ``cache_spec``.

    The cache is built fresh on each call — do not reuse ``cache_spec`` to
    try to carry cache state across calls; that is not what it is for.
    """
    device = next(model.parameters()).device
    ctx_ids, cont_ids = _tokenize_pair(tokenizer, context, continuation, device)
    cache = build_cache(cache_spec, model_config=model.config)
    return _score_against_cache(model, ctx_ids, cont_ids, cache)
