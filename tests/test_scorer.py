"""Tests for greekkv.eval.scorer.

The two acceptance invariants of this module (explicit in the project spec):

1. **FP16 ``FakeQuantCache`` is a strict no-op** — same per-token NLLs as a
   vanilla ``DynamicCache`` when both are scored through this function.
2. **An aggressive cache destroys quality** — zeroing every context KV via
   ``eviction-sink(0, 0)`` yields a dramatically higher NLL than FP16.
   This proves the cache is actually participating in the computation and
   the function is not accidentally scoring with a fresh FP16 cache under
   the hood.
"""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from greekkv.eval.scorer import ScoreResult, _score_against_cache, score_continuation
from greekkv.quant import CacheSpec, FakeQuantCache

TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"


def _try_load_mistral():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        pytest.skip(f"transformers not importable: {e}")
    try:
        model = AutoModelForCausalLM.from_pretrained(TINY_MISTRAL, dtype=torch.float16)
        tok = AutoTokenizer.from_pretrained(TINY_MISTRAL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot load {TINY_MISTRAL}: {e}")
    model.eval()
    # tiny-random-Mistral ships without pad_token — not needed here but silences warnings.
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


CTX = "The quick brown fox jumps over the lazy dog."
CONT = " The weather today is sunny."


# ---------------------------------------------------------------------------
# ScoreResult shape + edge cases
# ---------------------------------------------------------------------------


def test_score_result_shape_and_mean():
    model, tok = _try_load_mistral()
    res = score_continuation(model, tok, CTX, CONT, CacheSpec(scheme="fp16"))
    assert isinstance(res, ScoreResult)
    assert res.n_tokens == len(res.nll_per_token) > 0
    assert res.nll_sum == pytest.approx(sum(res.nll_per_token))
    assert res.nll_mean == pytest.approx(res.nll_sum / res.n_tokens)


def test_score_empty_continuation_raises():
    model, tok = _try_load_mistral()
    with pytest.raises(ValueError, match="continuation"):
        score_continuation(model, tok, CTX, "", CacheSpec(scheme="fp16"))


def test_score_is_deterministic():
    model, tok = _try_load_mistral()
    a = score_continuation(model, tok, CTX, CONT, CacheSpec(scheme="fp16"))
    b = score_continuation(model, tok, CTX, CONT, CacheSpec(scheme="fp16"))
    assert a.nll_per_token == b.nll_per_token


# ---------------------------------------------------------------------------
# Sanity 1 — FP16 FakeQuantCache bit-equal with vanilla DynamicCache
# ---------------------------------------------------------------------------


def test_fp16_fakequant_cache_bit_equal_with_vanilla():
    """FakeQuantCache(bits=16) is specified to be a strict no-op. Through the
    full scorer that means per-token NLLs are bit-identical to vanilla."""
    model, tok = _try_load_mistral()
    device = next(model.parameters()).device
    ctx_ids = tok.encode(CTX, add_special_tokens=True, return_tensors="pt").to(device)
    cont_ids = tok.encode(CONT, add_special_tokens=False, return_tensors="pt").to(device)

    vanilla = _score_against_cache(model, ctx_ids, cont_ids, DynamicCache())
    fake16 = _score_against_cache(model, ctx_ids, cont_ids, FakeQuantCache(bits=16, group_size=16))

    # The per-token NLLs come from float32 log_softmax of fp16 logits, so even
    # tiny FP arithmetic differences would surface here. Any divergence is a
    # bug in FakeQuantCache's no-op branch.
    for a, b in zip(vanilla.nll_per_token, fake16.nll_per_token, strict=True):
        assert a == b, f"NLLs differ: {a} vs {b}"


# ---------------------------------------------------------------------------
# Sanity 2 — destroyer cache visibly worsens NLL
# ---------------------------------------------------------------------------


def test_zero_eviction_cache_destroys_quality():
    """eviction-sink with n_sink=0, n_recent=0 zeroes every context KV.
    Attention then reads only zeros from context positions → NLL should rocket.
    """
    model, tok = _try_load_mistral()
    baseline = score_continuation(model, tok, CTX, CONT, CacheSpec(scheme="fp16"))
    destroyer = score_continuation(
        model,
        tok,
        CTX,
        CONT,
        CacheSpec(scheme="eviction-sink", n_sink=0, n_recent=0),
    )
    # Not equal (cache actually participates) and visibly worse.
    assert destroyer.nll_per_token != baseline.nll_per_token
    assert destroyer.nll_mean > baseline.nll_mean


def test_int4_fakequant_perturbs_logits_vs_fp16():
    """INT4 FakeQuant should visibly change the per-token NLLs vs FP16 (not
    necessarily worse — this is a tiny-random model with uninformative weights
    — but strictly different)."""
    model, tok = _try_load_mistral()
    baseline = score_continuation(model, tok, CTX, CONT, CacheSpec(scheme="fp16"))
    int4 = score_continuation(
        model,
        tok,
        CTX,
        CONT,
        CacheSpec(scheme="int4-fakequant", bits=4, group_size=4),
    )
    assert int4.nll_per_token != baseline.nll_per_token


# ---------------------------------------------------------------------------
# Dispatch — every native cache scheme runs end-to-end through the scorer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        CacheSpec(scheme="fp16"),
        CacheSpec(scheme="int8-fakequant", bits=8, group_size=4),
        CacheSpec(scheme="int4-fakequant", bits=4, group_size=4),
        CacheSpec(scheme="int8-kivi", bits=8, group_size=4, residual=2),
        CacheSpec(scheme="int4-kivi", bits=4, group_size=4, residual=2),
        CacheSpec(scheme="eviction-sink", n_sink=2, n_recent=4),
    ],
)
def test_all_native_schemes_produce_valid_scores(spec):
    model, tok = _try_load_mistral()
    res = score_continuation(model, tok, CTX, CONT, spec)
    assert res.n_tokens > 0
    assert all(isinstance(x, float) for x in res.nll_per_token)
    assert all(x >= 0 for x in res.nll_per_token)  # NLL is non-negative
