"""Tests for greekkv.eval.multiple_choice.predict_mc."""

from __future__ import annotations

import pytest
import torch

from greekkv.eval import predict_mc
from greekkv.quant import CacheSpec

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
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def test_predict_mc_returns_valid_index():
    model, tok = _try_load_mistral()
    pred = predict_mc(
        model,
        tok,
        question="What is the color of the sky?",
        choices=["blue", "green", "fire"],
        cache_spec=CacheSpec(scheme="fp16"),
        answer_prefix=" ",
    )
    assert 0 <= pred.pred_index < 3
    assert len(pred.nll_per_choice) == 3
    assert all(isinstance(x, float) for x in pred.nll_per_choice)


def test_predict_mc_rejects_empty_choices():
    model, tok = _try_load_mistral()
    with pytest.raises(ValueError, match="choices"):
        predict_mc(model, tok, "q?", [], CacheSpec(scheme="fp16"))


def test_predict_mc_deterministic():
    model, tok = _try_load_mistral()
    a = predict_mc(model, tok, "What is 2+2?", ["4", "5", "π"], CacheSpec(scheme="fp16"))
    b = predict_mc(model, tok, "What is 2+2?", ["4", "5", "π"], CacheSpec(scheme="fp16"))
    assert a.pred_index == b.pred_index
    assert a.nll_per_choice == b.nll_per_choice


def test_predict_mc_fp16_vs_destroyer_cache_differ():
    """FP16 vs an aggressive destroyer cache: NLLs must differ (cache actually
    participates in the computation). Tiny random weights don't guarantee the
    destroyer is *monotonically worse* on every choice, but they do guarantee
    the two paths produce different numbers."""
    model, tok = _try_load_mistral()
    q = "The quick brown fox jumps over the"
    choices = ["lazy dog", "tall tree", "wide river"]

    fp16 = predict_mc(model, tok, q, choices, CacheSpec(scheme="fp16"), answer_prefix=" ")
    destroyer = predict_mc(
        model,
        tok,
        q,
        choices,
        CacheSpec(scheme="eviction-sink", n_sink=0, n_recent=0),
        answer_prefix=" ",
    )
    assert destroyer.nll_per_choice != fp16.nll_per_choice


def test_predict_mc_length_normalisation_prevents_short_bias():
    """Without length normalisation, longer choices get a systematic NLL
    penalty just from summing more terms. With normalisation, a very short
    and a very long choice can compete on equal footing."""
    model, tok = _try_load_mistral()
    # Use a nonsense prompt so the tiny-random model has no strong preference.
    q = "xyz"
    short = "a"
    long = "a b c d e f g h i j k l m"
    pred = predict_mc(model, tok, q, [short, long], CacheSpec(scheme="fp16"), answer_prefix=" ")
    # With normalisation both NLLs should be of similar order of magnitude.
    # Without it the long one would dwarf the short one by ~13×.
    assert pred.nll_per_choice[0] / pred.nll_per_choice[1] < 3.0
    assert pred.nll_per_choice[1] / pred.nll_per_choice[0] < 3.0
