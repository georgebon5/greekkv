"""Tests for greekkv.quant.fakequant_cache.FakeQuantCache.

Two layers:
- Unit: construction, FP16 no-op, quantization effect, growing cache length.
- Integration: end-to-end forward pass on a tiny public HF model, comparing
  the fake-quant cache against the vanilla ``DynamicCache``.
"""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from greekkv.quant import FakeQuantCache

TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"


# ---------------------------------------------------------------------------
# Construction + argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_construct_valid_bits(bits):
    c = FakeQuantCache(bits=bits, group_size=32)
    assert c.bits == bits


def test_construct_rejects_bad_bits():
    with pytest.raises(ValueError, match="bits"):
        FakeQuantCache(bits=2, group_size=32)


# ---------------------------------------------------------------------------
# FP16 mode must be a bit-identical no-op
# ---------------------------------------------------------------------------


def _random_kv(
    seed: int, batch: int, heads: int, n_ctx: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(batch, heads, n_ctx, head_dim, generator=g, dtype=torch.float16)
    v = torch.randn(batch, heads, n_ctx, head_dim, generator=g, dtype=torch.float16)
    return k, v


def test_fp16_single_update_bit_identical():
    k, v = _random_kv(seed=0, batch=1, heads=4, n_ctx=8, head_dim=16)
    vanilla = DynamicCache()
    fake16 = FakeQuantCache(bits=16, group_size=16)
    k_v, v_v = vanilla.update(k, v, layer_idx=0)
    k_f, v_f = fake16.update(k, v, layer_idx=0)
    assert torch.equal(k_v, k_f)
    assert torch.equal(v_v, v_f)


def test_fp16_streaming_updates_bit_identical():
    """Multiple update() calls (simulating decoding) must still match vanilla."""
    vanilla = DynamicCache()
    fake16 = FakeQuantCache(bits=16, group_size=16)
    for step in range(3):
        k, v = _random_kv(seed=100 + step, batch=1, heads=4, n_ctx=2, head_dim=16)
        kv_out = vanilla.update(k, v, layer_idx=0)
        kf_out = fake16.update(k, v, layer_idx=0)
        assert torch.equal(kv_out[0], kf_out[0])
        assert torch.equal(kv_out[1], kf_out[1])


# ---------------------------------------------------------------------------
# INT8 / INT4 — quantization has effect and preserves shapes / dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [8, 4])
def test_quant_changes_output_but_preserves_shape_and_dtype(bits):
    k, v = _random_kv(seed=1, batch=1, heads=4, n_ctx=8, head_dim=32)
    vanilla = DynamicCache()
    quant = FakeQuantCache(bits=bits, group_size=32)
    k_v, v_v = vanilla.update(k, v, layer_idx=0)
    k_q, v_q = quant.update(k, v, layer_idx=0)
    assert k_q.shape == k_v.shape
    assert v_q.shape == v_v.shape
    assert k_q.dtype == k_v.dtype == torch.float16
    assert not torch.equal(k_q, k_v)  # quantization must change something
    assert not torch.equal(v_q, v_v)


def test_int8_error_smaller_than_int4():
    """Monotonicity across the cache boundary: higher precision → smaller drift."""
    k, v = _random_kv(seed=2, batch=1, heads=4, n_ctx=16, head_dim=64)
    vanilla_k, vanilla_v = DynamicCache().update(k, v, layer_idx=0)
    k8, v8 = FakeQuantCache(bits=8, group_size=32).update(k, v, layer_idx=0)
    k4, v4 = FakeQuantCache(bits=4, group_size=32).update(k, v, layer_idx=0)
    err_k8 = (k8.float() - vanilla_k.float()).norm()
    err_k4 = (k4.float() - vanilla_k.float()).norm()
    err_v8 = (v8.float() - vanilla_v.float()).norm()
    err_v4 = (v4.float() - vanilla_v.float()).norm()
    assert err_k8 < err_k4
    assert err_v8 < err_v4


def test_streaming_cache_grows_like_vanilla():
    """After N streaming updates, cache length must equal vanilla's."""
    vanilla = DynamicCache()
    quant = FakeQuantCache(bits=8, group_size=16)
    for step in range(4):
        k, v = _random_kv(seed=300 + step, batch=1, heads=4, n_ctx=2, head_dim=16)
        vanilla.update(k, v, layer_idx=0)
        quant.update(k, v, layer_idx=0)
    assert quant.get_seq_length(0) == vanilla.get_seq_length(0) == 8


# ---------------------------------------------------------------------------
# Integration: real tiny HF model forward pass
# ---------------------------------------------------------------------------


def _try_load_mistral():
    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:  # pragma: no cover
        pytest.skip(f"transformers not importable: {e}")
    try:
        return AutoModelForCausalLM.from_pretrained(TINY_MISTRAL, dtype=torch.float16)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot load {TINY_MISTRAL}: {e}")


def test_fp16_fakequant_cache_matches_vanilla_logits():
    """End-to-end: FP16 FakeQuantCache must produce bit-identical logits."""
    model = _try_load_mistral()
    model.eval()
    tokens = torch.randint(0, 100, (1, 16))

    with torch.no_grad():
        out_vanilla = model(tokens, past_key_values=DynamicCache(), use_cache=True)
        out_fake = model(tokens, past_key_values=FakeQuantCache(bits=16), use_cache=True)

    assert torch.equal(out_vanilla.logits, out_fake.logits)


def test_int8_fakequant_cache_changes_logits_but_keeps_shape():
    """End-to-end: INT8 cache visibly perturbs logits but preserves the shape."""
    model = _try_load_mistral()
    model.eval()
    tokens = torch.randint(0, 100, (1, 16))

    with torch.no_grad():
        out_vanilla = model(tokens, past_key_values=DynamicCache(), use_cache=True)
        out_int8 = model(
            tokens,
            past_key_values=FakeQuantCache(bits=8, group_size=4),
            use_cache=True,
        )
        out_int4 = model(
            tokens,
            past_key_values=FakeQuantCache(bits=4, group_size=4),
            use_cache=True,
        )

    assert out_int8.logits.shape == out_vanilla.logits.shape
    assert not torch.equal(out_int8.logits, out_vanilla.logits)

    # Monotonicity: INT4 moves logits farther than INT8.
    err8 = (out_int8.logits.float() - out_vanilla.logits.float()).norm()
    err4 = (out_int4.logits.float() - out_vanilla.logits.float()).norm()
    assert err8 < err4
