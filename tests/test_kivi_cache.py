"""Tests for greekkv.quant.kivi_cache.

Three tiers:
- Config validation.
- Cache-level invariants: FP16 no-op, residual window boundaries, mid-stream
  eviction idempotency, parity vs vanilla when nothing is evicted.
- Helper-level: per-channel shares scale across sequence, per-token shares
  scale across head_dim (we assert this by injecting known sparsity patterns).
- Integration: real tiny Mistral forward pass.
"""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from greekkv.quant import KiviCache, KiviCacheConfig
from greekkv.quant.kivi_cache import _fake_quant_per_channel, _fake_quant_per_token

TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"


def _random_kv(
    seed: int, batch: int, heads: int, n_ctx: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(batch, heads, n_ctx, head_dim, generator=g, dtype=torch.float16)
    v = torch.randn(batch, heads, n_ctx, head_dim, generator=g, dtype=torch.float16)
    return k, v


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_config_accepts_valid_bits(bits):
    cfg = KiviCacheConfig(bits=bits)
    assert cfg.bits == bits


@pytest.mark.parametrize("bad", [0, 1, 3, 5, 7, 2])
def test_config_rejects_bad_bits(bad):
    with pytest.raises(ValueError, match="bits"):
        KiviCacheConfig(bits=bad)


def test_config_rejects_bad_group_size():
    with pytest.raises(ValueError, match="group_size"):
        KiviCacheConfig(group_size=0)


def test_config_rejects_negative_residual():
    with pytest.raises(ValueError, match="residual"):
        KiviCacheConfig(residual=-1)


# ---------------------------------------------------------------------------
# FP16 mode must be a bit-identical no-op
# ---------------------------------------------------------------------------


def test_fp16_single_update_bit_identical():
    k, v = _random_kv(seed=0, batch=1, heads=4, n_ctx=8, head_dim=16)
    vanilla = DynamicCache()
    kivi = KiviCache(KiviCacheConfig(bits=16, group_size=8, residual=4))
    k_v, v_v = vanilla.update(k, v, layer_idx=0)
    k_f, v_f = kivi.update(k, v, layer_idx=0)
    assert torch.equal(k_v, k_f)
    assert torch.equal(v_v, v_f)


# ---------------------------------------------------------------------------
# Residual window: parity when nothing is ever evicted
# ---------------------------------------------------------------------------


def test_parity_when_residual_exceeds_seq_len():
    """Core acceptance: R >= seq_len → no eviction → bit-identical with vanilla."""
    n_ctx = 10
    k, v = _random_kv(seed=42, batch=1, heads=4, n_ctx=n_ctx, head_dim=32)
    vanilla = DynamicCache()
    kivi = KiviCache(KiviCacheConfig(bits=4, group_size=16, residual=n_ctx))
    k_v, v_v = vanilla.update(k, v, layer_idx=0)
    k_q, v_q = kivi.update(k, v, layer_idx=0)
    assert torch.equal(k_v, k_q)
    assert torch.equal(v_v, v_q)


def test_parity_when_residual_much_larger_than_seq_len():
    k, v = _random_kv(seed=1, batch=1, heads=4, n_ctx=5, head_dim=16)
    vanilla = DynamicCache()
    kivi = KiviCache(KiviCacheConfig(bits=4, group_size=8, residual=1000))
    k_v, _ = vanilla.update(k, v, layer_idx=0)
    k_q, _ = kivi.update(k, v, layer_idx=0)
    assert torch.equal(k_v, k_q)


# ---------------------------------------------------------------------------
# Residual window: quantizes the leading tail, keeps the last R positions
# ---------------------------------------------------------------------------


def test_residual_keeps_last_R_tokens_lossless():
    """Positions after ``seq_len - R`` must be bit-identical to vanilla."""
    n_ctx = 10
    r = 3
    k, v = _random_kv(seed=7, batch=1, heads=4, n_ctx=n_ctx, head_dim=16)
    vanilla = DynamicCache()
    kivi = KiviCache(KiviCacheConfig(bits=4, group_size=8, residual=r))
    k_v, v_v = vanilla.update(k, v, layer_idx=0)
    k_q, v_q = kivi.update(k, v, layer_idx=0)
    # The last r positions are untouched.
    assert torch.equal(k_v[..., -r:, :], k_q[..., -r:, :])
    assert torch.equal(v_v[..., -r:, :], v_q[..., -r:, :])
    # The earlier positions are perturbed (quantization is real).
    assert not torch.equal(k_v[..., :-r, :], k_q[..., :-r, :])
    assert not torch.equal(v_v[..., :-r, :], v_q[..., :-r, :])


def test_residual_zero_quantizes_everything():
    """``residual=0`` → every position is immediately quantized."""
    k, v = _random_kv(seed=2, batch=1, heads=4, n_ctx=8, head_dim=32)
    vanilla = DynamicCache()
    kivi = KiviCache(KiviCacheConfig(bits=4, group_size=16, residual=0))
    k_v, v_v = vanilla.update(k, v, layer_idx=0)
    k_q, v_q = kivi.update(k, v, layer_idx=0)
    # Nothing identical; everything perturbed.
    assert not torch.equal(k_v, k_q)
    assert not torch.equal(v_v, v_q)


# ---------------------------------------------------------------------------
# Streaming eviction: previously-quantized positions are not re-quantized
# ---------------------------------------------------------------------------


def test_already_quantized_positions_are_not_requantized():
    """On a subsequent update, positions quantized in a prior step must not
    change again — otherwise error would accumulate on repeated calls.
    """
    r = 2
    g = 16
    kivi = KiviCache(KiviCacheConfig(bits=4, group_size=g, residual=r))
    # First update: feed enough tokens to trigger an eviction (5 tokens, r=2).
    k0, v0 = _random_kv(seed=5, batch=1, heads=4, n_ctx=5, head_dim=g)
    k_after_1, v_after_1 = kivi.update(k0, v0, layer_idx=0)
    snapshot_k = k_after_1[..., :3, :].clone()  # the quantized portion (seq_len - r = 3)
    snapshot_v = v_after_1[..., :3, :].clone()

    # Second update: more tokens. Already-quantized positions [0:3] must stay bit-identical.
    k1, v1 = _random_kv(seed=6, batch=1, heads=4, n_ctx=4, head_dim=g)
    k_after_2, v_after_2 = kivi.update(k1, v1, layer_idx=0)
    assert torch.equal(k_after_2[..., :3, :], snapshot_k)
    assert torch.equal(v_after_2[..., :3, :], snapshot_v)


def test_streaming_cache_length_grows_like_vanilla():
    r = 2
    g = 8
    vanilla = DynamicCache()
    kivi = KiviCache(KiviCacheConfig(bits=4, group_size=g, residual=r))
    for step in range(4):
        k, v = _random_kv(seed=300 + step, batch=1, heads=4, n_ctx=2, head_dim=g)
        vanilla.update(k, v, layer_idx=0)
        kivi.update(k, v, layer_idx=0)
    assert kivi.get_seq_length(0) == vanilla.get_seq_length(0) == 8


# ---------------------------------------------------------------------------
# Scheme invariants: per-channel shares across seq; per-token shares across head_dim
# ---------------------------------------------------------------------------


def test_per_channel_confines_outlier_to_its_channel_group():
    """Blow up one channel; per-channel must *confine* the inflated scale to
    the channel group that contains the outlier and leave other channel
    groups (and their per-position reconstructions) untouched."""
    g = 4
    # head_dim = 16 → 4 channel groups; seq_len = 4.
    x = torch.randn(1, 2, 4, 16, dtype=torch.float32)
    x[:, :, 0, 0] = 100.0  # outlier at channel 0 (channel-group 0)
    y = _fake_quant_per_channel(x, bits=4, group_size=g, symmetric=True)
    err = (y - x).abs()
    # Group 0 covers channels [0:4]; its scale is inflated by the outlier →
    # positions 1..3 within group 0 reconstruct coarsely → high error there.
    # Groups 1..3 are untouched by the outlier → tight reconstruction.
    err_group0 = err[..., :g].max()
    err_other = err[..., g:].max()
    assert err_group0 > err_other * 5  # at least an order of magnitude apart


def test_per_token_confines_outlier_to_its_token_group():
    """Symmetric check for values: blow up one position in one channel; the
    per-token scheme must inflate the scale of the token group containing
    that position and leave other token groups tight."""
    g = 4
    # seq_len = 16 → 4 token groups; head_dim = 8.
    x = torch.randn(1, 2, 16, 8, dtype=torch.float32)
    x[:, :, 0, 0] = 100.0  # outlier at position 0 (token-group 0)
    y = _fake_quant_per_token(x, bits=4, group_size=g, symmetric=True)
    err = (y - x).abs()
    err_group0 = err[:, :, :g, :].max()  # token group 0 = positions [0:4]
    err_other = err[:, :, g:, :].max()  # token groups 1..3
    assert err_group0 > err_other * 5


def test_per_channel_padded_tail_roundtrips_shape():
    # head_dim 10 not divisible by group_size 4 → padding path exercised
    x = torch.randn(1, 2, 3, 10, dtype=torch.float32)
    y = _fake_quant_per_channel(x, bits=8, group_size=4, symmetric=True)
    assert y.shape == x.shape


def test_per_token_padded_tail_roundtrips_shape():
    # seq_len 5 not divisible by group_size 2 → padding path exercised
    x = torch.randn(1, 2, 5, 8, dtype=torch.float32)
    y = _fake_quant_per_token(x, bits=8, group_size=2, symmetric=True)
    assert y.shape == x.shape


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


def test_fp16_kivi_matches_vanilla_logits():
    model = _try_load_mistral()
    model.eval()
    tokens = torch.randint(0, 100, (1, 16))

    cfg = KiviCacheConfig(bits=16, group_size=8, residual=4)
    with torch.no_grad():
        out_vanilla = model(tokens, past_key_values=DynamicCache(), use_cache=True)
        out_kivi = model(tokens, past_key_values=KiviCache(cfg), use_cache=True)

    assert torch.equal(out_vanilla.logits, out_kivi.logits)


def test_kivi_with_large_residual_matches_vanilla_logits():
    """R >= total tokens → no eviction → still bit-identical to vanilla."""
    model = _try_load_mistral()
    model.eval()
    tokens = torch.randint(0, 100, (1, 16))

    cfg = KiviCacheConfig(bits=4, group_size=4, residual=100)  # R >> 16
    with torch.no_grad():
        out_vanilla = model(tokens, past_key_values=DynamicCache(), use_cache=True)
        out_kivi = model(tokens, past_key_values=KiviCache(cfg), use_cache=True)

    assert torch.equal(out_vanilla.logits, out_kivi.logits)


def test_kivi_int4_changes_logits_but_keeps_shape():
    model = _try_load_mistral()
    model.eval()
    tokens = torch.randint(0, 100, (1, 16))

    cfg = KiviCacheConfig(bits=4, group_size=4, residual=2)  # small residual → real eviction
    with torch.no_grad():
        out_vanilla = model(tokens, past_key_values=DynamicCache(), use_cache=True)
        out_kivi = model(tokens, past_key_values=KiviCache(cfg), use_cache=True)

    assert out_kivi.logits.shape == out_vanilla.logits.shape
    assert not torch.equal(out_kivi.logits, out_vanilla.logits)
