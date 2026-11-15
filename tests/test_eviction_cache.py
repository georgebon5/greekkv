"""Tests for greekkv.quant.eviction_cache.StreamingEvictionCache."""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from greekkv.quant import CacheSpec, StreamingEvictionCache, build_cache

TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"


def _random_kv(
    seed: int, batch: int, heads: int, n_ctx: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(batch, heads, n_ctx, head_dim, generator=g, dtype=torch.float16)
    v = torch.randn(batch, heads, n_ctx, head_dim, generator=g, dtype=torch.float16)
    return k, v


# ---------------------------------------------------------------------------
# Construction + arg validation
# ---------------------------------------------------------------------------


def test_construct_valid_args():
    c = StreamingEvictionCache(n_sink=4, n_recent=16)
    assert c.n_sink == 4
    assert c.n_recent == 16


def test_rejects_negative_sink():
    with pytest.raises(ValueError, match="n_sink"):
        StreamingEvictionCache(n_sink=-1, n_recent=16)


def test_rejects_negative_recent():
    with pytest.raises(ValueError, match="n_recent"):
        StreamingEvictionCache(n_sink=4, n_recent=-1)


# ---------------------------------------------------------------------------
# No-eviction regime: total length ≤ budget → bit-identical with vanilla
# ---------------------------------------------------------------------------


def test_no_eviction_when_total_fits_budget():
    """n_sink+n_recent ≥ seq_len → cache is bit-identical with vanilla."""
    k, v = _random_kv(seed=0, batch=1, heads=4, n_ctx=10, head_dim=16)
    vanilla = DynamicCache()
    cache = StreamingEvictionCache(n_sink=4, n_recent=16)  # budget = 20 > 10
    k_v, v_v = vanilla.update(k, v, layer_idx=0)
    k_c, v_c = cache.update(k, v, layer_idx=0)
    assert torch.equal(k_v, k_c)
    assert torch.equal(v_v, v_c)


# ---------------------------------------------------------------------------
# Eviction regime: middle positions zeroed, sink + recent preserved
# ---------------------------------------------------------------------------


def test_eviction_zeros_middle_preserves_sink_and_recent():
    k, v = _random_kv(seed=1, batch=1, heads=4, n_ctx=20, head_dim=16)
    vanilla = DynamicCache()
    cache = StreamingEvictionCache(n_sink=2, n_recent=4)  # budget = 6, total = 20
    k_v, v_v = vanilla.update(k, v, layer_idx=0)
    k_c, v_c = cache.update(k, v, layer_idx=0)

    # Sink positions untouched.
    assert torch.equal(k_v[..., :2, :], k_c[..., :2, :])
    assert torch.equal(v_v[..., :2, :], v_c[..., :2, :])

    # Recent positions untouched.
    assert torch.equal(k_v[..., -4:, :], k_c[..., -4:, :])
    assert torch.equal(v_v[..., -4:, :], v_c[..., -4:, :])

    # Middle positions zeroed.
    assert torch.all(k_c[..., 2:-4, :] == 0)
    assert torch.all(v_c[..., 2:-4, :] == 0)


def test_eviction_preserves_total_shape():
    """Simplified eviction does NOT shrink the tensor — shape is preserved."""
    k, v = _random_kv(seed=2, batch=1, heads=4, n_ctx=20, head_dim=16)
    cache = StreamingEvictionCache(n_sink=2, n_recent=4)
    k_c, v_c = cache.update(k, v, layer_idx=0)
    assert k_c.shape == (1, 4, 20, 16)
    assert v_c.shape == (1, 4, 20, 16)


def test_edge_case_zero_sink_zero_recent():
    """n_sink=0, n_recent=0 → every position zeroed as soon as total > 0."""
    k, v = _random_kv(seed=3, batch=1, heads=2, n_ctx=5, head_dim=8)
    cache = StreamingEvictionCache(n_sink=0, n_recent=0)
    k_c, v_c = cache.update(k, v, layer_idx=0)
    assert torch.all(k_c == 0)
    assert torch.all(v_c == 0)


# ---------------------------------------------------------------------------
# Streaming: eviction applied on each update as the cache grows
# ---------------------------------------------------------------------------


def test_streaming_evicts_as_cache_grows_past_budget():
    cache = StreamingEvictionCache(n_sink=1, n_recent=2)
    for step in range(5):
        k, v = _random_kv(seed=100 + step, batch=1, heads=2, n_ctx=2, head_dim=8)
        k_c, v_c = cache.update(k, v, layer_idx=0)
    # After 5 updates of 2 tokens each = 10 total. sink=1, recent=2 → middle 7 zeroed.
    assert k_c.shape == (1, 2, 10, 8)
    # Middle positions [1:-2] = [1:8] all zero.
    assert torch.all(k_c[..., 1:-2, :] == 0)
    assert torch.all(v_c[..., 1:-2, :] == 0)
    # First position (sink) preserved from the very first update.
    assert not torch.all(k_c[..., :1, :] == 0)


# ---------------------------------------------------------------------------
# Dispatch: eviction-sink scheme wired into build_cache
# ---------------------------------------------------------------------------


def test_build_cache_eviction_scheme_returns_streaming_cache():
    cache = build_cache(CacheSpec(scheme="eviction-sink", n_sink=8, n_recent=64))
    assert isinstance(cache, StreamingEvictionCache)
    assert cache.n_sink == 8
    assert cache.n_recent == 64


def test_cachespec_rejects_negative_sink():
    with pytest.raises(ValueError, match="n_sink"):
        CacheSpec(scheme="eviction-sink", n_sink=-1)


def test_cachespec_rejects_negative_recent():
    with pytest.raises(ValueError, match="n_recent"):
        CacheSpec(scheme="eviction-sink", n_recent=-1)


# ---------------------------------------------------------------------------
# Integration: real forward pass changes logits
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


def test_eviction_cache_changes_logits_but_keeps_shape():
    """End-to-end: aggressive eviction perturbs logits but preserves shape."""
    model = _try_load_mistral()
    model.eval()
    tokens = torch.randint(0, 100, (1, 16))

    with torch.no_grad():
        out_vanilla = model(tokens, past_key_values=DynamicCache(), use_cache=True)
        out_evict = model(
            tokens,
            past_key_values=StreamingEvictionCache(n_sink=2, n_recent=4),
            use_cache=True,
        )

    assert out_evict.logits.shape == out_vanilla.logits.shape
    assert not torch.equal(out_evict.logits, out_vanilla.logits)
