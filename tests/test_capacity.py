"""Tests for the T28 capacity model.

Byte bookkeeping is the whole point — if a row in the output table
cannot be reproduced by hand from the scheme spec, something broke.
Tests pin the arithmetic against hand-computed values for the common
schemes and against ``kv_bytes`` for the derived ones.
"""

from __future__ import annotations

import math

import pytest

from greekkv.memory import (
    ALL_SCHEMES,
    DEFAULT_OVERHEAD_BYTES,
    DEFAULT_SCHEME_SPECS,
    ArchConfig,
    CapacityInputs,
    SchemeSpec,
    capacity,
    kv_bytes,
    weights_bytes,
)


def _qwen25_0_5b() -> ArchConfig:
    """Qwen2.5-0.5B dimensions — matches the T17 / T23 measured runs."""
    return ArchConfig(
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_size=896,
        head_dim=64,
        num_parameters=494_000_000,
        vocab_size=151_936,
        intermediate_size=4864,
    )


# ---------------------------------------------------------------------------
# Hard-coded invariants
# ---------------------------------------------------------------------------


def test_all_six_schemes_have_default_specs():
    assert set(DEFAULT_SCHEME_SPECS) == set(ALL_SCHEMES)


def test_fp16_scheme_bytes_match_kv_bytes_formula():
    cfg = _qwen25_0_5b()
    inp = CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=4096, scheme="fp16")
    r = capacity(inp)
    expected = kv_bytes(cfg, n_ctx=4096, bits=16, group_size=0)
    assert r.kv_bytes_per_user == expected


def test_int8_fakequant_scheme_uses_bits_8_groups_64():
    cfg = _qwen25_0_5b()
    r = capacity(
        CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=4096, scheme="int8-fakequant")
    )
    expected = kv_bytes(cfg, n_ctx=4096, bits=8, group_size=64)
    assert r.kv_bytes_per_user == expected


def test_int4_scheme_is_strictly_cheaper_than_int8():
    cfg = _qwen25_0_5b()
    i8 = capacity(
        CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=4096, scheme="int8-fakequant")
    )
    i4 = capacity(
        CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=4096, scheme="int4-fakequant")
    )
    assert i4.kv_bytes_per_user < i8.kv_bytes_per_user
    assert i4.max_concurrent_users > i8.max_concurrent_users


# ---------------------------------------------------------------------------
# KIVI residual window
# ---------------------------------------------------------------------------


def test_kivi_adds_fp16_residual_window():
    cfg = _qwen25_0_5b()
    n_ctx = 4096
    kivi = capacity(
        CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=n_ctx, scheme="int8-kivi")
    )
    fakequant = capacity(
        CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=n_ctx, scheme="int8-fakequant")
    )
    # KIVI pays the FP16 cost on the residual window, so it must be strictly
    # more expensive than pure int8-fakequant.
    assert kivi.kv_bytes_per_user > fakequant.kv_bytes_per_user

    spec = DEFAULT_SCHEME_SPECS["int8-kivi"]
    expected_residual = kv_bytes(cfg, n_ctx=spec.residual_tokens, bits=16, group_size=0)
    expected_tail = kv_bytes(cfg, n_ctx=n_ctx - spec.residual_tokens, bits=8, group_size=64)
    assert kivi.kv_bytes_per_user == expected_residual + expected_tail


def test_kivi_degenerates_to_fp16_when_n_ctx_fits_in_residual():
    cfg = _qwen25_0_5b()
    spec = DEFAULT_SCHEME_SPECS["int8-kivi"]
    r = capacity(
        CapacityInputs(
            cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=spec.residual_tokens, scheme="int8-kivi"
        )
    )
    expected = kv_bytes(cfg, n_ctx=spec.residual_tokens, bits=16, group_size=0)
    assert r.kv_bytes_per_user == expected


# ---------------------------------------------------------------------------
# Eviction-sink: bounded cache
# ---------------------------------------------------------------------------


def test_eviction_sink_cache_bytes_do_not_grow_with_n_ctx():
    cfg = _qwen25_0_5b()
    low = capacity(
        CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=1024, scheme="eviction-sink")
    )
    high = capacity(
        CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=32768, scheme="eviction-sink")
    )
    assert low.kv_bytes_per_user == high.kv_bytes_per_user


def test_eviction_sink_matches_residual_window_at_fp16():
    cfg = _qwen25_0_5b()
    spec = DEFAULT_SCHEME_SPECS["eviction-sink"]
    r = capacity(
        CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=4096, scheme="eviction-sink")
    )
    expected = kv_bytes(cfg, n_ctx=spec.residual_tokens, bits=16, group_size=0)
    assert r.kv_bytes_per_user == expected


# ---------------------------------------------------------------------------
# Capacity arithmetic
# ---------------------------------------------------------------------------


def test_capacity_formula_matches_hand_computation():
    cfg = _qwen25_0_5b()
    vram = 16 * 1024**3
    n_ctx = 4096
    r = capacity(CapacityInputs(cfg=cfg, vram_bytes=vram, n_ctx=n_ctx, scheme="fp16"))
    w = weights_bytes(cfg, bits=16, group_size=0)
    kv = kv_bytes(cfg, n_ctx=n_ctx, bits=16, group_size=0)
    expected_users = (vram - w - DEFAULT_OVERHEAD_BYTES) // kv
    assert r.max_concurrent_users == expected_users


def test_zero_users_when_weights_alone_exceed_vram():
    cfg = _qwen25_0_5b()
    # Pretend weights are allowed 10 MB — they need much more.
    tiny = 10 * 1024 * 1024
    r = capacity(CapacityInputs(cfg=cfg, vram_bytes=tiny, n_ctx=4096, scheme="fp16"))
    assert r.max_concurrent_users == 0
    assert r.free_bytes_after_weights == 0


def test_zero_ctx_implies_infinite_kv_users_capped_by_formula():
    # n_ctx == 0 means no cache at all, so kv_bytes_per_user == 0 and the
    # formula returns 0 (no cache → no "capacity per cache slot").
    cfg = _qwen25_0_5b()
    r = capacity(CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=0, scheme="int8-fakequant"))
    assert r.kv_bytes_per_user == 0
    assert r.max_concurrent_users == 0


def test_quant_weights_reduce_weight_bytes_by_expected_ratio():
    cfg = _qwen25_0_5b()
    fp16 = capacity(CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=4096, scheme="fp16"))
    int8 = capacity(
        CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=4096, scheme="fp16", bits_weights=8)
    )
    ratio = fp16.weight_bytes / int8.weight_bytes
    assert math.isclose(ratio, 2.0, rel_tol=0.001)


# ---------------------------------------------------------------------------
# Scheme spec override
# ---------------------------------------------------------------------------


def test_scheme_spec_override_flows_through():
    cfg = _qwen25_0_5b()
    # Make the KIVI residual window 128 tokens instead of 36.
    r = capacity(
        CapacityInputs(
            cfg=cfg,
            vram_bytes=16 * 1024**3,
            n_ctx=4096,
            scheme="int8-kivi",
            scheme_specs={"int8-kivi": SchemeSpec(bits_kv=8, group_size=64, residual_tokens=128)},
        )
    )
    expected_residual = kv_bytes(cfg, n_ctx=128, bits=16, group_size=0)
    expected_tail = kv_bytes(cfg, n_ctx=4096 - 128, bits=8, group_size=64)
    assert r.kv_bytes_per_user == expected_residual + expected_tail


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unknown_scheme_raises():
    cfg = _qwen25_0_5b()
    with pytest.raises(ValueError, match="unknown scheme"):
        capacity(CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=4096, scheme="nf4-magic"))


def test_negative_vram_raises():
    cfg = _qwen25_0_5b()
    with pytest.raises(ValueError, match="vram_bytes"):
        capacity(CapacityInputs(cfg=cfg, vram_bytes=-1, n_ctx=4096, scheme="fp16"))


def test_negative_n_ctx_raises():
    cfg = _qwen25_0_5b()
    with pytest.raises(ValueError, match="n_ctx"):
        capacity(CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=-1, scheme="fp16"))


# ---------------------------------------------------------------------------
# Source tagging — the whole point of this task
# ---------------------------------------------------------------------------


def test_measured_source_passes_through_to_result():
    cfg = _qwen25_0_5b()
    r = capacity(
        CapacityInputs(
            cfg=cfg,
            vram_bytes=16 * 1024**3,
            n_ctx=4096,
            scheme="int8-kivi",
            source="measured",
            measured_ppl=4.66,
            measured_decode_ms=5.6,
        )
    )
    assert r.source == "measured"
    assert r.measured_ppl == pytest.approx(4.66)
    assert r.measured_decode_ms == pytest.approx(5.6)


def test_default_source_is_estimated():
    cfg = _qwen25_0_5b()
    r = capacity(CapacityInputs(cfg=cfg, vram_bytes=16 * 1024**3, n_ctx=4096, scheme="fp16"))
    assert r.source == "estimated"
    assert r.measured_ppl is None
    assert r.measured_decode_ms is None
