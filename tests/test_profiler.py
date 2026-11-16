"""Tests for greekkv.systems.profiler.profile_scheme."""

from __future__ import annotations

import pytest
import torch

from greekkv.quant import CacheSpec
from greekkv.systems import ProfileResult, profile_scheme

TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"


def _try_load_mistral():
    try:
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        pytest.skip(f"transformers not importable: {e}")
    try:
        return AutoModelForCausalLM.from_pretrained(TINY_MISTRAL, dtype=torch.float16)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot load {TINY_MISTRAL}: {e}")


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_profile_rejects_nonpositive_n_ctx():
    model = _try_load_mistral()
    with pytest.raises(ValueError, match="n_ctx"):
        profile_scheme(model, CacheSpec(scheme="fp16"), n_ctx=0, n_decode=4)


def test_profile_rejects_nonpositive_n_decode():
    model = _try_load_mistral()
    with pytest.raises(ValueError, match="n_decode"):
        profile_scheme(model, CacheSpec(scheme="fp16"), n_ctx=8, n_decode=0)


# ---------------------------------------------------------------------------
# Return shape + metric sanity
# ---------------------------------------------------------------------------


def test_profile_returns_valid_result():
    model = _try_load_mistral()
    model.eval()
    res = profile_scheme(model, CacheSpec(scheme="fp16"), n_ctx=16, n_decode=4)
    assert isinstance(res, ProfileResult)
    assert res.scheme == "fp16"
    assert res.n_ctx == 16
    assert res.n_decode == 4
    assert res.peak_mem_mb >= 0
    assert res.peak_mem_delta_mb >= 0
    assert res.prefill_wall_s > 0
    assert res.decode_wall_s > 0
    assert res.decode_ms_per_token > 0
    assert res.decode_tokens_per_s > 0
    # tokens/s and ms/token are reciprocals (to within rounding).
    assert res.decode_tokens_per_s == pytest.approx(1000.0 / res.decode_ms_per_token, rel=1e-6)


# ---------------------------------------------------------------------------
# Scheme dispatch — every native scheme can be profiled end-to-end
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
def test_profile_every_native_scheme(spec):
    model = _try_load_mistral()
    model.eval()
    res = profile_scheme(model, spec, n_ctx=16, n_decode=4)
    assert res.scheme == spec.scheme
    assert res.decode_tokens_per_s > 0


# ---------------------------------------------------------------------------
# Peak memory scales with n_ctx for FP16 (monotonicity under growth)
# ---------------------------------------------------------------------------


def test_peak_memory_delta_grows_with_n_ctx():
    """Doubling n_ctx should produce a non-decreasing peak memory delta on FP16
    (larger cache → more tensors live). Tiny random weights don't invalidate
    the monotonicity direction."""
    model = _try_load_mistral()
    model.eval()
    small = profile_scheme(model, CacheSpec(scheme="fp16"), n_ctx=16, n_decode=4)
    large = profile_scheme(model, CacheSpec(scheme="fp16"), n_ctx=64, n_decode=4)
    # Allow equal — on CPU the deltas can be identical if the allocator reuses pages.
    assert large.peak_mem_delta_mb >= small.peak_mem_delta_mb - 1.0
