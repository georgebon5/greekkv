"""Tests for greekkv.quant.adapters.

Three layers:
- CacheSpec validation (string + range).
- build_cache dispatch for each native scheme (fp16, fakequant, kivi).
- Optional-backend behaviour: quanto/HQQ raise clean install hints when their
  packages are not present; vllm-fp8 raises NotImplementedError. Protocol
  isinstance check for every reachable cache.
"""

from __future__ import annotations

import importlib

import pytest
import torch
from transformers.cache_utils import DynamicCache

from greekkv.quant import (
    CacheSpec,
    FakeQuantCache,
    KiviCache,
    KVCacheBackend,
    build_cache,
)
from greekkv.quant.adapters.base import SUPPORTED_SCHEMES


def _is_installed(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"


# ---------------------------------------------------------------------------
# CacheSpec
# ---------------------------------------------------------------------------


def test_cachespec_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unknown cache scheme"):
        CacheSpec(scheme="not-a-scheme")


@pytest.mark.parametrize("scheme", SUPPORTED_SCHEMES)
def test_cachespec_accepts_every_supported_scheme(scheme):
    spec = CacheSpec(scheme=scheme)
    assert spec.scheme == scheme


def test_cachespec_rejects_invalid_numbers():
    with pytest.raises(ValueError, match="bits"):
        CacheSpec(scheme="fp16", bits=0)
    with pytest.raises(ValueError, match="group_size"):
        CacheSpec(scheme="fp16", group_size=0)
    with pytest.raises(ValueError, match="residual"):
        CacheSpec(scheme="fp16", residual=-1)


def test_cachespec_is_frozen():
    spec = CacheSpec(scheme="fp16")
    with pytest.raises((AttributeError, TypeError)):
        spec.bits = 4  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_cache dispatch — native schemes
# ---------------------------------------------------------------------------


def test_build_fp16_returns_dynamic_cache():
    cache = build_cache(CacheSpec(scheme="fp16"))
    assert isinstance(cache, DynamicCache)
    # FakeQuantCache / KiviCache are DynamicCache subclasses, so disambiguate:
    assert type(cache) is DynamicCache


@pytest.mark.parametrize("scheme, bits", [("int4-fakequant", 4), ("int8-fakequant", 8)])
def test_build_fakequant_forwards_bits(scheme, bits):
    cache = build_cache(CacheSpec(scheme=scheme, bits=bits, group_size=32))
    assert isinstance(cache, FakeQuantCache)
    assert cache.bits == bits
    assert cache.group_size == 32


@pytest.mark.parametrize("scheme, bits", [("int4-kivi", 4), ("int8-kivi", 8)])
def test_build_kivi_forwards_config(scheme, bits):
    cache = build_cache(CacheSpec(scheme=scheme, bits=bits, group_size=64, residual=16))
    assert isinstance(cache, KiviCache)
    assert cache.cfg.bits == bits
    assert cache.cfg.group_size == 64
    assert cache.cfg.residual == 16


# ---------------------------------------------------------------------------
# build_cache dispatch — optional backends
# ---------------------------------------------------------------------------


def test_build_vllm_fp8_raises_optional_notice():
    with pytest.raises(NotImplementedError, match="OPTIONAL"):
        build_cache(CacheSpec(scheme="vllm-fp8"))


def test_build_quanto_or_hqq_requires_model_config():
    for s in ("quanto", "hqq"):
        with pytest.raises(ValueError, match="model_config"):
            build_cache(CacheSpec(scheme=s, bits=4, group_size=64))


@pytest.mark.skipif(
    _is_installed("optimum.quanto"),
    reason="optimum-quanto is installed; this test checks the missing-dep path",
)
def test_build_quanto_without_dep_raises_install_hint():
    # Need a dummy model_config to reach the optional-import check.
    class _DummyCfg:
        pass

    with pytest.raises(ModuleNotFoundError, match="optimum-quanto"):
        build_cache(CacheSpec(scheme="quanto", bits=4, group_size=64), model_config=_DummyCfg())


@pytest.mark.skipif(
    _is_installed("hqq"),
    reason="hqq is installed; this test checks the missing-dep path",
)
def test_build_hqq_without_dep_raises_install_hint():
    class _DummyCfg:
        pass

    with pytest.raises(ModuleNotFoundError, match="hqq"):
        build_cache(CacheSpec(scheme="hqq", bits=4, group_size=64), model_config=_DummyCfg())


# ---------------------------------------------------------------------------
# Protocol satisfaction — runtime isinstance for every reachable cache
# ---------------------------------------------------------------------------


def test_native_caches_satisfy_protocol():
    assert isinstance(build_cache(CacheSpec(scheme="fp16")), KVCacheBackend)
    assert isinstance(build_cache(CacheSpec(scheme="int4-fakequant", bits=4)), KVCacheBackend)
    assert isinstance(
        build_cache(CacheSpec(scheme="int4-kivi", bits=4, residual=4)), KVCacheBackend
    )


# ---------------------------------------------------------------------------
# End-to-end: dispatch wired into a real forward pass
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


def test_fp16_dispatch_is_bit_identical_in_forward():
    model = _try_load_mistral()
    model.eval()
    tokens = torch.randint(0, 100, (1, 12))
    with torch.no_grad():
        out_vanilla = model(tokens, past_key_values=DynamicCache(), use_cache=True)
        out_disp = model(
            tokens, past_key_values=build_cache(CacheSpec(scheme="fp16")), use_cache=True
        )
    assert torch.equal(out_vanilla.logits, out_disp.logits)


def test_int4_kivi_dispatch_runs_and_changes_logits():
    model = _try_load_mistral()
    model.eval()
    tokens = torch.randint(0, 100, (1, 12))
    spec = CacheSpec(scheme="int4-kivi", bits=4, group_size=4, residual=2)
    with torch.no_grad():
        out_vanilla = model(tokens, past_key_values=DynamicCache(), use_cache=True)
        out_kivi = model(tokens, past_key_values=build_cache(spec), use_cache=True)
    assert out_kivi.logits.shape == out_vanilla.logits.shape
    assert not torch.equal(out_kivi.logits, out_vanilla.logits)
