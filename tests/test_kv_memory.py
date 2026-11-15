"""Tests for greekkv.memory.kv.

Two tiers:
- Algebraic tests — pure formula, no network, no torch.
- Integration tests — load a tiny public HF model and compare the formula
  against the real PyTorch tensors. Skipped if the model cannot be loaded
  (e.g. offline CI).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from greekkv.memory import (
    ArchConfig,
    arch_from_hf,
    head_dim_of,
    kv_bytes,
    weights_bytes,
)

# ---------------------------------------------------------------------------
# Algebraic tests (no torch, no network)
# ---------------------------------------------------------------------------


def _cfg(**over: Any) -> ArchConfig:
    base = dict(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=16,
    )
    base.update(over)
    return ArchConfig(**base)


def test_head_dim_derived_from_hidden():
    assert head_dim_of(_cfg(hidden_size=128, num_attention_heads=8)) == 16


def test_head_dim_uses_explicit_when_set():
    # Llama 3.1/3.2 pattern: hidden_size / num_heads disagrees with head_dim.
    cfg = _cfg(hidden_size=8, num_attention_heads=4, head_dim=8)
    assert head_dim_of(cfg) == 8


def test_head_dim_raises_when_not_divisible():
    with pytest.raises(ValueError):
        head_dim_of(_cfg(hidden_size=10, num_attention_heads=4))


def test_kv_bytes_fp16_formula():
    # L=2, H_kv=4, d_head=4, n_ctx=8, bits=16, batch=1
    # elems = 2 * 2 * 4 * 8 * 4 = 512 (K+V together)
    # bytes = 512 * 2 = 1024
    assert kv_bytes(_cfg(), n_ctx=8, bits=16) == 1024


def test_kv_bytes_zero_ctx_is_zero():
    assert kv_bytes(_cfg(), n_ctx=0, bits=16) == 0


def test_kv_bytes_bits_scaling():
    base = kv_bytes(_cfg(), n_ctx=64, bits=16, group_size=0)
    half = kv_bytes(_cfg(), n_ctx=64, bits=8, group_size=0)
    quarter = kv_bytes(_cfg(), n_ctx=64, bits=4, group_size=0)
    assert half == base // 2
    assert quarter == base // 4


def test_kv_bytes_gqa_reduces_by_ratio():
    non_gqa = kv_bytes(_cfg(num_key_value_heads=4), n_ctx=64, bits=16)
    gqa_2to1 = kv_bytes(_cfg(num_key_value_heads=2), n_ctx=64, bits=16)
    gqa_4to1 = kv_bytes(_cfg(num_key_value_heads=1), n_ctx=64, bits=16)
    assert gqa_2to1 == non_gqa // 2
    assert gqa_4to1 == non_gqa // 4


def test_kv_bytes_batch_scales_linearly():
    one = kv_bytes(_cfg(), n_ctx=32, bits=16, batch_size=1)
    four = kv_bytes(_cfg(), n_ctx=32, bits=16, batch_size=4)
    assert four == one * 4


def test_kv_bytes_quant_overhead_decreases_with_larger_group():
    args = dict(cfg=_cfg(), n_ctx=256, bits=4)
    small = kv_bytes(**args, group_size=32)
    large = kv_bytes(**args, group_size=256)
    # Same data bytes; large group → fewer scale slots → smaller total.
    assert large < small


def test_kv_bytes_asymmetric_adds_zero_point_overhead():
    args = dict(cfg=_cfg(), n_ctx=256, bits=4, group_size=64)
    sym = kv_bytes(**args, asymmetric=False)
    asym = kv_bytes(**args, asymmetric=True)
    assert asym > sym


def test_kv_bytes_group_zero_is_no_overhead():
    # group_size=0 means raw packed bits (as if the scale were already elsewhere).
    raw = kv_bytes(_cfg(), n_ctx=64, bits=8, group_size=0)
    data_only = int(2 * 2 * 4 * 4 * 64 * 8 / 8)  # 2 * L * H_kv * d * n_ctx * bits/8
    assert raw == data_only


def test_kv_bytes_rejects_invalid_args():
    cfg = _cfg()
    with pytest.raises(ValueError):
        kv_bytes(cfg, n_ctx=-1, bits=16)
    with pytest.raises(ValueError):
        kv_bytes(cfg, n_ctx=8, bits=0)
    with pytest.raises(ValueError):
        kv_bytes(cfg, n_ctx=8, bits=16, group_size=-1)
    with pytest.raises(ValueError):
        kv_bytes(cfg, n_ctx=8, bits=16, batch_size=0)


def test_weights_bytes_requires_num_parameters():
    cfg = _cfg(num_parameters=None)
    with pytest.raises(ValueError):
        weights_bytes(cfg, bits=16)


def test_weights_bytes_bits_scaling():
    cfg = _cfg(num_parameters=1_000_000)
    assert weights_bytes(cfg, bits=16) == 2_000_000
    assert weights_bytes(cfg, bits=8) == 1_000_000
    assert weights_bytes(cfg, bits=4) == 500_000


def test_weights_bytes_quant_overhead():
    cfg = _cfg(num_parameters=1024)
    raw_4bit = weights_bytes(cfg, bits=4, group_size=0)
    grouped_4bit = weights_bytes(cfg, bits=4, group_size=64)
    # raw=1024*4/8=512; overhead=ceil(1024/64)*2=32; total=544
    assert raw_4bit == 512
    assert grouped_4bit == 544


# ---------------------------------------------------------------------------
# arch_from_hf
# ---------------------------------------------------------------------------


def test_arch_from_hf_dict_non_gqa():
    cfg = arch_from_hf(
        dict(
            num_hidden_layers=12,
            num_attention_heads=16,
            hidden_size=1024,
            vocab_size=32000,
        )
    )
    assert cfg.num_key_value_heads == 16  # defaults to num_attention_heads
    assert cfg.head_dim is None
    assert head_dim_of(cfg) == 64


def test_arch_from_hf_dict_gqa():
    cfg = arch_from_hf(
        dict(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        )
    )
    assert cfg.num_key_value_heads == 8
    assert head_dim_of(cfg) == 128


def test_arch_from_hf_dict_explicit_head_dim():
    cfg = arch_from_hf(
        dict(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=8,
            head_dim=8,  # Llama 3.1+ style
        )
    )
    assert cfg.head_dim == 8
    assert head_dim_of(cfg) == 8


# ---------------------------------------------------------------------------
# Integration tests with real tiny HF models (torch + network)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

try:  # noqa: SIM105 — pytest.skip requires module-level condition
    from transformers import AutoConfig, AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoConfig = AutoModelForCausalLM = None  # type: ignore[assignment]


TINY_NON_GQA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_GQA = "hf-internal-testing/tiny-random-MistralForCausalLM"
TINY_LLAMA32 = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


def _try_load(model_id: str):
    """Load a tiny HF model or skip the test if offline / gated."""
    if AutoModelForCausalLM is None:
        pytest.skip("transformers not importable")
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot load {model_id} (likely offline): {e}")


def _iter_kv_tensors(pkv) -> list:
    """Yield K and V tensors from past_key_values, handling DynamicCache + tuple formats."""
    tensors = []
    if hasattr(pkv, "layers"):  # DynamicCache (transformers >= ~4.44 refactor)
        for layer in pkv.layers:
            for name in ("keys", "values", "key_cache", "value_cache"):
                t = getattr(layer, name, None)
                if isinstance(t, torch.Tensor):
                    tensors.append(t)
    else:  # legacy tuple-of-tuples
        for entry in pkv:
            k, v = entry[0], entry[1]
            tensors.extend([k, v])
    return tensors


def _real_kv_bytes(model, n_ctx: int) -> int:
    tokens = torch.randint(0, 100, (1, n_ctx))
    model.eval()
    with torch.no_grad():
        out = model(tokens, use_cache=True)
    tensors = _iter_kv_tensors(out.past_key_values)
    # Each layer should contribute exactly one K and one V tensor.
    expected_n = 2 * model.config.num_hidden_layers
    assert len(tensors) == expected_n, f"got {len(tensors)} KV tensors, expected {expected_n}"
    return sum(t.numel() * t.element_size() for t in tensors)


@pytest.mark.parametrize(
    "model_id, expect_gqa, expect_explicit_head_dim",
    [
        (TINY_NON_GQA, False, False),
        (TINY_GQA, True, False),
        (TINY_LLAMA32, True, True),
    ],
)
def test_kv_bytes_matches_real_tensors(model_id, expect_gqa, expect_explicit_head_dim):
    model = _try_load(model_id)
    cfg = arch_from_hf(model.config)

    # Sanity: config shape matches what we think the model is.
    if expect_gqa:
        assert cfg.num_key_value_heads < cfg.num_attention_heads
    else:
        assert cfg.num_key_value_heads == cfg.num_attention_heads
    if expect_explicit_head_dim:
        assert cfg.head_dim is not None

    for n_ctx in (4, 16):
        real = _real_kv_bytes(model, n_ctx=n_ctx)
        pred = kv_bytes(cfg, n_ctx=n_ctx, bits=16, group_size=0)
        assert pred == real, (
            f"{model_id} n_ctx={n_ctx}: formula={pred} real={real} "
            f"(cfg: L={cfg.num_hidden_layers} H_kv={cfg.num_key_value_heads} "
            f"d_head={head_dim_of(cfg)})"
        )


def test_weights_bytes_matches_real_parameters():
    model = _try_load(TINY_NON_GQA)
    real_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    real_params = sum(p.numel() for p in model.parameters())

    cfg = arch_from_hf(model.config)
    cfg = ArchConfig(**{**cfg.__dict__, "num_parameters": real_params})

    assert weights_bytes(cfg, bits=16, group_size=0) == real_bytes
    # Int8: raw count. Allow equality within 1 byte for the math.ceil rounding.
    assert math.isclose(weights_bytes(cfg, bits=8, group_size=0), real_params, abs_tol=1)
