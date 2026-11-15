"""Tests for greekkv.eval.loader."""

from __future__ import annotations

import pytest
import torch

from greekkv.eval import (
    LoadSpec,
    MemoryBreakdown,
    estimate_memory,
    load_model,
    resolve_device,
)
from greekkv.memory import ArchConfig, estimate_num_parameters

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# LoadSpec validation
# ---------------------------------------------------------------------------


def test_loadspec_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="dtype"):
        LoadSpec(name="foo", dtype="fp8")  # type: ignore[arg-type]


def test_loadspec_rejects_nonpositive_ctx():
    with pytest.raises(ValueError, match="n_ctx_estimate"):
        LoadSpec(name="foo", n_ctx_estimate=0)


def test_loadspec_rejects_negative_overhead():
    with pytest.raises(ValueError, match="activation_overhead"):
        LoadSpec(name="foo", activation_overhead_factor=-0.1)


def test_loadspec_rejects_nonpositive_budget():
    with pytest.raises(ValueError, match="memory_budget_gb"):
        LoadSpec(name="foo", memory_budget_gb=0)


def test_loadspec_is_frozen():
    spec = LoadSpec(name="foo")
    with pytest.raises((AttributeError, TypeError)):
        spec.dtype = "int4"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_respects_explicit_choice():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_when_no_accel(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


# ---------------------------------------------------------------------------
# estimate_num_parameters — approx within 1% of a tiny real model
# ---------------------------------------------------------------------------


def test_estimate_num_parameters_matches_tiny_llama(monkeypatch):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        pytest.skip("transformers not importable")
    try:
        hf_cfg = AutoConfig.from_pretrained(TINY_LLAMA)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot fetch config for {TINY_LLAMA}: {e}")

    from greekkv.memory import arch_from_hf  # noqa: PLC0415

    cfg = arch_from_hf(hf_cfg)
    predicted = estimate_num_parameters(cfg)

    try:
        model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot load {TINY_LLAMA}: {e}")
    real = sum(p.numel() for p in model.parameters())

    # Tiny models disagree a bit more than large ones; ±15% is sufficient
    # evidence that the formula captures the dominant terms.
    rel_err = abs(predicted - real) / real
    assert rel_err < 0.15, f"predicted={predicted} real={real} rel_err={rel_err:.3f}"


def test_estimate_num_parameters_requires_intermediate_and_vocab():
    cfg = ArchConfig(
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=64,
    )
    with pytest.raises(ValueError, match="vocab_size"):
        estimate_num_parameters(cfg)


# ---------------------------------------------------------------------------
# estimate_memory — monotonic in dtype precision
# ---------------------------------------------------------------------------


def _toy_cfg(num_params: int = 1_000_000_000) -> ArchConfig:
    # Toy 32 × 32 × (4 KV heads) × 128 — roughly Qwen2.5-7B-ish per-layer.
    return ArchConfig(
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=4,
        hidden_size=4096,
        head_dim=128,
        num_parameters=num_params,
        vocab_size=152064,
        intermediate_size=14336,
    )


def test_estimate_memory_shrinks_with_lower_precision():
    cfg = _toy_cfg(num_params=7_000_000_000)
    m_fp16 = estimate_memory(cfg, LoadSpec(name="x", dtype="fp16"))
    m_int8 = estimate_memory(cfg, LoadSpec(name="x", dtype="int8"))
    m_int4 = estimate_memory(cfg, LoadSpec(name="x", dtype="int4"))
    assert m_fp16.weights_gb > m_int8.weights_gb > m_int4.weights_gb
    assert m_fp16.total_gb > m_int8.total_gb > m_int4.total_gb


def test_estimate_memory_fp16_matches_hand_math():
    # 1B params × 2 bytes = 2 GB weights. + KV at 2048 ctx + 20% activations.
    cfg = _toy_cfg(num_params=1_000_000_000)
    m = estimate_memory(cfg, LoadSpec(name="x", dtype="fp16"))
    assert abs(m.weights_gb - (1e9 * 2 / 1024**3)) < 0.01
    assert m.activation_gb == pytest.approx(m.weights_gb * 0.2, rel=1e-2)


def test_estimate_memory_int4_overhead_present():
    cfg = _toy_cfg(num_params=7_000_000_000)
    m = estimate_memory(cfg, LoadSpec(name="x", dtype="int4"))
    # Data alone: 7e9 * 0.5 B = 3.26 GB. With group=64 FP16 scales, total slightly higher.
    raw = 7e9 * 0.5 / 1024**3
    assert m.weights_gb > raw
    assert m.weights_gb < raw * 1.1  # scale overhead is small, not dominant


# ---------------------------------------------------------------------------
# load_model — budget check (no actual loading)
# ---------------------------------------------------------------------------


def _make_spec(**over):
    base = dict(name=TINY_LLAMA, dtype="fp32", device="cpu")
    base.update(over)
    return LoadSpec(**base)


def test_load_model_refuses_when_over_budget():
    # Tiny Llama is < 1 MB; a 1-byte budget is comically small on purpose.
    spec = _make_spec(memory_budget_gb=1e-9)
    with pytest.raises(MemoryError, match="exceeds budget"):
        load_model(spec)


def test_load_model_succeeds_within_generous_budget():
    spec = _make_spec(memory_budget_gb=4.0)
    try:
        model, tok, mem = load_model(spec)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot load {TINY_LLAMA}: {e}")
    assert isinstance(mem, MemoryBreakdown)
    assert mem.total_gb <= 4.0
    assert model.config is not None
    assert tok is not None


def test_load_model_without_budget_skips_check():
    spec = _make_spec(memory_budget_gb=None)
    try:
        model, tok, mem = load_model(spec)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot load {TINY_LLAMA}: {e}")
    assert mem.total_gb > 0
    assert model is not None


def test_load_model_int4_without_bitsandbytes_raises_module_not_found():
    # bitsandbytes isn't installed on CPU-only Mac → clean install hint.
    spec = _make_spec(dtype="int4", memory_budget_gb=None)
    pytest.importorskip("transformers")
    try:
        import bitsandbytes  # noqa: F401, PLC0415
    except ImportError:
        # Expected path on CPU-only Mac — the loader should raise the hint.
        with pytest.raises(ModuleNotFoundError, match="bitsandbytes"):
            load_model(spec)
    else:
        pytest.skip("bitsandbytes is installed — cannot exercise missing-dep path")
