"""Model loader with a pre-flight VRAM budget check.

Design
------
The loader does three things in order:

1. Pull the HF config and derive :class:`greekkv.memory.ArchConfig`, filling
   in an estimated ``num_parameters`` if the config doesn't publish it.
2. Estimate weights + KV + activation memory at the requested precision
   (:func:`estimate_memory`). If a ``memory_budget_gb`` is set, refuse to
   load when the estimate exceeds it — a loud ``MemoryError`` beats an OOM
   crash deep inside bitsandbytes.
3. Build the appropriate HF loading kwargs (dtype for FP paths, a
   ``BitsAndBytesConfig`` for int8/int4) and call ``AutoModelForCausalLM``.

What's NOT in scope
-------------------
- Actual perplexity / benchmark evaluation (T16–T18).
- CPU-offloading and multi-GPU device maps (deferred until we have the
  hardware to test them honestly).
- Loading from local paths — only HF Hub IDs for now.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch

from greekkv.memory import (
    ArchConfig,
    arch_from_hf,
    estimate_num_parameters,
    kv_bytes,
    weights_bytes,
)

log = logging.getLogger(__name__)

Dtype = Literal["fp16", "bf16", "fp32", "int8", "int4"]
Device = Literal["auto", "cuda", "cpu", "mps"]

_BITS_FOR_DTYPE: dict[str, int] = {
    "fp32": 32,
    "fp16": 16,
    "bf16": 16,
    "int8": 8,
    "int4": 4,
}


@dataclass(frozen=True)
class LoadSpec:
    """Serializable load configuration.

    ``memory_budget_gb`` is the pre-flight ceiling. ``n_ctx_estimate`` is the
    context length the KV estimate is sized for (defaults to a common 2k).
    """

    name: str
    dtype: Dtype = "fp16"
    device: Device = "auto"
    memory_budget_gb: float | None = None
    n_ctx_estimate: int = 2048
    activation_overhead_factor: float = 0.2
    trust_remote_code: bool = False
    extra_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dtype not in _BITS_FOR_DTYPE:
            raise ValueError(f"unknown dtype {self.dtype!r}")
        if self.n_ctx_estimate <= 0:
            raise ValueError(f"n_ctx_estimate must be positive, got {self.n_ctx_estimate}")
        if self.activation_overhead_factor < 0:
            raise ValueError(
                f"activation_overhead_factor must be non-negative, "
                f"got {self.activation_overhead_factor}"
            )
        if self.memory_budget_gb is not None and self.memory_budget_gb <= 0:
            raise ValueError(
                f"memory_budget_gb must be positive if set, got {self.memory_budget_gb}"
            )


@dataclass(frozen=True)
class MemoryBreakdown:
    """Pre-flight memory estimate in gigabytes."""

    weights_gb: float
    kv_gb: float
    activation_gb: float
    total_gb: float


def resolve_device(device: Device) -> str:
    """Pick a concrete torch device string from the ``"auto"`` sentinel."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    # MPS is Apple Silicon. Fine for fp16/fp32; bitsandbytes needs CUDA.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_num_parameters(cfg: ArchConfig, name: str) -> ArchConfig:
    """Return ``cfg`` with ``num_parameters`` filled in (estimate if missing)."""
    if cfg.num_parameters is not None:
        return cfg
    try:
        n = estimate_num_parameters(cfg)
    except ValueError as e:
        raise ValueError(
            f"cannot estimate parameter count for {name!r}; "
            f"HF config is missing intermediate_size or vocab_size"
        ) from e
    return replace(cfg, num_parameters=n)


def estimate_memory(cfg: ArchConfig, spec: LoadSpec) -> MemoryBreakdown:
    """Estimate total runtime memory for loading + a short forward pass.

    KV is sized at ``n_ctx_estimate`` tokens, batch = 1, FP16 (no quantization)
    because the cache always starts fresh at FP16 until a quantized scheme
    kicks in.
    """
    bits = _BITS_FOR_DTYPE[spec.dtype]
    group_size = 64 if bits < 16 else 0  # matches bitsandbytes NF4 default
    w = weights_bytes(cfg, bits=bits, group_size=group_size)
    kv = kv_bytes(cfg, n_ctx=spec.n_ctx_estimate, bits=16)
    act = int(w * spec.activation_overhead_factor)
    gb = 1024**3
    return MemoryBreakdown(
        weights_gb=w / gb,
        kv_gb=kv / gb,
        activation_gb=act / gb,
        total_gb=(w + kv + act) / gb,
    )


def _require_bitsandbytes() -> None:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as e:
        raise ModuleNotFoundError(
            "int8/int4 model loading requires `bitsandbytes` (CUDA-only). "
            "Install with: pip install bitsandbytes"
        ) from e


def _build_from_pretrained_kwargs(spec: LoadSpec, device: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(spec.extra_from_pretrained_kwargs)
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True

    if spec.dtype == "fp16":
        kwargs["dtype"] = torch.float16
    elif spec.dtype == "bf16":
        kwargs["dtype"] = torch.bfloat16
    elif spec.dtype == "fp32":
        kwargs["dtype"] = torch.float32
    elif spec.dtype in ("int8", "int4"):
        _require_bitsandbytes()
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        if spec.dtype == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

    # bitsandbytes handles its own device placement.
    if device != "cpu" and spec.dtype not in ("int8", "int4"):
        kwargs["device_map"] = device
    return kwargs


def load_model(spec: LoadSpec) -> tuple[Any, Any, MemoryBreakdown]:
    """Load model + tokenizer with a pre-flight memory budget check.

    Returns ``(model, tokenizer, memory_breakdown)``. Raises
    :class:`MemoryError` if the estimate exceeds ``spec.memory_budget_gb``.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    device = resolve_device(spec.device)

    hf_cfg = AutoConfig.from_pretrained(spec.name, trust_remote_code=spec.trust_remote_code)
    cfg = arch_from_hf(hf_cfg)
    cfg = _ensure_num_parameters(cfg, spec.name)
    mem = estimate_memory(cfg, spec)

    if spec.memory_budget_gb is not None and mem.total_gb > spec.memory_budget_gb:
        raise MemoryError(
            f"Predicted memory {mem.total_gb:.2f} GB "
            f"(weights {mem.weights_gb:.2f}, KV {mem.kv_gb:.2f}, "
            f"activations {mem.activation_gb:.2f}) "
            f"exceeds budget {spec.memory_budget_gb:.2f} GB "
            f"for {spec.name!r} @ dtype={spec.dtype!r} on {device!r}."
        )

    log.info(
        "loading %s @ %s on %s — estimated %.2f GB total",
        spec.name,
        spec.dtype,
        device,
        mem.total_gb,
    )
    kwargs = _build_from_pretrained_kwargs(spec, device)
    model = AutoModelForCausalLM.from_pretrained(spec.name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(spec.name, trust_remote_code=spec.trust_remote_code)
    return model, tokenizer, mem
