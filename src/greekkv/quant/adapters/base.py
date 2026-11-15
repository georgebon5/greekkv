"""Common surface for the KV-cache backends used by the eval harness.

The protocol is deliberately **thin**: anything that can be handed to a
``model.forward(past_key_values=...)`` call of a modern transformers model
already satisfies it, including:

- ``transformers.cache_utils.DynamicCache`` (the FP16 baseline)
- :class:`greekkv.quant.FakeQuantCache` (T10)
- :class:`greekkv.quant.KiviCache` (T11)
- ``transformers.cache_utils.QuantizedCache`` with backend ``"quanto"`` or
  ``"HQQ"`` (via :mod:`greekkv.quant.adapters.hf_quantized`)

The point of the adapter layer is **not** to wrap these — they already share
an interface — but to give the eval harness a single string-keyed dispatch
(``CacheSpec.scheme``) so results tagged with that scheme can be reproduced
later from a JSONL row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

SUPPORTED_SCHEMES: tuple[str, ...] = (
    "fp16",
    "int8-fakequant",
    "int4-fakequant",
    "int8-kivi",
    "int4-kivi",
    "quanto",
    "hqq",
    "vllm-fp8",
)


@runtime_checkable
class KVCacheBackend(Protocol):
    """Minimal surface of a KV cache that the eval harness uses.

    Any ``transformers.cache_utils.Cache`` subclass satisfies this.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def get_seq_length(self, layer_idx: int = 0) -> int: ...


@dataclass(frozen=True)
class CacheSpec:
    """Serializable description of a KV-cache choice.

    ``scheme`` is the primary dispatch key for :func:`build_cache`. The
    remaining fields are interpreted per scheme (e.g. ``residual`` is only
    meaningful for KIVI and HF QuantizedCache; FP16 ignores it).
    """

    scheme: str
    bits: int = 16
    group_size: int = 128
    residual: int = 0
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.scheme not in SUPPORTED_SCHEMES:
            raise ValueError(
                f"unknown cache scheme: {self.scheme!r}; supported: {SUPPORTED_SCHEMES}"
            )
        if self.bits <= 0:
            raise ValueError(f"bits must be positive, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.residual < 0:
            raise ValueError(f"residual must be non-negative, got {self.residual}")


def build_cache(spec: CacheSpec, model_config: Any = None) -> KVCacheBackend:
    """Instantiate a cache per ``spec``.

    Parameters
    ----------
    spec
        The :class:`CacheSpec`.
    model_config
        Required for HF QuantizedCache backends (``quanto``, ``hqq``) because
        ``transformers.cache_utils.QuantizedCache`` needs the model's
        ``PretrainedConfig`` at construction. Ignored for native schemes.
    """
    # Local imports keep this file torch-dep-free at import time and sidestep
    # circular imports with greekkv.quant.
    from transformers.cache_utils import DynamicCache  # noqa: PLC0415

    from greekkv.quant.fakequant_cache import FakeQuantCache  # noqa: PLC0415
    from greekkv.quant.kivi_cache import KiviCache, KiviCacheConfig  # noqa: PLC0415

    s = spec.scheme
    if s == "fp16":
        return DynamicCache()

    if s in ("int4-fakequant", "int8-fakequant"):
        return FakeQuantCache(
            bits=spec.bits,
            group_size=spec.group_size,
            symmetric=spec.symmetric,
        )

    if s in ("int4-kivi", "int8-kivi"):
        return KiviCache(
            KiviCacheConfig(
                bits=spec.bits,
                group_size=spec.group_size,
                residual=spec.residual,
                symmetric=spec.symmetric,
            )
        )

    if s in ("quanto", "hqq"):
        if model_config is None:
            raise ValueError(
                f"model_config is required for scheme={s!r} "
                "(HF QuantizedCache needs the model's PretrainedConfig)"
            )
        from greekkv.quant.adapters.hf_quantized import build_hf_quantized  # noqa: PLC0415

        return build_hf_quantized(backend=s, model_config=model_config, spec=spec)

    if s == "vllm-fp8":
        from greekkv.quant.adapters.vllm_fp8 import build_vllm_fp8  # noqa: PLC0415

        return build_vllm_fp8(spec)

    # Unreachable thanks to __post_init__; explicit for defence in depth.
    raise ValueError(f"unknown cache scheme: {s!r}")
