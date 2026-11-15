"""KV-cache and weight memory formulas.

The central formula for KV-cache size (per sequence, FP16, no overhead) is::

    M_KV = 2 · L · H_kv · d_head · n_ctx · (bits / 8)

where ``L`` is the number of transformer layers, ``H_kv`` is the number of
*key-value* heads (``= num_attention_heads`` without GQA, smaller with it),
``d_head`` is the per-head dimension, ``n_ctx`` is the sequence length, and the
factor of ``2`` accounts for K and V both being stored.

For sub-16-bit precisions we add the overhead of group-wise scales (and
optionally zero-points for asymmetric quantization). The approximation
``n_groups = ceil(n_elems / group_size)`` treats the tensor as a flat buffer,
which is correct in aggregate and matches how packed weights are laid out in
practice (e.g. bitsandbytes NF4, AWQ, GPTQ).

This module is torch-free: it only consumes config data. Comparisons against
real PyTorch tensors live in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# All group-wise quantizers we care about keep the scale in FP16 (2 bytes).
BYTES_PER_SCALE = 2


@dataclass(frozen=True)
class ArchConfig:
    """Normalized subset of a transformers ``PretrainedConfig``.

    - ``num_key_value_heads`` must equal ``num_attention_heads`` for non-GQA
      models.
    - ``head_dim`` is optional because many architectures derive it from
      ``hidden_size // num_attention_heads``. Llama 3.1+ sets it independently,
      which is why we expose the field.
    - ``num_parameters`` is required for :func:`weights_bytes` but not for
      :func:`kv_bytes`.
    """

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int | None = None
    num_parameters: int | None = None
    vocab_size: int | None = None


def head_dim_of(cfg: ArchConfig) -> int:
    """Return the effective per-head dimension.

    Uses the explicit ``head_dim`` if set (Llama 3.1+ style) and otherwise
    computes ``hidden_size // num_attention_heads``.
    """
    if cfg.head_dim is not None:
        return cfg.head_dim
    if cfg.hidden_size % cfg.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size={cfg.hidden_size} not divisible by "
            f"num_attention_heads={cfg.num_attention_heads}; set head_dim explicitly"
        )
    return cfg.hidden_size // cfg.num_attention_heads


def arch_from_hf(hf_cfg: Any) -> ArchConfig:
    """Build an :class:`ArchConfig` from a transformers ``PretrainedConfig`` or dict.

    Handles missing ``num_key_value_heads`` (pre-GQA models default to
    ``num_attention_heads``) and optional ``head_dim`` /
    ``num_parameters`` / ``vocab_size``.
    """
    if isinstance(hf_cfg, dict):

        def _g(k: str) -> Any:
            return hf_cfg.get(k)
    else:

        def _g(k: str) -> Any:
            return getattr(hf_cfg, k, None)

    n_attn = _g("num_attention_heads")
    n_kv = _g("num_key_value_heads")
    if n_kv is None:
        n_kv = n_attn  # pre-GQA models (Llama 1/2) omit this field
    return ArchConfig(
        num_hidden_layers=_g("num_hidden_layers"),
        num_attention_heads=n_attn,
        num_key_value_heads=n_kv,
        hidden_size=_g("hidden_size"),
        head_dim=_g("head_dim"),
        num_parameters=_g("num_parameters"),
        vocab_size=_g("vocab_size"),
    )


def _quant_overhead_bytes(n_elems: int, bits: int, group_size: int, asymmetric: bool) -> float:
    """Bytes consumed by per-group scales (and zero-points if ``asymmetric``).

    For ``bits >= 16`` or ``group_size <= 0`` we assume no quantization, so
    the overhead is zero.
    """
    if bits >= 16 or group_size <= 0:
        return 0.0
    n_groups = math.ceil(n_elems / group_size)
    scale_bytes = n_groups * BYTES_PER_SCALE
    zp_bytes = n_groups * bits / 8 if asymmetric else 0.0
    return scale_bytes + zp_bytes


def kv_bytes(
    cfg: ArchConfig,
    n_ctx: int,
    bits: int = 16,
    group_size: int = 0,
    asymmetric: bool = False,
    batch_size: int = 1,
) -> int:
    """Bytes for the KV cache at precision ``bits``.

    Parameters
    ----------
    cfg
        Architecture config. ``head_dim`` is used if set (Llama 3.1+), else
        derived from ``hidden_size / num_attention_heads``.
    n_ctx
        Sequence length (tokens held in cache). ``0`` returns ``0``.
    bits
        Bits per stored KV value. Values ``>=16`` skip the overhead term.
    group_size
        Size of per-group scale window. ``0`` means no grouping (and no
        overhead, which is the FP16 case).
    asymmetric
        If True, add a per-group zero-point of the same bit-width.
    batch_size
        Number of parallel sequences in the cache (default 1).
    """
    if n_ctx < 0 or bits <= 0 or group_size < 0 or batch_size < 1:
        raise ValueError(
            f"invalid args: n_ctx={n_ctx} bits={bits} "
            f"group_size={group_size} batch_size={batch_size}"
        )
    if n_ctx == 0:
        return 0

    d = head_dim_of(cfg)
    # Per K-tensor element count; V is identical, hence the ``2 *`` below.
    per_tensor = batch_size * cfg.num_hidden_layers * cfg.num_key_value_heads * n_ctx * d
    kv_elems = 2 * per_tensor
    data_bytes = kv_elems * bits / 8
    overhead = _quant_overhead_bytes(kv_elems, bits, group_size, asymmetric)
    return math.ceil(data_bytes + overhead)


def weights_bytes(
    cfg: ArchConfig,
    bits: int = 16,
    group_size: int = 0,
    asymmetric: bool = False,
) -> int:
    """Bytes for model weights at precision ``bits``.

    Requires ``cfg.num_parameters`` to be set — we never guess parameter count
    from architecture because the number of tied/untied embeddings alone can
    swing the answer by hundreds of MB on large vocabularies.
    """
    if cfg.num_parameters is None:
        raise ValueError(
            "ArchConfig.num_parameters must be set for weights_bytes "
            "(use arch_from_hf on a config that has num_parameters, or set it manually)"
        )
    if bits <= 0 or group_size < 0:
        raise ValueError(f"invalid args: bits={bits} group_size={group_size}")

    data_bytes = cfg.num_parameters * bits / 8
    overhead = _quant_overhead_bytes(cfg.num_parameters, bits, group_size, asymmetric)
    return math.ceil(data_bytes + overhead)
