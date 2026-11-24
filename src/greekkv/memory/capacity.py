"""Capacity model (T28).

Translates a (model, VRAM budget, sequence length, cache scheme) tuple
into the **maximum number of concurrent users** that fit alongside the
weights and a fixed runtime overhead. The arithmetic itself is
boring: it stacks :mod:`greekkv.memory.kv` byte formulas and divides.
What matters is making it *honest* about which inputs are measured vs
estimated — the only way a capacity number is useful is if the reader
can tell whether it came out of the profiler or out of a formula.

Three cache behaviours are modelled explicitly, matching the schemes
measured in T17 / T20 / T23:

- ``fp16`` / ``int8-fakequant`` / ``int4-fakequant`` — the full KV cache
  lives at the stated precision. Overhead per group is already handled
  by :func:`greekkv.memory.kv.kv_bytes`.
- ``int8-kivi`` / ``int4-kivi`` — recent tokens stay in FP16 inside a
  residual window of ``n_sink + n_recent`` tokens; the evicted tail is
  quantized. Bytes decompose as ``fp16(min(n_ctx, window)) +
  quant(max(0, n_ctx - window))``.
- ``eviction-sink`` — the cache is bounded to the sink+recent window
  regardless of ``n_ctx``. Bytes are ``fp16(min(n_ctx, window))``.

The runtime overhead (CUDA context, activations, framework allocator)
is held in one place — :data:`DEFAULT_OVERHEAD_BYTES` — so a single
knob moves every row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from greekkv.memory.kv import ArchConfig, kv_bytes, weights_bytes

# 650 MB is the median post-load allocator state we have seen on the
# T23 profiler across schemes on free-tier Kaggle T4 / Colab T4, after
# subtracting the weight bytes. Reviewers who have a better local
# number can override it with `CapacityInputs.overhead_bytes`.
DEFAULT_OVERHEAD_BYTES = 650 * 1024 * 1024

MIB = 1024 * 1024

SchemeName = Literal[
    "fp16",
    "int8-fakequant",
    "int8-kivi",
    "int4-fakequant",
    "int4-kivi",
    "eviction-sink",
]

ALL_SCHEMES: tuple[SchemeName, ...] = (
    "fp16",
    "int8-fakequant",
    "int8-kivi",
    "int4-fakequant",
    "int4-kivi",
    "eviction-sink",
)


@dataclass(frozen=True)
class SchemeSpec:
    """Byte-level description of a cache scheme.

    - ``bits_kv`` — precision of the quantized portion of the cache.
    - ``group_size`` — group-wise scale granularity (KIVI/fake-quant).
    - ``residual_tokens`` — FP16 residual window used by KIVI /
      eviction-sink schemes. ``0`` for the non-residual variants.
    - ``is_bounded`` — if True, the cache is capped at
      ``residual_tokens`` regardless of ``n_ctx`` (``eviction-sink``).
    """

    bits_kv: int
    group_size: int = 0
    residual_tokens: int = 0
    is_bounded: bool = False


# Canonical specs. These mirror the knobs recorded in the T17 / T20 /
# T23 JSONL rows (``metrics.bits``, ``metrics.group_size``,
# ``metrics.n_sink + metrics.n_recent``). Keeping them here lets the
# capacity model stay data-driven without silently guessing when a
# measured row is absent.
DEFAULT_SCHEME_SPECS: dict[SchemeName, SchemeSpec] = {
    "fp16": SchemeSpec(bits_kv=16),
    "int8-fakequant": SchemeSpec(bits_kv=8, group_size=64),
    "int8-kivi": SchemeSpec(bits_kv=8, group_size=64, residual_tokens=36),
    "int4-fakequant": SchemeSpec(bits_kv=4, group_size=64),
    "int4-kivi": SchemeSpec(bits_kv=4, group_size=64, residual_tokens=36),
    "eviction-sink": SchemeSpec(bits_kv=16, residual_tokens=36, is_bounded=True),
}


@dataclass(frozen=True)
class CapacityInputs:
    """Arguments to :func:`capacity`.

    ``measured_*`` fields are optional and get copied into the result for
    reporting; they do not affect the byte arithmetic (that's
    deliberately pure so capacity stays comparable across measurements).
    The ``source`` field on the result records which path was taken.
    """

    cfg: ArchConfig
    vram_bytes: int
    n_ctx: int
    scheme: SchemeName
    bits_weights: int = 16
    weight_group_size: int = 0
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES
    scheme_specs: dict[SchemeName, SchemeSpec] = field(default_factory=dict)

    # Measured passthroughs — not part of the arithmetic.
    measured_ppl: float | None = None
    measured_decode_ms: float | None = None
    source: Literal["measured", "estimated"] = "estimated"


@dataclass(frozen=True)
class CapacityResult:
    """Output of :func:`capacity`.

    Every byte number is a positive integer so downstream code (CLI,
    tests, planner web app) can do strict arithmetic without worrying
    about float rounding.
    """

    scheme: SchemeName
    n_ctx: int
    weight_bytes: int
    overhead_bytes: int
    kv_bytes_per_user: int
    free_bytes_after_weights: int
    max_concurrent_users: int
    source: Literal["measured", "estimated"]
    measured_ppl: float | None = None
    measured_decode_ms: float | None = None


def _kv_bytes_for_scheme(cfg: ArchConfig, n_ctx: int, spec: SchemeSpec) -> int:
    """Bytes for one user's KV cache under ``spec``.

    Combines FP16 residual + quantized tail for KIVI, bounds the cache
    for eviction-sink, otherwise reduces to the plain :func:`kv_bytes`
    formula.
    """
    if n_ctx == 0:
        return 0

    if spec.is_bounded:
        # Cache is capped at the residual window at FP16, regardless of n_ctx.
        tokens = min(n_ctx, spec.residual_tokens)
        return kv_bytes(cfg, tokens, bits=16, group_size=0)

    if spec.residual_tokens > 0 and spec.bits_kv < 16:
        residual = min(n_ctx, spec.residual_tokens)
        tail = max(0, n_ctx - residual)
        fp16_part = kv_bytes(cfg, residual, bits=16, group_size=0)
        quant_part = kv_bytes(cfg, tail, bits=spec.bits_kv, group_size=spec.group_size)
        return fp16_part + quant_part

    return kv_bytes(cfg, n_ctx, bits=spec.bits_kv, group_size=spec.group_size)


def capacity(inputs: CapacityInputs) -> CapacityResult:
    """Return the capacity breakdown for ``inputs``.

    The scheme is resolved against ``inputs.scheme_specs`` first, then
    against :data:`DEFAULT_SCHEME_SPECS`, so callers can override one
    scheme (for example, ``int8-kivi`` with ``residual_tokens=64``)
    without hand-copying the rest.

    Raises
    ------
    ValueError
        If ``scheme`` is unknown, or ``vram_bytes`` / ``n_ctx`` are
        negative.
    """
    if inputs.vram_bytes < 0:
        raise ValueError(f"vram_bytes must be non-negative, got {inputs.vram_bytes}")
    if inputs.n_ctx < 0:
        raise ValueError(f"n_ctx must be non-negative, got {inputs.n_ctx}")

    scheme = inputs.scheme
    spec = inputs.scheme_specs.get(scheme) or DEFAULT_SCHEME_SPECS.get(scheme)
    if spec is None:
        raise ValueError(f"unknown scheme {scheme!r}; known: {sorted(DEFAULT_SCHEME_SPECS)}")

    w_bytes = weights_bytes(
        inputs.cfg, bits=inputs.bits_weights, group_size=inputs.weight_group_size
    )
    kv_per_user = _kv_bytes_for_scheme(inputs.cfg, inputs.n_ctx, spec)

    free_after_weights = inputs.vram_bytes - w_bytes - inputs.overhead_bytes
    users = 0 if kv_per_user <= 0 or free_after_weights <= 0 else free_after_weights // kv_per_user

    return CapacityResult(
        scheme=scheme,
        n_ctx=inputs.n_ctx,
        weight_bytes=w_bytes,
        overhead_bytes=inputs.overhead_bytes,
        kv_bytes_per_user=kv_per_user,
        free_bytes_after_weights=max(0, free_after_weights),
        max_concurrent_users=users,
        source=inputs.source,
        measured_ppl=inputs.measured_ppl,
        measured_decode_ms=inputs.measured_decode_ms,
    )
