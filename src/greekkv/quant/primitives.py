"""Low-level quantization primitives used by every cache in this project.

Four public functions:

- :func:`quantize`      — any tensor → :class:`QuantizedTensor`.
- :func:`dequantize`    — :class:`QuantizedTensor` → float tensor.
- :func:`pack_int4`     — pack an int tensor with values in ``[-8, 7]`` or
  ``[0, 15]`` into ``uint8`` (two values per byte).
- :func:`unpack_int4`   — inverse of :func:`pack_int4`.

Design choices
--------------
- **Grouping along one axis.** A quantizer is fully specified by
  ``(bits, group_size, axis, symmetric)``. Groups of ``group_size`` consecutive
  elements along ``axis`` share one scale (and one zero-point when asymmetric).
  All other axes are independent.
- **Zero-padding the tail.** If the group axis is not divisible by
  ``group_size`` we pad with zeros to the next multiple. Zeros do not change
  ``max(|x|)`` so the first (``n_real``-wide) group's scale is preserved.
  After :func:`dequantize` we slice the padding off, so round-trip shape
  equals input shape exactly.
- **INT4 is always packed.** :func:`quantize` with ``bits=4`` stores a packed
  ``uint8`` buffer internally. :func:`pack_int4` / :func:`unpack_int4` are
  exposed for callers that want to manipulate packed tensors directly.
- **Scales stored in FP16.** Matches what every real KV-cache implementation
  does (KIVI, bitsandbytes, AWQ, GPTQ).

Shape conventions
-----------------
For ``x`` of shape ``(..., D)`` quantized along the last axis with
``group_size = G`` (``D % G == 0`` after padding):

- ``data.shape`` is ``(..., D)`` when ``bits == 8`` and ``(..., D // 2)`` when
  ``bits == 4``.
- ``scales.shape`` is ``(..., D // G)``.
- ``zero_points`` has the same shape as ``scales`` (``None`` when symmetric).

The module is CPU-only and uses only vanilla ``torch`` ops — no CUDA kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

# FP16 is universal for quant scales in KV-cache land.
SCALE_DTYPE = torch.float16

# Tiny floor to avoid 0 scales for all-zero groups.
_EPS = 1e-8


@dataclass(frozen=True)
class QuantMeta:
    bits: int
    group_size: int
    axis: int
    symmetric: bool
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype


@dataclass
class QuantizedTensor:
    """The packed result of :func:`quantize`.

    - For ``bits=8``: ``data`` is ``torch.int8`` (symmetric) or ``torch.uint8``
      (asymmetric); shape matches the padded tensor.
    - For ``bits=4``: ``data`` is ``torch.uint8`` and holds two nibbles per
      byte (last axis halved).
    - Metadata is enough to reverse every shape operation and recover the
      original tensor up to quantization error.
    """

    data: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    meta: QuantMeta

    @property
    def bits(self) -> int:
        return self.meta.bits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quantize(
    x: torch.Tensor,
    bits: int,
    group_size: int = 128,
    axis: int = -1,
    symmetric: bool = True,
) -> QuantizedTensor:
    """Quantize ``x`` along ``axis`` with per-group scales.

    Parameters
    ----------
    x
        Any floating tensor. Shape is preserved by the round-trip.
    bits
        4 or 8. Any other value raises.
    group_size
        Elements per group along ``axis``. The axis is zero-padded to a
        multiple of ``group_size`` before quantization.
    axis
        Axis to group along. Negative indexing supported.
    symmetric
        If True, zero-point is 0 and the range is ``[-qmax, qmax]``. If False,
        zero-point is per-group and the range is ``[0, 2*qmax + 1]``.
    """
    _validate_args(bits=bits, group_size=group_size)
    if not x.is_floating_point():
        raise TypeError(f"quantize expects a float tensor, got {x.dtype}")

    axis_pos = _normalize_axis(axis, x.ndim)
    original_shape = tuple(x.shape)
    original_dtype = x.dtype

    # Move the group axis to the end so we only ever reshape the last dim.
    x_last = x.movedim(axis_pos, -1).contiguous()
    original_last = x_last.shape[-1]
    pad = (-original_last) % group_size
    if pad:
        x_last = F.pad(x_last, (0, pad))
    assert x_last.shape[-1] % group_size == 0

    # Shape to (..., n_groups, G) — stats + quantize on the last dim.
    g_shape = (*x_last.shape[:-1], x_last.shape[-1] // group_size, group_size)
    xg = x_last.reshape(g_shape).to(torch.float32)

    if symmetric:
        qmax = (1 << (bits - 1)) - 1  # 7 for int4, 127 for int8
        max_abs = xg.abs().amax(dim=-1, keepdim=True).clamp_min(_EPS)
        scales = max_abs / qmax
        q = (xg / scales).round().clamp_(-qmax, qmax)
        zero_points: torch.Tensor | None = None
    else:
        qmax = (1 << bits) - 1  # 15 for int4, 255 for int8
        x_min = xg.amin(dim=-1, keepdim=True)
        x_max = xg.amax(dim=-1, keepdim=True)
        scales = ((x_max - x_min) / qmax).clamp_min(_EPS)
        zero_points = (-x_min / scales).round().clamp_(0, qmax)
        q = (xg / scales + zero_points).round().clamp_(0, qmax)

    # Scales + zero-points: squeeze the group-interior dim and keep FP16.
    scales_out = scales.squeeze(-1).to(SCALE_DTYPE)
    zp_out: torch.Tensor | None = None
    if zero_points is not None:
        # For int4 asymmetric we keep zero-points in uint8 (values fit in 4 bits
        # but we store full bytes; packing the zero-points is out of scope).
        zp_out = zero_points.squeeze(-1).to(torch.uint8)

    # Flatten the group dim back; store integers (and pack if bits == 4).
    q_flat = q.reshape(*x_last.shape).to(torch.int8 if symmetric else torch.uint8)
    if bits == 4:
        signed = symmetric
        data = pack_int4(q_flat, signed=signed)
    else:
        data = q_flat

    # Undo the axis move for data and scales so callers see familiar layouts.
    data = data.movedim(-1, axis_pos)
    scales_out = scales_out.movedim(-1, axis_pos)
    if zp_out is not None:
        zp_out = zp_out.movedim(-1, axis_pos)

    meta = QuantMeta(
        bits=bits,
        group_size=group_size,
        axis=axis_pos,
        symmetric=symmetric,
        original_shape=original_shape,
        original_dtype=original_dtype,
    )
    return QuantizedTensor(data=data, scales=scales_out, zero_points=zp_out, meta=meta)


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """Inverse of :func:`quantize`. Returns a tensor with the input shape + dtype."""
    bits = qt.meta.bits
    axis = qt.meta.axis
    group_size = qt.meta.group_size
    original_shape = qt.meta.original_shape
    original_dtype = qt.meta.original_dtype
    original_last = original_shape[axis if axis >= 0 else axis + len(original_shape)]

    # Move the group axis to the last position to mirror quantize().
    data = qt.data.movedim(axis, -1)
    scales = qt.scales.movedim(axis, -1)
    zp = qt.zero_points.movedim(axis, -1) if qt.zero_points is not None else None

    # Unpack if int4.
    if bits == 4:
        padded_last = data.shape[-1] * 2
        q_int = unpack_int4(data, n_values=padded_last, signed=qt.meta.symmetric)
    else:
        q_int = data

    padded_last = q_int.shape[-1]
    # Shape (..., n_groups, G)
    g_shape = (*q_int.shape[:-1], padded_last // group_size, group_size)
    qg = q_int.reshape(g_shape).to(torch.float32)
    sg = scales.unsqueeze(-1).to(torch.float32)
    if zp is None:
        xg = qg * sg
    else:
        zpg = zp.unsqueeze(-1).to(torch.float32)
        xg = (qg - zpg) * sg

    x_last = xg.reshape(*q_int.shape)
    # Slice off padding.
    x_last = x_last[..., :original_last]
    out = x_last.movedim(-1, axis)
    return out.to(original_dtype)


def pack_int4(x: torch.Tensor, signed: bool = True) -> torch.Tensor:
    """Pack an integer tensor whose last axis has an even length into ``uint8``.

    ``signed=True`` expects values in ``[-8, 7]``; ``signed=False`` expects
    ``[0, 15]``. Negative values are stored as their two's-complement nibbles,
    which :func:`unpack_int4` reverses.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last axis must be even for int4 packing, got {x.shape[-1]}")
    lo = x[..., 0::2]
    hi = x[..., 1::2]
    if signed:
        if (lo < -8).any() or (lo > 7).any() or (hi < -8).any() or (hi > 7).any():
            raise ValueError("signed int4 values must be in [-8, 7]")
    else:
        if (lo < 0).any() or (lo > 15).any() or (hi < 0).any() or (hi > 15).any():
            raise ValueError("unsigned int4 values must be in [0, 15]")
    lo_u = lo.to(torch.int16) & 0x0F
    hi_u = hi.to(torch.int16) & 0x0F
    packed = ((hi_u << 4) | lo_u).to(torch.uint8)
    return packed


def unpack_int4(packed: torch.Tensor, n_values: int, signed: bool = True) -> torch.Tensor:
    """Inverse of :func:`pack_int4`.

    ``n_values`` is the original value count along the last axis. It must
    equal ``2 * packed.shape[-1]`` (we double-check and raise otherwise).
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"unpack_int4 expects uint8 packed input, got {packed.dtype}")
    if n_values != 2 * packed.shape[-1]:
        raise ValueError(
            f"n_values={n_values} does not match 2 * packed.shape[-1]={2 * packed.shape[-1]}"
        )
    packed_i = packed.to(torch.int16)
    lo = packed_i & 0x0F
    hi = (packed_i >> 4) & 0x0F
    if signed:
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
    # Interleave: out[..., 0::2] = lo, out[..., 1::2] = hi
    out = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], n_values)
    return out.to(torch.int8)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_args(bits: int, group_size: int) -> None:
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} out of range for ndim={ndim}")
    return axis if axis >= 0 else axis + ndim
