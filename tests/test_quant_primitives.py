"""Property-style tests for greekkv.quant.primitives.

Invariants we rely on in every downstream task:

- Round-trip preserves shape and dtype.
- Round-trip relative error has a tight ceiling: tighter for higher bits.
- More bits → strictly less error (on the same input).
- Symmetric keeps zero-point implicit; asymmetric reaches both sign bounds.
- pack / unpack is bit-exact in both signed and unsigned modes.
- Padding is transparent: non-divisible group axis is still faithfully
  round-tripped.
"""

from __future__ import annotations

import pytest
import torch

from greekkv.quant import (
    QuantizedTensor,
    dequantize,
    pack_int4,
    quantize,
    unpack_int4,
)

# Loose per-bits relative-error ceilings. Sized so a normal(0, 1) tensor of a
# few thousand elements comfortably clears them; tightened empirically.
REL_ERR_BUDGET = {
    8: 0.01,
    4: 0.15,
}


def _rel_err(y: torch.Tensor, x: torch.Tensor) -> float:
    return float((y - x).norm() / x.norm())


def _rand(
    shape: tuple[int, ...], seed: int = 0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype)


# ---------------------------------------------------------------------------
# pack_int4 / unpack_int4
# ---------------------------------------------------------------------------


def test_pack_unpack_signed_identity():
    x = torch.arange(-8, 8, dtype=torch.int8)  # covers full [-8, 7] range
    packed = pack_int4(x, signed=True)
    assert packed.dtype == torch.uint8
    assert packed.shape[-1] == x.shape[-1] // 2
    out = unpack_int4(packed, n_values=x.shape[-1], signed=True)
    assert torch.equal(out, x)


def test_pack_unpack_unsigned_identity():
    x = torch.arange(0, 16, dtype=torch.int8)
    packed = pack_int4(x, signed=False)
    out = unpack_int4(packed, n_values=x.shape[-1], signed=False)
    assert torch.equal(out, x)


def test_pack_unpack_identity_higher_rank():
    x = torch.randint(-8, 8, (3, 4, 10), dtype=torch.int8)
    packed = pack_int4(x, signed=True)
    assert packed.shape == (3, 4, 5)
    out = unpack_int4(packed, n_values=10, signed=True)
    assert torch.equal(out, x)


def test_pack_rejects_odd_last_dim():
    with pytest.raises(ValueError, match="even"):
        pack_int4(torch.zeros(5, dtype=torch.int8), signed=True)


def test_pack_rejects_out_of_range_signed():
    with pytest.raises(ValueError, match=r"\[-8, 7\]"):
        pack_int4(torch.tensor([0, 8], dtype=torch.int8), signed=True)


def test_pack_rejects_out_of_range_unsigned():
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        pack_int4(torch.tensor([0, 16], dtype=torch.int8), signed=False)


def test_unpack_wrong_n_values_raises():
    packed = pack_int4(torch.zeros(4, dtype=torch.int8), signed=True)
    with pytest.raises(ValueError, match="n_values"):
        unpack_int4(packed, n_values=5, signed=True)


# ---------------------------------------------------------------------------
# quantize / dequantize — invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("symmetric", [True, False])
def test_roundtrip_preserves_shape_and_dtype(bits, group_size, symmetric):
    x = _rand((2, 3, 256), seed=bits * 31 + group_size)
    qt = quantize(x, bits=bits, group_size=group_size, symmetric=symmetric)
    y = dequantize(qt)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert isinstance(qt, QuantizedTensor)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("symmetric", [True, False])
def test_roundtrip_error_under_budget(bits, symmetric):
    x = _rand((4, 512), seed=42)
    qt = quantize(x, bits=bits, group_size=128, symmetric=symmetric)
    err = _rel_err(dequantize(qt), x)
    assert err < REL_ERR_BUDGET[bits], f"err={err:.4f} over budget {REL_ERR_BUDGET[bits]}"


def test_more_bits_strictly_fewer_errors():
    x = _rand((4, 512), seed=7)
    err8 = _rel_err(dequantize(quantize(x, bits=8, group_size=128)), x)
    err4 = _rel_err(dequantize(quantize(x, bits=4, group_size=128)), x)
    assert err8 < err4, f"int8 err {err8} should be < int4 err {err4}"


def test_smaller_group_size_fewer_errors():
    # Smaller groups track outliers more finely → lower reconstruction error.
    x = _rand((4, 512), seed=11)
    err_32 = _rel_err(dequantize(quantize(x, bits=4, group_size=32)), x)
    err_128 = _rel_err(dequantize(quantize(x, bits=4, group_size=128)), x)
    assert err_32 < err_128


def test_asymmetric_wins_on_skewed_input():
    # Heavily non-centered data benefits from zero-point.
    g = torch.Generator().manual_seed(3)
    x = torch.randn(4, 512, generator=g) * 2.0 + 5.0  # mean ~5
    sym_err = _rel_err(dequantize(quantize(x, bits=4, group_size=128, symmetric=True)), x)
    asym_err = _rel_err(dequantize(quantize(x, bits=4, group_size=128, symmetric=False)), x)
    assert asym_err < sym_err


def test_padded_tail_roundtrips():
    # last dim 100 is NOT divisible by group_size 32 → padded + sliced
    x = _rand((3, 100), seed=5)
    qt = quantize(x, bits=8, group_size=32, symmetric=True)
    y = dequantize(qt)
    assert y.shape == x.shape
    assert _rel_err(y, x) < REL_ERR_BUDGET[8]


def test_axis_argument_quantizes_correct_dim():
    x = _rand((4, 256), seed=9)
    # Quantizing along axis=0 (len 4) with group_size=4 means a single scale per column.
    qt0 = quantize(x, bits=8, group_size=4, axis=0, symmetric=True)
    # scales shape mirrors the grouped dim: (4 // 4) along axis 0 → size 1, plus 256 along axis 1.
    assert qt0.scales.shape == (1, 256)
    y = dequantize(qt0)
    assert y.shape == x.shape
    assert _rel_err(y, x) < REL_ERR_BUDGET[8]


def test_zero_input_roundtrips_to_zero():
    x = torch.zeros(2, 128)
    qt = quantize(x, bits=4, group_size=32, symmetric=True)
    y = dequantize(qt)
    assert torch.all(y == 0)


def test_deterministic():
    x = _rand((4, 128), seed=1)
    qt1 = quantize(x, bits=4, group_size=32, symmetric=False)
    qt2 = quantize(x, bits=4, group_size=32, symmetric=False)
    assert torch.equal(qt1.data, qt2.data)
    assert torch.equal(qt1.scales, qt2.scales)
    assert qt1.zero_points is not None and qt2.zero_points is not None
    assert torch.equal(qt1.zero_points, qt2.zero_points)


def test_symmetric_has_no_zero_point():
    qt = quantize(_rand((2, 64)), bits=8, group_size=32, symmetric=True)
    assert qt.zero_points is None


def test_asymmetric_zero_point_shape_matches_scales():
    qt = quantize(_rand((2, 64)), bits=8, group_size=32, symmetric=False)
    assert qt.zero_points is not None
    assert qt.zero_points.shape == qt.scales.shape


# ---------------------------------------------------------------------------
# quantize — storage format invariants
# ---------------------------------------------------------------------------


def test_int4_storage_is_packed_uint8():
    qt = quantize(_rand((2, 128)), bits=4, group_size=32, symmetric=True)
    assert qt.data.dtype == torch.uint8
    # Last axis halved.
    assert qt.data.shape[-1] == 64


def test_int8_symmetric_storage_is_int8():
    qt = quantize(_rand((2, 128)), bits=8, group_size=32, symmetric=True)
    assert qt.data.dtype == torch.int8
    assert qt.data.shape[-1] == 128


def test_int8_asymmetric_storage_is_uint8():
    qt = quantize(_rand((2, 128)), bits=8, group_size=32, symmetric=False)
    assert qt.data.dtype == torch.uint8


def test_scales_stored_as_fp16():
    qt = quantize(_rand((2, 128)), bits=8, group_size=32)
    assert qt.scales.dtype == torch.float16


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [0, 1, 3, 5, 7, 16])
def test_quantize_rejects_bad_bits(bits):
    with pytest.raises(ValueError, match="bits"):
        quantize(_rand((2, 32)), bits=bits)


@pytest.mark.parametrize("gs", [0, -1])
def test_quantize_rejects_bad_group_size(gs):
    with pytest.raises(ValueError, match="group_size"):
        quantize(_rand((2, 32)), bits=8, group_size=gs)


def test_quantize_rejects_integer_tensor():
    with pytest.raises(TypeError, match="float"):
        quantize(torch.zeros(4, dtype=torch.int32), bits=8, group_size=4)


def test_quantize_rejects_out_of_range_axis():
    with pytest.raises(ValueError, match="axis"):
        quantize(_rand((2, 32)), bits=8, group_size=4, axis=2)
