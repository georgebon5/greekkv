"""Tests for greekkv.eval.fidelity.compute_fidelity."""

from __future__ import annotations

import pytest
import torch

from greekkv.eval import FidelityResult, compute_fidelity


def _rand_logits(n: int, v: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, v, generator=g, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Identity: logits vs themselves ⇒ zero KL, full top-1/top-5 agreement
# ---------------------------------------------------------------------------


def test_fidelity_is_identity_for_matching_logits():
    x = _rand_logits(32, 128, seed=0)
    res = compute_fidelity(x, x)
    assert isinstance(res, FidelityResult)
    assert res.kl_mean == pytest.approx(0.0, abs=1e-6)
    assert res.kl_p95 == pytest.approx(0.0, abs=1e-6)
    assert res.top1_agree == 1.0
    assert res.top5_agree == 1.0
    assert res.n_positions == 32


# ---------------------------------------------------------------------------
# Monotonicity: larger perturbations ⇒ larger KL, lower agreement
# ---------------------------------------------------------------------------


def test_small_perturbation_has_small_fidelity_cost():
    ref = _rand_logits(64, 256, seed=1)
    tiny_noise = _rand_logits(64, 256, seed=2) * 0.01
    big_noise = _rand_logits(64, 256, seed=2) * 1.0
    small = compute_fidelity(ref, ref + tiny_noise)
    large = compute_fidelity(ref, ref + big_noise)
    assert small.kl_mean < large.kl_mean
    assert small.top1_agree > large.top1_agree
    assert small.top5_agree >= large.top5_agree


# ---------------------------------------------------------------------------
# KL direction: KL(ref || q) ≠ KL(q || ref) in general
# ---------------------------------------------------------------------------


def test_kl_is_asymmetric():
    # KL(P || Q) ≠ KL(Q || P) in general. We use a sharply-peaked reference
    # against a near-uniform q — the two KLs differ by orders of magnitude.
    ref = torch.zeros(50, 64)
    ref[:, 0] = 10.0  # sharp peak at token 0
    q = torch.zeros(50, 64)  # uniform logits → uniform probs
    a = compute_fidelity(ref, q).kl_mean  # KL(sharp || uniform) — large
    b = compute_fidelity(q, ref).kl_mean  # KL(uniform || sharp) — different
    assert abs(a - b) > 0.5


# ---------------------------------------------------------------------------
# Top-1 vs Top-5 relation
# ---------------------------------------------------------------------------


def test_top5_agreement_is_at_least_top1():
    ref = _rand_logits(40, 128, seed=5)
    q = ref + _rand_logits(40, 128, seed=6) * 0.3
    res = compute_fidelity(ref, q)
    assert res.top5_agree >= res.top1_agree


# ---------------------------------------------------------------------------
# KL percentile ≥ mean (by distributional construction)
# ---------------------------------------------------------------------------


def test_kl_p95_is_at_least_kl_mean():
    ref = _rand_logits(200, 64, seed=7)
    q = ref + _rand_logits(200, 64, seed=8)
    res = compute_fidelity(ref, q)
    assert res.kl_p95 >= res.kl_mean


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_fidelity_rejects_shape_mismatch():
    a = _rand_logits(10, 64, seed=0)
    b = _rand_logits(10, 128, seed=0)
    with pytest.raises(ValueError, match="shape"):
        compute_fidelity(a, b)


def test_fidelity_rejects_non_2d():
    a = torch.randn(10)
    b = torch.randn(10)
    with pytest.raises(ValueError, match="2D"):
        compute_fidelity(a, b)


def test_fidelity_rejects_empty_positions():
    a = torch.empty(0, 64)
    b = torch.empty(0, 64)
    with pytest.raises(ValueError, match="zero positions"):
        compute_fidelity(a, b)


def test_fidelity_handles_fp16_input():
    ref = _rand_logits(20, 64, seed=10).to(torch.float16)
    q = ref.clone()
    res = compute_fidelity(ref, q)
    assert res.kl_mean == pytest.approx(0.0, abs=1e-3)
    assert res.top1_agree == 1.0
