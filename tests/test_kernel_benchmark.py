"""Tests for the T25 kernel-benchmark script.

Only the pure-function pieces of ``experiments/kernel_benchmark.py`` are
covered here: shape construction, the timing harness, and the JSONL
schema. The script's ``main()`` is exercised end-to-end with a tiny
``N_ctx`` so it stays within CPU budget.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "kernel_benchmark.py"


@pytest.fixture(scope="module")
def kb_mod():
    spec = importlib.util.spec_from_file_location("kernel_benchmark", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["kernel_benchmark"] = module
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _build_shapes — KIVI layout invariants
# ---------------------------------------------------------------------------


def test_build_shapes_returns_kivi_layout(kb_mod):
    q, k, v, k_data, k_scales, v_data, v_scales = kb_mod._build_shapes(
        h=4, h_kv=2, n=64, d=32, group_size=32, seed=0
    )
    assert q.shape == (4, 32)
    assert k.shape == (2, 64, 32) and v.shape == (2, 64, 32)
    assert k_data.shape == (2, 64, 32) and v_data.shape == (2, 64, 32)
    assert k_data.dtype == torch.int8 and v_data.dtype == torch.int8
    # K: per-channel along head_dim → [H_kv, D // G]
    assert k_scales.shape == (2, 1)
    # V: per-token along seq → [H_kv, N // G]
    assert v_scales.shape == (2, 2)
    # All quantized values stay in int8 range.
    assert int(k_data.abs().max()) <= 127
    assert int(v_data.abs().max()) <= 127


def test_build_shapes_is_deterministic(kb_mod):
    a = kb_mod._build_shapes(4, 2, 64, 32, 32, seed=123)
    b = kb_mod._build_shapes(4, 2, 64, 32, 32, seed=123)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


# ---------------------------------------------------------------------------
# _fp16_attention — single-query GQA sanity
# ---------------------------------------------------------------------------


def test_fp16_attention_matches_torch_softmax(kb_mod):
    torch.manual_seed(0)
    h, h_kv, n, d = 4, 2, 32, 16
    q = torch.randn((h, d), dtype=torch.float16)
    k = torch.randn((h_kv, n, d), dtype=torch.float16)
    v = torch.randn((h_kv, n, d), dtype=torch.float16)
    out = kb_mod._fp16_attention(q, k, v, scale=1.0 / (d**0.5))
    assert out.shape == (h, d)
    assert out.dtype == torch.float16
    # Compare one head against an independent FP32 reference.
    scores0 = (q[0].float() @ k[0].float().T) * (1.0 / (d**0.5))
    expect0 = torch.softmax(scores0, dim=-1) @ v[0].float()
    assert torch.allclose(out[0].float(), expect0, atol=5e-3)


# ---------------------------------------------------------------------------
# _time_callable — returns sane (mean, p95)
# ---------------------------------------------------------------------------


def test_time_callable_returns_positive_mean_and_p95(kb_mod):
    counter = {"n": 0}

    def tick():
        counter["n"] += 1

    mean, p95 = kb_mod._time_callable(tick, n_iters=5, warmup=1)
    assert counter["n"] == 6  # 1 warmup + 5 iters
    assert mean >= 0.0
    assert p95 >= 0.0
    assert p95 >= mean - 1e-6  # p95 ≥ mean up to float noise


# ---------------------------------------------------------------------------
# End-to-end smoke — main() writes the expected JSONL schema.
# ---------------------------------------------------------------------------


def test_main_writes_expected_jsonl_schema(kb_mod, tmp_path, monkeypatch):
    out = tmp_path / "kb.jsonl"
    figs = tmp_path / "figs"
    argv = [
        "kernel_benchmark.py",
        "--n-ctx",
        "64",
        "--H",
        "4",
        "--H-kv",
        "2",
        "--D",
        "32",
        "--group-size",
        "32",
        "--n-iters",
        "2",
        "--out",
        str(out),
        "--figures-dir",
        str(figs),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = kb_mod.main()
    assert rc == 0
    assert out.exists()
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    # CPU-only host: fp16_baseline + reference_unfused only.
    assert len(rows) >= 2
    impls = {r["scheme"] for r in rows}
    assert {"fp16_baseline", "reference_unfused"}.issubset(impls)
    required_metric_keys = {
        "impl",
        "N",
        "H",
        "H_kv",
        "D",
        "group_size",
        "wall_ms_mean",
        "wall_ms_p95",
        "n_iters",
        "device",
    }
    for r in rows:
        assert r["experiment"] == "kernel_bench"
        assert required_metric_keys.issubset(r["metrics"].keys())
        assert r["metrics"]["N"] == 64
        assert r["metrics"]["wall_ms_mean"] >= 0.0
    # Figure must land in figures-dir.
    assert (figs / "kernel_benchmark.png").exists()


def test_main_skips_n_not_divisible_by_group_size(kb_mod, tmp_path, monkeypatch, capsys):
    out = tmp_path / "kb.jsonl"
    figs = tmp_path / "figs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kernel_benchmark.py",
            "--n-ctx",
            "65",
            "--H",
            "4",
            "--H-kv",
            "2",
            "--D",
            "32",
            "--group-size",
            "32",
            "--n-iters",
            "2",
            "--out",
            str(out),
            "--figures-dir",
            str(figs),
        ],
    )
    kb_mod.main()
    captured = capsys.readouterr().out
    assert "SKIP N=65" in captured
    # No rows should have been written for the skipped size.
    if out.exists():
        assert out.read_text().strip() == ""
    # Empty row set → no figure should be produced either.
    assert not (figs / "kernel_benchmark.png").exists()
