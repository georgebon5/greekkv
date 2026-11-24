"""Tests for the T22 report builder.

Only the pure-function pieces of ``scripts/build_report.py`` are covered
here — the section builders. Each gets a tiny fixture and we assert the
table shape and a handful of specific values. The script itself is
exercised end-to-end by ``test_build_report_runs_against_committed_rows``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_REPORT_PATH = REPO_ROOT / "scripts" / "build_report.py"


@pytest.fixture(scope="module")
def report_mod():
    spec = importlib.util.spec_from_file_location("build_report", BUILD_REPORT_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["build_report"] = module
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# ensure_files_exist — loud failure on missing inputs
# ---------------------------------------------------------------------------


def test_ensure_files_exist_passes_when_committed_data_is_present(report_mod):
    # The committed results files from T06–T21 must all exist on main.
    report_mod.ensure_files_exist()


def test_ensure_files_exist_raises_when_any_file_is_missing(
    report_mod, tmp_path: Path, monkeypatch
):
    bad_files = {
        "fertility": tmp_path / "nope.csv",
        "kv_cost": tmp_path / "nope.csv",
        "perplexity": tmp_path / "nope.jsonl",
        "benchmarks": tmp_path / "nope.jsonl",
        "needle": tmp_path / "nope.jsonl",
        "kernel_bench": tmp_path / "nope.jsonl",
    }
    monkeypatch.setattr(report_mod, "RESULT_FILES", bad_files)
    with pytest.raises(SystemExit) as exc:
        report_mod.ensure_files_exist()
    assert "Missing result files" in str(exc.value)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def test_fertility_section_includes_ratio_column(report_mod):
    rows = [
        {"tokenizer": "tok-A", "language": "en", "mean_tpw": "1.30"},
        {"tokenizer": "tok-A", "language": "el", "mean_tpw": "1.56"},
        {"tokenizer": "tok-B", "language": "en", "mean_tpw": "1.20"},
        {"tokenizer": "tok-B", "language": "el", "mean_tpw": "4.80"},
    ]
    md = report_mod.fertility_section(rows)
    # Both tokenizers should appear.
    assert "`tok-A`" in md and "`tok-B`" in md
    # Ratio header must be present.
    assert "EL / EN" in md
    # tok-B's ratio = 4.0×.
    assert "**4.00×**" in md


def test_kv_cost_section_shapes_three_columns_per_lang(report_mod):
    rows = [
        {
            "model": "X/Y",
            "label": "Krikri-8B",
            "language": "en",
            "tpw": "1",
            "n_tokens": "1269",
            "kv_mb": "158.6",
        },
        {
            "model": "X/Y",
            "label": "Krikri-8B",
            "language": "el",
            "tpw": "1",
            "n_tokens": "1529",
            "kv_mb": "191.1",
        },
    ]
    md = report_mod.kv_cost_section(rows)
    assert "Krikri-8B" in md
    assert "158.6" in md and "191.1" in md
    # Ratio ≈ 191.1 / 158.6 ≈ 1.20.
    assert "**1.20×**" in md


def _ppl_row(scheme: str, n_ctx: int, ppl: float, top1: float | None = None):
    m = {
        "n_ctx": n_ctx,
        "perplexity": ppl,
        "nll_mean": 0.0,
    }
    if top1 is not None:
        m.update(top1_agree=top1, top5_agree=1.0, kl_mean=0.0, kl_p95=0.0)
    return {"scheme": scheme, "metrics": m}


def test_perplexity_section_pivot_and_fidelity_block(report_mod):
    rows = [
        _ppl_row("fp16", 256, 3.9, top1=1.0),
        _ppl_row("fp16", 512, 4.0, top1=1.0),
        _ppl_row("int4-kivi", 256, 29.6, top1=0.19),
        _ppl_row("int4-kivi", 512, 34.5, top1=0.18),
    ]
    md = report_mod.perplexity_section(rows)
    # Scheme × n_ctx table + fidelity block.
    assert "PPL n_ctx=256" in md and "PPL n_ctx=512" in md
    assert "Fidelity vs FP16" in md
    assert "**fp16**" in md and "int4-kivi" in md


def _bench_row(bench: str, scheme: str, correct: int, n: int):
    return {
        "scheme": scheme,
        "metrics": {
            "benchmark": bench,
            "correct": correct,
            "n_samples": n,
            "accuracy": correct / n,
        },
    }


def test_benchmarks_section_includes_ci(report_mod):
    rows = [
        _bench_row("arc_el", "fp16", 12, 50),
        _bench_row("arc_el", "int4-kivi", 13, 50),
        _bench_row("hellaswag_el", "fp16", 12, 50),
        _bench_row("hellaswag_el", "int4-kivi", 18, 50),
    ]
    md = report_mod.benchmarks_section(rows)
    assert "arc_el" in md and "hellaswag_el" in md
    # Should include fraction + rounded accuracy + CI block.
    assert "12/50" in md
    assert "0.24" in md


def _needle_row(scheme: str, correct: int):
    return {"scheme": scheme, "metrics": {"correct": correct}}


def test_needle_section_overall_accuracy_and_ci(report_mod):
    rows = [
        _needle_row("fp16", 1),
        _needle_row("fp16", 1),
        _needle_row("fp16", 1),
        _needle_row("int4-kivi", 1),
        _needle_row("int4-kivi", 0),
        _needle_row("int4-kivi", 1),
    ]
    md = report_mod.needle_section(rows)
    assert "3/3" in md  # fp16 perfect
    assert "2/3" in md  # int4-kivi 2 out of 3
    # CI column present on both rows.
    assert md.count("(") >= 2


def _kb_row(impl: str, n: int, mean_ms: float, p95_ms: float, device: str = "cpu") -> dict:
    return {
        "scheme": impl,
        "metrics": {
            "impl": impl,
            "N": n,
            "wall_ms_mean": mean_ms,
            "wall_ms_p95": p95_ms,
            "device": device,
        },
    }


def test_kernel_bench_section_pivot_and_cpu_note(report_mod):
    rows = [
        _kb_row("fp16_baseline", 256, 0.19, 0.21),
        _kb_row("reference_unfused", 256, 0.13, 0.14),
        _kb_row("fp16_baseline", 1024, 0.25, 0.26),
        _kb_row("reference_unfused", 1024, 0.25, 0.26),
    ]
    md = report_mod.kernel_bench_section(rows)
    assert "## M4 — Decode-attention kernel benchmark (T25)" in md
    assert "N = 256" in md and "N = 1024" in md
    assert "`fp16_baseline`" in md and "`reference_unfused`" in md
    # 0.13 ms cell must be rendered.
    assert "0.13" in md
    # CPU-only fixture → Triton note should appear.
    assert "triton_fused" in md
    assert "skipped" in md.lower()


def test_kernel_bench_section_no_cpu_note_when_cuda_rows_present(report_mod):
    rows = [
        _kb_row("fp16_baseline", 256, 0.19, 0.21),
        _kb_row("triton_fused", 256, 0.02, 0.03, device="cuda"),
    ]
    md = report_mod.kernel_bench_section(rows)
    assert "`triton_fused`" in md
    assert "skipped" not in md.lower()


# ---------------------------------------------------------------------------
# End-to-end: build the report from committed results
# ---------------------------------------------------------------------------


def test_build_report_runs_against_committed_rows(report_mod):
    text = report_mod.build_report()
    # Headline sections must all be there.
    for marker in (
        "## M1 — Tokenizer fertility (T06)",
        "## M1 — KV-cache cost per document (T08)",
        "## M3 — Perplexity sweep (T17 + T20 fidelity)",
        "## M3 — Greek multi-choice benchmark accuracy (T18)",
        "## M3 — Needle-in-a-haystack retrieval (T19)",
        "## M4 — Decode-attention kernel benchmark (T25)",
    ):
        assert marker in text, f"missing section: {marker}"
    # The report should be non-trivial — at least a few KB of content.
    assert len(text) > 2000
