"""Tests for the T28.3 measured-vs-estimated resolver."""

from __future__ import annotations

import json
from pathlib import Path

from greekkv.memory import (
    DEFAULT_PERPLEXITY_JSONL,
    DEFAULT_PROFILE_JSONL,
    load_measurements,
    pick,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_measurements_joins_perplexity_and_profile(tmp_path):
    ppl = tmp_path / "ppl.jsonl"
    prof = tmp_path / "prof.jsonl"
    _write_jsonl(
        ppl,
        [
            {"model": "Qwen/x", "scheme": "fp16", "metrics": {"n_ctx": 256, "perplexity": 4.0}},
            {
                "model": "Qwen/x",
                "scheme": "int8-fakequant",
                "metrics": {"n_ctx": 256, "perplexity": 4.5},
            },
        ],
    )
    _write_jsonl(
        prof,
        [
            {
                "model": "Qwen/x",
                "scheme": "fp16",
                "metrics": {"n_ctx": 256, "decode_ms_per_token": 7.0, "peak_mem_mb": 375.0},
            },
        ],
    )
    idx = load_measurements(perplexity_jsonl=ppl, profile_jsonl=prof)
    m_fp16 = pick(idx, "Qwen/x", "fp16", 256)
    assert m_fp16 is not None
    assert m_fp16.perplexity == 4.0
    assert m_fp16.decode_ms_per_token == 7.0
    assert m_fp16.peak_mem_mb == 375.0

    m_int8 = pick(idx, "Qwen/x", "int8-fakequant", 256)
    assert m_int8 is not None
    assert m_int8.perplexity == 4.5
    # profile file has no int8 row → decode_ms_per_token stays None
    assert m_int8.decode_ms_per_token is None
    assert m_int8.peak_mem_mb is None


def test_pick_returns_none_for_missing_key(tmp_path):
    ppl = tmp_path / "ppl.jsonl"
    _write_jsonl(
        ppl, [{"model": "Qwen/x", "scheme": "fp16", "metrics": {"n_ctx": 256, "perplexity": 4.0}}]
    )
    idx = load_measurements(perplexity_jsonl=ppl, profile_jsonl=tmp_path / "missing.jsonl")
    assert pick(idx, "Qwen/x", "fp16", 1024) is None  # wrong n_ctx
    assert pick(idx, "Qwen/x", "int8", 256) is None  # wrong scheme
    assert pick(idx, "Other", "fp16", 256) is None  # wrong model


def test_missing_files_are_not_errors(tmp_path):
    idx = load_measurements(
        perplexity_jsonl=tmp_path / "nope1.jsonl",
        profile_jsonl=tmp_path / "nope2.jsonl",
    )
    assert idx == {}


def test_load_measurements_against_committed_artefacts():
    # Must work against the real committed files under results/.
    idx = load_measurements()
    # Known cell from docs/RESULTS.md (T17 + T23 for Qwen2.5-0.5B, fp16, n_ctx=256):
    #   perplexity = 3.89, decode_ms_per_token ≈ 7.3
    m = pick(idx, "Qwen/Qwen2.5-0.5B", "fp16", 256)
    assert m is not None
    assert m.perplexity is not None and 3.8 <= m.perplexity <= 4.0
    assert m.decode_ms_per_token is not None and m.decode_ms_per_token > 0.0


def test_default_paths_exist_or_are_tolerated():
    # The resolver must not crash even if DEFAULT_PERPLEXITY_JSONL is absent;
    # that is a deployment choice, not a bug. Here we just assert the paths
    # are declared relative so future refactors don't silently absolutise.
    assert not DEFAULT_PERPLEXITY_JSONL.is_absolute()
    assert not DEFAULT_PROFILE_JSONL.is_absolute()
