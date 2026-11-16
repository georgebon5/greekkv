"""Tests for the T21 sweep runner's resume logic.

The sweep script itself is a thin orchestrator; the pure functions worth
unit-testing are:

- :func:`scripts.run_sweep.load_existing_hashes` — reads the JSONL and
  returns the set of ``config_hash`` keys it contains.
- :func:`scripts.run_sweep.plan_cells` — marks each ``(scheme, n_ctx)``
  cell as TODO or SKIP based on that set.

We import the script via :mod:`importlib.util` because it lives under
``scripts/`` rather than inside the installed package.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from greekkv.eval import spec_hash
from greekkv.quant import CacheSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SWEEP_PATH = REPO_ROOT / "scripts" / "run_sweep.py"


@pytest.fixture(scope="module")
def run_sweep_module():
    spec = importlib.util.spec_from_file_location("run_sweep", RUN_SWEEP_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["run_sweep"] = module
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# load_existing_hashes
# ---------------------------------------------------------------------------


def test_load_existing_hashes_empty_file(run_sweep_module, tmp_path: Path):
    assert run_sweep_module.load_existing_hashes(tmp_path / "missing.jsonl") == set()


def test_load_existing_hashes_collects_all_rows(run_sweep_module, tmp_path: Path):
    p = tmp_path / "rows.jsonl"
    rows = [
        {"config_hash": "aaa111111111", "model": "m", "scheme": "fp16"},
        {"config_hash": "bbb222222222", "model": "m", "scheme": "int4-kivi"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    hashes = run_sweep_module.load_existing_hashes(p)
    assert hashes == {"aaa111111111", "bbb222222222"}


def test_load_existing_hashes_skips_malformed_lines(run_sweep_module, tmp_path: Path, capsys):
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        '{"config_hash": "good00000001"}\nnot-valid-json\n{"config_hash": "good00000002"}\n',
        encoding="utf-8",
    )
    hashes = run_sweep_module.load_existing_hashes(p)
    assert hashes == {"good00000001", "good00000002"}
    assert "malformed" in capsys.readouterr().err


def test_load_existing_hashes_ignores_rows_without_hash(run_sweep_module, tmp_path: Path):
    p = tmp_path / "partial.jsonl"
    p.write_text(
        '{"scheme": "fp16"}\n{"config_hash": "ok1234567890"}\n',
        encoding="utf-8",
    )
    assert run_sweep_module.load_existing_hashes(p) == {"ok1234567890"}


# ---------------------------------------------------------------------------
# plan_cells — SKIP/TODO decision mirrors spec_hash
# ---------------------------------------------------------------------------


def test_plan_cells_all_todo_when_empty(run_sweep_module):
    schemes = [CacheSpec(scheme="fp16"), CacheSpec(scheme="int4-kivi", bits=4, residual=8)]
    cells = run_sweep_module.plan_cells(schemes, [256, 512], done=set())
    assert len(cells) == 4
    assert all(not done for _, _, done in cells)


def test_plan_cells_marks_matching_hash_as_done(run_sweep_module):
    s_fp16 = CacheSpec(scheme="fp16")
    s_kivi = CacheSpec(scheme="int4-kivi", bits=4, residual=8)
    done = {spec_hash(s_fp16, 256), spec_hash(s_kivi, 512)}
    cells = run_sweep_module.plan_cells([s_fp16, s_kivi], [256, 512], done=done)
    done_flags = {(spec.scheme, n_ctx, flag) for spec, n_ctx, flag in cells}
    assert ("fp16", 256, True) in done_flags
    assert ("fp16", 512, False) in done_flags
    assert ("int4-kivi", 256, False) in done_flags
    assert ("int4-kivi", 512, True) in done_flags


def test_plan_cells_order_is_stable(run_sweep_module):
    schemes = [CacheSpec(scheme="fp16"), CacheSpec(scheme="int4-kivi", bits=4, residual=8)]
    first = run_sweep_module.plan_cells(schemes, [256, 512], done=set())
    again = run_sweep_module.plan_cells(schemes, [256, 512], done=set())
    assert [(c[0].scheme, c[1]) for c in first] == [(c[0].scheme, c[1]) for c in again]
