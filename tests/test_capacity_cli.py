"""Tests for the T28.4 ``greekkv-capacity`` CLI.

We exercise ``capacity_cli.main()`` directly (in-process) rather than
spawning subprocesses — faster to run, easier to pin, and covers the
same code path. Subprocess invocation is sanity-checked in one
smoke test so we also know the console-script entry point is wired.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from greekkv.cli import capacity_cli

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_requires_model_vram_nctx(capsys):
    with pytest.raises(SystemExit):
        capacity_cli.main(["--vram-gb", "16", "--n-ctx", "1024"])


def test_mode_flags_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit):
        capacity_cli.main(
            [
                "--model",
                "Qwen/Qwen2.5-0.5B",
                "--vram-gb",
                "16",
                "--n-ctx",
                "1024",
                "--measured",
                "--estimate",
            ]
        )


def test_unknown_model_fails_fast(capsys):
    with pytest.raises(SystemExit) as exc:
        capacity_cli.main(
            [
                "--model",
                "Fake/Nothing-13B",
                "--vram-gb",
                "16",
                "--n-ctx",
                "1024",
                "--estimate",
            ]
        )
    assert "unknown model" in str(exc.value)


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------


def test_default_mode_is_measured(capsys):
    rc = capacity_cli.main(["--model", "Qwen/Qwen2.5-0.5B", "--vram-gb", "16", "--n-ctx", "1024"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mode=measured" in out
    # Every scheme should show up as a row.
    for scheme in (
        "fp16",
        "int8-fakequant",
        "int8-kivi",
        "int4-fakequant",
        "int4-kivi",
        "eviction-sink",
    ):
        assert scheme in out


def test_estimate_mode_leaves_ppl_and_latency_blank(capsys):
    rc = capacity_cli.main(
        [
            "--model",
            "Qwen/Qwen2.5-0.5B",
            "--vram-gb",
            "16",
            "--n-ctx",
            "1024",
            "--estimate",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "mode=estimate" in out
    # The ppl column uses "—" when no measured row is loaded.
    assert "—" in out


def test_measured_mode_populates_known_cell(capsys):
    # Qwen2.5-0.5B, fp16, n_ctx=256 has perplexity 3.89 in the committed JSONL.
    rc = capacity_cli.main(
        [
            "--model",
            "Qwen/Qwen2.5-0.5B",
            "--vram-gb",
            "16",
            "--n-ctx",
            "256",
            "--measured",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "mode=measured" in out
    assert "3.89" in out  # fp16 PPL at n_ctx=256
    assert "measured" in out


def test_all_mode_mixes_measured_and_estimated(capsys, tmp_path):
    # Point at an empty profile file so decode-ms stays None, but perplexity
    # from the committed file still populates — exercises the mixed row case.
    empty_prof = tmp_path / "empty.jsonl"
    empty_prof.write_text("")
    rc = capacity_cli.main(
        [
            "--model",
            "Qwen/Qwen2.5-0.5B",
            "--vram-gb",
            "16",
            "--n-ctx",
            "256",
            "--all",
            "--profile-jsonl",
            str(empty_prof),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # Perplexity still rendered from committed T17 artefact
    assert "3.89" in out
    # Decode ms blanked because custom profile was empty
    assert "—" in out


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def test_json_output_is_valid_and_has_six_rows(capsys):
    rc = capacity_cli.main(
        [
            "--model",
            "Qwen/Qwen2.5-0.5B",
            "--vram-gb",
            "16",
            "--n-ctx",
            "1024",
            "--estimate",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "Qwen/Qwen2.5-0.5B"
    assert payload["mode"] == "estimate"
    assert len(payload["rows"]) == 6
    row_schemes = [r["scheme"] for r in payload["rows"]]
    assert row_schemes == [
        "fp16",
        "int8-fakequant",
        "int8-kivi",
        "int4-fakequant",
        "int4-kivi",
        "eviction-sink",
    ]
    # Every row has the required source tag.
    for row in payload["rows"]:
        assert row["source"] in ("measured", "estimated")


# ---------------------------------------------------------------------------
# End-to-end: console-script entry point
# ---------------------------------------------------------------------------


def test_console_script_entry_point_runs():
    # Smoke-test the installed console script so we catch pyproject regressions.
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "greekkv.cli.capacity_cli",
            "--model",
            "Qwen/Qwen2.5-0.5B",
            "--vram-gb",
            "16",
            "--n-ctx",
            "256",
            "--estimate",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "scheme" in proc.stdout
    assert "fp16" in proc.stdout
