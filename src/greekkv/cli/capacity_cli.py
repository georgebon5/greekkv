"""``greekkv-capacity`` command-line entry point (T28.4).

Prints a capacity table for a given (model, VRAM, n_ctx) combination
across all six schemes. Every row is tagged **measured** or
**estimated** depending on whether the T17 / T23 JSONL files have a
matching cell. This is the whole product value of the CLI: the
hardware number comes from the formula either way, but the quality /
latency annotation is only trustworthy when a measurement exists.

Usage
-----
::

    greekkv-capacity \\
        --model Qwen/Qwen2.5-0.5B \\
        --vram-gb 16 --n-ctx 1024 \\
        --measured

    greekkv-capacity --model Qwen/Qwen2.5-0.5B --vram-gb 16 --n-ctx 4096 --estimate

    greekkv-capacity ... --all   # does the full scheme grid, defaults to measured
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from greekkv.memory import (
    ALL_SCHEMES,
    DEFAULT_OVERHEAD_BYTES,
    ArchConfig,
    CapacityInputs,
    CapacityResult,
    capacity,
    load_measurements,
    pick,
)

# Minimal hard-coded registry of known models so the CLI stays usable
# offline (no HF download). Each entry mirrors ArchConfig; keeping the
# number small is deliberate — the project's measured grid only covers
# Qwen2.5-0.5B today.
_KNOWN_MODELS: dict[str, ArchConfig] = {
    "Qwen/Qwen2.5-0.5B": ArchConfig(
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_size=896,
        head_dim=64,
        num_parameters=494_000_000,
        vocab_size=151_936,
        intermediate_size=4864,
    ),
    "Qwen/Qwen2.5-7B": ArchConfig(
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_size=3584,
        head_dim=128,
        num_parameters=7_615_616_512,
        vocab_size=152_064,
        intermediate_size=18944,
    ),
    "ilsp/Llama-Krikri-8B-Instruct": ArchConfig(
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        head_dim=128,
        num_parameters=8_030_000_000,
        vocab_size=128_256,
        intermediate_size=14336,
    ),
    "ilsp/Meltemi-7B-Instruct-v1.5": ArchConfig(
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        head_dim=128,
        num_parameters=7_240_000_000,
        vocab_size=32000,
        intermediate_size=14336,
    ),
}


SourceMode = Literal["measured", "estimate", "all"]


def _format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:,.1f} {units[i]}"


def _format_row(r: CapacityResult) -> list[str]:
    ppl = "—" if r.measured_ppl is None else f"{r.measured_ppl:.2f}"
    ms = "—" if r.measured_decode_ms is None else f"{r.measured_decode_ms:.2f}"
    return [
        r.scheme,
        f"{r.max_concurrent_users:,}",
        _format_bytes(r.kv_bytes_per_user),
        _format_bytes(r.weight_bytes),
        ppl,
        ms,
        r.source,
    ]


def _render_table(rows: list[list[str]], headers: list[str]) -> str:
    cols = list(zip(headers, *rows, strict=False))
    widths = [max(len(c) for c in col) for col in cols]

    def line(cells: list[str]) -> str:
        return " | ".join(f"{c:<{w}}" for c, w in zip(cells, widths, strict=False))

    sep = "-+-".join("-" * w for w in widths)
    out = [line(headers), sep, *(line(r) for r in rows)]
    return "\n".join(out)


def _resolve_source(args: argparse.Namespace) -> SourceMode:
    if args.all:
        return "all"
    if args.measured:
        return "measured"
    if args.estimate:
        return "estimate"
    return "measured"  # default is the more informative option


def _resolve_cfg(model: str) -> ArchConfig:
    if model in _KNOWN_MODELS:
        return _KNOWN_MODELS[model]
    raise SystemExit(
        f"unknown model {model!r}. Known: {sorted(_KNOWN_MODELS)}. "
        "To add a new model, extend _KNOWN_MODELS in capacity_cli.py "
        "(or supply the ArchConfig fields on the capacity() call directly)."
    )


def _run_grid(
    cfg: ArchConfig,
    model: str,
    vram_bytes: int,
    n_ctx: int,
    overhead_bytes: int,
    mode: SourceMode,
    measured_index: dict,
) -> list[CapacityResult]:
    results: list[CapacityResult] = []
    for scheme in ALL_SCHEMES:
        measurement = pick(measured_index, model, scheme, n_ctx)
        want_measured = mode in ("measured", "all") and measurement is not None
        inp = CapacityInputs(
            cfg=cfg,
            vram_bytes=vram_bytes,
            n_ctx=n_ctx,
            scheme=scheme,
            overhead_bytes=overhead_bytes,
            source="measured" if want_measured else "estimated",
            measured_ppl=measurement.perplexity if want_measured else None,
            measured_decode_ms=(
                measurement.decode_ms_per_token if want_measured else None
            ),
        )
        results.append(capacity(inp))
    return results


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="greekkv-capacity",
        description=(
            "Compute max concurrent users on a VRAM budget, per KV cache "
            "scheme. Pass --measured to only trust rows that appear in the "
            "committed T17/T23 JSONL; --estimate to skip the lookup entirely."
        ),
    )
    ap.add_argument("--model", required=True, help="Hugging Face model id (see _KNOWN_MODELS for coverage)")
    ap.add_argument("--vram-gb", type=float, required=True, help="GPU VRAM budget in GB (power of 2 base, i.e. GiB)")
    ap.add_argument("--n-ctx", type=int, required=True, help="Sequence length per user")
    ap.add_argument(
        "--overhead-mb",
        type=float,
        default=DEFAULT_OVERHEAD_BYTES / (1024 * 1024),
        help="Runtime overhead (framework + activations) in MB. Default: T23 median.",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--measured", action="store_true", help="Only annotate rows that have a measured JSONL cell")
    g.add_argument("--estimate", action="store_true", help="Skip the measured JSONL lookup entirely")
    g.add_argument("--all", action="store_true", help="Both: use measurements where available, estimate otherwise")

    ap.add_argument(
        "--perplexity-jsonl",
        type=Path,
        default=None,
        help="Override path for T17 perplexity JSONL",
    )
    ap.add_argument(
        "--profile-jsonl",
        type=Path,
        default=None,
        help="Override path for T23 profile JSONL",
    )
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _resolve_cfg(args.model)

    mode = _resolve_source(args)
    index = {}
    if mode in ("measured", "all"):
        index = load_measurements(
            perplexity_jsonl=args.perplexity_jsonl,
            profile_jsonl=args.profile_jsonl,
        )

    vram_bytes = int(round(args.vram_gb * (1024**3)))
    overhead_bytes = int(round(args.overhead_mb * 1024 * 1024))
    results = _run_grid(cfg, args.model, vram_bytes, args.n_ctx, overhead_bytes, mode, index)

    if args.json:
        payload = {
            "model": args.model,
            "vram_gb": args.vram_gb,
            "n_ctx": args.n_ctx,
            "overhead_mb": args.overhead_mb,
            "mode": mode,
            "rows": [asdict(r) for r in results],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    headers = ["scheme", "users", "kv/user", "weights", "ppl", "ms/tok", "source"]
    rows = [_format_row(r) for r in results]
    header_line = (
        f"model={args.model}  vram={args.vram_gb} GiB  n_ctx={args.n_ctx}  "
        f"overhead={args.overhead_mb:.0f} MiB  mode={mode}"
    )
    print(header_line)
    print()
    print(_render_table(rows, headers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
