"""T23 — Sweep the memory + latency profiler across (scheme, n_ctx).

Loads one model, runs :func:`greekkv.systems.profile_scheme` for every
``(scheme, n_ctx)`` combination in the hard-coded list, writes a JSONL
row per combination and three figures (peak memory, prefill latency,
decode throughput) side-by-side.

Usage
-----
::

    python experiments/profile.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --n-ctx 256 512 1024 \\
        --n-decode 16 \\
        --out results/profile/qwen2_5_0_5b.jsonl \\
        --figures-dir results/figures/profile/
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from greekkv.eval import LoadSpec, load_model
from greekkv.quant import CacheSpec
from greekkv.systems import profile_scheme
from greekkv.utils import ResultRecord, append_result, git_sha, seed_all

DEFAULT_SCHEMES: list[CacheSpec] = [
    CacheSpec(scheme="fp16"),
    CacheSpec(scheme="int8-fakequant", bits=8, group_size=64),
    CacheSpec(scheme="int4-fakequant", bits=4, group_size=64),
    CacheSpec(scheme="int8-kivi", bits=8, group_size=64, residual=16),
    CacheSpec(scheme="int4-kivi", bits=4, group_size=64, residual=16),
    CacheSpec(scheme="eviction-sink", n_sink=4, n_recent=32),
]

SCHEME_COLOR = {
    "fp16": "#2E7D32",
    "int8-fakequant": "#1976D2",
    "int4-fakequant": "#7B1FA2",
    "int8-kivi": "#F57C00",
    "int4-kivi": "#D32F2F",
    "eviction-sink": "#5D4037",
}


def _spec_hash(spec: CacheSpec, n_ctx: int, n_decode: int) -> str:
    key = (
        f"profile|{spec.scheme}|bits={spec.bits}|g={spec.group_size}|r={spec.residual}|"
        f"sym={int(spec.symmetric)}|sink={spec.n_sink}|recent={spec.n_recent}|"
        f"ctx={n_ctx}|decode={n_decode}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _plot_triptych(rows: list[dict], out_dir: Path, model_name: str) -> None:
    # Group: scheme → list of (n_ctx, peak_mb, prefill_s, decode_tok_s)
    by_scheme: dict[str, list[tuple[int, float, float, float, float]]] = {}
    for r in rows:
        m = r["metrics"]
        by_scheme.setdefault(r["scheme"], []).append(
            (
                int(m["n_ctx"]),
                float(m["peak_mem_delta_mb"]),
                float(m["prefill_wall_s"]),
                float(m["decode_tokens_per_s"]),
                float(m["decode_ms_per_token"]),
            )
        )
    for s in by_scheme:
        by_scheme[s].sort()

    schemes = [s for s in SCHEME_COLOR if s in by_scheme]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = [
        "Peak memory Δ vs baseline (MB)",
        "Prefill wall time (s)",
        "Decode latency (ms / token)",
    ]
    columns = [1, 2, 4]  # peak_mb, prefill_s, decode_ms_per_token
    for ax, title, col in zip(axes, titles, columns, strict=True):
        for s in schemes:
            data = by_scheme[s]
            xs = [row[0] for row in data]
            ys = [row[col] for row in data]
            ax.plot(xs, ys, marker="o", linewidth=2, label=s, color=SCHEME_COLOR[s])
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({row[0] for s in schemes for row in by_scheme[s]}))
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.set_xlabel("n_ctx")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")

    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(f"Systems profile — {model_name}", fontsize=12)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "profile_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-ctx", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--n-decode", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/profile/qwen2_5_0_5b.jsonl"))
    ap.add_argument("--figures-dir", type=Path, default=Path("results/figures/profile"))
    args = ap.parse_args()

    seed_all(args.seed)

    print(f"Loading {args.model}...")
    model, tok, mem = load_model(LoadSpec(name=args.model, dtype="fp32", device="auto"))
    model.eval()
    print(f"  loaded (≈{mem.total_gb:.2f} GB est.)")

    # Clear prior file so plots and tables stay aligned.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    all_rows: list[dict] = []
    for n_ctx in args.n_ctx:
        for spec in DEFAULT_SCHEMES:
            print(f"  profiling scheme={spec.scheme:20s} n_ctx={n_ctx:5d} ...", flush=True)
            try:
                res = profile_scheme(
                    model=model,
                    cache_spec=spec,
                    n_ctx=n_ctx,
                    n_decode=args.n_decode,
                    seed=args.seed,
                )
            except Exception as e:  # noqa: BLE001
                print(f"    FAILED: {e}")
                continue

            record = ResultRecord(
                experiment="profile",
                model=args.model,
                scheme=spec.scheme,
                git_sha=git_sha(),
                config_hash=_spec_hash(spec, n_ctx, args.n_decode),
                metrics={
                    **asdict(res),
                    "bits": spec.bits,
                    "group_size": spec.group_size,
                    "residual": spec.residual,
                    "symmetric": spec.symmetric,
                    "n_sink": spec.n_sink,
                    "n_recent": spec.n_recent,
                },
                tokens=n_ctx + args.n_decode,
            )
            append_result(args.out, record)
            all_rows.append(record.model_dump())
            print(
                f"    peak Δ={res.peak_mem_delta_mb:7.1f} MB  "
                f"prefill={res.prefill_wall_s:6.2f}s  "
                f"decode={res.decode_ms_per_token:6.1f}ms/tok  "
                f"({res.decode_tokens_per_s:5.1f} tok/s)"
            )

    print(f"\nWrote {len(all_rows)} rows to {args.out}")
    _plot_triptych(all_rows, args.figures_dir, args.model)
    print(f"Wrote figure to {args.figures_dir}/profile_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
