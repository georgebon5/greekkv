"""T18 — Plot benchmark accuracy by scheme, one panel per benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCHEME_ORDER = [
    "fp16",
    "int8-fakequant",
    "int4-fakequant",
    "int8-kivi",
    "int4-kivi",
    "eviction-sink",
]

SCHEME_COLOR = {
    "fp16": "#2E7D32",
    "int8-fakequant": "#1976D2",
    "int4-fakequant": "#7B1FA2",
    "int8-kivi": "#F57C00",
    "int4-kivi": "#D32F2F",
    "eviction-sink": "#5D4037",
}


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot(rows: list[dict], out: Path) -> None:
    # Group by benchmark → {scheme: accuracy}
    by_bench: dict[str, dict[str, float]] = defaultdict(dict)
    n_by_bench: dict[str, int] = {}
    for r in rows:
        m = r["metrics"]
        by_bench[m["benchmark"]][r["scheme"]] = m["accuracy"]
        n_by_bench[m["benchmark"]] = m["n_samples"]

    benchmarks = sorted(by_bench)
    model_name = rows[0]["model"] if rows else "unknown"

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(4.5 * len(benchmarks), 4.5), sharey=True)
    if len(benchmarks) == 1:
        axes = [axes]

    for ax, bench in zip(axes, benchmarks, strict=True):
        schemes_here = [s for s in SCHEME_ORDER if s in by_bench[bench]]
        accs = [by_bench[bench][s] for s in schemes_here]
        colors = [SCHEME_COLOR[s] for s in schemes_here]
        xs = np.arange(len(schemes_here))
        bars = ax.bar(xs, accs, color=colors)
        baseline = by_bench[bench].get("fp16")
        if baseline is not None:
            ax.axhline(y=baseline, color="#2E7D32", ls="--", lw=0.8, alpha=0.6)

        for rect, acc in zip(bars, accs, strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.01,
                f"{acc:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(xs)
        ax.set_xticklabels(schemes_here, rotation=35, ha="right")
        ax.set_ylim(0, max(0.6, max(accs) * 1.15))
        ax.set_title(f"{bench}  (n={n_by_bench[bench]})")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Accuracy")
    fig.suptitle(f"Greek multiple-choice accuracy — {model_name}", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("results/benchmarks/qwen2_5_0_5b.jsonl"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures/benchmarks_qwen2_5_0_5b.png"),
    )
    args = ap.parse_args()
    rows = load_rows(args.in_path)
    if not rows:
        raise SystemExit(f"no rows in {args.in_path}")
    plot(rows, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
