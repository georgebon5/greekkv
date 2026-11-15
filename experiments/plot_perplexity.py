"""T17 — plot sliding-window PPL across cache schemes and context lengths."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

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
    # Group by scheme → list of (n_ctx, ppl)
    by_scheme: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        m = r["metrics"]
        by_scheme[r["scheme"]].append((m["n_ctx"], m["perplexity"]))
    for s in by_scheme:
        by_scheme[s].sort()

    model_name = rows[0]["model"] if rows else "unknown"

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for scheme in SCHEME_ORDER:
        if scheme not in by_scheme:
            continue
        xs, ys = zip(*by_scheme[scheme], strict=True)
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            label=scheme,
            color=SCHEME_COLOR.get(scheme, "black"),
        )

    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r["metrics"]["n_ctx"] for r in rows}))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlabel("Context length (tokens, log₂ scale)")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_title(f"Sliding-window PPL on Greek text — {model_name}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
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
        default=Path("results/perplexity/qwen2_5_0_5b.jsonl"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures/perplexity_qwen2_5_0_5b.png"),
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
