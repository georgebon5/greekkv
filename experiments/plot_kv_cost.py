"""T08 — Greek-vs-English KV cost figure.

Consumes ``results/fertility.csv`` (T06) + the KV formula (T07) to show how
many more bytes of KV cache a given document of fixed word count costs in
Greek vs English, for each candidate model.

Usage
-----
::

    python experiments/plot_kv_cost.py \\
        --fertility results/fertility.csv \\
        --out results/figures/kv_greek_vs_english.png \\
        --n-words 1000 \\
        --bits 16

The grouped bar chart has one pair of bars per model (English vs Greek) with
the EL/EN ratio annotated and token counts below. The underlying numbers are
also written as sidecar CSV ``<out>.csv`` for downstream use (T30 dashboard).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoConfig

from greekkv.memory import arch_from_hf, kv_bytes

# Short labels used on the plot; order controls left-to-right grouping.
MODEL_LABEL = {
    "ilsp/Llama-Krikri-8B-Instruct": "Krikri-8B",
    "ilsp/Meltemi-7B-Instruct-v1.5": "Meltemi-7B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
}

COLOR_EN = "#4C72B0"
COLOR_EL = "#DD8452"


@dataclass(frozen=True)
class Row:
    model: str
    label: str
    language: str
    tpw: float
    n_tokens: int
    kv_mb: float


def load_fertility(path: Path) -> dict[tuple[str, str], float]:
    """Return ``{(tokenizer, language): mean_tpw}`` from the T06 CSV."""
    rows: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["tokenizer"], r["language"])] = float(r["mean_tpw"])
    return rows


def compute_rows(tpw: dict[tuple[str, str], float], n_words: int, bits: int) -> list[Row]:
    out: list[Row] = []
    for model in MODEL_LABEL:
        try:
            cfg = arch_from_hf(AutoConfig.from_pretrained(model))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot load config for {model}: {e}") from e
        for lang in ("en", "el"):
            key = (model, lang)
            if key not in tpw:
                raise KeyError(f"missing fertility row for {key} in CSV")
            n_tok = int(round(n_words * tpw[key]))
            mb = kv_bytes(cfg, n_ctx=n_tok, bits=bits) / (1024**2)
            out.append(
                Row(
                    model=model,
                    label=MODEL_LABEL[model],
                    language=lang,
                    tpw=tpw[key],
                    n_tokens=n_tok,
                    kv_mb=mb,
                )
            )
    return out


def plot(rows: list[Row], n_words: int, bits: int, out_path: Path) -> None:
    labels = [MODEL_LABEL[m] for m in MODEL_LABEL]
    by_label = {(r.label, r.language): r for r in rows}
    en = [by_label[(lbl, "en")].kv_mb for lbl in labels]
    el = [by_label[(lbl, "el")].kv_mb for lbl in labels]
    en_tok = [by_label[(lbl, "en")].n_tokens for lbl in labels]
    el_tok = [by_label[(lbl, "el")].n_tokens for lbl in labels]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(labels))
    w = 0.36
    b_en = ax.bar(x - w / 2, en, w, label="English", color=COLOR_EN)
    b_el = ax.bar(x + w / 2, el, w, label="Greek", color=COLOR_EL)

    y_max = max(en + el)
    for i, (ve, vl) in enumerate(zip(en, el, strict=True)):
        ratio = vl / ve if ve > 0 else float("nan")
        ax.text(
            x[i],
            max(ve, vl) + y_max * 0.04,
            f"{ratio:.2f}×",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    for bar, counts in ((b_en, en_tok), (b_el, el_tok)):
        for rect, n in zip(bar, counts, strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() / 2,
                f"{n:,} tok",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"KV cache (MB) — {bits}-bit, batch = 1")
    ax.set_title(
        f"KV-cache cost for a {n_words}-word document\n"
        "Greek vs English (same content, FLORES-200 devtest)"
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, y_max * 1.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_sidecar_csv(rows: list[Row], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "label", "language", "tpw", "n_tokens", "kv_mb"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fertility", type=Path, default=Path("results/fertility.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/kv_greek_vs_english.png"))
    ap.add_argument("--n-words", type=int, default=1000)
    ap.add_argument("--bits", type=int, default=16)
    args = ap.parse_args()

    if not args.fertility.exists():
        raise SystemExit(
            f"fertility CSV not found: {args.fertility}. "
            "Run T06 first: python experiments/fertility.py"
        )
    tpw = load_fertility(args.fertility)
    rows = compute_rows(tpw, n_words=args.n_words, bits=args.bits)

    plot(rows, n_words=args.n_words, bits=args.bits, out_path=args.out)
    sidecar = args.out.with_suffix(".csv")
    write_sidecar_csv(rows, sidecar)

    for r in rows:
        print(f"  {r.label:12s} {r.language}  n_tok={r.n_tokens:>6,}  KV={r.kv_mb:7.1f} MB")
    print(f"\nWrote {args.out}")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
