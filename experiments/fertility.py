"""T06 — Tokenizer fertility study (CPU, ~minutes on first run due to downloads).

Measures tokens-per-word and tokens-per-char for each candidate tokenizer on
FLORES-200 Greek and English (same content, both languages).

Usage:
    python experiments/fertility.py \\
        --out results/fertility.csv \\
        --n-samples 500 \\
        --seed 42

The output CSV is consumed by `docs/fertility.md` and the T08 figure.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from greekkv.tokenization import fertility_report, load_flores_pair
from greekkv.utils import seed_all

# Tokenizers verified ungated via HF API (see T06 PR). Keep this list small —
# fertility is cheap to recompute if we extend it later.
TOKENIZERS = [
    "ilsp/Llama-Krikri-8B-Instruct",
    "ilsp/Meltemi-7B-Instruct-v1.5",
    "Qwen/Qwen2.5-7B-Instruct",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/fertility.csv"))
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tokenizers",
        nargs="+",
        default=TOKENIZERS,
        help="HF model IDs whose tokenizers to evaluate.",
    )
    args = ap.parse_args()

    seed_all(args.seed)

    print(f"Loading FLORES devtest (eng_Latn + ell_Grek, limit={args.n_samples})...")
    pairs = load_flores_pair(limit=args.n_samples)
    en_texts = [p.src for p in pairs]
    el_texts = [p.tgt for p in pairs]
    print(f"  -> {len(pairs)} parallel pairs loaded.")

    rows: list[dict] = []
    for name in args.tokenizers:
        print(f"\nTokenizer: {name}")
        tok = AutoTokenizer.from_pretrained(name)
        for lang, texts in (("en", en_texts), ("el", el_texts)):
            stats = fertility_report(
                texts=texts,
                tokenizer=tok,
                tokenizer_name=name,
                language=lang,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            rows.append(asdict(stats))
            print(
                f"  {lang}: TPW={stats.mean_tpw:.3f} "
                f"[{stats.ci_lo_tpw:.3f}, {stats.ci_hi_tpw:.3f}]  "
                f"TPC={stats.mean_tpc:.3f} "
                f"[{stats.ci_lo_tpc:.3f}, {stats.ci_hi_tpc:.3f}]"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
