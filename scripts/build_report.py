#!/usr/bin/env python
"""T22 — Regenerate ``docs/RESULTS.md`` from committed result files.

Reads every artefact produced by M1–M3 (CSV + JSONL + JSON summaries),
aggregates into markdown tables, and writes a single canonical report.
The script is **pure data wrangling**: it does not score anything, does
not call a model, and fails loudly if a referenced result file is missing.

Usage
-----
::

    python scripts/build_report.py
    # or, when wired into the Makefile:
    make reproduce

CI benefit: a reviewer can audit ``docs/RESULTS.md`` for consistency with
the committed JSONL in a single grep.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from greekkv.tokenization.fertility import bootstrap_mean_ci

REPO_ROOT = Path(__file__).resolve().parents[1]

RESULT_FILES: dict[str, Path] = {
    "fertility": REPO_ROOT / "results/fertility.csv",
    "kv_cost": REPO_ROOT / "results/figures/kv_greek_vs_english.csv",
    "perplexity": REPO_ROOT / "results/perplexity/qwen2_5_0_5b.jsonl",
    "benchmarks": REPO_ROOT / "results/benchmarks/qwen2_5_0_5b.jsonl",
    "needle": REPO_ROOT / "results/needle/qwen2_5_0_5b.jsonl",
}

OUTPUT = REPO_ROOT / "docs/RESULTS.md"

SCHEME_ORDER = [
    "fp16",
    "int8-fakequant",
    "int8-kivi",
    "int4-fakequant",
    "int4-kivi",
    "eviction-sink",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_files_exist() -> None:
    missing = [name for name, p in RESULT_FILES.items() if not p.exists()]
    if missing:
        detail = "\n  ".join(f"{n}: {RESULT_FILES[n]}" for n in missing)
        raise SystemExit(f"Missing result files — cannot build report:\n  {detail}")


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _by_scheme_order(rows: list[dict], key: str = "scheme") -> list[dict]:
    return sorted(rows, key=lambda r: SCHEME_ORDER.index(r[key]) if r[key] in SCHEME_ORDER else 999)


def fertility_section(rows: list[dict]) -> str:
    # Group by tokenizer, pivot EN vs EL on tokens_per_word.
    by_tok: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_tok[r["tokenizer"]][r["language"]] = float(r["mean_tpw"])

    lines = [
        "## M1 — Tokenizer fertility (T06)",
        "",
        "Tokens per whitespace-separated word (EN vs EL, FLORES-200 devtest, 500 samples, 95% bootstrap CIs in `results/fertility.csv`):",
        "",
        "| Tokenizer | EN | EL | **EL / EN** |",
        "|---|---:|---:|---:|",
    ]
    for tok, cells in by_tok.items():
        en = cells.get("en")
        el = cells.get("el")
        if en is None or el is None:
            continue
        ratio = el / en if en else float("nan")
        lines.append(f"| `{tok}` | {en:.2f} | {el:.2f} | **{ratio:.2f}×** |")
    return "\n".join(lines) + "\n"


def kv_cost_section(rows: list[dict]) -> str:
    # Group by model label.
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_model[r["label"]][r["language"]] = r

    lines = [
        "## M1 — KV-cache cost per document (T08)",
        "",
        "1000-word Greek Wikipedia doc, FP16 cache, batch = 1:",
        "",
        "| Model | EN tokens | EN KV (MB) | EL tokens | EL KV (MB) | **EL / EN** |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, cells in by_model.items():
        if "en" not in cells or "el" not in cells:
            continue
        en = cells["en"]
        el = cells["el"]
        ratio = float(el["kv_mb"]) / float(en["kv_mb"]) if float(en["kv_mb"]) > 0 else float("nan")
        lines.append(
            f"| {model} | {int(float(en['n_tokens'])):,} | {float(en['kv_mb']):.1f} | "
            f"{int(float(el['n_tokens'])):,} | {float(el['kv_mb']):.1f} | **{ratio:.2f}×** |"
        )
    return "\n".join(lines) + "\n"


def perplexity_section(rows: list[dict]) -> str:
    # Pivot: scheme × n_ctx → PPL. Keep fidelity columns at n_ctx=512 as a representative slice.
    by_scheme_ctx: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        m = r["metrics"]
        by_scheme_ctx[r["scheme"]][int(m["n_ctx"])] = m

    n_ctx_values = sorted({int(r["metrics"]["n_ctx"]) for r in rows})
    header = ["| Scheme |"] + [f" PPL n_ctx={n} |" for n in n_ctx_values]
    sep = ["|---|"] + ["---:|" for _ in n_ctx_values]
    lines = [
        "## M3 — Perplexity sweep (T17 + T20 fidelity)",
        "",
        "Sliding-window PPL on Greek Wikipedia, `Qwen/Qwen2.5-0.5B`, 15 chunks per cell:",
        "",
        "".join(header),
        "".join(sep),
    ]
    for scheme in SCHEME_ORDER:
        if scheme not in by_scheme_ctx:
            continue
        cells = by_scheme_ctx[scheme]
        row = [f"| **{scheme}** |"]
        for n in n_ctx_values:
            if n in cells:
                row.append(f" {cells[n]['perplexity']:.2f} |")
            else:
                row.append(" — |")
        lines.append("".join(row))

    # Fidelity at the middle n_ctx.
    middle_n = sorted(n_ctx_values)[len(n_ctx_values) // 2]
    lines.append("")
    lines.append(f"### Fidelity vs FP16 (n_ctx = {middle_n})")
    lines.append("")
    lines.append("| Scheme | top-1 agree | top-5 agree | KL mean | KL p95 |")
    lines.append("|---|---:|---:|---:|---:|")
    for scheme in SCHEME_ORDER:
        if scheme not in by_scheme_ctx or middle_n not in by_scheme_ctx[scheme]:
            continue
        m = by_scheme_ctx[scheme][middle_n]
        top1 = m.get("top1_agree", float("nan"))
        top5 = m.get("top5_agree", float("nan"))
        kl_mean = m.get("kl_mean", float("nan"))
        kl_p95 = m.get("kl_p95", float("nan"))
        lines.append(f"| {scheme} | {top1:.3f} | {top5:.3f} | {kl_mean:.3f} | {kl_p95:.3f} |")
    return "\n".join(lines) + "\n"


def benchmarks_section(rows: list[dict]) -> str:
    # Group by benchmark → scheme → accuracy.
    by_bench: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        m = r["metrics"]
        by_bench[m["benchmark"]][r["scheme"]] = m

    benchmarks = sorted(by_bench)
    header = ["| Scheme |"] + [f" {b} |" for b in benchmarks]
    sep = ["|---|"] + ["---:|" for _ in benchmarks]
    lines = [
        "## M3 — Greek multi-choice benchmark accuracy (T18)",
        "",
        "ILSP Greek translations, `Qwen/Qwen2.5-0.5B`. Numbers in parentheses are "
        "bootstrap 95% CIs on accuracy, 1000 resamples.",
        "",
        "".join(header),
        "".join(sep),
    ]
    for scheme in SCHEME_ORDER:
        row = [f"| **{scheme}** |"]
        for b in benchmarks:
            m = by_bench[b].get(scheme)
            if m is None:
                row.append(" — |")
                continue
            # Build the per-question correct list to bootstrap on.
            n = int(m["n_samples"])
            c = int(m["correct"])
            samples = [1.0] * c + [0.0] * (n - c)
            if n >= 10:
                _, lo, hi = bootstrap_mean_ci(samples, n_boot=1000, seed=0)
                row.append(f" {c}/{n} = {c / n:.2f} ({lo:.2f}, {hi:.2f}) |")
            else:
                row.append(f" {c}/{n} = {c / n:.2f} |")
        lines.append("".join(row))
    return "\n".join(lines) + "\n"


def needle_section(rows: list[dict]) -> str:
    # Overall retrieval accuracy per scheme.
    by_scheme: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_scheme[r["scheme"]].append(int(r["metrics"]["correct"]))

    lines = [
        "## M3 — Needle-in-a-haystack retrieval (T19)",
        "",
        "Greek haystack, 3 needle categories (number / name / place), grid of "
        "context lengths × depths. Overall accuracy and bootstrap 95% CI across trials:",
        "",
        "| Scheme | Retrieval accuracy | Bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for scheme in SCHEME_ORDER:
        if scheme not in by_scheme:
            continue
        xs = by_scheme[scheme]
        n = len(xs)
        acc = sum(xs) / n
        mean, lo, hi = bootstrap_mean_ci([float(x) for x in xs], n_boot=1000, seed=0)
        lines.append(f"| {scheme} | {sum(xs)}/{n} = {acc:.2f} | ({lo:.2f}, {hi:.2f}) |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


HEAD = """# GreekKV — aggregated results

**Regenerated by `scripts/build_report.py`** from committed JSONL / CSV
result files. Do **not** hand-edit this file; edit the sources and re-run.

Scope: this report aggregates the measured numbers from M1–M3. Model
coverage is limited to `Qwen/Qwen2.5-0.5B` while the project runs on
free-tier CPU. M4 Kaggle T4 re-runs will extend every table to Krikri /
Meltemi at the same configuration.
"""


def build_report() -> str:
    fertility = load_csv(RESULT_FILES["fertility"])
    kv_cost = load_csv(RESULT_FILES["kv_cost"])
    perplexity = load_jsonl(RESULT_FILES["perplexity"])
    benchmarks = load_jsonl(RESULT_FILES["benchmarks"])
    needle = load_jsonl(RESULT_FILES["needle"])

    parts = [HEAD]
    parts.append(fertility_section(fertility))
    parts.append(kv_cost_section(kv_cost))
    parts.append(perplexity_section(perplexity))
    parts.append(benchmarks_section(benchmarks))
    parts.append(needle_section(needle))
    parts.append(
        "## Where to go next\n\n"
        "- `docs/fertility.md`, `docs/kv_cost.md`, `docs/kv_outliers.md`, "
        "`docs/perplexity.md`, `docs/benchmarks.md`, `docs/needle_haystack.md` "
        "carry the full reading of each experiment.\n"
        "- `scripts/run_sweep.py` (T21) resumes the perplexity + fidelity sweep "
        "from any JSONL checkpoint.\n"
        "- Figures live under `results/figures/`; regenerate them with the "
        "experiment-specific plot scripts under `experiments/plot_*.py`.\n"
    )
    return "\n".join(parts)


def main() -> int:
    ensure_files_exist()
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}  ({len(report)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
