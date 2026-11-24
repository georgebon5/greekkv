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
from datetime import UTC, datetime
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
    "kernel_bench": REPO_ROOT / "results/kernel_bench/qwen2_5_0_5b_shape.jsonl",
    "profile": REPO_ROOT / "results/profile/qwen2_5_0_5b.jsonl",
}

OUTPUT = REPO_ROOT / "docs/RESULTS.md"
PLANNER_DATA_OUTPUT = REPO_ROOT / "planner/public/data.json"

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


def kernel_bench_section(rows: list[dict]) -> str:
    # Pivot to one row per impl, one column per N.
    by_impl: dict[str, dict[int, tuple[float, float, str]]] = defaultdict(dict)
    for r in rows:
        m = r["metrics"]
        by_impl[r["scheme"]][int(m["N"])] = (
            float(m["wall_ms_mean"]),
            float(m["wall_ms_p95"]),
            str(m["device"]),
        )
    ns = sorted({int(r["metrics"]["N"]) for r in rows})
    impl_order = ["fp16_baseline", "reference_unfused", "triton_fused"]

    header_cells = " | ".join(f"N = {n}" for n in ns)
    sep_cells = " | ".join(["---:"] * len(ns))
    lines = [
        "## M4 — Decode-attention kernel benchmark (T25)",
        "",
        "Three single-query decode-attention implementations across `N_ctx` on "
        "synthetic shape (H=8, H_kv=2, D=64, G=64). Wall time in ms per call, "
        "mean over timed iterations after 3 warmups.",
        "",
        f"| Impl | {header_cells} |",
        f"|---|{sep_cells}|",
    ]
    for impl in impl_order:
        if impl not in by_impl:
            continue
        row_cells = []
        for n in ns:
            if n not in by_impl[impl]:
                row_cells.append("—")
            else:
                mean_ms, _p95, _dev = by_impl[impl][n]
                row_cells.append(f"{mean_ms:.2f}")
        lines.append(f"| `{impl}` | " + " | ".join(row_cells) + " |")

    devices = {dev for impl_data in by_impl.values() for _, _, dev in impl_data.values()}
    if "cuda" not in devices:
        lines.append("")
        lines.append(
            "_`triton_fused` is skipped on this run — CUDA + Triton unavailable on "
            "the free-tier host. The M4 Kaggle T4 re-run will fill that row._"
        )
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
    kernel_bench = load_jsonl(RESULT_FILES["kernel_bench"])

    parts = [HEAD]
    parts.append(fertility_section(fertility))
    parts.append(kv_cost_section(kv_cost))
    parts.append(perplexity_section(perplexity))
    parts.append(benchmarks_section(benchmarks))
    parts.append(needle_section(needle))
    parts.append(kernel_bench_section(kernel_bench))
    parts.append(
        "## Where to go next\n\n"
        "- `docs/fertility.md`, `docs/kv_cost.md`, `docs/kv_outliers.md`, "
        "`docs/perplexity.md`, `docs/benchmarks.md`, `docs/needle_haystack.md`, "
        "`docs/kernel_bench.md` "
        "carry the full reading of each experiment.\n"
        "- `scripts/run_sweep.py` (T21) resumes the perplexity + fidelity sweep "
        "from any JSONL checkpoint.\n"
        "- Figures live under `results/figures/`; regenerate them with the "
        "experiment-specific plot scripts under `experiments/plot_*.py`.\n"
    )
    return "\n".join(parts)


def build_planner_data() -> dict:
    """Produce the snapshot the static planner app reads at load.

    The planner recomputes capacity numbers in-browser (TypeScript port
    of :func:`greekkv.memory.capacity.capacity`) but it leans on this
    file for anything *measured*: perplexity, decode latency, and peak
    memory. Writing it from the same build_report entry point keeps
    docs/RESULTS.md and the planner UI in lock-step.
    """
    perplexity_rows = load_jsonl(RESULT_FILES["perplexity"])
    profile_rows = load_jsonl(RESULT_FILES["profile"])

    # Group perplexity rows by (model, scheme, n_ctx) → headline metrics.
    ppl_index: dict[tuple[str, str, int], dict[str, float]] = {}
    for r in perplexity_rows:
        m = r["metrics"]
        key = (r["model"], r["scheme"], int(m["n_ctx"]))
        ppl_index[key] = {
            "perplexity": float(m["perplexity"]),
            "kl_mean": float(m.get("kl_mean", 0.0)),
            "top1_agree": float(m.get("top1_agree", 0.0)),
        }

    prof_index: dict[tuple[str, str, int], dict[str, float]] = {}
    for r in profile_rows:
        m = r["metrics"]
        if "decode_ms_per_token" not in m:
            continue
        key = (r["model"], r["scheme"], int(m["n_ctx"]))
        prof_index[key] = {
            "decode_ms_per_token": float(m["decode_ms_per_token"]),
            "peak_mem_mb": float(m.get("peak_mem_mb", 0.0)),
        }

    measurements: list[dict] = []
    for key in sorted(set(ppl_index) | set(prof_index)):
        model, scheme, n_ctx = key
        entry: dict = {"model": model, "scheme": scheme, "n_ctx": n_ctx}
        entry.update(ppl_index.get(key, {}))
        entry.update(prof_index.get(key, {}))
        measurements.append(entry)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": sorted({m["model"] for m in measurements}),
        "schemes": sorted({m["scheme"] for m in measurements}),
        "n_ctx_grid": sorted({m["n_ctx"] for m in measurements}),
        "measurements": measurements,
    }


def main() -> int:
    ensure_files_exist()
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}  ({len(report)} chars)")

    planner_data = build_planner_data()
    PLANNER_DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PLANNER_DATA_OUTPUT.write_text(
        json.dumps(planner_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {PLANNER_DATA_OUTPUT.relative_to(REPO_ROOT)}  "
        f"({len(planner_data['measurements'])} measurement rows)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
