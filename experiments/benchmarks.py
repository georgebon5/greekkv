"""T18 — Greek multiple-choice benchmark runner.

Evaluates accuracy on three Greek benchmarks (ARC-Challenge, HellaSwag,
MMLU) across cache schemes. Predictions come from :func:`predict_mc` which
uses the T16 scorer under the hood — every scheme is evaluated under the
same prefill-then-score pattern as perplexity.

Usage
-----
::

    python experiments/benchmarks.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --n-samples 50 \\
        --benchmarks arc_el hellaswag_el mmlu_el \\
        --out results/benchmarks/qwen2_5_0_5b.jsonl

Output
------
One ``ResultRecord`` (T03) per ``(benchmark, scheme)`` with accuracy,
per-choice length-normalised NLLs are not stored (too large) but the fields
in ``metrics`` are enough for T22 to regenerate the comparison table.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset

from greekkv.eval import LoadSpec, load_model, predict_mc
from greekkv.quant import CacheSpec
from greekkv.utils import ResultRecord, append_result, git_sha, seed_all

DEFAULT_SCHEMES: list[CacheSpec] = [
    CacheSpec(scheme="fp16"),
    CacheSpec(scheme="int8-fakequant", bits=8, group_size=64),
    CacheSpec(scheme="int4-fakequant", bits=4, group_size=64),
    CacheSpec(scheme="int8-kivi", bits=8, group_size=64, residual=16),
    CacheSpec(scheme="int4-kivi", bits=4, group_size=64, residual=16),
    CacheSpec(scheme="eviction-sink", n_sink=4, n_recent=32),
]


BENCHMARKS: dict[str, dict[str, Any]] = {
    "arc_el": {
        "dataset": "ilsp/arc_greek",
        "config": "ARC-Challenge",
        "split": "test",
        "normaliser": "arc",
    },
    "hellaswag_el": {
        "dataset": "ilsp/hellaswag_greek",
        "config": None,
        "split": "validation",
        "normaliser": "hellaswag",
    },
    "mmlu_el": {
        "dataset": "ilsp/mmlu_greek",
        "config": "all",
        "split": "test",
        "normaliser": "mmlu",
    },
}


def _normalise(row: dict, kind: str) -> tuple[str, list[str], int]:
    """Return ``(question, choices, gold_index)`` from a raw benchmark row."""
    if kind == "arc":
        q = row["question"]
        choices_dict = row["choices"]
        labels = choices_dict["label"]
        texts = choices_dict["text"]
        gold_key = row["answerKey"]
        gold_idx = labels.index(gold_key)
        return q, texts, gold_idx

    if kind == "hellaswag":
        q = row["ctx"]
        texts = row["endings"]
        gold_idx = int(row["label"])
        return q, texts, gold_idx

    if kind == "mmlu":
        q = row["question"]
        texts = row["choices"]
        gold_idx = int(row["answer"])
        return q, texts, gold_idx

    raise ValueError(f"unknown normaliser {kind!r}")


def _spec_hash(spec: CacheSpec, benchmark: str, n_samples: int, seed: int) -> str:
    key = (
        f"{benchmark}|n={n_samples}|seed={seed}|{spec.scheme}|"
        f"bits={spec.bits}|g={spec.group_size}|r={spec.residual}|"
        f"sym={int(spec.symmetric)}|sink={spec.n_sink}|recent={spec.n_recent}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _evaluate(
    model,
    tokenizer,
    rows: list[dict],
    kind: str,
    cache_spec: CacheSpec,
) -> tuple[int, int, float]:
    """Score every row, return ``(correct, total, wall_seconds)``."""
    start = time.perf_counter()
    correct = 0
    for row in rows:
        try:
            question, choices, gold = _normalise(row, kind)
        except Exception:  # noqa: BLE001 — bad row, skip
            continue
        if not choices or gold < 0 or gold >= len(choices):
            continue
        pred = predict_mc(model, tokenizer, question, choices, cache_spec)
        if pred.pred_index == gold:
            correct += 1
    wall = time.perf_counter() - start
    return correct, len(rows), wall


def _load_benchmark_rows(info: dict, n_samples: int, seed: int) -> list[dict]:
    """Shuffle (seeded) and take the first ``n_samples`` rows."""
    ds = load_dataset(
        info["dataset"],
        info.get("config"),
        split=info["split"],
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed)
    n = min(n_samples, len(ds))
    return list(ds.select(range(n)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARKS),
        choices=list(BENCHMARKS),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/benchmarks/qwen2_5_0_5b.jsonl"),
    )
    args = ap.parse_args()

    seed_all(args.seed)

    print(f"Loading {args.model}...")
    spec = LoadSpec(name=args.model, dtype="fp32", device="auto")
    model, tok, mem = load_model(spec)
    model.eval()
    print(f"  loaded (≈{mem.total_gb:.2f} GB est.)")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    for bench_name in args.benchmarks:
        info = BENCHMARKS[bench_name]
        rows = _load_benchmark_rows(info, args.n_samples, args.seed)
        kind = info["normaliser"]
        print(f"\n=== {bench_name}  ({info['dataset']}, n={len(rows)}) ===")

        for cache_spec in DEFAULT_SCHEMES:
            try:
                correct, total, wall = _evaluate(model, tok, rows, kind, cache_spec)
            except Exception as e:  # noqa: BLE001
                print(f"  {cache_spec.scheme:20s}  FAILED: {e}")
                continue
            acc = correct / total if total else 0.0

            record = ResultRecord(
                experiment=f"benchmark:{bench_name}",
                model=args.model,
                scheme=cache_spec.scheme,
                git_sha=git_sha(),
                config_hash=_spec_hash(cache_spec, bench_name, args.n_samples, args.seed),
                metrics={
                    "benchmark": bench_name,
                    "dataset": info["dataset"],
                    "config": info.get("config"),
                    "split": info["split"],
                    "n_samples": total,
                    "correct": correct,
                    "accuracy": acc,
                    "wall_time_s": wall,
                    "bits": cache_spec.bits,
                    "group_size": cache_spec.group_size,
                    "residual": cache_spec.residual,
                    "symmetric": cache_spec.symmetric,
                    "n_sink": cache_spec.n_sink,
                    "n_recent": cache_spec.n_recent,
                },
                tokens=total,
            )
            append_result(args.out, record)
            print(
                f"  {cache_spec.scheme:20s}  acc={acc:.3f}  ({correct}/{total})  wall={wall:5.1f}s"
            )

    print(f"\nWrote results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
