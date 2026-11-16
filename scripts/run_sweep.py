#!/usr/bin/env python
"""T21 — Resumable experiment sweep runner.

Reads a sweep YAML that enumerates ``(cache scheme, context length)``
cells, scores each one via :func:`greekkv.eval.run_perplexity_cell`, and
appends one JSONL row per cell to the output file. **Interrupted runs
resume cleanly**: the runner first reads the existing output JSONL and
skips any cell whose ``(scheme, n_ctx)`` hash is already present.

Usage
-----
::

    scripts/run_sweep.py --config configs/sweep_perplexity_qwen_0_5b.yaml

The sweep YAML shape is::

    experiment: perplexity
    model: Qwen/Qwen2.5-0.5B
    seed: 42
    n_samples: 8
    n_chunks: 15
    ctx_frac: 0.5
    n_ctx: [256, 512, 1024]
    schemes:
      - {scheme: fp16}
      - {scheme: int4-kivi, bits: 4, group_size: 64, residual: 16}
    output: results/perplexity/qwen2_5_0_5b.jsonl

Only the ``perplexity`` experiment kind is wired up here. ``benchmark``
and ``needle`` sweeps use the same resume pattern and will join this
runner in M4; the T22 report regenerates downstream tables regardless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from greekkv.eval import (
    DEFAULT_SCHEMES,
    LoadSpec,
    chunks_from_ids,
    collect_fp16_logits,
    load_model,
    run_perplexity_cell,
    spec_hash,
)
from greekkv.quant import CacheSpec
from greekkv.tokenization import load_el_wikipedia_sample
from greekkv.utils import seed_all


def load_existing_hashes(path: Path) -> set[str]:
    """Collect ``config_hash`` of every row currently in ``path``.

    Rows with malformed JSON are reported and skipped — we don't want a
    stray bad line to block a resumable run.
    """
    hashes: set[str] = set()
    if not path.exists():
        return hashes
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN: malformed line {i} in {path}: {e}", file=sys.stderr)
                continue
            h = row.get("config_hash")
            if isinstance(h, str):
                hashes.add(h)
    return hashes


def plan_cells(
    schemes: list[CacheSpec], n_ctx_list: list[int], done: set[str]
) -> list[tuple[CacheSpec, int, bool]]:
    """Enumerate ``(scheme, n_ctx, already_done)`` triples in a stable order."""
    cells: list[tuple[CacheSpec, int, bool]] = []
    for n_ctx in n_ctx_list:
        for spec in schemes:
            cells.append((spec, n_ctx, spec_hash(spec, n_ctx) in done))
    return cells


def _schemes_from_yaml(items: list[dict[str, Any]] | None) -> list[CacheSpec]:
    if not items:
        return list(DEFAULT_SCHEMES)
    return [CacheSpec(**item) for item in items]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned cells and exit without loading the model.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg.get("experiment", "perplexity") != "perplexity":
        raise SystemExit(
            f"run_sweep.py only supports experiment=perplexity, got {cfg.get('experiment')!r}"
        )

    model_name: str = cfg["model"]
    seed: int = int(cfg.get("seed", 42))
    n_samples: int = int(cfg.get("n_samples", 6))
    n_chunks: int = int(cfg.get("n_chunks", 15))
    ctx_frac: float = float(cfg.get("ctx_frac", 0.5))
    n_ctx_list: list[int] = [int(x) for x in cfg["n_ctx"]]
    schemes: list[CacheSpec] = _schemes_from_yaml(cfg.get("schemes"))
    out_path: Path = Path(cfg["output"])

    done = load_existing_hashes(out_path)
    cells = plan_cells(schemes, n_ctx_list, done)
    todo = [c for c in cells if not c[2]]
    skipped = [c for c in cells if c[2]]

    print(f"Sweep config:  {args.config}")
    print(f"Output:        {out_path}")
    print(f"Existing rows: {len(done)} (dedupe keys)")
    print(f"Planned cells: {len(cells)}  →  TODO: {len(todo)},  SKIP: {len(skipped)}")
    for spec, n_ctx, is_done in cells:
        tag = "SKIP" if is_done else "TODO"
        print(f"  [{tag}] {spec.scheme:20s}  n_ctx={n_ctx}")

    if args.dry_run or not todo:
        if args.dry_run:
            print("\n(dry-run: not loading model or running cells)")
        elif not todo:
            print("\nNothing to do — all cells already scored.")
        return 0

    seed_all(seed)

    print(f"\nLoading {model_name}...")
    model, tok, mem = load_model(LoadSpec(name=model_name, dtype="fp32", device="auto"))
    model.eval()
    print(f"  loaded (≈{mem.total_gb:.2f} GB est.)")

    articles = load_el_wikipedia_sample(n=n_samples, seed=seed)
    text = "\n\n".join(articles)
    all_ids = tok.encode(text, add_special_tokens=False)
    print(f"Loaded {len(articles)} Greek Wikipedia articles → {len(all_ids)} tokens")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Group remaining cells by n_ctx so the fp16 reference logits are collected once per n_ctx.
    by_ctx: dict[int, list[CacheSpec]] = {}
    for spec, n_ctx, _ in todo:
        by_ctx.setdefault(n_ctx, []).append(spec)

    n_done = 0
    for n_ctx in sorted(by_ctx):
        chunks = chunks_from_ids(all_ids, n_ctx, n_chunks)
        if not chunks:
            print(f"SKIP n_ctx={n_ctx}: not enough tokens for even one chunk")
            continue
        print(f"\nn_ctx={n_ctx}  → {len(chunks)} chunks × {len(by_ctx[n_ctx])} schemes to run")

        # fp16 logits are only needed if some non-fp16 scheme is TODO for this n_ctx.
        non_fp16_todo = [s for s in by_ctx[n_ctx] if s.scheme != "fp16"]
        fp16_logits = None
        if non_fp16_todo:
            print(f"  (collecting fp16 reference logits over {len(chunks)} chunks)")
            fp16_logits = collect_fp16_logits(model, chunks, int(n_ctx * ctx_frac))

        for spec in by_ctx[n_ctx]:
            fp16_ref = None if spec.scheme == "fp16" else fp16_logits
            try:
                row = run_perplexity_cell(
                    model=model,
                    chunks=chunks,
                    cache_spec=spec,
                    n_ctx=n_ctx,
                    ctx_frac=ctx_frac,
                    model_name=model_name,
                    out=out_path,
                    fp16_logits_by_chunk=fp16_ref,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {spec.scheme:20s} n_ctx={n_ctx:5d}  FAILED: {e}")
                continue
            n_done += 1
            extra = f"  top1={row['top1']:.3f}  KL={row['kl_mean']:.3f}" if "top1" in row else ""
            print(
                f"  {row['scheme']:20s} n_ctx={row['n_ctx']:5d}  "
                f"PPL={row['ppl']:8.2f}  wall={row['wall_s']:5.1f}s{extra}  "
                f"[{n_done}/{len(todo)}]"
            )

    print(f"\nWrote {n_done} new rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
