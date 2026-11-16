"""T19 — Greek needle-in-a-haystack retrieval test.

For each (context length, needle depth, cache scheme) we build a context by
inserting a Greek needle sentence into a Greek Wikipedia haystack, append
the question at the end, then score two candidate answers (correct vs
distractor) using the T16 prefill+score pattern. The correct candidate
must beat the distractor on length-normalised NLL — that is the retrieval
success criterion.

Needles come from ``configs/needles_el.yaml`` (number / name / place).
Grid + haystack params also come from the config so the experiment is
reproducible from one file.

Usage
-----
::

    python experiments/needle.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --config configs/needles_el.yaml \\
        --out results/needle/qwen2_5_0_5b.jsonl \\
        --figures-dir results/figures/needle/
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from greekkv.eval import LoadSpec, load_model
from greekkv.eval.scorer import _score_against_cache
from greekkv.quant import CacheSpec, build_cache
from greekkv.tokenization import load_el_wikipedia_sample
from greekkv.utils import ResultRecord, append_result, git_sha, seed_all

DEFAULT_SCHEMES: list[CacheSpec] = [
    CacheSpec(scheme="fp16"),
    CacheSpec(scheme="int8-fakequant", bits=8, group_size=64),
    CacheSpec(scheme="int4-fakequant", bits=4, group_size=64),
    CacheSpec(scheme="int8-kivi", bits=8, group_size=64, residual=16),
    CacheSpec(scheme="int4-kivi", bits=4, group_size=64, residual=16),
    CacheSpec(scheme="eviction-sink", n_sink=4, n_recent=64),
]

QUESTION_PREFIX = "\n\n"
ANSWER_TAG = "\nΑπάντηση:"


def _spec_hash(spec: CacheSpec, length: int, depth: float, needle_idx: int) -> str:
    key = (
        f"{spec.scheme}|bits={spec.bits}|g={spec.group_size}|r={spec.residual}|"
        f"sym={int(spec.symmetric)}|sink={spec.n_sink}|recent={spec.n_recent}|"
        f"L={length}|d={depth:.2f}|n={needle_idx}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _build_context_ids(
    tokenizer,
    haystack_ids: list[int],
    needle_text: str,
    question_text: str,
    total_length: int,
    depth_frac: float,
) -> list[int]:
    """Insert needle at ``depth_frac`` within haystack, append question, trim to length."""
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    tail_ids = tokenizer.encode(
        QUESTION_PREFIX + question_text + ANSWER_TAG, add_special_tokens=False
    )
    budget = total_length - len(needle_ids) - len(tail_ids)
    if budget <= 0:
        raise ValueError(
            f"context length {total_length} too small for needle+question "
            f"({len(needle_ids) + len(tail_ids)} tokens)"
        )
    if len(haystack_ids) < budget:
        raise ValueError(f"haystack has {len(haystack_ids)} tokens, need at least {budget}")
    hay = haystack_ids[:budget]
    insert_pos = int(len(hay) * depth_frac)
    # Clip insert_pos so the needle lives fully inside the haystack slice.
    insert_pos = max(0, min(insert_pos, len(hay)))
    ctx_ids = hay[:insert_pos] + needle_ids + hay[insert_pos:] + tail_ids
    return ctx_ids[:total_length]


def _retrieval_success(
    model, tokenizer, context_ids: list[int], correct: str, distractor: str, spec: CacheSpec
) -> tuple[int, float, float]:
    """Return ``(1 if correct wins else 0, nll_correct, nll_distractor)``."""
    device = next(model.parameters()).device
    ctx_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
    correct_ids = tokenizer.encode(" " + correct, add_special_tokens=False, return_tensors="pt").to(
        device
    )
    distractor_ids = tokenizer.encode(
        " " + distractor, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    cache_c = build_cache(spec, model_config=model.config)
    cache_d = build_cache(spec, model_config=model.config)
    res_c = _score_against_cache(model, ctx_tensor, correct_ids, cache_c)
    res_d = _score_against_cache(model, ctx_tensor, distractor_ids, cache_d)
    return int(res_c.nll_mean < res_d.nll_mean), res_c.nll_mean, res_d.nll_mean


def _load_haystack_tokens(tokenizer, n_articles: int, seed: int) -> list[int]:
    articles = load_el_wikipedia_sample(n=n_articles, seed=seed)
    text = "\n\n".join(articles)
    return tokenizer.encode(text, add_special_tokens=False)


def _plot_heatmaps(
    rows: list[dict], schemes: list[str], lengths: list[int], depths: list[float], out_dir: Path
) -> None:
    """One heatmap per scheme, depth × length cells coloured by accuracy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Aggregate: scheme → (L, d) → list of successes
    agg: dict[str, dict[tuple[int, float], list[int]]] = {s: {} for s in schemes}
    for r in rows:
        m = r["metrics"]
        key = (m["length"], m["depth"])
        agg.setdefault(r["scheme"], {}).setdefault(key, []).append(m["correct"])

    for scheme in schemes:
        heat = np.zeros((len(depths), len(lengths)))
        for i, d in enumerate(depths):
            for j, ln in enumerate(lengths):
                v = agg.get(scheme, {}).get((ln, d), [])
                heat[i, j] = sum(v) / len(v) if v else float("nan")

        fig, ax = plt.subplots(figsize=(1.2 * len(lengths) + 2.5, 1.2 * len(depths) + 2))
        im = ax.imshow(heat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([str(ln) for ln in lengths])
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f"{d:.0%}" for d in depths])
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel("Needle depth")
        ax.set_title(f"Needle retrieval — {scheme}")
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                val = heat[i, j]
                label = f"{val:.2f}" if np.isfinite(val) else "—"
                ax.text(j, i, label, ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, label="accuracy")
        fig.tight_layout()
        fig.savefig(out_dir / f"needle_{scheme}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--config", type=Path, default=Path("configs/needles_el.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/needle/qwen2_5_0_5b.jsonl"))
    ap.add_argument("--figures-dir", type=Path, default=Path("results/figures/needle"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    needles = cfg["needles"]
    lengths = cfg["grid"]["context_lengths"]
    depths = cfg["grid"]["depths"]
    n_articles = cfg["haystack"]["n_articles"]
    haystack_seed = cfg["haystack"]["seed"]

    print(f"Loading {args.model}...")
    spec = LoadSpec(name=args.model, dtype="fp32", device="auto")
    model, tok, mem = load_model(spec)
    model.eval()
    print(f"  loaded (≈{mem.total_gb:.2f} GB est.)")

    print(f"Building haystack from {n_articles} Greek Wikipedia articles...")
    haystack_ids = _load_haystack_tokens(tok, n_articles, haystack_seed)
    print(f"  {len(haystack_ids)} tokens available")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Clear prior file so a fresh run doesn't mix with old rows.
    if args.out.exists():
        args.out.unlink()

    scheme_names = [s.scheme for s in DEFAULT_SCHEMES]
    total_cells = len(lengths) * len(depths) * len(DEFAULT_SCHEMES) * len(needles)
    print(
        f"Grid: {len(lengths)} lengths × {len(depths)} depths × {len(DEFAULT_SCHEMES)} schemes × {len(needles)} needles = {total_cells} trials"
    )

    rows: list[dict] = []
    for ln in lengths:
        for d in depths:
            for needle_idx, n in enumerate(needles):
                context_ids = _build_context_ids(
                    tok,
                    haystack_ids,
                    n["needle"],
                    n["question"],
                    total_length=ln,
                    depth_frac=d,
                )
                for scheme in DEFAULT_SCHEMES:
                    start = time.perf_counter()
                    try:
                        correct, nll_c, nll_d = _retrieval_success(
                            model, tok, context_ids, n["correct"], n["distractor"], scheme
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"  L={ln} d={d} needle={needle_idx} {scheme.scheme} FAILED: {e}")
                        continue
                    wall = time.perf_counter() - start

                    record = ResultRecord(
                        experiment="needle_haystack",
                        model=args.model,
                        scheme=scheme.scheme,
                        git_sha=git_sha(),
                        config_hash=_spec_hash(scheme, ln, d, needle_idx),
                        metrics={
                            "length": ln,
                            "depth": d,
                            "needle_idx": needle_idx,
                            "needle_category": n["category"],
                            "correct": correct,
                            "nll_correct": nll_c,
                            "nll_distractor": nll_d,
                            "bits": scheme.bits,
                            "group_size": scheme.group_size,
                            "residual": scheme.residual,
                            "n_sink": scheme.n_sink,
                            "n_recent": scheme.n_recent,
                            "wall_time_s": wall,
                        },
                        tokens=len(context_ids),
                    )
                    append_result(args.out, record)
                    rows.append(record.model_dump())
                    print(
                        f"  L={ln:5d} d={d:.2f} needle={needle_idx} {scheme.scheme:20s} "
                        f"→ {'✓' if correct else '✗'}  (NLL c/d = {nll_c:.3f}/{nll_d:.3f})  {wall:4.1f}s"
                    )

    print(f"\nWriting heatmaps to {args.figures_dir}")
    _plot_heatmaps(rows, scheme_names, lengths, depths, args.figures_dir)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
