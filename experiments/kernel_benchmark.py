"""T25 — Benchmark three decode-attention implementations across ``N_ctx``.

Three candidates per cell:

1. **FP16 baseline** — standard single-query attention with FP16 K/V
   tensors. No quantization. This is the ceiling on today's GPUs.
2. **Unfused reference** — the pure-PyTorch :func:`decode_attention_reference`
   from T24. On every call it dequantizes K and V from int8 scales, then runs
   the ``Q@K^T`` + softmax + ``@V`` pipeline with torch ops. This is what
   `FakeQuantCache` effectively pays per step (24 layers worth).
3. **Triton kernel** — :func:`decode_attention_triton` from T24. One fused
   GPU launch per head. Runs only on CUDA; skipped here on CPU hosts and
   flagged in the output JSONL.

The acceptance criterion for this task is honest reporting — including
situations where the unfused reference on CPU is *slower* than FP16
baseline on CPU (a predictable outcome given that fake-quant adds dequant
work for zero storage benefit at CPU inference time).

Usage
-----
::

    python experiments/kernel_benchmark.py \\
        --n-ctx 256 1024 4096 \\
        --n-iters 30 \\
        --out results/kernel_bench/qwen2_5_0_5b_shape.jsonl \\
        --figures-dir results/figures/kernel_bench/
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from greekkv.systems.attention import (
    decode_attention_reference,
    decode_attention_triton,
)
from greekkv.utils import ResultRecord, append_result, git_sha, seed_all

IMPL_COLOR = {
    "fp16_baseline": "#2E7D32",
    "reference_unfused": "#D32F2F",
    "triton_fused": "#1976D2",
}


def _fp16_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Standard GQA single-query attention in FP16 — the no-quant ceiling."""
    h, d = q.shape
    h_kv = k.shape[0]
    kv_ratio = h // h_kv
    out = torch.empty_like(q)
    for hi in range(h):
        hk = hi // kv_ratio
        scores = (q[hi] @ k[hk].T) * scale
        w = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
        out[hi] = w @ v[hk]
    return out


def _build_shapes(h: int, h_kv: int, n: int, d: int, group_size: int, seed: int):
    """Return (q_fp16, k_fp16, v_fp16, k_data_int8, k_scales, v_data_int8, v_scales)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn((h, d), generator=g, dtype=torch.float32)
    k = torch.randn((h_kv, n, d), generator=g, dtype=torch.float32)
    v = torch.randn((h_kv, n, d), generator=g, dtype=torch.float32)

    # K per-channel quant
    k_max = k.abs().reshape(h_kv, n, d // group_size, group_size).amax(dim=(1, 3))
    k_scales = (k_max / 127.0).clamp_min(1e-6)
    k_scales_e = k_scales.repeat_interleave(group_size, dim=-1)
    k_data = (k / k_scales_e.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)

    # V per-token quant
    v_max = v.abs().reshape(h_kv, n // group_size, group_size, d).amax(dim=(2, 3))
    v_scales = (v_max / 127.0).clamp_min(1e-6)
    v_scales_e = v_scales.repeat_interleave(group_size, dim=-1)
    v_data = (v / v_scales_e.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)

    return (
        q.to(torch.float16),
        k.to(torch.float16),
        v.to(torch.float16),
        k_data,
        k_scales,
        v_data,
        v_scales,
    )


def _time_callable(fn, n_iters: int, warmup: int = 3) -> tuple[float, float]:
    """Return ``(mean_ms, p95_ms)`` over ``n_iters`` runs after a few warmups."""
    for _ in range(warmup):
        fn()
    times_ms: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    mean = statistics.mean(times_ms)
    p95 = times_ms[min(len(times_ms) - 1, int(0.95 * len(times_ms)))]
    return mean, p95


def _plot(rows: list[dict], out_dir: Path, shape_label: str) -> None:
    # Group by impl → list of (N, mean_ms, p95_ms)
    by_impl: dict[str, list[tuple[int, float, float]]] = {}
    for r in rows:
        m = r["metrics"]
        by_impl.setdefault(r["scheme"], []).append(
            (int(m["N"]), float(m["wall_ms_mean"]), float(m["wall_ms_p95"]))
        )
    for impl in by_impl:
        by_impl[impl].sort()

    if not by_impl:
        # Nothing to plot (e.g. every N was skipped). Stay silent and return.
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for impl, data in by_impl.items():
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        ax.plot(xs, ys, marker="o", linewidth=2, label=impl, color=IMPL_COLOR.get(impl, "black"))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted({d[0] for impl in by_impl for d in by_impl[impl]}))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlabel("N_ctx (sequence length, log₂)")
    ax.set_ylabel("Wall time (ms, log)")
    ax.set_title(f"Decode-attention kernel benchmark — {shape_label}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "kernel_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-ctx", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--H", dest="h", type=int, default=8)
    ap.add_argument("--H-kv", dest="h_kv", type=int, default=2)
    ap.add_argument("--D", dest="d", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--n-iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/kernel_bench/qwen2_5_0_5b_shape.jsonl"),
    )
    ap.add_argument("--figures-dir", type=Path, default=Path("results/figures/kernel_bench"))
    args = ap.parse_args()

    seed_all(args.seed)

    shape_label = f"H={args.h}, H_kv={args.h_kv}, D={args.d}, G={args.group_size}"
    print(f"Kernel benchmark — shape: {shape_label}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            import triton  # noqa: F401

            triton_available = True
        except ImportError:
            triton_available = False
    else:
        triton_available = False

    if not (cuda_available and triton_available):
        print("  (CUDA + Triton not available — Triton benchmark will be skipped)")

    all_rows: list[dict] = []
    for n in args.n_ctx:
        if n % args.group_size != 0:
            print(f"SKIP N={n}: not divisible by group_size={args.group_size}")
            continue
        print(f"\nN={n}:")
        q, k, v, k_data, k_scales, v_data, v_scales = _build_shapes(
            args.h, args.h_kv, n, args.d, args.group_size, seed=args.seed
        )
        scale = 1.0 / math.sqrt(args.d)

        candidates: list[tuple[str, Callable[[], torch.Tensor], str]] = [
            (
                "fp16_baseline",
                partial(_fp16_attention, q, k, v, scale),
                "cpu",
            ),
            (
                "reference_unfused",
                partial(
                    decode_attention_reference,
                    q.to(torch.float32),
                    k_data,
                    k_scales,
                    v_data,
                    v_scales,
                    args.group_size,
                ),
                "cpu",
            ),
        ]
        if cuda_available and triton_available:
            q_c = q.cuda()
            k_data_c = k_data.cuda()
            k_scales_c = k_scales.to(torch.float16).cuda()
            v_data_c = v_data.cuda()
            v_scales_c = v_scales.to(torch.float16).cuda()
            candidates.append(
                (
                    "triton_fused",
                    partial(
                        decode_attention_triton,
                        q_c,
                        k_data_c,
                        k_scales_c,
                        v_data_c,
                        v_scales_c,
                        args.group_size,
                    ),
                    "cuda",
                )
            )

        for name, fn, device in candidates:
            mean_ms, p95_ms = _time_callable(fn, n_iters=args.n_iters)
            record = ResultRecord(
                experiment="kernel_bench",
                model="synthetic",
                scheme=name,
                git_sha=git_sha(),
                config_hash=f"{name[:4]}_{n}",
                metrics={
                    "impl": name,
                    "N": n,
                    "H": args.h,
                    "H_kv": args.h_kv,
                    "D": args.d,
                    "group_size": args.group_size,
                    "wall_ms_mean": mean_ms,
                    "wall_ms_p95": p95_ms,
                    "n_iters": args.n_iters,
                    "device": device,
                },
                tokens=n,
            )
            append_result(args.out, record)
            all_rows.append(record.model_dump())
            print(f"  {name:20s}  mean={mean_ms:7.2f} ms  p95={p95_ms:7.2f} ms  device={device}")

    print(f"\nWrote {len(all_rows)} rows to {args.out}")
    _plot(all_rows, args.figures_dir, shape_label)
    print(f"Wrote figure to {args.figures_dir}/kernel_benchmark.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
