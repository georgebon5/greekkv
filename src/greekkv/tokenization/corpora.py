"""Parallel + monolingual corpus loaders for Greek-vs-English KV-cost work.

Two public functions:

- `load_flores_pair(...)`  → parallel (src, tgt) sentence pairs from FLORES-200.
- `load_el_wikipedia_sample(n, seed)`  → `n` Greek Wikipedia article texts,
  deterministic given `seed`.

Both accept a `local_path` escape hatch that reads a tiny JSONL fixture
instead of hitting HuggingFace. Tests use this path so CI stays offline and
fast; real experiments leave it as `None` and let `datasets` download+cache.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Chosen for ungated access + widely-used mirrors. Verified against the HF Hub
# API at project T05 (see PR description). Change here if the hub paths move.
FLORES_DEFAULT = "Muennighoff/flores200"
WIKIPEDIA_DEFAULT = ("wikimedia/wikipedia", "20231101.el")

DEFAULT_CACHE = Path("data/hf_cache")


@dataclass(frozen=True)
class ParallelPair:
    src: str
    tgt: str


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pairs_from_jsonl(path: Path) -> list[ParallelPair]:
    rows = _read_jsonl(path)
    return [ParallelPair(src=r["src"], tgt=r["tgt"]) for r in rows]


def load_flores_pair(
    src_lang: str = "eng_Latn",
    tgt_lang: str = "ell_Grek",
    split: str = "devtest",
    cache_dir: str | Path = DEFAULT_CACHE,
    local_path: str | Path | None = None,
    limit: int | None = None,
) -> list[ParallelPair]:
    """Return aligned (src, tgt) sentences from FLORES-200.

    Parameters
    ----------
    src_lang, tgt_lang
        FLORES-200 language codes (e.g. ``eng_Latn``, ``ell_Grek``).
    split
        ``dev`` or ``devtest``. FLORES has no public test split.
    cache_dir
        Where ``datasets`` persists its cache.
    local_path
        If set, read pairs from this JSONL file instead of hitting HF. Each
        line must be ``{"src": "...", "tgt": "..."}``. Used by tests.
    limit
        Return at most this many pairs (applied last, deterministic order).
    """
    if local_path is not None:
        pairs = _pairs_from_jsonl(Path(local_path))
    else:
        pairs = _load_flores_from_hub(src_lang, tgt_lang, split, Path(cache_dir))
    return pairs[:limit] if limit is not None else pairs


def _load_flores_from_hub(
    src_lang: str, tgt_lang: str, split: str, cache_dir: Path
) -> list[ParallelPair]:
    # Import inside the function so importing greekkv.tokenization doesn't drag
    # in datasets' ~100 MB of transitive deps until we actually hit the hub.
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    ds_src = load_dataset(FLORES_DEFAULT, src_lang, split=split, cache_dir=str(cache_dir))
    ds_tgt = load_dataset(FLORES_DEFAULT, tgt_lang, split=split, cache_dir=str(cache_dir))
    texts_src = _column(ds_src, ("sentence", "text"))
    texts_tgt = _column(ds_tgt, ("sentence", "text"))
    if len(texts_src) != len(texts_tgt):
        raise RuntimeError(
            f"FLORES misaligned: {src_lang}={len(texts_src)} vs {tgt_lang}={len(texts_tgt)}"
        )
    return [ParallelPair(src=s, tgt=t) for s, t in zip(texts_src, texts_tgt, strict=True)]


def _column(ds, candidates: Iterable[str]) -> list[str]:
    for name in candidates:
        if name in ds.column_names:
            return list(ds[name])
    raise KeyError(f"None of {list(candidates)} found in columns {ds.column_names}")


def load_el_wikipedia_sample(
    n: int,
    seed: int = 42,
    cache_dir: str | Path = DEFAULT_CACHE,
    local_path: str | Path | None = None,
) -> list[str]:
    """Return ``n`` Greek Wikipedia article texts, deterministic given ``seed``.

    - If ``local_path`` is set, reads JSONL rows ``{"text": "..."}``.
    - Otherwise streams ``wikimedia/wikipedia`` ``20231101.el`` and takes the
      first ``n * 10`` articles, then samples ``n`` of them with ``seed``.
      Streaming keeps memory bounded on free-tier machines.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if local_path is not None:
        rows = _read_jsonl(Path(local_path))
        texts = [r["text"] for r in rows]
    else:
        texts = _load_el_wikipedia_from_hub(n_pool=max(n * 10, 100), cache_dir=Path(cache_dir))

    if len(texts) < n:
        raise RuntimeError(f"Requested {n} articles but only {len(texts)} available")

    rng = random.Random(seed)
    return rng.sample(texts, n)


def _load_el_wikipedia_from_hub(n_pool: int, cache_dir: Path) -> list[str]:
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    name, config = WIKIPEDIA_DEFAULT
    stream = load_dataset(name, config, split="train", streaming=True)
    texts: list[str] = []
    for i, row in enumerate(stream):
        if i >= n_pool:
            break
        texts.append(row["text"])
    return texts
