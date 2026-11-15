"""Tokenizer fertility measurement.

Fertility = how many sub-word tokens a tokenizer uses to encode one linguistic
"unit" of input. Two units here:

- **tokens-per-word** (TPW): whitespace-split words. Natural, but sensitive to
  punctuation attachment ("Αθήνα," counts as one word).
- **tokens-per-char** (TPC): non-whitespace characters. Punctuation-robust and
  comparable across languages with different word lengths.

Both are reported with bootstrap 95% confidence intervals so comparisons
across tokenizers / languages carry a sample-size signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Tokenizer(Protocol):
    """Minimal subset of the HF tokenizer surface this module uses."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class FertilityStats:
    tokenizer: str
    language: str
    n_samples: int
    seed: int
    n_boot: int
    mean_tpw: float
    ci_lo_tpw: float
    ci_hi_tpw: float
    mean_tpc: float
    ci_lo_tpc: float
    ci_hi_tpc: float


def _word_count(text: str) -> int:
    return len(text.split())


def _nonspace_char_count(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def tokens_per_word(text: str, tokenizer: Tokenizer) -> float:
    words = _word_count(text)
    if words == 0:
        return float("nan")
    n_tok = len(tokenizer.encode(text, add_special_tokens=False))
    return n_tok / words


def tokens_per_char(text: str, tokenizer: Tokenizer) -> float:
    chars = _nonspace_char_count(text)
    if chars == 0:
        return float("nan")
    n_tok = len(tokenizer.encode(text, add_special_tokens=False))
    return n_tok / chars


def bootstrap_mean_ci(
    values: list[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) where lo/hi are percentile-bootstrap bounds at level ``ci``.

    Drops NaNs before resampling. Raises ``ValueError`` on empty input.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("no finite values to bootstrap")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return float(arr.mean()), float(lo), float(hi)


def fertility_report(
    texts: list[str],
    tokenizer: Tokenizer,
    tokenizer_name: str,
    language: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> FertilityStats:
    tpw = [tokens_per_word(t, tokenizer) for t in texts]
    tpc = [tokens_per_char(t, tokenizer) for t in texts]
    m_w, lo_w, hi_w = bootstrap_mean_ci(tpw, n_boot=n_boot, seed=seed)
    m_c, lo_c, hi_c = bootstrap_mean_ci(tpc, n_boot=n_boot, seed=seed)
    return FertilityStats(
        tokenizer=tokenizer_name,
        language=language,
        n_samples=len(texts),
        seed=seed,
        n_boot=n_boot,
        mean_tpw=m_w,
        ci_lo_tpw=lo_w,
        ci_hi_tpw=hi_w,
        mean_tpc=m_c,
        ci_lo_tpc=lo_c,
        ci_hi_tpc=hi_c,
    )
