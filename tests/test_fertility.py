import math

import numpy as np
import pytest

from greekkv.tokenization import (
    FertilityStats,
    bootstrap_mean_ci,
    fertility_report,
    tokens_per_char,
    tokens_per_word,
)


class _WhitespaceTokenizer:
    """Simple tokenizer: each whitespace-separated word → 1 token."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(text.split())


class _CharTokenizer:
    """Each non-whitespace char → 1 token (worst-case fertility)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0 for c in text if not c.isspace()]


# ---------------------------------------------------------------------------
# Per-text metrics
# ---------------------------------------------------------------------------


def test_tokens_per_word_whitespace_tokenizer_is_one():
    tok = _WhitespaceTokenizer()
    assert tokens_per_word("hello world foo", tok) == 1.0
    assert tokens_per_word("Η Αθήνα είναι πρωτεύουσα", tok) == 1.0


def test_tokens_per_word_char_tokenizer_matches_avg_word_length():
    tok = _CharTokenizer()
    text = "abc de"  # 2 words, 5 non-space chars, so TPW = 5/2
    assert tokens_per_word(text, tok) == pytest.approx(2.5)


def test_tokens_per_char_whitespace_tokenizer():
    tok = _WhitespaceTokenizer()
    # "abc de" -> 2 tokens, 5 non-space chars -> 2/5
    assert tokens_per_char("abc de", tok) == pytest.approx(0.4)


def test_tokens_per_char_char_tokenizer_is_one():
    tok = _CharTokenizer()
    assert tokens_per_char("Αθήνα", tok) == 1.0


def test_empty_text_returns_nan():
    tok = _WhitespaceTokenizer()
    assert math.isnan(tokens_per_word("", tok))
    assert math.isnan(tokens_per_char("", tok))


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def test_bootstrap_is_deterministic_given_seed():
    rng = np.random.default_rng(0)
    data = rng.normal(size=200).tolist()
    a = bootstrap_mean_ci(data, n_boot=500, seed=7)
    b = bootstrap_mean_ci(data, n_boot=500, seed=7)
    assert a == b


def test_bootstrap_ci_covers_mean_of_constant_series():
    mean, lo, hi = bootstrap_mean_ci([3.0] * 50, n_boot=500, seed=1)
    assert mean == 3.0
    assert lo == 3.0 and hi == 3.0


def test_bootstrap_ci_tightens_with_more_samples():
    rng = np.random.default_rng(0)
    small = rng.normal(size=20).tolist()
    large = rng.normal(size=2000).tolist()
    _, lo_s, hi_s = bootstrap_mean_ci(small, n_boot=1000, seed=1)
    _, lo_l, hi_l = bootstrap_mean_ci(large, n_boot=1000, seed=1)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([], n_boot=100, seed=0)


def test_bootstrap_drops_nans():
    mean, _, _ = bootstrap_mean_ci([1.0, float("nan"), 3.0], n_boot=200, seed=0)
    assert mean == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# End-to-end report
# ---------------------------------------------------------------------------


def test_fertility_report_shape_and_bounds():
    texts = ["one two three", "foo bar", "a b c d"]
    stats = fertility_report(
        texts=texts,
        tokenizer=_CharTokenizer(),
        tokenizer_name="char",
        language="xx",
        n_boot=200,
        seed=0,
    )
    assert isinstance(stats, FertilityStats)
    assert stats.n_samples == 3
    assert stats.tokenizer == "char"
    assert stats.language == "xx"
    assert stats.ci_lo_tpw <= stats.mean_tpw <= stats.ci_hi_tpw
    assert stats.ci_lo_tpc <= stats.mean_tpc <= stats.ci_hi_tpc
    # Char tokenizer: TPC must be exactly 1.0 (non-space char -> 1 token).
    assert stats.mean_tpc == pytest.approx(1.0)
