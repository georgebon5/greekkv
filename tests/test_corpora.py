from pathlib import Path

import pytest

from greekkv.tokenization import (
    ParallelPair,
    load_el_wikipedia_sample,
    load_flores_pair,
)

FIX = Path(__file__).parent / "fixtures" / "corpora"
FLORES = FIX / "flores_mini.jsonl"
WIKI = FIX / "el_wiki_mini.jsonl"


# ---------------------------------------------------------------------------
# FLORES
# ---------------------------------------------------------------------------


def test_flores_local_shape():
    pairs = load_flores_pair(local_path=FLORES)
    assert len(pairs) == 5
    assert all(isinstance(p, ParallelPair) for p in pairs)
    assert all(p.src and p.tgt for p in pairs)


def test_flores_greek_text_decodes_correctly():
    pairs = load_flores_pair(local_path=FLORES)
    # "Η" (capital eta) is U+0397 — must survive round-trip without mojibake.
    assert pairs[1].tgt.startswith("Η Αθήνα")
    # Every target string must contain at least one Greek letter.
    greek_ranges = [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]
    for p in pairs:
        assert any(any(lo <= ord(c) <= hi for lo, hi in greek_ranges) for c in p.tgt), (
            f"no Greek letter in: {p.tgt!r}"
        )


def test_flores_limit_is_deterministic():
    full = load_flores_pair(local_path=FLORES)
    first3 = load_flores_pair(local_path=FLORES, limit=3)
    assert first3 == full[:3]


# ---------------------------------------------------------------------------
# Greek Wikipedia
# ---------------------------------------------------------------------------


def test_wiki_sample_shape_and_type():
    texts = load_el_wikipedia_sample(5, seed=0, local_path=WIKI)
    assert len(texts) == 5
    assert all(isinstance(t, str) and t for t in texts)


def test_wiki_sample_is_deterministic_given_seed():
    a = load_el_wikipedia_sample(5, seed=123, local_path=WIKI)
    b = load_el_wikipedia_sample(5, seed=123, local_path=WIKI)
    assert a == b


def test_wiki_different_seeds_give_different_samples():
    a = load_el_wikipedia_sample(5, seed=1, local_path=WIKI)
    b = load_el_wikipedia_sample(5, seed=2, local_path=WIKI)
    # 12 items, picking 5 → astronomically unlikely to match under different seeds
    assert a != b


def test_wiki_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        load_el_wikipedia_sample(0, local_path=WIKI)


def test_wiki_raises_when_pool_too_small():
    with pytest.raises(RuntimeError):
        load_el_wikipedia_sample(100, local_path=WIKI)  # fixture has 12
