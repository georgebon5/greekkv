"""Measured-vs-estimated resolver (T28.3).

Reads the committed JSONL artefacts from T17 (perplexity sweep) and
T23 (systems profile) and offers a single ``load_measurements()``
helper that returns the measured ``(perplexity, decode_ms_per_token)``
for a ``(model, scheme, n_ctx)`` key — or ``None`` when no such row
exists, so the capacity CLI can fall back to the pure estimate without
silently making one up.

Resolution rule: exact match on ``(model, scheme, n_ctx)``. We do not
interpolate between measured ``n_ctx`` values; a value that was not
measured is ``None``, period. Interpolation would hide gaps in the
measurement grid, which is the opposite of what the ``source`` tag on
``CapacityResult`` is supposed to communicate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The canonical committed artefacts. Kept in sync with
# ``scripts/build_report.py`` so a `make reproduce` and a
# ``greekkv-capacity --measured`` run use the same source of truth.
DEFAULT_PERPLEXITY_JSONL = Path("results/perplexity/qwen2_5_0_5b.jsonl")
DEFAULT_PROFILE_JSONL = Path("results/profile/qwen2_5_0_5b.jsonl")


@dataclass(frozen=True)
class Measurement:
    """Per-``(model, scheme, n_ctx)`` measured block.

    Fields hold ``None`` when the corresponding artefact lacked the
    row; this keeps the CLI honest about partial coverage.
    """

    model: str
    scheme: str
    n_ctx: int
    perplexity: float | None = None
    decode_ms_per_token: float | None = None
    peak_mem_mb: float | None = None

    def has_any(self) -> bool:
        return (
            self.perplexity is not None
            or self.decode_ms_per_token is not None
            or self.peak_mem_mb is not None
        )


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def load_measurements(
    perplexity_jsonl: Path | None = None,
    profile_jsonl: Path | None = None,
) -> dict[tuple[str, str, int], Measurement]:
    """Build a ``(model, scheme, n_ctx) → Measurement`` index.

    Missing artefact files are not errors: they simply mean the
    corresponding column stays ``None`` across the board. This matches
    the ADR-0001 CI story (reruns may land on different hosts).
    """
    ppl_path = perplexity_jsonl or DEFAULT_PERPLEXITY_JSONL
    prof_path = profile_jsonl or DEFAULT_PROFILE_JSONL

    # Collect per-key partials from each file, then zip them together.
    perplexity: dict[tuple[str, str, int], float] = {}
    if ppl_path.exists():
        for r in _iter_jsonl(ppl_path):
            m = r.get("metrics", {})
            n_ctx = m.get("n_ctx")
            if n_ctx is None:
                continue
            key = (r["model"], r["scheme"], int(n_ctx))
            if "perplexity" in m:
                perplexity[key] = float(m["perplexity"])

    profile: dict[tuple[str, str, int], tuple[float, float]] = {}
    if prof_path.exists():
        for r in _iter_jsonl(prof_path):
            m = r.get("metrics", {})
            n_ctx = m.get("n_ctx")
            if n_ctx is None:
                continue
            key = (r["model"], r["scheme"], int(n_ctx))
            decode_ms = m.get("decode_ms_per_token")
            peak_mem = m.get("peak_mem_mb")
            profile[key] = (
                float(decode_ms) if decode_ms is not None else float("nan"),
                float(peak_mem) if peak_mem is not None else float("nan"),
            )

    out: dict[tuple[str, str, int], Measurement] = {}
    for key in set(perplexity) | set(profile):
        model, scheme, n_ctx = key
        prof = profile.get(key)
        decode_ms, peak_mem = prof or (None, None)
        out[key] = Measurement(
            model=model,
            scheme=scheme,
            n_ctx=n_ctx,
            perplexity=perplexity.get(key),
            decode_ms_per_token=_nan_to_none(decode_ms),
            peak_mem_mb=_nan_to_none(peak_mem),
        )
    return out


def _nan_to_none(x: float | None) -> float | None:
    if x is None:
        return None
    return None if x != x else x  # NaN != NaN


def pick(
    index: dict[tuple[str, str, int], Measurement],
    model: str,
    scheme: str,
    n_ctx: int,
) -> Measurement | None:
    """Exact-match lookup. ``None`` if no row was measured for this key."""
    return index.get((model, scheme, int(n_ctx)))
