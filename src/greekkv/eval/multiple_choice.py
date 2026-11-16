"""Multiple-choice scoring via length-normalised NLL.

For a question and a list of candidate answers, score each candidate using
the T16 context-then-continuation scorer, then pick the candidate with the
lowest **per-token** NLL. Length normalisation is important: without it the
scorer would systematically prefer shorter answers, which has nothing to do
with the model's actual preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from greekkv.eval.scorer import score_continuation
from greekkv.quant import CacheSpec


@dataclass(frozen=True)
class MCPrediction:
    pred_index: int
    nll_per_choice: list[float]  # length-normalised


def predict_mc(
    model: Any,
    tokenizer: Any,
    question: str,
    choices: list[str],
    cache_spec: CacheSpec,
    answer_prefix: str = "\nΑπάντηση: ",
) -> MCPrediction:
    """Return the predicted index and the per-choice length-normalised NLLs.

    The default ``answer_prefix`` is Greek ("Answer: ") because the project
    evaluates Greek benchmarks. Callers can override for other prompt styles.
    """
    if not choices:
        raise ValueError("choices must be non-empty")

    nlls: list[float] = []
    for choice in choices:
        continuation = answer_prefix + choice
        res = score_continuation(model, tokenizer, question, continuation, cache_spec)
        nlls.append(res.nll_mean)

    best = min(range(len(nlls)), key=lambda i: nlls[i])
    return MCPrediction(pred_index=best, nll_per_choice=nlls)
