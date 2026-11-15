from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def git_sha(short: bool = True) -> str:
    """Return current git SHA, or "unknown" if git is unavailable or not a repo."""
    args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.decode().strip() or "unknown"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ResultRecord(BaseModel):
    """Single row in `results/*.jsonl`.

    `metrics` is intentionally free-form so each experiment can emit whatever
    scalar it wants (e.g. {"ppl": 12.3, "kl_mean": 0.07}) without schema churn.
    The structural fields below are the invariants every row must carry so we
    can always reproduce it.
    """

    model_config = ConfigDict(extra="forbid")

    experiment: str = Field(description="ExperimentConfig.name")
    model: str = Field(description="ModelConfig.name")
    scheme: str = Field(description="QuantConfig.scheme (fp16/int8/int4/...)")
    git_sha: str
    config_hash: str
    timestamp: str = Field(default_factory=_utcnow_iso)
    metrics: dict[str, Any] = Field(default_factory=dict)
    tokens: int | None = None


def append_result(path: str | Path, record: ResultRecord) -> None:
    """Append one JSON line to `path`, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
