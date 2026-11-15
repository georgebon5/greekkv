from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Strict):
    name: str
    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    load_in_4bit: bool = False


class QuantConfig(_Strict):
    scheme: Literal["fp16", "int8", "int4"] = "fp16"
    group_size: int = Field(default=128, ge=1)
    residual: int = Field(default=0, ge=0, description="FP16 window of most-recent tokens")
    key_axis: Literal["per_channel", "per_token"] = "per_channel"
    value_axis: Literal["per_channel", "per_token"] = "per_token"
    symmetric: bool = True


class DataConfig(_Strict):
    name: str
    split: str = "test"
    n_samples: int = Field(default=100, ge=1)
    n_ctx: int = Field(default=2048, ge=16)
    language: Literal["el", "en", "mixed"] = "el"


class ExperimentConfig(_Strict):
    name: str
    seed: int = 42
    model: ModelConfig
    quant: QuantConfig
    data: DataConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open() as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig.model_validate(raw)


def config_hash(cfg: ExperimentConfig, length: int = 12) -> str:
    # Stable: model_dump_json is deterministic for frozen pydantic models.
    payload = cfg.model_dump_json().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
