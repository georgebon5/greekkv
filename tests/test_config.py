from pathlib import Path

import pytest
from pydantic import ValidationError

from greekkv.utils import ExperimentConfig, config_hash, load_config

EXAMPLE = Path(__file__).parent.parent / "configs" / "example.yaml"


def test_load_example_config():
    cfg = load_config(EXAMPLE)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "krikri_fp16_baseline"
    assert cfg.seed == 42
    assert cfg.model.name == "ilsp/Llama-Krikri-8B-Instruct"
    assert cfg.quant.scheme == "fp16"
    assert cfg.data.language == "el"


def test_unknown_field_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\nmodel: {name: foo, mystery_field: 1}\nquant: {scheme: fp16}\ndata: {name: y}\n"
    )
    with pytest.raises(ValidationError):
        load_config(bad)


def test_invalid_scheme_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\n"
        "model: {name: foo}\n"
        "quant: {scheme: int2}\n"  # not in Literal
        "data: {name: y}\n"
    )
    with pytest.raises(ValidationError):
        load_config(bad)


def test_config_hash_is_deterministic():
    cfg1 = load_config(EXAMPLE)
    cfg2 = load_config(EXAMPLE)
    assert config_hash(cfg1) == config_hash(cfg2)
    assert len(config_hash(cfg1)) == 12


def test_config_hash_changes_with_payload(tmp_path: Path):
    cfg1 = load_config(EXAMPLE)
    alt = tmp_path / "alt.yaml"
    alt.write_text(EXAMPLE.read_text().replace("seed: 42", "seed: 43"))
    cfg2 = load_config(alt)
    assert config_hash(cfg1) != config_hash(cfg2)
