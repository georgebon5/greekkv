import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from greekkv.utils import ResultRecord, append_result, git_sha


def _make_record(**over):
    base = dict(
        experiment="exp1",
        model="ilsp/Meltemi-7B-Instruct-v1.5",
        scheme="int8",
        git_sha="abc1234",
        config_hash="deadbeef1234",
        metrics={"ppl": 12.3, "kl_mean": 0.07},
        tokens=1024,
    )
    base.update(over)
    return ResultRecord(**base)


def test_record_round_trip(tmp_path: Path):
    r = _make_record()
    out = tmp_path / "results" / "demo.jsonl"
    append_result(out, r)

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["experiment"] == "exp1"
    assert row["metrics"]["ppl"] == 12.3
    assert row["tokens"] == 1024
    assert row["timestamp"].endswith("Z")


def test_append_result_appends_not_overwrites(tmp_path: Path):
    out = tmp_path / "results.jsonl"
    append_result(out, _make_record())
    append_result(out, _make_record(scheme="int4"))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["scheme"] for r in rows] == ["int8", "int4"]


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        ResultRecord(
            experiment="e",
            model="m",
            scheme="fp16",
            git_sha="abc",
            config_hash="abc",
            typo_field=1,  # extra='forbid'
        )


def test_git_sha_returns_something():
    sha = git_sha()
    assert isinstance(sha, str) and sha
