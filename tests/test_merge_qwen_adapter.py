from __future__ import annotations

import hashlib
import json

import pytest

from scamguard.prompts import SYSTEM_PROMPT
from training.merge_qwen_adapter import load_release_calibration


def write_calibration(path, **overrides: object) -> None:
    values = {
        "base_model": "Qwen/example",
        "base_model_revision": "pinned",
        "labels": ["SAFE", "UNCERTAIN", "SCAM"],
        "safe_threshold_semantics": "minimum_safe_probability",
        "sequence_bucket_size": 64,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    } | overrides
    path.mkdir()
    (path / "scamguard_calibration.json").write_text(
        json.dumps(values), encoding="utf-8"
    )


def test_release_merge_requires_product_shaped_calibration(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    write_calibration(adapter)

    calibration, path = load_release_calibration(
        adapter, "Qwen/example", "pinned"
    )

    assert calibration["sequence_bucket_size"] == 64
    assert path.name == "scamguard_calibration.json"


def test_release_merge_rejects_dynamic_shape_calibration(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    write_calibration(adapter, sequence_bucket_size=0)

    with pytest.raises(ValueError, match="64-token"):
        load_release_calibration(adapter, "Qwen/example", "pinned")
