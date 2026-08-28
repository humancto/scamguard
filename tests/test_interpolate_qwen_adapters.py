"""LoRA interpolation produces one reproducible, fail-closed adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.interpolate_qwen_adapters import interpolate


def adapter(path: Path, value: float, *, rank: int = 2) -> Path:
    path.mkdir()
    config = {
        "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "r": rank,
        "target_modules": ["v_proj", "q_proj"],
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {"layer.lora_A.weight": torch.full((rank, 3), value, dtype=torch.bfloat16)},
        path / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )
    return path


def test_interpolation_is_single_adapter_with_hash_bound_parents(tmp_path: Path) -> None:
    left = adapter(tmp_path / "left", 2.0)
    right = adapter(tmp_path / "right", 6.0)
    # Target-module ordering is semantically irrelevant.
    right_config = json.loads((right / "adapter_config.json").read_text())
    right_config["target_modules"].reverse()
    (right / "adapter_config.json").write_text(json.dumps(right_config), encoding="utf-8")

    manifest = interpolate(left, right, tmp_path / "output", right_weight=0.25)
    tensors = load_file(tmp_path / "output" / "adapter_model.safetensors")

    assert torch.all(tensors["layer.lora_A.weight"] == 3.0)
    assert manifest["runtime_adapter_count"] == 1
    assert manifest["regression_splits_used_for_weight_selection"] == 0
    assert manifest["sealed_primary_test_v8_opened"] is False
    assert manifest["quantization_authorized"] is False


def test_interpolation_rejects_overwrite_weight_and_contract_drift(tmp_path: Path) -> None:
    left = adapter(tmp_path / "left", 2.0)
    right = adapter(tmp_path / "right", 6.0)

    with pytest.raises(ValueError, match="right_weight"):
        interpolate(left, right, tmp_path / "bad-weight", right_weight=1.1)

    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        interpolate(left, right, output, right_weight=0.5)

    drifted = adapter(tmp_path / "drifted", 4.0, rank=3)
    with pytest.raises(ValueError, match="configurations"):
        interpolate(left, drifted, tmp_path / "bad-contract", right_weight=0.5)
