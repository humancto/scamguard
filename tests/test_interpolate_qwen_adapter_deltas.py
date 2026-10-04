"""Rank-doubled LoRA concatenation preserves the exact weighted parent delta."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from scamguard.metrics import file_sha256
from scripts.interpolate_qwen_adapter_deltas import concatenate_deltas


def adapter(path: Path, a: torch.Tensor, b: torch.Tensor) -> Path:
    path.mkdir()
    config = {
        "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "r": 2,
        "lora_alpha": 4,
        "rank_pattern": {},
        "alpha_pattern": {},
        "target_modules": ["q_proj"],
        "use_rslora": False,
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {
            "layer.q_proj.lora_A.weight": a,
            "layer.q_proj.lora_B.weight": b,
        },
        path / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )
    return path


def frozen_config(left: Path, right: Path, output: Path, path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "method": "linear_lora_delta_concat_v1",
                "right_weight": 0.25,
                "left_weight": 0.75,
                "parent_rank": 2,
                "output_rank": 4,
                "checkpoint_output": str(output),
                "parents": {
                    side: {
                        "path": str(parent),
                        "adapter_sha256": file_sha256(
                            parent / "adapter_model.safetensors"
                        ),
                        "config_sha256": file_sha256(parent / "adapter_config.json"),
                    }
                    for side, parent in (("left", left), ("right", right))
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_rank_doubling_preserves_exact_weighted_delta(tmp_path: Path) -> None:
    left_a = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    left_b = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
    right_a = torch.tensor([[2.0, 1.0, 0.0], [1.0, 3.0, 2.0]])
    right_b = torch.tensor([[0.0, 2.0], [1.0, 3.0]])
    left = adapter(tmp_path / "left", left_a, left_b)
    right = adapter(tmp_path / "right", right_a, right_b)
    output = tmp_path / "output"
    config = frozen_config(left, right, output, tmp_path / "experiment.json")

    manifest = concatenate_deltas(
        left, right, output, right_weight=0.25, experiment_config=config
    )
    tensors = load_file(output / "adapter_model.safetensors")
    actual = tensors["layer.q_proj.lora_B.weight"] @ tensors[
        "layer.q_proj.lora_A.weight"
    ]
    expected = 0.75 * (left_b @ left_a) + 0.25 * (right_b @ right_a)

    assert torch.allclose(actual, expected)
    assert manifest["exact_parent_delta_mix"] is True
    assert manifest["parent_scaling"] == manifest["output_scaling"] == 2.0
    assert manifest["output_rank"] == 4
    assert json.loads((output / "adapter_config.json").read_text())["r"] == 4


def test_delta_interpolation_rejects_config_drift(tmp_path: Path) -> None:
    matrix_a = torch.ones((2, 3))
    matrix_b = torch.ones((2, 2))
    left = adapter(tmp_path / "left", matrix_a, matrix_b)
    right = adapter(tmp_path / "right", matrix_a, matrix_b)
    output = tmp_path / "output"
    config = frozen_config(left, right, output, tmp_path / "experiment.json")
    payload = json.loads(config.read_text())
    payload["right_weight"] = 0.5
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="experiment config"):
        concatenate_deltas(
            left, right, output, right_weight=0.25, experiment_config=config
        )
