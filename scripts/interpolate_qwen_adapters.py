#!/usr/bin/env python3
"""Create one hash-bound LoRA adapter along a frozen parent-to-child trajectory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scamguard.metrics import file_sha256


def normalized_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(config.get("target_modules"), list):
        config["target_modules"] = sorted(config["target_modules"])
    return config


def tensor_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def interpolate(
    left_directory: Path,
    right_directory: Path,
    output: Path,
    *,
    right_weight: float,
    selection_split: str = "dev_only",
    regression_splits_used_for_weight_selection: int = 0,
    selection_note: str | None = None,
    experiment_config: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite adapter: {output}")
    if not 0.0 <= right_weight <= 1.0:
        raise ValueError("right_weight must be in [0, 1]")
    if not selection_split.strip():
        raise ValueError("selection_split must be non-empty")
    if regression_splits_used_for_weight_selection < 0:
        raise ValueError("regression split count must be non-negative")
    left_weights = left_directory / "adapter_model.safetensors"
    right_weights = right_directory / "adapter_model.safetensors"
    left_config = left_directory / "adapter_config.json"
    right_config = right_directory / "adapter_config.json"
    for path in (left_weights, right_weights, left_config, right_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if normalized_config(left_config) != normalized_config(right_config):
        raise ValueError("adapter configurations are not semantically identical")

    config_binding = None
    if experiment_config is not None:
        config = json.loads(experiment_config.read_text(encoding="utf-8"))
        expected = {
            "method": "linear_lora_weight_space_v1",
            "right_weight": right_weight,
            "left_weight": 1.0 - right_weight,
            "checkpoint_output": str(output),
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("experiment config differs from requested interpolation")
        for side, directory, weights, adapter_config in (
            ("left", left_directory, left_weights, left_config),
            ("right", right_directory, right_weights, right_config),
        ):
            parent = config.get("parents", {}).get(side, {})
            required_parent = {
                "path": str(directory),
                "adapter_sha256": file_sha256(weights),
                "config_sha256": file_sha256(adapter_config),
            }
            if any(parent.get(key) != value for key, value in required_parent.items()):
                raise ValueError(f"experiment config {side} parent binding differs")
        configured_splits = config.get("selection", {}).get(
            "prior_open_validation_splits_used", []
        )
        if len(configured_splits) != regression_splits_used_for_weight_selection:
            raise ValueError("experiment config selection split count differs")
        config_binding = {
            "path": str(experiment_config),
            "sha256": file_sha256(experiment_config),
        }

    left = load_file(left_weights, device="cpu")
    right = load_file(right_weights, device="cpu")
    if set(left) != set(right):
        raise ValueError("adapter tensor keys differ")
    result = {}
    for key in sorted(left):
        if left[key].shape != right[key].shape or left[key].dtype != right[key].dtype:
            raise ValueError(f"adapter tensor contract differs: {key}")
        blended = torch.lerp(
            left[key].to(dtype=torch.float32),
            right[key].to(dtype=torch.float32),
            right_weight,
        )
        result[key] = blended.to(dtype=left[key].dtype).contiguous()

    output.mkdir(parents=True)
    output_weights = output / "adapter_model.safetensors"
    metadata = tensor_metadata(left_weights) | {
        "scamguard_interpolation": "linear_lora_weight_space_v1",
        "scamguard_right_weight": format(right_weight, ".17g"),
    }
    save_file(result, output_weights, metadata=metadata)
    shutil.copy2(left_config, output / "adapter_config.json")
    manifest = {
        "schema_version": 1,
        "method": "linear_lora_weight_space_v1",
        "right_weight": right_weight,
        "left_weight": 1.0 - right_weight,
        "parents": {
            "left": {
                "directory": str(left_directory),
                "adapter_sha256": file_sha256(left_weights),
                "config_sha256": file_sha256(left_config),
            },
            "right": {
                "directory": str(right_directory),
                "adapter_sha256": file_sha256(right_weights),
                "config_sha256": file_sha256(right_config),
            },
        },
        "output": {
            "adapter_sha256": file_sha256(output_weights),
            "config_sha256": file_sha256(output / "adapter_config.json"),
            "tensor_count": len(result),
        },
        "selection_split": selection_split,
        "regression_splits_used_for_weight_selection": (
            regression_splits_used_for_weight_selection
        ),
        "selection_note": selection_note,
        "experiment_config": config_binding,
        "sealed_primary_test_v8_opened": False,
        "runtime_adapter_count": 1,
        "quantization_authorized": False,
        "publication_authorized": False,
    }
    manifest_path = output / "interpolation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-split", default="dev_only")
    parser.add_argument("--regression-splits-used-for-weight-selection", type=int, default=0)
    parser.add_argument("--selection-note")
    parser.add_argument("--experiment-config", type=Path)
    args = parser.parse_args()
    interpolate(
        args.left,
        args.right,
        args.output,
        right_weight=args.right_weight,
        selection_split=args.selection_split,
        regression_splits_used_for_weight_selection=(
            args.regression_splits_used_for_weight_selection
        ),
        selection_note=args.selection_note,
        experiment_config=args.experiment_config,
    )


if __name__ == "__main__":
    main()
