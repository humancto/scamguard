#!/usr/bin/env python3
"""Create one exact rank-doubled LoRA adapter from two parent weight deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scamguard.metrics import file_sha256
from scripts.interpolate_qwen_adapters import normalized_config


def tensor_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def paired_key(key: str, source: str, target: str) -> str:
    marker = f".lora_{source}.weight"
    if not key.endswith(marker):
        raise ValueError(f"unexpected LoRA tensor key: {key}")
    return key[: -len(marker)] + f".lora_{target}.weight"


def concatenate_deltas(
    left_directory: Path,
    right_directory: Path,
    output: Path,
    *,
    right_weight: float,
    experiment_config: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite adapter: {output}")
    if not 0.0 <= right_weight <= 1.0:
        raise ValueError("right_weight must be in [0, 1]")

    left_weights_path = left_directory / "adapter_model.safetensors"
    right_weights_path = right_directory / "adapter_model.safetensors"
    left_config_path = left_directory / "adapter_config.json"
    right_config_path = right_directory / "adapter_config.json"
    for path in (
        left_weights_path,
        right_weights_path,
        left_config_path,
        right_config_path,
        experiment_config,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    left_config = normalized_config(left_config_path)
    right_config = normalized_config(right_config_path)
    if left_config != right_config:
        raise ValueError("adapter configurations are not semantically identical")
    if left_config.get("use_rslora") is not False:
        raise ValueError("delta concatenation requires standard, non-RSLoRA scaling")
    parent_rank = int(left_config["r"])
    parent_alpha = int(left_config["lora_alpha"])
    output_rank = parent_rank * 2
    output_alpha = parent_alpha * 2

    frozen = json.loads(experiment_config.read_text(encoding="utf-8"))
    expected = {
        "method": "linear_lora_delta_concat_v1",
        "right_weight": right_weight,
        "left_weight": 1.0 - right_weight,
        "parent_rank": parent_rank,
        "output_rank": output_rank,
        "checkpoint_output": str(output),
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise ValueError("experiment config differs from requested delta interpolation")
    for side, directory, weights, config_path in (
        ("left", left_directory, left_weights_path, left_config_path),
        ("right", right_directory, right_weights_path, right_config_path),
    ):
        parent = frozen.get("parents", {}).get(side, {})
        required = {
            "path": str(directory),
            "adapter_sha256": file_sha256(weights),
            "config_sha256": file_sha256(config_path),
        }
        if any(parent.get(key) != value for key, value in required.items()):
            raise ValueError(f"experiment config {side} parent binding differs")

    left = load_file(left_weights_path, device="cpu")
    right = load_file(right_weights_path, device="cpu")
    if set(left) != set(right):
        raise ValueError("adapter tensor keys differ")
    a_keys = sorted(key for key in left if key.endswith(".lora_A.weight"))
    if not a_keys or len(left) != 2 * len(a_keys):
        raise ValueError("adapter tensors are not complete LoRA A/B pairs")

    result: dict[str, torch.Tensor] = {}
    for a_key in a_keys:
        b_key = paired_key(a_key, "A", "B")
        if b_key not in left:
            raise ValueError(f"missing LoRA B tensor for {a_key}")
        left_a, right_a = left[a_key], right[a_key]
        left_b, right_b = left[b_key], right[b_key]
        if (
            left_a.shape != right_a.shape
            or left_b.shape != right_b.shape
            or left_a.dtype != right_a.dtype
            or left_b.dtype != right_b.dtype
            or left_a.shape[0] != parent_rank
            or left_b.shape[1] != parent_rank
        ):
            raise ValueError(f"adapter tensor contract differs: {a_key}")
        result[a_key] = torch.cat((left_a, right_a), dim=0).contiguous()
        result[b_key] = torch.cat(
            ((1.0 - right_weight) * left_b, right_weight * right_b), dim=1
        ).contiguous()

    output_config = dict(left_config)
    output_config["r"] = output_rank
    output_config["lora_alpha"] = output_alpha
    output_config["rank_pattern"] = {}
    output_config["alpha_pattern"] = {}

    output.mkdir(parents=True)
    output_weights = output / "adapter_model.safetensors"
    metadata = tensor_metadata(left_weights_path) | {
        "scamguard_interpolation": "linear_lora_delta_concat_v1",
        "scamguard_right_weight": format(right_weight, ".17g"),
        "scamguard_exact_parent_delta_mix": "true",
    }
    save_file(result, output_weights, metadata=metadata)
    output_config_path = output / "adapter_config.json"
    output_config_path.write_text(
        json.dumps(output_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "method": "linear_lora_delta_concat_v1",
        "right_weight": right_weight,
        "left_weight": 1.0 - right_weight,
        "parent_rank": parent_rank,
        "output_rank": output_rank,
        "parent_scaling": parent_alpha / parent_rank,
        "output_scaling": output_alpha / output_rank,
        "exact_parent_delta_mix": True,
        "parents": {
            "left": {
                "directory": str(left_directory),
                "adapter_sha256": file_sha256(left_weights_path),
                "config_sha256": file_sha256(left_config_path),
            },
            "right": {
                "directory": str(right_directory),
                "adapter_sha256": file_sha256(right_weights_path),
                "config_sha256": file_sha256(right_config_path),
            },
        },
        "output": {
            "adapter_sha256": file_sha256(output_weights),
            "config_sha256": file_sha256(output_config_path),
            "tensor_count": len(result),
        },
        "experiment_config": {
            "path": str(experiment_config),
            "sha256": file_sha256(experiment_config),
        },
        "selection_split": "dev_and_prior_open_validation",
        "regression_splits_used_for_weight_selection": 2,
        "fresh_confirmation": False,
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
    parser.add_argument("--experiment-config", type=Path, required=True)
    args = parser.parse_args()
    concatenate_deltas(
        args.left,
        args.right,
        args.output,
        right_weight=args.right_weight,
        experiment_config=args.experiment_config,
    )


if __name__ == "__main__":
    main()
