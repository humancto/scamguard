#!/usr/bin/env python3
"""Convert the frozen ScamGuard encoder directly from PyTorch to a Core ML ML Program."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.modernbert import modeling_modernbert

from scamguard.metrics import file_sha256
from scamguard.preprocessing import DIALOGUE_POLICIES


class CoreMLLogitsOnly(torch.nn.Module):
    """Fixed-shape ModernBERT wrapper with export-friendly attention masks."""

    mask_bias_value = -10_000.0

    def __init__(self, model: torch.nn.Module, sequence_length: int) -> None:
        super().__init__()
        self.model = model
        positions = torch.arange(sequence_length, dtype=torch.int64).unsqueeze(0)
        query_positions = positions.unsqueeze(-1)
        key_positions = positions.unsqueeze(-2)
        local_allowed = (query_positions - key_positions).abs() <= model.config.sliding_window
        local_bias = torch.where(
            local_allowed.unsqueeze(1),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(self.mask_bias_value, dtype=torch.float32),
        )
        self.register_buffer("position_ids", positions)
        self.register_buffer("local_bias", local_bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Transformers builds these masks through higher-order Python functions whose
        # new_ones op is not convertible by coremltools. These tensor expressions are
        # equivalent for the fixed, cache-free encoder input used by ScamGuard.
        key_bias = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * self.mask_bias_value
        attention_masks = {
            "full_attention": key_bias,
            "sliding_attention": self.local_bias + key_bias,
        }
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            position_ids=self.position_ids,
        )
        hidden = outputs[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return self.model.classifier(self.model.drop(self.model.head(pooled)))


def check_wrapper_parity(
    model: torch.nn.Module,
    wrapper: torch.nn.Module,
    tokenizer: Any,
    sequence_length: int,
) -> float:
    probes = [
        "ScamGuard Core ML conversion verification message.",
        "Your bank account is safe. Do not share a verification code with anyone.",
        "URGENT: send the one-time code now or your package will be returned.",
        "Agent: I can reverse the charge. Customer: What information do you need?",
    ]
    max_error = 0.0
    with torch.inference_mode():
        for probe in probes:
            encoded = tokenizer(
                probe,
                return_tensors="pt",
                max_length=sequence_length,
                truncation=True,
                padding="max_length",
            )
            input_ids = encoded["input_ids"].to(torch.int32)
            attention_mask = encoded["attention_mask"].to(torch.int32)
            reference = model(input_ids=input_ids, attention_mask=attention_mask).logits
            candidate = wrapper(input_ids, attention_mask)
            max_error = max(max_error, float(torch.max(torch.abs(reference - candidate))))
    return max_error


def install_fixed_rotate_half(head_dimension: int) -> None:
    """Replace shape-derived rotary slicing with an equivalent fixed slice."""
    if head_dimension % 2:
        raise ValueError(f"rotary head dimension must be even: {head_dimension}")
    midpoint = head_dimension // 2

    def fixed_rotate_half(tensor: torch.Tensor) -> torch.Tensor:
        first = tensor[..., :midpoint]
        second = tensor[..., midpoint:]
        return torch.cat((-second, first), dim=-1)

    probe = torch.randn(1, 2, 3, head_dimension)
    original = modeling_modernbert.rotate_half(probe)
    candidate = fixed_rotate_half(probe)
    if not torch.equal(original, candidate):
        raise RuntimeError("fixed rotary rewrite is not exactly equivalent")
    modeling_modernbert.rotate_half = fixed_rotate_half


def directory_identity(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(block)
                digest.update(block)
    return {"files": len(files), "bytes": total, "tree_sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-coreml-logit-error", type=float, default=0.25)
    parser.add_argument(
        "--dialogue-policy",
        choices=sorted(DIALOGUE_POLICIES),
        default="speaker-neutral-v1",
    )
    args = parser.parse_args()

    required = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "scamguard_calibration.json",
    ]
    missing = [filename for filename in required if not (args.checkpoint / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing required files: {missing}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Core ML pack: {args.output_dir}")
    if args.sequence_length < 8:
        raise ValueError("sequence length must be at least 8")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        attn_implementation="eager",
    ).to("cpu")
    model.eval()
    wrapper = CoreMLLogitsOnly(model, args.sequence_length).eval()
    wrapper_max_error = check_wrapper_parity(model, wrapper, tokenizer, args.sequence_length)
    if wrapper_max_error > 1e-5:
        raise RuntimeError(f"fixed-mask wrapper parity failure: {wrapper_max_error}")
    encoded = tokenizer(
        "ScamGuard Core ML conversion verification message.",
        return_tensors="pt",
        max_length=args.sequence_length,
        truncation=True,
        padding="max_length",
    )
    example_inputs = (
        encoded["input_ids"].to(torch.int32),
        encoded["attention_mask"].to(torch.int32),
    )
    with torch.inference_mode():
        pre_rewrite_logits = wrapper(*example_inputs).detach().numpy()
        head_dimension = model.config.hidden_size // model.config.num_attention_heads
        install_fixed_rotate_half(head_dimension)
        reference_logits = wrapper(*example_inputs).detach().numpy()
        rewrite_max_error = float(np.max(np.abs(pre_rewrite_logits - reference_logits)))
        if rewrite_max_error > 1e-5:
            raise RuntimeError(f"fixed rotary rewrite parity failure: {rewrite_max_error}")
        traced = torch.jit.trace(wrapper, example_inputs, strict=True)
        traced_logits = traced(*example_inputs).detach().numpy()
    trace_max_error = float(np.max(np.abs(reference_logits - traced_logits)))
    if trace_max_error > 1e-5:
        raise RuntimeError(f"TorchScript trace parity failure: {trace_max_error}")

    compute_precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    coreml_model = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, args.sequence_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, args.sequence_length), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS17,
    )

    args.output_dir.mkdir(parents=True)
    package_name = f"scamguard-modernbert-seq{args.sequence_length}-{args.precision}.mlpackage"
    package = args.output_dir / package_name
    coreml_model.save(package)
    for filename in ("tokenizer.json", "tokenizer_config.json", "scamguard_calibration.json"):
        shutil.copy2(args.checkpoint / filename, args.output_dir / filename)

    prediction = coreml_model.predict(
        {
            "input_ids": example_inputs[0].numpy(),
            "attention_mask": example_inputs[1].numpy(),
        }
    )
    converted_logits = np.asarray(prediction["logits"])
    coreml_max_error = float(np.max(np.abs(reference_logits - converted_logits)))
    if not np.isfinite(converted_logits).all() or coreml_max_error > args.max_coreml_logit_error:
        raise RuntimeError(
            "Core ML parity failure: "
            f"max_abs_logit_error={coreml_max_error}, "
            f"limit={args.max_coreml_logit_error}, "
            f"finite={bool(np.isfinite(converted_logits).all())}"
        )
    manifest = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_files": {name: file_sha256(args.checkpoint / name) for name in required},
        "model": package.name,
        "model_identity": directory_identity(package),
        "sequence_length": args.sequence_length,
        "batch_size": 1,
        "precision": args.precision,
        "minimum_deployment_target": "iOS17",
        "input_transform": {"dialogue_policy": args.dialogue_policy},
        "export_rewrites": {
            "fixed_attention_masks": True,
            "attention_mask_bias": CoreMLLogitsOnly.mask_bias_value,
            "fixed_rotary_half_slice": True,
        },
        "input_contract": {
            "input_ids": {"shape": [1, args.sequence_length], "dtype": "int32"},
            "attention_mask": {"shape": [1, args.sequence_length], "dtype": "int32"},
        },
        "output_contract": {"logits": {"shape": [1, 3], "dtype": "float32"}},
        "parity_probe": {
            "fixed_mask_wrapper_max_abs_logit_error": wrapper_max_error,
            "fixed_rotary_rewrite_max_abs_logit_error": rewrite_max_error,
            "torchscript_max_abs_logit_error": trace_max_error,
            "coreml_max_abs_logit_error": coreml_max_error,
            "output_finite": bool(np.isfinite(converted_logits).all()),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "coremltools": ct.__version__,
            "coremltools_torch_compatibility": (
                "coremltools 9.0 reports torch 2.13.0 as newer than its tested torch 2.7.0"
            ),
        },
        "runtime_files": {
            filename: {
                "bytes": (args.output_dir / filename).stat().st_size,
                "sha256": file_sha256(args.output_dir / filename),
            }
            for filename in (
                "tokenizer.json",
                "tokenizer_config.json",
                "scamguard_calibration.json",
            )
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Core ML export failed: {error}", file=sys.stderr)
        raise
