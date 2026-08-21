#!/usr/bin/env python3
"""Export a frozen ScamGuard encoder to validated FP32 and dynamic-INT8 ONNX artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scamguard.metrics import file_sha256
from scamguard.preprocessing import DIALOGUE_POLICIES


class LogitsOnly(torch.nn.Module):
    """Make the exported graph contract explicit and independent of HF output containers."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")


def model_metadata(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    return {
        "inputs": [item.name for item in model.graph.input],
        "outputs": [item.name for item in model.graph.output],
        "opset": max((entry.version for entry in model.opset_import), default=None),
        "nodes": len(model.graph.node),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def verify_runtime(
    path: Path, sequence_length: int, *, dynamic_sequence: bool
) -> dict[str, Any]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    tested_lengths = [sequence_length]
    if dynamic_sequence:
        tested_lengths.append(max(8, sequence_length // 2))
    output_shapes = []
    output_finite = True
    for tested_length in tested_lengths:
        inputs = {
            "input_ids": torch.ones((1, tested_length), dtype=torch.int64).numpy(),
            "attention_mask": torch.ones((1, tested_length), dtype=torch.int64).numpy(),
        }
        outputs = session.run(["logits"], inputs)
        if len(outputs) != 1 or outputs[0].shape != (1, 3):
            raise RuntimeError(
                f"unexpected ONNX output contract: {[item.shape for item in outputs]}"
            )
        output_shapes.append(list(outputs[0].shape))
        output_finite = output_finite and bool(torch.isfinite(torch.from_numpy(outputs[0])).all())
    return {
        "providers": session.get_providers(),
        "tested_sequence_lengths": tested_lengths,
        "output_shapes": output_shapes,
        "output_finite": output_finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-sequence", action="store_true")
    parser.add_argument(
        "--dialogue-policy",
        choices=sorted(DIALOGUE_POLICIES),
        default="speaker-neutral-v1",
    )
    args = parser.parse_args()

    if args.sequence_length < 8:
        raise ValueError("sequence length must be at least 8")
    required = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "scamguard_calibration.json",
    ]
    missing = [name for name in required if not (args.checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing required files: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape_name = "dynamic" if args.dynamic_sequence else str(args.sequence_length)
    stem = f"scamguard-modernbert-seq{shape_name}"
    fp32_path = args.output_dir / f"{stem}-fp32.onnx"
    int8_path = args.output_dir / f"{stem}-int8.onnx"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    runtime_files = [
        args.output_dir / "tokenizer.json",
        args.output_dir / "tokenizer_config.json",
        args.output_dir / "scamguard_calibration.json",
    ]
    for path in (fp32_path, int8_path, manifest_path, *runtime_files):
        require_new_output(path)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        attn_implementation="eager",
    ).to("cpu")
    model.eval()
    if int(model.config.num_labels) != 3:
        raise ValueError(f"expected three labels, found {model.config.num_labels}")

    encoded = tokenizer(
        "ScamGuard static-shape export verification message.",
        return_tensors="pt",
        max_length=args.sequence_length,
        truncation=True,
        padding="max_length",
    )
    wrapper = LogitsOnly(model)
    with torch.inference_mode():
        reference_logits = wrapper(encoded["input_ids"], encoded["attention_mask"])

    torch.onnx.export(
        wrapper,
        (encoded["input_ids"], encoded["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamo=False,
        external_data=False,
        dynamic_axes=(
            {
                "input_ids": {1: "sequence_length"},
                "attention_mask": {1: "sequence_length"},
            }
            if args.dynamic_sequence
            else None
        ),
        do_constant_folding=True,
    )
    fp32 = model_metadata(fp32_path)
    fp32_runtime = verify_runtime(
        fp32_path,
        args.sequence_length,
        dynamic_sequence=args.dynamic_sequence,
    )

    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    fp32_logits = fp32_session.run(
        ["logits"],
        {
            "input_ids": encoded["input_ids"].numpy(),
            "attention_mask": encoded["attention_mask"].numpy(),
        },
    )[0]
    reference = reference_logits.detach().numpy()
    fp32_max_error = float(abs(reference - fp32_logits).max())
    if fp32_max_error > 1e-4:
        raise RuntimeError(f"FP32 export parity failure: maximum logit error {fp32_max_error}")

    quantize_dynamic(
        fp32_path,
        int8_path,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=True,
        weight_type=QuantType.QInt8,
        use_external_data_format=False,
    )
    int8 = model_metadata(int8_path)
    int8_runtime = verify_runtime(
        int8_path,
        args.sequence_length,
        dynamic_sequence=args.dynamic_sequence,
    )
    for filename in ("tokenizer.json", "tokenizer_config.json", "scamguard_calibration.json"):
        shutil.copy2(args.checkpoint / filename, args.output_dir / filename)

    manifest = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_files": {name: file_sha256(args.checkpoint / name) for name in required},
        "sequence_length": args.sequence_length,
        "batch_size": 1,
        "labels": ["SAFE", "UNCERTAIN", "SCAM"],
        "input_transform": {"dialogue_policy": args.dialogue_policy},
        "input_contract": {
            "input_ids": [1, "dynamic" if args.dynamic_sequence else args.sequence_length],
            "attention_mask": [1, "dynamic" if args.dynamic_sequence else args.sequence_length],
        },
        "output_contract": {"logits": [1, 3]},
        "export": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "opset": args.opset,
            "attention_implementation": "eager",
            "dynamic_sequence": args.dynamic_sequence,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "fp32": {**fp32, "runtime": fp32_runtime, "pytorch_max_abs_logit_error": fp32_max_error},
        "int8_dynamic": {
            **int8,
            "runtime": int8_runtime,
            "quantized_ops": ["MatMul", "Gemm"],
            "per_channel": True,
            "weight_type": "QInt8",
        },
        "runtime_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in runtime_files
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"export failed: {error}", file=sys.stderr)
        raise
