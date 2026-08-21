#!/usr/bin/env python3
"""Score one external diagnostic with a frozen Qwen adapter and calibration report."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from scamguard.metrics import file_sha256

try:
    from training.eval_qwen import (
        LABELS,
        evaluate_slice,
        predict_with_abstention,
        read_jsonl,
        score_messages,
        softmax,
    )
except ModuleNotFoundError:  # Direct execution places training/ rather than the repo on sys.path.
    from eval_qwen import (  # type: ignore[no-redef]
        LABELS,
        evaluate_slice,
        predict_with_abstention,
        read_jsonl,
        score_messages,
        softmax,
    )


def artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()

    if not args.adapter.is_dir():
        raise FileNotFoundError(f"missing adapter: {args.adapter}")
    if not args.calibration_report.is_file():
        raise FileNotFoundError(f"missing calibration report: {args.calibration_report}")
    if not args.data.is_file():
        raise FileNotFoundError(f"missing diagnostic: {args.data}")
    mps_available = torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError("MPS is required but unavailable")
    device = torch.device("mps" if mps_available else "cpu")

    calibration_source = json.loads(args.calibration_report.read_text())
    temperature = float(calibration_source["temperature"])
    scam_threshold = float(calibration_source["scam_threshold"])
    safe_threshold = float(calibration_source["safe_threshold"])
    rows = read_jsonl(args.data)

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16 if device.type == "mps" else torch.float32,
        low_cpu_mem_usage=True,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision and resolved_revision != args.revision:
        raise RuntimeError(
            f"loaded base revision {resolved_revision} differs from requested {args.revision}"
        )
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter).to(device).eval()
    memory_telemetry: dict[str, int] = {}
    scores = score_messages(
        model,
        processor,
        [str(row["text"]) for row in rows],
        device,
        batch_size=args.batch_size,
        memory_telemetry=memory_telemetry,
        progress_label=args.split,
    )

    manifest: dict[str, Any] | None = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
    adapter_path = args.adapter / "adapter_model.safetensors"
    result: dict[str, Any] = {
        "model": args.model,
        "requested_revision": args.revision,
        "base_model_revision": resolved_revision or args.revision,
        "adapter": str(args.adapter),
        "adapter_sha256": file_sha256(adapter_path),
        "adapter_bytes": artifact_size(args.adapter),
        "calibration_report": str(args.calibration_report),
        "calibration_report_sha256": file_sha256(args.calibration_report),
        "calibration_scope": "frozen historical schema-v6 development calibration",
        "temperature": temperature,
        "scam_threshold": scam_threshold,
        "safe_threshold": safe_threshold,
        "split": args.split,
        "data_sha256": file_sha256(args.data),
        "external_manifest": manifest,
        "data_use": "selection diagnostic only; no fitting or threshold selection",
        "scoring": "length-normalized teacher-forced verdict likelihood",
        "memory_footprint_bytes": model.get_memory_footprint(),
        "memory": memory_telemetry
        | {"measurement": "sampled after every scoring batch; instantaneous lower bound"},
        "environment": {
            "python_arch": platform.machine(),
            "torch": torch.__version__,
            "device": str(device),
            "mps_available": mps_available,
            "batch_size": args.batch_size,
        },
        "metrics": evaluate_slice(
            rows,
            scores,
            temperature,
            scam_threshold,
            safe_threshold=safe_threshold,
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.predictions:
        probabilities = softmax(scores, temperature)
        calibrated = predict_with_abstention(probabilities, scam_threshold, safe_threshold)
        ledger = []
        for row, values, verdict in zip(rows, probabilities, calibrated, strict=True):
            ledger.append(
                {
                    "id": row["id"],
                    "family_id": row.get("family_id"),
                    "label": row["label"],
                    "argmax_label": LABELS[int(np.argmax(values))],
                    "calibrated_verdict": LABELS[int(verdict)],
                    "scam_probability": float(values[LABELS.index("SCAM")]),
                    "scam_at_frozen_threshold": bool(
                        values[LABELS.index("SCAM")] >= scam_threshold
                    ),
                }
            )
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        args.predictions.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in ledger),
            encoding="utf-8",
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
