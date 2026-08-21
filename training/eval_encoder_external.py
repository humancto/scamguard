#!/usr/bin/env python3
"""Score one external diagnostic with a frozen encoder and frozen calibration."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from scamguard.metrics import file_sha256
from scamguard.preprocessing import DIALOGUE_POLICIES, prepare_model_text

try:
    from training.train_encoder import EncodedDataset, read_jsonl, report_slice, softmax
except ModuleNotFoundError:  # Direct execution places training/ rather than the repo on sys.path.
    from train_encoder import (  # type: ignore[no-redef]
        EncodedDataset,
        read_jsonl,
        report_slice,
        softmax,
    )


METADATA_SLICE_FIELDS = ("source_accent", "source_domain", "source_window")


def metadata_slices(
    rows: list[dict[str, object]],
    predictions: np.ndarray,
    temperature: float,
    scam_threshold: float,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for field in METADATA_SLICE_FIELDS:
        values = sorted({str(row[field]) for row in rows if row.get(field) is not None})
        if len(values) < 2:
            continue
        slices: dict[str, object] = {}
        for value in values:
            indexes = [index for index, row in enumerate(rows) if str(row.get(field)) == value]
            metrics = report_slice(
                [rows[index] for index in indexes],
                predictions[indexes],
                temperature,
                scam_threshold,
            )
            metrics.pop("by_language", None)
            slices[value] = metrics
        result[field] = slices
    return result


def artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="ood_external")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--truncation-side",
        choices=("left", "right"),
        default="right",
        help=(
            "Which end of an overlength conversation to discard. Use left to evaluate the "
            "most-recent-token policy used by an incremental conversation scanner."
        ),
    )
    parser.add_argument("--dialogue-policy", choices=DIALOGUE_POLICIES, default="none")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional text-free JSONL ledger for permitted local error analysis.",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"missing checkpoint: {args.checkpoint}")
    if not args.data.is_file():
        raise FileNotFoundError(f"missing diagnostic: {args.data}")
    calibration_path = args.checkpoint / "scamguard_calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"missing frozen calibration: {calibration_path}")

    mps_available = torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError("MPS is required but unavailable")
    device = torch.device("mps" if mps_available else "cpu")
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("external diagnostic is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    tokenizer.truncation_side = args.truncation_side
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, local_files_only=True
    ).to(device)
    model.eval()
    dataset = EncodedDataset(
        rows,
        tokenizer,
        args.max_length,
        dialogue_policy=args.dialogue_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            output = model(**{key: value.to(device) for key, value in batch.items()})
            logits.append(output.logits.float().cpu().numpy())
    predictions = np.concatenate(logits)
    if len(predictions) != len(rows):
        raise RuntimeError("external prediction count differs from source row count")

    calibration = json.loads(calibration_path.read_text())
    raw_encodings = tokenizer(
        [prepare_model_text(str(row["text"]), args.dialogue_policy) for row in rows],
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )
    token_lengths = np.array([len(values) for values in raw_encodings["input_ids"]])
    manifest: dict[str, Any] | None = None
    if args.manifest:
        if not args.manifest.is_file():
            raise FileNotFoundError(f"missing external manifest: {args.manifest}")
        manifest = json.loads(args.manifest.read_text())
    model_file = args.checkpoint / "model.safetensors"
    if not model_file.is_file():
        model_file = args.checkpoint / "pytorch_model.bin"

    temperature = float(calibration["temperature"])
    scam_threshold = float(calibration["scam_threshold"])
    result: dict[str, Any] = {
        "model_id": args.checkpoint.name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_model_sha256": file_sha256(model_file),
        "checkpoint_artifact_bytes": artifact_size(args.checkpoint),
        "calibration_sha256": file_sha256(calibration_path),
        "calibration": calibration,
        "split": args.split,
        "data_sha256": file_sha256(args.data),
        "external_manifest": manifest,
        "data_use": "diagnostic only; no fitting or threshold selection",
        "input_transform": {"dialogue_policy": args.dialogue_policy},
        "sequence_window": {
            "max_tokens": args.max_length,
            "truncated_examples": int(np.sum(token_lengths > args.max_length)),
            "truncated_fraction": float(np.mean(token_lengths > args.max_length)),
            "token_length_p50": int(np.percentile(token_lengths, 50)),
            "token_length_p95": int(np.percentile(token_lengths, 95)),
            "token_length_max": int(token_lengths.max()),
            "truncation_side": tokenizer.truncation_side,
        },
        "environment": {
            "python_arch": platform.machine(),
            "torch": torch.__version__,
            "device": str(device),
            "mps_available": mps_available,
            "batch_size": args.batch_size,
        },
        "metrics": report_slice(
            rows,
            predictions,
            temperature,
            scam_threshold,
        ),
        "metadata_slices": metadata_slices(
            rows,
            predictions,
            temperature,
            scam_threshold,
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.predictions:
        probabilities = softmax(predictions, temperature)
        ledger = []
        for row, scores in zip(rows, probabilities, strict=True):
            ledger.append(
                {
                    "id": row["id"],
                    "family_id": row.get("family_id"),
                    "label": row["label"],
                    "argmax_label": ("SAFE", "UNCERTAIN", "SCAM")[int(np.argmax(scores))],
                    "scam_probability": float(scores[2]),
                    "scam_at_frozen_threshold": bool(
                        scores[2] >= scam_threshold
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
