#!/usr/bin/env python3
"""Evaluate an encoder as an incremental per-incoming-turn dialogue scanner."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from scamguard.metrics import file_sha256
from scamguard.preprocessing import parse_dialogue_turns

try:
    from training.train_encoder import LABELS, read_jsonl, report_slice, softmax
except ModuleNotFoundError:  # Direct execution places training/ rather than the repo on sys.path.
    from train_encoder import LABELS, read_jsonl, report_slice, softmax  # type: ignore[no-redef]


class TurnDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, texts: list[str], tokenizer: Any, max_length: int) -> None:
        self.encodings = tokenizer(texts, max_length=max_length, truncation=True, padding=False)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: torch.tensor(values[index]) for key, values in self.encodings.items()}


def artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()

    calibration_path = args.checkpoint / "scamguard_calibration.json"
    if not args.checkpoint.is_dir() or not calibration_path.is_file():
        raise FileNotFoundError(f"missing calibrated checkpoint: {args.checkpoint}")
    if not args.data.is_file():
        raise FileNotFoundError(f"missing diagnostic: {args.data}")
    mps_available = torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError("MPS is required but unavailable")
    device = torch.device("mps" if mps_available else "cpu")

    rows = read_jsonl(args.data)
    turn_texts: list[str] = []
    turn_owners: list[int] = []
    turns_per_dialogue: list[int] = []
    for row_index, row in enumerate(rows):
        turns = parse_dialogue_turns(str(row["text"]))
        if not turns:
            raise ValueError(f"row is not a recognized multi-turn dialogue: {row['id']}")
        incoming_speaker = turns[0][0]
        incoming_turns = [content for speaker, content in turns if speaker == incoming_speaker]
        if not incoming_turns:
            raise RuntimeError(f"dialogue has no first-speaker turns: {row['id']}")
        turns_per_dialogue.append(len(incoming_turns))
        turn_texts.extend(incoming_turns)
        turn_owners.extend([row_index] * len(incoming_turns))

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, local_files_only=True
    ).to(device)
    model.eval()
    dataset = TurnDataset(turn_texts, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            output = model(**{key: value.to(device) for key, value in batch.items()})
            batches.append(output.logits.float().cpu().numpy())
    turn_logits = np.concatenate(batches)
    if len(turn_logits) != len(turn_texts):
        raise RuntimeError("turn prediction count differs from materialized turn count")

    calibration = json.loads(calibration_path.read_text())
    temperature = float(calibration["temperature"])
    turn_probabilities = softmax(turn_logits, temperature)
    selected_logits = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    selected_turn_ordinal = np.zeros(len(rows), dtype=np.int64)
    owner_array = np.asarray(turn_owners)
    for row_index in range(len(rows)):
        candidates = np.flatnonzero(owner_array == row_index)
        best_ordinal = int(np.argmax(turn_probabilities[candidates, 2]))
        best = int(candidates[best_ordinal])
        selected_logits[row_index] = turn_logits[best]
        selected_turn_ordinal[row_index] = best_ordinal

    manifest = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
    model_path = args.checkpoint / "model.safetensors"
    result: dict[str, Any] = {
        "model_id": args.checkpoint.name,
        "checkpoint_model_sha256": file_sha256(model_path),
        "checkpoint_artifact_bytes": artifact_size(args.checkpoint),
        "calibration_sha256": file_sha256(calibration_path),
        "calibration": calibration,
        "split": args.split,
        "data_sha256": file_sha256(args.data),
        "external_manifest": manifest,
        "data_use": "selection diagnostic only; no fitting or threshold selection",
        "aggregation_policy": {
            "id": "first-speaker-turn-max-v1",
            "contract": (
                "score each first-speaker turn independently and use the turn with maximum "
                "calibrated scam probability as the dialogue verdict"
            ),
            "product_assumption": "first speaker represents the remote/incoming party",
        },
        "turns": {
            "dialogues": len(rows),
            "scored_incoming_turns": len(turn_texts),
            "per_dialogue_min": min(turns_per_dialogue),
            "per_dialogue_p50": float(np.percentile(turns_per_dialogue, 50)),
            "per_dialogue_p95": float(np.percentile(turns_per_dialogue, 95)),
            "per_dialogue_max": max(turns_per_dialogue),
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
            selected_logits,
            temperature,
            float(calibration["scam_threshold"]),
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.predictions:
        selected_probabilities = softmax(selected_logits, temperature)
        ledger = []
        for row_index, (row, scores) in enumerate(
            zip(rows, selected_probabilities, strict=True)
        ):
            ledger.append(
                {
                    "id": row["id"],
                    "family_id": row.get("family_id"),
                    "label": row["label"],
                    "argmax_label": LABELS[int(np.argmax(scores))],
                    "scam_probability": float(scores[2]),
                    "scam_at_frozen_threshold": bool(
                        scores[2] >= float(calibration["scam_threshold"])
                    ),
                    "selected_incoming_turn_ordinal": int(selected_turn_ordinal[row_index]),
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
