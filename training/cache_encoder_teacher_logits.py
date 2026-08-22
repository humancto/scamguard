#!/usr/bin/env python3
"""Cache a frozen encoder's text-free logits for continual-learning retention."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from scamguard.metrics import file_sha256
from scamguard.preprocessing import DIALOGUE_POLICIES

try:
    from training.train_encoder import ACTION_TARGETS, LABELS, EncodedDataset, read_jsonl
except ModuleNotFoundError:  # Direct execution places training/ rather than repo on sys.path.
    from train_encoder import (  # type: ignore[no-redef]
        ACTION_TARGETS,
        LABELS,
        EncodedDataset,
        read_jsonl,
    )


def checkpoint_model_file(checkpoint: Path) -> Path:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = checkpoint / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing model weights in {checkpoint}")


def verdict_logits(logits: torch.Tensor) -> torch.Tensor:
    """Strip auxiliary heads so the retention ledger preserves only the verdict contract."""

    if logits.ndim != 2 or logits.shape[1] < len(LABELS):
        raise ValueError("teacher output does not contain all verdict logits")
    return logits[:, : len(LABELS)]


def existing_cache(
    output: Path,
    manifest_path: Path,
    checkpoint: Path,
    data: Path,
    dialogue_policy: str,
) -> dict[str, object] | None:
    if not output.exists() and not manifest_path.exists():
        return None
    if not output.is_file() or not manifest_path.is_file():
        raise RuntimeError("teacher cache is partial; remove it before rebuilding")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "ledger_sha256": file_sha256(output),
        "checkpoint_model_sha256": file_sha256(checkpoint_model_file(checkpoint)),
        "data_sha256": file_sha256(data),
        "dialogue_policy": dialogue_policy,
    }
    mismatches = {
        field: (manifest.get(field), value)
        for field, value in expected.items()
        if manifest.get(field) != value
    }
    if mismatches:
        raise RuntimeError(f"existing teacher cache differs from request: {mismatches}")
    return manifest


def cache(
    checkpoint: Path,
    data: Path,
    output: Path,
    manifest_path: Path,
    *,
    dialogue_policy: str = "speaker-neutral-v1",
    max_length: int = 256,
    batch_size: int = 32,
    require_mps: bool = False,
) -> dict[str, object]:
    prior = existing_cache(output, manifest_path, checkpoint, data, dialogue_policy)
    if prior is not None:
        print(json.dumps(prior, indent=2, sort_keys=True))
        return prior
    if not checkpoint.is_dir() or not data.is_file():
        raise FileNotFoundError("teacher checkpoint or source training data is missing")
    rows = read_jsonl(data)
    identifiers = [str(row["id"]) for row in rows]
    if not rows or len(set(identifiers)) != len(rows):
        raise ValueError("teacher source data is empty or has duplicate IDs")

    mps_available = torch.backends.mps.is_available()
    if require_mps and not mps_available:
        raise RuntimeError("MPS is required but unavailable")
    device = torch.device("mps" if mps_available else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to(device)
    model.eval()
    dataset = EncodedDataset(
        rows,
        tokenizer,
        max_length,
        dialogue_policy=dialogue_policy,
        action_target_names=(
            ACTION_TARGETS
            if any(isinstance(row.get("action_targets"), dict) for row in rows)
            else ()
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    records: list[dict[str, object]] = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            for metadata_key in (
                "labels",
                "action_targets",
                "action_mask",
                "verdict_weight",
            ):
                batch.pop(metadata_key, None)
            logits = verdict_logits(
                model(**{key: value.to(device) for key, value in batch.items()}).logits
            )
            values = logits.float().cpu().tolist()
            for identifier, row_logits in zip(
                identifiers[offset : offset + len(values)],
                values,
                strict=True,
            ):
                records.append({"id": identifier, "logits": row_logits})
            offset += len(values)
    if offset != len(rows):
        raise RuntimeError("teacher cache prediction count differs from source data")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "role": "text-free frozen-logit retention targets; never labels or source text",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": file_sha256(checkpoint_model_file(checkpoint)),
        "data": str(data),
        "data_sha256": file_sha256(data),
        "rows": len(records),
        "labels": list(LABELS),
        "logit_scope": "first three verdict logits only",
        "dialogue_policy": dialogue_policy,
        "max_length": max_length,
        "ledger": str(output),
        "ledger_sha256": file_sha256(output),
        "contains_text": False,
        "environment": {
            "python_arch": platform.machine(),
            "torch": torch.__version__,
            "device": str(device),
            "mps_available": mps_available,
            "batch_size": batch_size,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/sg-modernbert-schema13-dose16"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/experiments/schema13-dose16/processed/train.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/schema16-retention-alpha05-w2/teacher/"
            "schema13-train-logits.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/experiments/schema16-retention-alpha05-w2/teacher/manifest.json"
        ),
    )
    parser.add_argument(
        "--dialogue-policy",
        choices=DIALOGUE_POLICIES,
        default="speaker-neutral-v1",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args()
    cache(
        args.checkpoint,
        args.data,
        args.output,
        args.manifest,
        dialogue_policy=args.dialogue_policy,
        max_length=args.max_length,
        batch_size=args.batch_size,
        require_mps=args.require_mps,
    )


if __name__ == "__main__":
    main()
