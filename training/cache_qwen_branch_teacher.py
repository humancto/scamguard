#!/usr/bin/env python3
"""Cache text-free Qwen verdict-branch logits from an immutable teacher adapter."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from training.qwen_branch import LABELS, BranchCollator, BranchDataset
    from training.train_qwen_lora import adapter_identity, require_loaded_revision, sha256
except ModuleNotFoundError:  # Direct execution: python training/cache_qwen_branch_teacher.py
    from qwen_branch import LABELS, BranchCollator, BranchDataset
    from train_qwen_lora import adapter_identity, require_loaded_revision, sha256


def cache_split(
    model: Any,
    dataset: BranchDataset,
    collator: BranchCollator,
    device: torch.device,
    *,
    split: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    records: list[dict[str, Any]] = []
    maximum_tokens = 0
    total = len(loader)
    for index, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        branch_tokens = batch["branch_token_ids"].to(device)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
            ).logits[:, -1, :]
            selected = logits.gather(1, branch_tokens).float().cpu()
        lengths = attention_mask.sum(dim=1).cpu().tolist()
        maximum_tokens = max(maximum_tokens, max(int(value) for value in lengths))
        for sample_id, target, values, tokens in zip(
            batch["sample_ids"],
            batch["targets"].tolist(),
            selected.tolist(),
            lengths,
            strict=True,
        ):
            records.append(
                {
                    "split": split,
                    "id": sample_id,
                    "label": LABELS[int(target)],
                    "teacher_logits": [float(value) for value in values],
                    "prompt_tokens": int(tokens),
                }
            )
        if index == total or index % 25 == 0:
            print(
                f"{split}: {index}/{total} batches ({len(records)}/{len(dataset)} rows)",
                flush=True,
            )
    return records, maximum_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args()

    mps_available = torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError("--require-mps was set but MPS is unavailable")
    device = torch.device("mps" if mps_available else "cpu")
    print(f"teacher-cache accelerator: {device}")
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16 if mps_available else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    require_loaded_revision(
        requested=args.revision,
        loaded=getattr(base.config, "_commit_hash", None),
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).to(device)
    model.eval()
    collator = BranchCollator(processor.tokenizer.pad_token_id)
    all_records: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    split_maximums: dict[str, int] = {}
    for split in ("train", "dev"):
        dataset = BranchDataset(
            args.data / f"{split}.jsonl",
            processor,
            args.max_length,
            split=split,
        )
        records, maximum = cache_split(
            model,
            dataset,
            collator,
            device,
            split=split,
            batch_size=args.batch_size,
        )
        all_records.extend(records)
        split_counts[split] = len(records)
        split_maximums[split] = maximum

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in all_records),
        encoding="utf-8",
    )
    manifest = {
        "artifact_schema_version": 1,
        "role": "text-free immutable Stage 3 verdict-branch retention targets",
        "contains_message_text": False,
        "model": args.model,
        "base_model_revision": args.revision,
        "adapter": adapter_identity(args.adapter),
        "labels": list(LABELS),
        "scoring_mode": "branch_token",
        "data": {
            split: {
                "path": str(args.data / f"{split}.jsonl"),
                "sha256": sha256(args.data / f"{split}.jsonl"),
                "rows": split_counts[split],
                "maximum_prompt_tokens": split_maximums[split],
            }
            for split in ("train", "dev")
        },
        "cache": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "rows": len(all_records),
        },
        "environment": {
            "python_arch": platform.machine(),
            "torch": torch.__version__,
            "device": str(device),
            "mps_available": mps_available,
            "local_files_only": args.local_files_only,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
