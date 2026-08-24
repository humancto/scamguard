#!/usr/bin/env python3
"""Audit full chat lengths before any Qwen run can truncate supervision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoProcessor

try:
    from training.train_qwen_lora import completion_token_start
except ModuleNotFoundError:  # Direct `python scripts/audit_qwen_tokens.py` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.train_qwen_lora import completion_token_start


def describe(values: list[int]) -> dict[str, int]:
    return {
        "min": min(values),
        "p50": int(np.percentile(values, 50)),
        "p95": int(np.percentile(values, 95)),
        "p99": int(np.percentile(values, 99)),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/qwen_sft"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--comparison-length", type=int, default=384)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/runs/qwen-token-audit-v0.3.json")
    )
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    full_lengths: list[int] = []
    prompt_lengths: list[int] = []
    supervised_lengths: list[int] = []
    split_counts: dict[str, int] = {}
    over_limit: list[dict[str, Any]] = []
    for split in ("train", "dev"):
        count = 0
        with (args.data / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row["messages"]
                prompt = processor.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
                complete = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                encoded = processor.tokenizer(
                    complete,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                full_ids = encoded["input_ids"]
                completion_start = completion_token_start(
                    prompt, complete, encoded["offset_mapping"]
                )
                supervised = len(full_ids) - completion_start
                if supervised <= 0:
                    raise ValueError(f"{split}:{row['id']} has no supervised tokens")
                prompt_lengths.append(completion_start)
                full_lengths.append(len(full_ids))
                supervised_lengths.append(supervised)
                if len(full_ids) > args.max_length:
                    over_limit.append(
                        {"split": split, "id": row["id"], "full_tokens": len(full_ids)}
                    )
                count += 1
        split_counts[split] = count

    result = {
        "model": args.model,
        "revision": args.revision,
        "max_length": args.max_length,
        "examples": len(full_lengths),
        "split_counts": split_counts,
        "prompt_tokens": describe(prompt_lengths),
        "full_tokens": describe(full_lengths),
        "supervised_tokens": describe(supervised_lengths),
        "full_over_max_length": len(over_limit),
        "full_over_comparison_length": sum(
            length > args.comparison_length for length in full_lengths
        ),
        "comparison_length": args.comparison_length,
        "over_limit_examples": over_limit[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if over_limit:
        raise SystemExit(f"token audit failed: {len(over_limit)} examples exceed max length")


if __name__ == "__main__":
    main()
