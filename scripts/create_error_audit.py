#!/usr/bin/env python3
"""Create a deterministic, local-only audit workbook from model errors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/audit/qwen_error_audit.csv"))
    parser.add_argument("--per-bucket", type=int, default=12)
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    splits = sorted({str(record["split"]) for record in predictions})
    examples: dict[tuple[str, str], dict[str, Any]] = {}
    for split in splits:
        for row in read_jsonl(args.data / f"{split}.jsonl"):
            examples[(split, str(row["id"]))] = row

    buckets: list[tuple[str, list[dict[str, Any]]]] = []
    for split in splits:
        records = [record for record in predictions if record["split"] == split]
        buckets.extend(
            [
                (
                    f"{split}:false_negative",
                    [
                        record
                        for record in records
                        if record["truth"] == "SCAM" and not record["threshold_scam"]
                    ],
                ),
                (
                    f"{split}:false_positive",
                    [
                        record
                        for record in records
                        if record["truth"] == "SAFE" and record["threshold_scam"]
                    ],
                ),
                (f"{split}:highest_nll", records),
            ]
        )

    selected: list[tuple[str, dict[str, Any]]] = []
    selected_ids: set[tuple[str, str]] = set()
    for reason, records in buckets:
        ordered = sorted(
            records,
            key=lambda record: (-float(record["negative_log_likelihood"]), str(record["id"])),
        )
        accepted = 0
        for record in ordered:
            key = (str(record["split"]), str(record["id"]))
            if key in selected_ids:
                continue
            selected.append((reason, record))
            selected_ids.add(key)
            accepted += 1
            if accepted >= args.per_bucket:
                break

    fieldnames = [
        "audit_reason",
        "id",
        "dataset_split",
        "source",
        "source_label",
        "label",
        "category",
        "argmax",
        "threshold_scam",
        "safe_probability",
        "uncertain_probability",
        "scam_probability",
        "negative_log_likelihood",
        "text",
        "auditor_label",
        "label_correct",
        "contains_sensitive_data",
        "notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for reason, prediction in selected:
            example = examples[(str(prediction["split"]), str(prediction["id"]))]
            probabilities = prediction["probabilities"]
            writer.writerow(
                {
                    "audit_reason": reason,
                    "id": prediction["id"],
                    "dataset_split": prediction["split"],
                    "source": prediction["source"],
                    "source_label": example.get("source_label", ""),
                    "label": prediction["truth"],
                    "category": prediction["category"],
                    "argmax": prediction["argmax"],
                    "threshold_scam": prediction["threshold_scam"],
                    "safe_probability": probabilities["SAFE"],
                    "uncertain_probability": probabilities["UNCERTAIN"],
                    "scam_probability": probabilities["SCAM"],
                    "negative_log_likelihood": prediction["negative_log_likelihood"],
                    "text": example["text"],
                    "auditor_label": "",
                    "label_correct": "",
                    "contains_sensitive_data": "",
                    "notes": "",
                }
            )
    print(f"wrote {len(selected)} rows to {args.output}")


if __name__ == "__main__":
    main()
