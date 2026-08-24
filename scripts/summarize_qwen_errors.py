#!/usr/bin/env python3
"""Summarize a text-free Qwen prediction ledger by split and data slice."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

LABELS = ("SAFE", "UNCERTAIN", "SCAM")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    truth = Counter(str(record["truth"]) for record in records)
    predicted = Counter(str(record["calibrated_verdict"]) for record in records)
    confusion = {
        actual: {
            verdict: sum(
                record["truth"] == actual and record["calibrated_verdict"] == verdict
                for record in records
            )
            for verdict in LABELS
        }
        for actual in LABELS
    }
    false_positives = sum(
        record["truth"] == "SAFE" and bool(record["threshold_scam"])
        for record in records
    )
    false_negatives = sum(
        record["truth"] == "SCAM" and not bool(record["threshold_scam"])
        for record in records
    )
    verdict_errors = sum(
        record["truth"] != record["calibrated_verdict"] for record in records
    )
    return {
        "examples": len(records),
        "truth": dict(sorted(truth.items())),
        "calibrated_verdicts": dict(sorted(predicted.items())),
        "confusion_truth_to_calibrated": confusion,
        "verdict_errors": verdict_errors,
        "verdict_error_rate": verdict_errors / len(records) if records else None,
        "safe_false_positives": false_positives,
        "safe_false_positive_rate": (
            false_positives / truth["SAFE"] if truth["SAFE"] else None
        ),
        "scam_false_negatives": false_negatives,
        "scam_false_negative_rate": (
            false_negatives / truth["SCAM"] if truth["SCAM"] else None
        ),
    }


def summarize_ledger(
    records: list[dict[str, Any]], *, hardest_limit: int = 25
) -> dict[str, Any]:
    required = {
        "id",
        "split",
        "source",
        "category",
        "truth",
        "calibrated_verdict",
        "threshold_scam",
        "negative_log_likelihood",
    }
    for index, record in enumerate(records, start=1):
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"prediction row {index} missing fields: {missing}")
        if "text" in record:
            raise ValueError(f"prediction row {index} contains message text")
        if record["truth"] not in LABELS or record["calibrated_verdict"] not in LABELS:
            raise ValueError(f"prediction row {index} has an invalid label")

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[str(record["split"])].append(record)
        by_source[str(record["source"])].append(record)
        if record.get("source_language"):
            by_source_language[str(record["source_language"])].append(record)
        if record.get("source_domain"):
            by_source_domain[str(record["source_domain"])].append(record)
        by_category[str(record["category"])].append(record)

    errors = [
        record for record in records if record["truth"] != record["calibrated_verdict"]
    ]
    hardest = sorted(
        errors,
        key=lambda record: (-float(record["negative_log_likelihood"]), str(record["id"])),
    )[:hardest_limit]
    return {
        "schema_version": 1,
        "contains_message_text": False,
        "overall": summarize_records(records),
        "by_split": {
            key: summarize_records(values) for key, values in sorted(by_split.items())
        },
        "by_source": {
            key: summarize_records(values) for key, values in sorted(by_source.items())
        },
        "by_source_language": {
            key: summarize_records(values)
            for key, values in sorted(by_source_language.items())
        },
        "by_source_domain": {
            key: summarize_records(values)
            for key, values in sorted(by_source_domain.items())
        },
        "by_category": {
            key: summarize_records(values) for key, values in sorted(by_category.items())
        },
        "hardest_calibrated_errors": [
            {
                "id": record["id"],
                "split": record["split"],
                "source": record["source"],
                "source_language": record.get("source_language"),
                "source_domain": record.get("source_domain"),
                "category": record["category"],
                "truth": record["truth"],
                "calibrated_verdict": record["calibrated_verdict"],
                "threshold_scam": record["threshold_scam"],
                "negative_log_likelihood": record["negative_log_likelihood"],
            }
            for record in hardest
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardest-limit", type=int, default=25)
    args = parser.parse_args()
    if args.hardest_limit < 0:
        raise ValueError("--hardest-limit must be non-negative")
    result = summarize_ledger(read_jsonl(args.predictions), hardest_limit=args.hardest_limit)
    result["prediction_ledger"] = {
        "path": str(args.predictions),
        "sha256": file_sha256(args.predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
