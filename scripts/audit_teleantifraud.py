#!/usr/bin/env python3
"""Create a text-free admission audit for gated TeleAntiFraud binary metadata."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from scamguard.metrics import file_sha256

EXPECTED_SPLITS = {"train.json": 4000, "test.json": 400}
EXPECTED_LABELS = {"fraud", "normal"}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_OR_LONG_DIGITS_RE = re.compile(
    r"(?<![A-Za-z0-9])\+?\d(?:[\d ()-]{5,}\d)(?![A-Za-z0-9])|\d{10,}"
)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    selected: dict[str, zipfile.ZipInfo] = {}
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or member.is_dir():
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
            continue
        basename = path.name
        if basename in EXPECTED_SPLITS:
            if basename in selected:
                raise ValueError(f"duplicate {basename} in archive")
            selected[basename] = member
    if set(selected) != set(EXPECTED_SPLITS):
        raise ValueError(f"archive split files differ from expected: {sorted(selected)}")
    return selected


def text_and_audio(record: dict[str, Any]) -> tuple[str, list[str], Counter[str]]:
    prompt = record.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError("record prompt must be a list")
    text_parts: list[str] = []
    audio_paths: list[str] = []
    shape: Counter[str] = Counter()
    for turn in prompt:
        if not isinstance(turn, dict) or not isinstance(turn.get("role"), str):
            raise ValueError("prompt turn must have a string role")
        role = str(turn["role"])
        content = turn.get("content")
        shape[f"role:{role}"] += 1
        if isinstance(content, str):
            text_parts.append(content)
            shape[f"content:{role}:string"] += 1
            continue
        if not isinstance(content, list):
            raise ValueError("prompt content must be a string or list")
        for item in content:
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                raise ValueError("prompt content item must have a string type")
            item_type = str(item["type"])
            shape[f"content:{role}:{item_type}"] += 1
            if item_type == "text" and isinstance(item.get("text"), str):
                text_parts.append(str(item["text"]))
            elif item_type == "audio" and isinstance(item.get("audio_url"), str):
                audio_paths.append(str(item["audio_url"]))
    return "\n".join(text_parts).strip(), audio_paths, shape


def audit(source: Path, receipt_path: Path | None = None) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    receipt = None
    if receipt_path:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        metadata = receipt.get("files", {}).get(source.name, {})
        if metadata.get("sha256") != file_sha256(source):
            raise ValueError("source archive differs from the pinned download receipt")

    aggregate_labels: Counter[str] = Counter()
    aggregate_keys: Counter[str] = Counter()
    aggregate_shape: Counter[str] = Counter()
    split_reports: dict[str, Any] = {}
    all_text: Counter[str] = Counter()
    with zipfile.ZipFile(source) as archive:
        members = safe_members(archive)
        for split_name, expected_count in EXPECTED_SPLITS.items():
            payload = json.loads(archive.read(members[split_name]))
            if not isinstance(payload, list) or len(payload) != expected_count:
                raise ValueError(
                    f"{split_name} must contain exactly {expected_count} records"
                )
            labels: Counter[str] = Counter()
            keys: Counter[str] = Counter()
            shapes: Counter[str] = Counter()
            pii: Counter[str] = Counter()
            audio_count = 0
            missing_text = 0
            split_text: Counter[str] = Counter()
            provenance_fields: Counter[str] = Counter()
            for record in payload:
                if not isinstance(record, dict):
                    raise ValueError("every source record must be an object")
                keys.update(map(str, record))
                label = str(record.get("answer", "")).strip().casefold()
                if label not in EXPECTED_LABELS:
                    raise ValueError(f"unexpected TeleAntiFraud answer: {label!r}")
                labels[label] += 1
                text, audio_paths, shape = text_and_audio(record)
                shapes.update(shape)
                audio_count += len(audio_paths)
                if not text:
                    missing_text += 1
                else:
                    normalized_text = normalized(text)
                    split_text[normalized_text] += 1
                    all_text[normalized_text] += 1
                    pii["email_like_rows"] += bool(EMAIL_RE.search(text))
                    pii["phone_or_long_digit_rows"] += bool(PHONE_OR_LONG_DIGITS_RE.search(text))
                    pii["url_like_rows"] += bool(URL_RE.search(text))
                for field in ("provenance", "source", "construction", "is_synthetic"):
                    if field in record:
                        provenance_fields[field] += 1
            aggregate_labels.update(labels)
            aggregate_keys.update(keys)
            aggregate_shape.update(shapes)
            split_reports[split_name.removesuffix(".json")] = {
                "rows": len(payload),
                "labels": dict(labels),
                "record_key_presence": dict(keys),
                "prompt_shape": dict(shapes),
                "audio_references": audio_count,
                "missing_text_rows": missing_text,
                "exact_duplicate_text_groups": sum(count > 1 for count in split_text.values()),
                "exact_duplicate_text_rows_beyond_first": sum(
                    count - 1 for count in split_text.values() if count > 1
                ),
                "privacy_like_counts": dict(pii),
                "row_level_provenance_fields": dict(provenance_fields),
            }

    cross_split_duplicate_groups = sum(count > 1 for count in all_text.values())
    report = {
        "audit_schema_version": 1,
        "source": {
            "repository": "JimmyMa99/TeleAntiFraud",
            "revision": "0872e54b584b28d34e0911dffcf696f0b2e5e49a",
            "license_declared_by_publisher": "Apache-2.0",
            "archive_sha256": file_sha256(source),
            "receipt_verified": receipt is not None,
        },
        "splits": split_reports,
        "aggregate": {
            "rows": sum(aggregate_labels.values()),
            "labels": dict(aggregate_labels),
            "record_key_presence": dict(aggregate_keys),
            "prompt_shape": dict(aggregate_shape),
            "cross_split_or_internal_exact_duplicate_text_groups": cross_split_duplicate_groups,
        },
        "admission": {
            "status": "PENDING_MANUAL_SCHEMA_AND_PRIVACY_REVIEW",
            "train_rows_admitted": 0,
            "test_rows_admitted": 0,
            "publisher_test_must_remain_external": True,
            "requires_row_level_real_vs_augmented_vs_synthetic_provenance": True,
            "requires_chinese_native_review": True,
            "requires_privacy_normalization_before_materialization": True,
            "raw_text_written_to_report": False,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/teleantifraud/binary_classification.zip"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/raw/teleantifraud/download_receipt.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/source-audits/teleantifraud.json"),
    )
    args = parser.parse_args()
    report = audit(args.source, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
