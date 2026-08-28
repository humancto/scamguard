#!/usr/bin/env python3
"""Build a training-only curriculum for Qwen verdict-branch correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

LABELS = ("SAFE", "UNCERTAIN", "SCAM")
DEFAULT_TARGET_SOURCES = (
    "imc25_public_forum_smishing",
    "mendeley_sms_phishing",
    "scamguard_synthetic_v5",
    "uci_sms_spam",
    "wspr_sms_phishing",
    "youtube_scam_calls_cc0",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def verdict(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row.get('id')} has no messages")
    try:
        value = json.loads(messages[-1]["content"])["verdict"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"row {row.get('id')} has an invalid assistant verdict") from error
    if value not in LABELS:
        raise ValueError(f"row {row.get('id')} has unsupported verdict {value!r}")
    return str(value)


def rank(row: dict[str, Any], salt: str) -> str:
    identifier = str(row.get("id", ""))
    return hashlib.sha256(f"{salt}\0{identifier}".encode()).hexdigest()


def take_ranked(rows: list[dict[str, Any]], count: int, salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: rank(row, salt))[:count]


def build(
    parent: Path,
    output: Path,
    *,
    target_sources: tuple[str, ...] = DEFAULT_TARGET_SOURCES,
    retention_per_label: int = 1024,
    salt: str = "scamguard-qwen08-branch-stage6-v1",
) -> dict[str, Any]:
    if retention_per_label < 0:
        raise ValueError("retention_per_label must be non-negative")
    parent_train = parent / "train.jsonl"
    parent_dev = parent / "dev.jsonl"
    parent_manifest = parent / "manifest.json"
    for path in (parent_train, parent_dev, parent_manifest):
        if not path.is_file():
            raise ValueError(f"missing parent artifact: {path}")

    train = read_jsonl(parent_train)
    dev = read_jsonl(parent_dev)
    identifiers = [str(row.get("id", "")) for row in train]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("parent train IDs must be non-empty and unique")
    dev_identifiers = {str(row.get("id", "")) for row in dev}
    if not all(dev_identifiers) or set(identifiers) & dev_identifiers:
        raise ValueError("parent train and dev IDs must be non-empty and disjoint")
    train_families = {str(row.get("family_id", "")) for row in train}
    dev_families = {str(row.get("family_id", "")) for row in dev}
    if not all(train_families | dev_families) or train_families & dev_families:
        raise ValueError("parent train and dev families must be non-empty and disjoint")

    selected: dict[str, dict[str, Any]] = {}
    target_counts: dict[str, dict[str, int]] = {}
    for source in target_sources:
        source_rows = [row for row in train if row.get("source") == source]
        uncertain = [row for row in source_rows if verdict(row) == "UNCERTAIN"]
        if not uncertain:
            continue
        per_source: dict[str, int] = {"UNCERTAIN": len(uncertain)}
        for row in uncertain:
            selected[str(row["id"])] = row
        for label in ("SAFE", "SCAM"):
            controls = [row for row in source_rows if verdict(row) == label]
            chosen = take_ranked(
                controls,
                min(len(controls), len(uncertain)),
                f"{salt}:{source}:{label}",
            )
            per_source[label] = len(chosen)
            for row in chosen:
                selected[str(row["id"])] = row
        target_counts[source] = per_source

    retention_counts: dict[str, int] = {}
    for label in LABELS:
        candidates = [
            row
            for row in train
            if verdict(row) == label and str(row["id"]) not in selected
        ]
        chosen = take_ranked(
            candidates,
            min(retention_per_label, len(candidates)),
            f"{salt}:retention:{label}",
        )
        retention_counts[label] = len(chosen)
        for row in chosen:
            selected[str(row["id"])] = row

    output_train = sorted(selected.values(), key=lambda row: rank(row, f"{salt}:output"))
    output_dev = dev
    train_path = output / "qwen_sft" / "train.jsonl"
    dev_path = output / "qwen_sft" / "dev.jsonl"
    write_jsonl(train_path, output_train)
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parent_dev, dev_path)

    train_verdicts = Counter(verdict(row) for row in output_train)
    train_sources = Counter(str(row.get("source", "UNKNOWN")) for row in output_train)
    manifest: dict[str, Any] = {
        "artifact_schema_version": 1,
        "experiment_kind": "qwen_branch_focal_kl_stage6_curriculum",
        "schema_version": 24,
        "release_eligible": False,
        "publication_authorized": False,
        "parent": {
            "directory": str(parent),
            "manifest_sha256": file_sha256(parent_manifest),
            "train_sha256": file_sha256(parent_train),
            "dev_sha256": file_sha256(parent_dev),
            "train_rows": len(train),
            "dev_rows": len(dev),
        },
        "selection": {
            "policy": (
                "all target-source UNCERTAIN rows, source-matched SAFE/SCAM controls, "
                "then deterministic cross-source retention"
            ),
            "salt": salt,
            "target_sources": list(target_sources),
            "target_counts": target_counts,
            "retention_per_label": retention_per_label,
            "retention_counts": retention_counts,
            "held_rows_used_for_fitting": 0,
            "primary_test_rows_used": 0,
            "bothbosu_rows_used": 0,
            "family_cross_split": False,
        },
        "splits": {
            "train": {
                "rows": len(output_train),
                "families": len({str(row.get("family_id")) for row in output_train}),
                "sha256": file_sha256(train_path),
                "verdicts": dict(sorted(train_verdicts.items())),
                "sources": dict(sorted(train_sources.items())),
            },
            "dev": {
                "rows": len(output_dev),
                "families": len({str(row.get("family_id")) for row in output_dev}),
                "sha256": file_sha256(dev_path),
                "byte_identical_to_parent": file_sha256(dev_path) == file_sha256(parent_dev),
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sft_manifest = {
        "artifact_schema_version": 1,
        "input_manifest_sha256": file_sha256(manifest_path),
        "policy": {
            "training_only_selection": True,
            "held_rows_used_for_fitting": 0,
            "dev_byte_identical_to_parent": True,
        },
        "splits": manifest["splits"],
    }
    (output / "qwen_sft" / "manifest.json").write_text(
        json.dumps(sft_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retention-per-label", type=int, default=1024)
    parser.add_argument("--salt", default="scamguard-qwen08-branch-stage6-v1")
    args = parser.parse_args()
    manifest = build(
        args.parent,
        args.output,
        retention_per_label=args.retention_per_label,
        salt=args.salt,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
