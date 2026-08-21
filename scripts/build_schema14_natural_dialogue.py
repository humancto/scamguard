#!/usr/bin/env python3
"""Append the isolated real-call training increment to schema v13."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from scamguard.metrics import file_sha256

PRESERVED_FILES = (
    "dev.jsonl",
    "test.jsonl",
    "ood_financial.jsonl",
    "ood_wspr.jsonl",
    "forum_validation.jsonl",
    "ood_forum.jsonl",
    "ood_azsc.jsonl",
    "quarantine_label_conflicts.jsonl",
)
EXPECTED_SOURCE = "youtube_scam_calls_cc0"
EXPECTED_SOURCE_LICENSE = "CC0-1.0"
EXPECTED_SOURCE_POLICY = "publisher_positive_only_scam_call_collection"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(
    parent: Path,
    source_train: Path,
    source_manifest_path: Path,
    output: Path,
    *,
    source_window: str = "early",
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite schema-v14 output: {output}")
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("schema_version") != 13:
        raise ValueError("schema-v14 parent must be schema version 13")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_train_hash = source_manifest.get("artifacts", {}).get("train", {}).get("sha256")
    if file_sha256(source_train) != expected_train_hash:
        raise ValueError("real-call training artifact differs from its source manifest")

    parent_train = read_jsonl(parent / "train.jsonl")
    source_rows = [
        row for row in read_jsonl(source_train) if row.get("source_window") == source_window
    ]
    if not source_rows:
        raise ValueError(f"no real-call training rows for source window {source_window!r}")
    for row in source_rows:
        if (
            row.get("source") != EXPECTED_SOURCE
            or row.get("license") != EXPECTED_SOURCE_LICENSE
            or row.get("label") != "SCAM"
            or row.get("label_policy") != EXPECTED_SOURCE_POLICY
            or row.get("split") != "train"
            or row.get("is_synthetic") is not False
        ):
            raise ValueError(f"unexpected real-call training row contract: {row.get('id')}")

    parent_ids = {str(row["id"]) for row in parent_train}
    source_ids = {str(row["id"]) for row in source_rows}
    if len(source_ids) != len(source_rows) or parent_ids & source_ids:
        raise ValueError("real-call increment has duplicate or parent-colliding IDs")
    parent_families = {str(row["family_id"]) for row in parent_train}
    source_families = {str(row["family_id"]) for row in source_rows}
    if parent_families & source_families:
        raise ValueError("real-call increment has a parent-colliding family")

    output.mkdir(parents=True)
    combined_train = parent_train + sorted(source_rows, key=lambda row: str(row["id"]))
    write_jsonl(output / "train.jsonl", combined_train)
    for filename in PRESERVED_FILES:
        source_path = parent / filename
        if source_path.is_file():
            shutil.copy2(source_path, output / filename)

    counts = dict(parent_manifest["counts"])
    counts["train"] = len(combined_train)
    label_counts = Counter(str(row["label"]) for row in combined_train)
    for split in ("dev", "test"):
        for row in read_jsonl(output / f"{split}.jsonl"):
            label_counts[str(row["label"])] += 1
    sources = dict(parent_manifest["sources"])
    sources[EXPECTED_SOURCE] = len(source_rows)
    manifest = dict(parent_manifest)
    manifest.update(
        {
            "schema_version": 14,
            "counts": counts,
            "labels": dict(label_counts),
            "sources": sources,
            "parent": {
                "schema_version": 13,
                "manifest_sha256": file_sha256(parent_manifest_path),
                "train_sha256": file_sha256(parent / "train.jsonl"),
            },
            "schema14_increment": {
                "source": EXPECTED_SOURCE,
                "source_manifest_sha256": file_sha256(source_manifest_path),
                "source_train_sha256": file_sha256(source_train),
                "source_window": source_window,
                "rows": len(source_rows),
                "families": len(source_families),
                "label": "SCAM",
                "license": EXPECTED_SOURCE_LICENSE,
                "provenance": "real scam-call-derived positive-only dialogue",
                "used_for_fitting": True,
                "open_validation_used_for_fitting_or_threshold": False,
                "sealed_ood_opened": False,
            },
            "preserved_parent_artifacts": {
                filename: {
                    "sha256": file_sha256(output / filename),
                    "byte_identical_to_parent": file_sha256(output / filename)
                    == file_sha256(parent / filename),
                }
                for filename in PRESERVED_FILES
                if (output / filename).is_file()
            },
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("data/experiments/schema13-dose16/processed"),
    )
    parser.add_argument(
        "--source-train",
        type=Path,
        default=Path("data/external/youtube_scam_calls/youtube_scam_train.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/external/youtube_scam_calls/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/schema14-natural-dialogue/processed"),
    )
    parser.add_argument("--source-window", choices=("early", "recent"), default="early")
    args = parser.parse_args()
    build(
        args.parent,
        args.source_train,
        args.source_manifest,
        args.output,
        source_window=args.source_window,
    )


if __name__ == "__main__":
    main()
