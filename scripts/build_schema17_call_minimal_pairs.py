#!/usr/bin/env python3
"""Build schema v17 from schema v14 plus family-held call minimal pairs."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.metrics import file_sha256
from scamguard.signals import extract_signal_matches

try:
    from scripts.generate_call_minimal_pairs import (
        HOLDOUT_SCENARIOS,
        SOURCE,
        SYNTHETIC_METHOD,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from generate_call_minimal_pairs import (  # type: ignore[no-redef]
        HOLDOUT_SCENARIOS,
        SOURCE,
        SYNTHETIC_METHOD,
    )

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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_pair_rows(rows: list[dict[str, object]]) -> None:
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("source") != SOURCE
            or row.get("license") != "Apache-2.0"
            or row.get("split") != "train"
            or row.get("is_synthetic") is not True
            or row.get("synthetic_method") != SYNTHETIC_METHOD
            or row.get("external_benchmark_text_copied") is not False
        ):
            raise ValueError(f"unexpected call-pair row contract: {row.get('id')}")
        if row.get("label") == "SCAM" and not extract_signal_matches(str(row["text"])):
            raise ValueError(f"call-pair SCAM row lacks extractive evidence: {row.get('id')}")
        grouped[str(row["family_id"])].append(row)
    for family_id, pair in grouped.items():
        if len(pair) != 2 or {str(row["label"]) for row in pair} != {"SAFE", "SCAM"}:
            raise ValueError(f"invalid call minimal-pair family: {family_id}")
        if len({str(row["shared_context_sha256"]) for row in pair}) != 1:
            raise ValueError(f"call minimal-pair context hash differs: {family_id}")


def build(
    parent: Path,
    source_data: Path,
    source_manifest_path: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite schema-v17 output: {output}")
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("schema_version") != 14:
        raise ValueError("schema-v17 parent must be schema version 14")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("source") != SOURCE:
        raise ValueError("call-pair generator manifest has an unexpected source")
    if file_sha256(source_data) != source_manifest.get("sha256"):
        raise ValueError("call-pair artifact differs from its generator manifest")

    source_rows = read_jsonl(source_data)
    validate_pair_rows(source_rows)
    expected_holdouts = set(HOLDOUT_SCENARIOS)
    actual_scenarios = {str(row["scenario"]) for row in source_rows}
    if not expected_holdouts < actual_scenarios:
        raise ValueError("call-pair holdout scenarios are missing or consume every scenario")

    train_increment: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for source_row in source_rows:
        row = dict(source_row)
        if str(row["scenario"]) in expected_holdouts:
            row["split"] = "validation"
            validation_rows.append(row)
        else:
            row["split"] = "train"
            train_increment.append(row)

    train_families = {str(row["family_id"]) for row in train_increment}
    validation_families = {str(row["family_id"]) for row in validation_rows}
    if train_families & validation_families:
        raise ValueError("call-pair family crosses train and validation")
    for partition in (train_increment, validation_rows):
        partition_counts = Counter(str(row["label"]) for row in partition)
        if partition_counts["SAFE"] != partition_counts["SCAM"]:
            raise ValueError(f"call-pair partition is not label balanced: {partition_counts}")

    parent_train = read_jsonl(parent / "train.jsonl")
    parent_ids = {str(row["id"]) for row in parent_train}
    source_ids = {str(row["id"]) for row in source_rows}
    if len(source_ids) != len(source_rows) or parent_ids & source_ids:
        raise ValueError("call-pair increment has duplicate or parent-colliding IDs")

    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", parent_train + train_increment)
    write_jsonl(output / "call_pair_validation.jsonl", validation_rows)
    for filename in PRESERVED_FILES:
        source_path = parent / filename
        if source_path.is_file():
            shutil.copy2(source_path, output / filename)

    counts = dict(parent_manifest["counts"])
    counts["train"] = len(parent_train) + len(train_increment)
    counts["call_pair_validation"] = len(validation_rows)
    development_rows = read_jsonl(output / "train.jsonl")
    for split in ("dev", "test"):
        development_rows.extend(read_jsonl(output / f"{split}.jsonl"))
    manifest = dict(parent_manifest)
    manifest.update(
        {
            "schema_version": 17,
            "counts": counts,
            "labels": dict(Counter(str(row["label"]) for row in development_rows)),
            "sources": dict(Counter(str(row["source"]) for row in development_rows)),
            "parent": {
                "schema_version": 14,
                "manifest_sha256": file_sha256(parent_manifest_path),
                "train_sha256": file_sha256(parent / "train.jsonl"),
            },
            "schema17_increment": {
                "source": SOURCE,
                "source_manifest_sha256": file_sha256(source_manifest_path),
                "source_data_sha256": file_sha256(source_data),
                "train_rows": len(train_increment),
                "validation_rows": len(validation_rows),
                "train_pair_families": len(train_families),
                "validation_pair_families": len(validation_families),
                "train_labels": dict(
                    Counter(str(row["label"]) for row in train_increment)
                ),
                "validation_labels": dict(
                    Counter(str(row["label"]) for row in validation_rows)
                ),
                "holdout_scenarios": sorted(expected_holdouts),
                "license": "Apache-2.0",
                "provenance": (
                    "original deterministic structure-matched advisory-grounded minimal pairs"
                ),
                "minimal_contrast_field": "final_agent_action",
                "used_for_fitting": True,
                "used_for_threshold": False,
                "apptek_rows_used_for_fitting": 0,
                "apptek_ood_opened": False,
                "youtube_ood_opened": False,
                "bothbosu_ood_opened": False,
                "moz_holdout_opened": False,
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
        default=Path("data/experiments/schema14-natural-dialogue/processed"),
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=Path("data/generated/call_minimal_pairs.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/generated/call_minimal_pairs_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/schema17-call-minimal-pairs/processed"),
    )
    args = parser.parse_args()
    build(args.parent, args.source_data, args.source_manifest, args.output)


if __name__ == "__main__":
    main()
