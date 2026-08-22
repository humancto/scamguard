#!/usr/bin/env python3
"""Build schema v24 by admitting audited, split-safe MultiDoGO hard negatives."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

try:
    from scripts.build_multidogo_dialogues import LICENSE as MULTIDOGO_LICENSE
    from scripts.build_multidogo_dialogues import SOURCE as MULTIDOGO_SOURCE
    from scripts.build_schema19_call_windows import read_jsonl, write_jsonl
    from scripts.build_schema23_evidence_compaction import remove_reference_overlap_families
    from scripts.fetch_multidogo import REVISION as MULTIDOGO_REVISION
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_multidogo_dialogues import (  # type: ignore[no-redef]
        LICENSE as MULTIDOGO_LICENSE,
    )
    from build_multidogo_dialogues import (  # type: ignore[no-redef]
        SOURCE as MULTIDOGO_SOURCE,
    )
    from build_schema19_call_windows import (  # type: ignore[no-redef]
        read_jsonl,
        write_jsonl,
    )
    from build_schema23_evidence_compaction import (  # type: ignore[no-redef]
        remove_reference_overlap_families,
    )
    from fetch_multidogo import REVISION as MULTIDOGO_REVISION  # type: ignore[no-redef]

SCHEMA_VERSION = 24


def curriculum_rows(
    directory: Path, manifest: dict[str, Any], split: str
) -> list[dict[str, object]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(split), dict):
        raise ValueError(f"annotation curriculum lacks {split} artifact")
    metadata = artifacts[split]
    path = directory / f"{split}.jsonl"
    if metadata.get("sha256") != file_sha256(path):
        raise ValueError(f"annotation curriculum {split} hash differs")
    rows = read_jsonl(path)
    if metadata.get("rows") != len(rows):
        raise ValueError(f"annotation curriculum {split} count differs")
    return rows


def validate_curriculum_rows(rows: list[dict[str, object]], paper_split: str) -> None:
    for row in rows:
        if (
            row.get("source") != MULTIDOGO_SOURCE
            or row.get("license") != MULTIDOGO_LICENSE
            or row.get("label") != "SAFE"
            or row.get("is_synthetic") is not False
            or row.get("publisher_annotation_split") != paper_split
            or row.get("publisher_annotation_granularity") != "turn"
            or row.get("annotation_audit_sha256") is None
            or "legitimate-domain weak label" not in str(row.get("annotation_label_scope"))
        ):
            raise ValueError(f"invalid annotation curriculum row: {row.get('id')}")


def reference_rows(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def build(parent: Path, curriculum: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite schema-v24 output: {output}")
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("schema_version") != 23:
        raise ValueError("schema-v24 parent must be schema version 23")

    curriculum_manifest_path = curriculum / "manifest.json"
    curriculum_manifest = json.loads(
        curriculum_manifest_path.read_text(encoding="utf-8")
    )
    policy = curriculum_manifest.get("policy")
    if (
        curriculum_manifest.get("artifact_schema_version") != 1
        or curriculum_manifest.get("source") != MULTIDOGO_SOURCE
        or curriculum_manifest.get("license") != MULTIDOGO_LICENSE
        or curriculum_manifest.get("revision") != MULTIDOGO_REVISION
        or not isinstance(policy, dict)
        or policy.get("publisher_annotations_are_not_independent_scam_labels") is not True
        or policy.get("paper_dev_test_rows_enter_fitting") is not False
        or policy.get("existing_source_train_validation_boundary_preserved") is not True
    ):
        raise ValueError("annotation curriculum differs from the schema-v24 contract")
    source_train = curriculum_rows(curriculum, curriculum_manifest, "train")
    annotation_dev = curriculum_rows(curriculum, curriculum_manifest, "dev")
    annotation_test = curriculum_rows(curriculum, curriculum_manifest, "test")
    validate_curriculum_rows(source_train, "train")
    validate_curriculum_rows(annotation_dev, "dev")
    validate_curriculum_rows(annotation_test, "test")

    parent_references = reference_rows(parent)
    parent_train = read_jsonl(parent / "train.jsonl")
    parent_ids = {str(row["id"]) for row in parent_references}
    parent_families = {str(row["family_id"]) for row in parent_references}
    held_annotation_families = {
        str(row["family_id"]) for row in annotation_dev + annotation_test
    }
    collision_families = {
        str(row["family_id"])
        for row in source_train
        if str(row["id"]) in parent_ids
        or str(row["family_id"]) in parent_families
        or str(row["family_id"]) in held_annotation_families
    }
    collision_controlled = [
        row for row in source_train if str(row["family_id"]) not in collision_families
    ]
    admitted, overlap_stats = remove_reference_overlap_families(
        collision_controlled,
        parent_references + annotation_dev + annotation_test,
    )
    admitted = sorted(
        (
            row
            | {
                "split": "train",
                "schema24_admitted": True,
                "schema24_admission_policy": (
                    "publisher_train_annotation_intersection_after_parent_and_held_overlap_control"
                ),
            }
            for row in admitted
        ),
        key=lambda row: str(row["id"]),
    )
    if not admitted:
        raise ValueError("schema-v24 annotation increment is empty after overlap control")
    admitted_ids = {str(row["id"]) for row in admitted}
    admitted_families = {str(row["family_id"]) for row in admitted}
    if admitted_ids & parent_ids or admitted_families & (
        parent_families | held_annotation_families
    ):
        raise ValueError("schema-v24 admitted row overlaps parent or held annotation families")

    output.mkdir(parents=True)
    combined_train = parent_train + admitted
    write_jsonl(output / "train.jsonl", combined_train)
    write_jsonl(output / "multidogo_annotation_dev.jsonl", annotation_dev)
    write_jsonl(output / "multidogo_annotation_test.jsonl", annotation_test)
    preserved_files: list[str] = []
    for source_path in sorted(parent.glob("*.jsonl")):
        if source_path.name == "train.jsonl":
            continue
        shutil.copy2(source_path, output / source_path.name)
        preserved_files.append(source_path.name)

    development_rows = list(combined_train)
    for split in ("dev", "test"):
        development_rows.extend(read_jsonl(output / f"{split}.jsonl"))
    counts = dict(parent_manifest["counts"])
    counts.update(
        {
            "train": len(combined_train),
            "multidogo_annotation_dev": len(annotation_dev),
            "multidogo_annotation_test": len(annotation_test),
        }
    )
    manifest = dict(parent_manifest)
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "counts": counts,
            "labels": dict(Counter(str(row["label"]) for row in development_rows)),
            "sources": dict(Counter(str(row["source"]) for row in development_rows)),
            "parent": {
                "schema_version": 23,
                "manifest_sha256": file_sha256(parent_manifest_path),
                "train_sha256": file_sha256(parent / "train.jsonl"),
            },
            "schema24_increment": {
                "annotation_curriculum_manifest": str(curriculum_manifest_path),
                "annotation_curriculum_manifest_sha256": file_sha256(
                    curriculum_manifest_path
                ),
                "annotation_source_rows": len(source_train),
                "annotation_train_rows": len(admitted),
                "annotation_train_families": len(admitted_families),
                "annotation_dev_rows": len(annotation_dev),
                "annotation_dev_families": len(
                    {str(row["family_id"]) for row in annotation_dev}
                ),
                "annotation_test_rows": len(annotation_test),
                "annotation_test_families": len(
                    {str(row["family_id"]) for row in annotation_test}
                ),
                "collision_families_removed": len(collision_families),
                "collision_rows_removed": len(source_train) - len(collision_controlled),
                "near_overlap_control": overlap_stats,
                "near_overlap_reference_rows": len(
                    parent_references + annotation_dev + annotation_test
                ),
                "paper_train_rows_used_for_fitting": True,
                "paper_dev_test_rows_used_for_fitting": False,
                "publisher_annotations_are_not_independent_scam_labels": True,
                "independent_human_label_audit_required_before_training": True,
                "label_audit_is_post_build_hash_bound_sidecar": True,
                "reddit_rows_directly_scraped": 0,
                "sealed_ood_opened": False,
            },
            "preserved_parent_artifacts": {
                filename: {
                    "sha256": file_sha256(output / filename),
                    "byte_identical_to_parent": file_sha256(output / filename)
                    == file_sha256(parent / filename),
                }
                for filename in preserved_files
            },
        }
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("data/experiments/schema23-evidence-compaction/processed"),
    )
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=Path("data/external/multidogo_annotated"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/schema24-annotated-hard-negatives/processed"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.parent, args.curriculum, args.output), indent=2))


if __name__ == "__main__":
    main()
