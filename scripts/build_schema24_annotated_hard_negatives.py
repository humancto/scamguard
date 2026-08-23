#!/usr/bin/env python3
"""Build schema v24 by admitting audited, split-safe MultiDoGO hard negatives."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256
from scamguard.privacy import (
    CONTEXTUAL_PRIVACY_REVISION,
    mask_contextual_sensitive_values,
)

try:
    from scripts.build_dataset import family_skeleton, simhash64, simhash_bands
    from scripts.build_multidogo_dialogues import LICENSE as MULTIDOGO_LICENSE
    from scripts.build_multidogo_dialogues import SOURCE as MULTIDOGO_SOURCE
    from scripts.build_schema19_call_windows import read_jsonl, write_jsonl
    from scripts.build_schema23_evidence_compaction import remove_reference_overlap_families
    from scripts.fetch_multidogo import REVISION as MULTIDOGO_REVISION
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_dataset import (  # type: ignore[no-redef]
        family_skeleton,
        simhash64,
        simhash_bands,
    )
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


def normalize_schema24_row(row: dict[str, object]) -> dict[str, object]:
    if not isinstance(row.get("text"), str):
        return dict(row)
    result = mask_contextual_sensitive_values(str(row["text"]))
    return row | {
        "text": result.text,
        "schema24_privacy_normalization": CONTEXTUAL_PRIVACY_REVISION,
        "schema24_privacy_values_replaced": result.changed,
        "schema24_privacy_replacement_counts": result.replacement_counts,
    }


def normalize_schema24_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [normalize_schema24_row(row) for row in rows]


def privacy_counts(rows: list[dict[str, object]]) -> tuple[int, Counter[str]]:
    replacements: Counter[str] = Counter()
    changed = 0
    for row in rows:
        if row.get("schema24_privacy_values_replaced") is True:
            changed += 1
        counts = row.get("schema24_privacy_replacement_counts")
        if isinstance(counts, dict):
            replacements.update(
                {
                    str(key): int(value)
                    for key, value in counts.items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }
            )
    return changed, replacements


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


def reference_rows(
    directory: Path,
    *,
    excluded_names: set[str] | None = None,
    normalize_privacy: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.jsonl")):
        if excluded_names and path.name in excluded_names:
            continue
        candidates = [
            row
            for row in read_jsonl(path)
            if {"id", "family_id", "text"} <= row.keys()
        ]
        rows.extend(normalize_schema24_rows(candidates) if normalize_privacy else candidates)
    return rows


def remove_new_privacy_overlap_families(
    original_train: list[dict[str, object]],
    normalized_train: list[dict[str, object]],
    original_references: list[dict[str, object]],
    normalized_references: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Remove only train-family overlaps introduced by contextual value masking."""

    original_train_by_id = {str(row["id"]): row for row in original_train}
    original_reference_by_id = {str(row["id"]): row for row in original_references}
    original_train_signatures = {
        identifier: simhash64(family_skeleton(str(row["text"])))
        for identifier, row in original_train_by_id.items()
    }
    original_reference_signatures = {
        identifier: simhash64(family_skeleton(str(row["text"])))
        for identifier, row in original_reference_by_id.items()
    }
    original_train_text = {
        identifier: " ".join(str(row["text"]).casefold().split())
        for identifier, row in original_train_by_id.items()
    }
    original_reference_text = {
        identifier: " ".join(str(row["text"]).casefold().split())
        for identifier, row in original_reference_by_id.items()
    }
    changed_train_ids = {
        str(row["id"])
        for row in normalized_train
        if " ".join(str(row["text"]).casefold().split())
        != original_train_text[str(row["id"])]
    }
    changed_reference_ids = {
        str(row["id"])
        for row in normalized_references
        if " ".join(str(row["text"]).casefold().split())
        != original_reference_text[str(row["id"])]
    }
    reference_buckets: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = {}
    for row in normalized_references:
        signature = simhash64(family_skeleton(str(row["text"])))
        for band in simhash_bands(signature, max_hamming=6):
            reference_buckets.setdefault(band, []).append((signature, row))

    removed_families: set[str] = set()
    exact_pairs = 0
    near_pairs = 0
    for row in normalized_train:
        identifier = str(row["id"])
        signature = simhash64(family_skeleton(str(row["text"])))
        candidates: dict[str, tuple[int, dict[str, object]]] = {}
        for band in simhash_bands(signature, max_hamming=6):
            for reference_signature, reference in reference_buckets.get(band, []):
                candidates[str(reference["id"])] = (reference_signature, reference)
        for reference_id, (reference_signature, reference) in candidates.items():
            if (signature ^ reference_signature).bit_count() > 6:
                continue
            if identifier not in changed_train_ids and reference_id not in changed_reference_ids:
                continue
            after_exact = " ".join(str(row["text"]).casefold().split()) == " ".join(
                str(reference["text"]).casefold().split()
            )
            before_exact = original_train_text[identifier] == original_reference_text[
                reference_id
            ]
            if (
                not after_exact
                and (
                    original_train_signatures[identifier]
                    ^ original_reference_signatures[reference_id]
                ).bit_count()
                <= 6
            ) or (after_exact and before_exact):
                continue
            removed_families.add(str(row["family_id"]))
            if after_exact:
                exact_pairs += 1
            else:
                near_pairs += 1

    retained = [
        row for row in normalized_train if str(row["family_id"]) not in removed_families
    ]
    return retained, {
        "candidate_rows_before_overlap_control": len(normalized_train),
        "new_exact_collision_pairs": exact_pairs,
        "new_near_collision_pairs": near_pairs,
        "families_removed": len(removed_families),
        "rows_removed_with_families": len(normalized_train) - len(retained),
        "candidate_rows_after_overlap_control": len(retained),
        "near_hamming_max": 6,
        "reference_rows": len(normalized_references),
    }


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
        curriculum_manifest.get("artifact_schema_version") != 2
        or curriculum_manifest.get("source") != MULTIDOGO_SOURCE
        or curriculum_manifest.get("license") != MULTIDOGO_LICENSE
        or curriculum_manifest.get("revision") != MULTIDOGO_REVISION
        or not isinstance(policy, dict)
        or policy.get("publisher_annotations_are_not_independent_scam_labels") is not True
        or policy.get("paper_dev_test_rows_enter_fitting") is not False
        or policy.get("publisher_paper_split_boundary_preserved") is not True
    ):
        raise ValueError("annotation curriculum differs from the schema-v24 contract")
    source_train = normalize_schema24_rows(
        curriculum_rows(curriculum, curriculum_manifest, "train")
    )
    annotation_dev = normalize_schema24_rows(
        curriculum_rows(curriculum, curriculum_manifest, "dev")
    )
    annotation_test = normalize_schema24_rows(
        curriculum_rows(curriculum, curriculum_manifest, "test")
    )
    validate_curriculum_rows(source_train, "train")
    validate_curriculum_rows(annotation_dev, "dev")
    validate_curriculum_rows(annotation_test, "test")

    original_parent_train = read_jsonl(parent / "train.jsonl")
    parent_train = normalize_schema24_rows(original_parent_train)
    original_parent_held_references = reference_rows(
        parent,
        excluded_names={"train.jsonl"},
        normalize_privacy=False,
    )
    parent_held_references = normalize_schema24_rows(original_parent_held_references)
    parent_train, parent_privacy_overlap_stats = remove_new_privacy_overlap_families(
        original_parent_train,
        parent_train,
        original_parent_held_references,
        parent_held_references,
    )
    parent_references = parent_train + parent_held_references
    annotation_test_source_rows = len(annotation_test)
    annotation_test, test_overlap_stats = remove_reference_overlap_families(
        annotation_test,
        parent_references,
    )
    annotation_dev_source_rows = len(annotation_dev)
    annotation_dev, dev_overlap_stats = remove_reference_overlap_families(
        annotation_dev,
        parent_references + annotation_test,
    )
    if not annotation_dev or not annotation_test:
        raise ValueError("schema-v24 held annotation slices are empty after overlap control")
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
    output_rows = combined_train + annotation_dev + annotation_test
    for source_path in sorted(parent.glob("*.jsonl")):
        if source_path.name == "train.jsonl":
            continue
        normalized_rows = normalize_schema24_rows(read_jsonl(source_path))
        write_jsonl(output / source_path.name, normalized_rows)
        output_rows.extend(normalized_rows)
        preserved_files.append(source_path.name)

    privacy_changed_rows, privacy_replacements = privacy_counts(output_rows)

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
    policy = dict(parent_manifest.get("policy", {}))
    policy["real_source_privacy_normalization"] = (
        "emails, phone/account-like digit sequences, contextual access codes, "
        "account fragments, postal codes, and credential-like values replaced "
        "with typed placeholders"
    )
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "counts": counts,
            "labels": dict(Counter(str(row["label"]) for row in development_rows)),
            "sources": dict(Counter(str(row["source"]) for row in development_rows)),
            "policy": policy,
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
                "annotation_dev_source_rows": annotation_dev_source_rows,
                "annotation_dev_families": len(
                    {str(row["family_id"]) for row in annotation_dev}
                ),
                "annotation_test_rows": len(annotation_test),
                "annotation_test_source_rows": annotation_test_source_rows,
                "annotation_test_families": len(
                    {str(row["family_id"]) for row in annotation_test}
                ),
                "collision_families_removed": len(collision_families),
                "collision_rows_removed": len(source_train) - len(collision_controlled),
                "near_overlap_control": overlap_stats,
                "held_near_overlap_control": {
                    "dev": dev_overlap_stats,
                    "test": test_overlap_stats,
                },
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
                "parent_train_post_privacy_overlap_control": (
                    parent_privacy_overlap_stats
                ),
            },
            "schema24_privacy": {
                "revision": CONTEXTUAL_PRIVACY_REVISION,
                "rows_processed": len(output_rows),
                "rows_with_replacements": privacy_changed_rows,
                "replacement_counts": dict(privacy_replacements),
                "access_codes_are_never_training_features": True,
                "applied_before_overlap_control": True,
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
