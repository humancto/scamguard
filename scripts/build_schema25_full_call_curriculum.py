#!/usr/bin/env python3
"""Build schema v25 with licensed full calls and bounded synthetic scam dialogues."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from scamguard.metrics import file_sha256

try:
    from scripts.build_multidogo_dialogues import LICENSE as MULTIDOGO_LICENSE
    from scripts.build_multidogo_dialogues import SOURCE as MULTIDOGO_SOURCE
    from scripts.build_phone_scam_synthetic import SOURCE as PHONE_SOURCE
    from scripts.build_schema19_call_windows import read_jsonl, write_jsonl
    from scripts.build_schema23_evidence_compaction import remove_reference_overlap_families
    from scripts.build_schema24_annotated_hard_negatives import (
        normalize_schema24_rows,
        privacy_counts,
    )
    from scripts.fetch_multidogo import REVISION as MULTIDOGO_REVISION
    from scripts.fetch_phone_scam_synthetic import (
        LICENSE as PHONE_LICENSE,
    )
    from scripts.fetch_phone_scam_synthetic import (
        REVISION as PHONE_REVISION,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_multidogo_dialogues import (  # type: ignore[no-redef]
        LICENSE as MULTIDOGO_LICENSE,
    )
    from build_multidogo_dialogues import (  # type: ignore[no-redef]
        SOURCE as MULTIDOGO_SOURCE,
    )
    from build_phone_scam_synthetic import SOURCE as PHONE_SOURCE  # type: ignore[no-redef]
    from build_schema19_call_windows import read_jsonl, write_jsonl  # type: ignore[no-redef]
    from build_schema23_evidence_compaction import (  # type: ignore[no-redef]
        remove_reference_overlap_families,
    )
    from build_schema24_annotated_hard_negatives import (  # type: ignore[no-redef]
        normalize_schema24_rows,
        privacy_counts,
    )
    from fetch_multidogo import REVISION as MULTIDOGO_REVISION  # type: ignore[no-redef]
    from fetch_phone_scam_synthetic import (  # type: ignore[no-redef]
        LICENSE as PHONE_LICENSE,
    )
    from fetch_phone_scam_synthetic import (
        REVISION as PHONE_REVISION,
    )

SCHEMA_VERSION = 25
PHONE_REQUIRED_POLICY = {
    "synthetic": True,
    "counted_as_real_call_data": False,
    "publisher_split_boundary_preserved": True,
    "test_prediction_sealed_until_candidate_freeze": True,
    "source_metadata_used_as_model_input": False,
    "variation_style_is_perfectly_label_confounded": True,
    "validation_and_test_are_not_final_sota_evidence": True,
}
WEAK_ACTION_FIELDS = {
    "action_label_method",
    "action_targets",
    "action_verdict_weight",
}


def artifact_rows(
    directory: Path, manifest: dict[str, object], name: str, filename: str
) -> list[dict[str, object]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(name), dict):
        raise ValueError(f"manifest is missing {name} artifact")
    contract = artifacts[name]
    path = directory / filename
    if file_sha256(path) != contract.get("sha256"):
        raise ValueError(f"{name} artifact differs from its manifest")
    rows = read_jsonl(path)
    if len(rows) != contract.get("rows"):
        raise ValueError(f"{name} row count differs from its manifest")
    return rows


def full_call_row(row: dict[str, object]) -> dict[str, object]:
    if (
        row.get("source") != MULTIDOGO_SOURCE
        or row.get("license") != MULTIDOGO_LICENSE
        or row.get("source_revision") != MULTIDOGO_REVISION
        or row.get("source_window") != "recent_complete_turns"
        or row.get("label") != "SAFE"
        or row.get("is_synthetic") is not False
    ):
        raise ValueError(f"invalid MultiDoGO full-call row: {row.get('id')}")
    result = {key: value for key, value in row.items() if key not in WEAK_ACTION_FIELDS}
    result.update(
        {
            "split": "train",
            "schema25_admitted": True,
            "schema25_role": "licensed_human_spoken_roleplay_full_call_safe_evidence",
            "schema25_action_supervision": "removed_weak_heuristic_targets",
        }
    )
    return result


def phone_row(row: dict[str, object], split: str) -> dict[str, object]:
    if (
        row.get("source") != PHONE_SOURCE
        or row.get("license") != PHONE_LICENSE
        or row.get("source_revision") != PHONE_REVISION
        or row.get("is_synthetic") is not True
        or row.get("metadata_used_as_model_input") is not False
        or row.get("split") != split
    ):
        raise ValueError(f"invalid phone-scam {split} row: {row.get('id')}")
    return row | {
        "schema25_admitted": True,
        "schema25_role": (
            "synthetic_training_curriculum"
            if split == "train"
            else "confounded_development_diagnostic_only"
        ),
    }


def reference_rows(directory: Path, excluded_names: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.jsonl")):
        if path.name not in excluded_names:
            rows.extend(row for row in read_jsonl(path) if {"id", "family_id", "text"} <= set(row))
    return rows


def build(parent: Path, multidogo: Path, phone: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite schema-v25 output: {output}")
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("schema_version") != 24:
        raise ValueError("schema-v25 parent must be schema version 24")

    multidogo_manifest_path = multidogo / "manifest.json"
    multidogo_manifest = json.loads(multidogo_manifest_path.read_text(encoding="utf-8"))
    if multidogo_manifest.get("revision") != MULTIDOGO_REVISION:
        raise ValueError("MultiDoGO revision differs from the schema-v25 contract")
    multidogo_rows = artifact_rows(
        multidogo,
        multidogo_manifest,
        "real_train",
        "multidogo_real_train.jsonl",
    )

    phone_manifest_path = phone / "manifest.json"
    phone_manifest = json.loads(phone_manifest_path.read_text(encoding="utf-8"))
    phone_policy = phone_manifest.get("policy")
    if (
        phone_manifest.get("artifact_schema_version") != 1
        or phone_manifest.get("source") != PHONE_SOURCE
        or phone_manifest.get("revision") != PHONE_REVISION
        or phone_manifest.get("license_declared_by_publisher") != PHONE_LICENSE
        or not isinstance(phone_policy, dict)
        or any(phone_policy.get(key) is not value for key, value in PHONE_REQUIRED_POLICY.items())
    ):
        raise ValueError("phone-scam derivative differs from the schema-v25 contract")
    source_phone_train = artifact_rows(phone, phone_manifest, "train", "train.jsonl")
    phone_train = normalize_schema24_rows([phone_row(row, "train") for row in source_phone_train])
    phone_validation = normalize_schema24_rows(
        [
            phone_row(row, "validation")
            for row in artifact_rows(phone, phone_manifest, "validation", "validation.jsonl")
        ]
    )
    phone_test = normalize_schema24_rows(artifact_rows(phone, phone_manifest, "test", "test.jsonl"))

    parent_train_source = read_jsonl(parent / "train.jsonl")
    parent_train = [row for row in parent_train_source if row.get("schema24_admitted") is not True]
    excluded_unaudited_rows = len(parent_train_source) - len(parent_train)
    if excluded_unaudited_rows <= 0:
        raise ValueError("schema-v25 expected unaudited schema-v24 rows to exclude")
    calibration = read_jsonl(parent / "action_calibration.jsonl")
    calibration_families = {str(row["family_id"]) for row in calibration}
    complete_calls = normalize_schema24_rows(
        [
            full_call_row(row)
            for row in multidogo_rows
            if row.get("source_window") == "recent_complete_turns"
            and str(row.get("family_id")) not in calibration_families
        ]
    )
    if not complete_calls:
        raise ValueError("schema-v25 has no eligible MultiDoGO full calls")

    eligible_complete_calls = len(complete_calls)
    replaced_families = {str(row["family_id"]) for row in complete_calls}
    parent_without_agent_windows = [
        row
        for row in parent_train
        if not (
            row.get("source") == MULTIDOGO_SOURCE
            and row.get("source_window") == "highest_risk_agent_turn"
            and str(row.get("family_id")) in replaced_families
        )
    ]
    removed_agent_windows = len(parent_train) - len(parent_without_agent_windows)

    parent_held = reference_rows(parent, {"train.jsonl"})
    complete_calls, full_call_overlap = remove_reference_overlap_families(
        complete_calls,
        parent_held + phone_validation + phone_test,
    )
    phone_train, phone_train_overlap = remove_reference_overlap_families(
        phone_train,
        parent_held + phone_validation + phone_test + complete_calls,
    )
    if not complete_calls or not phone_train:
        raise ValueError("schema-v25 increment is empty after overlap control")

    increment = sorted(complete_calls + phone_train, key=lambda row: str(row["id"]))
    combined_train = parent_without_agent_windows + increment
    fit_ids = [str(row["id"]) for row in combined_train]
    held_ids = [str(row["id"]) for row in parent_held + phone_validation + phone_test]
    if len(fit_ids) != len(set(fit_ids)) or set(fit_ids) & set(held_ids):
        raise ValueError("schema-v25 fitting IDs collide or overlap held data")
    increment_families = {str(row["family_id"]) for row in increment}
    held_families = {str(row["family_id"]) for row in parent_held + phone_validation + phone_test}
    if increment_families & held_families:
        raise ValueError("schema-v25 increment family overlaps held data")

    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", combined_train)
    write_jsonl(output / "phone_scam_validation.jsonl", phone_validation)
    retention_source = output / "retention" / "parent_train.jsonl"
    retention_source.parent.mkdir(parents=True)
    write_jsonl(retention_source, parent_without_agent_windows)
    preserved_files: list[str] = []
    for source_path in sorted(parent.glob("*.jsonl")):
        if source_path.name == "train.jsonl":
            continue
        shutil.copy2(source_path, output / source_path.name)
        preserved_files.append(source_path.name)

    output_rows = combined_train + phone_validation
    for filename in preserved_files:
        output_rows.extend(read_jsonl(output / filename))
    privacy_changed_rows, privacy_replacements = privacy_counts(output_rows)

    development_rows = list(combined_train)
    for split in ("dev", "test"):
        development_rows.extend(read_jsonl(output / f"{split}.jsonl"))
    counts = dict(parent_manifest["counts"])
    counts.update({"train": len(combined_train), "phone_scam_validation": len(phone_validation)})
    manifest = dict(parent_manifest)
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "counts": counts,
            "labels": dict(Counter(str(row["label"]) for row in development_rows)),
            "sources": dict(Counter(str(row["source"]) for row in development_rows)),
            "parent": {
                "schema_version": 24,
                "manifest_sha256": file_sha256(parent_manifest_path),
                "train_sha256": file_sha256(parent / "train.jsonl"),
            },
            "schema24_privacy": {
                "revision": parent_manifest["schema24_privacy"]["revision"],
                "rows_processed": len(output_rows),
                "rows_with_replacements": privacy_changed_rows,
                "replacement_counts": dict(privacy_replacements),
                "access_codes_are_never_training_features": True,
                "applied_before_overlap_control": True,
            },
            "schema25_increment": {
                "role": "full-call caller-control curriculum plus synthetic scam coverage",
                "multidogo_manifest_sha256": file_sha256(multidogo_manifest_path),
                "multidogo_revision": MULTIDOGO_REVISION,
                "multidogo_license": MULTIDOGO_LICENSE,
                "multidogo_complete_calls_before_overlap": eligible_complete_calls,
                "multidogo_complete_calls_admitted": len(complete_calls),
                "multidogo_single_agent_windows_replaced": removed_agent_windows,
                "schema24_unaudited_rows_excluded": excluded_unaudited_rows,
                "schema24_unaudited_rows_used_for_fitting": False,
                "retention_anchor_rows": len(parent_without_agent_windows),
                "retention_anchor_sha256": file_sha256(retention_source),
                "retention_anchor_excludes_new_curriculum": True,
                "multidogo_action_targets_removed": True,
                "multidogo_full_call_overlap_control": full_call_overlap,
                "phone_manifest_sha256": file_sha256(phone_manifest_path),
                "phone_revision": PHONE_REVISION,
                "phone_license_declared_by_publisher": PHONE_LICENSE,
                "phone_train_source_rows": len(source_phone_train),
                "phone_train_admitted": len(phone_train),
                "phone_validation_rows": len(phone_validation),
                "phone_test_rows_sealed": len(phone_test),
                "phone_train_overlap_control": phone_train_overlap,
                "phone_test_predictions_opened": False,
                "phone_validation_is_final_sota_evidence": False,
                "phone_style_label_confound_acknowledged": True,
                "real_call_rows_added": len(complete_calls),
                "synthetic_rows_added": len(phone_train),
                "direct_reddit_rows_scraped": 0,
                "public_visibility_treated_as_training_permission": False,
                "sealed_primary_holdouts_opened": False,
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
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("data/experiments/schema24-annotated-hard-negatives/processed"),
    )
    parser.add_argument("--multidogo", type=Path, default=Path("data/external/multidogo"))
    parser.add_argument("--phone", type=Path, default=Path("data/external/phone_scam_synthetic"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/schema25-full-call-curriculum/processed"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.parent, args.multidogo, args.phone, args.output), indent=2))


if __name__ == "__main__":
    main()
