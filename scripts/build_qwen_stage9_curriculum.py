#!/usr/bin/env python3
"""Build Stage 9 from text-free error evidence and quality-filtered training rows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from scripts.audit_source_overlap import near_overlap_indices
from scripts.build_qwen_phone_curriculum import (
    PHONE_SOURCE,
    QWEN_MODEL,
    QWEN_REVISION,
    sft_as_text_rows,
    validate_phone_manifest,
)
from scripts.build_qwen_ppone_curriculum import (
    PPONE_REVISION,
    PPONE_SOURCE,
    family_key,
    identity_sha,
    one_per_family,
    parent_anchors,
    read_jsonl,
    target,
    token_filter,
    validate_parent,
    validate_ppone,
    verdict,
    write_jsonl,
)
from training.build_qwen_sft import convert_supported_rows

EXPERIMENT_KIND = "qwen_evidence_retention_stage9_curriculum"
POLICY_SOURCE = "scamguard_ftc_pattern_action_states_v1"
CRITICAL_CATEGORIES = (
    "CREDENTIAL_MFA",
    "FINANCIAL_IMPERSONATION",
    "GOVERNMENT_LEGAL",
    "JOB_OPPORTUNITY",
)
EXCLUSION_FLAGS = {
    "generation_artifact",
    "publisher_safe_has_high_risk_caller_action",
    "publisher_scam_lacks_detected_runtime_evidence",
    "speaker_parse_failed",
}


def validate_phone_audit(path: Path, phone_train: Path) -> tuple[dict[str, Any], set[str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    policy = report.get("policy", {})
    inputs = report.get("inputs", [])
    matching_inputs = [row for row in inputs if row.get("path") == str(phone_train)]
    findings = report.get("findings", [])
    if (
        report.get("artifact_schema_version") != 1
        or report.get("contains_message_text") is not False
        or policy.get("automatic_relabeling_allowed") is not False
        or policy.get("test_inspected") is not False
        or len(matching_inputs) != 1
        or matching_inputs[0].get("sha256") != file_sha256(phone_train)
        or not isinstance(findings, list)
    ):
        raise ValueError("phone label audit differs from the frozen text-free contract")
    excluded: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict) or "id" not in finding or "flags" not in finding:
            raise ValueError("phone label audit contains an invalid finding")
        flags = finding["flags"]
        if not isinstance(flags, list) or not set(flags) <= EXCLUSION_FLAGS:
            raise ValueError("phone label audit contains an unknown finding flag")
        if finding.get("split") == "train" and set(flags) & EXCLUSION_FLAGS:
            excluded.add(str(finding["id"]))
    if not excluded:
        raise ValueError("phone label audit does not exclude any training rows")
    alignment = report.get("prediction_alignment", {})
    if not {"stage7", "stage8b"} <= set(alignment):
        raise ValueError("phone label audit lacks the frozen transition baselines")
    return report, excluded


def category_anchors(
    rows: list[dict[str, Any]], categories: tuple[str, ...], limit: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category in categories:
        candidates = [row for row in rows if target(row).get("category") == category]
        candidates = sorted(one_per_family(candidates), key=lambda row: str(row["id"]))
        selected.extend(candidates[:limit])
    return selected


def phone_scam_targets(
    rows: list[dict[str, Any]], excluded_ids: set[str], limit_per_type: int
) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("source") == PHONE_SOURCE
            and row.get("label") == "SCAM"
            and row.get("source_length_category") == "long"
            and str(row.get("id")) not in excluded_ids
        ):
            groups[str(row.get("source_dialogue_type"))].append(row)
    if set(groups) != {"refund", "ssn", "support"}:
        raise ValueError("quality-filtered phone targets lack a dialogue type")
    selected: list[dict[str, Any]] = []
    for dialogue_type in sorted(groups):
        candidates = sorted(one_per_family(groups[dialogue_type]), key=lambda row: str(row["id"]))
        selected.extend(candidates[:limit_per_type])
    return selected


def add_rows(
    selected: dict[str, dict[str, Any]],
    reasons: defaultdict[str, set[str]],
    rows: list[dict[str, Any]],
    reason: str,
) -> None:
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or not all(family_key(row)):
            raise ValueError("Stage 9 row has an empty ID, source, or family")
        selected[identifier] = row
        reasons[identifier].add(reason)


def build(
    parent: Path,
    ppone_manifest_path: Path,
    ppone_train_path: Path,
    phone_manifest_path: Path,
    phone_train_path: Path,
    phone_audit_path: Path,
    output: Path,
    *,
    overlap_references: tuple[Path, ...] = (),
    tokenizer: Any | None = None,
    max_length: int = 640,
    anchors_per_source_verdict: int = 20,
    critical_category_families: int = 64,
    phone_scam_families_per_type: int = 24,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite curriculum: {output}")
    if min(
        anchors_per_source_verdict,
        critical_category_families,
        phone_scam_families_per_type,
    ) < 1:
        raise ValueError("Stage 9 selection limits must be positive")

    parent_manifest, parent_train, parent_dev = validate_parent(parent)
    ppone_manifest = validate_ppone(ppone_manifest_path, ppone_train_path)
    phone_manifest = json.loads(phone_manifest_path.read_text(encoding="utf-8"))
    validate_phone_manifest(phone_manifest, phone_train_path)
    phone_audit, excluded_phone_ids = validate_phone_audit(
        phone_audit_path, phone_train_path
    )

    references: list[dict[str, Any]] = []
    reference_contracts: list[dict[str, Any]] = []
    for path in overlap_references:
        rows = read_jsonl(path)
        references.extend(rows)
        reference_contracts.append(
            {"path": str(path), "rows": len(rows), "sha256": file_sha256(path)}
        )
    overlap_rows = references + sft_as_text_rows(parent_dev)

    ppone_raw = read_jsonl(ppone_train_path)
    if any(
        row.get("source") != PPONE_SOURCE
        or row.get("split") != "train"
        or row.get("label") not in {"SCAM", "UNCERTAIN"}
        or row.get("individual_scam_ground_truth") is not False
        for row in ppone_raw
    ):
        raise ValueError("PPoNE training rows violate the frozen label boundary")
    if near_overlap_indices(ppone_raw, overlap_rows, 6):
        raise ValueError("PPoNE training rows near-overlap held or parent dev rows")
    ppone_sft, ppone_unsupported = convert_supported_rows(ppone_raw)
    ppone_sft, ppone_over_length = token_filter(ppone_sft, tokenizer, max_length)
    if ppone_unsupported:
        raise ValueError("PPoNE contains unsupported rows")

    phone_raw = read_jsonl(phone_train_path)
    phone_target_raw = phone_scam_targets(
        phone_raw, excluded_phone_ids, phone_scam_families_per_type
    )
    if near_overlap_indices(phone_target_raw, overlap_rows, 6):
        raise ValueError("phone targets near-overlap held or parent dev rows")
    phone_sft, phone_unsupported = convert_supported_rows(phone_target_raw)
    phone_sft, phone_over_length = token_filter(phone_sft, tokenizer, max_length)
    if not phone_sft:
        raise ValueError("quality-filtered phone target set is empty")

    selected: dict[str, dict[str, Any]] = {}
    reasons: defaultdict[str, set[str]] = defaultdict(set)
    add_rows(
        selected,
        reasons,
        parent_anchors(parent_train, anchors_per_source_verdict),
        "source_verdict_anchor",
    )
    add_rows(
        selected,
        reasons,
        category_anchors(parent_train, CRITICAL_CATEGORIES, critical_category_families),
        "critical_category_anchor",
    )
    policy_rows = [row for row in parent_train if row.get("source") == POLICY_SOURCE]
    if not policy_rows or {verdict(row) for row in policy_rows} != {
        "SAFE",
        "SCAM",
        "UNCERTAIN",
    }:
        raise ValueError("parent lacks the complete three-way action-state policy source")
    add_rows(selected, reasons, policy_rows, "complete_action_state_policy")
    add_rows(selected, reasons, phone_sft, "quality_filtered_long_phone_scam")
    add_rows(selected, reasons, ppone_sft, "real_ppone_abstention")

    train, final_over_length = token_filter(list(selected.values()), tokenizer, max_length)
    final_excluded = {str(row["id"]) for row in final_over_length}
    train = sorted(
        (row for row in train if str(row["id"]) not in excluded_phone_ids),
        key=lambda row: str(row["id"]),
    )
    train_ids = [str(row["id"]) for row in train]
    dev_ids = {str(row["id"]) for row in parent_dev}
    if (
        len(train_ids) != len(set(train_ids))
        or set(train_ids) & dev_ids
        or set(train_ids) & excluded_phone_ids
    ):
        raise ValueError("Stage 9 train identities violate the frozen boundary")
    if {family_key(row) for row in train} & {family_key(row) for row in parent_dev}:
        raise ValueError("Stage 9 train families cross parent dev")

    output_sft = output / "qwen_sft"
    output_sft.mkdir(parents=True)
    train_path = output_sft / "train.jsonl"
    dev_path = output_sft / "dev.jsonl"
    write_jsonl(train_path, train)
    shutil.copy2(parent / "qwen_sft/dev.jsonl", dev_path)

    verdict_counts = Counter(verdict(row) for row in train)
    source_counts = Counter(str(row["source"]) for row in train)
    reason_counts = Counter(
        reason for identifier in train_ids for reason in reasons[identifier]
    )
    manifest: dict[str, Any] = {
        "artifact_schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "schema_version": 25,
        "release_eligible": False,
        "publication_authorized": False,
        "parent": {
            "directory": str(parent),
            "manifest_sha256": file_sha256(parent / "manifest.json"),
            "train_sha256": parent_manifest["splits"]["train"]["sha256"],
            "dev_sha256": parent_manifest["splits"]["dev"]["sha256"],
            "full_replay_rows": 0,
        },
        "selection": {
            "policy": (
                "family-diverse Stage 7 retention, complete three-way FTC action states, "
                "quality-filtered long phone scams, and PPoNE train-only robocalls"
            ),
            "anchors_per_source_verdict": anchors_per_source_verdict,
            "critical_categories": list(CRITICAL_CATEGORIES),
            "critical_category_family_limit": critical_category_families,
            "phone_scam_family_limit_per_type": phone_scam_families_per_type,
            "reason_row_counts": dict(sorted(reason_counts.items())),
            "selected_ids_sha256": identity_sha(train_ids),
            "held_rows_used_for_fitting": 0,
            "phone_validation_rows_used_for_fitting": 0,
            "phone_test_rows_read": 0,
            "ppone_validation_rows_read": 0,
            "ppone_test_rows_read": 0,
            "sealed_predictions_opened": False,
        },
        "phone_quality": {
            "audit_path": str(phone_audit_path),
            "audit_sha256": file_sha256(phone_audit_path),
            "audit_test_inspected": phone_audit["policy"]["test_inspected"],
            "excluded_training_ids": len(excluded_phone_ids),
            "excluded_training_ids_sha256": identity_sha(list(excluded_phone_ids)),
            "automatic_relabeling": False,
            "raw_target_rows": len(phone_target_raw),
            "converted_target_rows": len(phone_sft),
            "unsupported_target_rows": len(phone_unsupported),
            "over_length_target_rows": len(phone_over_length),
        },
        "ppone": {
            "manifest_path": str(ppone_manifest_path),
            "manifest_sha256": file_sha256(ppone_manifest_path),
            "train_path": str(ppone_train_path),
            "train_sha256": file_sha256(ppone_train_path),
            "source_revision": PPONE_REVISION,
            "source_rows_converted": len(ppone_sft),
            "over_length_rows_excluded": len(ppone_over_length),
            "individual_scam_ground_truth": ppone_manifest["policy"][
                "individual_scam_ground_truth"
            ],
        },
        "token_length_filter": {
            "enforced": tokenizer is not None,
            "model": QWEN_MODEL if tokenizer is not None else None,
            "revision": QWEN_REVISION if tokenizer is not None else None,
            "max_length": max_length,
            "final_over_length_rows_excluded": len(final_over_length),
            "final_over_length_ids_sha256": identity_sha(list(final_excluded)),
        },
        "overlap": {
            "near_overlap_radius": 6,
            "new_training_overlap_with_held_or_dev_rows": 0,
            "references": reference_contracts,
        },
        "splits": {
            "train": {
                "rows": len(train),
                "families": len({family_key(row) for row in train}),
                "sha256": file_sha256(train_path),
                "verdicts": dict(sorted(verdict_counts.items())),
                "sources": dict(sorted(source_counts.items())),
            },
            "dev": {
                "rows": len(parent_dev),
                "families": len({family_key(row) for row in parent_dev}),
                "sha256": file_sha256(dev_path),
                "byte_identical_to_parent": file_sha256(dev_path)
                == file_sha256(parent / "qwen_sft/dev.jsonl"),
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_sft / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "input_directory": str(output),
                "input_manifest_sha256": file_sha256(manifest_path),
                "policy": {
                    "targeted_continuation_only": True,
                    "held_rows_used_for_fitting": 0,
                    "sealed_predictions_opened": False,
                },
                "splits": manifest["splits"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--ppone-manifest", type=Path, required=True)
    parser.add_argument("--ppone-train", type=Path, required=True)
    parser.add_argument("--phone-manifest", type=Path, required=True)
    parser.add_argument("--phone-train", type=Path, required=True)
    parser.add_argument("--phone-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-reference", type=Path, action="append", default=[])
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--revision", default=QWEN_REVISION)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--anchors-per-source-verdict", type=int, default=20)
    parser.add_argument("--critical-category-families", type=int, default=64)
    parser.add_argument("--phone-scam-families-per-type", type=int, default=24)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.model != QWEN_MODEL or args.revision != QWEN_REVISION:
        raise ValueError("Stage 9 requires the exact pinned Qwen model and revision")
    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    print(
        json.dumps(
            build(
                args.parent,
                args.ppone_manifest,
                args.ppone_train,
                args.phone_manifest,
                args.phone_train,
                args.phone_audit,
                args.output,
                overlap_references=tuple(args.overlap_reference),
                tokenizer=tokenizer,
                max_length=args.max_length,
                anchors_per_source_verdict=args.anchors_per_source_verdict,
                critical_category_families=args.critical_category_families,
                phone_scam_families_per_type=args.phone_scam_families_per_type,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
