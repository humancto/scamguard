#!/usr/bin/env python3
"""Build Stage 10 matched-evidence retention curriculum from the Stage 7 reference."""

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
    parent_anchors,
    read_jsonl,
    token_filter,
    validate_parent,
    validate_ppone,
    verdict,
    write_jsonl,
)
from scripts.build_qwen_stage9_curriculum import (
    POLICY_SOURCE,
    add_rows,
    phone_scam_targets,
    validate_phone_audit,
)
from scripts.generate_government_evidence_contrasts import (
    LICENSE as CONTRAST_LICENSE,
)
from scripts.generate_government_evidence_contrasts import (
    SOURCE as CONTRAST_SOURCE,
)
from training.build_qwen_sft import convert_supported_rows

EXPERIMENT_KIND = "qwen_government_evidence_stage10_curriculum"
LONG_CALL_SOURCE = "scamguard_synthetic_long_call_action_states_v1"
DIALOGUE_SOURCE = "scamguard_synthetic_dialogue_v2"
PERSISTENCE_SOURCE = "scamguard_synthetic_evidence_persistence_v1"


def validate_contrasts(manifest_path: Path, rows_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(rows_path)
    if (
        manifest.get("artifact_schema_version") != 1
        or manifest.get("source") != CONTRAST_SOURCE
        or manifest.get("license") != CONTRAST_LICENSE
        or manifest.get("used_for_fitting") is not True
        or manifest.get("used_for_threshold") is not False
        or manifest.get("held_rows_copied") != 0
        or manifest.get("evaluation_rows_read_by_generator") != 0
        or manifest.get("sha256") != file_sha256(rows_path)
        or manifest.get("rows") != len(rows)
        or manifest.get("labels") != {"SAFE": 32, "SCAM": 32, "UNCERTAIN": 32}
        or any(
            row.get("source") != CONTRAST_SOURCE
            or row.get("split") != "train"
            or row.get("license") != CONTRAST_LICENSE
            or row.get("is_synthetic") is not True
            for row in rows
        )
    ):
        raise ValueError("government evidence contrasts differ from the frozen contract")
    return manifest


def government_replay(
    rows: list[dict[str, Any]], long_call_family_limit: int
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if (
            row.get("source") in {DIALOGUE_SOURCE, PERSISTENCE_SOURCE}
            and "government_case" in str(row.get("family_id"))
        )
    ]
    long_groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("source") == LONG_CALL_SOURCE:
            long_groups[family_key(row)].append(row)
    qualifying = [
        group
        for group in long_groups.values()
        if any(
            json.loads(str(row["messages"][-1]["content"])).get("category")
            == "GOVERNMENT_LEGAL"
            for row in group
        )
    ]
    qualifying.sort(key=lambda group: family_key(group[0]))
    for group in qualifying[:long_call_family_limit]:
        selected.extend(group)
    if not selected or not qualifying:
        raise ValueError("Stage 7 lacks matched government evidence replay families")
    return selected


def build(
    parent: Path,
    ppone_manifest_path: Path,
    ppone_train_path: Path,
    phone_manifest_path: Path,
    phone_train_path: Path,
    phone_audit_path: Path,
    contrast_manifest_path: Path,
    contrast_rows_path: Path,
    output: Path,
    *,
    overlap_references: tuple[Path, ...] = (),
    tokenizer: Any | None = None,
    max_length: int = 640,
    anchors_per_source_verdict: int = 12,
    phone_scam_families_per_type: int = 24,
    long_call_government_families: int = 48,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite curriculum: {output}")
    if min(
        anchors_per_source_verdict,
        phone_scam_families_per_type,
        long_call_government_families,
    ) < 1:
        raise ValueError("Stage 10 selection limits must be positive")

    parent_manifest, parent_train, parent_dev = validate_parent(parent)
    ppone_manifest = validate_ppone(ppone_manifest_path, ppone_train_path)
    phone_manifest = json.loads(phone_manifest_path.read_text(encoding="utf-8"))
    validate_phone_manifest(phone_manifest, phone_train_path)
    phone_audit, excluded_phone_ids = validate_phone_audit(
        phone_audit_path, phone_train_path
    )
    contrast_manifest = validate_contrasts(contrast_manifest_path, contrast_rows_path)

    references: list[dict[str, Any]] = []
    reference_contracts: list[dict[str, Any]] = []
    for path in overlap_references:
        rows = read_jsonl(path)
        references.extend(rows)
        reference_contracts.append(
            {"path": str(path), "rows": len(rows), "sha256": file_sha256(path)}
        )
    overlap_rows = references + sft_as_text_rows(parent_dev)

    contrast_raw = read_jsonl(contrast_rows_path)
    if near_overlap_indices(contrast_raw, overlap_rows, 6):
        raise ValueError("government evidence contrasts near-overlap held or dev rows")
    contrast_sft, contrast_unsupported = convert_supported_rows(contrast_raw)
    contrast_sft, contrast_over_length = token_filter(
        contrast_sft, tokenizer, max_length
    )
    if contrast_unsupported or contrast_over_length or len(contrast_sft) != len(contrast_raw):
        raise ValueError("government evidence contrasts fail conversion or token limits")

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
        raise ValueError("PPoNE training rows near-overlap held or dev rows")
    ppone_sft, ppone_unsupported = convert_supported_rows(ppone_raw)
    ppone_sft, ppone_over_length = token_filter(ppone_sft, tokenizer, max_length)
    if ppone_unsupported:
        raise ValueError("PPoNE contains unsupported rows")

    phone_raw = read_jsonl(phone_train_path)
    phone_target_raw = phone_scam_targets(
        phone_raw, excluded_phone_ids, phone_scam_families_per_type
    )
    if near_overlap_indices(phone_target_raw, overlap_rows, 6):
        raise ValueError("phone targets near-overlap held or dev rows")
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
    policy_rows = [row for row in parent_train if row.get("source") == POLICY_SOURCE]
    if not policy_rows or {verdict(row) for row in policy_rows} != {
        "SAFE",
        "SCAM",
        "UNCERTAIN",
    }:
        raise ValueError("parent lacks the complete action-state policy source")
    add_rows(selected, reasons, policy_rows, "complete_action_state_policy")
    add_rows(
        selected,
        reasons,
        government_replay(parent_train, long_call_government_families),
        "matched_government_replay",
    )
    add_rows(selected, reasons, contrast_sft, "new_government_evidence_contrast")
    add_rows(selected, reasons, phone_sft, "quality_filtered_long_phone_scam")
    add_rows(selected, reasons, ppone_sft, "real_ppone_abstention")

    train, final_over_length = token_filter(list(selected.values()), tokenizer, max_length)
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
        or {family_key(row) for row in train} & {family_key(row) for row in parent_dev}
    ):
        raise ValueError("Stage 10 train identities violate the frozen boundary")

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
                "Stage 7 retention, complete action states, matched government evidence "
                "replay and new contrasts, filtered long phone scams, and PPoNE train rows"
            ),
            "anchors_per_source_verdict": anchors_per_source_verdict,
            "phone_scam_family_limit_per_type": phone_scam_families_per_type,
            "long_call_government_family_limit": long_call_government_families,
            "reason_row_counts": dict(sorted(reason_counts.items())),
            "selected_ids_sha256": identity_sha(train_ids),
            "held_rows_used_for_fitting": 0,
            "phone_validation_rows_used_for_fitting": 0,
            "phone_test_rows_read": 0,
            "ppone_validation_rows_used_for_fitting": 0,
            "ppone_test_rows_read": 0,
            "sealed_predictions_opened": False,
            "open_stage9_errors_informed_abstract_design": True,
            "open_stage9_rows_copied_or_transformed": 0,
        },
        "contrast_source": {
            "manifest_path": str(contrast_manifest_path),
            "manifest_sha256": file_sha256(contrast_manifest_path),
            "rows_path": str(contrast_rows_path),
            "rows_sha256": file_sha256(contrast_rows_path),
            "rows": len(contrast_sft),
            "families": contrast_manifest["families"],
            "labels": contrast_manifest["labels"],
            "held_rows_copied": contrast_manifest["held_rows_copied"],
        },
        "phone_quality": {
            "audit_path": str(phone_audit_path),
            "audit_sha256": file_sha256(phone_audit_path),
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
    parser.add_argument("--contrast-manifest", type=Path, required=True)
    parser.add_argument("--contrast-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-reference", type=Path, action="append", default=[])
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--revision", default=QWEN_REVISION)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--anchors-per-source-verdict", type=int, default=12)
    parser.add_argument("--phone-scam-families-per-type", type=int, default=24)
    parser.add_argument("--long-call-government-families", type=int, default=48)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.model != QWEN_MODEL or args.revision != QWEN_REVISION:
        raise ValueError("Stage 10 requires the exact pinned Qwen model and revision")
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
                args.contrast_manifest,
                args.contrast_rows,
                args.output,
                overlap_references=tuple(args.overlap_reference),
                tokenizer=tokenizer,
                max_length=args.max_length,
                anchors_per_source_verdict=args.anchors_per_source_verdict,
                phone_scam_families_per_type=args.phone_scam_families_per_type,
                long_call_government_families=args.long_call_government_families,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
