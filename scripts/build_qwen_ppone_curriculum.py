#!/usr/bin/env python3
"""Build the frozen Stage 8 abstention and long-call recovery curriculum."""

from __future__ import annotations

import argparse
import hashlib
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
from training.build_qwen_sft import convert_supported_rows

PARENT_KIND = "qwen_phone_generalization_stage7_curriculum"
EXPERIMENT_KIND = "qwen_ppone_abstention_stage8_curriculum"
PPONE_REPOSITORY = "wspr-ncsu/robocall-audio-dataset"
PPONE_REVISION = "5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272"
PPONE_SOURCE = "wspr_ncsu_ppone_robocalls"
CRITICAL_CATEGORIES = ("CREDENTIAL_MFA", "JOB_OPPORTUNITY")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def target(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(row["messages"][-1]["content"]))


def verdict(row: dict[str, Any]) -> str:
    return str(target(row)["verdict"])


def family_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("source", "")), str(row.get("family_id", ""))


def identity_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def validate_parent(
    parent: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = parent / "manifest.json"
    sft_manifest_path = parent / "qwen_sft/manifest.json"
    train_path = parent / "qwen_sft/train.jsonl"
    dev_path = parent / "qwen_sft/dev.jsonl"
    for path in (manifest_path, sft_manifest_path, train_path, dev_path):
        if not path.is_file():
            raise ValueError(f"missing Stage 7 artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sft_manifest = json.loads(sft_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_kind") != PARENT_KIND
        or manifest.get("release_eligible") is not False
        or manifest.get("publication_authorized") is not False
        or sft_manifest.get("input_manifest_sha256") != file_sha256(manifest_path)
        or manifest.get("splits", {}).get("train", {}).get("sha256")
        != file_sha256(train_path)
        or manifest.get("splits", {}).get("dev", {}).get("sha256")
        != file_sha256(dev_path)
    ):
        raise ValueError("parent is not the frozen non-release Stage 7 curriculum")
    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)
    train_ids = [str(row.get("id", "")) for row in train]
    dev_ids = [str(row.get("id", "")) for row in dev]
    train_families = {family_key(row) for row in train}
    dev_families = {family_key(row) for row in dev}
    if (
        not all(train_ids + dev_ids)
        or len(train_ids) != len(set(train_ids))
        or len(dev_ids) != len(set(dev_ids))
        or set(train_ids) & set(dev_ids)
        or train_families & dev_families
    ):
        raise ValueError("Stage 7 train/dev identity contract is invalid")
    return manifest, train, dev


def validate_ppone(manifest_path: Path, train_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    policy = manifest.get("policy", {})
    artifact = manifest.get("artifacts", {}).get("train", {})
    if (
        source.get("repository") != PPONE_REPOSITORY
        or source.get("revision") != PPONE_REVISION
        or "public domain" not in str(source.get("rights", "")).casefold()
        or artifact.get("sha256") != file_sha256(train_path)
        or artifact.get("rows") != sum(1 for line in train_path.open() if line.strip())
        or policy.get("individual_scam_ground_truth") is not False
        or policy.get("safe_rows_supplied_by_this_source") != 0
        or policy.get("validation_used_for_fitting_or_threshold") is not False
        or policy.get("test_prediction_sealed_until_candidate_freeze") is not True
        or policy.get("independent_human_label_review_complete") is not False
    ):
        raise ValueError("PPoNE source differs from its pinned rights and split contract")
    return manifest


def one_per_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[family_key(row)].append(row)
    return [min(group, key=lambda row: str(row["id"])) for group in groups.values()]


def parent_anchors(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("source")), verdict(row))].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        candidates = sorted(one_per_family(groups[key]), key=lambda row: str(row["id"]))
        selected.extend(candidates[:limit])
    return selected


def category_anchors(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category in CRITICAL_CATEGORIES:
        candidates = [row for row in rows if target(row).get("category") == category]
        candidates = sorted(one_per_family(candidates), key=lambda row: str(row["id"]))
        selected.extend(candidates[:limit])
    return selected


def token_filter(
    rows: list[dict[str, Any]], tokenizer: Any | None, max_length: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if tokenizer is None:
        return rows, []
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        complete = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        count = len(tokenizer.tokenizer(complete, add_special_tokens=False)["input_ids"])
        if count > max_length:
            excluded.append({"id": str(row["id"]), "full_tokens": count})
        else:
            retained.append(row)
    return retained, excluded


def add_rows(
    selected: dict[tuple[str, str], dict[str, Any]],
    reasons: defaultdict[tuple[str, str], set[str]],
    rows: list[dict[str, Any]],
    reason: str,
    *,
    replace: bool = False,
) -> None:
    for row in rows:
        key = family_key(row)
        if not all(key):
            raise ValueError(f"row {row.get('id')} has an empty source or family")
        reasons[key].add(reason)
        if key not in selected or replace:
            selected[key] = row


def build(
    parent: Path,
    ppone_manifest_path: Path,
    ppone_train_path: Path,
    phone_manifest_path: Path,
    phone_train_path: Path,
    output: Path,
    *,
    overlap_references: tuple[Path, ...] = (),
    tokenizer: Any | None = None,
    max_length: int = 640,
    anchors_per_source_verdict: int = 24,
    critical_category_families: int = 64,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite curriculum: {output}")
    if anchors_per_source_verdict < 1 or critical_category_families < 1:
        raise ValueError("Stage 8 selection limits must be positive")
    parent_manifest, parent_train, parent_dev = validate_parent(parent)
    ppone_manifest = validate_ppone(ppone_manifest_path, ppone_train_path)
    phone_manifest = json.loads(phone_manifest_path.read_text(encoding="utf-8"))
    validate_phone_manifest(phone_manifest, phone_train_path)

    ppone_raw = read_jsonl(ppone_train_path)
    if any(
        row.get("source") != PPONE_SOURCE
        or row.get("split") != "train"
        or row.get("label") not in {"SCAM", "UNCERTAIN"}
        or row.get("individual_scam_ground_truth") is not False
        for row in ppone_raw
    ):
        raise ValueError("PPoNE training rows violate the frozen label boundary")
    ppone_sft, ppone_unsupported = convert_supported_rows(ppone_raw)
    ppone_sft, ppone_over_length = token_filter(ppone_sft, tokenizer, max_length)
    if ppone_unsupported:
        raise ValueError("PPoNE contains rows unsupported by the frozen Qwen schema")

    reference_rows: list[dict[str, Any]] = []
    reference_contracts: list[dict[str, Any]] = []
    for path in overlap_references:
        rows = read_jsonl(path)
        reference_rows.extend(rows)
        reference_contracts.append(
            {"path": str(path), "rows": len(rows), "sha256": file_sha256(path)}
        )
    overlap_rows = reference_rows + sft_as_text_rows(parent_dev)
    if near_overlap_indices(ppone_raw, overlap_rows, 6):
        raise ValueError("PPoNE training rows near-overlap held or Stage 7 dev rows")

    phone_raw = read_jsonl(phone_train_path)
    phone_target_raw = [
        row
        for row in phone_raw
        if row.get("source") == PHONE_SOURCE
        and row.get("source_dialogue_type") == "refund"
        and row.get("source_length_category") == "long"
        and row.get("label") in {"SAFE", "SCAM"}
    ]
    phone_sft, phone_unsupported = convert_supported_rows(phone_target_raw)
    phone_sft, phone_over_length = token_filter(phone_sft, tokenizer, max_length)
    if not phone_sft:
        raise ValueError("long refund-call target set is empty")

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    reasons: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    anchors = parent_anchors(parent_train, anchors_per_source_verdict)
    critical = category_anchors(parent_train, critical_category_families)
    add_rows(selected, reasons, anchors, "source_verdict_anchor")
    add_rows(selected, reasons, critical, "critical_category_anchor", replace=True)
    add_rows(selected, reasons, phone_sft, "long_refund_call_contrast", replace=True)
    add_rows(selected, reasons, ppone_sft, "real_ppone_abstention", replace=True)
    train, final_over_length = token_filter(list(selected.values()), tokenizer, max_length)
    if final_over_length:
        excluded_keys = {str(row["id"]) for row in final_over_length}
        selected = {
            key: row for key, row in selected.items() if str(row["id"]) not in excluded_keys
        }
    train = sorted(selected.values(), key=lambda row: str(row["id"]))

    train_ids = [str(row["id"]) for row in train]
    dev_ids = {str(row["id"]) for row in parent_dev}
    if len(train_ids) != len(set(train_ids)) or set(train_ids) & dev_ids:
        raise ValueError("Stage 8 train IDs are not unique or cross Stage 7 dev")
    if {family_key(row) for row in train} & {family_key(row) for row in parent_dev}:
        raise ValueError("Stage 8 train families cross Stage 7 dev")

    output_sft = output / "qwen_sft"
    output_sft.mkdir(parents=True)
    train_path = output_sft / "train.jsonl"
    dev_path = output_sft / "dev.jsonl"
    write_jsonl(train_path, train)
    shutil.copy2(parent / "qwen_sft/dev.jsonl", dev_path)

    verdict_counts = Counter(verdict(row) for row in train)
    source_counts = Counter(str(row["source"]) for row in train)
    reason_counts = Counter(
        reason for key in {family_key(row) for row in train} for reason in reasons[key]
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
                "family-diverse Stage 7 anchors plus critical-category anchors, matched long "
                "refund calls, and PPoNE train-only real robocalls"
            ),
            "anchors_per_source_verdict": anchors_per_source_verdict,
            "critical_categories": list(CRITICAL_CATEGORIES),
            "critical_category_family_limit": critical_category_families,
            "reason_family_counts": dict(sorted(reason_counts.items())),
            "selected_ids_sha256": identity_sha(train_ids),
            "held_rows_used_for_fitting": 0,
            "primary_test_rows_used_for_fitting": 0,
            "bothbosu_rows_used_for_fitting": 0,
            "phone_validation_rows_used_for_fitting": 0,
            "phone_test_rows_used_for_fitting": 0,
            "ppone_validation_rows_read": 0,
            "ppone_test_rows_read": 0,
            "ppone_test_predictions_opened": False,
        },
        "ppone": {
            "manifest_path": str(ppone_manifest_path),
            "manifest_sha256": file_sha256(ppone_manifest_path),
            "train_path": str(ppone_train_path),
            "train_sha256": file_sha256(ppone_train_path),
            "source_revision": PPONE_REVISION,
            "source_rows_converted": len(ppone_sft),
            "over_length_rows_excluded": len(ppone_over_length),
            "positive_or_suspicious_source_only": ppone_manifest["policy"][
                "positive_or_suspicious_source_only"
            ],
            "individual_scam_ground_truth": False,
        },
        "phone_target": {
            "manifest_path": str(phone_manifest_path),
            "manifest_sha256": file_sha256(phone_manifest_path),
            "raw_target_rows": len(phone_target_raw),
            "converted_rows": len(phone_sft),
            "unsupported_rows_excluded": len(phone_unsupported),
            "over_length_rows_excluded": len(phone_over_length),
            "criteria": "long refund calls, both SAFE and SCAM",
        },
        "token_length_filter": {
            "enforced": tokenizer is not None,
            "model": QWEN_MODEL if tokenizer is not None else None,
            "revision": QWEN_REVISION if tokenizer is not None else None,
            "max_length": max_length,
            "final_over_length_rows_excluded": len(final_over_length),
            "final_over_length_ids_sha256": identity_sha(
                [str(row["id"]) for row in final_over_length]
            ),
        },
        "overlap": {
            "near_overlap_radius": 6,
            "ppone_train_overlap_with_held_or_dev_rows": 0,
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
                    "ppone_test_rows_read": 0,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-reference", type=Path, action="append", default=[])
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--revision", default=QWEN_REVISION)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--anchors-per-source-verdict", type=int, default=24)
    parser.add_argument("--critical-category-families", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.model != QWEN_MODEL or args.revision != QWEN_REVISION:
        raise ValueError("Stage 8 requires the exact pinned Qwen model and revision")
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
                args.output,
                overlap_references=tuple(args.overlap_reference),
                tokenizer=tokenizer,
                max_length=args.max_length,
                anchors_per_source_verdict=args.anchors_per_source_verdict,
                critical_category_families=args.critical_category_families,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
