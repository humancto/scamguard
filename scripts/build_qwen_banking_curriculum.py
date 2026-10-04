#!/usr/bin/env python3
"""Build the split-safe Stage 13 BANKING77 boundary curriculum."""

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
from scripts.build_banking77_safe import (
    LICENSE_NAME as BANKING_LICENSE,
)
from scripts.build_banking77_safe import (
    SOURCE as BANKING_SOURCE,
)
from scripts.build_banking77_safe import (
    SOURCE_REPOSITORY as BANKING_REPOSITORY,
)
from scripts.build_banking77_safe import (
    SOURCE_REVISION as BANKING_REVISION,
)
from scripts.build_qwen_phone_curriculum import (
    QWEN_MODEL,
    QWEN_REVISION,
    sft_as_text_rows,
)
from scripts.build_qwen_ppone_curriculum import (
    family_key,
    identity_sha,
    one_per_family,
    parent_anchors,
    read_jsonl,
    token_filter,
    validate_parent,
    verdict,
    write_jsonl,
)
from scripts.build_qwen_stage9_curriculum import POLICY_SOURCE, add_rows
from training.build_qwen_sft import convert_supported_rows

EXPERIMENT_KIND = "qwen_banking_boundary_stage13_curriculum"
PHONE_SOURCE = "shakeleoatmeal_phone_scam_synthetic"
REAL_CALL_SOURCE = "youtube_scam_calls_cc0"
HARD_INTENTS = (
    "cancel_transfer",
    "cash_withdrawal_not_recognised",
    "pin_blocked",
    "request_refund",
    "top_up_reverted",
)
RISK_CATEGORIES = (
    "CREDENTIAL_MFA",
    "FINANCIAL_IMPERSONATION",
    "GOVERNMENT_LEGAL",
    "PAYMENT_INVOICE",
)
SELECTION_SEED = "scamguard-qwen35-08b-banking-boundary-stage13-v1"


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def target(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(row["messages"][-1]["content"]))


def validate_banking_source(
    manifest_path: Path, fit_path: Path, validation_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind the fit pool and open validation boundary without reading sealed test text."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fit = read_jsonl(fit_path)
    validation = read_jsonl(validation_path)
    source = manifest.get("source", {})
    policy = manifest.get("policy", {})
    artifacts = manifest.get("artifacts", {})
    fit_contract = artifacts.get("fit", {})
    validation_contract = artifacts.get("validation", {})
    sealed_contract = artifacts.get("test_sealed", {})
    if (
        manifest.get("dataset_schema_version") != 1
        or source.get("repository") != BANKING_REPOSITORY
        or source.get("revision") != BANKING_REVISION
        or source.get("license") != BANKING_LICENSE
        or policy.get("validation_used_for_fitting") is not False
        or policy.get("validation_used_for_threshold") is not False
        or policy.get("official_test_predictions_opened") is not False
        or policy.get("official_test_used_for_fitting_or_selection") is not False
        or fit_contract.get("path") != str(fit_path)
        or fit_contract.get("rows") != len(fit)
        or fit_contract.get("sha256") != file_sha256(fit_path)
        or validation_contract.get("path") != str(validation_path)
        or validation_contract.get("rows") != len(validation)
        or validation_contract.get("sha256") != file_sha256(validation_path)
        or not isinstance(sealed_contract.get("sha256"), str)
        or sealed_contract.get("rows", 0) < 1
    ):
        raise ValueError("BANKING77 differs from its pinned rights and split contract")
    if any(
        row.get("source") != BANKING_SOURCE
        or row.get("split") != "banking77_fit"
        or row.get("label") != "SAFE"
        or row.get("license") != BANKING_LICENSE
        or row.get("individual_safe_ground_truth") is not False
        for row in fit
    ):
        raise ValueError("BANKING77 fit rows violate the weak-SAFE source contract")
    if any(
        row.get("source") != BANKING_SOURCE
        or row.get("split") != "banking77_validation"
        for row in validation
    ):
        raise ValueError("BANKING77 validation rows violate the open boundary contract")
    if (
        {str(row["id"]) for row in fit} & {str(row["id"]) for row in validation}
        or {family_key(row) for row in fit} & {family_key(row) for row in validation}
    ):
        raise ValueError("BANKING77 fit and validation identities overlap")
    return manifest, fit, validation


def select_banking_rows(
    rows: list[dict[str, Any]], general_per_intent: int, hard_total_per_intent: int
) -> list[dict[str, Any]]:
    if general_per_intent < 1 or hard_total_per_intent < general_per_intent:
        raise ValueError("invalid BANKING77 per-intent selection limits")
    by_intent: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intent[str(row.get("source_intent", ""))].append(row)
    if len(by_intent) != 77 or not set(HARD_INTENTS) <= set(by_intent):
        raise ValueError("BANKING77 fit pool lacks the frozen 77-intent inventory")
    selected: list[dict[str, Any]] = []
    for intent, candidates in sorted(by_intent.items()):
        limit = hard_total_per_intent if intent in HARD_INTENTS else general_per_intent
        ordered = sorted(
            one_per_family(candidates),
            key=lambda row: stable_hash(f"{SELECTION_SEED}:{intent}:{row['id']}"),
        )
        if len(ordered) < limit:
            raise ValueError(f"too few BANKING77 fit families for {intent}")
        selected.extend(ordered[:limit])
    return selected


def grouped_parent_rows(
    rows: list[dict[str, Any]], *, group_field: str, values: tuple[str, ...], limit: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for value in values:
        candidates = [
            row
            for row in rows
            if str(target(row).get(group_field, "")) == value
        ]
        by_verdict: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_verdict[verdict(row)].append(row)
        for label in sorted(by_verdict):
            ordered = sorted(
                one_per_family(by_verdict[label]), key=lambda row: str(row["id"])
            )
            selected.extend(ordered[:limit])
    return selected


def source_verdict_rows(
    rows: list[dict[str, Any]], source: str, per_verdict: int
) -> list[dict[str, Any]]:
    by_verdict: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("source") == source:
            by_verdict[verdict(row)].append(row)
    selected: list[dict[str, Any]] = []
    for label in sorted(by_verdict):
        ordered = sorted(
            one_per_family(by_verdict[label]), key=lambda row: str(row["id"])
        )
        selected.extend(ordered[:per_verdict])
    return selected


def build(
    parent: Path,
    banking_manifest_path: Path,
    banking_fit_path: Path,
    banking_validation_path: Path,
    output: Path,
    *,
    overlap_references: tuple[Path, ...] = (),
    tokenizer: Any | None = None,
    max_length: int = 640,
    general_per_intent: int = 4,
    hard_total_per_intent: int = 20,
    anchors_per_source_verdict: int = 12,
    risk_category_families_per_verdict: int = 24,
    phone_families_per_verdict: int = 48,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite curriculum: {output}")
    if min(
        anchors_per_source_verdict,
        risk_category_families_per_verdict,
        phone_families_per_verdict,
    ) < 1:
        raise ValueError("Stage 13 retention limits must be positive")

    parent_manifest, parent_train, parent_dev = validate_parent(parent)
    banking_manifest, banking_fit, banking_validation = validate_banking_source(
        banking_manifest_path, banking_fit_path, banking_validation_path
    )
    banking_raw = select_banking_rows(
        banking_fit, general_per_intent, hard_total_per_intent
    )

    references = sft_as_text_rows(parent_dev) + banking_validation
    reference_contracts: list[dict[str, Any]] = [
        {
            "path": str(banking_validation_path),
            "purpose": "open selection boundary audit only",
            "rows": len(banking_validation),
            "sha256": file_sha256(banking_validation_path),
        }
    ]
    for path in overlap_references:
        rows = read_jsonl(path)
        references.extend(rows)
        reference_contracts.append(
            {"path": str(path), "rows": len(rows), "sha256": file_sha256(path)}
        )
    if near_overlap_indices(banking_raw, references, 6):
        raise ValueError("selected BANKING77 fit rows near-overlap held or dev rows")

    banking_sft, banking_unsupported = convert_supported_rows(banking_raw)
    banking_sft, banking_over_length = token_filter(
        banking_sft, tokenizer, max_length
    )
    if banking_unsupported or banking_over_length or len(banking_sft) != len(banking_raw):
        raise ValueError("selected BANKING77 rows fail conversion or token limits")

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
        grouped_parent_rows(
            parent_train,
            group_field="category",
            values=RISK_CATEGORIES,
            limit=risk_category_families_per_verdict,
        ),
        "high_risk_category_retention",
    )
    real_calls = [row for row in parent_train if row.get("source") == REAL_CALL_SOURCE]
    if not real_calls:
        raise ValueError("parent lacks real-call retention rows")
    add_rows(selected, reasons, real_calls, "real_call_retention")
    phone_rows = source_verdict_rows(
        parent_train, PHONE_SOURCE, phone_families_per_verdict
    )
    if {verdict(row) for row in phone_rows} != {"SAFE", "SCAM"}:
        raise ValueError("parent lacks balanced phone SAFE/SCAM retention rows")
    add_rows(selected, reasons, phone_rows, "phone_boundary_retention")
    add_rows(selected, reasons, banking_sft, "new_banking_safe_boundary")

    train, final_over_length = token_filter(
        list(selected.values()), tokenizer, max_length
    )
    if final_over_length:
        raise ValueError("retention rows exceed the frozen token limit")
    train = sorted(train, key=lambda row: str(row["id"]))
    train_ids = [str(row["id"]) for row in train]
    dev_ids = {str(row["id"]) for row in parent_dev}
    validation_ids = {str(row["id"]) for row in banking_validation}
    if (
        len(train_ids) != len(set(train_ids))
        or set(train_ids) & dev_ids
        or set(train_ids) & validation_ids
        or {family_key(row) for row in train} & {family_key(row) for row in parent_dev}
        or {family_key(row) for row in train}
        & {family_key(row) for row in banking_validation}
    ):
        raise ValueError("Stage 13 train identities violate the frozen boundary")

    output_sft = output / "qwen_sft"
    output_sft.mkdir(parents=True)
    train_path = output_sft / "train.jsonl"
    dev_path = output_sft / "dev.jsonl"
    write_jsonl(train_path, train)
    shutil.copy2(parent / "qwen_sft/dev.jsonl", dev_path)

    train_families = {family_key(row) for row in train}
    reason_counts = Counter(
        reason for identifier in train_ids for reason in reasons[identifier]
    )
    banking_intent_counts = Counter(str(row["source_intent"]) for row in banking_raw)
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
                "small intent-balanced BANKING77 SAFE dose plus Stage 7 source/verdict, "
                "action-state, high-risk-category, real-call, and phone-boundary retention"
            ),
            "seed": SELECTION_SEED,
            "general_rows_per_intent": general_per_intent,
            "hard_intents": list(HARD_INTENTS),
            "hard_total_rows_per_intent": hard_total_per_intent,
            "anchors_per_source_verdict": anchors_per_source_verdict,
            "risk_categories": list(RISK_CATEGORIES),
            "risk_category_families_per_verdict": risk_category_families_per_verdict,
            "phone_families_per_verdict": phone_families_per_verdict,
            "reason_family_counts": dict(sorted(reason_counts.items())),
            "selected_ids_sha256": identity_sha(train_ids),
            "held_rows_used_for_fitting": 0,
            "banking_validation_rows_used_for_fitting": 0,
            "banking_validation_rows_used_for_threshold": 0,
            "banking_validation_rows_read_for_overlap_audit": len(banking_validation),
            "banking_validation_errors_informed_abstract_intent_oversampling": True,
            "banking_validation_rows_copied_or_transformed": 0,
            "banking_official_test_rows_read": 0,
            "banking_official_test_predictions_opened": False,
        },
        "banking77": {
            "manifest_path": str(banking_manifest_path),
            "manifest_sha256": file_sha256(banking_manifest_path),
            "fit_path": str(banking_fit_path),
            "fit_sha256": file_sha256(banking_fit_path),
            "source_revision": BANKING_REVISION,
            "license": BANKING_LICENSE,
            "fit_pool_rows": len(banking_fit),
            "selected_rows": len(banking_sft),
            "selected_intents": len(banking_intent_counts),
            "selected_intent_counts": dict(sorted(banking_intent_counts.items())),
            "weak_safe_supervision": True,
            "individual_safe_ground_truth": False,
            "sealed_test_contract_sha256": banking_manifest["artifacts"]["test_sealed"][
                "sha256"
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
                "families": len(train_families),
                "sha256": file_sha256(train_path),
                "verdicts": dict(sorted(Counter(verdict(row) for row in train).items())),
                "sources": dict(
                    sorted(Counter(str(row["source"]) for row in train).items())
                ),
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
                    "banking_validation_rows_used_for_fitting": 0,
                    "banking_official_test_rows_read": 0,
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
    parser.add_argument("--banking-manifest", type=Path, required=True)
    parser.add_argument("--banking-fit", type=Path, required=True)
    parser.add_argument("--banking-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-reference", type=Path, action="append", default=[])
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--revision", default=QWEN_REVISION)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--general-per-intent", type=int, default=4)
    parser.add_argument("--hard-total-per-intent", type=int, default=20)
    parser.add_argument("--anchors-per-source-verdict", type=int, default=12)
    parser.add_argument("--risk-category-families-per-verdict", type=int, default=24)
    parser.add_argument("--phone-families-per-verdict", type=int, default=48)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.model != QWEN_MODEL or args.revision != QWEN_REVISION:
        raise ValueError("Stage 13 requires the exact pinned Qwen model and revision")
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
                args.banking_manifest,
                args.banking_fit,
                args.banking_validation,
                args.output,
                overlap_references=tuple(args.overlap_reference),
                tokenizer=tokenizer,
                max_length=args.max_length,
                general_per_intent=args.general_per_intent,
                hard_total_per_intent=args.hard_total_per_intent,
                anchors_per_source_verdict=args.anchors_per_source_verdict,
                risk_category_families_per_verdict=args.risk_category_families_per_verdict,
                phone_families_per_verdict=args.phone_families_per_verdict,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
