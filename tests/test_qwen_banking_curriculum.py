from __future__ import annotations

import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.build_banking77_safe import (
    LICENSE_NAME,
    SOURCE,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
)
from scripts.build_qwen_banking_curriculum import (
    HARD_INTENTS,
    PHONE_SOURCE,
    POLICY_SOURCE,
    REAL_CALL_SOURCE,
    build,
    validate_banking_source,
)
from tests.test_qwen_ppone_curriculum import fixture, qwen_row, write_jsonl


def prepare_parent(tmp_path: Path) -> Path:
    parent, *_ = fixture(tmp_path)
    train_path = parent / "qwen_sft/train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    rows.extend(
        [
            qwen_row(
                "policy-safe",
                "policy-family",
                "SAFE",
                "Open the official app yourself",
                source=POLICY_SOURCE,
            ),
            qwen_row(
                "policy-uncertain",
                "policy-family",
                "UNCERTAIN",
                "A recorded account notice needs checking",
                source=POLICY_SOURCE,
            ),
            qwen_row(
                "policy-scam",
                "policy-family",
                "SCAM",
                "Pay a penalty now or face arrest",
                source=POLICY_SOURCE,
                category="GOVERNMENT_LEGAL",
            ),
            qwen_row(
                "real-call-scam",
                "real-call-family",
                "SCAM",
                "Transfer the money while I stay on the line",
                source=REAL_CALL_SOURCE,
                category="FINANCIAL_IMPERSONATION",
            ),
            qwen_row(
                "phone-safe",
                "phone-safe-family",
                "SAFE",
                "Your appointment is confirmed",
                source=PHONE_SOURCE,
            ),
            qwen_row(
                "phone-scam",
                "phone-scam-family",
                "SCAM",
                "Read the verification code to me now",
                source=PHONE_SOURCE,
                category="CREDENTIAL_MFA",
            ),
        ]
    )
    write_jsonl(train_path, rows)
    parent_manifest_path = parent / "manifest.json"
    manifest = json.loads(parent_manifest_path.read_text())
    manifest["splits"]["train"]["sha256"] = file_sha256(train_path)
    parent_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (parent / "qwen_sft/manifest.json").write_text(
        json.dumps({"input_manifest_sha256": file_sha256(parent_manifest_path)}),
        encoding="utf-8",
    )
    return parent


def banking_row(identifier: str, intent: str, split: str, text: str) -> dict[str, object]:
    return {
        "id": identifier,
        "family_id": f"family-{identifier}",
        "source": SOURCE,
        "source_label": intent,
        "source_intent": intent,
        "license": LICENSE_NAME,
        "label": "SAFE",
        "category": "NONE",
        "text": text,
        "split": split,
        "is_synthetic": False,
        "individual_safe_ground_truth": False,
    }


def banking_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    intents = [*HARD_INTENTS, *(f"intent_{index:02d}" for index in range(72))]
    fit = tmp_path / "banking77_fit.jsonl"
    validation = tmp_path / "banking77_validation.jsonl"
    fit_rows = [
        banking_row(
            f"fit-{index:02d}",
            intent,
            "banking77_fit",
            f"Customer request {index:02d} regarding {intent} through the official bank",
        )
        for index, intent in enumerate(intents)
    ]
    validation_rows = [
        banking_row(
            "validation-only",
            "intent_00",
            "banking77_validation",
            "A separate card service question for validation only",
        )
    ]
    write_jsonl(fit, fit_rows)
    write_jsonl(validation, validation_rows)
    manifest_path = tmp_path / "banking-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_schema_version": 1,
                "source": {
                    "repository": SOURCE_REPOSITORY,
                    "revision": SOURCE_REVISION,
                    "license": LICENSE_NAME,
                },
                "policy": {
                    "validation_used_for_fitting": False,
                    "validation_used_for_threshold": False,
                    "official_test_predictions_opened": False,
                    "official_test_used_for_fitting_or_selection": False,
                },
                "artifacts": {
                    "fit": {
                        "path": str(fit),
                        "rows": len(fit_rows),
                        "sha256": file_sha256(fit),
                    },
                    "validation": {
                        "path": str(validation),
                        "rows": len(validation_rows),
                        "sha256": file_sha256(validation),
                    },
                    "test_sealed": {"rows": 3, "sha256": "sealed-contract-hash"},
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, fit, validation


def test_stage13_uses_fit_only_and_preserves_retention_boundaries(tmp_path: Path) -> None:
    parent = prepare_parent(tmp_path)
    banking_manifest, banking_fit, banking_validation = banking_fixture(tmp_path)
    output = tmp_path / "stage13"
    manifest = build(
        parent,
        banking_manifest,
        banking_fit,
        banking_validation,
        output,
        general_per_intent=1,
        hard_total_per_intent=1,
        anchors_per_source_verdict=1,
        risk_category_families_per_verdict=1,
        phone_families_per_verdict=1,
    )

    rows = [
        json.loads(line)
        for line in (output / "qwen_sft/train.jsonl").read_text().splitlines()
    ]
    identifiers = {str(row["id"]) for row in rows}
    assert len([row for row in rows if row["source"] == SOURCE]) == 77
    assert "validation-only" not in identifiers
    assert {
        "policy-safe",
        "policy-uncertain",
        "policy-scam",
        "real-call-scam",
        "phone-safe",
        "phone-scam",
    } <= identifiers
    assert manifest["selection"]["banking_validation_rows_used_for_fitting"] == 0
    assert manifest["selection"]["banking_official_test_rows_read"] == 0
    assert manifest["banking77"]["selected_intents"] == 77
    assert manifest["selection"]["reason_family_counts"][
        "new_banking_safe_boundary"
    ] == 77
    assert (output / "qwen_sft/dev.jsonl").read_bytes() == (
        parent / "qwen_sft/dev.jsonl"
    ).read_bytes()


def test_stage13_rejects_drifted_sealed_test_policy(tmp_path: Path) -> None:
    manifest_path, fit, validation = banking_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["policy"]["official_test_predictions_opened"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned rights and split contract"):
        validate_banking_source(manifest_path, fit, validation)
