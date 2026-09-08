from __future__ import annotations

import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.build_qwen_phone_curriculum import PHONE_SOURCE
from scripts.build_qwen_stage9_curriculum import (
    POLICY_SOURCE,
    build,
    validate_phone_audit,
)
from tests.test_qwen_ppone_curriculum import fixture, qwen_row, raw_row, write_jsonl


def prepare_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    parent, ppone_manifest, ppone_train, phone_manifest, phone_train = fixture(tmp_path)
    parent_train_path = parent / "qwen_sft/train.jsonl"
    parent_rows = [json.loads(line) for line in parent_train_path.read_text().splitlines()]
    parent_rows.extend(
        [
            qwen_row(
                "policy-safe",
                "policy-family",
                "SAFE",
                "Use the official app yourself",
                source=POLICY_SOURCE,
            ),
            qwen_row(
                "policy-uncertain",
                "policy-family",
                "UNCERTAIN",
                "A recorded offer asks you to press one",
                source=POLICY_SOURCE,
            ),
            qwen_row(
                "policy-scam",
                "policy-family",
                "SCAM",
                "Pay now or face arrest",
                source=POLICY_SOURCE,
                category="GOVERNMENT_LEGAL",
            ),
        ]
    )
    write_jsonl(parent_train_path, parent_rows)
    parent_data = json.loads((parent / "manifest.json").read_text())
    parent_data["splits"]["train"]["sha256"] = file_sha256(parent_train_path)
    (parent / "manifest.json").write_text(json.dumps(parent_data), encoding="utf-8")
    (parent / "qwen_sft/manifest.json").write_text(
        json.dumps({"input_manifest_sha256": file_sha256(parent / "manifest.json")}),
        encoding="utf-8",
    )

    phone_rows = [
        raw_row(
            "phone-flagged-safe",
            "phone-flagged-family",
            "SAFE",
            "Use this link and read the code back to us",
            source=PHONE_SOURCE,
        )
        | {"source_dialogue_type": "refund", "source_length_category": "long"},
        *[
            raw_row(
                f"phone-{dialogue_type}-scam",
                f"phone-{dialogue_type}-family",
                "SCAM",
                f"Pay the {dialogue_type} fee with gift cards immediately",
                source=PHONE_SOURCE,
            )
            | {
                "source_dialogue_type": dialogue_type,
                "source_length_category": "long",
            }
            for dialogue_type in ("refund", "ssn", "support")
        ],
    ]
    write_jsonl(phone_train, phone_rows)
    phone_data = json.loads(phone_manifest.read_text())
    phone_data["artifacts"]["train"] = {
        "rows": len(phone_rows),
        "sha256": file_sha256(phone_train),
    }
    phone_manifest.write_text(json.dumps(phone_data), encoding="utf-8")

    phone_audit = tmp_path / "phone-audit.json"
    phone_audit.write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "contains_message_text": False,
                "policy": {
                    "automatic_relabeling_allowed": False,
                    "test_inspected": False,
                },
                "inputs": [
                    {"path": str(phone_train), "sha256": file_sha256(phone_train)}
                ],
                "prediction_alignment": {"stage7": {}, "stage8b": {}},
                "findings": [
                    {
                        "id": "phone-flagged-safe",
                        "split": "train",
                        "flags": ["publisher_safe_has_high_risk_caller_action"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return (
        parent,
        ppone_manifest,
        ppone_train,
        phone_manifest,
        phone_train,
        phone_audit,
    )


def test_stage9_filters_phone_labels_and_preserves_three_way_policy(tmp_path: Path) -> None:
    inputs = prepare_fixture(tmp_path)
    output = tmp_path / "output"
    manifest = build(
        *inputs,
        output,
        anchors_per_source_verdict=1,
        critical_category_families=1,
        phone_scam_families_per_type=1,
    )

    rows = [
        json.loads(line)
        for line in (output / "qwen_sft/train.jsonl").read_text().splitlines()
    ]
    identifiers = {row["id"] for row in rows}
    assert "phone-flagged-safe" not in identifiers
    assert {
        "phone-refund-scam",
        "phone-ssn-scam",
        "phone-support-scam",
        "policy-safe",
        "policy-uncertain",
        "policy-scam",
        "ppone-scam",
        "ppone-uncertain",
    } <= identifiers
    assert manifest["phone_quality"]["automatic_relabeling"] is False
    assert manifest["selection"]["phone_validation_rows_used_for_fitting"] == 0
    assert manifest["selection"]["sealed_predictions_opened"] is False
    assert (output / "qwen_sft/dev.jsonl").read_bytes() == (
        inputs[0] / "qwen_sft/dev.jsonl"
    ).read_bytes()


def test_stage9_rejects_drifted_phone_audit(tmp_path: Path) -> None:
    *_, phone_train, phone_audit = prepare_fixture(tmp_path)
    report = json.loads(phone_audit.read_text())
    report["policy"]["test_inspected"] = True
    phone_audit.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen text-free contract"):
        validate_phone_audit(phone_audit, phone_train)
