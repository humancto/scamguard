#!/usr/bin/env python3
"""Apply the frozen Stage 9 promotion contract with a quality-stratified phone FPR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from scripts.audit_phone_source_labels import prediction_alignment
from scripts.check_qwen_stage8_promotion import (
    MAX_DEV_FPR,
    MIN_DEV_RECALL,
    MIN_PPONE_MACRO_GAIN,
    MIN_PPONE_UNCERTAIN_RECALL,
    metric,
    uncertain_recall,
)


def phone_index(audit: dict[str, Any]) -> tuple[dict[str, str], set[str]]:
    index = audit.get("validation_index", [])
    labels: dict[str, str] = {}
    for row in index:
        if not isinstance(row, dict) or row.get("publisher_label") not in {"SAFE", "SCAM"}:
            raise ValueError("phone audit has an invalid validation index")
        identifier = str(row.get("id", ""))
        if not identifier or identifier in labels:
            raise ValueError("phone audit has duplicate or empty validation IDs")
        labels[identifier] = str(row["publisher_label"])
    high_risk = {
        str(row["id"])
        for row in audit.get("findings", [])
        if row.get("split") == "validation"
        and row.get("publisher_label") == "SAFE"
        and "publisher_safe_has_high_risk_caller_action" in row.get("flags", [])
    }
    if not labels or not high_risk or not high_risk < set(labels):
        raise ValueError("phone audit lacks a usable high-risk validation stratum")
    return labels, high_risk


def validate_ledger_binding(report: dict[str, Any], path: Path) -> None:
    ledger = report.get("prediction_ledger", {})
    if ledger.get("path") != str(path) or ledger.get("sha256") != file_sha256(path):
        raise ValueError("quality report is not bound to the supplied prediction ledger")


def check(
    candidate_path: Path,
    candidate_predictions: Path,
    stage7_full_path: Path,
    stage7_predictions: Path,
    stage7_ppone_path: Path,
    phone_audit_path: Path,
    output: Path,
) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    stage7 = json.loads(stage7_full_path.read_text(encoding="utf-8"))
    stage7_ppone = json.loads(stage7_ppone_path.read_text(encoding="utf-8"))
    audit = json.loads(phone_audit_path.read_text(encoding="utf-8"))
    if (
        candidate.get("selection_screen_only") is not True
        or candidate.get("release_gate_report") is not False
        or candidate.get("frozen_calibration_source") is None
        or audit.get("contains_message_text") is not False
        or audit.get("policy", {}).get("automatic_relabeling_allowed") is not False
        or audit.get("policy", {}).get("test_inspected") is not False
    ):
        raise ValueError("Stage 9 requires frozen selection and text-free phone-audit reports")
    required = {"dev", "phone_scam_validation", "ppone_validation"}
    if not required <= candidate.keys():
        raise ValueError("Stage 9 selection report is missing a required split")
    validate_ledger_binding(candidate, candidate_predictions)
    validate_ledger_binding(stage7, stage7_predictions)

    labels, high_risk = phone_index(audit)
    candidate_phone = prediction_alignment(candidate_predictions, labels, high_risk)
    stage7_phone = prediction_alignment(stage7_predictions, labels, high_risk)
    candidate_values = {
        "dev_recall": metric(candidate, "dev", "binary_safety", "scam_recall"),
        "dev_fpr": metric(candidate, "dev", "binary_safety", "false_positive_rate"),
        "dev_macro_f1": metric(candidate, "dev", "calibrated_decision", "macro_f1"),
        "phone_recall": metric(
            candidate, "phone_scam_validation", "binary_safety", "scam_recall"
        ),
        "phone_publisher_fpr_diagnostic": metric(
            candidate, "phone_scam_validation", "binary_safety", "false_positive_rate"
        ),
        "phone_other_safe_fpr": float(candidate_phone["other_safe_trigger_rate"]),
        "phone_high_risk_safe_alarm_rate_diagnostic": float(
            candidate_phone["high_risk_action_trigger_rate"]
        ),
        "ppone_recall": metric(
            candidate, "ppone_validation", "binary_safety", "scam_recall"
        ),
        "ppone_macro_f1": metric(
            candidate, "ppone_validation", "calibrated_decision", "macro_f1"
        ),
        "ppone_uncertain_recall": uncertain_recall(candidate, "ppone_validation"),
    }
    baseline_values = {
        "dev_macro_f1": metric(stage7, "dev", "calibrated_decision", "macro_f1"),
        "phone_recall": metric(
            stage7, "phone_scam_validation", "binary_safety", "scam_recall"
        ),
        "phone_publisher_fpr_diagnostic": metric(
            stage7, "phone_scam_validation", "binary_safety", "false_positive_rate"
        ),
        "phone_other_safe_fpr": float(stage7_phone["other_safe_trigger_rate"]),
        "phone_high_risk_safe_alarm_rate_diagnostic": float(
            stage7_phone["high_risk_action_trigger_rate"]
        ),
        "ppone_recall": metric(
            stage7_ppone, "ppone_validation", "binary_safety", "scam_recall"
        ),
        "ppone_macro_f1": metric(
            stage7_ppone, "ppone_validation", "calibrated_decision", "macro_f1"
        ),
    }
    gates = {
        "dev_recall": candidate_values["dev_recall"] >= MIN_DEV_RECALL,
        "dev_fpr": candidate_values["dev_fpr"] <= MAX_DEV_FPR,
        "dev_macro_non_regression": candidate_values["dev_macro_f1"]
        >= baseline_values["dev_macro_f1"],
        "phone_recall_non_regression": candidate_values["phone_recall"]
        >= baseline_values["phone_recall"],
        "phone_other_safe_fpr_non_regression": candidate_values["phone_other_safe_fpr"]
        <= baseline_values["phone_other_safe_fpr"],
        "ppone_recall_non_regression": candidate_values["ppone_recall"]
        >= baseline_values["ppone_recall"],
        "ppone_macro_gain": candidate_values["ppone_macro_f1"]
        >= baseline_values["ppone_macro_f1"] + MIN_PPONE_MACRO_GAIN,
        "ppone_uncertain_recall": candidate_values["ppone_uncertain_recall"]
        >= MIN_PPONE_UNCERTAIN_RECALL,
    }
    passed = all(gates.values())
    result = {
        "artifact_schema_version": 1,
        "candidate": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
        "candidate_predictions": {
            "path": str(candidate_predictions),
            "sha256": file_sha256(candidate_predictions),
        },
        "stage7_full": {
            "path": str(stage7_full_path),
            "sha256": file_sha256(stage7_full_path),
        },
        "stage7_predictions": {
            "path": str(stage7_predictions),
            "sha256": file_sha256(stage7_predictions),
        },
        "stage7_ppone": {
            "path": str(stage7_ppone_path),
            "sha256": file_sha256(stage7_ppone_path),
        },
        "phone_label_audit": {
            "path": str(phone_audit_path),
            "sha256": file_sha256(phone_audit_path),
            "high_risk_publisher_safe_rows": len(high_risk),
            "other_publisher_safe_rows": sum(label == "SAFE" for label in labels.values())
            - len(high_risk),
            "publisher_labels_changed": 0,
        },
        "frozen_thresholds": {
            "minimum_dev_recall": MIN_DEV_RECALL,
            "maximum_dev_fpr": MAX_DEV_FPR,
            "minimum_ppone_macro_f1_gain": MIN_PPONE_MACRO_GAIN,
            "minimum_ppone_uncertain_recall": MIN_PPONE_UNCERTAIN_RECALL,
            "stage7_non_regression": [
                "dev_macro_f1",
                "phone_recall",
                "phone_other_safe_fpr",
                "ppone_recall",
            ],
            "publisher_phone_fpr_is_diagnostic_only": True,
        },
        "candidate_metrics": candidate_values,
        "stage7_metrics": baseline_values,
        "phone_alignment": {"candidate": candidate_phone, "stage7": stage7_phone},
        "gates": gates,
        "passed": passed,
        "next_action": (
            "run frozen full regression without opening source tests"
            if passed
            else "reject Stage 9 candidate before full regression"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--stage7-full", type=Path, required=True)
    parser.add_argument("--stage7-predictions", type=Path, required=True)
    parser.add_argument("--stage7-ppone", type=Path, required=True)
    parser.add_argument("--phone-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(
        args.candidate,
        args.candidate_predictions,
        args.stage7_full,
        args.stage7_predictions,
        args.stage7_ppone,
        args.phone_audit,
        args.output,
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
