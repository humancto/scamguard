#!/usr/bin/env python3
"""Gate the Stage 13 banking-boundary candidate against frozen Stage 7 evidence."""

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
    metric,
    uncertain_recall,
)
from scripts.check_qwen_stage9_promotion import phone_index, validate_ledger_binding

MAX_BANKING_FPR = 0.005
MAX_BANKING_FPR_CI95_UPPER = 0.0125
MAX_BANKING_FALSE_POSITIVES = 3


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(
    candidate_path: Path,
    candidate_predictions: Path,
    stage7_full_path: Path,
    stage7_predictions: Path,
    stage7_ppone_path: Path,
    stage7_banking_path: Path,
    stage7_vystadial_path: Path,
    stage7_international_path: Path,
    phone_audit_path: Path,
    output: Path,
) -> dict[str, Any]:
    candidate = load(candidate_path)
    stage7 = load(stage7_full_path)
    stage7_ppone = load(stage7_ppone_path)
    stage7_banking = load(stage7_banking_path)
    stage7_vystadial = load(stage7_vystadial_path)
    stage7_international = load(stage7_international_path)
    audit = load(phone_audit_path)
    required = {
        "dev",
        "banking77_validation",
        "phone_scam_validation",
        "ppone_validation",
        "vystadial_safe",
        "international_robocalls",
    }
    if (
        candidate.get("selection_screen_only") is not True
        or candidate.get("release_gate_report") is not False
        or candidate.get("frozen_calibration_source") is None
        or not required <= candidate.keys()
        or audit.get("contains_message_text") is not False
        or audit.get("policy", {}).get("automatic_relabeling_allowed") is not False
        or audit.get("policy", {}).get("test_inspected") is not False
    ):
        raise ValueError("candidate lacks the frozen Stage 13 selection contract")
    validate_ledger_binding(candidate, candidate_predictions)
    validate_ledger_binding(stage7, stage7_predictions)
    labels, high_risk = phone_index(audit)
    candidate_phone = prediction_alignment(candidate_predictions, labels, high_risk)
    stage7_phone = prediction_alignment(stage7_predictions, labels, high_risk)

    banking_binary = candidate["banking77_validation"]["binary_safety"]
    candidate_values = {
        "dev_recall": metric(candidate, "dev", "binary_safety", "scam_recall"),
        "dev_fpr": metric(candidate, "dev", "binary_safety", "false_positive_rate"),
        "dev_macro_f1": metric(candidate, "dev", "calibrated_decision", "macro_f1"),
        "banking_fpr": float(banking_binary["false_positive_rate"]),
        "banking_fpr_ci95_upper": float(banking_binary["false_positive_rate_ci95"][1]),
        "banking_false_positives": int(banking_binary["fp"]),
        "phone_recall": metric(
            candidate, "phone_scam_validation", "binary_safety", "scam_recall"
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
        "vystadial_fpr": metric(
            candidate, "vystadial_safe", "binary_safety", "false_positive_rate"
        ),
        "international_recall": metric(
            candidate, "international_robocalls", "binary_safety", "scam_recall"
        ),
    }
    baseline_values = {
        "dev_macro_f1": metric(stage7, "dev", "calibrated_decision", "macro_f1"),
        "banking_fpr": metric(
            stage7_banking,
            "banking77_validation",
            "binary_safety",
            "false_positive_rate",
        ),
        "banking_false_positives": int(
            stage7_banking["banking77_validation"]["binary_safety"]["fp"]
        ),
        "phone_recall": metric(
            stage7, "phone_scam_validation", "binary_safety", "scam_recall"
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
        "ppone_uncertain_recall": uncertain_recall(
            stage7_ppone, "ppone_validation"
        ),
        "vystadial_fpr": metric(
            stage7_vystadial,
            "vystadial_safe",
            "binary_safety",
            "false_positive_rate",
        ),
        "international_recall": metric(
            stage7_international,
            "international_robocalls",
            "binary_safety",
            "scam_recall",
        ),
    }
    gates = {
        "dev_recall": candidate_values["dev_recall"] >= MIN_DEV_RECALL,
        "dev_fpr": candidate_values["dev_fpr"] <= MAX_DEV_FPR,
        "dev_macro_non_regression": candidate_values["dev_macro_f1"]
        >= baseline_values["dev_macro_f1"],
        "banking_fpr_absolute": candidate_values["banking_fpr"] <= MAX_BANKING_FPR,
        "banking_fpr_ci95_upper": candidate_values["banking_fpr_ci95_upper"]
        <= MAX_BANKING_FPR_CI95_UPPER,
        "banking_false_positive_budget": candidate_values["banking_false_positives"]
        <= MAX_BANKING_FALSE_POSITIVES,
        "banking_strict_improvement": candidate_values["banking_fpr"]
        < baseline_values["banking_fpr"],
        "phone_recall_non_regression": candidate_values["phone_recall"]
        >= baseline_values["phone_recall"],
        "phone_other_safe_fpr_non_regression": candidate_values["phone_other_safe_fpr"]
        <= baseline_values["phone_other_safe_fpr"],
        "ppone_recall_non_regression": candidate_values["ppone_recall"]
        >= baseline_values["ppone_recall"],
        "ppone_macro_non_regression": candidate_values["ppone_macro_f1"]
        >= baseline_values["ppone_macro_f1"],
        "ppone_uncertain_non_regression": candidate_values["ppone_uncertain_recall"]
        >= baseline_values["ppone_uncertain_recall"],
        "vystadial_fpr_non_regression": candidate_values["vystadial_fpr"]
        <= baseline_values["vystadial_fpr"],
        "international_recall_non_regression": candidate_values["international_recall"]
        >= baseline_values["international_recall"],
    }
    passed = all(gates.values())
    paths = {
        "candidate": candidate_path,
        "candidate_predictions": candidate_predictions,
        "stage7_full": stage7_full_path,
        "stage7_predictions": stage7_predictions,
        "stage7_ppone": stage7_ppone_path,
        "stage7_banking": stage7_banking_path,
        "stage7_vystadial": stage7_vystadial_path,
        "stage7_international": stage7_international_path,
        "phone_label_audit": phone_audit_path,
    }
    result: dict[str, Any] = {
        "artifact_schema_version": 1,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "frozen_thresholds": {
            "minimum_dev_recall": MIN_DEV_RECALL,
            "maximum_dev_fpr": MAX_DEV_FPR,
            "maximum_banking_fpr": MAX_BANKING_FPR,
            "maximum_banking_fpr_ci95_upper": MAX_BANKING_FPR_CI95_UPPER,
            "maximum_banking_false_positives": MAX_BANKING_FALSE_POSITIVES,
            "all_other_metrics_require_stage7_non_regression": True,
        },
        "candidate_metrics": candidate_values,
        "stage7_metrics": baseline_values,
        "phone_alignment": {"candidate": candidate_phone, "stage7": stage7_phone},
        "phone_label_policy": {
            "publisher_labels_changed": 0,
            "high_risk_publisher_safe_rows_diagnostic_only": len(high_risk),
        },
        "gates": gates,
        "passed": passed,
        "next_action": (
            "run frozen full regression without opening BANKING77 official test"
            if passed
            else "reject Stage 13 candidate before full regression"
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
    parser.add_argument("--stage7-banking", type=Path, required=True)
    parser.add_argument("--stage7-vystadial", type=Path, required=True)
    parser.add_argument("--stage7-international", type=Path, required=True)
    parser.add_argument("--phone-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(
        args.candidate,
        args.candidate_predictions,
        args.stage7_full,
        args.stage7_predictions,
        args.stage7_ppone,
        args.stage7_banking,
        args.stage7_vystadial,
        args.stage7_international,
        args.phone_audit,
        args.output,
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
