#!/usr/bin/env python3
"""Apply the frozen development-only Stage 8 promotion contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

MIN_DEV_RECALL = 0.97
MAX_DEV_FPR = 0.02
MIN_PPONE_MACRO_GAIN = 0.03
MIN_PPONE_UNCERTAIN_RECALL = 3 / 21


def metric(report: dict[str, Any], split: str, group: str, key: str) -> float:
    return float(report[split][group][key])


def uncertain_recall(report: dict[str, Any], split: str) -> float:
    confusion = report[split]["calibrated_decision"]["confusion"]
    if len(confusion) != 3 or any(len(row) != 3 for row in confusion):
        raise ValueError(f"{split} does not have a three-way confusion matrix")
    actual_uncertain = sum(int(value) for value in confusion[1])
    return float(confusion[1][1]) / actual_uncertain if actual_uncertain else 0.0


def check(
    candidate_path: Path,
    stage7_full_path: Path,
    stage7_ppone_path: Path,
    output: Path,
) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    stage7 = json.loads(stage7_full_path.read_text(encoding="utf-8"))
    stage7_ppone = json.loads(stage7_ppone_path.read_text(encoding="utf-8"))
    if (
        candidate.get("selection_screen_only") is not True
        or candidate.get("release_gate_report") is not False
        or candidate.get("frozen_calibration_source") is None
    ):
        raise ValueError("Stage 8 promotion requires a frozen selection-screen report")
    required = {"dev", "phone_scam_validation", "ppone_validation"}
    if not required <= candidate.keys():
        raise ValueError("Stage 8 development report is missing a required split")

    candidate_values = {
        "dev_recall": metric(candidate, "dev", "binary_safety", "scam_recall"),
        "dev_fpr": metric(candidate, "dev", "binary_safety", "false_positive_rate"),
        "dev_macro_f1": metric(candidate, "dev", "calibrated_decision", "macro_f1"),
        "phone_recall": metric(
            candidate, "phone_scam_validation", "binary_safety", "scam_recall"
        ),
        "phone_fpr": metric(
            candidate, "phone_scam_validation", "binary_safety", "false_positive_rate"
        ),
        "phone_macro_f1": metric(
            candidate, "phone_scam_validation", "calibrated_decision", "macro_f1"
        ),
        "ppone_recall": metric(
            candidate, "ppone_validation", "binary_safety", "scam_recall"
        ),
        "ppone_macro_f1": metric(
            candidate, "ppone_validation", "calibrated_decision", "macro_f1"
        ),
        "ppone_uncertain_recall": uncertain_recall(candidate, "ppone_validation"),
    }
    source_values = {
        "dev_macro_f1": metric(stage7, "dev", "calibrated_decision", "macro_f1"),
        "phone_recall": metric(
            stage7, "phone_scam_validation", "binary_safety", "scam_recall"
        ),
        "phone_fpr": metric(
            stage7, "phone_scam_validation", "binary_safety", "false_positive_rate"
        ),
        "phone_macro_f1": metric(
            stage7, "phone_scam_validation", "calibrated_decision", "macro_f1"
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
        >= source_values["dev_macro_f1"],
        "phone_recall_non_regression": candidate_values["phone_recall"]
        >= source_values["phone_recall"],
        "phone_fpr_non_regression": candidate_values["phone_fpr"]
        <= source_values["phone_fpr"],
        "phone_macro_non_regression": candidate_values["phone_macro_f1"]
        >= source_values["phone_macro_f1"],
        "ppone_recall_non_regression": candidate_values["ppone_recall"]
        >= source_values["ppone_recall"],
        "ppone_macro_gain": candidate_values["ppone_macro_f1"]
        >= source_values["ppone_macro_f1"] + MIN_PPONE_MACRO_GAIN,
        "ppone_uncertain_recall": candidate_values["ppone_uncertain_recall"]
        >= MIN_PPONE_UNCERTAIN_RECALL,
    }
    report = {
        "artifact_schema_version": 1,
        "candidate": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
        "stage7_full": {
            "path": str(stage7_full_path),
            "sha256": file_sha256(stage7_full_path),
        },
        "stage7_ppone": {
            "path": str(stage7_ppone_path),
            "sha256": file_sha256(stage7_ppone_path),
        },
        "frozen_thresholds": {
            "minimum_dev_recall": MIN_DEV_RECALL,
            "maximum_dev_fpr": MAX_DEV_FPR,
            "minimum_ppone_macro_f1_gain": MIN_PPONE_MACRO_GAIN,
            "minimum_ppone_uncertain_recall": MIN_PPONE_UNCERTAIN_RECALL,
            "all_stage7_comparisons_require_non_regression": True,
        },
        "candidate_metrics": candidate_values,
        "stage7_metrics": source_values,
        "gates": gates,
        "passed": all(gates.values()),
        "next_action": (
            "run frozen full regression without opening PPoNE test"
            if all(gates.values())
            else "reject Stage 8 candidate before full regression"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stage7-full", type=Path, required=True)
    parser.add_argument("--stage7-ppone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.candidate, args.stage7_full, args.stage7_ppone, args.output)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
