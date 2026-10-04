#!/usr/bin/env python3
"""Fail closed on development quality before any open selection split is scored."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from scripts.check_qwen_stage8_promotion import MAX_DEV_FPR, MIN_DEV_RECALL, metric


def check(candidate_path: Path, baseline_path: Path, output: Path) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        candidate.get("development_screen_only") is not True
        or candidate.get("selection_screen_only") is not False
        or candidate.get("release_gate_report") is not False
        or candidate.get("frozen_calibration_source") is not None
        or "dev" not in candidate
        or "dev" not in baseline
    ):
        raise ValueError("candidate lacks the frozen development-only contract")
    values = {
        "dev_recall": metric(candidate, "dev", "binary_safety", "scam_recall"),
        "dev_fpr": metric(candidate, "dev", "binary_safety", "false_positive_rate"),
        "dev_macro_f1": metric(candidate, "dev", "calibrated_decision", "macro_f1"),
    }
    baseline_values = {
        "dev_recall": metric(baseline, "dev", "binary_safety", "scam_recall"),
        "dev_fpr": metric(baseline, "dev", "binary_safety", "false_positive_rate"),
        "dev_macro_f1": metric(baseline, "dev", "calibrated_decision", "macro_f1"),
    }
    gates = {
        "dev_recall": values["dev_recall"] >= MIN_DEV_RECALL,
        "dev_fpr": values["dev_fpr"] <= MAX_DEV_FPR,
        "dev_macro_non_regression": values["dev_macro_f1"]
        >= baseline_values["dev_macro_f1"],
    }
    passed = all(gates.values())
    result = {
        "artifact_schema_version": 1,
        "candidate": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
        },
        "baseline": {
            "path": str(baseline_path),
            "sha256": file_sha256(baseline_path),
        },
        "frozen_thresholds": {
            "minimum_dev_recall": MIN_DEV_RECALL,
            "maximum_dev_fpr": MAX_DEV_FPR,
            "minimum_dev_macro_f1": baseline_values["dev_macro_f1"],
        },
        "candidate_metrics": values,
        "baseline_metrics": baseline_values,
        "gates": gates,
        "passed": passed,
        "next_action": (
            "score frozen open selection splits"
            if passed
            else "reject candidate without scoring open selection splits"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.candidate, args.baseline, args.output)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
