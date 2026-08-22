#!/usr/bin/env python3
"""Apply the frozen schema-23 quality gates without opening sealed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(
    "configs/encoder-schema23-evidencecompact-ret4-aw05-vw025-lr2e6-right.json"
)
EXPECTED_MULTIDOGO_CALL_DOMAINS = {
    "airline",
    "fastfood",
    "finance",
    "insurance",
    "media",
    "software",
}


def nested(mapping: dict[str, Any], *path: str) -> float:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing result metric: {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, int | float):
        raise ValueError(f"non-numeric result metric: {'.'.join(path)}")
    return float(value)


def evaluate_gates(config: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    quality = config["quality_acceptance"]
    gates: list[dict[str, Any]] = []

    def minimum(name: str, actual: float, threshold_key: str) -> None:
        required = float(quality[threshold_key])
        gates.append(
            {
                "name": name,
                "actual": actual,
                "operator": ">=",
                "required": required,
                "passed": actual >= required,
            }
        )

    def maximum(name: str, actual: float, threshold_key: str) -> None:
        required = float(quality[threshold_key])
        gates.append(
            {
                "name": name,
                "actual": actual,
                "operator": "<=",
                "required": required,
                "passed": actual <= required,
            }
        )

    minimum(
        "development recall",
        nested(report, "dev", "binary_safety", "scam_recall"),
        "development_recall_min",
    )
    maximum(
        "development SAFE FPR",
        nested(report, "dev", "binary_safety", "false_positive_rate"),
        "development_fpr_max",
    )
    minimum(
        "unchanged regression recall",
        nested(report, "test", "binary_safety", "scam_recall"),
        "regression_recall_min",
    )
    maximum(
        "unchanged regression SAFE FPR",
        nested(report, "test", "binary_safety", "false_positive_rate"),
        "regression_fpr_max",
    )

    for prefix, report_key, verdict_prefix, action_prefix in (
        (
            "original state",
            "call_state_validation",
            "original_state",
            "original_state_action",
        ),
        (
            "FTC pattern holdout",
            "ftc_pattern_validation",
            "ftc_pattern",
            "ftc_pattern_action",
        ),
        (
            "held-domain MultiDoGO state",
            "multidogo_state_validation",
            "multidogo_state",
            "multidogo_action",
        ),
    ):
        minimum(
            f"{prefix} harmful recall",
            nested(
                report,
                report_key,
                "state_verdict_metrics",
                "by_state",
                "harmful_scam",
                "threshold_scam_rate",
            ),
            f"{verdict_prefix}_harmful_scam_recall_min",
        )
        maximum(
            f"{prefix} routine SAFE FPR",
            nested(
                report,
                report_key,
                "state_verdict_metrics",
                "by_state",
                "routine_safe",
                "threshold_scam_rate",
            ),
            f"{verdict_prefix}_routine_safe_fpr_max",
        )
        maximum(
            f"{prefix} verified SAFE FPR",
            nested(
                report,
                report_key,
                "state_verdict_metrics",
                "by_state",
                "verified_safe",
                "threshold_scam_rate",
            ),
            f"{verdict_prefix}_verified_safe_fpr_max",
        )
        maximum(
            f"{prefix} unresolved scam rate",
            nested(
                report,
                report_key,
                "state_verdict_metrics",
                "by_state",
                "unresolved",
                "threshold_scam_rate",
            ),
            f"{verdict_prefix}_unresolved_scam_rate_max",
        )
        minimum(
            f"{prefix} ordered contrasts",
            nested(report, report_key, "state_verdict_metrics", "ordered_contrast_rate"),
            f"{verdict_prefix}_ordered_contrast_rate_min",
        )
        minimum(
            f"{prefix} action macro AUC",
            nested(report, report_key, "action_target_metrics", "macro_roc_auc"),
            f"{action_prefix}_macro_auc_min",
        )
        minimum(
            f"{prefix} calibrated action exact match",
            nested(
                report,
                report_key,
                "action_target_metrics",
                "exact_match_at_calibrated",
            ),
            f"{action_prefix}_exact_match_min",
        )

    maximum(
        "MultiDoGO original-call SAFE FPR",
        nested(
            report,
            "multidogo_call_validation",
            "binary_safety",
            "false_positive_rate",
        ),
        "multidogo_call_validation_fpr_max",
    )
    domains = report.get("multidogo_call_validation", {}).get("by_source_domain")
    if not isinstance(domains, dict) or set(domains) != EXPECTED_MULTIDOGO_CALL_DOMAINS:
        raise ValueError("MultiDoGO per-domain metrics differ from the six-domain contract")
    for domain, values in sorted(domains.items()):
        maximum(
            f"MultiDoGO {domain} SAFE FPR",
            nested(values, "binary_safety", "false_positive_rate"),
            "multidogo_call_domain_fpr_max",
        )

    maximum(
        "long-call SAFE FPR",
        nested(report, "call_window_validation", "binary_safety", "false_positive_rate"),
        "call_window_validation_fpr_max",
    )
    maximum(
        "Taskmaster SAFE FPR",
        nested(report, "taskmaster_validation", "binary_safety", "false_positive_rate"),
        "taskmaster_selection_fpr_max",
    )
    minimum(
        "prior-open BothBosu recall",
        nested(report, "scam_dialogue_validation", "binary_safety", "scam_recall"),
        "bothbosu_recall_min",
    )
    maximum(
        "prior-open BothBosu SAFE FPR",
        nested(
            report,
            "scam_dialogue_validation",
            "binary_safety",
            "false_positive_rate",
        ),
        "bothbosu_fpr_max",
    )

    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "experiment_id": config["experiment_id"],
        "quality_status": "passed" if passed else "rejected",
        "passed_gates": sum(bool(gate["passed"]) for gate in gates),
        "total_gates": len(gates),
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "gates": gates,
        "external_selection_authorized": passed,
        "distillation_or_export_authorized": passed,
        "sealed_evaluation_authorized": False,
        "runtime_status": "not evaluated until quality passes",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report_path = args.report or Path(config["outputs"]["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = evaluate_gates(config, report)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["quality_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
