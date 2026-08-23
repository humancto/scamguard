#!/usr/bin/env python3
"""Apply the frozen Qwen3.5-0.8B schema-24 quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CORE_MIN_EXAMPLES = 20
EXPECTED_DOMAINS = {"airline", "fastfood", "finance", "insurance", "media", "software"}


def nested(mapping: dict[str, Any], *path: str) -> float:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing Qwen result metric: {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"non-numeric Qwen result metric: {'.'.join(path)}")
    return float(value)


def evaluate_gates(report: dict[str, Any]) -> dict[str, object]:
    gates: list[dict[str, object]] = []

    def minimum(name: str, actual: float, required: float) -> None:
        gates.append(
            {
                "name": name,
                "actual": actual,
                "operator": ">=",
                "required": required,
                "passed": actual >= required,
            }
        )

    def maximum(name: str, actual: float, required: float) -> None:
        gates.append(
            {
                "name": name,
                "actual": actual,
                "operator": "<=",
                "required": required,
                "passed": actual <= required,
            }
        )

    def exactly(name: str, actual: float, required: float) -> None:
        gates.append(
            {
                "name": name,
                "actual": actual,
                "operator": "==",
                "required": required,
                "passed": actual == required,
            }
        )

    exactly(
        "frozen quality message batch size",
        nested(report, "score_cache", "message_batch_size"),
        1.0,
    )
    exactly(
        "frozen quality candidate batch size",
        nested(report, "score_cache", "candidate_batch_size"),
        3.0,
    )
    exactly(
        "frozen quality sequence bucket size",
        nested(report, "score_cache", "sequence_bucket_size"),
        64.0,
    )

    for split, label in (("dev", "development"), ("test", "unchanged regression")):
        minimum(
            f"{label} scam recall",
            nested(report, split, "binary_safety", "scam_recall"),
            0.97,
        )
        maximum(
            f"{label} SAFE FPR",
            nested(report, split, "binary_safety", "false_positive_rate"),
            0.02,
        )
    minimum(
        "unchanged regression calibrated macro F1",
        nested(report, "test", "calibrated_decision", "macro_f1"),
        0.94,
    )
    categories = report.get("test", {}).get("scam_by_category")
    if not isinstance(categories, dict):
        raise ValueError("Qwen test report lacks scam category slices")
    eligible_categories = {
        category: values
        for category, values in categories.items()
        if isinstance(values, dict) and int(values.get("examples", 0)) >= CORE_MIN_EXAMPLES
    }
    if not eligible_categories:
        raise ValueError("Qwen test report lacks eligible scam category gates")
    for category, values in sorted(eligible_categories.items()):
        minimum(
            f"unchanged regression {category} recall",
            nested(values, "recall"),
            0.97,
        )

    for split, label, overall_max, domain_max in (
        ("multidogo_annotation_dev", "publisher annotation dev", 0.02, 0.03),
        ("multidogo_annotation_test", "publisher annotation test", 0.02, 0.03),
        ("multidogo_call_validation", "MultiDoGO original calls", 0.02, 0.03),
    ):
        maximum(
            f"{label} SAFE FPR",
            nested(report, split, "binary_safety", "false_positive_rate"),
            overall_max,
        )
        domains = report.get(split, {}).get("by_source_domain")
        if not isinstance(domains, dict) or set(domains) != EXPECTED_DOMAINS:
            raise ValueError(f"{label} domain coverage differs from the six-domain contract")
        for domain, values in sorted(domains.items()):
            maximum(
                f"{label} {domain} SAFE FPR",
                nested(values, "binary_safety", "false_positive_rate"),
                domain_max,
            )

    maximum(
        "long-call SAFE FPR",
        nested(report, "call_window_validation", "binary_safety", "false_positive_rate"),
        0.02,
    )
    maximum(
        "Taskmaster SAFE FPR",
        nested(report, "taskmaster_validation", "binary_safety", "false_positive_rate"),
        0.02,
    )
    minimum(
        "prior-open BothBosu scam recall",
        nested(report, "scam_dialogue_validation", "binary_safety", "scam_recall"),
        0.97,
    )
    maximum(
        "prior-open BothBosu SAFE FPR",
        nested(
            report,
            "scam_dialogue_validation",
            "binary_safety",
            "false_positive_rate",
        ),
        0.02,
    )
    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "quality_status": "passed" if passed else "rejected",
        "passed_gates": sum(bool(gate["passed"]) for gate in gates),
        "total_gates": len(gates),
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "gates": gates,
        "quantization_authorized": passed,
        "huggingface_publication_authorized": False,
        "publication_note": (
            "runtime parity, generation, quantization, mobile, and release audits remain"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_gates(json.loads(args.report.read_text(encoding="utf-8")))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["quality_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
