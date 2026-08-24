#!/usr/bin/env python3
"""Apply frozen grounded-product-output gates before sealed evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_SPLITS = ("test", "scam_dialogue_validation")
MIN_TRUE_SCAMS = 20
MIN_END_TO_END_GROUNDED_EXPLANATION_RECALL = 0.97
MIN_EMITTED_SCAM_GROUNDED_EVIDENCE_RATE = 0.98
MIN_TRUE_POSITIVE_SCAM_GROUNDED_EVIDENCE_RATE = 0.99
MIN_EMITTED_SCAM_KNOWN_CATEGORY_RATE = 0.98
MIN_EMITTED_SCAM_SPECIFIC_ACTION_RATE = 0.90


def nested(mapping: dict[str, Any], *path: str) -> float:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing product-contract metric: {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"non-numeric product-contract metric: {'.'.join(path)}")
    return float(value)


def evaluate_gates(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("artifact_schema_version") != 1:
        raise ValueError("product-contract report schema differs from the frozen gate")
    if report.get("contains_message_text") is not False:
        raise ValueError("product-contract report is not text-free")
    if report.get("semantic_correctness_established") is not False:
        raise ValueError("coverage audit must not claim semantic human correctness")
    by_split = report.get("by_split")
    if not isinstance(by_split, dict):
        raise ValueError("product-contract report lacks split metrics")

    gates: list[dict[str, Any]] = []

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

    for split in REQUIRED_SPLITS:
        values = by_split.get(split)
        if not isinstance(values, dict):
            raise ValueError(f"product-contract report lacks required split {split}")
        minimum(
            f"{split} true scam denominator",
            nested(values, "truth", "SCAM"),
            float(MIN_TRUE_SCAMS),
        )
        minimum(
            f"{split} end-to-end grounded explanation recall",
            nested(values, "true_scam_grounded_explanation_recall"),
            MIN_END_TO_END_GROUNDED_EXPLANATION_RECALL,
        )
        minimum(
            f"{split} emitted SCAM grounded evidence",
            nested(values, "emitted_scam_grounded_evidence_rate"),
            MIN_EMITTED_SCAM_GROUNDED_EVIDENCE_RATE,
        )
        minimum(
            f"{split} true-positive SCAM grounded evidence",
            nested(values, "true_positive_scam_grounded_evidence_rate"),
            MIN_TRUE_POSITIVE_SCAM_GROUNDED_EVIDENCE_RATE,
        )
        minimum(
            f"{split} emitted SCAM known category",
            nested(values, "emitted_scam_known_category_rate"),
            MIN_EMITTED_SCAM_KNOWN_CATEGORY_RATE,
        )
        minimum(
            f"{split} emitted SCAM specific action",
            nested(values, "emitted_scam_specific_action_rate"),
            MIN_EMITTED_SCAM_SPECIFIC_ACTION_RATE,
        )

    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "artifact_schema_version": 1,
        "quality_status": "passed" if passed else "rejected",
        "passed_gates": sum(bool(gate["passed"]) for gate in gates),
        "total_gates": len(gates),
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "gates": gates,
        "sealed_primary_authorized": passed,
        "semantic_human_review_complete": False,
        "huggingface_publication_authorized": False,
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
