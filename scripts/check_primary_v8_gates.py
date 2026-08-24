#!/usr/bin/env python3
"""Check the final quantized candidate on the separately reported primary_test_v8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def metric(report: dict[str, Any], *path: str) -> float:
    value: Any = report
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing primary-v8 metric: {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"non-numeric primary-v8 metric: {'.'.join(path)}")
    return float(value)


def evaluate(report: dict[str, Any]) -> dict[str, object]:
    declaration = report.get("final_artifact_declaration")
    manifest = report.get("primary_test_v8_manifest")
    if (
        not isinstance(declaration, dict)
        or declaration.get("state") != "FINAL_QUANTIZED_CANDIDATE_FROZEN"
        or declaration.get("threshold_refit_after_primary_forbidden") is not True
        or not isinstance(manifest, dict)
        or manifest.get("source", {}).get("local_evaluation_only") is not True
    ):
        raise ValueError("primary-v8 report lacks the final frozen local-only contract")
    recall = metric(report, "primary_test_v8", "binary_safety", "scam_recall")
    fpr = metric(
        report, "primary_test_v8", "binary_safety", "false_positive_rate"
    )
    gates = [
        {
            "name": "primary_test_v8 scam recall",
            "actual": recall,
            "operator": ">=",
            "required": 0.97,
            "passed": recall >= 0.97,
        },
        {
            "name": "primary_test_v8 SAFE FPR",
            "actual": fpr,
            "operator": "<=",
            "required": 0.02,
            "passed": fpr <= 0.02,
        },
    ]
    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "quality_status": "passed" if passed else "rejected",
        "passed_gates": sum(bool(gate["passed"]) for gate in gates),
        "total_gates": len(gates),
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "gates": gates,
        "quantized_primary_quality_passed": passed,
        "local_evaluation_only": True,
        "huggingface_publication_authorized": False,
        "publication_note": (
            "dataset license clarification, human audit, mobile evidence, model card, "
            "and complete release verification remain mandatory"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["quality_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
