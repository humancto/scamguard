#!/usr/bin/env python3
"""Freeze a passing quantized candidate before opening primary_test_v8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.gguf_runtime import GGUF_SCORING_VERSION
from scamguard.metrics import file_sha256
from training.eval_qwen import validate_primary_test_v8


def freeze(
    *,
    model: Path,
    runner: Path,
    regression_report_path: Path,
    gate_report_path: Path,
    product_contract_report_path: Path,
    product_contract_gate_report_path: Path,
    primary_test: Path,
    quantization: str,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final artifact declaration: {output}")
    for path in (
        model,
        runner,
        regression_report_path,
        gate_report_path,
        product_contract_report_path,
        product_contract_gate_report_path,
        primary_test,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    regression = json.loads(regression_report_path.read_text(encoding="utf-8"))
    gates = json.loads(gate_report_path.read_text(encoding="utf-8"))
    product_contract = json.loads(
        product_contract_report_path.read_text(encoding="utf-8")
    )
    product_gates = json.loads(
        product_contract_gate_report_path.read_text(encoding="utf-8")
    )
    validate_primary_test_v8(primary_test)
    model_sha256 = file_sha256(model)
    runner_sha256 = file_sha256(runner)
    if (
        regression.get("model_sha256") != model_sha256
        or regression.get("runner_sha256") != runner_sha256
        or regression.get("protocol_version") != 3
        or regression.get("scoring_mode") != "branch_token"
        or regression.get("scoring_version") != GGUF_SCORING_VERSION
        or regression.get("quantization_parity", {}).get("release_gate_passed") is not True
        or regression.get("prediction_ledger", {}).get("contains_message_text") is not False
    ):
        raise ValueError("quantized regression report is not a passing bound artifact")
    if (
        gates.get("quality_status") != "passed"
        or gates.get("quantization_authorized") is not True
        or gates.get("passed_gates") != gates.get("total_gates")
    ):
        raise ValueError("quantized candidate has not passed every pre-sealed quality gate")
    if (
        product_contract.get("artifact_schema_version") != 1
        or product_contract.get("contains_message_text") is not False
        or product_contract.get("semantic_correctness_established") is not False
        or product_contract.get("prediction_ledger", {}).get("sha256")
        != regression.get("prediction_ledger", {}).get("sha256")
    ):
        raise ValueError("product-contract audit is not bound to the quantized prediction ledger")
    if (
        product_gates.get("quality_status") != "passed"
        or product_gates.get("sealed_primary_authorized") is not True
        or product_gates.get("passed_gates") != product_gates.get("total_gates")
        or product_gates.get("product_contract_report", {}).get("sha256")
        != file_sha256(product_contract_report_path)
    ):
        raise ValueError("quantized candidate has not passed every product-contract gate")
    if quantization not in {"Q4_K_M", "Q5_K_M", "Q6_K"}:
        raise ValueError("unsupported frozen quantization")

    record: dict[str, object] = {
        "artifact_schema_version": 1,
        "state": "FINAL_QUANTIZED_CANDIDATE_FROZEN",
        "quantization_frozen": True,
        "quantization": quantization,
        "model": str(model),
        "model_sha256": model_sha256,
        "runner": str(runner),
        "runner_sha256": runner_sha256,
        "calibration_report": str(regression_report_path),
        "calibration_report_sha256": file_sha256(regression_report_path),
        "gate_report": str(gate_report_path),
        "gate_report_sha256": file_sha256(gate_report_path),
        "product_contract_report": str(product_contract_report_path),
        "product_contract_report_sha256": file_sha256(product_contract_report_path),
        "product_contract_gate_report": str(product_contract_gate_report_path),
        "product_contract_gate_report_sha256": file_sha256(
            product_contract_gate_report_path
        ),
        "primary_test_v8": str(primary_test),
        "primary_test_v8_sha256": file_sha256(primary_test),
        "protocol_version": 3,
        "scoring_version": GGUF_SCORING_VERSION,
        "threshold_refit_after_primary_forbidden": True,
        "publication_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--regression-report", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--product-contract-report", type=Path, required=True)
    parser.add_argument("--product-contract-gate-report", type=Path, required=True)
    parser.add_argument("--primary-test-v8", type=Path, required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                model=args.model,
                runner=args.runner,
                regression_report_path=args.regression_report,
                gate_report_path=args.gate_report,
                product_contract_report_path=args.product_contract_report,
                product_contract_gate_report_path=args.product_contract_gate_report,
                primary_test=args.primary_test_v8,
                quantization=args.quantization,
                output=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
