from __future__ import annotations

import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.freeze_gguf_primary_v8 import freeze


def fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": tmp_path / "model-q4-k-m.gguf",
        "runner": tmp_path / "runner",
        "regression": tmp_path / "regression.json",
        "gates": tmp_path / "gates.json",
        "product": tmp_path / "product.json",
        "product_gates": tmp_path / "product-gates.json",
        "primary": tmp_path / "primary_test_v8.jsonl",
    }
    paths["model"].write_bytes(b"model")
    paths["runner"].write_bytes(b"runner")
    paths["primary"].write_text('{"id":"one"}\n', encoding="utf-8")
    (tmp_path / "primary_test_v8.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 8,
                "benchmark_state": "SEALED_MODEL_PREDICTIONS_NOT_RUN",
                "source": {
                    "local_evaluation_only": True,
                    "training_allowed_by_project": False,
                },
                "artifact": {"sha256": file_sha256(paths["primary"])},
                "counts": {"final_rows": 1},
            }
        ),
        encoding="utf-8",
    )
    paths["regression"].write_text(
        json.dumps(
            {
                "model_sha256": file_sha256(paths["model"]),
                "runner_sha256": file_sha256(paths["runner"]),
                "protocol_version": 3,
                "scoring_mode": "branch_token",
                "scoring_version": "qwen-verdict-branch-token-v1",
                "quantization_parity": {"release_gate_passed": True},
                "prediction_ledger": {
                    "contains_message_text": False,
                    "sha256": "ledger-sha",
                },
            }
        ),
        encoding="utf-8",
    )
    paths["gates"].write_text(
        json.dumps(
            {
                "quality_status": "passed",
                "quantization_authorized": True,
                "passed_gates": 39,
                "total_gates": 39,
            }
        ),
        encoding="utf-8",
    )
    paths["product"].write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "contains_message_text": False,
                "semantic_correctness_established": False,
                "prediction_ledger": {"sha256": "ledger-sha"},
            }
        ),
        encoding="utf-8",
    )
    paths["product_gates"].write_text(
        json.dumps(
            {
                "quality_status": "passed",
                "sealed_primary_authorized": True,
                "passed_gates": 12,
                "total_gates": 12,
                "product_contract_report": {
                    "sha256": file_sha256(paths["product"])
                },
            }
        ),
        encoding="utf-8",
    )
    return paths


def run(paths: dict[str, Path], output: Path) -> dict[str, object]:
    return freeze(
        model=paths["model"],
        runner=paths["runner"],
        regression_report_path=paths["regression"],
        gate_report_path=paths["gates"],
        product_contract_report_path=paths["product"],
        product_contract_gate_report_path=paths["product_gates"],
        primary_test=paths["primary"],
        quantization="Q4_K_M",
        output=output,
    )


def test_freezes_passing_quantized_candidate_before_sealed_evaluation(
    tmp_path: Path,
) -> None:
    paths = fixture(tmp_path)

    record = run(paths, tmp_path / "declaration.json")

    assert record["state"] == "FINAL_QUANTIZED_CANDIDATE_FROZEN"
    assert record["quantization_frozen"] is True
    assert record["threshold_refit_after_primary_forbidden"] is True
    assert record["publication_authorized"] is False


def test_rejects_failed_gate_or_quantization_parity(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    gates = json.loads(paths["gates"].read_text())
    gates["quality_status"] = "rejected"
    paths["gates"].write_text(json.dumps(gates), encoding="utf-8")

    with pytest.raises(ValueError, match="pre-sealed quality gate"):
        run(paths, tmp_path / "declaration.json")

    paths = fixture(tmp_path / "second")
    regression = json.loads(paths["regression"].read_text())
    regression["quantization_parity"]["release_gate_passed"] = False
    paths["regression"].write_text(json.dumps(regression), encoding="utf-8")
    with pytest.raises(ValueError, match="passing bound artifact"):
        run(paths, tmp_path / "second-declaration.json")


def test_rejects_failed_or_unbound_product_contract(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    product_gates = json.loads(paths["product_gates"].read_text())
    product_gates["quality_status"] = "rejected"
    paths["product_gates"].write_text(json.dumps(product_gates), encoding="utf-8")

    with pytest.raises(ValueError, match="product-contract gate"):
        run(paths, tmp_path / "declaration.json")

    paths = fixture(tmp_path / "unbound")
    product = json.loads(paths["product"].read_text())
    product["prediction_ledger"]["sha256"] = "different"
    paths["product"].write_text(json.dumps(product), encoding="utf-8")
    product_gates = json.loads(paths["product_gates"].read_text())
    product_gates["product_contract_report"]["sha256"] = file_sha256(paths["product"])
    paths["product_gates"].write_text(json.dumps(product_gates), encoding="utf-8")
    with pytest.raises(ValueError, match="quantized prediction ledger"):
        run(paths, tmp_path / "unbound-declaration.json")
