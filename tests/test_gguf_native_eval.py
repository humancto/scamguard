from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scamguard.metrics import file_sha256
from training.eval_gguf_native import (
    GGUF_SCORING_VERSION,
    cache_identity,
    calibration_from_report,
    load_cache,
    save_cache,
    score_rows,
    validate_final_declaration,
)


def identity() -> dict[str, object]:
    return cache_identity(
        split="dev",
        rows=2,
        data_sha256="data",
        model_sha256="model",
        runner_sha256="runner",
        ctx_size=640,
        batch_size=640,
        ubatch_size=128,
        threads=4,
        n_gpu_layers=99,
    )


def test_native_score_cache_requires_exact_identity(tmp_path: Path) -> None:
    scores = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    save_cache(tmp_path, "dev", identity(), scores, [1.0, 2.0])

    loaded = load_cache(tmp_path, "dev", identity())

    assert loaded is not None
    np.testing.assert_array_equal(loaded[0], scores)
    assert loaded[1] == [1.0, 2.0]
    assert load_cache(tmp_path, "dev", identity() | {"model_sha256": "changed"}) is None


def test_native_calibration_is_bound_to_quantized_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "calibration": {
                    "backend_type": "qwen_gguf_verdict_branch_token",
                    "model_sha256": "model",
                    "runner_sha256": "runner",
                    "protocol_version": 3,
                    "scoring_mode": "branch_token",
                    "scoring_version": GGUF_SCORING_VERSION,
                    "dev_data_sha256": "dev",
                    "maximum_safe_fpr": 0.02,
                    "minimum_dev_recall": 0.97,
                    "temperature": 1.0,
                    "scam_threshold": 0.2,
                    "safe_threshold": 0.7,
                }
            }
        ),
        encoding="utf-8",
    )

    record = calibration_from_report(
        path,
        model_sha256="model",
        runner_sha256="runner",
        dev_sha256="dev",
        max_fpr=0.02,
        min_recall=0.97,
    )

    assert record["temperature"] == 1.0
    with pytest.raises(ValueError, match="differs"):
        calibration_from_report(
            path,
            model_sha256="changed",
            runner_sha256="runner",
            dev_sha256="dev",
            max_fpr=0.02,
            min_recall=0.97,
        )


def test_final_declaration_binds_every_sealed_input(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    runner = tmp_path / "runner"
    calibration = tmp_path / "calibration.json"
    primary = tmp_path / "primary_test_v8.jsonl"
    product_contract = tmp_path / "product-contract.json"
    product_gates = tmp_path / "product-gates.json"
    for path, content in (
        (model, b"model"),
        (runner, b"runner"),
        (calibration, b"calibration"),
        (primary, b"primary"),
    ):
        path.write_bytes(content)
    product_contract.write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "contains_message_text": False,
                "semantic_correctness_established": False,
            }
        ),
        encoding="utf-8",
    )
    product_gates.write_text(
        json.dumps(
            {
                "quality_status": "passed",
                "sealed_primary_authorized": True,
                "passed_gates": 12,
                "total_gates": 12,
                "product_contract_report": {
                    "sha256": file_sha256(product_contract)
                },
            }
        ),
        encoding="utf-8",
    )
    declaration = tmp_path / "declaration.json"
    record = {
        "artifact_schema_version": 1,
        "state": "FINAL_QUANTIZED_CANDIDATE_FROZEN",
        "quantization_frozen": True,
        "model_sha256": file_sha256(model),
        "runner_sha256": file_sha256(runner),
        "calibration_report_sha256": file_sha256(calibration),
        "primary_test_v8_sha256": file_sha256(primary),
        "product_contract_report": str(product_contract),
        "product_contract_report_sha256": file_sha256(product_contract),
        "product_contract_gate_report": str(product_gates),
        "product_contract_gate_report_sha256": file_sha256(product_gates),
        "protocol_version": 3,
        "scoring_version": GGUF_SCORING_VERSION,
    }
    declaration.write_text(json.dumps(record), encoding="utf-8")

    assert validate_final_declaration(
        declaration,
        model=model,
        runner=runner,
        calibration_report=calibration,
        primary_test=primary,
    ) == record
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        validate_final_declaration(
            declaration,
            model=model,
            runner=runner,
            calibration_report=calibration,
            primary_test=primary,
        )


def test_score_rows_uses_frozen_prompt_and_text_free_ids() -> None:
    class FakeScorer:
        questions: list[tuple[str, str]] = []

        def score(self, identifier: str, question: str, *, timeout_seconds: float) -> object:
            assert timeout_seconds == 120.0
            self.questions.append((identifier, question))
            return SimpleNamespace(raw_scores=(-1.0, -2.0, -3.0), round_trip_ms=4.0)

    scorer = FakeScorer()
    scores, timings = score_rows(
        scorer,  # type: ignore[arg-type]
        "dev",
        [{"id": "private-id", "text": "ordinary message"}],
    )

    assert scorer.questions[0][0] == "dev-0"
    assert "ordinary message" in scorer.questions[0][1]
    assert "private-id" not in scorer.questions[0][1]
    np.testing.assert_array_equal(scores, [[-1.0, -2.0, -3.0]])
    assert timings == [4.0]
