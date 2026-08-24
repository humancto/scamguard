"""Grounded-output gates block weak explanations before sealed evaluation."""

from __future__ import annotations

import pytest

from scripts.check_product_contract_gates import evaluate_gates


def split_metrics() -> dict[str, object]:
    return {
        "truth": {"SCAM": 100},
        "true_scam_grounded_explanation_recall": 0.98,
        "emitted_scam_grounded_evidence_rate": 0.99,
        "true_positive_scam_grounded_evidence_rate": 1.0,
        "emitted_scam_known_category_rate": 0.99,
        "emitted_scam_specific_action_rate": 0.95,
    }


def passing_report() -> dict[str, object]:
    return {
        "artifact_schema_version": 1,
        "contains_message_text": False,
        "semantic_correctness_established": False,
        "by_split": {
            "test": split_metrics(),
            "scam_dialogue_validation": split_metrics(),
        },
    }


def test_product_contract_gates_accept_strong_grounded_outputs() -> None:
    result = evaluate_gates(passing_report())

    assert result["quality_status"] == "passed"
    assert result["passed_gates"] == result["total_gates"] == 12
    assert result["sealed_primary_authorized"] is True
    assert result["huggingface_publication_authorized"] is False


def test_product_contract_gates_reject_weak_end_to_end_explanation() -> None:
    report = passing_report()
    report["by_split"]["test"]["true_scam_grounded_explanation_recall"] = 0.96  # type: ignore[index]

    result = evaluate_gates(report)

    assert result["quality_status"] == "rejected"
    assert "test end-to-end grounded explanation recall" in result["failed_gates"]
    assert result["sealed_primary_authorized"] is False


def test_product_contract_gates_require_text_free_audit() -> None:
    report = passing_report()
    report["contains_message_text"] = True

    with pytest.raises(ValueError, match="not text-free"):
        evaluate_gates(report)
