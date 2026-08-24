"""Product-contract audits stay grounded, complete, and text-free."""

from __future__ import annotations

import json

import pytest

from scripts.evaluate_product_contract import evaluate_product_contract


def data(identifier: str, text: str, label: str, category: str) -> dict[str, object]:
    return {
        "id": identifier,
        "text": text,
        "label": label,
        "category": category,
        "source": "fixture",
    }


def prediction(
    identifier: str, truth: str, category: str, verdict: str
) -> dict[str, object]:
    return {
        "id": identifier,
        "split": "test",
        "source": "fixture",
        "category": category,
        "truth": truth,
        "calibrated_verdict": verdict,
    }


def test_product_contract_reports_grounded_end_to_end_coverage() -> None:
    rows = [
        data(
            "scam-explained",
            "Urgent: share your verification code now.",
            "SCAM",
            "CREDENTIAL_THEFT",
        ),
        data("scam-missed", "A subtle manipulation.", "SCAM", "UNKNOWN"),
        data("safe", "Dinner is at six.", "SAFE", "NONE"),
        data("false-positive", "Ordinary project update.", "SAFE", "NONE"),
    ]
    predictions = [
        prediction("scam-explained", "SCAM", "CREDENTIAL_THEFT", "SCAM"),
        prediction("scam-missed", "SCAM", "UNKNOWN", "UNCERTAIN"),
        prediction("safe", "SAFE", "NONE", "SAFE"),
        prediction("false-positive", "SAFE", "NONE", "SCAM"),
    ]

    report = evaluate_product_contract({"test": rows}, predictions)

    overall = report["overall"]
    assert report["contains_message_text"] is False
    assert report["semantic_correctness_established"] is False
    assert overall["emitted_scams"] == 2
    assert overall["emitted_scam_grounded_evidence_rate"] == 0.5
    assert overall["true_positive_scam_grounded_evidence_rate"] == 1.0
    assert overall["true_scam_grounded_explanation_recall"] == 0.5
    assert overall["recommended_actions"]["DO_NOT_SHARE_CODE"] == 1
    assert "Dinner is at six" not in json.dumps(report)


def test_product_contract_rejects_prediction_text() -> None:
    row = data("safe", "Dinner is at six.", "SAFE", "NONE")
    ledger = prediction("safe", "SAFE", "NONE", "SAFE")
    ledger["text"] = "must not survive"

    with pytest.raises(ValueError, match="contains message text"):
        evaluate_product_contract({"test": [row]}, [ledger])


def test_product_contract_requires_exact_data_ledger_binding() -> None:
    row = data("safe", "Dinner is at six.", "SAFE", "NONE")
    ledger = prediction("safe", "SCAM", "NONE", "SAFE")

    with pytest.raises(ValueError, match="label differs"):
        evaluate_product_contract({"test": [row]}, [ledger])
