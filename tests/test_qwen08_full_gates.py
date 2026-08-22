from __future__ import annotations

import pytest

from scripts.check_qwen08_full_gates import EXPECTED_DOMAINS, evaluate_gates


def binary(recall: float = 0.99, fpr: float = 0.01) -> dict[str, float]:
    return {"scam_recall": recall, "false_positive_rate": fpr}


def safe_domain_report(fpr: float = 0.01) -> dict[str, object]:
    return {
        "binary_safety": binary(fpr=fpr),
        "by_source_domain": {
            domain: {"binary_safety": binary(fpr=fpr)} for domain in EXPECTED_DOMAINS
        },
    }


def passing_report() -> dict[str, object]:
    return {
        "dev": {"binary_safety": binary()},
        "test": {
            "binary_safety": binary(),
            "calibrated_decision": {"macro_f1": 0.95},
            "scam_by_category": {
                "FINANCIAL": {"examples": 40, "recall": 0.98},
            },
        },
        "multidogo_annotation_dev": safe_domain_report(),
        "multidogo_annotation_test": safe_domain_report(),
        "multidogo_call_validation": safe_domain_report(),
        "call_window_validation": {"binary_safety": binary(fpr=0.01)},
        "taskmaster_validation": {"binary_safety": binary(fpr=0.01)},
        "scam_dialogue_validation": {"binary_safety": binary()},
    }


def test_full_gate_accepts_only_complete_passing_report() -> None:
    result = evaluate_gates(passing_report())

    assert result["quality_status"] == "passed"
    assert result["quantization_authorized"] is True
    assert result["huggingface_publication_authorized"] is False


def test_full_gate_rejects_publisher_domain_false_positives() -> None:
    report = passing_report()
    report["multidogo_annotation_test"]["by_source_domain"]["media"][  # type: ignore[index]
        "binary_safety"
    ]["false_positive_rate"] = 0.04  # type: ignore[index]

    result = evaluate_gates(report)

    assert result["quality_status"] == "rejected"
    assert "publisher annotation test media SAFE FPR" in result["failed_gates"]


def test_full_gate_requires_all_six_domains() -> None:
    report = passing_report()
    del report["multidogo_annotation_dev"]["by_source_domain"]["media"]  # type: ignore[index]

    with pytest.raises(ValueError, match="six-domain"):
        evaluate_gates(report)
