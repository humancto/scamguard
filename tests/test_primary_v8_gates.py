from __future__ import annotations

import pytest

from scripts.check_primary_v8_gates import evaluate


def report() -> dict[str, object]:
    return {
        "final_artifact_declaration": {
            "state": "FINAL_QUANTIZED_CANDIDATE_FROZEN",
            "threshold_refit_after_primary_forbidden": True,
        },
        "primary_test_v8_manifest": {"source": {"local_evaluation_only": True}},
        "primary_test_v8": {
            "binary_safety": {"scam_recall": 0.98, "false_positive_rate": 0.01}
        },
    }


def test_primary_v8_final_quantized_gates_pass() -> None:
    result = evaluate(report())

    assert result["quality_status"] == "passed"
    assert result["quantized_primary_quality_passed"] is True
    assert result["huggingface_publication_authorized"] is False


def test_primary_v8_rejects_quality_or_missing_freeze() -> None:
    candidate = report()
    candidate["primary_test_v8"]["binary_safety"]["scam_recall"] = 0.96  # type: ignore[index]
    result = evaluate(candidate)
    assert result["quality_status"] == "rejected"

    candidate = report()
    candidate["final_artifact_declaration"] = None
    with pytest.raises(ValueError, match="final frozen"):
        evaluate(candidate)
