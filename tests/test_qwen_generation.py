"""Generated Qwen JSON must meet syntactic and semantic runtime constraints."""

import json

from training.eval_qwen_generation import sample_rows, validate_output


def test_generated_output_accepts_grounded_contract() -> None:
    text = "Urgent: share your verification code now."
    generated = json.dumps(
        {
            "verdict": "SCAM",
            "category": "CREDENTIAL_MFA",
            "signals": ["artificial_urgency", "otp_request"],
            "evidence": ["Urgent", "share your verification code"],
            "recommended_action": "DO_NOT_SHARE_CODE",
        },
        separators=(",", ":"),
    )

    payload, errors = validate_output(text, generated)

    assert payload is not None
    assert errors == []


def test_generated_output_rejects_semantic_inconsistency() -> None:
    text = "Dinner is at six."
    generated = json.dumps(
        {
            "verdict": "SAFE",
            "category": "OTHER_SCAM",
            "signals": [],
            "evidence": ["Dinner"],
            "recommended_action": "DO_NOT_REPLY",
        },
        separators=(",", ":"),
    )

    _, errors = validate_output(text, generated)

    assert "signal_evidence_length_mismatch" in errors
    assert "safe_category_not_none" in errors
    assert "safe_action_not_none" in errors
    assert "safe_has_risk_evidence" in errors


def test_generation_sample_fills_limit_with_imbalanced_strata() -> None:
    rows = [{"id": f"large-{index}", "source": "large", "label": "SCAM"} for index in range(20)] + [
        {"id": "small-0", "source": "small", "label": "SAFE"}
    ]

    sample = sample_rows(rows, limit=10, seed="fixed")

    assert len(sample) == 10
    assert any(row["source"] == "small" for row in sample)
    assert sample == sample_rows(rows, limit=10, seed="fixed")
