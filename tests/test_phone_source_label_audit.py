from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_phone_source_labels import (
    audit,
    caller_risk_reasons,
    parse_turns,
    prediction_alignment,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def row(identifier: str, label: str, text: str, split: str = "validation") -> dict:
    return {
        "id": identifier,
        "text": text,
        "label": label,
        "split": split,
        "source": "phone",
        "family_id": f"family-{identifier}",
        "source_variation_style": "neutral",
    }


def test_turn_parser_and_caller_risk_ignore_receiver_request() -> None:
    text = (
        "caller: Use our secure portal, then read the confirmation code back to us. "
        "receiver: Can you confirm my social security number? "
        "caller: I cannot disclose that."
    )
    assert [speaker for speaker, _ in parse_turns(text)] == [
        "caller",
        "receiver",
        "caller",
    ]
    assert caller_risk_reasons(text) == [
        "caller_sends_or_directs_link",
        "caller_requests_code_return",
    ]
    receiver_only = "caller: I cannot disclose it. receiver: Please confirm my password."
    assert caller_risk_reasons(receiver_only) == []


def test_audit_flags_conflicts_without_copying_text(tmp_path: Path) -> None:
    path = tmp_path / "validation.jsonl"
    risky_text = (
        "caller: I will email a secure link. Please enter your password, then read the "
        "confirmation code back to us. receiver: I am not comfortable with that."
    )
    artifact_text = "caller: Hello. receiver: Hi. (I will stop here to meet the requirement)"
    write_jsonl(
        path,
        [
            row("safe-risk", "SAFE", risky_text),
            row("scam-weak", "SCAM", "caller: Hello there. receiver: Goodbye."),
            row("artifact", "SAFE", artifact_text),
        ],
    )

    result = audit([path])

    assert result["flag_counts"] == {
        "generation_artifact": 1,
        "publisher_safe_has_high_risk_caller_action": 1,
        "publisher_scam_lacks_detected_runtime_evidence": 1,
    }
    serialized = json.dumps(result)
    assert risky_text not in serialized
    assert artifact_text not in serialized
    assert result["contains_message_text"] is False
    assert result["policy"]["automatic_relabeling_allowed"] is False
    assert len(result["validation_index"]) == 3
    assert {row["publisher_label"] for row in result["validation_index"]} == {
        "SAFE",
        "SCAM",
    }


def test_audit_refuses_prediction_sealed_test(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    write_jsonl(path, [row("sealed", "SAFE", "caller: Hello.", split="test")])
    with pytest.raises(ValueError, match="prediction-sealed"):
        audit([path])


def test_prediction_alignment_separates_questionable_safe_rows(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "id": "flagged",
            "split": "phone_scam_validation",
            "truth": "SAFE",
            "threshold_scam": True,
            "calibrated_verdict": "SCAM",
        },
        {
            "id": "other-safe",
            "split": "phone_scam_validation",
            "truth": "SAFE",
            "threshold_scam": False,
            "calibrated_verdict": "SAFE",
        },
        {
            "id": "scam",
            "split": "phone_scam_validation",
            "truth": "SCAM",
            "threshold_scam": True,
            "calibrated_verdict": "SCAM",
        },
    ]
    write_jsonl(path, rows)

    result = prediction_alignment(
        path,
        {"flagged": "SAFE", "other-safe": "SAFE", "scam": "SCAM"},
        {"flagged"},
    )

    assert result["safe_false_positives"] == 1
    assert result["false_positives_with_high_risk_caller_action"] == 1
    assert result["false_positives_without_high_risk_caller_action"] == 0
    assert result["high_risk_action_trigger_rate"] == 1.0
    assert result["other_safe_trigger_rate"] == 0.0
