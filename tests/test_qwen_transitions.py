from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_qwen_transitions import (
    compare,
    read_prediction_split,
    read_reference,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def prediction(
    identifier: str,
    truth: str,
    verdict: str,
    threshold_scam: bool,
    probabilities: dict[str, float],
) -> dict:
    return {
        "id": identifier,
        "split": "open",
        "source": "source",
        "category": "UNKNOWN",
        "truth": truth,
        "calibrated_verdict": verdict,
        "threshold_scam": threshold_scam,
        "probabilities": probabilities,
    }


def test_transition_audit_is_exact_and_text_free(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    baseline_rows = [
        prediction(
            "a",
            "UNCERTAIN",
            "UNCERTAIN",
            False,
            {"SAFE": 0.2, "UNCERTAIN": 0.7, "SCAM": 0.1},
        ),
        prediction("b", "SCAM", "UNCERTAIN", False, {"SAFE": 0.1, "UNCERTAIN": 0.6, "SCAM": 0.3}),
        prediction("c", "SAFE", "SAFE", False, {"SAFE": 0.8, "UNCERTAIN": 0.1, "SCAM": 0.1}),
    ]
    candidate_rows = [
        prediction("a", "UNCERTAIN", "SCAM", True, {"SAFE": 0.1, "UNCERTAIN": 0.2, "SCAM": 0.7}),
        prediction("b", "SCAM", "SCAM", True, {"SAFE": 0.1, "UNCERTAIN": 0.1, "SCAM": 0.8}),
        prediction("c", "SAFE", "SAFE", False, {"SAFE": 0.9, "UNCERTAIN": 0.05, "SCAM": 0.05}),
    ]
    reference_rows = [
        {
            "id": "a",
            "text": "Press 1 about a $200 charge.",
            "label": "UNCERTAIN",
            "family_id": "fa",
        },
        {"id": "b", "text": "Pay immediately or face arrest.", "label": "SCAM", "family_id": "fb"},
        {"id": "c", "text": "Use the official app.", "label": "SAFE", "family_id": "fc"},
    ]
    write_jsonl(baseline_path, baseline_rows)
    write_jsonl(candidate_path, candidate_rows)
    write_jsonl(reference_path, reference_rows)

    result = compare(
        read_prediction_split(baseline_path, "open"),
        read_prediction_split(candidate_path, "open"),
        read_reference(reference_path),
        split="open",
    )

    assert result["changed_examples"] == 2
    assert result["verdict_outcomes"]["improved"] == 1
    assert result["verdict_outcomes"]["regressed"] == 1
    assert result["binary_outcomes"]["improved"] == 1
    assert result["changed_rows"][0]["surface_profile"]["cues"]["press_digit"] is True
    serialized = json.dumps(result)
    assert "Press 1" not in serialized
    assert result["contains_message_text"] is False


def test_transition_audit_rejects_mismatched_ids(tmp_path: Path) -> None:
    baseline = {
        "a": prediction("a", "SAFE", "SAFE", False, {"SAFE": 0.8, "UNCERTAIN": 0.1, "SCAM": 0.1})
    }
    candidate = {
        "b": prediction("b", "SAFE", "SAFE", False, {"SAFE": 0.8, "UNCERTAIN": 0.1, "SCAM": 0.1})
    }
    reference = {"a": {"id": "a", "text": "hello", "label": "SAFE"}}
    with pytest.raises(ValueError, match="ID sets must match exactly"):
        compare(baseline, candidate, reference, split="open")


def test_prediction_reader_rejects_text_bearing_ledger(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    row = prediction(
        "a", "SAFE", "SAFE", False, {"SAFE": 0.8, "UNCERTAIN": 0.1, "SCAM": 0.1}
    )
    row["message"] = "secret"
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match="text-bearing fields are forbidden"):
        read_prediction_split(path, "open")
