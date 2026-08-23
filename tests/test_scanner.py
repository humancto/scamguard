from __future__ import annotations

from scamguard.model import ModelScores
from scamguard.scanner import Scanner
from scamguard.taxonomy import RecommendedAction, Verdict


class StubBackend:
    model_id = "test-stub"
    scam_threshold = 0.80
    safe_threshold = 0.20
    safe_probability_threshold = 0.80
    safe_max_scam_probability = 0.20

    def __init__(self, scores: ModelScores) -> None:
        self.scores = scores

    def predict(self, text: str) -> ModelScores:
        return self.scores


def test_scam_result_has_evidence_and_action() -> None:
    scanner = Scanner(backend=StubBackend(ModelScores(safe=0.01, uncertain=0.04, scam=0.95)))
    message = "Urgent: share the verification code now so we can stop the wire transfer."
    result = scanner.scan(message)

    assert result.verdict is Verdict.SCAM
    assert result.is_scam is True
    assert result.evidence_spans
    assert all(message[item.start : item.end] == item.text for item in result.evidence_spans)
    assert result.recommended_action is RecommendedAction.DO_NOT_SHARE_CODE


def test_uncertain_is_not_coerced_to_boolean() -> None:
    scanner = Scanner(backend=StubBackend(ModelScores(safe=0.30, uncertain=0.50, scam=0.20)))
    result = scanner.scan("Please review this account notification.")

    assert result.verdict is Verdict.UNCERTAIN
    assert result.is_scam is None
    assert result.uncertain is True


def test_safe_result_has_no_action_or_scam_category() -> None:
    scanner = Scanner(backend=StubBackend(ModelScores(safe=0.95, uncertain=0.04, scam=0.01)))
    result = scanner.scan(
        "Security reminder: never share a verification code; use https://bank.example."
    )

    assert result.verdict is Verdict.SAFE
    assert result.is_scam is False
    assert result.recommended_action is RecommendedAction.NO_ACTION
    assert result.category.value == "NONE"
    assert result.signals == ()
    assert result.evidence_spans == ()


def test_backend_can_expose_qwen_direct_safe_probability_semantics() -> None:
    backend = StubBackend(ModelScores(safe=0.65, uncertain=0.25, scam=0.10))
    backend.safe_probability_threshold = 0.60
    backend.safe_max_scam_probability = None

    result = Scanner(backend=backend).scan("Routine account notice")

    assert result.verdict is Verdict.SAFE


def test_scam_threshold_has_precedence_over_safe_probability() -> None:
    backend = StubBackend(ModelScores(safe=0.55, uncertain=0.05, scam=0.40))
    backend.scam_threshold = 0.35
    backend.safe_probability_threshold = 0.50
    backend.safe_max_scam_probability = None

    result = Scanner(backend=backend).scan("Please inspect this notice")

    assert result.verdict is Verdict.SCAM


def test_empty_message_fails_closed() -> None:
    scanner = Scanner(backend=StubBackend(ModelScores(safe=0.95, uncertain=0.04, scam=0.01)))
    try:
        scanner.scan("  ")
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty input should fail")
