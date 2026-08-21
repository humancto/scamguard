"""One-call ScamGuard SDK."""

from __future__ import annotations

from .model import ConservativeHeuristicBackend, ModelBackend, load_backend
from .schema import ScanResult
from .signals import choose_action, extract_signal_matches, infer_category
from .taxonomy import Category, RecommendedAction, Verdict


class Scanner:
    def __init__(
        self, backend: ModelBackend | None = None, *, model_path: str | None = None
    ) -> None:
        if backend is not None and model_path is not None:
            raise ValueError("provide backend or model_path, not both")
        self.backend = backend or (
            load_backend(model_path) if model_path else ConservativeHeuristicBackend()
        )

    def scan(self, message: str) -> ScanResult:
        text = message.strip()
        if not text:
            raise ValueError("message must not be empty")
        if len(text) > 100_000:
            raise ValueError("message exceeds the 100,000 character safety limit")

        scores = self.backend.predict(text)
        signal_matches = extract_signal_matches(text)
        signals = tuple(match.signal for match in signal_matches)

        if scores.scam >= self.backend.scam_threshold:
            verdict = Verdict.SCAM
            is_scam: bool | None = True
        elif (
            scores.safe >= 1.0 - self.backend.safe_threshold
            and scores.scam < self.backend.safe_threshold
        ):
            verdict = Verdict.SAFE
            is_scam = False
        else:
            verdict = Verdict.UNCERTAIN
            is_scam = None

        if verdict is Verdict.SAFE:
            signal_matches = ()
            signals = ()
        category = Category.NONE if verdict is Verdict.SAFE else infer_category(text, signals)
        action = choose_action(signals)
        if verdict is Verdict.SAFE:
            action = RecommendedAction.NO_ACTION
        elif action is RecommendedAction.NO_ACTION:
            action = RecommendedAction.VERIFY_OFFICIAL_CHANNEL

        return ScanResult(
            verdict=verdict,
            is_scam=is_scam,
            risk=scores.scam,
            category=category,
            signals=signals,
            evidence_spans=tuple(match.evidence for match in signal_matches),
            recommended_action=action,
            uncertain=verdict is Verdict.UNCERTAIN,
            model_id=self.backend.model_id,
        )


def scan(message: str, *, model_path: str | None = None) -> ScanResult:
    """Classify one message. Supply a trusted local model artifact for benchmarked output."""

    return Scanner(model_path=model_path).scan(message)
