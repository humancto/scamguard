"""Stable, JSON-serializable output contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .taxonomy import Category, RecommendedAction, Signal, Verdict


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.text or self.start < 0 or self.end <= self.start:
            raise ValueError("invalid evidence span")


@dataclass(frozen=True, slots=True)
class ScanResult:
    verdict: Verdict
    is_scam: bool | None
    risk: float
    category: Category
    signals: tuple[Signal, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    recommended_action: RecommendedAction
    uncertain: bool
    model_id: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk must be between 0 and 1")
        expected = {
            Verdict.SAFE: False,
            Verdict.UNCERTAIN: None,
            Verdict.SCAM: True,
        }[self.verdict]
        if self.is_scam is not expected:
            raise ValueError("is_scam must agree with verdict")
        if self.uncertain is not (self.verdict is Verdict.UNCERTAIN):
            raise ValueError("uncertain must agree with verdict")
        if self.verdict is Verdict.SAFE and self.category is not Category.NONE:
            raise ValueError("SAFE results must use category NONE")
        if self.verdict is Verdict.SAFE and (self.signals or self.evidence_spans):
            raise ValueError("SAFE results must not expose scam signals or evidence")
        if (
            self.verdict is Verdict.SAFE
            and self.recommended_action is not RecommendedAction.NO_ACTION
        ):
            raise ValueError("SAFE results must use NO_ACTION")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        data["category"] = self.category.value
        data["signals"] = [signal.value for signal in self.signals]
        data["evidence_spans"] = [asdict(span) for span in self.evidence_spans]
        data["recommended_action"] = self.recommended_action.value
        return data
