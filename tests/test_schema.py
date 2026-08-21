from __future__ import annotations

import pytest

from scamguard.schema import EvidenceSpan, ScanResult
from scamguard.taxonomy import Category, RecommendedAction, Signal, Verdict


def test_evidence_span_rejects_invalid_offsets() -> None:
    with pytest.raises(ValueError):
        EvidenceSpan(text="urgent", start=8, end=8)


def test_scan_result_rejects_false_safe_boolean() -> None:
    with pytest.raises(ValueError):
        ScanResult(
            verdict=Verdict.SCAM,
            is_scam=False,
            risk=0.9,
            category=Category.OTHER_SCAM,
            signals=(),
            evidence_spans=(),
            recommended_action=RecommendedAction.VERIFY_OFFICIAL_CHANNEL,
            uncertain=False,
            model_id="broken",
        )


def test_scan_result_serializes_the_prd_contract_with_offset_evidence() -> None:
    result = ScanResult(
        verdict=Verdict.SCAM,
        is_scam=True,
        risk=0.994,
        category=Category.DELIVERY_TOLL_PARKING,
        signals=(Signal.ARTIFICIAL_URGENCY, Signal.SUSPICIOUS_LINK),
        evidence_spans=(EvidenceSpan(text="within 12 hours", start=8, end=23),),
        recommended_action=RecommendedAction.DO_NOT_OPEN_LINK,
        uncertain=False,
        model_id="scamguard-test",
    )

    assert result.to_dict() == {
        "verdict": "SCAM",
        "is_scam": True,
        "risk": 0.994,
        "category": "DELIVERY_TOLL_PARKING",
        "signals": ["artificial_urgency", "suspicious_link"],
        "evidence_spans": [
            {"text": "within 12 hours", "start": 8, "end": 23}
        ],
        "recommended_action": "DO_NOT_OPEN_LINK",
        "uncertain": False,
        "model_id": "scamguard-test",
    }
