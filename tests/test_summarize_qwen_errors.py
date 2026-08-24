"""Text-free Qwen error summaries expose actionable aggregate slices."""

from __future__ import annotations

import pytest

from scripts.summarize_qwen_errors import summarize_ledger


def prediction(
    identifier: str,
    truth: str,
    verdict: str,
    *,
    threshold_scam: bool,
    nll: float,
) -> dict[str, object]:
    return {
        "id": identifier,
        "split": "test",
        "source": "fixture",
        "category": "IMPERSONATION",
        "truth": truth,
        "calibrated_verdict": verdict,
        "threshold_scam": threshold_scam,
        "negative_log_likelihood": nll,
    }


def test_summary_reports_false_positives_misses_and_hardest_errors() -> None:
    rows = [
        prediction("safe-ok", "SAFE", "SAFE", threshold_scam=False, nll=0.1),
        prediction("safe-fp", "SAFE", "SCAM", threshold_scam=True, nll=1.2),
        prediction("scam-fn", "SCAM", "UNCERTAIN", threshold_scam=False, nll=2.0),
        prediction("scam-ok", "SCAM", "SCAM", threshold_scam=True, nll=0.2),
    ]

    report = summarize_ledger(rows, hardest_limit=1)

    assert report["contains_message_text"] is False
    assert report["overall"]["safe_false_positive_rate"] == 0.5
    assert report["overall"]["scam_false_negative_rate"] == 0.5
    assert report["overall"]["verdict_error_rate"] == 0.5
    assert report["by_split"]["test"]["examples"] == 4
    assert report["hardest_calibrated_errors"][0]["id"] == "scam-fn"


def test_summary_rejects_message_text() -> None:
    row = prediction("unsafe", "SAFE", "SAFE", threshold_scam=False, nll=0.1)
    row["text"] = "must not leak"

    with pytest.raises(ValueError, match="contains message text"):
        summarize_ledger([row])
