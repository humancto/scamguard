from __future__ import annotations

import pytest

from benchmarks.benchmark_native_gguf_prefix import score_parity


def test_score_parity_reports_probability_drift_without_threshold_crossing() -> None:
    result = score_parity(
        [(1.0, 0.0, -1.0), (-1.0, 0.0, 1.0)],
        [(1.001, 0.0, -1.0), (-1.0, 0.0, 1.001)],
        temperature=1.0,
        scam_threshold=0.6,
        safe_threshold=0.6,
    )

    assert result["maximum_absolute_raw_score_error"] == pytest.approx(0.001)
    assert result["maximum_absolute_probability_error"] > 0.0
    assert result["calibrated_verdict_mismatch_count"] == 0
    assert result["release_gate_passed"] is True


def test_score_parity_rejects_empty_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="equal non-empty"):
        score_parity(
            [],
            [],
            temperature=1.0,
            scam_threshold=0.5,
            safe_threshold=0.5,
        )
