from __future__ import annotations

import pytest

from benchmarks.benchmark_gguf_verdict_latency import (
    score_phase_seconds,
    summarize,
    timestamp_microseconds,
)


def test_timestamp_microseconds_parses_llama_elapsed_format() -> None:
    assert timestamp_microseconds(("1", "02", "003", "004")) == 62_003_004


def test_score_phase_seconds_uses_internal_runtime_timestamps() -> None:
    output = (
        "0.00.530.112 I multiple_choice_score : calculating score\n"
        "0.00.867.525 I Final result: 0.0\n"
    )

    assert score_phase_seconds(output) == pytest.approx(0.337413)


def test_score_phase_seconds_requires_one_interval() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        score_phase_seconds("no timestamps")


def test_summarize_reports_distribution() -> None:
    result = summarize([10.0, 20.0, 30.0])

    assert result["mean"] == 20.0
    assert result["median"] == 20.0
    assert result["p95"] == pytest.approx(29.0)
