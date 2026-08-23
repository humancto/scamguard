from __future__ import annotations

import pytest

from benchmarks.benchmark_gguf_runtime import (
    derive_measurement,
    parse_max_rss,
    select_prompt_measurement,
)


def row(**overrides: object) -> dict[str, object]:
    return {
        "samples_ns": [10_000_000, 20_000_000, 30_000_000],
        "n_prompt": 192,
        "n_gen": 0,
        "n_threads": 12,
        "n_gpu_layers": 99,
        "backends": "MTL,BLAS",
        "avg_ts": 9_600.0,
    } | overrides


def test_derive_measurement_reports_latency_distribution() -> None:
    result = derive_measurement(row())

    assert result["kind"] == "prompt_processing"
    assert result["tokens"] == 192
    assert result["mean_ms"] == 20.0
    assert result["median_ms"] == 20.0
    assert result["p95_ms"] == pytest.approx(29.0)


def test_derive_measurement_rejects_mixed_workload() -> None:
    with pytest.raises(ValueError, match="prompt or generation"):
        derive_measurement(row(n_gen=1))


def test_parse_max_rss_uses_final_time_record() -> None:
    stderr = "logs\n  123 maximum resident set size\n  456 maximum resident set size\n"

    assert parse_max_rss(stderr) == 456
    assert parse_max_rss("logs only") is None


def test_select_prompt_measurement_requires_exact_identity() -> None:
    measurement = derive_measurement(row())

    assert select_prompt_measurement([measurement], 192, 99, 12) == measurement
    with pytest.raises(ValueError, match="exactly once"):
        select_prompt_measurement([], 192, 99, 12)
