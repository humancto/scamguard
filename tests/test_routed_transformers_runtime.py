import json
from pathlib import Path

import pytest

from benchmarks.benchmark_routed_transformers_runtime import (
    adapter_identity,
    latency_report,
    trace_requests,
    verify_backend_calibration,
    verify_score_cache_identity,
)
from scamguard.model import ModelScores


class StubRuntime:
    model_id = "stub"
    scam_threshold = 0.6
    safe_probability_threshold = 0.8
    safe_max_scam_probability = 0.2

    def __init__(self, scores: dict[str, ModelScores]) -> None:
        self.scores = scores

    def predict(self, text: str) -> ModelScores:
        return self.scores[text]


def prediction(
    identifier: str,
    verdict: str,
    scores: ModelScores,
) -> dict[str, object]:
    return {
        "id": identifier,
        "calibrated_verdict": verdict,
        "probabilities": {
            "SAFE": scores.safe,
            "UNCERTAIN": scores.uncertain,
            "SCAM": scores.scam,
        },
    }


def test_persistent_route_trace_preserves_decisions_and_exposes_tail() -> None:
    safe_scores = ModelScores(safe=0.9, uncertain=0.05, scam=0.05)
    uncertain_scores = ModelScores(safe=0.3, uncertain=0.4, scam=0.3)
    scam_scores = ModelScores(safe=0.05, uncertain=0.05, scam=0.9)
    rows = [
        {"id": "safe", "split": "test", "text": "private safe phrase"},
        {"id": "uncertain", "split": "test", "text": "private uncertain phrase"},
    ]
    expected_router = {
        "safe": prediction("safe", "SAFE", safe_scores),
        "uncertain": prediction("uncertain", "UNCERTAIN", uncertain_scores),
    }
    expected_specialist = {
        "safe": prediction("safe", "SAFE", safe_scores),
        "uncertain": prediction("uncertain", "SCAM", scam_scores),
    }
    expected_final = {
        "safe": {"escalated": False, "final_verdict": "SAFE"},
        "uncertain": {"escalated": True, "final_verdict": "SCAM"},
    }
    router = StubRuntime(
        {"private safe phrase": safe_scores, "private uncertain phrase": uncertain_scores}
    )
    specialist = StubRuntime(
        {"private safe phrase": safe_scores, "private uncertain phrase": scam_scores}
    )

    traces, parity = trace_requests(
        rows,
        expected_router,
        expected_specialist,
        expected_final,
        router,
        specialist,
        margin_max=-1.0,
        repetitions=2,
        probability_tolerance=1e-8,
    )
    report = latency_report(traces)

    assert len(traces) == 4
    assert sum(record["escalated"] for record in traces) == 2
    assert report["escalation_rate"] == 0.5
    assert report["fast_path_total"]["samples"] == 2
    assert report["escalated_path_total"]["samples"] == 2
    assert parity["all_final_decisions_match"] is True
    assert parity["release_gate_passed"] is True
    assert "private" not in json.dumps(traces)


def test_persistent_route_trace_reports_probability_drift() -> None:
    scores = ModelScores(safe=0.9, uncertain=0.05, scam=0.05)
    rows = [{"id": "row", "split": "test", "text": "private phrase"}]
    expected_router = {"row": prediction("row", "SAFE", scores)}
    expected_router["row"]["probabilities"]["SAFE"] = 0.8  # type: ignore[index]
    expected_specialist = {"row": prediction("row", "SAFE", scores)}
    expected_final = {"row": {"escalated": False, "final_verdict": "SAFE"}}
    backend = StubRuntime({"private phrase": scores})

    _traces, parity = trace_requests(
        rows,
        expected_router,
        expected_specialist,
        expected_final,
        backend,
        backend,
        margin_max=-1.0,
        repetitions=1,
        probability_tolerance=1e-8,
    )

    assert parity["unique_router_probability_drift_ids"] == 1
    assert parity["all_router_decisions_match"] is True


def test_persistent_route_trace_rejects_decision_drift_with_small_probability_error() -> None:
    runtime_scores = ModelScores(safe=0.79, uncertain=0.16, scam=0.05)
    ledger_scores = ModelScores(safe=0.81, uncertain=0.14, scam=0.05)
    rows = [{"id": "row", "split": "test", "text": "private phrase"}]
    expected_router = {"row": prediction("row", "SAFE", ledger_scores)}
    expected_specialist = {"row": prediction("row", "SAFE", ledger_scores)}
    expected_final = {"row": {"escalated": False, "final_verdict": "SAFE"}}
    backend = StubRuntime({"private phrase": runtime_scores})

    _traces, parity = trace_requests(
        rows,
        expected_router,
        expected_specialist,
        expected_final,
        backend,
        backend,
        margin_max=-1.0,
        repetitions=1,
        probability_tolerance=0.05,
    )

    assert parity["unique_router_probability_drift_ids"] == 0
    assert parity["unique_router_decision_mismatch_ids"] == 1
    assert parity["release_gate_passed"] is False


def test_score_cache_identity_fails_closed_on_batch_contract() -> None:
    metadata = {"model": "pinned", "batch_size": 16}
    verify_score_cache_identity(metadata, {"model": "pinned"})

    with pytest.raises(ValueError, match="batch_size"):
        verify_score_cache_identity({"model": "pinned", "batch_size": 0}, {"model": "pinned"})


def test_adapter_identity_binds_report_path_and_weights(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"pinned weights")

    digest, record = adapter_identity({"adapter": str(adapter)}, adapter)

    assert digest == record["adapter_sha256"]
    assert record["kind"] == "lora_adapter"
    with pytest.raises(ValueError, match="different LoRA adapter"):
        adapter_identity({"adapter": str(tmp_path / "other")}, adapter)


def test_runtime_backend_calibration_must_match_report() -> None:
    backend = StubRuntime({})
    backend.temperature = 1.5  # type: ignore[attr-defined]
    backend.sequence_bucket_size = 64  # type: ignore[attr-defined]
    report = {
        "model": "Qwen/example",
        "base_model_revision": "pinned",
        "temperature": 1.5,
        "scam_threshold": backend.scam_threshold,
        "safe_threshold": backend.safe_probability_threshold,
        "score_cache": {"sequence_bucket_size": 64},
    }

    verify_backend_calibration(
        backend, report, model="Qwen/example", revision="pinned"
    )
    report["safe_threshold"] = 0.7
    with pytest.raises(ValueError, match="safe_probability_threshold"):
        verify_backend_calibration(
            backend, report, model="Qwen/example", revision="pinned"
        )
