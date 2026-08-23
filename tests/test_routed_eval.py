import json
from pathlib import Path

import pytest

from training.eval_routed import (
    component_records,
    evaluate_records,
    fit_policy,
    join_split,
    latency_report,
    read_prediction_ledger,
    route_records,
)


def record(
    identifier: str,
    truth: str,
    verdict: str,
    probabilities: tuple[float, float, float],
    *,
    split: str = "dev",
    category: str = "NONE",
) -> dict[str, object]:
    return {
        "id": identifier,
        "split": split,
        "source": "fixture",
        "source_language": "English",
        "category": category,
        "truth": truth,
        "argmax": verdict,
        "calibrated_verdict": verdict,
        "threshold_scam": verdict == "SCAM",
        "probabilities": dict(zip(("SAFE", "UNCERTAIN", "SCAM"), probabilities, strict=True)),
    }


def test_policy_is_fit_on_dev_with_uncertain_rows_always_escalated() -> None:
    router = [
        record("safe-confident", "SAFE", "SAFE", (0.9, 0.05, 0.05)),
        record("scam-uncertain", "SCAM", "UNCERTAIN", (0.3, 0.4, 0.3), category="OTP"),
        record("scam-margin", "SCAM", "SAFE", (0.5, 0.4, 0.1), category="OTP"),
        record("safe-margin", "SAFE", "SAFE", (0.55, 0.35, 0.1)),
    ]
    specialist = [
        record("safe-confident", "SAFE", "SAFE", (0.9, 0.05, 0.05)),
        record("scam-uncertain", "SCAM", "SCAM", (0.05, 0.05, 0.9), category="OTP"),
        record("scam-margin", "SCAM", "SCAM", (0.05, 0.05, 0.9), category="OTP"),
        record("safe-margin", "SAFE", "SCAM", (0.05, 0.05, 0.9)),
    ]
    joined = list(zip(router, specialist, strict=True))

    margin_max, policy = fit_policy(joined, max_escalation_rate=0.5, max_fpr=0.0)
    routed = route_records(joined, margin_max)
    metrics = evaluate_records(routed)

    assert margin_max == pytest.approx(0.1)
    assert policy["selection_feasible"] is True
    assert metrics["escalation_rate"] == pytest.approx(0.5)
    assert metrics["binary_safety"]["scam_recall"] == 1.0
    assert metrics["binary_safety"]["false_positive_rate"] == 0.0


def test_ledger_reader_rejects_message_text_and_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    first = record("row-1", "SAFE", "SAFE", (0.9, 0.05, 0.05))
    path.write_text(json.dumps(first | {"text": "secret"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text-bearing fields"):
        read_prediction_ledger(path)

    path.write_text(
        json.dumps(first | {"metadata": {"transcript": "secret"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="text-bearing fields"):
        read_prediction_ledger(path)

    path.write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        read_prediction_ledger(path)


def test_join_rejects_truth_or_source_drift() -> None:
    router_record = record("row-1", "SAFE", "SAFE", (0.9, 0.05, 0.05))
    specialist_record = record("row-1", "SCAM", "SCAM", (0.05, 0.05, 0.9))
    key = ("dev", "row-1")

    with pytest.raises(ValueError, match="metadata mismatch.*truth"):
        join_split({key: router_record}, {key: specialist_record}, "dev")


def test_component_percentiles_do_not_masquerade_as_routed_p95() -> None:
    report = latency_report(
        escalation_rate=0.1,
        router_mean_ms=8.0,
        router_p95_ms=12.0,
        specialist_mean_ms=45.0,
        specialist_p95_ms=55.0,
    )

    assert report["analytical_expected_mean_ms"] == pytest.approx(12.5)
    assert report["specialist_path_conservative_p95_upper_bound_ms"] == 67.0
    assert report["routed_end_to_end_p95_ms"] is None


def test_component_baselines_bypass_routing_policy() -> None:
    router = record("row", "SCAM", "UNCERTAIN", (0.3, 0.4, 0.3))
    specialist = record("row", "SCAM", "SCAM", (0.05, 0.05, 0.9))
    joined = [(router, specialist)]

    router_only = component_records(joined, "router")
    specialist_only = component_records(joined, "specialist")

    assert router_only[0]["final_verdict"] == "UNCERTAIN"
    assert router_only[0]["escalated"] is False
    assert specialist_only[0]["final_verdict"] == "SCAM"
    assert specialist_only[0]["escalated"] is True
