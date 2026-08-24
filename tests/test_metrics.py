from __future__ import annotations

import numpy as np

from scamguard.linear_baseline import build_pipeline
from scamguard.metrics import (
    binary_safety_metrics,
    choose_threshold,
    choose_threshold_for_gates,
    wilson_interval,
)
from training.train_linear import evaluate


def test_threshold_selection_honors_fpr_cap_then_maximizes_recall() -> None:
    truth = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.01, 0.20, 0.51, 0.40, 0.70, 0.90])

    threshold = choose_threshold(truth, probabilities, max_fpr=0.0)
    metrics = binary_safety_metrics(truth, probabilities, threshold)

    assert threshold == 0.7
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["scam_recall"] == 2 / 3


def test_gate_threshold_chooses_highest_value_meeting_recall_and_fpr() -> None:
    truth = np.array([0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.01, 0.20, 0.51, 0.40, 0.60, 0.70, 0.90])

    threshold = choose_threshold_for_gates(
        truth, probabilities, min_recall=0.75, max_fpr=0.0
    )

    assert threshold == 0.6


def test_gate_threshold_returns_none_when_contract_is_infeasible() -> None:
    threshold = choose_threshold_for_gates(
        np.array([0, 1]), np.array([0.8, 0.2]), min_recall=1.0, max_fpr=0.0
    )

    assert threshold is None


def test_metrics_publish_raw_confusion_counts() -> None:
    metrics = binary_safety_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.9]), threshold=0.75
    )

    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["scam_recall_ci95"] == wilson_interval(1, 2)
    assert metrics["false_positive_rate_ci95"] == wilson_interval(1, 2)


def test_wilson_interval_handles_boundary_and_empty_samples() -> None:
    assert wilson_interval(0, 0) == [0.0, 1.0]
    lower, upper = wilson_interval(0, 100)
    assert lower == 0.0
    assert 0.03 < upper < 0.04


def test_slice_with_only_uncertain_rows_has_no_binary_metric() -> None:
    pipeline = build_pipeline()
    pipeline.fit(
        [
            "safe message dinner",
            "safe message lunch",
            "uncertain message advertising",
            "uncertain message promotion",
            "scam message password link",
            "scam message urgent link",
        ],
        ["SAFE", "SAFE", "UNCERTAIN", "UNCERTAIN", "SCAM", "SCAM"],
    )

    result = evaluate(
        pipeline,
        [
            {
                "text": "ambiguous message",
                "label": "UNCERTAIN",
                "category": "NONE",
                "source": "test",
            }
        ],
        threshold=0.5,
    )

    assert result["binary_safety"] is None
    assert result["binary_subset_empty"] is True
