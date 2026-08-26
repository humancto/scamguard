"""Dev-only Qwen score interpolation is deterministic and leakage-resistant."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_qwen_score_blend import (
    blend_probabilities,
    fit_blend,
    select_scam_threshold,
)

LABELS = ("SAFE", "UNCERTAIN", "SCAM")


def record(
    identifier: str,
    truth: str,
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[dict[str, object], dict[str, object]]:
    shared: dict[str, object] = {
        "id": identifier,
        "split": "dev",
        "source": "fixture",
        "source_language": "en",
        "category": "NONE" if truth == "SAFE" else "IMPERSONATION",
        "truth": truth,
    }

    def side(probabilities: tuple[float, float, float]) -> dict[str, object]:
        return shared | {
            "probabilities": dict(zip(LABELS, probabilities, strict=True)),
        }

    return side(left), side(right)


def test_blend_endpoints_and_log_linear_normalization() -> None:
    left = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    right = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])

    assert np.allclose(
        blend_probabilities(left, right, right_weight=0.0, method="arithmetic"), left
    )
    assert np.allclose(
        blend_probabilities(left, right, right_weight=1.0, method="log_linear"), right
    )
    middle = blend_probabilities(left, right, right_weight=0.5, method="log_linear")
    assert np.allclose(middle.sum(axis=1), 1.0)
    assert np.all(middle >= 0.0)


def test_fast_threshold_matches_recall_fpr_contract_and_fallback() -> None:
    truth = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.1])

    threshold, feasible = select_scam_threshold(
        truth, probabilities, min_recall=1.0, max_fpr=0.0
    )
    assert feasible is True
    assert threshold == pytest.approx(0.8)

    threshold, feasible = select_scam_threshold(
        truth, probabilities, min_recall=1.0, max_fpr=0.49
    )
    assert feasible is True
    assert threshold == pytest.approx(0.8)

    impossible = np.array([0.9, 0.1, 0.8, 0.7])
    threshold, feasible = select_scam_threshold(
        truth, impossible, min_recall=1.0, max_fpr=0.0
    )
    assert feasible is False
    assert threshold == pytest.approx(0.9)


def test_fit_blend_uses_only_supplied_dev_records_and_fits_safe_threshold() -> None:
    joined = [
        record("safe-1", "SAFE", (0.8, 0.1, 0.1), (0.9, 0.05, 0.05)),
        record("safe-2", "SAFE", (0.7, 0.2, 0.1), (0.8, 0.1, 0.1)),
        record("scam-1", "SCAM", (0.1, 0.1, 0.8), (0.05, 0.05, 0.9)),
        record("scam-2", "SCAM", (0.2, 0.1, 0.7), (0.1, 0.1, 0.8)),
        record("uncertain", "UNCERTAIN", (0.2, 0.7, 0.1), (0.1, 0.8, 0.1)),
    ]

    selected, candidates, probabilities = fit_blend(
        joined, alpha_steps=4, min_recall=1.0, max_fpr=0.0
    )

    assert selected["joint_dev_contract_satisfied"] is True
    assert selected["selection_used_non_dev_labels"] is False
    assert selected["dev_macro_f1"] == pytest.approx(1.0)
    assert 0.0 <= selected["safe_threshold"] <= 1.0
    assert len(candidates) == 10
    assert probabilities.shape == (5, 3)


def test_invalid_shapes_weights_and_methods_fail_closed() -> None:
    matrix = np.array([[0.8, 0.1, 0.1]])
    with pytest.raises(ValueError, match="matching N x 3"):
        blend_probabilities(matrix, matrix[:, :2], right_weight=0.5, method="arithmetic")
    with pytest.raises(ValueError, match="right_weight"):
        blend_probabilities(matrix, matrix, right_weight=1.1, method="arithmetic")
    with pytest.raises(ValueError, match="unsupported"):
        blend_probabilities(matrix, matrix, right_weight=0.5, method="unknown")
