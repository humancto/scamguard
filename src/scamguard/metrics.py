"""Safety-oriented metrics shared by all model tracks."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= lower) & (probabilities < upper)
        if upper == 1.0:
            mask |= probabilities == 1.0
        if not mask.any():
            continue
        accuracy = y_true[mask].mean()
        confidence = probabilities[mask].mean()
        result += mask.mean() * abs(float(accuracy - confidence))
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if total <= 0:
        return [0.0, 1.0]
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return [lower, upper]


def binary_safety_metrics(
    y_true: np.ndarray, scam_probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (scam_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "scam_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "scam_precision_ci95": wilson_interval(int(tp), int(tp + fp)),
        "scam_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "scam_recall_ci95": wilson_interval(int(tp), int(tp + fn)),
        "scam_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_positive_rate_ci95": wilson_interval(int(fp), int(fp + tn)),
        "brier": float(brier_score_loss(y_true, scam_probabilities)),
        "ece_15": expected_calibration_error(y_true, scam_probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, max_fpr: float = 0.02) -> float:
    candidates = sorted({float(value) for value in probabilities}, reverse=True)
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        metrics = binary_safety_metrics(y_true, probabilities, threshold)
        if metrics["false_positive_rate"] <= max_fpr:
            feasible.append(
                (float(metrics["scam_recall"]), float(metrics["scam_precision"]), threshold)
            )
    if not feasible:
        return 1.0
    return max(feasible)[2]
