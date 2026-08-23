"""Shared calibrated verdict policy for runtime and evaluation ledgers."""

from __future__ import annotations


def calibrated_verdict(
    *,
    safe_probability: float,
    scam_probability: float,
    scam_probability_threshold: float,
    safe_probability_threshold: float,
    safe_max_scam_probability: float | None,
) -> str:
    values = (
        safe_probability,
        scam_probability,
        scam_probability_threshold,
        safe_probability_threshold,
    )
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probabilities and thresholds must be in [0, 1]")
    if safe_max_scam_probability is not None and not 0.0 <= safe_max_scam_probability <= 1.0:
        raise ValueError("safe maximum scam probability must be in [0, 1]")
    if scam_probability >= scam_probability_threshold:
        return "SCAM"
    safe_allowed = (
        safe_max_scam_probability is None
        or scam_probability < safe_max_scam_probability
    )
    if safe_probability >= safe_probability_threshold and safe_allowed:
        return "SAFE"
    return "UNCERTAIN"
