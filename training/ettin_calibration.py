"""Development-only calibration using the batch-one Transformers serving arithmetic.

This module does not read datasets or write artifacts. The caller supplies only the
hash-pinned development rows and owns checkpoint/config bindings and persistence.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scamguard.decision import calibrated_verdict
from scamguard.metrics import binary_safety_metrics, verdict_alert_metrics
from scamguard.preprocessing import DIALOGUE_POLICIES, prepare_model_text
from training.train_encoder import LABEL_TO_ID, LABELS, fit_temperature

SERVING_CALIBRATION_VERSION = "ettin-serving-batch1-torch-fp32-v1"
MIN_INTERIOR_MARGIN_ULPS = 1.0


def decision_mask(scores: Sequence[float], threshold: float) -> list[bool]:
    """Match SDK Python-float >=, without NumPy casting the threshold to float32."""
    return [float(score) >= float(threshold) for score in scores]


def select_stable_threshold(
    truth: Sequence[int],
    scores: Sequence[float],
    *,
    all_scores: Sequence[float] | None = None,
    max_fpr: float = 0.02,
) -> tuple[float, dict[str, Any]]:
    """Preserve the optimal observed-score mask while moving inside its interval.

    The objective is max recall, then precision, then the highest observed threshold,
    subject to SAFE FPR <= max_fpr. All three labels' scores delimit the interval so
    moving a threshold cannot silently change UNCERTAIN-reference alerts either.
    """
    values = [float(value) for value in scores]
    labels = list(truth)
    population = values if all_scores is None else [float(value) for value in all_scores]
    if not values or len(values) != len(labels) or set(labels) != {0, 1}:
        raise ValueError("threshold fitting requires aligned SAFE and SCAM development rows")
    if not math.isfinite(max_fpr) or not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be finite and in [0, 1]")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in population + values):
        raise ValueError("development scores must be finite probabilities")
    if not set(values) <= set(population):
        raise ValueError("all_scores must contain the binary subset's scores")
    negatives, positives = labels.count(0), labels.count(1)

    def counts(threshold: float) -> tuple[int, int]:
        predicted = decision_mask(values, threshold)
        return (
            sum(flag and label == 1 for flag, label in zip(predicted, labels, strict=True)),
            sum(flag and label == 0 for flag, label in zip(predicted, labels, strict=True)),
        )

    feasible = []
    for candidate in sorted(set(values), reverse=True):
        tp, fp = counts(candidate)
        if fp / negatives <= max_fpr:
            feasible.append((tp / positives, tp / (tp + fp) if tp + fp else 0.0, candidate))
    selection_kind = "optimal_observed_score"
    if feasible:
        observed = max(feasible)[2]
    else:
        # 1.0 is not guaranteed to mean no alerts: saturated SAFE scores may equal 1.
        observed = 1.0
        _, fp = counts(observed)
        if fp / negatives > max_fpr:
            raise ValueError("no feasible probability threshold in [0, 1] for development FPR")
        selection_kind = "no_feasible_observed_score; verified_1.0_boundary"

    expected_mask = decision_mask(population, observed)
    excluded = [value for value, flag in zip(population, expected_mask, strict=True) if not flag]
    included = [value for value, flag in zip(population, expected_mask, strict=True) if flag]
    lower, upper = max(excluded, default=0.0), min(included, default=1.0)
    midpoint = lower + (upper - lower) / 2.0
    # Inputs originate as float32 serving probabilities. Require at least one local
    # float32 ULP of midpoint margin; this is not a model-drift robustness claim.
    ulp = max(float(np.spacing(np.float32(lower))), float(np.spacing(np.float32(upper))))
    candidate_margin = min(abs(value - midpoint) for value in population)
    interior = (
        lower < midpoint < upper
        and candidate_margin >= MIN_INTERIOR_MARGIN_ULPS * ulp
        and decision_mask(population, midpoint) == expected_mask
    )
    threshold = midpoint if interior else observed
    final_mask = decision_mask(population, threshold)
    if final_mask != expected_mask:
        raise RuntimeError("stable threshold changed the optimal observed decision mask")
    tp, fp = counts(threshold)
    if fp / negatives > max_fpr:
        raise RuntimeError("stable threshold violated development SAFE FPR")
    return threshold, {
        "objective": (
            "max recall under SAFE FPR cap, then precision, then highest observed threshold"
        ),
        "selection_kind": selection_kind,
        "observed_threshold": observed,
        "threshold": threshold,
        "threshold_location": "stable_interval_midpoint" if interior else "exact_boundary_fallback",
        "largest_excluded_score": max(excluded) if excluded else None,
        "smallest_included_score": min(included) if included else None,
        "minimum_score_margin": min(abs(value - threshold) for value in population),
        "minimum_interior_margin_ulps": MIN_INTERIOR_MARGIN_ULPS,
        "interval_float32_ulp": ulp,
        "mask_unchanged": True,
        "comparison": "Python float >= Python float; no float32 threshold cast",
        "max_safe_fpr": max_fpr,
        "safe_examples": negatives,
        "scam_examples": positives,
        "tp": tp,
        "fp": fp,
        "scam_recall": tp / positives,
        "safe_fpr": fp / negatives,
    }


def serving_probabilities(logits: np.ndarray, temperature: float, device: Any) -> list[list[float]]:
    """Replay each captured 3-vector with the SDK's one-dimensional Torch softmax."""
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise ValueError("serving logits must be finite N-by-3 vectors")
    if logits.dtype != np.float32:
        raise ValueError("serving calibration requires captured float32 logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    probabilities = []
    with torch.inference_mode():
        for values in logits:
            # Division and softmax stay on the model device, as in TransformersBackend.predict.
            vector = torch.tensor(values, dtype=torch.float32, device=device) / temperature
            scores = torch.softmax(vector, dim=-1).detach().cpu().tolist()
            if (
                any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores)
                or abs(sum(scores) - 1.0) > 1e-4
            ):
                raise ValueError("serving softmax returned invalid probabilities")
            probabilities.append(scores)
    return probabilities


def _metrics(rows: list[dict], records: list[dict], threshold: float) -> dict[str, Any]:
    truth = np.array([LABEL_TO_ID[row["label"]] for row in rows])
    argmax = np.array([LABEL_TO_ID[row["argmax"]] for row in records])
    verdicts = np.array([LABEL_TO_ID[row["calibrated_verdict"]] for row in records])
    indices = [index for index, row in enumerate(rows) if row["label"] in {"SAFE", "SCAM"}]
    binary_truth = np.array([int(rows[index]["label"] == "SCAM") for index in indices])
    # Explicit float64 preserves the SDK's Python-float threshold comparison.
    scores = np.array(
        [records[index]["probabilities"]["SCAM"] for index in indices], dtype=np.float64
    )
    return {
        "examples": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "accuracy_argmax": float(accuracy_score(truth, argmax)),
        "macro_f1_argmax": float(
            f1_score(truth, argmax, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "confusion_argmax": confusion_matrix(truth, argmax, labels=[0, 1, 2]).tolist(),
        "binary_safety": binary_safety_metrics(binary_truth, scores, threshold),
        "calibrated": {
            "accuracy": float(accuracy_score(truth, verdicts)),
            "macro_f1": float(
                f1_score(truth, verdicts, labels=[0, 1, 2], average="macro", zero_division=0)
            ),
            "confusion": confusion_matrix(truth, verdicts, labels=[0, 1, 2]).tolist(),
            "labels": list(LABELS),
            "alert_composition": verdict_alert_metrics(truth, verdicts),
        },
    }


def calibrate_serving_dev(
    model: Any,
    tokenizer: Any,
    rows: list[dict],
    max_length: int,
    dialogue_policy: str,
) -> tuple[dict, dict, list[dict]]:
    """Fit only supplied dev rows; CPU is supported for offline tests, not MPS evidence."""
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if dialogue_policy not in DIALOGUE_POLICIES:
        raise ValueError("unknown dialogue policy")
    if not rows or {row.get("label") for row in rows} != set(LABELS):
        raise ValueError("development calibration requires all three verdict labels")
    identifiers = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("development rows require stable string IDs")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate development IDs")
    if any(
        not isinstance(row.get("text"), str)
        or not row["text"].strip()
        or len(row["text"].strip()) > 100_000
        for row in rows
    ):
        raise ValueError("development text violates the SDK input contract")
    truncation_side = getattr(tokenizer, "truncation_side", None)
    if truncation_side not in {"left", "right"}:
        raise ValueError("tokenizer must declare a valid truncation side")
    device = torch.device(model.device)
    model.eval()
    captured = []
    with torch.inference_mode():
        for row in rows:
            prepared = prepare_model_text(row["text"].strip(), dialogue_policy)
            encoded = tokenizer(
                prepared, return_tensors="pt", truncation=True, max_length=max_length
            )
            inputs = {name: value.to(device) for name, value in encoded.items()}
            ids = inputs["input_ids"]
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise ValueError("serving calibration must tokenize exactly one message")
            output = model(**inputs).logits
            if output.shape != (1, 3) or output.dtype != torch.float32:
                raise ValueError(
                    "serving calibration requires batch-one, three-class float32 logits"
                )
            if output.device != device or not bool(torch.isfinite(output).all()):
                raise ValueError("serving logits must be finite and on the model device")
            captured.append(output[0, :3].detach().cpu().numpy().copy())
    logits = np.stack(captured)
    truth = np.array([LABEL_TO_ID[row["label"]] for row in rows])
    temperature = fit_temperature(logits, truth)
    probabilities = serving_probabilities(logits, temperature, device)
    binary_indices = [index for index, row in enumerate(rows) if row["label"] != "UNCERTAIN"]
    threshold, threshold_selection = select_stable_threshold(
        [int(rows[index]["label"] == "SCAM") for index in binary_indices],
        [probabilities[index][2] for index in binary_indices],
        all_scores=[values[2] for values in probabilities],
    )
    records = []
    for row, scores, raw in zip(rows, probabilities, logits, strict=True):
        verdict = calibrated_verdict(
            safe_probability=scores[0],
            scam_probability=scores[2],
            scam_probability_threshold=threshold,
            safe_probability_threshold=0.8,
            safe_max_scam_probability=0.2,
        )
        records.append(
            {
                "id": row["id"],
                "split": "dev",
                "source": row.get("source", "UNKNOWN"),
                "source_language": row.get("source_language"),
                "category": row.get("category", "UNKNOWN"),
                "truth": row["label"],
                "argmax": LABELS[max(range(3), key=scores.__getitem__)],
                "calibrated_verdict": verdict,
                "threshold_scam": scores[2] >= threshold,
                "probabilities": dict(zip(LABELS, scores, strict=True)),
                "logits": raw.tolist(),
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            }
        )
    calibration = {
        "temperature": temperature,
        "scam_threshold": threshold,
        "safe_threshold": 0.2,
        "safe_threshold_semantics": "maximum_safe_risk",
        "labels": list(LABELS),
        "threshold_source": "dev SAFE/SCAM only",
        "safe_threshold_source": "fixed existing encoder policy; not fitted",
        "calibration_version": SERVING_CALIBRATION_VERSION,
        "serving_contract": {
            "device": str(device),
            "message_batch_size": 1,
            "logits_dtype": "float32",
            "softmax": "torch.softmax((logits[0, :3] / temperature), dim=-1)",
            "softmax_vector_shape": [3],
            "padding": False,
            "text_preprocessing": "strip then prepare_model_text",
            "dialogue_policy": dialogue_policy,
            "max_length": max_length,
            "truncation_side": truncation_side,
        },
        "threshold_selection": threshold_selection,
    }
    return calibration, _metrics(rows, records, threshold), records
