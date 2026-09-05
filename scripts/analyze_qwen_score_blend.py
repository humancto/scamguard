#!/usr/bin/env python3
"""Test whether two frozen Qwen adapters have a useful dev-only score interpolation.

This is a diagnostic, not a deployable model: running both adapters would roughly
double adapter inference work and does not meet ScamGuard's mobile product contract.
The script consumes text-free prediction ledgers and never uses non-dev splits to
select the blend method, weight, or decision thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scamguard.metrics import (
    binary_safety_metrics,
    choose_safe_abstention_threshold,
    file_sha256,
    wilson_interval,
)

LABELS = ("SAFE", "UNCERTAIN", "SCAM")
METHODS = ("arithmetic", "log_linear")
REQUIRED_FIELDS = {"id", "split", "source", "category", "truth", "probabilities"}
FORBIDDEN_TEXT_FIELDS = {"text", "message", "prompt", "conversation", "transcript"}

choose_safe_threshold = choose_safe_abstention_threshold


def forbidden_text_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_TEXT_FIELDS:
                found.add(str(key))
            found.update(forbidden_text_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.update(forbidden_text_fields(child))
    return found


def read_prediction_ledger(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not REQUIRED_FIELDS <= set(record):
                raise ValueError(f"{path}:{line_number}: invalid prediction schema")
            forbidden = forbidden_text_fields(record)
            if forbidden:
                raise ValueError(
                    f"{path}:{line_number}: text-bearing fields are forbidden: {sorted(forbidden)}"
                )
            if record["truth"] not in LABELS:
                raise ValueError(f"{path}:{line_number}: invalid truth label")
            probabilities = record["probabilities"]
            if not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
                raise ValueError(f"{path}:{line_number}: invalid probability labels")
            values = [probabilities[label] for label in LABELS]
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and 0.0 <= value <= 1.0
                for value in values
            ) or not math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(f"{path}:{line_number}: invalid probabilities")
            key = (str(record["split"]), str(record["id"]))
            if not all(key):
                raise ValueError(f"{path}:{line_number}: invalid split or id")
            if key in records:
                raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
            records[key] = record
    if not records:
        raise ValueError(f"{path}: prediction ledger is empty")
    return records


def join_split(
    left: dict[tuple[str, str], dict[str, Any]],
    right: dict[tuple[str, str], dict[str, Any]],
    split: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left_keys = {key for key in left if key[0] == split}
    right_keys = {key for key in right if key[0] == split}
    if not left_keys or left_keys != right_keys:
        raise ValueError(f"ledger key mismatch for split {split!r}")
    joined = []
    for key in sorted(left_keys):
        left_record = left[key]
        right_record = right[key]
        for field in ("truth", "source", "source_language", "source_domain", "category"):
            if left_record.get(field) != right_record.get(field):
                raise ValueError(f"ledger metadata mismatch for {key!r}: {field}")
        joined.append((left_record, right_record))
    return joined


def probability_matrix(
    joined: list[tuple[dict[str, Any], dict[str, Any]]], side: int
) -> np.ndarray:
    """Return label-ordered probabilities for one side of a joined ledger."""

    return np.array(
        [
            [float(pair[side]["probabilities"][label]) for label in LABELS]
            for pair in joined
        ],
        dtype=np.float64,
    )


def blend_probabilities(
    left: np.ndarray,
    right: np.ndarray,
    *,
    right_weight: float,
    method: str,
) -> np.ndarray:
    """Blend calibrated class probabilities while preserving normalized rows."""

    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != len(LABELS):
        raise ValueError("probability matrices must have matching N x 3 shapes")
    if not 0.0 <= right_weight <= 1.0:
        raise ValueError("right_weight must be in [0, 1]")
    if method == "arithmetic":
        blended = (1.0 - right_weight) * left + right_weight * right
    elif method == "log_linear":
        epsilon = np.finfo(np.float64).tiny
        log_blended = (1.0 - right_weight) * np.log(np.clip(left, epsilon, 1.0))
        log_blended += right_weight * np.log(np.clip(right, epsilon, 1.0))
        log_blended -= log_blended.max(axis=1, keepdims=True)
        blended = np.exp(log_blended)
    else:
        raise ValueError(f"unsupported blend method: {method}")
    totals = blended.sum(axis=1, keepdims=True)
    if not np.isfinite(blended).all() or np.any(totals <= 0.0):
        raise ValueError("blend produced invalid probabilities")
    return blended / totals


def select_scam_threshold(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_recall: float,
    max_fpr: float,
) -> tuple[float, bool]:
    """Match the frozen evaluator's exact threshold policy in O(n log n)."""

    if len(truth) != len(probabilities) or not len(truth):
        raise ValueError("binary truth and probabilities must be non-empty and aligned")
    if not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("binary truth must contain only zero and one")
    if not np.isfinite(probabilities).all():
        raise ValueError("SCAM probabilities must be finite")

    positives = int(np.sum(truth == 1))
    negatives = int(np.sum(truth == 0))
    order = np.argsort(-probabilities, kind="stable")
    true_sorted = truth[order]
    scores_sorted = probabilities[order]
    tp = 0
    fp = 0
    fallback: list[tuple[float, float, float]] = []
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and scores_sorted[end] == scores_sorted[index]:
            end += 1
        group = true_sorted[index:end]
        tp += int(np.sum(group == 1))
        fp += int(np.sum(group == 0))
        recall = tp / max(positives, 1)
        fpr = fp / max(negatives, 1)
        precision = tp / max(tp + fp, 1)
        threshold = float(scores_sorted[index])
        if recall >= min_recall and fpr <= max_fpr:
            return threshold, True
        if fpr <= max_fpr:
            fallback.append((recall, precision, threshold))
        index = end
    return (max(fallback)[2] if fallback else 1.0), False


def predict_with_abstention(
    probabilities: np.ndarray, scam_threshold: float, safe_threshold: float
) -> np.ndarray:
    predicted = np.full(len(probabilities), LABELS.index("UNCERTAIN"), dtype=np.int64)
    predicted[probabilities[:, LABELS.index("SAFE")] >= safe_threshold] = LABELS.index("SAFE")
    predicted[probabilities[:, LABELS.index("SCAM")] >= scam_threshold] = LABELS.index("SCAM")
    return predicted


def binary_dev_arrays(
    joined: list[tuple[dict[str, Any], dict[str, Any]]], probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.array([left["truth"] in {"SAFE", "SCAM"} for left, _ in joined])
    truth = np.array([int(left["truth"] == "SCAM") for left, _ in joined], dtype=np.int64)
    return truth[mask], probabilities[mask, LABELS.index("SCAM")]


def candidate_summary(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    probabilities: np.ndarray,
    *,
    method: str,
    right_weight: float,
    min_recall: float,
    max_fpr: float,
) -> dict[str, Any]:
    binary_truth, scam_probabilities = binary_dev_arrays(joined, probabilities)
    threshold, contract_satisfied = select_scam_threshold(
        binary_truth,
        scam_probabilities,
        min_recall=min_recall,
        max_fpr=max_fpr,
    )
    metrics = binary_safety_metrics(binary_truth, scam_probabilities, threshold)
    return {
        "method": method,
        "right_weight": right_weight,
        "scam_threshold": threshold,
        "joint_dev_contract_satisfied": contract_satisfied,
        "scam_recall": metrics["scam_recall"],
        "false_positive_rate": metrics["false_positive_rate"],
        "scam_precision": metrics["scam_precision"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
    }


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, ...]:
    """Rank only dev safety, preferring a single adapter on exact metric ties."""

    feasible = bool(candidate["joint_dev_contract_satisfied"])
    endpoint_preference = abs(float(candidate["right_weight"]) - 0.5)
    method_preference = 1.0 if candidate["method"] == "arithmetic" else 0.0
    if feasible:
        return (
            1.0,
            -float(candidate["false_positive_rate"]),
            float(candidate["scam_recall"]),
            float(candidate["scam_precision"]),
            float(candidate["scam_threshold"]),
            endpoint_preference,
            method_preference,
            -float(candidate["right_weight"]),
        )
    return (
        0.0,
        float(candidate["scam_recall"]),
        -float(candidate["false_positive_rate"]),
        float(candidate["scam_precision"]),
        float(candidate["scam_threshold"]),
        endpoint_preference,
        method_preference,
        -float(candidate["right_weight"]),
    )


def fit_blend(
    dev_joined: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    alpha_steps: int,
    min_recall: float,
    max_fpr: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Select method, right-side weight, and both thresholds using dev only."""

    if alpha_steps < 1:
        raise ValueError("alpha_steps must be positive")
    left = probability_matrix(dev_joined, 0)
    right = probability_matrix(dev_joined, 1)
    dev_truth = np.array(
        [LABELS.index(str(left_record["truth"])) for left_record, _ in dev_joined]
    )
    candidates = []
    matrices: dict[tuple[str, float], np.ndarray] = {}
    for method in METHODS:
        for step in range(alpha_steps + 1):
            right_weight = step / alpha_steps
            probabilities = blend_probabilities(
                left, right, right_weight=right_weight, method=method
            )
            candidate = candidate_summary(
                dev_joined,
                probabilities,
                method=method,
                right_weight=right_weight,
                min_recall=min_recall,
                max_fpr=max_fpr,
            )
            safe_threshold = choose_safe_threshold(
                dev_truth, probabilities, float(candidate["scam_threshold"])
            )
            calibrated = predict_with_abstention(
                probabilities, float(candidate["scam_threshold"]), safe_threshold
            )
            candidate["safe_threshold"] = safe_threshold
            candidate["dev_accuracy"] = float(accuracy_score(dev_truth, calibrated))
            candidate["dev_macro_f1"] = float(
                f1_score(dev_truth, calibrated, average="macro", zero_division=0)
            )
            candidates.append(candidate)
            matrices[(method, right_weight)] = probabilities

    selected = max(candidates, key=candidate_rank).copy()
    selected_probabilities = matrices[(selected["method"], selected["right_weight"])]
    selected["selection_used_non_dev_labels"] = False
    return selected, candidates, selected_probabilities


def evaluate_split(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    probabilities: np.ndarray,
    *,
    scam_threshold: float,
    safe_threshold: float,
) -> dict[str, Any]:
    truth = np.array([LABELS.index(str(left["truth"])) for left, _ in joined])
    predicted = predict_with_abstention(probabilities, scam_threshold, safe_threshold)
    binary_truth, scam_probabilities = binary_dev_arrays(joined, probabilities)
    binary = binary_safety_metrics(binary_truth, scam_probabilities, scam_threshold)
    by_category = {}
    for category in sorted(
        {str(left["category"]) for left, _ in joined if left["truth"] == "SCAM"}
    ):
        indices = [
            index
            for index, (left, _) in enumerate(joined)
            if left["truth"] == "SCAM" and left["category"] == category
        ]
        detected = int(
            np.sum(probabilities[indices, LABELS.index("SCAM")] >= scam_threshold)
        )
        by_category[category] = {
            "examples": len(indices),
            "detected": detected,
            "recall": detected / len(indices),
            "recall_ci95": wilson_interval(detected, len(indices)),
        }
    return {
        "examples": len(joined),
        "labels": dict(Counter(str(left["truth"]) for left, _ in joined)),
        "binary_safety": binary,
        "calibrated_decision": {
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "confusion": confusion_matrix(
                truth, predicted, labels=list(range(len(LABELS)))
            ).tolist(),
        },
        "scam_by_category": by_category,
    }


def prediction_records(
    joined_by_split: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    probabilities_by_split: dict[str, np.ndarray],
    *,
    scam_threshold: float,
    safe_threshold: float,
) -> list[dict[str, Any]]:
    records = []
    for split in sorted(joined_by_split):
        joined = joined_by_split[split]
        probabilities = probabilities_by_split[split]
        calibrated = predict_with_abstention(probabilities, scam_threshold, safe_threshold)
        for (left, _), values, verdict in zip(joined, probabilities, calibrated, strict=True):
            records.append(
                {
                    "id": left["id"],
                    "split": split,
                    "source": left["source"],
                    "source_language": left.get("source_language"),
                    "source_domain": left.get("source_domain"),
                    "category": left["category"],
                    "truth": left["truth"],
                    "argmax": LABELS[int(values.argmax())],
                    "calibrated_verdict": LABELS[int(verdict)],
                    "threshold_scam": bool(values[LABELS.index("SCAM")] >= scam_threshold),
                    "probabilities": {
                        label: float(values[index]) for index, label in enumerate(LABELS)
                    },
                }
            )
    return records


def comparison_splits(
    left: dict[tuple[str, str], dict[str, Any]],
    right: dict[tuple[str, str], dict[str, Any]],
    *,
    dev_only: bool,
) -> list[str]:
    """Resolve explicitly comparable splits without silently dropping rows."""

    if dev_only:
        join_split(left, right, "dev")
        return ["dev"]
    if set(left) != set(right):
        missing = sorted(set(left) - set(right))[:3]
        extra = sorted(set(right) - set(left))[:3]
        raise ValueError(f"ledger key mismatch: missing right={missing}, extra right={extra}")
    splits = sorted({split for split, _ in left})
    if "dev" not in splits:
        raise ValueError("prediction ledgers must contain the dev split")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--alpha-steps", type=int, default=100)
    parser.add_argument("--minimum-dev-recall", type=float, default=0.97)
    parser.add_argument("--maximum-safe-fpr", type=float, default=0.02)
    parser.add_argument(
        "--dev-only",
        action="store_true",
        help="Compare only identical dev IDs when one input also contains other splits.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()

    left = read_prediction_ledger(args.left)
    right = read_prediction_ledger(args.right)
    splits = comparison_splits(left, right, dev_only=args.dev_only)
    joined_by_split = {split: join_split(left, right, split) for split in splits}
    selected, candidates, dev_probabilities = fit_blend(
        joined_by_split["dev"],
        alpha_steps=args.alpha_steps,
        min_recall=args.minimum_dev_recall,
        max_fpr=args.maximum_safe_fpr,
    )

    method = str(selected["method"])
    right_weight = float(selected["right_weight"])
    probabilities_by_split = {}
    for split, joined in joined_by_split.items():
        probabilities_by_split[split] = (
            dev_probabilities
            if split == "dev"
            else blend_probabilities(
                probability_matrix(joined, 0),
                probability_matrix(joined, 1),
                right_weight=right_weight,
                method=method,
            )
        )

    scam_threshold = float(selected["scam_threshold"])
    safe_threshold = float(selected["safe_threshold"])
    split_metrics = {
        split: evaluate_split(
            joined,
            probabilities_by_split[split],
            scam_threshold=scam_threshold,
            safe_threshold=safe_threshold,
        )
        for split, joined in joined_by_split.items()
    }
    prediction_path = args.predictions or args.report.with_suffix(".predictions.jsonl")
    records = prediction_records(
        joined_by_split,
        probabilities_by_split,
        scam_threshold=scam_threshold,
        safe_threshold=safe_threshold,
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    report = {
        "schema_version": 1,
        "experiment_type": "post_hoc_dev_only_score_interpolation_diagnostic",
        "inputs": {
            "left": {
                "name": args.left_name,
                "path": str(args.left),
                "sha256": file_sha256(args.left),
                "examples": len(left),
            },
            "right": {
                "name": args.right_name,
                "path": str(args.right),
                "sha256": file_sha256(args.right),
                "examples": len(right),
            },
            "contains_message_text": False,
        },
        "selection_policy": {
            "fit_split": "dev",
            "compared_splits": splits,
            "dev_only": args.dev_only,
            "selection_used_non_dev_labels": False,
            "methods": list(METHODS),
            "alpha_steps": args.alpha_steps,
            "candidate_count": len(candidates),
            "minimum_dev_recall": args.minimum_dev_recall,
            "maximum_safe_fpr": args.maximum_safe_fpr,
            "ranking": (
                "joint contract first; if feasible minimize FPR then maximize recall; "
                "otherwise maximize recall under FPR cap; exact metric ties prefer an endpoint"
            ),
            "safe_threshold": (
                "fit only after blend selection by dev three-way macro-F1, then accuracy, "
                "then higher abstention threshold"
            ),
        },
        "selected": selected,
        "best_feasible_by_macro_f1": max(
            (
                candidate
                for candidate in candidates
                if candidate["joint_dev_contract_satisfied"]
            ),
            key=lambda candidate: (
                candidate["dev_macro_f1"],
                -candidate["false_positive_rate"],
                candidate["scam_recall"],
            ),
            default=None,
        ),
        "dev_candidates": candidates,
        "split_metrics": split_metrics,
        "prediction_ledger": {
            "path": str(prediction_path),
            "sha256": file_sha256(prediction_path),
            "examples": len(records),
            "contains_message_text": False,
        },
        "research_limitations": {
            "post_hoc_hypothesis_after_prior_split_inspection": True,
            "previously_inspected_splits_are_diagnostic_not_fresh_confirmation": True,
            "sealed_primary_test_v8_opened": False,
            "two_adapter_runtime_measured": False,
            "two_adapter_ensemble_is_mobile_release_candidate": False,
            "intended_use": "infer the next single-adapter data or weight-space experiment",
        },
        "release_gates": {
            "quantization_allowed": False,
            "hugging_face_publication_allowed": False,
            "reason": "diagnostic ensemble is not a validated single mobile artifact",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
