#!/usr/bin/env python3
"""Audit a frozen dev decision policy from its bound, existing prediction ledger.

This diagnostic never reads datasets, changes thresholds, scores a model, or
authorizes candidate promotion. Non-dev ledger records are counted but ignored.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.decision import calibrated_verdict
from scamguard.metrics import file_sha256, verdict_alert_metrics

LABELS = ("SAFE", "UNCERTAIN", "SCAM")


def finite_number(value: Any, name: str, *, probability: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (probability and not 0.0 <= number <= 1.0):
        raise ValueError(f"{name} must be finite and within its permitted range")
    return number


def require_close(actual: Any, expected: float, name: str) -> None:
    number = finite_number(actual, name)
    if not math.isclose(number, expected, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f"{name} differs from the bound development ledger")


def require_count(actual: Any, expected: int, name: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
        raise ValueError(f"{name} differs from the bound development ledger")


def confusion(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    active = denominator > 0
    per_label = np.divide(
        2.0 * np.diag(matrix),
        denominator,
        out=np.zeros(3, dtype=float),
        where=active,
    )
    return float(per_label[active].mean()) if active.any() else 0.0


def fixed_scam_oracle_ceiling(matrix: np.ndarray) -> dict[str, Any]:
    """Upper-bound three-way macro F1 when the SCAM prediction set is immutable.

    Every movable true SAFE/UNCERTAIN row is optimally assigned its reference
    label. Only missed true SCAM rows remain: all allocations of those rows to
    SAFE versus UNCERTAIN are enumerated. No row can become or cease being SCAM.
    The oracle uses labels and is not a deployable policy or a measured score.
    """
    matrix = np.asarray(matrix)
    if (
        matrix.shape != (3, 3)
        or not np.issubdtype(matrix.dtype, np.integer)
        or (matrix < 0).any()
        or matrix.sum() == 0
    ):
        raise ValueError("oracle requires a non-empty nonnegative integer 3x3 confusion")
    totals = matrix.sum(axis=1)
    fixed = matrix[:, 2]
    missed_scams = int(totals[2] - fixed[2])
    best: tuple[float, int] | None = None
    best_matrix: np.ndarray | None = None
    for assigned_safe in range(missed_scams + 1):
        candidate = np.array(
            [
                [totals[0] - fixed[0], 0, fixed[0]],
                [0, totals[1] - fixed[1], fixed[1]],
                [assigned_safe, missed_scams - assigned_safe, fixed[2]],
            ],
            dtype=np.int64,
        )
        rank = (macro_f1(candidate), assigned_safe)
        if best is None or rank > best:
            best, best_matrix = rank, candidate
    assert best is not None and best_matrix is not None
    return {
        "macro_f1_upper_bound": best[0],
        "oracle_confusion": best_matrix.tolist(),
        "missed_scams_assigned_safe": best[1],
        "missed_scams_assigned_uncertain": missed_scams - best[1],
        "allocations_evaluated": missed_scams + 1,
        "definition": (
            "Diagnostic label-aware upper bound at the exact frozen SCAM set: "
            "correct all remaining SAFE/UNCERTAIN rows and exhaustively allocate "
            "missed SCAM rows between SAFE and UNCERTAIN. This is neither a "
            "deployable policy nor an achieved score."
        ),
    }


def audit(report_path: Path, ledger_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("development_screen_only") is not True
        or report.get("release_gate_report") is not False
        or ("selection_screen_only" in report and report["selection_screen_only"] is not False)
        or report.get("frozen_calibration_source") is not None
        or report.get("safe_threshold_semantics") != "minimum_safe_probability"
    ):
        raise ValueError("an existing development-only calibration report is required")
    binding = report.get("prediction_ledger", {})
    ledger_hash = file_sha256(ledger_path)
    if (
        binding.get("path") != str(ledger_path)
        or binding.get("sha256") != ledger_hash
        or binding.get("contains_message_text") is not False
    ):
        raise ValueError("report is not bound to the supplied text-free prediction ledger")
    scam_threshold = finite_number(report.get("scam_threshold"), "scam_threshold", probability=True)
    safe_threshold = finite_number(report.get("safe_threshold"), "safe_threshold", probability=True)
    temperature = finite_number(report.get("temperature"), "temperature")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    truth_values: list[int] = []
    predictions: list[int] = []
    argmax_values: list[int] = []
    probabilities: list[list[float]] = []
    sources: list[str] = []
    seen: set[str] = set()
    total_records = 0
    with ledger_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("split"), str):
                raise ValueError("ledger record must have an explicit split")
            total_records += 1
            if row["split"] != "dev":
                continue
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                raise ValueError("development IDs must be non-empty and unique")
            seen.add(identifier)
            if row.get("truth") not in LABELS:
                raise ValueError("development row has an unsupported reference label")
            source = row.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("development row requires a source")
            values = row.get("probabilities")
            if not isinstance(values, dict) or set(values) != set(LABELS):
                raise ValueError("probabilities require exactly SAFE, UNCERTAIN, SCAM")
            probability = [
                finite_number(values[label], f"p_{label}", probability=True) for label in LABELS
            ]
            if not math.isclose(sum(probability), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("probabilities must sum to one")
            predicted = calibrated_verdict(
                safe_probability=probability[0],
                scam_probability=probability[2],
                scam_probability_threshold=scam_threshold,
                safe_probability_threshold=safe_threshold,
                safe_max_scam_probability=None,
            )
            argmax = int(np.argmax(probability))
            if (
                row.get("calibrated_verdict") != predicted
                or row.get("argmax") != LABELS[argmax]
                or row.get("threshold_scam") is not (predicted == "SCAM")
            ):
                raise ValueError("ledger verdict differs from the frozen shared decision")
            truth_index = LABELS.index(row["truth"])
            require_close(
                row.get("negative_log_likelihood"),
                -math.log(max(probability[truth_index], 1e-9)),
                "ledger negative_log_likelihood",
            )
            truth_values.append(truth_index)
            predictions.append(LABELS.index(predicted))
            argmax_values.append(argmax)
            probabilities.append(probability)
            sources.append(source)
    require_count(binding.get("examples"), total_records, "prediction_ledger.examples")
    if not truth_values:
        raise ValueError("ledger contains no development rows")
    truth = np.array(truth_values)
    predicted = np.array(predictions)
    argmax = np.array(argmax_values)
    probability_array = np.array(probabilities)
    matrix = confusion(truth, predicted)
    argmax_matrix = confusion(truth, argmax)
    dev = report.get("dev", {})
    require_count(dev.get("examples"), len(truth), "dev.examples")
    counts = Counter(LABELS[index] for index in truth_values)
    if dev.get("labels") != dict(counts):
        raise ValueError("dev.labels differs from the bound development ledger")
    decision = dev.get("calibrated_decision", {})
    for name, actual, expected in (
        ("dev.confusion_argmax", dev.get("confusion_argmax"), argmax_matrix),
        ("dev.calibrated_decision.confusion", decision.get("confusion"), matrix),
    ):
        if actual != expected.tolist():
            raise ValueError(f"{name} differs from the bound development ledger")
    for name, actual, expected in (
        ("dev.accuracy_argmax", dev.get("accuracy_argmax"), float((truth == argmax).mean())),
        ("dev.macro_f1_argmax", dev.get("macro_f1_argmax"), macro_f1(argmax_matrix)),
        ("dev.decision.accuracy", decision.get("accuracy"), float((truth == predicted).mean())),
        ("dev.decision.macro_f1", decision.get("macro_f1"), macro_f1(matrix)),
        ("dev.decision.safe_threshold", decision.get("safe_threshold"), safe_threshold),
        ("dev.calibration.temperature", dev.get("calibration", {}).get("temperature"), temperature),
    ):
        require_close(actual, expected, name)
    binary = dev.get("binary_safety", {})
    tn, fp, fn, tp = (
        int(matrix[0, :2].sum()),
        int(matrix[0, 2]),
        int(matrix[2, :2].sum()),
        int(matrix[2, 2]),
    )
    for name, expected in {"tn": tn, "fp": fp, "fn": fn, "tp": tp}.items():
        require_count(binary.get(name), expected, f"dev.binary_safety.{name}")
    for name, expected in {
        "threshold": scam_threshold,
        "scam_precision": tp / (tp + fp) if tp + fp else 0.0,
        "scam_recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "scam_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }.items():
        require_close(binary.get(name), expected, f"dev.binary_safety.{name}")
    selected_probability = probability_array[np.arange(len(truth)), truth]
    require_close(
        dev.get("calibration", {}).get("after_temperature", {}).get("negative_log_likelihood"),
        float(-np.log(np.clip(selected_probability, 1e-9, 1.0)).mean()),
        "dev.calibration.after_temperature.negative_log_likelihood",
    )

    uncertain = truth == 1
    by_source: dict[str, Any] = {}
    for source in sorted(set(sources)):
        mask = uncertain & (np.array(sources) == source)
        if mask.any():
            by_source[source] = {
                "reference_uncertain": int(mask.sum()),
                "argmax_to_decision_confusion": confusion(argmax[mask], predicted[mask]).tolist(),
            }
    return {
        "artifact_schema_version": 1,
        "diagnostic_only": True,
        "eligibility_authorization": False,
        "calibration_changed": False,
        "contains_message_text": False,
        "inputs": {
            "report": {"path": str(report_path), "sha256": file_sha256(report_path)},
            "prediction_ledger": {"path": str(ledger_path), "sha256": ledger_hash},
        },
        "scope": {
            "processed_split": "dev",
            "dev_examples": len(truth),
            "non_dev_records_ignored": total_records - len(truth),
            "legacy_report_without_selection_screen_only": "selection_screen_only" not in report,
            "new_inference": False,
            "threshold_search": False,
        },
        "calibration": {
            "temperature": temperature,
            "scam_threshold": scam_threshold,
            "safe_threshold": safe_threshold,
        },
        "label_order": list(LABELS),
        "labels": dict(counts),
        "argmax": {"confusion": argmax_matrix.tolist(), "macro_f1": macro_f1(argmax_matrix)},
        "frozen_decision": {"confusion": matrix.tolist(), "macro_f1": macro_f1(matrix)},
        "alerts": verdict_alert_metrics(truth, predicted),
        "reference_uncertain": {
            "examples": int(uncertain.sum()),
            "argmax_to_decision_confusion": confusion(
                argmax[uncertain], predicted[uncertain]
            ).tolist(),
            "by_source": by_source,
        },
        "fixed_scam_oracle": fixed_scam_oracle_ceiling(matrix),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replace", action="store_true", help="Replace an existing diagnostic only."
    )
    args = parser.parse_args()
    if args.output.resolve() in {args.report.resolve(), args.predictions.resolve()}:
        raise ValueError("diagnostic output cannot overwrite an input")
    if args.output.exists() and not args.replace:
        raise ValueError("diagnostic output already exists; use --replace explicitly")
    result = audit(args.report, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
