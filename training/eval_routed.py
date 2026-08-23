#!/usr/bin/env python3
"""Freeze and evaluate a fast-router plus specialist ScamGuard policy."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from scamguard.metrics import file_sha256, wilson_interval

LABELS = ("SAFE", "UNCERTAIN", "SCAM")
REQUIRED_FIELDS = {
    "id",
    "split",
    "source",
    "source_language",
    "category",
    "truth",
    "argmax",
    "calibrated_verdict",
    "threshold_scam",
    "probabilities",
}
FORBIDDEN_TEXT_FIELDS = {"text", "message", "prompt", "conversation", "transcript"}


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
                raise ValueError(f"{path}:{line_number}: prediction record has an invalid schema")
            forbidden = forbidden_text_fields(record)
            if forbidden:
                raise ValueError(
                    f"{path}:{line_number}: text-bearing fields are forbidden: {sorted(forbidden)}"
                )
            identifier = record["id"]
            split = record["split"]
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{path}:{line_number}: invalid id")
            if not isinstance(split, str) or not split:
                raise ValueError(f"{path}:{line_number}: invalid split")
            if not isinstance(record["source"], str) or not record["source"]:
                raise ValueError(f"{path}:{line_number}: invalid source")
            if not isinstance(record["category"], str) or not record["category"]:
                raise ValueError(f"{path}:{line_number}: invalid category")
            if record["source_language"] is not None and not isinstance(
                record["source_language"], str
            ):
                raise ValueError(f"{path}:{line_number}: invalid source_language")
            if not isinstance(record["threshold_scam"], bool):
                raise ValueError(f"{path}:{line_number}: invalid threshold_scam")
            for field in ("truth", "argmax", "calibrated_verdict"):
                if record[field] not in LABELS:
                    raise ValueError(f"{path}:{line_number}: invalid {field}")
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
            ):
                raise ValueError(f"{path}:{line_number}: invalid probabilities")
            if not math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(f"{path}:{line_number}: probabilities do not sum to one")
            key = (split, identifier)
            if key in records:
                raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
            records[key] = record
    if not records:
        raise ValueError(f"{path}: prediction ledger is empty")
    return records


def join_split(
    router: dict[tuple[str, str], dict[str, Any]],
    specialist: dict[tuple[str, str], dict[str, Any]],
    split: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    router_keys = {key for key in router if key[0] == split}
    specialist_keys = {key for key in specialist if key[0] == split}
    if not router_keys:
        raise ValueError(f"router ledger has no {split!r} records")
    if router_keys != specialist_keys:
        missing = sorted(router_keys - specialist_keys)[:3]
        extra = sorted(specialist_keys - router_keys)[:3]
        raise ValueError(
            f"ledger key mismatch for {split!r}: missing specialist={missing}, "
            f"extra specialist={extra}"
        )
    joined = []
    for key in sorted(router_keys):
        router_record = router[key]
        specialist_record = specialist[key]
        for field in ("truth", "source", "source_language", "category"):
            if router_record.get(field) != specialist_record.get(field):
                raise ValueError(f"ledger metadata mismatch for {key!r}: {field}")
        joined.append((router_record, specialist_record))
    return joined


def confidence_margin(record: dict[str, Any]) -> float:
    probabilities = sorted(
        (float(record["probabilities"][label]) for label in LABELS), reverse=True
    )
    return probabilities[0] - probabilities[1]


def should_escalate(record: dict[str, Any], margin_max: float) -> bool:
    return record["calibrated_verdict"] == "UNCERTAIN" or confidence_margin(record) <= margin_max


def route_records(
    joined: list[tuple[dict[str, Any], dict[str, Any]]], margin_max: float
) -> list[dict[str, Any]]:
    routed = []
    for router, specialist in joined:
        escalated = should_escalate(router, margin_max)
        routed.append(
            {
                "id": router["id"],
                "split": router["split"],
                "source": router["source"],
                "source_language": router.get("source_language"),
                "category": router["category"],
                "truth": router["truth"],
                "router_verdict": router["calibrated_verdict"],
                "specialist_verdict": specialist["calibrated_verdict"],
                "final_verdict": (
                    specialist["calibrated_verdict"]
                    if escalated
                    else router["calibrated_verdict"]
                ),
                "router_confidence_margin": confidence_margin(router),
                "escalated": escalated,
            }
        )
    return routed


def component_records(
    joined: list[tuple[dict[str, Any], dict[str, Any]]], component: str
) -> list[dict[str, Any]]:
    if component not in {"router", "specialist"}:
        raise ValueError("component must be router or specialist")
    records = route_records(joined, margin_max=-1.0)
    for output, (router, specialist) in zip(records, joined, strict=True):
        output["final_verdict"] = (
            router["calibrated_verdict"]
            if component == "router"
            else specialist["calibrated_verdict"]
        )
        output["escalated"] = component == "specialist"
    return records


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.array([LABELS.index(str(record["truth"])) for record in records])
    predicted = np.array([LABELS.index(str(record["final_verdict"])) for record in records])
    binary_indices = [
        index for index, record in enumerate(records) if record["truth"] in {"SAFE", "SCAM"}
    ]
    binary_truth = np.array(
        [int(records[index]["truth"] == "SCAM") for index in binary_indices]
    )
    binary_predicted = np.array(
        [int(records[index]["final_verdict"] == "SCAM") for index in binary_indices]
    )
    tn, fp, fn, tp = confusion_matrix(
        binary_truth, binary_predicted, labels=[0, 1]
    ).ravel()
    by_category = {}
    categories = sorted(
        {str(record["category"]) for record in records if record["truth"] == "SCAM"}
    )
    for category in categories:
        selected = [
            record
            for record in records
            if record["truth"] == "SCAM" and record["category"] == category
        ]
        detected = sum(record["final_verdict"] == "SCAM" for record in selected)
        by_category[category] = {
            "examples": len(selected),
            "detected": detected,
            "recall": detected / len(selected),
            "recall_ci95": wilson_interval(detected, len(selected)),
        }
    return {
        "examples": len(records),
        "labels": dict(Counter(str(record["truth"]) for record in records)),
        "escalated": sum(bool(record["escalated"]) for record in records),
        "escalation_rate": float(np.mean([record["escalated"] for record in records])),
        "accuracy": float(np.mean(truth == predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion": confusion_matrix(
            truth, predicted, labels=list(range(len(LABELS)))
        ).tolist(),
        "binary_safety": {
            "scam_precision": float(
                precision_score(binary_truth, binary_predicted, zero_division=0)
            ),
            "scam_precision_ci95": wilson_interval(int(tp), int(tp + fp)),
            "scam_recall": float(recall_score(binary_truth, binary_predicted, zero_division=0)),
            "scam_recall_ci95": wilson_interval(int(tp), int(tp + fn)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "false_positive_rate_ci95": wilson_interval(int(fp), int(fp + tn)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "scam_by_category": by_category,
    }


def fit_policy(
    joined_dev: list[tuple[dict[str, Any], dict[str, Any]]],
    max_escalation_rate: float,
    max_fpr: float,
) -> tuple[float, dict[str, Any]]:
    nonmandatory_margins = sorted(
        {
            confidence_margin(router)
            for router, _specialist in joined_dev
            if router["calibrated_verdict"] != "UNCERTAIN"
        }
    )
    candidates = [-1.0, *nonmandatory_margins]
    evaluated = []
    for margin_max in candidates:
        metrics = evaluate_records(route_records(joined_dev, margin_max))
        evaluated.append((margin_max, metrics))
    under_cap = [
        candidate
        for candidate in evaluated
        if candidate[1]["escalation_rate"] <= max_escalation_rate
    ]
    feasible = [
        candidate
        for candidate in under_cap
        if candidate[1]["binary_safety"]["false_positive_rate"] <= max_fpr
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda item: (
                item[1]["binary_safety"]["scam_recall"],
                item[1]["macro_f1"],
                -item[1]["escalation_rate"],
                -item[0],
            ),
        )
    else:
        pool = under_cap or evaluated[:1]
        selected = min(
            pool,
            key=lambda item: (
                item[1]["binary_safety"]["false_positive_rate"],
                -item[1]["binary_safety"]["scam_recall"],
                -item[1]["macro_f1"],
                item[1]["escalation_rate"],
            ),
        )
    margin_max, metrics = selected
    return margin_max, {
        "selection_split": "dev",
        "rule": (
            "escalate when router calibrated verdict is UNCERTAIN or its top-two "
            "probability margin is at most margin_max"
        ),
        "margin_max": margin_max,
        "mandatory_uncertain_escalation": True,
        "max_escalation_rate": max_escalation_rate,
        "max_false_positive_rate": max_fpr,
        "candidates_evaluated": len(evaluated),
        "selection_feasible": bool(feasible),
        "selected_escalation_cap_compliant": (
            metrics["escalation_rate"] <= max_escalation_rate
        ),
        "selected_fpr_cap_compliant": (
            metrics["binary_safety"]["false_positive_rate"] <= max_fpr
        ),
        "selected_dev_metrics": metrics,
    }


def latency_report(
    escalation_rate: float,
    router_mean_ms: float | None,
    router_p95_ms: float | None,
    specialist_mean_ms: float | None,
    specialist_p95_ms: float | None,
) -> dict[str, Any]:
    values = (router_mean_ms, router_p95_ms, specialist_mean_ms, specialist_p95_ms)
    if all(value is None for value in values):
        return {
            "status": "not supplied",
            "routed_end_to_end_p95_ms": None,
            "routed_end_to_end_p95_reason": (
                "requires end-to-end timing of actual routed requests"
            ),
        }
    if any(value is None or value < 0.0 for value in values):
        raise ValueError("latency inputs must be four non-negative values or all omitted")
    assert router_mean_ms is not None
    assert router_p95_ms is not None
    assert specialist_mean_ms is not None
    assert specialist_p95_ms is not None
    return {
        "status": "aggregate component estimates only",
        "router_mean_ms": router_mean_ms,
        "router_p95_ms": router_p95_ms,
        "specialist_mean_ms": specialist_mean_ms,
        "specialist_p95_ms": specialist_p95_ms,
        "analytical_expected_mean_ms": router_mean_ms
        + escalation_rate * specialist_mean_ms,
        "fast_path_p95_ms": router_p95_ms,
        "specialist_path_conservative_p95_upper_bound_ms": (
            router_p95_ms + specialist_p95_ms
        ),
        "routed_end_to_end_p95_ms": None,
        "routed_end_to_end_p95_reason": (
            "aggregate component percentiles cannot produce a routed request percentile; "
            "measure the frozen policy end to end"
        ),
    }


def test_gates(metrics: dict[str, Any], max_fpr: float) -> dict[str, Any]:
    core_categories = {
        category: values
        for category, values in metrics["scam_by_category"].items()
        if values["examples"] >= 20
    }
    return {
        "recall": metrics["binary_safety"]["scam_recall"] >= 0.97,
        "fpr": metrics["binary_safety"]["false_positive_rate"] <= max_fpr,
        "core_category_recall": bool(core_categories)
        and all(values["recall"] >= 0.97 for values in core_categories.values()),
        "core_category_min_examples": 20,
        "core_categories_evaluated": sorted(core_categories),
        "macro_f1_stretch": metrics["macro_f1"] >= 0.94,
        "routed_end_to_end_latency": False,
        "routed_end_to_end_latency_reason": (
            "not evaluable from component aggregates; requires frozen-policy traces"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-predictions", type=Path, required=True)
    parser.add_argument("--specialist-predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--max-escalation-rate", type=float, default=0.25)
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--router-mean-ms", type=float)
    parser.add_argument("--router-p95-ms", type=float)
    parser.add_argument("--specialist-mean-ms", type=float)
    parser.add_argument("--specialist-p95-ms", type=float)
    args = parser.parse_args()
    if not 0.0 <= args.max_escalation_rate <= 1.0:
        parser.error("--max-escalation-rate must be in [0, 1]")
    if not 0.0 <= args.max_fpr <= 1.0:
        parser.error("--max-fpr must be in [0, 1]")

    router = read_prediction_ledger(args.router_predictions)
    specialist = read_prediction_ledger(args.specialist_predictions)
    joined_dev = join_split(router, specialist, "dev")
    joined_test = join_split(router, specialist, "test")
    margin_max, policy = fit_policy(
        joined_dev,
        max_escalation_rate=args.max_escalation_rate,
        max_fpr=args.max_fpr,
    )
    routed_test = route_records(joined_test, margin_max)
    test_metrics = evaluate_records(routed_test)
    router_test_metrics = evaluate_records(component_records(joined_test, "router"))
    specialist_test_metrics = evaluate_records(component_records(joined_test, "specialist"))
    output_path = args.predictions or args.report.with_suffix(".predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in routed_test),
        encoding="utf-8",
    )
    result = {
        "architecture": "fast calibrated encoder router plus selective Qwen specialist",
        "inputs": {
            "router_predictions": str(args.router_predictions),
            "router_sha256": file_sha256(args.router_predictions),
            "specialist_predictions": str(args.specialist_predictions),
            "specialist_sha256": file_sha256(args.specialist_predictions),
        },
        "join": {
            "key": ["split", "id"],
            "metadata_checked": ["truth", "source", "source_language", "category"],
            "dev_examples": len(joined_dev),
            "test_examples": len(joined_test),
            "contains_message_text": False,
        },
        "policy": policy,
        "baselines": {
            "test_router_only": router_test_metrics,
            "test_specialist_only": specialist_test_metrics,
        },
        "test": test_metrics,
        "delta_vs_test_router": {
            "scam_recall": test_metrics["binary_safety"]["scam_recall"]
            - router_test_metrics["binary_safety"]["scam_recall"],
            "false_positive_rate": test_metrics["binary_safety"]["false_positive_rate"]
            - router_test_metrics["binary_safety"]["false_positive_rate"],
            "macro_f1": test_metrics["macro_f1"] - router_test_metrics["macro_f1"],
        },
        "test_gates": test_gates(test_metrics, args.max_fpr),
        "latency": latency_report(
            test_metrics["escalation_rate"],
            args.router_mean_ms,
            args.router_p95_ms,
            args.specialist_mean_ms,
            args.specialist_p95_ms,
        ),
        "prediction_ledger": {
            "path": str(output_path),
            "sha256": file_sha256(output_path),
            "examples": len(routed_test),
            "contains_message_text": False,
        },
        "limitations": [
            "Routing is selected on dev only; test is never used to choose margin_max.",
            "Aggregate component latency cannot establish routed end-to-end p95.",
            "Physical-device latency and final quantized-artifact parity remain separate gates.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
