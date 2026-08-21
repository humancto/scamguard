#!/usr/bin/env python3
"""Train the cheap lexical baseline and calibrate its safety operating point."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from scamguard.linear_baseline import build_pipeline
from scamguard.metrics import binary_safety_metrics, choose_threshold, file_sha256, wilson_interval

LABELS = ("SAFE", "UNCERTAIN", "SCAM")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def binary_subset(
    rows: list[dict[str, object]], probabilities: np.ndarray, classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    scam_index = classes.index("SCAM")
    mask = np.array([row["label"] in {"SAFE", "SCAM"} for row in rows])
    truth = np.array([int(row["label"] == "SCAM") for row in rows])[mask]
    return truth, probabilities[mask, scam_index]


def evaluate(
    pipeline: Pipeline,
    rows: list[dict[str, object]],
    threshold: float,
    *,
    include_sources: bool = True,
) -> dict[str, object]:
    texts = [str(row["text"]) for row in rows]
    truth_labels = [str(row["label"]) for row in rows]
    classes = list(pipeline.classes_)
    probabilities = pipeline.predict_proba(texts)
    predicted_labels = [classes[index] for index in probabilities.argmax(axis=1)]
    binary_truth, scam_probabilities = binary_subset(rows, probabilities, classes)
    binary = (
        binary_safety_metrics(binary_truth, scam_probabilities, threshold)
        if len(binary_truth)
        else None
    )
    result: dict[str, object] = {
        "examples": len(rows),
        "labels": {label: truth_labels.count(label) for label in LABELS},
        "accuracy_argmax": float(accuracy_score(truth_labels, predicted_labels)),
        "macro_f1_argmax": float(
            f1_score(
                truth_labels,
                predicted_labels,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_argmax": confusion_matrix(
            truth_labels, predicted_labels, labels=list(LABELS)
        ).tolist(),
        "binary_safety": binary,
    }
    scam_categories = sorted({str(row["category"]) for row in rows if row["label"] == "SCAM"})
    result["scam_by_category"] = {}
    for category in scam_categories:
        indices = [
            index
            for index, row in enumerate(rows)
            if row["label"] == "SCAM" and row["category"] == category
        ]
        category_probabilities = probabilities[indices, classes.index("SCAM")]
        detected = int(np.sum(category_probabilities >= threshold))
        result["scam_by_category"][category] = {
            "examples": len(indices),
            "detected": detected,
            "recall": float(detected / len(indices)),
            "recall_ci95": wilson_interval(detected, len(indices)),
            "mean_scam_probability": float(category_probabilities.mean()),
        }
    if not len(binary_truth):
        result["binary_subset_empty"] = True
    elif not any(row["label"] == "SAFE" for row in rows):
        result["positive_only"] = True
    if include_sources:
        sources = sorted({str(row["source"]) for row in rows})
        if len(sources) > 1:
            result["by_source"] = {
                source: evaluate(
                    pipeline,
                    [row for row in rows if row["source"] == source],
                    threshold,
                    include_sources=False,
                )
                for source in sources
            }
        if any(row.get("source_language") for row in rows):
            languages = sorted(
                {str(row.get("source_language") or "UNSPECIFIED") for row in rows}
            )
            result["by_language"] = {
                language: evaluate(
                    pipeline,
                    [
                        row
                        for row in rows
                        if str(row.get("source_language") or "UNSPECIFIED") == language
                    ],
                    threshold,
                    include_sources=False,
                )
                for language in languages
            }
    return result


def latency(pipeline: Pipeline, texts: list[str], runs: int = 250) -> dict[str, float]:
    samples = texts[: min(len(texts), runs)]
    pipeline.predict_proba(samples[:8])
    durations = []
    for text in samples:
        started = time.perf_counter_ns()
        pipeline.predict_proba([text])
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "samples": len(durations),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/sg-linear-v0.3.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/runs/sg-linear-v0.3.json"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--latency-p95-ms", type=float, default=20.0)
    args = parser.parse_args()

    train = read_jsonl(args.data / "train.jsonl")
    dev = read_jsonl(args.data / "dev.jsonl")
    test = read_jsonl(args.data / "test.jsonl")
    ood = read_jsonl(args.data / "ood_financial.jsonl")
    wspr_path = args.data / "ood_wspr.jsonl"
    wspr = read_jsonl(wspr_path) if wspr_path.exists() else []
    forum_validation_path = args.data / "forum_validation.jsonl"
    forum_validation = (
        read_jsonl(forum_validation_path) if forum_validation_path.exists() else []
    )
    forum_path = args.data / "ood_forum.jsonl"
    forum = read_jsonl(forum_path) if forum_path.exists() else []
    materialized_forum_path = args.data / "ood_forum_materialized.jsonl"
    materialized_forum = (
        read_jsonl(materialized_forum_path) if materialized_forum_path.exists() else []
    )
    adversarial_path = args.data / "adversarial.jsonl"
    adversarial = read_jsonl(adversarial_path) if adversarial_path.exists() else []

    pipeline = build_pipeline()
    pipeline.fit([str(row["text"]) for row in train], [str(row["label"]) for row in train])

    dev_probabilities = pipeline.predict_proba([str(row["text"]) for row in dev])
    dev_truth, dev_scam_probabilities = binary_subset(
        dev, dev_probabilities, list(pipeline.classes_)
    )
    threshold = choose_threshold(dev_truth, dev_scam_probabilities, args.max_fpr)

    results = {
        "model_id": "sg-linear-v0.3",
        "model_family": "word+character TF-IDF logistic regression",
        "threshold_fitted_on": "dev SAFE/SCAM only",
        "scam_threshold": threshold,
        "safe_threshold": 0.20,
        "targets": {
            "scam_recall_min": 0.97,
            "false_positive_rate_max": args.max_fpr,
            "macro_f1_stretch": 0.94,
            "desktop_fast_path_latency_p95_ms_max": args.latency_p95_ms,
            "mobile_latency_status": "must be measured on a physical target device",
        },
        "dev": evaluate(pipeline, dev, threshold),
        "test": evaluate(pipeline, test, threshold),
        "ood_financial": evaluate(pipeline, ood, threshold),
        "latency": latency(pipeline, [str(row["text"]) for row in test]),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "data_sha256": {
            split: file_sha256(args.data / f"{split}.jsonl")
            for split in ("train", "dev", "test", "ood_financial")
        },
    }
    if adversarial:
        results["adversarial"] = evaluate(pipeline, adversarial, threshold)
        results["data_sha256"]["adversarial"] = file_sha256(args.data / "adversarial.jsonl")
    if wspr:
        results["ood_wspr_positive_only"] = evaluate(pipeline, wspr, threshold)
        results["data_sha256"]["ood_wspr"] = file_sha256(args.data / "ood_wspr.jsonl")
    if forum_validation:
        results["forum_validation_selection_only"] = evaluate(
            pipeline, forum_validation, threshold
        )
        results["data_sha256"]["forum_validation"] = file_sha256(forum_validation_path)
    if forum:
        results["ood_forum_source_reported"] = evaluate(pipeline, forum, threshold)
        results["data_sha256"]["ood_forum"] = file_sha256(forum_path)
    if materialized_forum:
        results["ood_forum_materialized"] = evaluate(
            pipeline, materialized_forum, threshold
        )
        results["data_sha256"]["ood_forum_materialized"] = file_sha256(
            materialized_forum_path
        )
    test_binary = results["test"]["binary_safety"]
    core_categories = {
        category: values
        for category, values in results["test"]["scam_by_category"].items()
        if values["examples"] >= 20
    }
    results["test_gates"] = {
        "recall": test_binary["scam_recall"] >= 0.97,
        "fpr": test_binary["false_positive_rate"] <= args.max_fpr,
        "core_category_recall": bool(core_categories)
        and all(values["recall"] >= 0.97 for values in core_categories.values()),
        "core_category_min_examples": 20,
        "core_categories_evaluated": sorted(core_categories),
        "macro_f1_stretch": results["test"]["macro_f1_argmax"] >= 0.94,
        "desktop_fast_path_latency": results["latency"]["p95_ms"]
        <= args.latency_p95_ms,
    }

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "labels": list(pipeline.classes_),
        "model_id": results["model_id"],
        "scam_threshold": threshold,
        "safe_threshold": 0.20,
        "training_manifest": json.loads((args.data / "manifest.json").read_text()),
    }
    joblib.dump(payload, args.artifact, compress=3)
    results["artifact_bytes"] = args.artifact.stat().st_size

    prediction_path = args.predictions or args.report.with_suffix(".predictions.jsonl")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_records = []
    prediction_splits = {
        "dev": dev,
        "test": test,
        "ood_financial": ood,
        "ood_wspr": wspr,
        "forum_validation": forum_validation,
        "ood_forum": forum,
        "ood_forum_materialized": materialized_forum,
        "adversarial": adversarial,
    }
    for split, split_rows in prediction_splits.items():
        binary_rows = [row for row in split_rows if row["label"] in {"SAFE", "SCAM"}]
        if not binary_rows:
            continue
        probabilities = pipeline.predict_proba([str(row["text"]) for row in binary_rows])
        scam_index = list(pipeline.classes_).index("SCAM")
        for row, probability in zip(binary_rows, probabilities[:, scam_index], strict=True):
            prediction_records.append(
                {
                    "id": row["id"],
                    "split": split,
                    "source": row["source"],
                    "source_language": row.get("source_language"),
                    "category": row["category"],
                    "truth": row["label"],
                    "scam_probability": float(probability),
                    "threshold_scam": bool(probability >= threshold),
                }
            )
    prediction_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in prediction_records),
        encoding="utf-8",
    )
    results["prediction_ledger"] = {
        "path": str(prediction_path),
        "sha256": file_sha256(prediction_path),
        "examples": len(prediction_records),
        "contains_message_text": False,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
