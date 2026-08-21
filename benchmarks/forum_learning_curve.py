#!/usr/bin/env python3
"""Select a real-forum training cap without reading test or OOD outcomes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from scamguard.linear_baseline import build_pipeline
from scamguard.metrics import binary_safety_metrics, choose_threshold, file_sha256

LABELS = ("SAFE", "UNCERTAIN", "SCAM")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def binary_subset(
    rows: list[dict[str, Any]], probabilities: np.ndarray, classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    scam_index = classes.index("SCAM")
    mask = np.array([row["label"] in {"SAFE", "SCAM"} for row in rows])
    truth = np.array([int(row["label"] == "SCAM") for row in rows])[mask]
    return truth, probabilities[mask, scam_index]


def evaluate_selection(
    pipeline: Any, rows: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    labels = [str(row["label"]) for row in rows]
    probabilities = pipeline.predict_proba([str(row["text"]) for row in rows])
    classes = list(pipeline.classes_)
    predictions = [classes[index] for index in probabilities.argmax(axis=1)]
    binary_truth, scam_probabilities = binary_subset(rows, probabilities, classes)
    return {
        "examples": len(rows),
        "macro_f1_argmax": float(
            f1_score(
                labels,
                predictions,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "binary_safety": binary_safety_metrics(binary_truth, scam_probabilities, threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/learning_curve"))
    parser.add_argument("--caps", default="0,1000,3000,5672")
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--plateau-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/runs/forum-learning-curve-v0.3.json")
    )
    args = parser.parse_args()
    caps = [int(value) for value in args.caps.split(",")]
    if sorted(set(caps)) != caps or any(cap < 0 for cap in caps):
        raise ValueError("--caps must be unique, ascending, non-negative integers")

    records: list[dict[str, Any]] = []
    dev_hash: str | None = None
    validation_hash: str | None = None
    for cap in caps:
        data = args.root / f"forum-{cap}"
        manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        if manifest["forum_train_scam"] != cap:
            raise ValueError(f"{data} contains cap {manifest['forum_train_scam']}, expected {cap}")
        current_dev_hash = file_sha256(data / "dev.jsonl")
        current_validation_hash = file_sha256(data / "forum_validation.jsonl")
        dev_hash = dev_hash or current_dev_hash
        validation_hash = validation_hash or current_validation_hash
        if current_dev_hash != dev_hash or current_validation_hash != validation_hash:
            raise ValueError("development slices differ across cap candidates")

        # Deliberately do not open test.jsonl or any ood_*.jsonl here.
        train = read_jsonl(data / "train.jsonl")
        dev = read_jsonl(data / "dev.jsonl")
        forum_validation = read_jsonl(data / "forum_validation.jsonl")
        pipeline = build_pipeline()
        started = time.perf_counter()
        pipeline.fit([str(row["text"]) for row in train], [str(row["label"]) for row in train])
        fit_seconds = time.perf_counter() - started
        dev_probabilities = pipeline.predict_proba([str(row["text"]) for row in dev])
        dev_truth, dev_scam_probabilities = binary_subset(
            dev, dev_probabilities, list(pipeline.classes_)
        )
        threshold = choose_threshold(dev_truth, dev_scam_probabilities, args.max_fpr)
        dev_result = evaluate_selection(pipeline, dev, threshold)
        validation_result = evaluate_selection(pipeline, forum_validation, threshold)
        records.append(
            {
                "forum_train_scam": cap,
                "forum_train_uncertain": manifest["forum_train_uncertain"],
                "train_examples": len(train),
                "fit_seconds": fit_seconds,
                "threshold": threshold,
                "dev": dev_result,
                "forum_validation": validation_result,
            }
        )

    eligible = [
        record
        for record in records
        if record["dev"]["binary_safety"]["scam_recall"] >= 0.97
        and record["dev"]["binary_safety"]["false_positive_rate"] <= args.max_fpr
    ]
    release_gate_selected: int | None = None
    if eligible:
        best_forum_recall = max(
            record["forum_validation"]["binary_safety"]["scam_recall"]
            for record in eligible
        )
        plateau = [
            record
            for record in eligible
            if record["forum_validation"]["binary_safety"]["scam_recall"]
            >= best_forum_recall - args.plateau_tolerance
        ]
        release_gate_selected = min(record["forum_train_scam"] for record in plateau)

    # The cheap lexical model is a screening proxy, not the release candidate.
    # When no cap clears the absolute release gate, retain an explicitly
    # exploratory quality-first recommendation: stay within the observed forum
    # recall plateau, then maximize core-development recall at the same FPR cap.
    fpr_eligible = [
        record
        for record in records
        if record["dev"]["binary_safety"]["false_positive_rate"] <= args.max_fpr
    ]
    proxy_recommendation: int | None = None
    if fpr_eligible:
        best_forum_recall = max(
            record["forum_validation"]["binary_safety"]["scam_recall"]
            for record in fpr_eligible
        )
        forum_plateau = [
            record
            for record in fpr_eligible
            if record["forum_validation"]["binary_safety"]["scam_recall"]
            >= best_forum_recall - args.plateau_tolerance
        ]
        proxy_recommendation = max(
            forum_plateau,
            key=lambda record: (
                record["dev"]["binary_safety"]["scam_recall"],
                -record["forum_train_scam"],
            ),
        )["forum_train_scam"]

    result = {
        "protocol": "development-only forum training-size selection",
        "files_read_per_candidate": ["train.jsonl", "dev.jsonl", "forum_validation.jsonl"],
        "files_prohibited": [
            "test.jsonl",
            "ood_financial.jsonl",
            "ood_wspr.jsonl",
            "ood_forum.jsonl",
        ],
        "selection_rule": (
            "smallest cap within plateau_tolerance of best forum-validation SCAM recall, "
            "among caps meeting core-dev recall>=0.97 and FPR<=max_fpr"
        ),
        "max_fpr": args.max_fpr,
        "plateau_tolerance": args.plateau_tolerance,
        "dev_sha256": dev_hash,
        "forum_validation_sha256": validation_hash,
        "release_gate_selected_forum_train_scam": release_gate_selected,
        "quality_first_proxy_rule": (
            "exploratory only: among FPR-compliant caps within the forum-recall plateau, "
            "maximize core-development recall; confirm with the final model family"
        ),
        "quality_first_proxy_recommendation": proxy_recommendation,
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
