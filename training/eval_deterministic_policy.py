#!/usr/bin/env python3
"""Evaluate a frozen deterministic policy over a text-free model prediction ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from scamguard.metrics import file_sha256, wilson_interval
from scamguard.policy import POLICY_VERSION, deterministic_override
from scamguard.signals import extract_signal_matches
from scamguard.taxonomy import Verdict

LABELS = ("SAFE", "UNCERTAIN", "SCAM")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def binary_metrics(truth: list[str], predicted: list[bool]) -> dict[str, Any]:
    selected = [index for index, label in enumerate(truth) if label in {"SAFE", "SCAM"}]
    y_true = np.array([truth[index] == "SCAM" for index in selected], dtype=int)
    y_pred = np.array([predicted[index] for index in selected], dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "scam_recall": float(tp / max(tp + fn, 1)),
        "scam_recall_ci95": wilson_interval(int(tp), int(tp + fn)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_positive_rate_ci95": wilson_interval(int(fp), int(fp + tn)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    predictions = read_jsonl(args.predictions)
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows) or len(predictions) != len(rows):
        raise ValueError("data and prediction ledgers must contain the same unique row count")
    if {str(item["id"]) for item in predictions} != set(by_id):
        raise ValueError("prediction IDs differ from dataset IDs")

    truth: list[str] = []
    base_binary: list[bool] = []
    policy_binary: list[bool] = []
    base_labels: list[str] = []
    policy_labels: list[str] = []
    rule_counts: Counter[str] = Counter()
    rule_truth: Counter[str] = Counter()
    ledger: list[dict[str, Any]] = []

    for item in predictions:
        row = by_id[str(item["id"])]
        signals = tuple(match.signal for match in extract_signal_matches(str(row["text"])))
        override = deterministic_override(str(row["text"]), signals)
        base_is_scam = bool(item["scam_at_frozen_threshold"])
        final_is_scam = base_is_scam
        final_label = str(item["argmax_label"])
        if override:
            final_is_scam = override.verdict is Verdict.SCAM
            final_label = override.verdict.value
            rule_counts[override.rule_id] += 1
            rule_truth[f"{override.rule_id}:{row['label']}"] += 1

        truth.append(str(row["label"]))
        base_binary.append(base_is_scam)
        policy_binary.append(final_is_scam)
        base_labels.append(str(item["argmax_label"]))
        policy_labels.append(final_label)
        ledger.append(
            {
                "id": row["id"],
                "family_id": row.get("family_id"),
                "label": row["label"],
                "base_scam": base_is_scam,
                "policy_scam": final_is_scam,
                "base_argmax_label": item["argmax_label"],
                "policy_label": final_label,
                "override_rule": override.rule_id if override else None,
            }
        )

    report = {
        "split": args.split,
        "policy_version": POLICY_VERSION,
        "data_sha256": file_sha256(args.data),
        "predictions_sha256": file_sha256(args.predictions),
        "examples": len(rows),
        "labels": dict(Counter(truth)),
        "rule_counts": dict(rule_counts),
        "rule_truth": dict(rule_truth),
        "base_binary": binary_metrics(truth, base_binary),
        "policy_binary": binary_metrics(truth, policy_binary),
        "base_macro_f1_argmax": float(
            f1_score(truth, base_labels, labels=LABELS, average="macro", zero_division=0)
        ),
        "policy_macro_f1": float(
            f1_score(truth, policy_labels, labels=LABELS, average="macro", zero_division=0)
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.ledger:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        args.ledger.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in ledger),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
