#!/usr/bin/env python3
"""Paired ScamBench comparison for a Qwen candidate and reference detector."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_mcnemar_pvalue(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def rate(decisions: np.ndarray, truth: np.ndarray, positive_truth: bool) -> float:
    mask = truth if positive_truth else ~truth
    if not mask.any():
        return float("nan")
    return float(decisions[mask].mean())


def bootstrap_interval(
    first: np.ndarray,
    second: np.ndarray,
    truth: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    iterations: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        indices = rng.integers(0, len(truth), len(truth))
        differences.append(
            metric(first[indices], truth[indices]) - metric(second[indices], truth[indices])
        )
    return [float(value) for value in np.percentile(differences, [2.5, 97.5])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, default=Path("reports/runs/paired-comparison.json"))
    args = parser.parse_args()

    candidate = {
        str(row["id"]): row
        for row in read_jsonl(args.candidate)
        if row["split"] == args.split and row["truth"] in {"SAFE", "SCAM"}
    }
    reference = {
        str(row["id"]): row
        for row in read_jsonl(args.reference)
        if row["split"] == args.split and row["truth"] in {"SAFE", "SCAM"}
    }
    if set(candidate) != set(reference):
        raise ValueError("candidate and reference do not contain identical binary example IDs")
    ids = sorted(candidate)
    if not ids:
        raise ValueError("no paired examples")
    for identifier in ids:
        if candidate[identifier]["truth"] != reference[identifier]["truth"]:
            raise ValueError(f"truth mismatch for {identifier}")

    truth = np.array([candidate[identifier]["truth"] == "SCAM" for identifier in ids])
    candidate_decisions = np.array(
        [bool(candidate[identifier]["threshold_scam"]) for identifier in ids]
    )
    reference_decisions = np.array(
        [bool(reference[identifier]["scambench_threshold_scam"]) for identifier in ids]
    )
    candidate_correct = candidate_decisions == truth
    reference_correct = reference_decisions == truth
    candidate_only = int(np.sum(candidate_correct & ~reference_correct))
    reference_only = int(np.sum(~candidate_correct & reference_correct))

    def recall_metric(decisions: np.ndarray, labels: np.ndarray) -> float:
        return rate(decisions, labels, True)

    def fpr_metric(decisions: np.ndarray, labels: np.ndarray) -> float:
        return rate(decisions, labels, False)

    def accuracy_metric(decisions: np.ndarray, labels: np.ndarray) -> float:
        return float(np.mean(decisions == labels))

    result = {
        "split": args.split,
        "examples": len(ids),
        "scam_examples": int(truth.sum()),
        "safe_examples": int((~truth).sum()),
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "candidate_metrics": {
            "recall": recall_metric(candidate_decisions, truth),
            "false_positive_rate": fpr_metric(candidate_decisions, truth),
            "accuracy": accuracy_metric(candidate_decisions, truth),
        },
        "reference_metrics": {
            "recall": recall_metric(reference_decisions, truth),
            "false_positive_rate": fpr_metric(reference_decisions, truth),
            "accuracy": accuracy_metric(reference_decisions, truth),
        },
        "candidate_minus_reference": {
            "recall": recall_metric(candidate_decisions, truth)
            - recall_metric(reference_decisions, truth),
            "recall_ci95_paired_bootstrap": bootstrap_interval(
                candidate_decisions,
                reference_decisions,
                truth,
                recall_metric,
                iterations=args.iterations,
                seed=args.seed,
            ),
            "false_positive_rate": fpr_metric(candidate_decisions, truth)
            - fpr_metric(reference_decisions, truth),
            "false_positive_rate_ci95_paired_bootstrap": bootstrap_interval(
                candidate_decisions,
                reference_decisions,
                truth,
                fpr_metric,
                iterations=args.iterations,
                seed=args.seed + 1,
            ),
            "accuracy": accuracy_metric(candidate_decisions, truth)
            - accuracy_metric(reference_decisions, truth),
            "accuracy_ci95_paired_bootstrap": bootstrap_interval(
                candidate_decisions,
                reference_decisions,
                truth,
                accuracy_metric,
                iterations=args.iterations,
                seed=args.seed + 2,
            ),
        },
        "mcnemar_exact": {
            "candidate_only_correct": candidate_only,
            "reference_only_correct": reference_only,
            "two_sided_p_value": exact_mcnemar_pvalue(candidate_only, reference_only),
        },
        "bootstrap_iterations": args.iterations,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
