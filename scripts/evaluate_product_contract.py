#!/usr/bin/env python3
"""Audit grounded product metadata from a frozen text-free prediction ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256
from scamguard.model import ModelScores
from scamguard.scanner import Scanner
from scamguard.taxonomy import Category, RecommendedAction, Verdict


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_split_path(
    data: Path, external_data: Path, split: str, primary_test_v8: Path | None
) -> Path:
    if split == "primary_test_v8":
        if primary_test_v8 is None:
            raise ValueError("primary_test_v8 requires --primary-test-v8")
        return primary_test_v8
    candidates = [data / f"{split}.jsonl"]
    external_paths = {
        "ood_chichewa": external_data / "chichewa" / "ood_chichewa.jsonl",
        "scam_dialogue_validation": (
            external_data / "scam_dialogue" / "scam_dialogue_validation.jsonl"
        ),
        "taskmaster_validation": (
            external_data / "taskmaster" / "taskmaster_validation.jsonl"
        ),
    }
    if split in external_paths:
        candidates.append(external_paths[split])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no product-contract data found for split {split}")


@dataclass(slots=True)
class LedgerVerdictBackend:
    """Force Scanner through the exact verdict already frozen in the ledger."""

    verdict: Verdict
    model_id: str = "frozen-ledger-product-contract-audit"
    scam_threshold: float = 0.8
    safe_threshold: float = 0.2
    safe_probability_threshold: float = 0.8
    safe_max_scam_probability: float | None = None

    def predict(self, _text: str) -> ModelScores:
        return {
            Verdict.SAFE: ModelScores(safe=1.0, uncertain=0.0, scam=0.0),
            Verdict.UNCERTAIN: ModelScores(safe=0.0, uncertain=1.0, scam=0.0),
            Verdict.SCAM: ModelScores(safe=0.0, uncertain=0.0, scam=1.0),
        }[self.verdict]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    truth = Counter(str(output["truth"]) for output in outputs)
    verdicts = Counter(str(output["verdict"]) for output in outputs)
    emitted_scams = [output for output in outputs if output["verdict"] == "SCAM"]
    true_positive_scams = [
        output
        for output in emitted_scams
        if output["truth"] == "SCAM"
    ]
    risk_decisions = [output for output in outputs if output["verdict"] != "SAFE"]

    def with_evidence(rows: list[dict[str, Any]]) -> int:
        return sum(bool(row["has_grounded_evidence"]) for row in rows)

    def with_known_category(rows: list[dict[str, Any]]) -> int:
        return sum(
            row["product_category"] not in {Category.NONE.value, Category.UNKNOWN.value}
            for row in rows
        )

    def with_specific_action(rows: list[dict[str, Any]]) -> int:
        return sum(
            row["recommended_action"]
            not in {
                RecommendedAction.NO_ACTION.value,
                RecommendedAction.VERIFY_OFFICIAL_CHANNEL.value,
            }
            for row in rows
        )

    emitted_with_evidence = with_evidence(emitted_scams)
    true_positive_with_evidence = with_evidence(true_positive_scams)
    true_scam_explained = sum(
        output["truth"] == "SCAM"
        and output["verdict"] == "SCAM"
        and bool(output["has_grounded_evidence"])
        for output in outputs
    )
    return {
        "examples": len(outputs),
        "truth": dict(sorted(truth.items())),
        "calibrated_verdicts": dict(sorted(verdicts.items())),
        "risk_decisions": len(risk_decisions),
        "risk_decisions_with_grounded_evidence": with_evidence(risk_decisions),
        "risk_decision_grounded_evidence_rate": _rate(
            with_evidence(risk_decisions), len(risk_decisions)
        ),
        "emitted_scams": len(emitted_scams),
        "emitted_scams_with_grounded_evidence": emitted_with_evidence,
        "emitted_scam_grounded_evidence_rate": _rate(
            emitted_with_evidence, len(emitted_scams)
        ),
        "emitted_scams_with_known_category": with_known_category(emitted_scams),
        "emitted_scam_known_category_rate": _rate(
            with_known_category(emitted_scams), len(emitted_scams)
        ),
        "emitted_scams_with_specific_action": with_specific_action(emitted_scams),
        "emitted_scam_specific_action_rate": _rate(
            with_specific_action(emitted_scams), len(emitted_scams)
        ),
        "true_positive_scams": len(true_positive_scams),
        "true_positive_scams_with_grounded_evidence": true_positive_with_evidence,
        "true_positive_scam_grounded_evidence_rate": _rate(
            true_positive_with_evidence, len(true_positive_scams)
        ),
        "true_scam_grounded_explanation_recall": _rate(
            true_scam_explained, truth["SCAM"]
        ),
        "product_categories": dict(
            sorted(Counter(str(output["product_category"]) for output in outputs).items())
        ),
        "recommended_actions": dict(
            sorted(Counter(str(output["recommended_action"]) for output in outputs).items())
        ),
        "signals": dict(
            sorted(
                Counter(
                    signal
                    for output in outputs
                    for signal in output["signals"]
                ).items()
            )
        ),
    }


def evaluate_product_contract(
    rows_by_split: dict[str, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    data_index: dict[tuple[str, str], dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        for row_index, row in enumerate(rows, start=1):
            key = (split, str(row.get("id", "")))
            if not key[1]:
                raise ValueError(f"{split} data row {row_index} lacks an ID")
            if key in data_index:
                raise ValueError(f"duplicate data key: {key}")
            data_index[key] = row

    prediction_index: dict[tuple[str, str], dict[str, Any]] = {}
    required = {"id", "split", "source", "category", "truth", "calibrated_verdict"}
    for row_index, prediction in enumerate(predictions, start=1):
        missing = sorted(required - prediction.keys())
        if missing:
            raise ValueError(f"prediction row {row_index} missing fields: {missing}")
        if "text" in prediction:
            raise ValueError(f"prediction row {row_index} contains message text")
        key = (str(prediction["split"]), str(prediction["id"]))
        if key in prediction_index:
            raise ValueError(f"duplicate prediction key: {key}")
        prediction_index[key] = prediction

    if set(data_index) != set(prediction_index):
        missing_predictions = sorted(set(data_index) - set(prediction_index))
        missing_data = sorted(set(prediction_index) - set(data_index))
        raise ValueError(
            "data/prediction keys differ: "
            f"missing_predictions={missing_predictions[:3]}, missing_data={missing_data[:3]}"
        )

    outputs: list[dict[str, Any]] = []
    for key in sorted(data_index):
        split, _identifier = key
        row = data_index[key]
        prediction = prediction_index[key]
        for data_field, prediction_field in (
            ("label", "truth"),
            ("source", "source"),
            ("category", "category"),
        ):
            if str(row.get(data_field)) != str(prediction.get(prediction_field)):
                raise ValueError(f"{key} {data_field} differs from prediction ledger")
        verdict = Verdict(str(prediction["calibrated_verdict"]))
        result = Scanner(backend=LedgerVerdictBackend(verdict)).scan(str(row["text"]))
        for evidence in result.evidence_spans:
            if str(row["text"])[evidence.start : evidence.end] != evidence.text:
                raise AssertionError(f"{key} produced non-grounded evidence")
        outputs.append(
            {
                "split": split,
                "truth": str(row["label"]),
                "verdict": result.verdict.value,
                "has_grounded_evidence": bool(result.evidence_spans),
                "product_category": result.category.value,
                "recommended_action": result.recommended_action.value,
                "signals": [signal.value for signal in result.signals],
            }
        )

    return {
        "artifact_schema_version": 1,
        "measurement": "deterministic product metadata over frozen calibrated verdicts",
        "contains_message_text": False,
        "semantic_correctness_established": False,
        "semantic_correctness_note": (
            "Coverage and runtime invariants are measured here. Independent category, evidence, "
            "and action correctness still requires blinded human review."
        ),
        "overall": summarize_outputs(outputs),
        "by_split": {
            split: summarize_outputs([output for output in outputs if output["split"] == split])
            for split in sorted(rows_by_split)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--external-data", type=Path, default=Path("data/external"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--primary-test-v8", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    available_splits = sorted({str(row.get("split")) for row in predictions})
    splits = args.splits or available_splits
    unknown = sorted(set(splits) - set(available_splits))
    if unknown:
        raise ValueError(f"prediction ledger lacks requested splits: {unknown}")
    selected_predictions = [
        row for row in predictions if str(row.get("split")) in set(splits)
    ]
    split_paths = {
        split: resolve_split_path(args.data, args.external_data, split, args.primary_test_v8)
        for split in splits
    }
    result = evaluate_product_contract(
        {split: read_jsonl(path) for split, path in split_paths.items()},
        selected_predictions,
    )
    result["data_sha256"] = {
        split: file_sha256(path) for split, path in split_paths.items()
    }
    result["prediction_ledger"] = {
        "path": str(args.predictions),
        "sha256": file_sha256(args.predictions),
        "rows": len(predictions),
        "selected_rows": len(selected_predictions),
        "contains_message_text": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
