#!/usr/bin/env python3
"""Fine-tune and calibrate the ModernBERT ScamGuard classifier."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from scamguard.metrics import binary_safety_metrics, choose_threshold, file_sha256, wilson_interval
from scamguard.preprocessing import DIALOGUE_POLICIES, prepare_model_text

LABELS = ("SAFE", "UNCERTAIN", "SCAM")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class EncodedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        dialogue_policy: str = "none",
    ) -> None:
        self.rows = rows
        self.encodings = tokenizer(
            [prepare_model_text(str(row["text"]), dialogue_policy) for row in rows],
            max_length=max_length,
            truncation=True,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(LABEL_TO_ID[str(self.rows[index]["label"])])
        return item


class WeightedTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor,
        binary_loss_weight: float,
        binary_positive_weight: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.binary_loss_weight = binary_loss_weight
        self.binary_positive_weight = binary_positive_weight

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        multiclass_loss = functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        # Product safety is a binary boundary layered over the three-way response contract. The
        # auxiliary margin trains SCAM against SAFE-or-UNCERTAIN directly, matching calibration and
        # the release gate instead of asking generic three-class cross entropy to discover it.
        scam_margin = outputs.logits[:, LABEL_TO_ID["SCAM"]] - torch.logsumexp(
            outputs.logits[:, : LABEL_TO_ID["SCAM"]], dim=1
        )
        binary_labels = (labels == LABEL_TO_ID["SCAM"]).to(outputs.logits.dtype)
        binary_loss = functional.binary_cross_entropy_with_logits(
            scam_margin,
            binary_labels,
            pos_weight=torch.tensor(
                self.binary_positive_weight,
                dtype=outputs.logits.dtype,
                device=outputs.logits.device,
            ),
        )
        loss = multiclass_loss + self.binary_loss_weight * binary_loss
        return (loss, outputs) if return_outputs else loss


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    adjusted = logits / temperature
    adjusted -= adjusted.max(axis=1, keepdims=True)
    exponentials = np.exp(adjusted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, truth: np.ndarray) -> float:
    def negative_log_likelihood(log_temperature: float) -> float:
        probabilities = softmax(logits, math.exp(log_temperature))
        selected = probabilities[np.arange(len(truth)), truth]
        return float(-np.log(np.clip(selected, 1e-9, 1.0)).mean())

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(math.log(0.05), math.log(10.0)),
        method="bounded",
    )
    return float(math.exp(result.x))


def safety_selection_metrics(
    logits: np.ndarray, labels: np.ndarray, max_fpr: float
) -> dict[str, float]:
    """Compute the development-only checkpoint objective at the product FPR cap."""

    probabilities = softmax(logits)
    mask = np.isin(labels, [LABEL_TO_ID["SAFE"], LABEL_TO_ID["SCAM"]])
    binary_truth = (labels[mask] == LABEL_TO_ID["SCAM"]).astype(int)
    scam_probabilities = probabilities[mask, LABEL_TO_ID["SCAM"]]
    if not np.any(binary_truth == 0) or not np.any(binary_truth == 1):
        raise ValueError("safety selection requires both SAFE and SCAM development labels")
    threshold = choose_threshold(binary_truth, scam_probabilities, max_fpr)
    metrics = binary_safety_metrics(binary_truth, scam_probabilities, threshold)
    return {
        "safety_recall_at_fpr": float(metrics["scam_recall"]),
        "safety_precision_at_fpr": float(metrics["scam_precision"]),
        "safety_fpr": float(metrics["false_positive_rate"]),
        "safety_threshold": float(threshold),
    }


def binary_subset(
    rows: list[dict[str, Any]], probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.array([row["label"] in {"SAFE", "SCAM"} for row in rows])
    truth = np.array([int(row["label"] == "SCAM") for row in rows])[mask]
    return truth, probabilities[mask, LABEL_TO_ID["SCAM"]]


def predict_in_dataset_order(
    trainer: Trainer,
    dataset: Dataset[dict[str, torch.Tensor]],
    rows: list[dict[str, Any]],
) -> np.ndarray:
    """Return logits only when Trainer preserved the source row order.

    Transformers 5 can apply the length-grouped training sampler during prediction. The returned
    labels are reordered with the logits, so Trainer metrics remain correct while row metadata no
    longer lines up with the predictions. ScamGuard reports source, category, and evidence slices;
    accepting that silent permutation would invalidate every slice.
    """

    # The checkpoint-selection callback is deliberately defined only for development data that
    # contains both SAFE and SCAM examples. Some diagnostic splits are single-class by design, so
    # invoking that callback during ``predict`` would make otherwise valid reporting fail. Slice
    # metrics are computed below with the development-selected threshold.
    had_compute_metrics = hasattr(trainer, "compute_metrics")
    compute_metrics = getattr(trainer, "compute_metrics", None)
    trainer.compute_metrics = None
    try:
        output = trainer.predict(dataset)
    finally:
        if had_compute_metrics:
            trainer.compute_metrics = compute_metrics
        else:
            del trainer.compute_metrics
    expected_labels = np.array([LABEL_TO_ID[str(row["label"])] for row in rows])
    if output.label_ids is None or not np.array_equal(output.label_ids, expected_labels):
        raise RuntimeError(
            "prediction sampler changed row order; use sequential prediction before reporting"
        )
    return output.predictions


def report_slice(
    rows: list[dict[str, Any]],
    logits: np.ndarray,
    temperature: float,
    threshold: float,
    *,
    include_sources: bool = True,
) -> dict[str, Any]:
    probabilities = softmax(logits, temperature)
    truth = np.array([LABEL_TO_ID[str(row["label"])] for row in rows])
    predicted = probabilities.argmax(axis=1)
    binary_truth, scam_probabilities = binary_subset(rows, probabilities)
    result: dict[str, Any] = {
        "examples": len(rows),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "accuracy_argmax": float(accuracy_score(truth, predicted)),
        "macro_f1_argmax": float(
            f1_score(
                truth,
                predicted,
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_argmax": confusion_matrix(
            truth, predicted, labels=list(range(len(LABELS)))
        ).tolist(),
        "binary_safety": (
            binary_safety_metrics(binary_truth, scam_probabilities, threshold)
            if len(binary_truth)
            else None
        ),
    }
    scam_categories = sorted({str(row["category"]) for row in rows if row["label"] == "SCAM"})
    result["scam_by_category"] = {}
    for category in scam_categories:
        indices = [
            index
            for index, row in enumerate(rows)
            if row["label"] == "SCAM" and row["category"] == category
        ]
        category_probabilities = probabilities[indices, LABEL_TO_ID["SCAM"]]
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
            result["by_source"] = {}
            for source in sources:
                indices = [index for index, row in enumerate(rows) if row["source"] == source]
                result["by_source"][source] = report_slice(
                    [rows[index] for index in indices],
                    logits[indices],
                    temperature,
                    threshold,
                    include_sources=False,
                )
        if any(row.get("source_language") for row in rows):
            languages = sorted(
                {str(row.get("source_language") or "UNSPECIFIED") for row in rows}
            )
            result["by_language"] = {}
            for language in languages:
                indices = [
                    index
                    for index, row in enumerate(rows)
                    if str(row.get("source_language") or "UNSPECIFIED") == language
                ]
                result["by_language"][language] = report_slice(
                    [rows[index] for index in indices],
                    logits[indices],
                    temperature,
                    threshold,
                    include_sources=False,
                )
    return result


def latency(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    max_length: int,
    dialogue_policy: str,
) -> dict[str, float | int | str]:
    model.eval()

    def synchronize() -> None:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    forward_durations = []
    end_to_end_durations = []
    samples = texts[: min(len(texts), 250)]
    with torch.inference_mode():
        for text in samples[:8]:
            text = prepare_model_text(text, dialogue_policy)
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            model(**{key: value.to(device) for key, value in encoded.items()})
        synchronize()
        for text in samples:
            text = prepare_model_text(text, dialogue_policy)
            end_to_end_started = time.perf_counter_ns()
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {key: value.to(device) for key, value in encoded.items()}
            forward_started = time.perf_counter_ns()
            output = model(**inputs)
            torch.softmax(output.logits, dim=-1)
            synchronize()
            finished = time.perf_counter_ns()
            forward_durations.append((finished - forward_started) / 1_000_000)
            end_to_end_durations.append((finished - end_to_end_started) / 1_000_000)
    return {
        # Compatibility aliases retained for historical report readers.
        "median_ms": float(np.median(forward_durations)),
        "p95_ms": float(np.percentile(forward_durations, 95)),
        "model_forward_median_ms": float(np.median(forward_durations)),
        "model_forward_p95_ms": float(np.percentile(forward_durations, 95)),
        "end_to_end_median_ms": float(np.median(end_to_end_durations)),
        "end_to_end_p95_ms": float(np.percentile(end_to_end_durations, 95)),
        "samples": len(forward_durations),
        "scope": (
            "batch-one tokenizer plus device transfer plus model forward plus probability "
            "transform; excludes SDK evidence extraction and I/O"
        ),
    }


def sequence_window_report(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int, dialogue_policy: str
) -> dict[str, float | int | str]:
    """Make truncation visible instead of silently treating long dialogue as short text."""
    encodings = tokenizer(
        [prepare_model_text(str(row["text"]), dialogue_policy) for row in rows],
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )
    lengths = np.array([len(values) for values in encodings["input_ids"]])
    return {
        "max_tokens": max_length,
        "truncation_side": str(tokenizer.truncation_side),
        "truncated_examples": int(np.sum(lengths > max_length)),
        "truncated_fraction": float(np.mean(lengths > max_length)),
        "token_length_p50": int(np.percentile(lengths, 50)),
        "token_length_p95": int(np.percentile(lengths, 95)),
        "token_length_max": int(lengths.max()),
    }


def artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", default="8949b909ec900327062f0ebf497f51aef5e6f0c8")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--external-data", type=Path, default=Path("data/external"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/checkpoints/sg-modernbert-schema9-safety")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Existing Trainer checkpoint to load in evaluate-only mode; output remains the export "
            "path."
        ),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/runs/sg-modernbert-schema9-safety.json")
    )
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dialogue-policy", choices=DIALOGUE_POLICIES, default="none")
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--binary-positive-weight",
        type=float,
        default=0.0,
        help="Positive weight for the SCAM boundary; zero derives it from the training labels.",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume optimizer, scheduler, and RNG state from a Trainer checkpoint.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Recompute calibration and reports from the checkpoint already at --output.",
    )
    args = parser.parse_args()
    if args.evaluate_only and args.resume_from_checkpoint:
        parser.error("--resume-from-checkpoint cannot be combined with --evaluate-only")
    set_seed(args.seed)

    row_paths = {
        split: args.data / f"{split}.jsonl" for split in ("train", "dev", "test")
    }
    row_paths["ood_financial"] = args.data / "ood_financial.jsonl"
    wspr_path = args.data / "ood_wspr.jsonl"
    if wspr_path.exists():
        row_paths["ood_wspr"] = wspr_path
    for split in ("forum_validation", "ood_forum", "ood_forum_materialized", "ood_azsc"):
        path = args.data / f"{split}.jsonl"
        if path.exists():
            row_paths[split] = path
    adversarial_path = args.data / "adversarial.jsonl"
    if adversarial_path.exists():
        row_paths["adversarial"] = adversarial_path
    chichewa_path = args.external_data / "chichewa" / "ood_chichewa.jsonl"
    if chichewa_path.exists():
        row_paths["ood_chichewa"] = chichewa_path
    scam_dialogue_path = (
        args.external_data / "scam_dialogue" / "scam_dialogue_validation.jsonl"
    )
    if scam_dialogue_path.exists():
        row_paths["scam_dialogue_validation"] = scam_dialogue_path
    taskmaster_path = args.external_data / "taskmaster" / "taskmaster_validation.jsonl"
    if taskmaster_path.exists():
        row_paths["taskmaster_validation"] = taskmaster_path
    rows = {split: read_jsonl(path) for split, path in row_paths.items()}
    load_path = args.checkpoint or args.output
    if args.evaluate_only:
        if not load_path.is_dir():
            raise FileNotFoundError(f"missing encoder checkpoint: {load_path}")
        tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            load_path, local_files_only=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            revision=args.revision,
            num_labels=len(LABELS),
            id2label={index: label for label, index in LABEL_TO_ID.items()},
            label2id=LABEL_TO_ID,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    datasets = {
        split: EncodedDataset(
            split_rows,
            tokenizer,
            args.max_length,
            dialogue_policy=args.dialogue_policy,
        )
        for split, split_rows in rows.items()
    }

    counts = Counter(str(row["label"]) for row in rows["train"])
    total = sum(counts.values())
    # Square-root balancing avoids letting the minority weight dominate noisy public labels.
    weights = torch.tensor(
        [math.sqrt(total / (len(LABELS) * counts[label])) for label in LABELS],
        dtype=torch.float32,
    )
    binary_positive_weight = args.binary_positive_weight or (
        (counts["SAFE"] + counts["UNCERTAIN"]) / counts["SCAM"]
    )
    training_args = TrainingArguments(
        output_dir=str(args.output.parent / (args.output.name + "-trainer")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=0.08,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="safety_recall_at_fpr",
        greater_is_better=True,
        save_total_limit=1,
        report_to=[],
        train_sampling_strategy="random" if args.evaluate_only else "group_by_length",
        seed=args.seed,
        data_seed=args.seed,
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        predicted = prediction.predictions.argmax(axis=1)
        metrics = {
            "macro_f1": float(
                f1_score(prediction.label_ids, predicted, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(prediction.label_ids, predicted)),
        }
        metrics.update(
            safety_selection_metrics(prediction.predictions, prediction.label_ids, args.max_fpr)
        )
        return metrics

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=weights,
        binary_loss_weight=args.binary_loss_weight,
        binary_positive_weight=binary_positive_weight,
    )
    if args.evaluate_only:
        previous = json.loads(args.report.read_text()) if args.report.exists() else {}
        train_metrics = previous.get("train", {})
        trainer_state_path = load_path / "trainer_state.json"
        if not train_metrics and trainer_state_path.exists():
            trainer_state = json.loads(trainer_state_path.read_text())
            train_metrics = next(
                (
                    entry
                    for entry in reversed(trainer_state.get("log_history", []))
                    if "train_runtime" in entry
                ),
                {},
            )
    else:
        resume = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        train_metrics = trainer.train(resume_from_checkpoint=resume).metrics

    # Persist the best inference checkpoint before diagnostics. A reporting failure must never
    # strand a completed run only inside an optimizer-heavy Trainer checkpoint.
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only or load_path.resolve() != args.output.resolve():
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)

    # Prediction must be sequential because every report joins logits back to row metadata. The
    # explicit assertion below is the guardrail; merely trusting a sampler setting is insufficient.
    trainer.args.train_sampling_strategy = "random"
    predictions = {
        split: predict_in_dataset_order(trainer, dataset, rows[split])
        for split, dataset in datasets.items()
        if split != "train"
    }
    dev_truth = np.array([LABEL_TO_ID[str(row["label"])] for row in rows["dev"]])
    temperature = fit_temperature(predictions["dev"], dev_truth)
    dev_probabilities = softmax(predictions["dev"], temperature)
    dev_binary_truth, dev_scam_probabilities = binary_subset(rows["dev"], dev_probabilities)
    threshold = choose_threshold(dev_binary_truth, dev_scam_probabilities, args.max_fpr)

    calibration = {
        "model_id": args.output.name,
        "temperature": temperature,
        "scam_threshold": threshold,
        "safe_threshold": 0.20,
        "labels": list(LABELS),
        "threshold_source": "dev SAFE/SCAM only",
    }
    (args.output / "scamguard_calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )

    device = trainer.model.device
    external_data_manifests = {}
    for diagnostic in ("chichewa", "scam_dialogue", "taskmaster"):
        manifest_path = args.external_data / diagnostic / "manifest.json"
        if manifest_path.exists():
            external_data_manifests[diagnostic] = json.loads(manifest_path.read_text())
    result: dict[str, Any] = {
        "model_id": args.output.name,
        "base_model": args.model,
        "base_model_revision": args.revision,
        "seed": args.seed,
        "temperature": temperature,
        "scam_threshold": threshold,
        "class_weights": weights.tolist(),
        "training_objective": {
            "multiclass": "class-weighted cross entropy",
            "binary": "SCAM versus logsumexp(SAFE, UNCERTAIN) BCEWithLogits",
            "binary_loss_weight": args.binary_loss_weight,
            "binary_positive_weight": binary_positive_weight,
            "checkpoint_selection": "development recall at the configured maximum FPR",
            "gradient_accumulation": args.gradient_accumulation,
            "gradient_checkpointing": args.gradient_checkpointing,
            "resumed_from_checkpoint": (
                str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
            ),
        },
        "data_manifest": json.loads((args.data / "manifest.json").read_text()),
        "external_data_manifests": external_data_manifests,
        "data_sha256": {split: file_sha256(row_paths[split]) for split in rows},
        "input_transform": {
            "dialogue_policy": args.dialogue_policy,
            "application": "before tokenization for fitting, evaluation, and latency",
        },
        "sequence_windows": {
            split: sequence_window_report(
                split_rows, tokenizer, args.max_length, args.dialogue_policy
            )
            for split, split_rows in rows.items()
        },
        "train": train_metrics,
        "evaluation_mode": "checkpoint_only" if args.evaluate_only else "post_training",
        "targets": {
            "scam_recall_min": 0.97,
            "false_positive_rate_max": args.max_fpr,
            "macro_f1_stretch": 0.94,
            "desktop_fast_path_latency_p95_ms_max": 20.0,
            "mobile_latency_status": "must be measured on a physical target device",
        },
    }
    for split in (
        "dev",
        "test",
        "ood_financial",
        "ood_wspr",
        "forum_validation",
        "ood_forum",
        "ood_forum_materialized",
        "adversarial",
        "ood_azsc",
        "ood_chichewa",
        "scam_dialogue_validation",
        "taskmaster_validation",
    ):
        if split not in rows:
            continue
        result[split] = report_slice(rows[split], predictions[split], temperature, threshold)
    result["latency"] = latency(
        trainer.model,
        tokenizer,
        [str(row["text"]) for row in rows["test"]],
        device,
        args.max_length,
        args.dialogue_policy,
    )
    result["artifact_bytes"] = artifact_size(args.output)
    result["environment"] = {
        "python_arch": platform.machine(),
        "torch": torch.__version__,
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
    }
    test_binary = result["test"]["binary_safety"]
    core_categories = {
        category: values
        for category, values in result["test"]["scam_by_category"].items()
        if values["examples"] >= 20
    }
    result["test_gates"] = {
        "recall": test_binary["scam_recall"] >= 0.97,
        "fpr": test_binary["false_positive_rate"] <= args.max_fpr,
        "core_category_recall": bool(core_categories)
        and all(values["recall"] >= 0.97 for values in core_categories.values()),
        "core_category_min_examples": 20,
        "core_categories_evaluated": sorted(core_categories),
        "macro_f1_stretch": result["test"]["macro_f1_argmax"] >= 0.94,
        "desktop_fast_path_latency": result["latency"]["end_to_end_p95_ms"] <= 20.0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
