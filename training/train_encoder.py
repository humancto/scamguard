#!/usr/bin/env python3
"""Fine-tune and calibrate the ModernBERT ScamGuard classifier."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
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
        teacher_logits: dict[str, tuple[float, float, float]] | None = None,
        include_pair_metadata: bool = False,
    ) -> None:
        self.rows = rows
        self.teacher_logits = teacher_logits
        pair_ids = sorted(
            {str(row["pair_id"]) for row in rows if str(row.get("pair_id", "")).strip()}
        )
        self.pair_groups = {pair_id: index + 1 for index, pair_id in enumerate(pair_ids)}
        self.include_pair_metadata = include_pair_metadata
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
        if self.teacher_logits is not None:
            logits = self.teacher_logits.get(str(self.rows[index]["id"]))
            item["teacher_logits"] = torch.tensor(logits or (0.0, 0.0, 0.0))
            item["retention_mask"] = torch.tensor(float(logits is not None))
        if self.include_pair_metadata:
            pair_id = str(self.rows[index].get("pair_id", ""))
            item["pair_group"] = torch.tensor(self.pair_groups.get(pair_id, 0))
            item["pair_mask"] = torch.tensor(float(bool(pair_id)))
        return item


def load_teacher_logits(
    ledger_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("ledger_sha256") != file_sha256(ledger_path):
        raise ValueError("teacher-logit ledger differs from its manifest")
    records = read_jsonl(ledger_path)
    logits_by_id: dict[str, tuple[float, float, float]] = {}
    for record in records:
        if set(record) != {"id", "logits"}:
            raise ValueError("teacher-logit record has an unexpected schema")
        identifier = str(record["id"])
        values = record["logits"]
        if (
            not identifier
            or identifier in logits_by_id
            or not isinstance(values, list)
            or len(values) != len(LABELS)
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
        ):
            raise ValueError(f"invalid teacher-logit record: {identifier!r}")
        logits_by_id[identifier] = tuple(float(value) for value in values)  # type: ignore[assignment]
    if manifest.get("rows") != len(logits_by_id):
        raise ValueError("teacher-logit manifest row count differs from ledger")
    return logits_by_id, manifest


def source_sample_weights(
    rows: list[dict[str, Any]], alpha: float
) -> tuple[list[float], dict[str, float]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("source-balance alpha must be between zero and one")
    counts = Counter(str(row["source"]) for row in rows)
    weights = [counts[str(row["source"])] ** (-alpha) for row in rows]
    total_weight = sum(weights)
    probability = {
        source: sum(
            weight
            for row, weight in zip(rows, weights, strict=True)
            if str(row["source"]) == source
        )
        / total_weight
        for source in sorted(counts)
    }
    return weights, probability


def retention_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    retention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    selected = retention_mask.to(student_logits.device).bool()
    if not selected.any():
        return student_logits.sum() * 0.0
    teacher_probabilities = functional.softmax(
        teacher_logits.to(student_logits.device)[selected] / temperature,
        dim=-1,
    )
    student_log_probabilities = functional.log_softmax(
        student_logits[selected] / temperature,
        dim=-1,
    )
    return functional.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    ) * (temperature**2)


def pairwise_scam_margin_loss(
    scam_margins: torch.Tensor,
    labels: torch.Tensor,
    pair_groups: torch.Tensor,
    pair_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Require each matched SCAM row to outrank its exact-context SAFE partner."""
    selected = pair_mask.to(scam_margins.device).bool()
    if not selected.any():
        return scam_margins.sum() * 0.0
    selected_groups = pair_groups.to(scam_margins.device)[selected]
    selected_labels = labels.to(scam_margins.device)[selected]
    selected_margins = scam_margins[selected]
    losses: list[torch.Tensor] = []
    for group in torch.unique(selected_groups):
        group_mask = selected_groups == group
        group_labels = selected_labels[group_mask]
        group_size = group_mask.sum().item()
        if group.item() <= 0 or group_size < 2 or group_size % 2:
            raise ValueError("pair-aware batch contains an incomplete pair family")
        safe = group_labels == LABEL_TO_ID["SAFE"]
        scam = group_labels == LABEL_TO_ID["SCAM"]
        expected_per_label = group_size // 2
        if (
            safe.sum().item() != expected_per_label
            or scam.sum().item() != expected_per_label
        ):
            raise ValueError("pair-aware family must contain equal SAFE and SCAM rows")
        # Repeated-pair curricula can place two complete copies of the same family in one
        # batch. Average their dropout-perturbed margins so this remains one family-level
        # constraint, while still rejecting an actually split or label-imbalanced pair.
        group_margins = selected_margins[group_mask]
        difference = group_margins[scam].mean() - group_margins[safe].mean()
        losses.append(functional.softplus(scam_margins.new_tensor(margin) - difference))
    return torch.stack(losses).mean()


class PairPreservingSampler(Sampler[int]):
    """Shuffle examples while keeping complete minimal pairs inside the same even-sized batch."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        batch_size: int,
        seed: int,
        pair_repeats: int = 1,
    ) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("pair-aware training requires a positive even batch size")
        if pair_repeats < 1:
            raise ValueError("pair-aware training requires at least one pair repeat")
        grouped: dict[str, list[int]] = {}
        ordinary: list[int] = []
        for index, row in enumerate(rows):
            pair_id = str(row.get("pair_id", "")).strip()
            if pair_id:
                grouped.setdefault(pair_id, []).append(index)
            else:
                ordinary.append(index)
        if not grouped:
            raise ValueError("pair-aware training data contains no pair families")
        pairs: list[tuple[int, int]] = []
        for pair_id, indices in sorted(grouped.items()):
            if len(indices) != 2 or {str(rows[index]["label"]) for index in indices} != {
                "SAFE",
                "SCAM",
            }:
                raise ValueError(f"invalid pair-aware family: {pair_id}")
            pairs.append((indices[0], indices[1]))
        self.ordinary = ordinary
        self.pairs = pairs
        self.batch_size = batch_size
        self.seed = seed
        self.pair_repeats = pair_repeats
        self.epoch = 0
        self.dataset_size = len(rows)
        self.row_count = len(ordinary) + 2 * len(pairs) * pair_repeats

    def __len__(self) -> int:
        return self.row_count

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        ordinary_order = torch.randperm(len(self.ordinary), generator=generator).tolist()
        ordinary = [self.ordinary[index] for index in ordinary_order]
        pair_instances = self.pairs * self.pair_repeats
        pair_order = torch.randperm(len(pair_instances), generator=generator).tolist()
        pairs: list[int] = []
        for pair_position in pair_order:
            pair = pair_instances[pair_position]
            if torch.randint(0, 2, (1,), generator=generator).item():
                pair = (pair[1], pair[0])
            pairs.extend(pair)
        tail_size = len(ordinary) % self.batch_size
        ordinary_main = ordinary[:-tail_size] if tail_size else ordinary
        ordinary_tail = ordinary[-tail_size:] if tail_size else []
        return iter(ordinary_main + pairs + ordinary_tail)


class WeightedTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor,
        binary_loss_weight: float,
        binary_positive_weight: float,
        retention_weight: float = 0.0,
        retention_temperature: float = 2.0,
        pair_loss_weight: float = 0.0,
        pair_margin: float = 2.0,
        pair_sampler: Sampler[int] | None = None,
        sample_weights: list[float] | None = None,
        sampler_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.binary_loss_weight = binary_loss_weight
        self.binary_positive_weight = binary_positive_weight
        self.retention_weight = retention_weight
        self.retention_temperature = retention_temperature
        self.pair_loss_weight = pair_loss_weight
        self.pair_margin = pair_margin
        self.pair_sampler = pair_sampler
        self.sample_weights = sample_weights
        self.sampler_seed = sampler_seed

    def _get_train_sampler(self, train_dataset: Dataset | None = None) -> Any:
        if self.pair_sampler is not None:
            dataset = train_dataset or self.train_dataset
            if dataset is None or len(dataset) != self.pair_sampler.dataset_size:
                raise ValueError("pair-aware sampler differs from training dataset")
            return self.pair_sampler
        if self.sample_weights is None:
            return super()._get_train_sampler(train_dataset)
        dataset = train_dataset or self.train_dataset
        if dataset is None or len(dataset) != len(self.sample_weights):
            raise ValueError("source-balanced sample weights differ from training dataset")
        generator = torch.Generator()
        generator.manual_seed(self.sampler_seed)
        return WeightedRandomSampler(
            self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
            generator=generator,
        )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        teacher_logits = inputs.pop("teacher_logits", None)
        retention_mask = inputs.pop("retention_mask", None)
        pair_groups = inputs.pop("pair_group", None)
        pair_mask = inputs.pop("pair_mask", None)
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
        # Retention is a training-only objective. Evaluation datasets intentionally contain no
        # teacher fields so their metrics remain the ordinary supervised product contract.
        if self.retention_weight and model.training:
            if teacher_logits is None or retention_mask is None:
                raise ValueError("retention loss requires teacher logits and a retention mask")
            retention_loss = retention_kl_loss(
                outputs.logits,
                teacher_logits,
                retention_mask,
                self.retention_temperature,
            )
            loss = loss + self.retention_weight * retention_loss
        if self.pair_loss_weight and model.training:
            if pair_groups is None or pair_mask is None:
                raise ValueError("pair loss requires pair groups and a pair mask")
            pair_loss = pairwise_scam_margin_loss(
                scam_margin,
                labels,
                pair_groups,
                pair_mask,
                self.pair_margin,
            )
            loss = loss + self.pair_loss_weight * pair_loss
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


def paired_validation_metrics(
    rows: list[dict[str, Any]], logits: np.ndarray, temperature: float
) -> dict[str, float | int]:
    probabilities = softmax(logits, temperature)[:, LABEL_TO_ID["SCAM"]]
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError("paired validation row lacks pair_id")
        grouped.setdefault(pair_id, []).append(index)
    gaps: list[float] = []
    correctly_ordered = 0
    for pair_id, indices in grouped.items():
        if len(indices) != 2 or {str(rows[index]["label"]) for index in indices} != {
            "SAFE",
            "SCAM",
        }:
            raise ValueError(f"invalid paired validation family: {pair_id}")
        safe_index = next(index for index in indices if rows[index]["label"] == "SAFE")
        scam_index = next(index for index in indices if rows[index]["label"] == "SCAM")
        gap = float(probabilities[scam_index] - probabilities[safe_index])
        gaps.append(gap)
        correctly_ordered += int(gap > 0.0)
    return {
        "pairs": len(gaps),
        "correctly_ordered": correctly_ordered,
        "pair_order_accuracy": correctly_ordered / len(gaps),
        "scam_probability_gap_mean": float(np.mean(gaps)),
        "scam_probability_gap_p05": float(np.percentile(gaps, 5)),
        "scam_probability_gap_min": float(np.min(gaps)),
    }


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
        "--init-checkpoint",
        type=Path,
        help="Initialize a new continual-training run from this local classifier checkpoint.",
    )
    parser.add_argument(
        "--teacher-logits",
        type=Path,
        help="Text-free JSONL teacher logits used to retain the prior decision boundary.",
    )
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        help="Manifest pinning the teacher-logit ledger, source data, and checkpoint.",
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
    parser.add_argument("--truncation-side", choices=("left", "right"), default="right")
    parser.add_argument("--dialogue-policy", choices=DIALOGUE_POLICIES, default="none")
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument("--retention-weight", type=float, default=0.0)
    parser.add_argument("--retention-temperature", type=float, default=2.0)
    parser.add_argument("--pair-loss-weight", type=float, default=0.0)
    parser.add_argument("--pair-margin", type=float, default=2.0)
    parser.add_argument("--pair-repeats", type=int, default=1)
    parser.add_argument(
        "--source-balance-alpha",
        type=float,
        default=0.0,
        help="Per-row source weight is source_count ** -alpha; zero keeps random sampling.",
    )
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
    if args.evaluate_only and (args.resume_from_checkpoint or args.init_checkpoint):
        parser.error(
            "--resume-from-checkpoint/--init-checkpoint cannot be combined with --evaluate-only"
        )
    if args.init_checkpoint and args.resume_from_checkpoint:
        parser.error("--init-checkpoint and --resume-from-checkpoint are mutually exclusive")
    if args.retention_weight < 0 or args.retention_temperature <= 0:
        parser.error("retention weight must be nonnegative and temperature must be positive")
    if args.pair_loss_weight < 0 or args.pair_margin <= 0:
        parser.error("pair loss weight must be nonnegative and pair margin must be positive")
    if args.pair_repeats < 1:
        parser.error("pair repeats must be at least one")
    if args.pair_repeats != 1 and not args.pair_loss_weight:
        parser.error("pair repeats greater than one require pair-aware training")
    if args.pair_loss_weight and args.batch_size % 2:
        parser.error("pair-aware training requires an even --batch-size")
    if args.pair_loss_weight and args.source_balance_alpha:
        parser.error("pair-aware sampling and source-balanced replacement sampling are exclusive")
    if not 0.0 <= args.source_balance_alpha <= 1.0:
        parser.error("--source-balance-alpha must be between zero and one")
    if args.retention_weight and not (
        args.init_checkpoint and args.teacher_logits and args.teacher_manifest
    ):
        parser.error(
            "retention loss requires --init-checkpoint, --teacher-logits, and --teacher-manifest"
        )
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
    call_pair_validation_path = args.data / "call_pair_validation.jsonl"
    if call_pair_validation_path.exists():
        row_paths["call_pair_validation"] = call_pair_validation_path
    call_window_validation_path = args.data / "call_window_validation.jsonl"
    if call_window_validation_path.exists():
        row_paths["call_window_validation"] = call_window_validation_path
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
    elif args.init_checkpoint:
        if not args.init_checkpoint.is_dir():
            raise FileNotFoundError(f"missing initialization checkpoint: {args.init_checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.init_checkpoint,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.init_checkpoint,
            local_files_only=True,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
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
    tokenizer.truncation_side = args.truncation_side
    teacher_logits: dict[str, tuple[float, float, float]] | None = None
    teacher_manifest: dict[str, object] | None = None
    if args.teacher_logits or args.teacher_manifest:
        if not args.teacher_logits or not args.teacher_manifest:
            parser.error("--teacher-logits and --teacher-manifest must be provided together")
        teacher_logits, teacher_manifest = load_teacher_logits(
            args.teacher_logits,
            args.teacher_manifest,
        )
        train_ids = {str(row["id"]) for row in rows["train"]}
        unexpected_teacher_ids = set(teacher_logits) - train_ids
        if unexpected_teacher_ids:
            raise ValueError(
                f"teacher ledger contains {len(unexpected_teacher_ids)} IDs outside training data"
            )
        if args.init_checkpoint:
            init_model_file = args.init_checkpoint / "model.safetensors"
            if not init_model_file.is_file():
                init_model_file = args.init_checkpoint / "pytorch_model.bin"
            if teacher_manifest.get("checkpoint_model_sha256") != file_sha256(init_model_file):
                raise ValueError("teacher ledger checkpoint differs from initialization model")
    datasets = {}
    for split, split_rows in rows.items():
        datasets[split] = EncodedDataset(
            split_rows,
            tokenizer,
            args.max_length,
            dialogue_policy=args.dialogue_policy,
            teacher_logits=teacher_logits if split == "train" else None,
            include_pair_metadata=bool(args.pair_loss_weight and split == "train"),
        )

    sample_weights: list[float] | None = None
    source_sampling_probability: dict[str, float] | None = None
    if args.source_balance_alpha:
        sample_weights, source_sampling_probability = source_sample_weights(
            rows["train"], args.source_balance_alpha
        )
    pair_sampler: PairPreservingSampler | None = None
    if args.pair_loss_weight:
        pair_sampler = PairPreservingSampler(
            rows["train"], args.batch_size, args.seed, args.pair_repeats
        )

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
        remove_unused_columns=False,
        train_sampling_strategy=(
            "random"
            if args.evaluate_only or args.source_balance_alpha or args.pair_loss_weight
            else "group_by_length"
        ),
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
        retention_weight=args.retention_weight,
        retention_temperature=args.retention_temperature,
        pair_loss_weight=args.pair_loss_weight,
        pair_margin=args.pair_margin,
        pair_sampler=pair_sampler,
        sample_weights=sample_weights,
        sampler_seed=args.seed,
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
            "initialization_checkpoint": (
                str(args.init_checkpoint) if args.init_checkpoint else None
            ),
            "retention_weight": args.retention_weight,
            "retention_temperature": args.retention_temperature,
            "pairwise": (
                "softplus(target_margin - (SCAM scam-margin - matched SAFE scam-margin))"
            ),
            "pair_loss_weight": args.pair_loss_weight,
            "pair_margin": args.pair_margin,
            "pair_repeats": args.pair_repeats,
            "pair_sampler": (
                "complete pairs kept inside even-sized batches"
                if args.pair_loss_weight
                else None
            ),
            "teacher_logit_manifest": teacher_manifest,
            "teacher_anchor_rows": len(teacher_logits or {}),
            "source_balance_alpha": args.source_balance_alpha,
            "source_sampling_probability": source_sampling_probability,
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
            "truncation_side": tokenizer.truncation_side,
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
        "call_pair_validation",
        "call_window_validation",
    ):
        if split not in rows:
            continue
        result[split] = report_slice(rows[split], predictions[split], temperature, threshold)
    if "call_pair_validation" in rows:
        result["call_pair_validation"]["paired_ranking"] = paired_validation_metrics(
            rows["call_pair_validation"],
            predictions["call_pair_validation"],
            temperature,
        )
        pair_binary = result["call_pair_validation"]["binary_safety"]
        pair_ranking = result["call_pair_validation"]["paired_ranking"]
        result["call_pair_validation"]["gates"] = {
            "recall": pair_binary["scam_recall"] >= 0.97,
            "fpr": pair_binary["false_positive_rate"] <= args.max_fpr,
            "perfect_pair_order": pair_ranking["pair_order_accuracy"] == 1.0,
        }
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
