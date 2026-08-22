#!/usr/bin/env python3
"""Fine-tune and calibrate the ModernBERT ScamGuard classifier."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
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
ACTION_TARGETS = (
    "sensitive_action_language",
    "requested_disclosure_or_transfer",
    "caller_controls_target",
    "official_self_navigation",
    "independent_verification",
    "pressure_or_secrecy",
    "irreversible_action",
)


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
        action_target_names: tuple[str, ...] = (),
        action_verdict_weight: float = 1.0,
    ) -> None:
        self.rows = rows
        self.teacher_logits = teacher_logits
        pair_ids = sorted(
            {str(row["pair_id"]) for row in rows if str(row.get("pair_id", "")).strip()}
        )
        self.pair_groups = {pair_id: index + 1 for index, pair_id in enumerate(pair_ids)}
        self.include_pair_metadata = include_pair_metadata
        self.action_target_names = action_target_names
        self.action_verdict_weight = action_verdict_weight
        if not 0.0 < action_verdict_weight <= 1.0:
            raise ValueError("action verdict weight must be in (0, 1]")
        for row in rows:
            targets = row.get("action_targets")
            if targets is not None and (
                not isinstance(targets, dict)
                or tuple(targets) != action_target_names
                or not all(isinstance(value, bool) for value in targets.values())
            ):
                raise ValueError(f"invalid action targets for row {row.get('id')!r}")
            row_verdict_weight = row.get("action_verdict_weight")
            if row_verdict_weight is not None and (
                not isinstance(row_verdict_weight, (int, float))
                or isinstance(row_verdict_weight, bool)
                or not 0.0 < float(row_verdict_weight) <= 1.0
            ):
                raise ValueError(f"invalid action verdict weight for row {row.get('id')!r}")
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
        if self.action_target_names:
            targets = self.rows[index].get("action_targets")
            has_targets = isinstance(targets, dict)
            item["action_targets"] = torch.tensor(
                [float(targets[name]) for name in self.action_target_names]
                if has_targets
                else [0.0] * len(self.action_target_names)
            )
            item["action_mask"] = torch.tensor(float(has_targets))
            row_verdict_weight = self.rows[index].get("action_verdict_weight")
            item["verdict_weight"] = torch.tensor(
                float(row_verdict_weight)
                if row_verdict_weight is not None
                else (self.action_verdict_weight if has_targets else 1.0)
            )
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


def expand_classifier_for_action_targets(
    model: torch.nn.Module,
    action_target_names: tuple[str, ...],
    seed: int,
) -> None:
    """Add independent auxiliary logits while preserving the frozen verdict head exactly."""
    if not action_target_names:
        return
    if tuple(action_target_names) != ACTION_TARGETS:
        raise ValueError("action targets differ from the frozen ScamGuard target order")
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, torch.nn.Linear):
        raise TypeError("action-target expansion requires a linear sequence classifier")
    expected_outputs = len(LABELS) + len(action_target_names)
    if classifier.out_features == expected_outputs:
        saved_targets = tuple(getattr(model.config, "scamguard_action_targets", ()))
        if saved_targets != action_target_names:
            raise ValueError("expanded classifier action-target metadata differs")
        return
    if classifier.out_features != len(LABELS):
        raise ValueError(
            f"cannot expand classifier with {classifier.out_features} existing outputs"
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expanded = torch.nn.Linear(
            classifier.in_features,
            expected_outputs,
            bias=classifier.bias is not None,
        )
        torch.nn.init.normal_(expanded.weight, mean=0.0, std=0.02)
        if expanded.bias is not None:
            torch.nn.init.zeros_(expanded.bias)
    expanded = expanded.to(device=classifier.weight.device, dtype=classifier.weight.dtype)
    with torch.no_grad():
        expanded.weight[: len(LABELS)].copy_(classifier.weight)
        if classifier.bias is not None and expanded.bias is not None:
            expanded.bias[: len(LABELS)].copy_(classifier.bias)
    model.classifier = expanded
    model.num_labels = expected_outputs
    model.config.num_labels = expected_outputs
    model.config.id2label = {
        index: label
        for index, label in enumerate(
            LABELS + tuple(f"ACTION_{name}" for name in action_target_names)
        )
    }
    model.config.label2id = {label: index for index, label in model.config.id2label.items()}
    model.config.scamguard_verdict_labels = list(LABELS)
    model.config.scamguard_action_targets = list(action_target_names)


def masked_action_bce_loss(
    action_logits: torch.Tensor,
    action_targets: torch.Tensor,
    action_mask: torch.Tensor,
    positive_weights: torch.Tensor,
) -> torch.Tensor:
    selected = action_mask.to(action_logits.device).bool()
    if not selected.any():
        return action_logits.sum() * 0.0
    return functional.binary_cross_entropy_with_logits(
        action_logits[selected],
        action_targets.to(action_logits.device)[selected],
        pos_weight=positive_weights.to(action_logits.device),
    )


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
        action_target_names: tuple[str, ...] = (),
        action_loss_weight: float = 0.0,
        action_positive_weights: torch.Tensor | None = None,
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
        self.action_target_names = action_target_names
        self.action_loss_weight = action_loss_weight
        self.action_positive_weights = action_positive_weights

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
        action_targets = inputs.pop("action_targets", None)
        action_mask = inputs.pop("action_mask", None)
        verdict_weight = inputs.pop("verdict_weight", None)
        outputs = model(**inputs)
        if outputs.logits.shape[1] < len(LABELS):
            raise ValueError("classifier exposes fewer than three verdict logits")
        verdict_logits = outputs.logits[:, : len(LABELS)]
        row_weights = (
            verdict_weight.to(verdict_logits.device)
            if verdict_weight is not None
            else torch.ones_like(labels, dtype=verdict_logits.dtype)
        )
        multiclass_rows = functional.cross_entropy(
            verdict_logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
            reduction="none",
        )
        multiclass_loss = (multiclass_rows * row_weights).sum() / row_weights.sum()
        # Product safety is a binary boundary layered over the three-way response contract. The
        # auxiliary margin trains SCAM against SAFE-or-UNCERTAIN directly, matching calibration and
        # the release gate instead of asking generic three-class cross entropy to discover it.
        scam_margin = verdict_logits[:, LABEL_TO_ID["SCAM"]] - torch.logsumexp(
            verdict_logits[:, : LABEL_TO_ID["SCAM"]], dim=1
        )
        binary_labels = (labels == LABEL_TO_ID["SCAM"]).to(outputs.logits.dtype)
        binary_rows = functional.binary_cross_entropy_with_logits(
            scam_margin,
            binary_labels,
            pos_weight=torch.tensor(
                self.binary_positive_weight,
                dtype=outputs.logits.dtype,
                device=outputs.logits.device,
            ),
            reduction="none",
        )
        binary_loss = (binary_rows * row_weights).sum() / row_weights.sum()
        loss = multiclass_loss + self.binary_loss_weight * binary_loss
        # Retention is a training-only objective. Evaluation datasets intentionally contain no
        # teacher fields so their metrics remain the ordinary supervised product contract.
        if self.retention_weight and model.training:
            if teacher_logits is None or retention_mask is None:
                raise ValueError("retention loss requires teacher logits and a retention mask")
            retention_loss = retention_kl_loss(
                verdict_logits,
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
        if self.action_loss_weight and model.training:
            if (
                action_targets is None
                or action_mask is None
                or self.action_positive_weights is None
            ):
                raise ValueError("action loss requires targets, mask, and positive weights")
            expected_outputs = len(LABELS) + len(self.action_target_names)
            if outputs.logits.shape[1] != expected_outputs:
                raise ValueError("classifier output count differs from action-target contract")
            action_loss = masked_action_bce_loss(
                outputs.logits[:, len(LABELS) :],
                action_targets,
                action_mask,
                self.action_positive_weights,
            )
            loss = loss + self.action_loss_weight * action_loss
        return (loss, outputs) if return_outputs else loss


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    adjusted = logits / temperature
    adjusted -= adjusted.max(axis=1, keepdims=True)
    exponentials = np.exp(adjusted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, truth: np.ndarray) -> float:
    verdict = logits[:, : len(LABELS)]

    def negative_log_likelihood(log_temperature: float) -> float:
        probabilities = softmax(verdict, math.exp(log_temperature))
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

    probabilities = softmax(logits[:, : len(LABELS)])
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
    probabilities = softmax(logits[:, : len(LABELS)], temperature)
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
        if any(row.get("source_domain") for row in rows):
            domains = sorted({str(row.get("source_domain")) for row in rows})
            result["by_source_domain"] = {}
            for domain in domains:
                indices = [
                    index
                    for index, row in enumerate(rows)
                    if str(row.get("source_domain")) == domain
                ]
                result["by_source_domain"][domain] = report_slice(
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
    probabilities = softmax(logits[:, : len(LABELS)], temperature)[
        :, LABEL_TO_ID["SCAM"]
    ]
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


def action_target_metrics(
    rows: list[dict[str, Any]],
    logits: np.ndarray,
    action_target_names: tuple[str, ...],
    calibrated_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    indices = [
        index
        for index, row in enumerate(rows)
        if isinstance(row.get("action_targets"), dict)
    ]
    if not indices:
        return {"examples": 0, "targets": {}}
    expected_outputs = len(LABELS) + len(action_target_names)
    if logits.shape[1] != expected_outputs:
        raise ValueError("action metric logits differ from the target contract")
    truth = np.array(
        [
            [int(bool(rows[index]["action_targets"][name])) for name in action_target_names]
            for index in indices
        ],
        dtype=np.int64,
    )
    probabilities = 1.0 / (
        1.0 + np.exp(-logits[indices, len(LABELS) :].astype(np.float64))
    )
    predicted_at_0_5 = probabilities >= 0.5
    if calibrated_thresholds is not None:
        if set(calibrated_thresholds) != set(action_target_names):
            raise ValueError("calibrated action thresholds differ from target names")
        threshold_values = np.array(
            [calibrated_thresholds[name] for name in action_target_names], dtype=np.float64
        )
        if np.any((threshold_values < 0.0) | (threshold_values > 1.0)):
            raise ValueError("calibrated action threshold is outside [0, 1]")
        predicted_calibrated = probabilities >= threshold_values
    else:
        threshold_values = np.full(len(action_target_names), 0.5)
        predicted_calibrated = predicted_at_0_5
    target_reports: dict[str, Any] = {}
    f1_values_at_0_5: list[float] = []
    f1_values_calibrated: list[float] = []
    roc_auc_values: list[float] = []
    for target_index, name in enumerate(action_target_names):
        target_truth = truth[:, target_index]
        reports: dict[str, float | int] = {}
        for suffix, target_predicted in (
            ("at_0_5", predicted_at_0_5[:, target_index]),
            ("at_calibrated", predicted_calibrated[:, target_index]),
        ):
            tp = int(np.sum((target_truth == 1) & target_predicted))
            fp = int(np.sum((target_truth == 0) & target_predicted))
            fn = int(np.sum((target_truth == 1) & ~target_predicted))
            tn = int(np.sum((target_truth == 0) & ~target_predicted))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            if suffix == "at_0_5":
                f1_values_at_0_5.append(f1)
                reports.update(
                    {
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "precision_at_0_5": precision,
                        "recall_at_0_5": recall,
                        "f1_at_0_5": f1,
                    }
                )
            else:
                f1_values_calibrated.append(f1)
                reports.update(
                    {
                        "calibrated_threshold": float(threshold_values[target_index]),
                        "precision_at_calibrated": precision,
                        "recall_at_calibrated": recall,
                        "f1_at_calibrated": f1,
                    }
                )
        roc_auc = (
            float(roc_auc_score(target_truth, probabilities[:, target_index]))
            if len(np.unique(target_truth)) == 2
            else None
        )
        if roc_auc is not None:
            roc_auc_values.append(roc_auc)
        target_reports[name] = {
            "positives": int(target_truth.sum()),
            "negatives": int(len(target_truth) - target_truth.sum()),
            "roc_auc": roc_auc,
            **reports,
        }
    return {
        "examples": len(indices),
        "exact_match_at_0_5": float(np.mean(np.all(predicted_at_0_5 == truth, axis=1))),
        "macro_f1_at_0_5": float(np.mean(f1_values_at_0_5)),
        "exact_match_at_calibrated": float(
            np.mean(np.all(predicted_calibrated == truth, axis=1))
        ),
        "macro_f1_at_calibrated": float(np.mean(f1_values_calibrated)),
        "macro_roc_auc": float(np.mean(roc_auc_values)) if roc_auc_values else None,
        "targets": target_reports,
    }


def fit_action_thresholds(
    rows: list[dict[str, Any]],
    logits: np.ndarray,
    action_target_names: tuple[str, ...],
) -> dict[str, float]:
    """Fit each auxiliary threshold for F1 on a dedicated family-disjoint split."""

    indices = [
        index for index, row in enumerate(rows) if isinstance(row.get("action_targets"), dict)
    ]
    if not indices or not action_target_names:
        raise ValueError("action threshold fitting requires supervised rows and target names")
    expected_outputs = len(LABELS) + len(action_target_names)
    if logits.shape[1] != expected_outputs:
        raise ValueError("action calibration logits differ from target contract")
    truth = np.array(
        [
            [int(bool(rows[index]["action_targets"][name])) for name in action_target_names]
            for index in indices
        ],
        dtype=np.int64,
    )
    probabilities = 1.0 / (
        1.0 + np.exp(-logits[indices, len(LABELS) :].astype(np.float64))
    )
    thresholds: dict[str, float] = {}
    for target_index, name in enumerate(action_target_names):
        target_truth = truth[:, target_index]
        if len(np.unique(target_truth)) != 2:
            raise ValueError(f"action calibration target lacks both classes: {name}")
        candidates = np.unique(
            np.concatenate(
                (
                    np.array([0.0, 0.5, 1.0]),
                    probabilities[:, target_index],
                )
            )
        )
        best: tuple[float, float, float] | None = None
        best_threshold = 0.5
        for candidate in candidates:
            predicted = probabilities[:, target_index] >= candidate
            tp = int(np.sum((target_truth == 1) & predicted))
            fp = int(np.sum((target_truth == 0) & predicted))
            fn = int(np.sum((target_truth == 1) & ~predicted))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            objective = (f1, precision, float(candidate))
            if best is None or objective > best:
                best = objective
                best_threshold = float(candidate)
        thresholds[name] = best_threshold
    return thresholds


def action_state_verdict_metrics(
    rows: list[dict[str, Any]],
    logits: np.ndarray,
    temperature: float,
    scam_threshold: float,
) -> dict[str, Any]:
    probabilities = softmax(logits[:, : len(LABELS)], temperature)
    scam_probabilities = probabilities[:, LABEL_TO_ID["SCAM"]]
    states = sorted({str(row.get("contrast_state", "")) for row in rows})
    by_state: dict[str, Any] = {}
    for state in states:
        indices = [
            index for index, row in enumerate(rows) if str(row.get("contrast_state")) == state
        ]
        values = scam_probabilities[indices]
        argmax_labels = [LABELS[index] for index in probabilities[indices].argmax(axis=1)]
        by_state[state] = {
            "examples": len(indices),
            "threshold_scam": int(np.sum(values >= scam_threshold)),
            "threshold_scam_rate": float(np.mean(values >= scam_threshold)),
            "mean_scam_probability": float(np.mean(values)),
            "argmax_labels": dict(Counter(argmax_labels)),
        }

    grouped: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for index, row in enumerate(rows):
        contrast_id = str(row.get("contrast_id", "")).strip()
        state = str(row.get("contrast_state", "")).strip()
        if not contrast_id or not state:
            raise ValueError("action-state validation row lacks contrast metadata")
        if state in grouped[contrast_id]:
            raise ValueError(f"duplicate action state in contrast: {contrast_id}")
        grouped[contrast_id][state] = float(scam_probabilities[index])
    correctly_ordered = 0
    harmful_gaps: list[float] = []
    for contrast_id, scores in grouped.items():
        if set(scores) != {"routine_safe", "verified_safe", "unresolved", "harmful_scam"}:
            raise ValueError(f"incomplete action-state validation contrast: {contrast_id}")
        ordered = (
            scores["harmful_scam"] > scores["unresolved"]
            and scores["unresolved"] > scores["verified_safe"]
            and scores["harmful_scam"] > scores["routine_safe"]
        )
        correctly_ordered += int(ordered)
        harmful_gaps.append(
            scores["harmful_scam"]
            - max(scores["routine_safe"], scores["verified_safe"])
        )
    return {
        "by_state": by_state,
        "contrasts": len(grouped),
        "ordered_contrasts": correctly_ordered,
        "ordered_contrast_rate": correctly_ordered / len(grouped),
        "harmful_vs_safe_gap_mean": float(np.mean(harmful_gaps)),
        "harmful_vs_safe_gap_p05": float(np.percentile(harmful_gaps, 5)),
        "harmful_vs_safe_gap_min": float(np.min(harmful_gaps)),
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
            torch.softmax(output.logits[:, : len(LABELS)], dim=-1)
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
        "--action-targets",
        default="",
        help="Comma-separated frozen auxiliary target order; schema 20 uses all seven targets.",
    )
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--action-verdict-weight",
        type=float,
        default=1.0,
        help="Main verdict-loss multiplier for rows carrying dense action targets.",
    )
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
    if args.action_loss_weight < 0:
        parser.error("action loss weight must be nonnegative")
    if not 0.0 < args.action_verdict_weight <= 1.0:
        parser.error("action verdict weight must be in (0, 1]")
    requested_action_targets = tuple(
        value.strip() for value in args.action_targets.split(",") if value.strip()
    )
    if requested_action_targets and requested_action_targets != ACTION_TARGETS:
        parser.error("--action-targets differs from the frozen ScamGuard action-target order")
    if args.action_loss_weight and not requested_action_targets and not args.evaluate_only:
        parser.error("action loss requires --action-targets")
    if requested_action_targets and not args.action_loss_weight and not args.evaluate_only:
        parser.error("action targets require a positive --action-loss-weight")
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
    call_state_validation_path = args.data / "call_state_validation.jsonl"
    if call_state_validation_path.exists():
        row_paths["call_state_validation"] = call_state_validation_path
    for split in (
        "harper_call_validation",
        "harper_state_validation",
        "multidogo_call_validation",
        "multidogo_state_validation",
        "action_calibration",
        "ftc_pattern_validation",
    ):
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
    saved_action_targets = tuple(getattr(model.config, "scamguard_action_targets", ()))
    if requested_action_targets and saved_action_targets not in {
        (),
        requested_action_targets,
    }:
        parser.error("requested action targets differ from checkpoint metadata")
    action_target_names = requested_action_targets or saved_action_targets
    if action_target_names:
        expand_classifier_for_action_targets(model, action_target_names, args.seed)
    if args.evaluate_only and args.action_loss_weight and not action_target_names:
        parser.error("evaluate-only action loss requested for a verdict-only checkpoint")
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
            action_target_names=action_target_names,
            action_verdict_weight=args.action_verdict_weight,
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
    action_positive_weights: torch.Tensor | None = None
    action_target_counts: dict[str, dict[str, int]] | None = None
    if action_target_names:
        action_rows = [
            row for row in rows["train"] if isinstance(row.get("action_targets"), dict)
        ]
        if not action_rows:
            raise ValueError("action-target training requested but no labeled rows were found")
        positives = [
            sum(int(bool(row["action_targets"][name])) for row in action_rows)
            for name in action_target_names
        ]
        if any(value <= 0 or value >= len(action_rows) for value in positives):
            raise ValueError("each action target requires both positive and negative training rows")
        action_positive_weights = torch.tensor(
            [math.sqrt((len(action_rows) - value) / value) for value in positives],
            dtype=torch.float32,
        )
        action_target_counts = {
            name: {
                "positive": positives[index],
                "negative": len(action_rows) - positives[index],
            }
            for index, name in enumerate(action_target_names)
        }
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
        verdict_predictions = prediction.predictions[:, : len(LABELS)]
        predicted = verdict_predictions.argmax(axis=1)
        metrics = {
            "macro_f1": float(
                f1_score(prediction.label_ids, predicted, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(prediction.label_ids, predicted)),
        }
        metrics.update(
            safety_selection_metrics(verdict_predictions, prediction.label_ids, args.max_fpr)
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
        action_target_names=action_target_names,
        action_loss_weight=args.action_loss_weight,
        action_positive_weights=action_positive_weights,
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
    dev_probabilities = softmax(
        predictions["dev"][:, : len(LABELS)], temperature
    )
    dev_binary_truth, dev_scam_probabilities = binary_subset(rows["dev"], dev_probabilities)
    threshold = choose_threshold(dev_binary_truth, dev_scam_probabilities, args.max_fpr)
    action_thresholds = (
        fit_action_thresholds(
            rows["action_calibration"],
            predictions["action_calibration"],
            action_target_names,
        )
        if "action_calibration" in rows and action_target_names
        else None
    )

    calibration = {
        "model_id": args.output.name,
        "temperature": temperature,
        "scam_threshold": threshold,
        "safe_threshold": 0.20,
        "labels": list(LABELS),
        "threshold_source": "dev SAFE/SCAM only",
        "action_thresholds": action_thresholds,
        "action_threshold_source": (
            "family-disjoint action_calibration rows only" if action_thresholds else None
        ),
        "input_transform": {
            "dialogue_policy": args.dialogue_policy,
            "truncation_side": tokenizer.truncation_side,
            "max_length": args.max_length,
        },
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
            "action_target_names": list(action_target_names),
            "action_loss_weight": args.action_loss_weight,
            "action_verdict_weight": args.action_verdict_weight,
            "action_positive_weights": (
                action_positive_weights.tolist()
                if action_positive_weights is not None
                else None
            ),
            "action_target_counts": action_target_counts,
            "action_thresholds": action_thresholds,
            "action_threshold_source": (
                "family-disjoint action_calibration rows only" if action_thresholds else None
            ),
            "primary_alert_score": (
                "calibrated probability from the preserved first three verdict logits; "
                "auxiliary logits are training and diagnostic signals only"
            ),
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
        "call_state_validation",
        "harper_call_validation",
        "harper_state_validation",
        "multidogo_call_validation",
        "multidogo_state_validation",
        "action_calibration",
        "ftc_pattern_validation",
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
    if "call_state_validation" in rows:
        result["call_state_validation"]["action_target_metrics"] = action_target_metrics(
            rows["call_state_validation"],
            predictions["call_state_validation"],
            action_target_names,
        )
        result["call_state_validation"]["state_verdict_metrics"] = (
            action_state_verdict_metrics(
                rows["call_state_validation"],
                predictions["call_state_validation"],
                temperature,
                threshold,
            )
        )
    if "harper_state_validation" in rows:
        result["harper_state_validation"]["action_target_metrics"] = action_target_metrics(
            rows["harper_state_validation"],
            predictions["harper_state_validation"],
            action_target_names,
        )
        result["harper_state_validation"]["state_verdict_metrics"] = (
            action_state_verdict_metrics(
                rows["harper_state_validation"],
                predictions["harper_state_validation"],
                temperature,
                threshold,
            )
        )
    if "multidogo_state_validation" in rows:
        result["multidogo_state_validation"]["action_target_metrics"] = (
            action_target_metrics(
                rows["multidogo_state_validation"],
                predictions["multidogo_state_validation"],
                action_target_names,
                action_thresholds,
            )
        )
        result["multidogo_state_validation"]["state_verdict_metrics"] = (
            action_state_verdict_metrics(
                rows["multidogo_state_validation"],
                predictions["multidogo_state_validation"],
                temperature,
                threshold,
            )
        )
    if "action_calibration" in rows:
        result["action_calibration"]["action_target_metrics"] = action_target_metrics(
            rows["action_calibration"],
            predictions["action_calibration"],
            action_target_names,
            action_thresholds,
        )
    if "ftc_pattern_validation" in rows:
        result["ftc_pattern_validation"]["action_target_metrics"] = action_target_metrics(
            rows["ftc_pattern_validation"],
            predictions["ftc_pattern_validation"],
            action_target_names,
            action_thresholds,
        )
        result["ftc_pattern_validation"]["state_verdict_metrics"] = (
            action_state_verdict_metrics(
                rows["ftc_pattern_validation"],
                predictions["ftc_pattern_validation"],
                temperature,
                threshold,
            )
        )
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
