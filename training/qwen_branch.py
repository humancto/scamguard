"""Shared verdict-branch data and loss utilities for Qwen specialist training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from scamguard.qwen_scoring import candidate_token_sequences

LABELS = ("SAFE", "UNCERTAIN", "SCAM")


def row_verdict(row: dict[str, Any]) -> str:
    try:
        value = json.loads(row["messages"][-1]["content"])["verdict"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"row {row.get('id')} has an invalid assistant verdict") from error
    if value not in LABELS:
        raise ValueError(f"row {row.get('id')} has unsupported verdict {value!r}")
    return str(value)


def load_teacher_cache(path: Path) -> dict[tuple[str, str], list[float]]:
    cache: dict[tuple[str, str], list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["split"]), str(record["id"]))
            logits = record.get("teacher_logits")
            if key in cache:
                raise ValueError(f"duplicate teacher-cache identity: {key}")
            if (
                not isinstance(logits, list)
                or len(logits) != len(LABELS)
                or any(not isinstance(value, (int, float)) for value in logits)
            ):
                raise ValueError(f"invalid teacher logits for {key}")
            cache[key] = [float(value) for value in logits]
    return cache


class BranchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        path: Path,
        processor: Any,
        max_length: int,
        *,
        split: str,
        teacher_cache: dict[tuple[str, str], list[float]] | None = None,
    ) -> None:
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.teacher_cache = teacher_cache
        identities = [str(row.get("id", "")) for row in self.rows]
        if not all(identities) or len(identities) != len(set(identities)):
            raise ValueError(f"{split} IDs must be non-empty and unique")
        if teacher_cache is not None:
            expected = {(split, identifier) for identifier in identities}
            available = {key for key in teacher_cache if key[0] == split}
            missing = expected - available
            extra = available - expected
            if missing or extra:
                raise ValueError(
                    f"teacher cache differs from {split} data: "
                    f"missing={len(missing)} extra={len(extra)}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        identifier = str(row["id"])
        messages = row["messages"]
        prompt = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt += '{"verdict":"'
        candidates, common_prefix = candidate_token_sequences(
            self.processor.tokenizer, prompt, LABELS
        )
        input_ids = candidates[0][:common_prefix]
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"branch prompt {identifier} has {len(input_ids)} tokens above {self.max_length}"
            )
        item: dict[str, Any] = {
            "sample_id": identifier,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "branch_token_ids": torch.tensor(
                [candidate[common_prefix] for candidate in candidates], dtype=torch.long
            ),
            "target": torch.tensor(LABELS.index(row_verdict(row)), dtype=torch.long),
        }
        if self.teacher_cache is not None:
            item["teacher_logits"] = torch.tensor(
                self.teacher_cache[(self.split, identifier)], dtype=torch.float32
            )
        return item


class BranchCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        maximum = max(len(feature["input_ids"]) for feature in features)

        def left_padded(key: str, value: int) -> torch.Tensor:
            rows = []
            for feature in features:
                tensor = feature[key]
                padding = torch.full((maximum - len(tensor),), value, dtype=tensor.dtype)
                rows.append(torch.cat((padding, tensor)))
            return torch.stack(rows)

        batch: dict[str, Any] = {
            "sample_ids": [str(feature["sample_id"]) for feature in features],
            "input_ids": left_padded("input_ids", self.pad_token_id),
            "attention_mask": left_padded("attention_mask", 0),
            "branch_token_ids": torch.stack(
                [feature["branch_token_ids"] for feature in features]
            ),
            "targets": torch.stack([feature["target"] for feature in features]),
        }
        if "teacher_logits" in features[0]:
            if not all("teacher_logits" in feature for feature in features):
                raise ValueError("teacher logits must be present for every feature in a batch")
            batch["teacher_logits"] = torch.stack(
                [feature["teacher_logits"] for feature in features]
            )
        return batch


def branch_focal_kl_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    focal_gamma: float,
    kl_weight: float,
    kl_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher branch logits must have identical shapes")
    if student_logits.ndim != 2 or student_logits.shape[1] != len(LABELS):
        raise ValueError("branch logits must have shape [batch, 3]")
    if class_weights.shape != (len(LABELS),):
        raise ValueError("class_weights must have shape [3]")
    if focal_gamma < 0 or kl_weight < 0 or kl_temperature <= 0:
        raise ValueError("invalid focal/KL hyperparameters")

    student = student_logits.float()
    teacher = teacher_logits.float()
    cross_entropy = F.cross_entropy(student, targets, reduction="none")
    probabilities = torch.softmax(student, dim=-1)
    target_probability = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    alpha = class_weights.to(student.device)[targets]
    focal = (alpha * (1.0 - target_probability).pow(focal_gamma) * cross_entropy).mean()
    temperature = float(kl_temperature)
    retention = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    loss = focal + float(kl_weight) * retention
    return loss, {"focal": focal.detach(), "retention_kl": retention.detach()}
