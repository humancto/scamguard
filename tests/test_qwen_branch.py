from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from training.qwen_branch import BranchCollator, branch_focal_kl_loss, load_teacher_cache


def test_branch_focal_kl_loss_prefers_correct_logits_and_retains_teacher() -> None:
    targets = torch.tensor([0, 1, 2])
    teacher = torch.tensor([[4.0, 0.0, -1.0], [0.0, 4.0, -1.0], [-1.0, 0.0, 4.0]])
    correct, parts = branch_focal_kl_loss(
        teacher.clone(),
        targets,
        teacher,
        class_weights=torch.tensor([1.0, 3.0, 1.0]),
        focal_gamma=2.0,
        kl_weight=5.0,
        kl_temperature=1.0,
    )
    wrong, _ = branch_focal_kl_loss(
        -teacher,
        targets,
        teacher,
        class_weights=torch.tensor([1.0, 3.0, 1.0]),
        focal_gamma=2.0,
        kl_weight=5.0,
        kl_temperature=1.0,
    )

    assert correct < wrong
    assert parts["retention_kl"].item() == pytest.approx(0.0, abs=1e-7)


def test_branch_focal_kl_loss_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="shape"):
        branch_focal_kl_loss(
            torch.zeros((2, 2)),
            torch.zeros(2, dtype=torch.long),
            torch.zeros((2, 2)),
            class_weights=torch.ones(3),
            focal_gamma=2.0,
            kl_weight=1.0,
            kl_temperature=1.0,
        )


def test_branch_collator_left_pads_and_preserves_branch_metadata() -> None:
    collator = BranchCollator(pad_token_id=9)
    batch = collator(
        [
            {
                "sample_id": "short",
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "branch_token_ids": torch.tensor([3, 4, 5]),
                "target": torch.tensor(0),
                "teacher_logits": torch.tensor([1.0, 0.0, -1.0]),
            },
            {
                "sample_id": "long",
                "input_ids": torch.tensor([6, 7, 8]),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "branch_token_ids": torch.tensor([3, 4, 5]),
                "target": torch.tensor(1),
                "teacher_logits": torch.tensor([0.0, 1.0, -1.0]),
            },
        ]
    )

    assert batch["sample_ids"] == ["short", "long"]
    assert batch["input_ids"].tolist() == [[9, 1, 2], [6, 7, 8]]
    assert batch["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert batch["branch_token_ids"].shape == (2, 3)


def test_teacher_cache_is_text_free_and_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "teacher.jsonl"
    records = [
        {"split": "train", "id": "one", "teacher_logits": [1.0, 0.0, -1.0]},
        {"split": "dev", "id": "two", "teacher_logits": [0.0, 1.0, -1.0]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    assert load_teacher_cache(path)[("train", "one")] == [1.0, 0.0, -1.0]

    path.write_text(json.dumps(records[0]) + "\n" + json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_teacher_cache(path)
