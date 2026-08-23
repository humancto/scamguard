from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scamguard.metrics import file_sha256
from scripts.preflight_qwen08_batch import expanded_batch


def source_example() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
    }


def test_expanded_batch_exercises_full_attention_geometry() -> None:
    batch, supervised = expanded_batch(source_example(), batch_size=3, sequence_length=10)

    assert batch["input_ids"].shape == (3, 10)
    assert batch["attention_mask"].shape == (3, 10)
    assert batch["attention_mask"].sum().item() == 30
    assert (batch["labels"] != -100).sum().item() == 6
    assert supervised == 6


def test_expanded_batch_rejects_invalid_or_unsupervised_sources() -> None:
    with pytest.raises(ValueError, match="positive"):
        expanded_batch(source_example(), batch_size=0, sequence_length=10)
    unsupervised = source_example()
    unsupervised["labels"][:] = -100
    with pytest.raises(ValueError, match="unsupervised"):
        expanded_batch(unsupervised, batch_size=1, sequence_length=10)


def test_tracked_batch16_result_is_bound_and_exposes_memory_overage() -> None:
    repository = Path(__file__).resolve().parents[1]
    report = json.loads(
        (repository / "reports" / "QWEN08_BATCH16X640_PREFLIGHT.json").read_text(
            encoding="utf-8"
        )
    )
    environment = report["environment"]
    bindings = report["source_bindings"]

    assert report["passed"] is True
    assert report["parameter_update_performed"] is False
    assert report["contains_training_or_audit_rows"] is False
    assert report["geometry"]["microbatch_size"] == 16
    assert report["geometry"]["sequence_length"] == 640
    assert environment["mps_driver_allocated_bytes"] > environment["mps_recommended_max_bytes"]
    assert bindings == {
        "batch_preflight_sha256": file_sha256(
            repository / "scripts" / "preflight_qwen08_batch.py"
        ),
        "base_preflight_sha256": file_sha256(
            repository / "scripts" / "preflight_qwen08_training.py"
        ),
        "training_launcher_sha256": file_sha256(
            repository / "training" / "train_qwen_lora.py"
        ),
        "experiment_freezer_sha256": file_sha256(
            repository / "scripts" / "freeze_qwen08_full_experiment.py"
        ),
        "uv_lock_sha256": file_sha256(repository / "uv.lock"),
    }
