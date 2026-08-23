from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

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
    assert bindings["source_commit"] == "9e02f53fe85b84ef816e8947f67216a3604dd438"
    assert bindings["batch_preflight_sha256"] == (
        "fda6543a358ceeaa63e33ea12487e5839c109bfecb9b3ac5636a5b7ea2db3710"
    )
    assert bindings["base_preflight_sha256"] == (
        "a554a294c1de366b5cc97fbf781356afc488d8f36a6f480c4ffdadfaa166873b"
    )
    assert bindings["training_launcher_sha256"] == (
        "06e4a51ef7c682b4a35a0c5247e0360fe6f8147927cc7b737ab6c96347423663"
    )
    assert bindings["experiment_freezer_sha256"] == (
        "6d3545786f7136ac63a110b421be8145d798f8b9f2646e2f40269981461e17a4"
    )
    assert bindings["uv_lock_sha256"] == (
        "e6ec0814c4af6614ba98639d98421f2df523e58e3e2032aa068b7d1e910c4765"
    )
