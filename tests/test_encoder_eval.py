import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scamguard.metrics import file_sha256
from training.eval_encoder_external import metadata_slices
from training.train_encoder import (
    PairPreservingSampler,
    WeightedTrainer,
    load_teacher_logits,
    paired_validation_metrics,
    pairwise_scam_margin_loss,
    predict_in_dataset_order,
    retention_kl_loss,
    safety_selection_metrics,
    source_sample_weights,
)


class StubClassifier(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=input_ids.to(torch.float32))


def bare_weighted_trainer() -> WeightedTrainer:
    trainer = WeightedTrainer.__new__(WeightedTrainer)
    trainer.class_weights = torch.ones(3)
    trainer.binary_loss_weight = 1.0
    trainer.binary_positive_weight = 1.0
    trainer.retention_weight = 2.0
    trainer.retention_temperature = 2.0
    trainer.pair_loss_weight = 0.0
    trainer.pair_margin = 2.0
    return trainer


class StubTrainer:
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels

    def predict(self, dataset: object) -> SimpleNamespace:
        del dataset
        logits = np.eye(3, dtype=np.float32)[self.labels]
        return SimpleNamespace(predictions=logits, label_ids=np.array(self.labels))


def test_predict_in_dataset_order_accepts_sequential_output() -> None:
    rows = [{"label": "SAFE"}, {"label": "SCAM"}, {"label": "UNCERTAIN"}]

    logits = predict_in_dataset_order(StubTrainer([0, 2, 1]), object(), rows)

    assert logits.argmax(axis=1).tolist() == [0, 2, 1]


def test_predict_in_dataset_order_rejects_silent_sampler_permutation() -> None:
    rows = [{"label": "SAFE"}, {"label": "SCAM"}, {"label": "UNCERTAIN"}]

    with pytest.raises(RuntimeError, match="sampler changed row order"):
        predict_in_dataset_order(StubTrainer([2, 0, 1]), object(), rows)


def test_safety_selection_metrics_optimizes_recall_under_fpr_cap() -> None:
    # Two false-positive candidates are below the two SCAM scores. At a 25% cap, the selected
    # threshold admits one of four SAFE examples and both SCAM examples.
    logits = np.array(
        [
            [4.0, 0.0, 0.0],
            [3.0, 0.0, 1.0],
            [2.0, 0.0, 2.2],
            [2.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 2.5],
        ]
    )
    labels = np.array([0, 0, 0, 0, 2, 2])

    metrics = safety_selection_metrics(logits, labels, max_fpr=0.25)

    assert metrics["safety_recall_at_fpr"] == 1.0
    assert metrics["safety_fpr"] <= 0.25
    assert 0.0 < metrics["safety_threshold"] < 1.0


def test_safety_selection_metrics_requires_both_binary_classes() -> None:
    with pytest.raises(ValueError, match="both SAFE and SCAM"):
        safety_selection_metrics(np.zeros((2, 3)), np.array([0, 0]), max_fpr=0.02)


def test_external_metadata_slices_are_text_free() -> None:
    rows = [
        {
            "text": "private fixture phrase alpha",
            "label": "SAFE",
            "category": "NONE",
            "source": "fixture",
            "source_language": "English",
            "source_accent": "accent-a",
            "source_domain": "banking",
            "source_window": "early",
        },
        {
            "text": "private fixture phrase beta",
            "label": "SAFE",
            "category": "NONE",
            "source": "fixture",
            "source_language": "English",
            "source_accent": "accent-b",
            "source_domain": "delivery",
            "source_window": "recent",
        },
    ]
    logits = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

    slices = metadata_slices(rows, logits, temperature=1.0, scam_threshold=0.5)

    assert set(slices) == {"source_accent", "source_domain", "source_window"}
    assert slices["source_window"]["early"]["binary_safety"]["fp"] == 0
    assert slices["source_window"]["recent"]["binary_safety"]["fp"] == 1
    serialized = json.dumps(slices)
    assert "private fixture phrase" not in serialized
    assert "by_language" not in serialized


def test_source_balance_uses_sqrt_source_mass_at_alpha_half() -> None:
    rows = [
        {"source": "large"},
        {"source": "large"},
        {"source": "large"},
        {"source": "large"},
        {"source": "small"},
    ]

    weights, probability = source_sample_weights(rows, alpha=0.5)

    assert weights == [0.5, 0.5, 0.5, 0.5, 1.0]
    assert probability == pytest.approx({"large": 2 / 3, "small": 1 / 3})


def test_retention_kl_uses_only_anchored_rows() -> None:
    teacher = torch.tensor([[3.0, 0.0, -1.0], [0.0, 0.0, 3.0]])
    student = teacher.clone()
    student[1] = torch.tensor([3.0, 0.0, -1.0])

    anchored_first = retention_kl_loss(
        student,
        teacher,
        torch.tensor([1.0, 0.0]),
        temperature=2.0,
    )
    anchored_both = retention_kl_loss(
        student,
        teacher,
        torch.tensor([1.0, 1.0]),
        temperature=2.0,
    )

    assert anchored_first.item() == pytest.approx(0.0, abs=1e-6)
    assert anchored_both.item() > 0.1


def test_pairwise_margin_rewards_matched_scam_ranking() -> None:
    labels = torch.tensor([0, 2])
    groups = torch.tensor([1, 1])
    mask = torch.tensor([1.0, 1.0])

    ordered = pairwise_scam_margin_loss(
        torch.tensor([-2.0, 2.0]), labels, groups, mask, margin=2.0
    )
    reversed_pair = pairwise_scam_margin_loss(
        torch.tensor([2.0, -2.0]), labels, groups, mask, margin=2.0
    )

    assert ordered.item() < 0.2
    assert reversed_pair.item() > 5.0


def test_pairwise_margin_rejects_incomplete_pair() -> None:
    with pytest.raises(ValueError, match="incomplete pair"):
        pairwise_scam_margin_loss(
            torch.tensor([1.0]),
            torch.tensor([2]),
            torch.tensor([1]),
            torch.tensor([1.0]),
            margin=2.0,
        )


def test_pair_sampler_keeps_pairs_in_same_batch_and_covers_every_row() -> None:
    rows = [{"label": "SAFE"} for _ in range(5)]
    rows.extend(
        [
            {"label": "SAFE", "pair_id": "one"},
            {"label": "SCAM", "pair_id": "one"},
            {"label": "SAFE", "pair_id": "two"},
            {"label": "SCAM", "pair_id": "two"},
        ]
    )
    sampler = PairPreservingSampler(rows, batch_size=4, seed=7)

    order = list(sampler)
    batches = [order[index : index + 4] for index in range(0, len(order), 4)]

    assert sorted(order) == list(range(len(rows)))
    for pair_id in ("one", "two"):
        pair_indices = {index for index, row in enumerate(rows) if row.get("pair_id") == pair_id}
        assert any(pair_indices <= set(batch) for batch in batches)


def test_paired_validation_reports_probability_order() -> None:
    rows = [
        {"label": "SAFE", "pair_id": "one"},
        {"label": "SCAM", "pair_id": "one"},
        {"label": "SAFE", "pair_id": "two"},
        {"label": "SCAM", "pair_id": "two"},
    ]
    logits = np.array(
        [[3.0, 0.0, -1.0], [0.0, 0.0, 3.0], [2.0, 0.0, 1.0], [3.0, 0.0, 0.0]]
    )

    metrics = paired_validation_metrics(rows, logits, temperature=1.0)

    assert metrics["pairs"] == 2
    assert metrics["correctly_ordered"] == 1
    assert metrics["pair_order_accuracy"] == 0.5


def test_weighted_trainer_skips_retention_for_teacher_free_evaluation() -> None:
    trainer = bare_weighted_trainer()
    model = StubClassifier()
    model.eval()

    loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.tensor([[3.0, 0.0, -1.0]]),
            "labels": torch.tensor([0]),
        },
    )

    assert torch.isfinite(loss)


def test_weighted_trainer_requires_teacher_fields_during_retention_training() -> None:
    trainer = bare_weighted_trainer()
    model = StubClassifier()
    model.train()

    with pytest.raises(ValueError, match="retention loss requires teacher logits"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[3.0, 0.0, -1.0]]),
                "labels": torch.tensor([0]),
            },
        )


def test_weighted_trainer_requires_pair_fields_during_pair_training() -> None:
    trainer = bare_weighted_trainer()
    trainer.retention_weight = 0.0
    trainer.pair_loss_weight = 1.0
    model = StubClassifier()
    model.train()

    with pytest.raises(ValueError, match="pair loss requires pair groups"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[3.0, 0.0, -1.0]]),
                "labels": torch.tensor([0]),
            },
        )


def test_teacher_logit_ledger_is_hash_and_schema_pinned(tmp_path: Path) -> None:
    ledger = tmp_path / "teacher.jsonl"
    ledger.write_text(
        json.dumps({"id": "one", "logits": [1.0, 0.0, -1.0]}) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"rows": 1, "ledger_sha256": file_sha256(ledger)}),
        encoding="utf-8",
    )

    values, loaded_manifest = load_teacher_logits(ledger, manifest)

    assert values == {"one": (1.0, 0.0, -1.0)}
    assert loaded_manifest["rows"] == 1
    ledger.write_text(ledger.read_text() + "{}\n")
    with pytest.raises(ValueError, match="differs from its manifest"):
        load_teacher_logits(ledger, manifest)
