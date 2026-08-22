import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scamguard.metrics import file_sha256
from training.eval_encoder_external import action_target_names, metadata_slices
from training.train_encoder import (
    ACTION_TARGETS,
    EncodedDataset,
    PairPreservingSampler,
    WeightedTrainer,
    action_state_verdict_metrics,
    action_target_metrics,
    expand_classifier_for_action_targets,
    load_teacher_logits,
    masked_action_bce_loss,
    paired_validation_metrics,
    pairwise_scam_margin_loss,
    predict_in_dataset_order,
    report_slice,
    retention_kl_loss,
    safety_selection_metrics,
    source_sample_weights,
)


class StubClassifier(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=input_ids.to(torch.float32))


class StubTokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [[index + 1] for index, _text in enumerate(texts)],
            "attention_mask": [[1] for _text in texts],
        }


def test_encoded_dataset_allows_real_action_rows_to_keep_full_verdict_weight() -> None:
    targets = {name: False for name in ACTION_TARGETS}
    rows = [
        {
            "id": "real-call",
            "text": "ordinary human-authored banking call",
            "label": "SAFE",
            "action_targets": targets,
            "action_verdict_weight": 1.0,
        },
        {
            "id": "synthetic-state",
            "text": "controlled action-state variant",
            "label": "SAFE",
            "action_targets": targets,
        },
    ]

    dataset = EncodedDataset(
        rows,
        StubTokenizer(),
        max_length=256,
        action_target_names=ACTION_TARGETS,
        action_verdict_weight=0.25,
    )

    assert dataset[0]["verdict_weight"].item() == pytest.approx(1.0)
    assert dataset[1]["verdict_weight"].item() == pytest.approx(0.25)


def bare_weighted_trainer() -> WeightedTrainer:
    trainer = WeightedTrainer.__new__(WeightedTrainer)
    trainer.class_weights = torch.ones(3)
    trainer.binary_loss_weight = 1.0
    trainer.binary_positive_weight = 1.0
    trainer.retention_weight = 2.0
    trainer.retention_temperature = 2.0
    trainer.pair_loss_weight = 0.0
    trainer.pair_margin = 2.0
    trainer.action_target_names = ()
    trainer.action_loss_weight = 0.0
    trainer.action_positive_weights = None
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


def test_safety_selection_ignores_auxiliary_action_logits() -> None:
    verdict = np.array(
        [[4.0, 0.0, 0.0], [3.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 2.5]]
    )
    labels = np.array([0, 0, 2, 2])
    expanded = np.concatenate([verdict, np.full((4, len(ACTION_TARGETS)), 100.0)], axis=1)

    assert safety_selection_metrics(expanded, labels, max_fpr=0.5) == pytest.approx(
        safety_selection_metrics(verdict, labels, max_fpr=0.5)
    )


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


def test_external_action_target_names_follow_classifier_order() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_labels=5,
            id2label={
                0: "SAFE",
                1: "UNCERTAIN",
                2: "SCAM",
                3: "ACTION_requested_disclosure_or_transfer",
                4: "ACTION_independent_verification",
            },
        )
    )

    assert action_target_names(model) == [
        "requested_disclosure_or_transfer",
        "independent_verification",
    ]


def test_report_slice_includes_source_domain_false_positive_breakdown() -> None:
    rows = [
        {
            "text": "routine airline service",
            "label": "SAFE",
            "category": "NONE",
            "source": "fixture",
            "source_domain": "airline",
        },
        {
            "text": "routine airline service falsely flagged",
            "label": "SAFE",
            "category": "NONE",
            "source": "fixture",
            "source_domain": "airline",
        },
        {
            "text": "routine software service",
            "label": "SAFE",
            "category": "NONE",
            "source": "fixture",
            "source_domain": "software",
        },
    ]
    logits = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0], [5.0, 0.0, 0.0]])

    report = report_slice(rows, logits, temperature=1.0, threshold=0.5)

    assert report["by_source_domain"]["airline"]["binary_safety"]["fp"] == 1
    assert report["by_source_domain"]["software"]["binary_safety"]["fp"] == 0


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


def test_action_classifier_expansion_preserves_verdict_rows_exactly() -> None:
    model = SimpleNamespace(
        classifier=torch.nn.Linear(5, 3),
        config=SimpleNamespace(),
        num_labels=3,
    )
    original_weight = model.classifier.weight.detach().clone()
    original_bias = model.classifier.bias.detach().clone()

    expand_classifier_for_action_targets(model, ACTION_TARGETS, seed=17)

    assert model.classifier.out_features == 3 + len(ACTION_TARGETS)
    assert torch.equal(model.classifier.weight[:3], original_weight)
    assert torch.equal(model.classifier.bias[:3], original_bias)
    assert tuple(model.config.scamguard_action_targets) == ACTION_TARGETS


def test_masked_action_loss_uses_only_densely_labeled_rows() -> None:
    aligned = masked_action_bce_loss(
        torch.tensor([[8.0, -8.0], [-8.0, 8.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([1.0, 0.0]),
        torch.ones(2),
    )
    reversed_targets = masked_action_bce_loss(
        torch.tensor([[-8.0, 8.0], [8.0, -8.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([1.0, 0.0]),
        torch.ones(2),
    )

    assert aligned.item() < 0.01
    assert reversed_targets.item() > 7.0


def test_action_target_metrics_report_each_dense_head() -> None:
    rows = []
    logits = []
    for value in (False, True):
        targets = {name: value for name in ACTION_TARGETS}
        rows.append({"action_targets": targets})
        action_logits = [8.0] * len(ACTION_TARGETS) if value else [-8.0] * len(ACTION_TARGETS)
        logits.append([0.0, 0.0, 0.0] + action_logits)

    metrics = action_target_metrics(rows, np.array(logits), ACTION_TARGETS)

    assert metrics["examples"] == 2
    assert metrics["exact_match_at_0_5"] == 1.0
    assert metrics["macro_f1_at_0_5"] == 1.0
    assert metrics["macro_roc_auc"] == 1.0
    assert all(report["roc_auc"] == 1.0 for report in metrics["targets"].values())


def test_action_state_metrics_require_ordered_complete_contrasts() -> None:
    states = ("routine_safe", "verified_safe", "unresolved", "harmful_scam")
    rows = [
        {"contrast_id": "one", "contrast_state": state}
        for state in states
    ]
    scam_logits = (-4.0, -2.0, 1.0, 5.0)
    logits = np.array([[0.0, 0.0, value] for value in scam_logits])

    metrics = action_state_verdict_metrics(rows, logits, 1.0, 0.5)

    assert metrics["contrasts"] == 1
    assert metrics["ordered_contrast_rate"] == 1.0
    assert metrics["by_state"]["harmful_scam"]["threshold_scam_rate"] == 1.0
    assert metrics["by_state"]["routine_safe"]["threshold_scam_rate"] == 0.0


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


def test_pairwise_margin_accepts_multiple_complete_copies_of_family() -> None:
    repeated = pairwise_scam_margin_loss(
        torch.tensor([-2.0, 2.0, -1.8, 2.2]),
        torch.tensor([0, 2, 0, 2]),
        torch.tensor([1, 1, 1, 1]),
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
        margin=2.0,
    )

    assert repeated.item() < 0.2


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


def test_pair_sampler_can_repeat_only_complete_pair_rows() -> None:
    rows = [
        {"label": "SAFE"},
        {"label": "SAFE", "pair_id": "one"},
        {"label": "SCAM", "pair_id": "one"},
    ]
    sampler = PairPreservingSampler(rows, batch_size=4, seed=7, pair_repeats=3)

    order = list(sampler)
    batches = [order[index : index + 4] for index in range(0, len(order), 4)]

    assert len(order) == 7
    assert order.count(0) == 1
    assert order.count(1) == 3
    assert order.count(2) == 3
    assert all(batch.count(1) == batch.count(2) for batch in batches)


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


def test_weighted_trainer_accepts_expanded_action_head_outputs() -> None:
    trainer = bare_weighted_trainer()
    trainer.retention_weight = 0.0
    trainer.action_target_names = ACTION_TARGETS
    trainer.action_loss_weight = 0.5
    trainer.action_positive_weights = torch.ones(len(ACTION_TARGETS))
    model = StubClassifier()
    model.train()
    action_truth = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    )

    loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.tensor(
                [[3.0, 0.0, -1.0, 4.0, -4.0, 4.0, -4.0, -4.0, -4.0, 4.0]]
            ),
            "labels": torch.tensor([0]),
            "action_targets": action_truth,
            "action_mask": torch.tensor([1.0]),
            "verdict_weight": torch.tensor([0.25]),
        },
    )

    assert torch.isfinite(loss)


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
