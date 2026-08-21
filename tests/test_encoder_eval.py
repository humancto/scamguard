import json
from types import SimpleNamespace

import numpy as np
import pytest

from training.eval_encoder_external import metadata_slices
from training.train_encoder import predict_in_dataset_order, safety_selection_metrics


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
