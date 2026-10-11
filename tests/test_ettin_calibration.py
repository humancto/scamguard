"""Synthetic CPU-only checks: no model weights, GPU, dataset files, or decisions are produced."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scamguard.decision import calibrated_verdict
from scamguard.metrics import choose_threshold
from training import ettin_calibration as calibration


class FakeTokenizer:
    truncation_side = "right"

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {
            "input_ids": torch.tensor([[len(self.calls) - 1]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, logits):
        self.logits = logits
        self.calls = []
        self.evaluating = False

    def eval(self):
        self.evaluating = True
        return self

    def __call__(self, **inputs):
        assert self.evaluating
        assert torch.is_inference_mode_enabled()
        assert inputs["input_ids"].device.type == "cpu"
        assert inputs["input_ids"].shape == (1, 1)
        self.calls.append(inputs)
        index = int(inputs["input_ids"][0, 0])
        return SimpleNamespace(logits=self.logits[index : index + 1])


def fixture_rows():
    return [
        {
            "id": f"fixture-{label}",
            "text": f"  synthetic {label} fixture  ",
            "label": label,
            "source": "fixture",
            "category": "fixture-category",
        }
        for label in calibration.LABELS
    ]


def test_dev_scoring_uses_stripped_batch_one_inputs_and_one_dimensional_torch_softmax(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    rows = fixture_rows()
    probabilities = np.array(
        [[0.92, 0.06, 0.02], [0.05, 0.90, 0.05], [0.10, 0.05, 0.85]], dtype=np.float32
    )
    model = FakeModel(torch.tensor(np.log(probabilities)))
    tokenizer = FakeTokenizer()
    fit_calls = []

    def fit(logits, labels):
        fit_calls.append((logits.copy(), labels.copy()))
        return 1.0

    monkeypatch.setattr(calibration, "fit_temperature", fit)
    real_softmax = torch.softmax
    softmax_calls = []

    def checked_softmax(vector, *, dim):
        softmax_calls.append((tuple(vector.shape), vector.dtype, vector.device.type, dim))
        return real_softmax(vector, dim=dim)

    monkeypatch.setattr(torch, "softmax", checked_softmax)
    config, metrics, records = calibration.calibrate_serving_dev(
        model, tokenizer, rows, 256, "none"
    )
    assert len(model.calls) == len(tokenizer.calls) == 3
    assert [text for text, _ in tokenizer.calls] == [row["text"].strip() for row in rows]
    assert all(
        kwargs == {"return_tensors": "pt", "truncation": True, "max_length": 256}
        for _, kwargs in tokenizer.calls
    )
    assert softmax_calls == [((3,), torch.float32, "cpu", -1)] * 3
    assert len(fit_calls) == 1
    np.testing.assert_array_equal(fit_calls[0][1], [0, 1, 2])
    assert fit_calls[0][0].dtype == np.float32
    assert config["calibration_version"] == calibration.SERVING_CALIBRATION_VERSION
    assert config["serving_contract"]["message_batch_size"] == 1
    assert config["serving_contract"]["padding"] is False
    assert config["safe_threshold"] == 0.2
    assert config["threshold_selection"]["mask_unchanged"] is True
    assert metrics["binary_safety"]["scam_recall"] == 1
    assert metrics["binary_safety"]["false_positive_rate"] == 0
    assert metrics["macro_f1_argmax"] == metrics["calibrated"]["macro_f1"] == 1
    assert [record["calibrated_verdict"] for record in records] == list(calibration.LABELS)
    for row, record in zip(rows, records, strict=True):
        assert record["text_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest()
        assert record["split"] == "dev"
        assert len(record["logits"]) == 3
        assert record["threshold_scam"] == (record["calibrated_verdict"] == "SCAM")
    assert "synthetic SAFE fixture" not in json.dumps(records)
    assert list(tmp_path.iterdir()) == []


def test_dialogue_preprocessing_is_applied_after_stripping(monkeypatch):
    seen = []
    monkeypatch.setattr(
        calibration,
        "prepare_model_text",
        lambda text, policy: seen.append((text, policy)) or f"prepared:{text}",
    )
    monkeypatch.setattr(calibration, "fit_temperature", lambda *_: 1.0)
    model = FakeModel(torch.tensor([[5.0, 0.0, -5.0], [0.0, 5.0, -5.0], [-5.0, 0.0, 5.0]]))
    tokenizer = FakeTokenizer()
    rows = fixture_rows()
    calibration.calibrate_serving_dev(
        model, tokenizer, rows, 256, "speaker-neutral-evidence-recent-v2"
    )
    assert seen == [(row["text"].strip(), "speaker-neutral-evidence-recent-v2") for row in rows]
    assert all(text.startswith("prepared:") for text, _ in tokenizer.calls)


def test_v1_one_ulp_boundary_uses_torch_probabilities_not_numpy():
    logits = np.array(
        [[3.8481390476226807, -5.913032054901123, -1.1912144422531128]], dtype=np.float32
    )
    temperature = 2.558446290011105
    adjusted = logits / temperature
    exponentials = np.exp(adjusted - adjusted.max(axis=1, keepdims=True))
    numpy_score = float((exponentials / exponentials.sum(axis=1, keepdims=True))[0, 2])
    scores = calibration.serving_probabilities(logits, temperature, "cpu")[0]
    assert numpy_score == 0.12009981274604797
    assert scores[2] == 0.12009980529546738
    assert scores[2] < numpy_score
    threshold, info = calibration.select_stable_threshold(
        [0, 1], [0.01, scores[2]], all_scores=[0.01, 0.07, scores[2]]
    )
    assert info["observed_threshold"] == scores[2]
    assert 0.07 < threshold < scores[2]
    assert info["threshold_location"] == "stable_interval_midpoint"
    assert (
        calibrated_verdict(
            safe_probability=scores[0],
            scam_probability=scores[2],
            scam_probability_threshold=threshold,
            safe_probability_threshold=0.8,
            safe_max_scam_probability=0.2,
        )
        == "SCAM"
    )


def test_python_float_comparison_and_narrow_interval_fallback():
    lower = np.float32(0.5)
    upper = np.nextafter(lower, np.float32(1))
    midpoint = (float(lower) + float(upper)) / 2
    assert np.float32(midpoint) == lower  # A weak NumPy scalar cast would change the SAFE decision.
    assert calibration.decision_mask(np.array([lower, upper]), midpoint) == [False, True]
    threshold, info = calibration.select_stable_threshold([0, 1], [lower, upper])
    assert threshold == float(upper)
    assert info["threshold_location"] == "exact_boundary_fallback"
    assert info["minimum_score_margin"] == 0
    assert calibration.decision_mask([lower, upper], threshold) == [False, True]


def test_midpoint_preserves_uncertain_reference_mask_too():
    threshold, info = calibration.select_stable_threshold(
        [0, 1], [0.1, 0.9], all_scores=[0.1, 0.9, 0.89, 0.95]
    )
    assert 0.89 < threshold < 0.9
    assert calibration.decision_mask([0.1, 0.9, 0.89, 0.95], threshold) == [
        False,
        True,
        False,
        True,
    ]
    assert info["mask_unchanged"] is True


def test_recall_precision_and_conservative_tie_break_match_existing_objective():
    # Two SCAMs have 1.0 recall at .7 and .5, but .7 excludes the extra SAFE at .6.
    truth, scores = [0, 0, 1, 1], [0.6, 0.1, 0.9, 0.7]
    threshold, info = calibration.select_stable_threshold(truth, scores, max_fpr=0.5)
    expected = choose_threshold(np.array(truth), np.array(scores, dtype=np.float64), max_fpr=0.5)
    assert info["observed_threshold"] == expected == 0.7
    assert info["scam_recall"] == 1 and info["fp"] == 0
    assert calibration.decision_mask(scores, threshold) == calibration.decision_mask(
        scores, expected
    )


@pytest.mark.parametrize("scores", [[0.5, 0.5], [0.0, 0.0], [0.9, 0.2]])
def test_ties_and_no_feasible_observed_threshold_can_safely_choose_no_alerts(scores):
    threshold, info = calibration.select_stable_threshold([0, 1], scores)
    assert not any(calibration.decision_mask(scores, threshold))
    assert info["observed_threshold"] == 1
    assert info["fp"] == info["tp"] == 0
    assert info["scam_recall"] == 0  # No invented passing recall for a zero-alert policy.


def test_saturated_safe_score_has_no_in_range_fpr_feasible_threshold():
    with pytest.raises(ValueError, match="no feasible probability threshold"):
        calibration.select_stable_threshold([0, 1], [1.0, 1.0])


@pytest.mark.parametrize(
    "truth,scores",
    [([], []), ([0, 0], [0.1, 0.2]), ([0, 1], [float("nan"), 0.2]), ([0, 1], [-0.1, 0.2])],
)
def test_threshold_rejects_invalid_development_populations(truth, scores):
    with pytest.raises(ValueError):
        calibration.select_stable_threshold(truth, scores)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        calibration.serving_probabilities(np.zeros((3, 3), dtype=np.float32), temperature, "cpu")


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros((3, 4)),
        torch.zeros((3, 3), dtype=torch.float64),
        torch.full((3, 3), float("nan")),
    ],
)
def test_capture_rejects_shape_dtype_or_nonfinite_logits(logits):
    with pytest.raises(ValueError, match="logits"):
        calibration.calibrate_serving_dev(
            FakeModel(logits), FakeTokenizer(), fixture_rows(), 256, "none"
        )


@pytest.mark.parametrize("change", ["duplicate", "missing_class", "empty", "too_long"])
def test_bad_rows_fail_before_inference(change):
    rows = fixture_rows()
    if change == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif change == "missing_class":
        rows[1]["label"] = "SAFE"
    elif change == "empty":
        rows[0]["text"] = "  "
    else:
        rows[0]["text"] = "x" * 100001
    model = FakeModel(torch.zeros((3, 3)))
    with pytest.raises(ValueError):
        calibration.calibrate_serving_dev(model, FakeTokenizer(), rows, 256, "none")
    assert not model.calls
