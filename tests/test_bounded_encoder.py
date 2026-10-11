"""CPU-only contract checks; never load a model or any real evaluation rows."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training import run_bounded_encoder as bounded


@pytest.fixture
def config(tmp_path):
    value = {
        "experiment_id": "test-fresh-ettin",
        "model": {
            "repository": bounded.MODEL_REPOSITORY,
            "revision": bounded.MODEL_REVISION,
            "license": "MIT",
            "files_sha256": dict.fromkeys(bounded.BASE_FILES, "a" * 64),
        },
        "data": {},
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "max_steps": 3200,
            "max_wall_seconds": 3600,
            "max_length": 256,
            "truncation_side": "right",
            "dialogue_policy": "speaker-neutral-evidence-recent-v2",
            "seed": 20261010,
            "binary_loss_weight": 0.0,
            "hyperparameter_runs": 1,
            "resume_or_continuation_allowed": False,
        },
        "outputs": {
            "checkpoint": str(tmp_path / "checkpoint"),
            "report": str(tmp_path / "train.json"),
            "evaluation": str(tmp_path / "open.json"),
            "predictions": str(tmp_path / "open.predictions.jsonl"),
        },
        # The fit must not open or otherwise require evaluator files.
        "evaluation": {"splits": {"not_read": {"path": str(tmp_path / "absent.jsonl")}}},
    }
    for split in ("train", "dev"):
        path = tmp_path / f"{split}.jsonl"
        rows = [
            {
                "id": f"{split}-{label}",
                "label": label,
                "text": "synthetic fixture",
                "category": "fixture",
                "source": "fixture",
            }
            for label in bounded.LABELS
        ]
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        path.write_bytes(payload)
        value["data"][split] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(rows),
        }
    return value


def test_frozen_real_contract_accepts_metadata_without_opening_data():
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root / "configs/ettin150m-bounded-v1.json").read_text())
    bounded.validate_config(value)
    assert bounded.step_bound(value["data"]["train"]["rows"], value["training"]) == 3056


def test_bounded_steps_and_config(config):
    bounded.validate_config(config)
    assert bounded.step_bound(24436, config["training"]) == 3056
    assert bounded.step_bound(100000, config["training"]) == 3200
    assert bounded.step_bound(17, config["training"]) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 3),
        ("max_steps", 3201),
        ("max_wall_seconds", 3601),
        ("batch_size", True),
        ("learning_rate", float("nan")),
        ("binary_loss_weight", 0.1),
        ("hyperparameter_runs", 2),
        ("resume_or_continuation_allowed", True),
        ("resume_from_checkpoint", "old-rejected-adapter"),
    ],
)
def test_rejects_changed_fit_scope(config, field, value):
    config["training"][field] = value
    with pytest.raises(ValueError):
        bounded.validate_config(config)


def test_rejects_unpinned_or_old_model(config):
    config["model"]["revision"] = "main"
    with pytest.raises(ValueError, match="pinned fresh"):
        bounded.validate_config(config)
    config["model"]["revision"] = bounded.MODEL_REVISION
    del config["model"]["files_sha256"]["pytorch_model.bin"]
    with pytest.raises(ValueError, match="file hashes"):
        bounded.validate_config(config)


def test_rejects_extra_training_split_without_reading_it(config):
    config["data"]["test"] = {"path": "/not-readable/sealed.jsonl"}
    with pytest.raises(ValueError, match="only train and dev"):
        bounded.validate_config(config)


def test_output_paths_do_not_overlap_open_evaluation(config):
    config["outputs"]["predictions"] = config["outputs"]["report"]
    with pytest.raises(ValueError, match="distinct"):
        bounded.validate_config(config)


def test_fresh_guard_allows_preflight_then_one_fit(config):
    paths = bounded.artifact_paths(config)
    bounded.ensure_fresh_outputs(paths, preflight=True)
    paths["preflight"].write_text("{}")
    bounded.ensure_fresh_outputs(paths, preflight=False)
    with pytest.raises(FileExistsError):
        bounded.ensure_fresh_outputs(paths, preflight=True)
    paths["progress"].write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to resume"):
        bounded.ensure_fresh_outputs(paths, preflight=False)


def test_source_hashes_counts_and_only_train_dev(config):
    rows, metadata = bounded.load_rows(config)
    assert set(rows) == {"train", "dev"}
    assert metadata["train"]["rows"] == 3
    assert metadata["dev"]["labels"] == dict.fromkeys(bounded.LABELS, 1)
    config["data"]["train"]["rows"] += 1
    with pytest.raises(ValueError, match="frozen count"):
        bounded.load_rows(config)
    config["data"]["train"]["rows"] -= 1
    config["data"]["train"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="frozen hash"):
        bounded.load_rows(config)


def test_train_dev_overlap_rejected(config):
    config["data"]["dev"] = copy.deepcopy(config["data"]["train"])
    with pytest.raises(ValueError, match="IDs overlap"):
        bounded.load_rows(config)


def test_auxiliary_projection_preserves_original():
    original = [
        {
            "id": "fixture",
            "label": "SAFE",
            "text": "fixture",
            "action_targets": {"payment": True},
            "action_verdict_weight": 0.3,
        }
    ]
    before = copy.deepcopy(original)
    projected = bounded.verdict_rows(original)
    assert original == before
    assert projected[0] is not original[0]
    assert "action_targets" not in projected[0]
    assert "action_verdict_weight" not in projected[0]
    assert projected[0]["source"] == "UNKNOWN"


def test_base_file_hashes_enforced(tmp_path):
    model = tmp_path / "pytorch_model.bin"
    model.write_bytes(b"synthetic weights; not a real model")
    expected = {model.name: hashlib.sha256(model.read_bytes()).hexdigest()}
    assert bounded.verify_model_files(tmp_path, expected) == expected
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="cached base"):
        bounded.verify_model_files(tmp_path, expected)


def test_wall_bound_rejects_even_if_final_step_finishes_late(tmp_path):
    callback = bounded.BoundProgress(tmp_path / "progress.json", 5, 60, 10, clock=lambda: 71)
    control = SimpleNamespace(should_training_stop=False)
    state = SimpleNamespace(global_step=5)
    callback.on_step_end(None, state, control)
    assert control.should_training_stop
    assert callback.stop_reason == "wall_limit_reached"
    progress = json.loads(callback.path.read_text())
    assert progress["wall_seconds"] == 61
    assert progress["release_authorized"] is False


def test_step_bound_and_early_stop_distinguished(tmp_path):
    callback = bounded.BoundProgress(tmp_path / "progress.json", 5, 60, 10, clock=lambda: 11)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_step_end(None, SimpleNamespace(global_step=5), control)
    assert control.should_training_stop
    assert callback.stop_reason == "step_bound_reached"
    early = bounded.BoundProgress(tmp_path / "early.json", 5, 60, 10, clock=lambda: 11)
    early.on_train_end(None, SimpleNamespace(global_step=3), control)
    assert early.stop_reason == "stopped_early"


def test_dev_calibration_reports_true_three_class_and_text_free_ledger(monkeypatch):
    rows = [
        {
            "id": label,
            "label": label,
            "text": "DO NOT RETAIN THIS TEXT",
            "source": "fixture",
            "category": "fixture",
        }
        for label in bounded.LABELS
    ]
    monkeypatch.setattr(bounded, "fit_temperature", lambda *_: 1.0)
    monkeypatch.setattr(bounded, "choose_threshold", lambda *_, **__: 0.5)
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    calibration, metrics, records = bounded.calibrate_dev(rows, np.log(probabilities))
    assert calibration["safe_threshold"] == 0.2
    assert metrics["macro_f1_argmax"] == 1
    assert metrics["calibrated"]["macro_f1"] == pytest.approx(5 / 9)
    assert metrics["binary_safety"]["scam_recall"] == 1
    assert metrics["scam_by_category"]["fixture"]["recall"] == 1
    assert [row["calibrated_verdict"] for row in records] == ["UNCERTAIN", "UNCERTAIN", "SCAM"]
    assert all("text" not in row for row in records)
    assert all("probabilities" in row and "logits" in row for row in records)
    assert "DO NOT RETAIN THIS TEXT" not in json.dumps(records)


def test_dev_calibration_scam_first(monkeypatch):
    rows = [
        {"id": label, "label": label, "source": "fixture", "category": "fixture"}
        for label in bounded.LABELS
    ]
    monkeypatch.setattr(bounded, "fit_temperature", lambda *_: 1.0)
    monkeypatch.setattr(bounded, "choose_threshold", lambda *_, **__: 0.15)
    probabilities = np.array([[0.81, 0.01, 0.18], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    _, _, records = bounded.calibrate_dev(rows, np.log(probabilities))
    assert records[0]["calibrated_verdict"] == "SCAM"


@pytest.mark.parametrize("logits", [np.zeros((3, 4)), np.full((3, 3), np.nan)])
def test_calibration_rejects_invalid_logits(logits):
    with pytest.raises(ValueError, match="finite and exactly three-class"):
        bounded.calibrate_dev([{}] * 3, logits)
