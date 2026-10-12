"""CPU-only synthetic contract tests; never load weights or real corpus rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from training import run_bounded_encoder as bounded


@pytest.fixture
def serving_config(tmp_path, monkeypatch):
    snapshot = tmp_path / "base"
    snapshot.mkdir()
    hashes = {}
    for name in bounded.BASE_FILES:
        payload = f"synthetic file: {name}".encode()
        (snapshot / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()

    def offline_snapshot(**kwargs):
        assert kwargs == {
            "repo_id": bounded.MODEL_REPOSITORY,
            "revision": bounded.MODEL_REVISION,
            "allow_patterns": sorted(bounded.BASE_FILES),
            "local_files_only": True,
        }
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline_snapshot)
    config = {
        "experiment_id": "synthetic-serving-contract",
        "model": {
            "repository": bounded.MODEL_REPOSITORY,
            "revision": bounded.MODEL_REVISION,
            "files_sha256": hashes,
        },
        "data": {},
        "training": {
            "epochs": 2, "batch_size": 16, "learning_rate": 2e-5,
            "max_steps": 3200, "max_wall_seconds": 3600, "max_length": 256,
            "truncation_side": "right",
            "dialogue_policy": "speaker-neutral-action-preserving-v3",
            "seed": 20261011, "binary_loss_weight": 0,
        },
        "calibration": {"method": bounded.SERVING_CALIBRATION_VERSION},
        "outputs": {
            "checkpoint": str(tmp_path / "checkpoint"),
            "report": str(tmp_path / "train.json"),
        },
        "evaluation": {"splits": {"unread": {"path": "/does-not-exist/held.jsonl"}}},
    }
    for split in ("train", "dev"):
        rows = [
            {"id": f"{split}-{label}", "text": "original synthetic fixture", "label": label}
            for label in bounded.LABELS
        ]
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        path = tmp_path / f"{split}.jsonl"
        path.write_bytes(payload)
        config["data"][split] = {
            "path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "rows": 3,
        }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return config, path


@pytest.mark.parametrize("method", [None, "unknown", False, 1])
def test_explicit_unknown_method_rejected(serving_config, method):
    config, _ = serving_config
    config["calibration"]["method"] = method
    with pytest.raises(ValueError, match="calibration method"):
        bounded.validate_config(config)


def test_new_and_legacy_selectors_are_valid(serving_config):
    config, _ = serving_config
    bounded.validate_config(config)
    del config["calibration"]["method"]
    bounded.validate_config(config)
    del config["calibration"]
    bounded.validate_config(config)


def test_calibration_dispatch_preserves_legacy_and_routes_serving(serving_config, monkeypatch):
    config, _ = serving_config
    model, tokenizer, rows, dataset = object(), object(), [], object()
    trainer = SimpleNamespace(args=SimpleNamespace(train_sampling_strategy="group_by_length"))
    expected, logits = ({"method": "synthetic"}, {}, []), object()
    calls = []
    monkeypatch.setattr(
        bounded, "calibrate_serving_dev", lambda *args: calls.append(args) or expected
    )
    assert bounded.calibrate_checkpoint_dev(
        config, model, tokenizer, rows, trainer, dataset
    ) is expected
    assert calls == [(model, tokenizer, rows, 256, "speaker-neutral-action-preserving-v3")]
    assert trainer.args.train_sampling_strategy == "group_by_length"
    del config["calibration"]["method"]
    monkeypatch.setattr(
        bounded, "predict_in_dataset_order", lambda *args: calls.append(args) or logits
    )
    monkeypatch.setattr(bounded, "calibrate_dev", lambda *args: calls.append(args) or expected)
    assert bounded.calibrate_checkpoint_dev(
        config, model, tokenizer, rows, trainer, dataset
    ) is expected
    assert trainer.args.train_sampling_strategy == "random"
    assert calls[-2:] == [(trainer, dataset, rows), (rows, logits)]


def test_metadata_preflight_never_enters_model_runtime(serving_config, monkeypatch):
    config, path = serving_config

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata preflight entered runtime")

    monkeypatch.setattr(bounded, "environment", forbidden)
    monkeypatch.setattr(bounded.torch.backends.mps, "is_available", forbidden)
    monkeypatch.setattr(bounded.AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(bounded.AutoModelForSequenceClassification, "from_pretrained", forbidden)
    assert bounded.run(path, metadata_preflight=True) == 0
    paths = bounded.artifact_paths(config)
    assert [name for name, output in paths.items() if output.exists()] == ["preflight"]
    receipt = json.loads(paths["preflight"].read_text())
    assert receipt["effective_max_steps"] == 2
    assert receipt["config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["data"]["dev"]["sha256"] == config["data"]["dev"]["sha256"]
    assert receipt["base_model_files_sha256"] == config["model"]["files_sha256"]
    assert receipt["source_sha256"] == bounded.source_identity()
    assert receipt["model_forward_performed"] is False
    assert receipt["training_performed"] is False
    assert receipt["release_authorized"] is False
    with pytest.raises(FileExistsError):
        bounded.run(path, metadata_preflight=True)


def test_metadata_preflight_rejects_changed_local_base(serving_config):
    _, path = serving_config
    (path.parent / "base" / "config.json").write_text("changed synthetic base")
    with pytest.raises(ValueError, match="cached base"):
        bounded.run(path, metadata_preflight=True)


def test_serving_fit_requires_matching_metadata_receipt(serving_config, monkeypatch):
    config, path = serving_config
    with pytest.raises(ValueError, match="metadata preflight receipt"):
        bounded.run(path)
    assert bounded.run(path, metadata_preflight=True) == 0
    sources = bounded.source_identity()
    sources["training/ettin_calibration.py"] = "0" * 64
    monkeypatch.setattr(bounded, "source_identity", lambda: sources)
    with pytest.raises(ValueError, match="metadata preflight receipt"):
        bounded.run(path)
    assert not bounded.artifact_paths(config)["progress"].exists()


def test_preflight_modes_cannot_be_confused(serving_config):
    _, path = serving_config
    with pytest.raises(ValueError, match="only one preflight"):
        bounded.run(path, preflight=True, metadata_preflight=True)
    with pytest.raises(ValueError, match="require --metadata-preflight"):
        bounded.run(path, preflight=True)


@pytest.mark.parametrize("device,world_size", [("cpu", 1), ("mps", 2), ("mps", 1)])
def test_synthetic_fit_single_process_guard_and_dev_binding(
    serving_config, monkeypatch, device, world_size,
):
    config, path = serving_config
    assert bounded.run(path, metadata_preflight=True) == 0
    monkeypatch.setattr(bounded, "environment", lambda: {"synthetic": True})
    monkeypatch.setattr(bounded.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(bounded, "set_seed", lambda _: None)
    model = SimpleNamespace(
        device=SimpleNamespace(type=device),
        parameters=lambda: [SimpleNamespace(numel=lambda: 3)],
    )
    model.to = lambda _: model
    tokenizer = SimpleNamespace(
        save_pretrained=lambda output: (output / "tokenizer.json").write_text("synthetic")
    )
    monkeypatch.setattr(bounded.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(
        bounded.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: model
    )
    monkeypatch.setattr(bounded, "EncodedDataset", lambda *a, **k: object())
    monkeypatch.setattr(bounded, "DataCollatorWithPadding", lambda _: object())

    class Arguments(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, world_size=world_size)

        def to_dict(self):
            return vars(self)

    class Trainer:
        def __init__(self, **kwargs):
            self.model, self.args = kwargs["model"], kwargs["args"]
            self.progress = kwargs["callbacks"][0]
            self.state = SimpleNamespace(global_step=2)

        def train(self):
            self.progress.stop_reason = "step_bound_reached"
            return SimpleNamespace(metrics={"synthetic": True})

        def save_model(self, output):
            output.mkdir()
            (output / "model.safetensors").write_bytes(b"synthetic, not weights")

    monkeypatch.setattr(bounded, "TrainingArguments", Arguments)
    monkeypatch.setattr(bounded, "WeightedTrainer", Trainer)
    monkeypatch.setattr(
        bounded, "calibrate_serving_dev",
        lambda *args: (
            {}, {"binary_safety": {"scam_recall": 1, "false_positive_rate": 0},
                 "calibrated": {"macro_f1": 1}}, [],
        ),
    )
    if device != "mps" or world_size != 1:
        with pytest.raises(RuntimeError, match="one local MPS process"):
            bounded.run(path)
        assert not Path(config["outputs"]["checkpoint"]).exists()
    else:
        assert bounded.run(path) == 0
        receipt = json.loads(Path(config["outputs"]["report"]).read_text())
        assert receipt["status"] == "completed"
        assert receipt["training_completed"] is True
        assert receipt["metadata_preflight"] == {
            "path": str(bounded.artifact_paths(config)["preflight"]),
            "sha256": bounded.file_sha256(bounded.artifact_paths(config)["preflight"]),
        }
        calibration = receipt["calibration"]
        assert calibration["dev_data_sha256"] == config["data"]["dev"]["sha256"]
        assert calibration["checkpoint_model_sha256"] == receipt["checkpoint_model_sha256"]
        assert calibration["input_transform"]["dialogue_policy"] == (
            "speaker-neutral-action-preserving-v3"
        )
        assert receipt["release_authorized"] is False
