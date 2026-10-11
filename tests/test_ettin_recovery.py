"""CPU-only replay/lineage guards; no model, GPU, or evaluation-data access."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training import run_ettin_recovery as recovery


@pytest.fixture
def rows():
    return [
        {
            "id": "identity",
            "label": "SCAM",
            "category": "IDENTITY_IMPERSONATION",
            "source": "fixture",
        },
        {"id": "other-scam", "label": "SCAM", "category": "FINANCIAL", "source": "fixture"},
        {
            "id": "multi-safe",
            "label": "SAFE",
            "category": "NONE",
            "source": "multidogo_human_service_dialogues",
        },
        {
            "id": "phone-safe",
            "label": "SAFE",
            "category": "NONE",
            "source": "shakeleoatmeal_phone_scam_synthetic",
        },
        {
            "id": "taskmaster",
            "label": "SAFE",
            "category": "NONE",
            "source": "taskmaster1_woz_dialogues",
        },
        {"id": "uncertain", "label": "UNCERTAIN", "category": "NONE", "source": "fixture"},
    ]


@pytest.fixture
def config(tmp_path):
    return {
        "experiment_id": "sg-ettin-recovery-test",
        "model": {
            "repository": recovery.bounded.MODEL_REPOSITORY,
            "revision": recovery.bounded.MODEL_REVISION,
            "files_sha256": dict.fromkeys(recovery.bounded.BASE_FILES, "a" * 64),
        },
        "initialization": {
            "checkpoint": str(tmp_path / "parent"),
            "model_sha256": recovery.PARENT_MODEL_SHA256,
        },
        "data": {
            "train": {
                "path": str(tmp_path / "train.jsonl"),
                "sha256": recovery.TRAIN_SHA256,
                "rows": 24436,
            },
            "dev": {
                "path": str(tmp_path / "dev.jsonl"),
                "sha256": recovery.DEV_SHA256,
                "rows": 2634,
            },
        },
        "training": {
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 5e-6,
            "max_steps": 1853,
            "max_wall_seconds": 2700,
            "max_length": 256,
            "truncation_side": "right",
            "dialogue_policy": "speaker-neutral-evidence-recent-v2",
            "seed": 20261011,
            "binary_loss_weight": 0.0,
            "extra_target_repeats": 2,
            "expected_presentations": 29648,
            "expected_target_counts": dict(recovery.TARGET_COUNTS),
        },
        "outputs": {
            "checkpoint": str(tmp_path / "recovery"),
            "report": str(tmp_path / "train.json"),
            "replay_manifest": str(tmp_path / "replay.json"),
        },
    }


def test_contract_accepts_exact_config_without_reading_files(config):
    recovery.validate_config(config)


def test_root_recovery_config_is_accepted():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/ettin150m-recovery-v2.json").read_text())
    recovery.validate_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_steps", 1854),
        ("max_wall_seconds", 2701),
        ("epochs", 2),
        ("batch_size", 32),
        ("learning_rate", 2e-5),
        ("seed", 123),
        ("extra_target_repeats", 3),
        ("expected_presentations", 30078),
        ("binary_loss_weight", 0.5),
        ("retention_weight", 1),
        ("source_balance_alpha", 0.5),
        ("pair_loss_weight", 1),
        ("action_loss_weight", 1),
        ("resume_from_checkpoint", "parent-optimizer"),
        ("hyperparameter_runs", 2),
    ],
)
def test_changed_contract_is_rejected(config, key, value):
    config["training"][key] = value
    with pytest.raises(ValueError):
        recovery.validate_config(config)


def test_parent_weights_and_data_identity_are_pinned(config):
    config["initialization"]["model_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Ettin v1"):
        recovery.validate_config(config)
    config["initialization"]["model_sha256"] = recovery.PARENT_MODEL_SHA256
    config["data"]["train"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="unchanged schema25"):
        recovery.validate_config(config)


def test_no_eval_input_or_parent_overwrite(config):
    config["data"]["test"] = {"path": "never-open-this.jsonl"}
    with pytest.raises(ValueError, match="only train and dev"):
        recovery.validate_config(config)
    del config["data"]["test"]
    config["outputs"]["checkpoint"] = config["initialization"]["checkpoint"]
    with pytest.raises(ValueError, match="overwrite"):
        recovery.validate_config(config)


def test_every_original_once_and_targets_three_times(rows):
    original = json.dumps(rows, sort_keys=True)
    indices, manifest = recovery.build_replay(rows, seed=20261011)
    assert Counter(indices) == Counter({0: 3, 1: 1, 2: 3, 3: 3, 4: 1, 5: 1})
    assert len(indices) == 12
    assert manifest["replacement_sampling"] is False
    assert manifest["source_texts_or_labels_modified"] is False
    assert manifest["groups"]["identity_scam"]["unique_rows"] == 1
    assert manifest["groups"]["service_safe"]["unique_rows"] == 2
    assert manifest["groups"]["ordinary"]["unique_rows"] == 3
    assert original == json.dumps(rows, sort_keys=True)


def test_deterministic_order_and_hashes_bind_order(rows):
    indices, first = recovery.build_replay(rows, 20261011)
    assert recovery.build_replay(rows, 20261011) == (indices, first)
    different_indices, different = recovery.build_replay(rows, 20261012)
    assert different_indices != indices
    assert different["index_order_sha256"] != first["index_order_sha256"]
    assert different["row_id_order_sha256"] != first["row_id_order_sha256"]
    assert different["groups"] == first["groups"]


def test_new_id_changes_replay_identity(rows):
    _, original = recovery.build_replay(rows, 1)
    rows[0]["id"] = "changed"
    _, changed = recovery.build_replay(rows, 1)
    assert original["index_order_sha256"] == changed["index_order_sha256"]
    assert original["row_id_order_sha256"] != changed["row_id_order_sha256"]


def test_duplicate_ids_or_different_repeat_budget_rejected(rows):
    with pytest.raises(ValueError, match="two extra"):
        recovery.build_replay(rows, 1, extra_target_repeats=1)
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="unique"):
        recovery.build_replay(rows, 1)


def test_explicit_sampler_retains_exact_order_and_finite_length(rows):
    indices, _ = recovery.build_replay(rows, 1)
    sampler = recovery.FrozenReplaySampler(indices, len(rows))
    assert list(sampler) == indices
    assert list(sampler) == indices
    assert len(sampler) == len(indices)
    trainer = object.__new__(recovery.ReplayTrainer)
    trainer.train_dataset = rows
    trainer.replay_sampler = sampler
    assert trainer._get_train_sampler() is sampler
    with pytest.raises(ValueError, match="size changed"):
        trainer._get_train_sampler(rows[:-1])


@pytest.mark.parametrize("indices", [[], [-1], [6], [True], [1.0]])
def test_invalid_sampler_indices_rejected(indices):
    with pytest.raises(ValueError, match="invalid frozen"):
        recovery.FrozenReplaySampler(indices, dataset_size=6)


def test_frozen_counts_yield_exactly_1853_complete_batches():
    rows = []
    for index in range(24436):
        if index < 1206:
            label, category, source = "SCAM", "IDENTITY_IMPERSONATION", "fixture"
        elif index < 2606:
            label, category, source = "SAFE", "NONE", "multidogo_human_service_dialogues"
        else:
            label, category, source = "UNCERTAIN", "NONE", "fixture"
        rows.append({"id": str(index), "label": label, "category": category, "source": source})
    indices, manifest = recovery.build_replay(rows, seed=20261011)
    assert len(indices) == 29648
    assert math.ceil(len(indices) / 16) == 1853
    dataset = TensorDataset(torch.arange(len(rows)))
    loader = DataLoader(
        dataset, batch_size=16, sampler=recovery.FrozenReplaySampler(indices, len(rows))
    )
    batches = list(loader)
    assert len(batches) == 1853
    assert all(len(batch[0]) == 16 for batch in batches)
    assert torch.cat([batch[0] for batch in batches]).tolist() == indices
    assert manifest["groups"]["identity_scam"]["presentations"] == 3618
    assert manifest["groups"]["service_safe"]["presentations"] == 4200


def test_actual_trainer_dataloader_preserves_all_1853_cpu_batches(tmp_path):
    class IndexedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 24436

        def __getitem__(self, index):
            return {"input_ids": torch.tensor([index]), "labels": torch.tensor(index % 3)}

    class UnusedCPUModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, 3)

        def forward(self, input_ids, labels=None):
            raise AssertionError("data-loader verification must not perform a model forward")

    generator = torch.Generator(device="cpu").manual_seed(20261011)
    unshuffled = list(range(24436)) + list(range(2606)) * 2
    indices = [unshuffled[index] for index in torch.randperm(29648, generator=generator).tolist()]
    args = recovery.TrainingArguments(
        output_dir=str(tmp_path / "unused-trainer-output"),
        use_cpu=True,
        per_device_train_batch_size=16,
        num_train_epochs=1,
        max_steps=1853,
        dataloader_drop_last=False,
        dataloader_pin_memory=False,
        train_sampling_strategy="random",
        remove_unused_columns=False,
        report_to=[],
        disable_tqdm=True,
        seed=20261011,
        data_seed=20261011,
    )
    trainer = recovery.ReplayTrainer(
        model=UnusedCPUModel(),
        args=args,
        train_dataset=IndexedDataset(),
        class_weights=torch.ones(3),
        binary_loss_weight=0.0,
        binary_positive_weight=1.0,
        replay_indices=indices,
    )
    assert trainer.args.device.type == "cpu"
    loader = trainer.get_train_dataloader()
    assert len(loader) == 1853
    observed = []
    for batch in loader:
        assert batch["input_ids"].device.type == "cpu"
        assert tuple(batch["input_ids"].shape) == (16, 1)
        observed.extend(batch["input_ids"][:, 0].tolist())
    assert observed == indices
    assert len(observed) == 29648
    assert Counter(observed) == Counter(unshuffled)


def test_replay_receipt_fails_closed(tmp_path):
    path = tmp_path / "replay.json"
    expected = {"config_sha256": "a", "replay": {"index_order_sha256": "b"}}
    with pytest.raises(ValueError, match="preflight-frozen"):
        recovery.verify_replay_receipt(path, expected)
    path.write_text(json.dumps(expected))
    assert recovery.verify_replay_receipt(path, expected) == recovery.file_sha256(path)
    expected["replay"]["index_order_sha256"] = "changed"
    with pytest.raises(ValueError, match="preflight-frozen"):
        recovery.verify_replay_receipt(path, expected)


@pytest.fixture
def parent(tmp_path):
    checkpoint = tmp_path / "parent"
    checkpoint.mkdir()
    for name in (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "scamguard_calibration.json",
    ):
        (checkpoint / name).write_text("synthetic fixture: " + name)
    files = {path.name: recovery.file_sha256(path) for path in checkpoint.iterdir()}
    report = tmp_path / "parent.train.json"
    receipt = {
        "status": "completed",
        "training_completed": True,
        "global_step": 3056,
        "stop_reason": "step_bound_reached",
        "checkpoint_model_sha256": files["model.safetensors"],
        "checkpoint_files_sha256": files,
        "development_quality_passed": False,
    }
    report.write_text(json.dumps(receipt))
    initialization = {
        "checkpoint": str(checkpoint),
        "model_sha256": files["model.safetensors"],
        "parent_report": str(report),
        "parent_report_sha256": recovery.file_sha256(report),
        "files_sha256": files,
    }
    return initialization, report, checkpoint


def test_parent_can_fail_quality_but_must_be_completed_and_hash_exact(parent):
    initialization, report, checkpoint = parent
    evidence = recovery.verify_parent(initialization)
    assert evidence["optimizer_state_reused"] is False
    (checkpoint / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="files changed"):
        recovery.verify_parent(initialization)


def test_parent_partial_fit_rejected(parent):
    initialization, report, _ = parent
    receipt = json.loads(report.read_text())
    receipt["global_step"] = 3055
    report.write_text(json.dumps(receipt))
    initialization["parent_report_sha256"] = recovery.file_sha256(report)
    with pytest.raises(ValueError, match="complete v1"):
        recovery.verify_parent(initialization)


def test_parent_report_or_file_manifest_tampering_rejected(parent):
    initialization, report, _ = parent
    original_sha = initialization["parent_report_sha256"]
    initialization["parent_report_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="receipt hash"):
        recovery.verify_parent(initialization)
    initialization["parent_report_sha256"] = original_sha
    initialization["files_sha256"] = {"model.safetensors": initialization["model_sha256"]}
    with pytest.raises(ValueError, match="file contract"):
        recovery.verify_parent(initialization)
