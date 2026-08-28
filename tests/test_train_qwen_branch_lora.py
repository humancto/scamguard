from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from training.train_qwen_branch_lora import branch_experiment_errors
from training.train_qwen_lora import LANGUAGE_LORA_TARGETS, adapter_identity

TRANSFORMERS_REVISION = "0c92811846095910816a87aca50050d10c545270"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, object]]:
    processed = tmp_path / "processed"
    data = processed / "qwen_sft"
    data.mkdir(parents=True)
    (processed / "manifest.json").write_text("{}", encoding="utf-8")
    (data / "manifest.json").write_text("{}", encoding="utf-8")
    (data / "train.jsonl").write_text('{"id":"train"}\n', encoding="utf-8")
    (data / "dev.jsonl").write_text('{"id":"dev"}\n', encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / "teacher.jsonl"
    cache.write_text(
        json.dumps({"split": "train", "id": "train", "teacher_logits": [1, 0, -1]})
        + "\n",
        encoding="utf-8",
    )
    cache_manifest = tmp_path / "teacher-manifest.json"
    cache_manifest.write_text(
        json.dumps(
            {
                "contains_message_text": False,
                "base_model_revision": "a" * 40,
                "adapter": adapter_identity(adapter),
                "data": {
                    "train": {"sha256": digest(data / "train.jsonl")},
                    "dev": {"sha256": digest(data / "dev.jsonl")},
                },
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        model="Qwen/Qwen3.5-0.8B",
        revision="a" * 40,
        seed=20260829,
        epochs=1.0,
        batch_size=4,
        eval_batch_size=4,
        gradient_accumulation=4,
        gradient_checkpointing=True,
        learning_rate=0.000001,
        max_length=640,
        sampling_strategy="group_by_length",
        skip_eval=False,
        output=tmp_path / "output",
        data=data,
        initial_adapter=adapter,
        teacher_cache=cache,
        teacher_cache_manifest=cache_manifest,
        class_weights=[1.0, 3.0, 1.0],
        focal_gamma=2.0,
        kl_weight=5.0,
        kl_temperature=1.0,
    )
    config: dict[str, object] = {
        "run_kind": "exploratory_continuation",
        "base_model": args.model,
        "base_model_revision": args.revision,
        "transformers_revision": TRANSFORMERS_REVISION,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "sampling": args.sampling_strategy,
        "warmup_fraction": 0.05,
        "weight_decay": 0.01,
        "trainer_eval": True,
        "checkpoint_output": str(args.output),
        "initial_adapter": adapter_identity(adapter),
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": LANGUAGE_LORA_TARGETS,
            "scope": "language tower only; asserted at runtime",
        },
        "data": {
            "processed_directory": str(processed),
            "manifest_sha256": digest(processed / "manifest.json"),
            "sft_build_manifest_sha256": digest(data / "manifest.json"),
            "train_jsonl_sha256": digest(data / "train.jsonl"),
            "dev_jsonl_sha256": digest(data / "dev.jsonl"),
            "train_examples": 1,
            "dev_examples": 1,
        },
        "objective": {
            "type": "branch_token_focal_kl_v1",
            "labels": ["SAFE", "UNCERTAIN", "SCAM"],
            "class_weights": [1.0, 3.0, 1.0],
            "focal_gamma": 2.0,
            "kl_weight": 5.0,
            "kl_temperature": 1.0,
            "deployment_head_added": False,
            "teacher_cache": {
                "path": str(cache),
                "sha256": digest(cache),
                "manifest_path": str(cache_manifest),
                "manifest_sha256": digest(cache_manifest),
            },
        },
    }
    return args, config


def test_branch_experiment_binds_objective_cache_and_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    args, config = fixture(tmp_path)
    monkeypatch.setattr(
        "training.train_qwen_lora.installed_transformers_revision",
        lambda: TRANSFORMERS_REVISION,
    )
    assert branch_experiment_errors(args, config) == []


def test_branch_experiment_rejects_cache_or_objective_drift(tmp_path: Path, monkeypatch) -> None:
    args, config = fixture(tmp_path)
    monkeypatch.setattr(
        "training.train_qwen_lora.installed_transformers_revision",
        lambda: TRANSFORMERS_REVISION,
    )
    args.focal_gamma = 1.0
    args.teacher_cache.write_text("changed", encoding="utf-8")
    errors = branch_experiment_errors(args, config)
    assert any(error.startswith("objective.focal_gamma") for error in errors)
    assert any(error.startswith("teacher-cache hash mismatch") for error in errors)


def test_branch_trainer_loss_has_gradients() -> None:
    logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    from training.qwen_branch import branch_focal_kl_loss

    loss, _ = branch_focal_kl_loss(
        logits,
        torch.tensor([1]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        class_weights=torch.tensor([1.0, 3.0, 1.0]),
        focal_gamma=2.0,
        kl_weight=5.0,
        kl_temperature=1.0,
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
