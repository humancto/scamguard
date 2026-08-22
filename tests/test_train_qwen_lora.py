from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from training.train_qwen_lora import (
    LANGUAGE_LORA_TARGETS,
    completion_token_start,
    experiment_config_errors,
)


def test_completion_token_start_at_exact_token_boundary() -> None:
    assert completion_token_start(
        "abc",
        "abcdef",
        [(0, 2), (2, 3), (3, 5), (5, 6)],
    ) == 2


def test_completion_token_start_masks_bpe_token_crossing_boundary() -> None:
    assert completion_token_start(
        "abc",
        "abcdef",
        [(0, 2), (2, 4), (4, 6)],
    ) == 2


def test_completion_token_start_rejects_non_prefix_or_truncated_completion() -> None:
    with pytest.raises(ValueError, match="exact string prefix"):
        completion_token_start("abc", "abZdef", [(0, 2), (2, 4), (4, 6)])
    with pytest.raises(ValueError, match="no token wholly inside"):
        completion_token_start("abc", "abcd", [(0, 2), (2, 4)])


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_experiment(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, object]]:
    processed = tmp_path / "processed"
    sft = processed / "qwen_sft"
    sft.mkdir(parents=True)
    manifest = processed / "manifest.json"
    train = sft / "train.jsonl"
    dev = sft / "dev.jsonl"
    manifest.write_text(json.dumps({"schema_version": 24}), encoding="utf-8")
    train.write_text('{"id":"train-1"}\n', encoding="utf-8")
    dev.write_text('{"id":"dev-1"}\n', encoding="utf-8")
    token_audit = tmp_path / "token-audit.json"
    token_audit.write_text('{"full_over_max_length":0}', encoding="utf-8")
    label_audit = tmp_path / "label-audit.json"
    label_audit.write_text('{"release_gate_passed":true}', encoding="utf-8")
    output = tmp_path / "checkpoint"
    args = argparse.Namespace(
        model="Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        seed=20260820,
        epochs=1.0,
        batch_size=16,
        eval_batch_size=4,
        gradient_accumulation=1,
        gradient_checkpointing=True,
        learning_rate=0.0001,
        max_length=512,
        sampling_strategy="group_by_length",
        skip_eval=False,
        output=output,
        data=sft,
    )
    config: dict[str, object] = {
        "run_kind": "full",
        "base_model": args.model,
        "base_model_revision": args.revision,
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
        "checkpoint_output": str(output),
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": LANGUAGE_LORA_TARGETS,
            "scope": "language tower only; asserted at runtime",
        },
        "data": {
            "schema_version": 24,
            "processed_directory": str(processed),
            "manifest_sha256": file_sha256(manifest),
            "train_jsonl_sha256": file_sha256(train),
            "dev_jsonl_sha256": file_sha256(dev),
            "train_examples": 1,
            "dev_examples": 1,
            "token_length_audit": {
                "report_path": str(token_audit),
                "report_sha256": file_sha256(token_audit),
                "full_over_max_length": 0,
                "minimum_supervised_tokens": 8,
            },
            "evidence_audit": {"coverage": 1.0},
            "label_audit": {
                "report_path": str(label_audit),
                "report_sha256": file_sha256(label_audit),
                "release_gate_passed": True,
                "data_manifest_sha256": file_sha256(manifest),
            },
        },
    }
    return args, config


def test_experiment_config_binds_command_data_and_output(tmp_path: Path) -> None:
    args, config = frozen_experiment(tmp_path)

    assert experiment_config_errors(args, config) == []


def test_experiment_config_rejects_parameter_and_data_drift(tmp_path: Path) -> None:
    args, config = frozen_experiment(tmp_path)
    args.sampling_strategy = "random"
    (args.data / "train.jsonl").write_text('{"id":"changed"}\n', encoding="utf-8")

    errors = experiment_config_errors(args, config)

    assert any(error.startswith("sampling:") for error in errors)
    assert any(error.startswith("data hash mismatch:") for error in errors)


def test_full_experiment_rejects_tampered_token_audit(tmp_path: Path) -> None:
    args, config = frozen_experiment(tmp_path)
    token_audit = Path(str(config["data"]["token_length_audit"]["report_path"]))  # type: ignore[index]
    token_audit.write_text('{"full_over_max_length":1}', encoding="utf-8")

    errors = experiment_config_errors(args, config)

    assert any(error.startswith("token audit hash mismatch:") for error in errors)
