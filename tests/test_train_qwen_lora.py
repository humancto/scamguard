from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from training.train_qwen_lora import (
    LANGUAGE_LORA_TARGETS,
    adapter_identity,
    completion_token_start,
    experiment_config_errors,
    final_eval_metrics,
    require_loaded_revision,
)

TRANSFORMERS_REVISION = "0c92811846095910816a87aca50050d10c545270"


def test_final_eval_metrics_returns_last_eval_record_without_progress_fields() -> None:
    history = [
        {"loss": 0.2, "step": 10},
        {"eval_loss": 0.08, "eval_runtime": 4.0, "epoch": 0.5},
        {"eval_loss": 0.04, "eval_runtime": 5.0, "epoch": 1.0},
        {"train_runtime": 100.0, "epoch": 1.0},
    ]

    assert final_eval_metrics(history) == {
        "eval_loss": 0.04,
        "eval_runtime": 5.0,
    }


def test_final_eval_metrics_returns_none_when_eval_was_skipped() -> None:
    assert final_eval_metrics([{"train_loss": 0.1}]) is None


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
    sft_manifest = sft / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 24}), encoding="utf-8")
    train.write_text('{"id":"train-1"}\n', encoding="utf-8")
    dev.write_text('{"id":"dev-1"}\n', encoding="utf-8")
    sft_manifest.write_text('{"artifact_schema_version":1}', encoding="utf-8")
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
        batch_size=4,
        eval_batch_size=4,
        gradient_accumulation=4,
        gradient_checkpointing=True,
        learning_rate=0.0001,
        max_length=640,
        sampling_strategy="group_by_length",
        skip_eval=False,
        output=output,
        data=sft,
    )
    config: dict[str, object] = {
        "run_kind": "full",
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
        "checkpoint_output": str(output),
        "batch_geometry_selection": {},
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
            "sft_build_manifest_sha256": file_sha256(sft_manifest),
            "train_jsonl_sha256": file_sha256(train),
            "dev_jsonl_sha256": file_sha256(dev),
            "train_examples": 1,
            "dev_examples": 1,
            "sft_exclusions": {"train": 0, "dev": 0},
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
                "imported_from_blind_bundle": True,
                "data_manifest_sha256": file_sha256(manifest),
            },
        },
    }
    selection_path = tmp_path / "batch-selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selected": {
                    "microbatch_size": 4,
                    "gradient_accumulation": 4,
                    "effective_batch_size": 16,
                },
                "quality_contract": {
                    "sequence_length": 640,
                    "tokens_per_effective_batch": 10_240,
                    "optimizer_semantics_changed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    config["batch_geometry_selection"] = {
        "report_path": str(selection_path),
        "report_sha256": file_sha256(selection_path),
        "microbatch_size": 4,
        "gradient_accumulation": 4,
        "effective_batch_size": 16,
        "sequence_length": 640,
        "tokens_per_effective_batch": 10_240,
    }
    return args, config


def test_experiment_config_binds_command_data_and_output(tmp_path: Path) -> None:
    args, config = frozen_experiment(tmp_path)

    assert experiment_config_errors(
        args, config, transformers_revision=TRANSFORMERS_REVISION
    ) == []


def test_experiment_config_rejects_parameter_and_data_drift(tmp_path: Path) -> None:
    args, config = frozen_experiment(tmp_path)
    args.sampling_strategy = "random"
    (args.data / "train.jsonl").write_text('{"id":"changed"}\n', encoding="utf-8")

    errors = experiment_config_errors(
        args, config, transformers_revision=TRANSFORMERS_REVISION
    )

    assert any(error.startswith("sampling:") for error in errors)
    assert any(error.startswith("data hash mismatch:") for error in errors)


def test_full_experiment_rejects_tampered_token_audit(tmp_path: Path) -> None:
    args, config = frozen_experiment(tmp_path)
    token_audit = Path(str(config["data"]["token_length_audit"]["report_path"]))  # type: ignore[index]
    token_audit.write_text('{"full_over_max_length":1}', encoding="utf-8")

    errors = experiment_config_errors(
        args, config, transformers_revision=TRANSFORMERS_REVISION
    )

    assert any(error.startswith("token audit hash mismatch:") for error in errors)


def test_experiment_config_rejects_transformers_revision_drift(tmp_path: Path) -> None:
    args, config = frozen_experiment(tmp_path)

    errors = experiment_config_errors(args, config, transformers_revision="f" * 40)

    assert any(error.startswith("transformers_revision:") for error in errors)


def test_loaded_base_revision_must_be_present_and_exact() -> None:
    revision = "a" * 40

    require_loaded_revision(requested=revision, loaded=revision)
    with pytest.raises(RuntimeError, match="<missing>"):
        require_loaded_revision(requested=revision, loaded=None)
    with pytest.raises(RuntimeError, match="differs"):
        require_loaded_revision(requested=revision, loaded="b" * 40)


def test_continuation_adapter_is_hash_bound(tmp_path: Path) -> None:
    args, config = frozen_experiment(tmp_path)
    adapter = tmp_path / "initial-adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    args.initial_adapter = adapter
    config["initial_adapter"] = adapter_identity(adapter)

    assert experiment_config_errors(
        args, config, transformers_revision=TRANSFORMERS_REVISION
    ) == []

    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    errors = experiment_config_errors(
        args, config, transformers_revision=TRANSFORMERS_REVISION
    )
    assert "initial_adapter: path or immutable adapter hash differs" in errors


def test_full_experiment_rejects_unselected_batch_geometry(tmp_path: Path) -> None:
    args, config = frozen_experiment(tmp_path)
    args.batch_size = 2
    args.gradient_accumulation = 8

    errors = experiment_config_errors(
        args, config, transformers_revision=TRANSFORMERS_REVISION
    )

    assert any("requires selected microbatch 4" in error for error in errors)
    assert any("batch geometry" in error for error in errors)


def test_full_experiment_rejects_non_blind_label_audit(tmp_path: Path) -> None:
    args, config = frozen_experiment(tmp_path)
    config["data"]["label_audit"]["imported_from_blind_bundle"] = False  # type: ignore[index]

    errors = experiment_config_errors(
        args, config, transformers_revision=TRANSFORMERS_REVISION
    )

    assert "full run label audit was not imported from a blind bundle" in errors
