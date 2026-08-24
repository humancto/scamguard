#!/usr/bin/env python3
"""Freeze the non-release Qwen3.5-0.8B call-robustness continuation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from training.train_qwen_lora import LANGUAGE_LORA_TARGETS, adapter_identity

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TRANSFORMERS_REVISION = "0c92811846095910816a87aca50050d10c545270"


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def freeze(
    curriculum: Path,
    token_audit_path: Path,
    initial_adapter: Path,
    source_report_path: Path,
    output: Path,
    checkpoint_output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen experiment: {output}")
    manifest_path = curriculum / "manifest.json"
    sft = curriculum / "qwen_sft"
    sft_manifest_path = sft / "manifest.json"
    train_path = sft / "train.jsonl"
    dev_path = sft / "dev.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sft_manifest = json.loads(sft_manifest_path.read_text(encoding="utf-8"))
    token_audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    initial = adapter_identity(initial_adapter)
    if (
        manifest.get("experiment_kind")
        != "qwen_call_robustness_stage2_curriculum"
        or manifest.get("release_eligible") is not False
        or manifest.get("publication_authorized") is not False
        or sft_manifest.get("input_manifest_sha256") != file_sha256(manifest_path)
    ):
        raise ValueError("call-robustness curriculum changed its non-release contract")
    if (
        token_audit.get("revision") != BASE_REVISION
        or token_audit.get("max_length") != 640
        or token_audit.get("full_over_max_length") != 0
        or token_audit.get("split_counts", {}).get("train") != line_count(train_path)
        or token_audit.get("split_counts", {}).get("dev") != line_count(dev_path)
    ):
        raise ValueError("token audit differs from the call-robustness data contract")
    if source_report.get("adapter_sha256") != initial["adapter_model_sha256"]:
        raise ValueError("source evaluation report does not bind the initial adapter")

    config: dict[str, object] = {
        "experiment_id": "sg-qwen35-08b-call-robustness-stage2-v1",
        "run_kind": "exploratory_continuation",
        "role": "development-only call-context and abstention robustness continuation",
        "release_eligible": False,
        "publication_authorized": False,
        "checkpoint_output": str(checkpoint_output),
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "initial_adapter": initial,
        "source_evaluation": {
            "path": str(source_report_path),
            "sha256": file_sha256(source_report_path),
            "status": "rejected; regression evidence only for this continuation",
        },
        "seed": 20260824,
        "epochs": 1.0,
        "batch_size": 4,
        "eval_batch_size": 4,
        "gradient_accumulation": 4,
        "gradient_checkpointing": True,
        "trainer_eval": True,
        "evaluation_policy": (
            "development loss, then frozen regression and selection diagnostics; "
            "primary_test_v8 remains unopened until this adapter and threshold policy are frozen"
        ),
        "learning_rate": 0.00002,
        "warmup_fraction": 0.05,
        "weight_decay": 0.01,
        "max_length": 640,
        "sampling": "group_by_length",
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": LANGUAGE_LORA_TARGETS,
            "scope": "language tower only; asserted at runtime",
        },
        "data": {
            "schema_version": 24,
            "processed_directory": str(curriculum),
            "manifest_sha256": file_sha256(manifest_path),
            "train_jsonl_sha256": file_sha256(train_path),
            "dev_jsonl_sha256": file_sha256(dev_path),
            "train_examples": line_count(train_path),
            "dev_examples": line_count(dev_path),
            "train_families": manifest["splits"]["train"]["families"],
            "dev_families": manifest["splits"]["dev"]["families"],
            "sft_build_manifest_path": str(sft_manifest_path),
            "sft_build_manifest_sha256": file_sha256(sft_manifest_path),
            "token_length_audit": {
                "report_path": str(token_audit_path),
                "report_sha256": file_sha256(token_audit_path),
                "examples": token_audit["examples"],
                "full_p95": token_audit["full_tokens"]["p95"],
                "full_p99": token_audit["full_tokens"]["p99"],
                "full_max": token_audit["full_tokens"]["max"],
                "full_over_max_length": token_audit["full_over_max_length"],
                "minimum_supervised_tokens": token_audit["supervised_tokens"]["min"],
            },
            "held_rows_used_for_fitting": 0,
            "primary_test_rows_used_for_fitting": 0,
            "bothbosu_rows_used_for_fitting": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.curriculum,
                args.token_audit,
                args.initial_adapter,
                args.source_report,
                args.output,
                args.checkpoint_output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
