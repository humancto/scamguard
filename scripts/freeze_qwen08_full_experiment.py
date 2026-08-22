#!/usr/bin/env python3
"""Freeze a full schema-v24 Qwen3.5-0.8B experiment from measured artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256
from training.build_qwen_sft import validate_target
from training.train_qwen_lora import LANGUAGE_LORA_TARGETS

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TRANSFORMERS_REVISION = "0c92811846095910816a87aca50050d10c545270"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"Qwen SFT row {row.get('id')} must contain exactly three messages")
    user = messages[1]
    assistant = messages[2]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError(f"Qwen SFT row {row.get('id')} has an invalid chat role sequence")
    content = str(user.get("content", ""))
    prefix = "Classify this message:\n<message>"
    suffix = "</message>"
    if not content.startswith(prefix) or not content.endswith(suffix):
        raise ValueError(f"Qwen SFT row {row.get('id')} has an invalid user envelope")
    return content[len(prefix) : -len(suffix)]


def audit_sft(processed: Path) -> dict[str, object]:
    split_rows = {
        split: read_jsonl(processed / "qwen_sft" / f"{split}.jsonl")
        for split in ("train", "dev")
    }
    ids: set[str] = set()
    families: dict[str, set[str]] = {}
    scam_examples = 0
    scam_with_evidence = 0
    for split, rows in split_rows.items():
        families[split] = set()
        for row in rows:
            identifier = str(row.get("id", ""))
            family = str(row.get("family_id", ""))
            if not identifier or identifier in ids:
                raise ValueError(f"Qwen SFT row ID is empty or duplicated: {identifier!r}")
            if not family:
                raise ValueError(f"Qwen SFT row {identifier} lacks a family ID")
            ids.add(identifier)
            families[split].add(family)
            messages = row["messages"]
            target = json.loads(str(messages[2]["content"]))
            text = source_text(row)
            validate_target(target, text)
            if target["verdict"] == "SCAM":
                scam_examples += 1
                if target["evidence"]:
                    scam_with_evidence += 1
    if families["train"] & families["dev"]:
        raise ValueError("Qwen SFT family crosses train and development")
    return {
        "train_examples": len(split_rows["train"]),
        "dev_examples": len(split_rows["dev"]),
        "train_families": len(families["train"]),
        "dev_families": len(families["dev"]),
        "train_dev_scam_examples": scam_examples,
        "with_verbatim_evidence": scam_with_evidence,
        "evidence_coverage": scam_with_evidence / max(scam_examples, 1),
    }


def validate_schema24_manifest(manifest: dict[str, Any], processed: Path) -> None:
    if manifest.get("schema_version") != 24:
        raise ValueError("full Qwen 0.8B experiment requires schema version 24")
    increment = manifest.get("schema24_increment")
    if not isinstance(increment, dict):
        raise ValueError("schema-v24 manifest lacks its annotation increment")
    if increment.get("paper_dev_test_rows_used_for_fitting") is not False:
        raise ValueError("schema-v24 paper dev/test rows must remain outside fitting")
    for field in ("annotation_train_rows", "annotation_dev_rows", "annotation_test_rows"):
        value = increment.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"schema-v24 {field} must be a positive integer")
    curriculum_path = Path(str(increment.get("annotation_curriculum_manifest", "")))
    expected_hash = str(increment.get("annotation_curriculum_manifest_sha256", ""))
    if not curriculum_path.is_file() or not SHA256_RE.fullmatch(expected_hash):
        raise ValueError("schema-v24 annotation curriculum identity is incomplete")
    if file_sha256(curriculum_path) != expected_hash:
        raise ValueError("schema-v24 annotation curriculum hash differs")
    if not curriculum_path.resolve().is_relative_to(processed.parent.parent.parent.resolve()):
        # External data normally sits elsewhere in the repository, so only reject paths outside
        # the repository root inferred from data/experiments/<experiment>/processed.
        repository_root = processed.resolve().parents[3]
        if not curriculum_path.resolve().is_relative_to(repository_root):
            raise ValueError("schema-v24 annotation curriculum path escapes the repository")


def validate_token_audit(
    report: dict[str, Any], report_path: Path, sft_audit: dict[str, object]
) -> None:
    if (
        report.get("model") != BASE_MODEL
        or report.get("revision") != BASE_REVISION
        or report.get("max_length") != 512
        or report.get("full_over_max_length") != 0
    ):
        raise ValueError("Qwen 0.8B token audit did not pass the frozen 512-token contract")
    split_counts = report.get("split_counts")
    if not isinstance(split_counts, dict) or split_counts != {
        "train": sft_audit["train_examples"],
        "dev": sft_audit["dev_examples"],
    }:
        raise ValueError("Qwen token audit split counts differ from SFT data")
    if not report_path.is_file():
        raise ValueError("Qwen token audit report is missing")


def validate_label_audit(
    report: dict[str, Any], report_path: Path, data_manifest_path: Path
) -> None:
    if (
        report.get("release_gate_passed") is not True
        or report.get("rows") != report.get("complete_rows")
        or report.get("incorrect_label_rows") != 0
        or report.get("sensitive_data_rows") != 0
        or report.get("errors") != []
    ):
        raise ValueError("schema-v24 independent human label audit has not passed")
    if report.get("data_manifest_sha256") != file_sha256(data_manifest_path):
        raise ValueError("schema-v24 label audit is not bound to this data manifest")
    if not report_path.is_file():
        raise ValueError("schema-v24 label audit report is missing")


def freeze(
    processed: Path,
    token_audit_path: Path,
    label_audit_path: Path,
    output: Path,
    checkpoint_output: Path,
    experiment_id: str,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen Qwen experiment: {output}")
    manifest_path = processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_schema24_manifest(manifest, processed)
    sft_audit = audit_sft(processed)
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or (
        counts.get("train") != sft_audit["train_examples"]
        or counts.get("dev") != sft_audit["dev_examples"]
    ):
        raise ValueError("schema-v24 manifest counts differ from Qwen SFT data")
    token_audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
    validate_token_audit(token_audit, token_audit_path, sft_audit)
    label_audit = json.loads(label_audit_path.read_text(encoding="utf-8"))
    validate_label_audit(label_audit, label_audit_path, manifest_path)

    evaluation = {
        f"{path.stem}_sha256": file_sha256(path)
        for path in sorted(processed.glob("*.jsonl"))
        if path.name != "train.jsonl"
    }
    required_evaluation = {"dev_sha256", "test_sha256"}
    if not required_evaluation <= set(evaluation):
        raise ValueError("schema-v24 processed data lacks dev or test evaluation artifacts")
    config: dict[str, object] = {
        "experiment_id": experiment_id,
        "run_kind": "full",
        "role": "quality-first full 0.8B challenger",
        "checkpoint_output": str(checkpoint_output),
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "seed": 20260820,
        "epochs": 1.0,
        "batch_size": 16,
        "eval_batch_size": 4,
        "gradient_accumulation": 1,
        "gradient_checkpointing": True,
        "trainer_eval": True,
        "evaluation_policy": "development loss only; frozen benchmark evaluation after training",
        "learning_rate": 0.0001,
        "warmup_fraction": 0.05,
        "weight_decay": 0.01,
        "max_length": 512,
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
            "processed_directory": str(processed),
            "manifest_sha256": file_sha256(manifest_path),
            "train_jsonl_sha256": file_sha256(processed / "qwen_sft" / "train.jsonl"),
            "dev_jsonl_sha256": file_sha256(processed / "qwen_sft" / "dev.jsonl"),
            "train_examples": sft_audit["train_examples"],
            "dev_examples": sft_audit["dev_examples"],
            "train_families": sft_audit["train_families"],
            "dev_families": sft_audit["dev_families"],
            "evaluation": evaluation,
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
            "label_audit": {
                "report_path": str(label_audit_path),
                "report_sha256": file_sha256(label_audit_path),
                "rows": label_audit["rows"],
                "agreement": label_audit["agreement"],
                "release_gate_passed": label_audit["release_gate_passed"],
                "data_manifest_sha256": label_audit["data_manifest_sha256"],
            },
            "evidence_audit": {
                "train_dev_scam_examples": sft_audit["train_dev_scam_examples"],
                "with_verbatim_evidence": sft_audit["with_verbatim_evidence"],
                "coverage": sft_audit["evidence_coverage"],
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.processed,
                args.token_audit,
                args.label_audit,
                args.output,
                args.checkpoint_output,
                args.experiment_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
