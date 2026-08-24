#!/usr/bin/env python3
"""Freeze the non-release Qwen3.5-0.8B experiment using the AI audit overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scamguard.metrics import file_sha256

try:
    from scripts.freeze_qwen08_full_experiment import (
        BASE_MODEL,
        BASE_REVISION,
        TRANSFORMERS_REVISION,
        audit_sft,
        validate_batch_selection,
        validate_token_audit,
    )
    from training.train_qwen_lora import LANGUAGE_LORA_TARGETS
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from freeze_qwen08_full_experiment import (  # type: ignore[no-redef]
        BASE_MODEL,
        BASE_REVISION,
        TRANSFORMERS_REVISION,
        audit_sft,
        validate_batch_selection,
        validate_token_audit,
    )

    from training.train_qwen_lora import LANGUAGE_LORA_TARGETS


def freeze(
    processed: Path,
    token_audit_path: Path,
    internal_audit_path: Path,
    batch_selection_path: Path,
    output: Path,
    checkpoint_output: Path,
    experiment_id: str,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen experiment: {output}")
    manifest_path = processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    internal_audit = json.loads(internal_audit_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_kind")
        != "ai_internal_exploratory_correction_overlay"
        or manifest.get("release_eligible") is not False
        or manifest.get("publication_authorized") is not False
        or manifest.get("internal_ai_audit_report_sha256")
        != file_sha256(internal_audit_path)
        or internal_audit.get("review_kind") != "ai_internal_blind"
        or internal_audit.get("independent_human_review") is not False
        or internal_audit.get("release_gate_passed") is not False
        or internal_audit.get("publication_authorized") is not False
    ):
        raise ValueError("AI-internal overlay or audit changed its non-release contract")

    sft_manifest_path = processed / "qwen_sft" / "manifest.json"
    sft_manifest = json.loads(sft_manifest_path.read_text(encoding="utf-8"))
    if sft_manifest.get("input_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Qwen SFT data is not bound to the AI-internal overlay")
    sft_audit = audit_sft(processed)
    token_audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
    validate_token_audit(token_audit, token_audit_path, sft_audit)
    batch_selection = validate_batch_selection(batch_selection_path)

    evaluation = {
        f"{path.stem}_sha256": file_sha256(path)
        for path in sorted(processed.glob("*.jsonl"))
        if path.name not in {"train.jsonl", "test.jsonl"}
    }
    config: dict[str, object] = {
        "experiment_id": experiment_id,
        "run_kind": "exploratory",
        "role": "AI-audited non-release 0.8B label-correction experiment",
        "release_eligible": False,
        "publication_authorized": False,
        "checkpoint_output": str(checkpoint_output),
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "seed": 20260820,
        "epochs": 1.0,
        "batch_size": 4,
        "eval_batch_size": 4,
        "gradient_accumulation": 4,
        "gradient_checkpointing": True,
        "trainer_eval": True,
        "evaluation_policy": (
            "development loss and open diagnostics only; no sealed or release benchmark claim"
        ),
        "learning_rate": 0.0001,
        "warmup_fraction": 0.05,
        "weight_decay": 0.01,
        "max_length": 640,
        "sampling": "group_by_length",
        "batch_geometry_selection": batch_selection,
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
            "sft_build_manifest_path": str(sft_manifest_path),
            "sft_build_manifest_sha256": file_sha256(sft_manifest_path),
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
            "internal_ai_label_audit": {
                "report_path": str(internal_audit_path),
                "report_sha256": file_sha256(internal_audit_path),
                "rows": internal_audit["rows"],
                "agreement": internal_audit["agreement"],
                "cohen_kappa": internal_audit["cohen_kappa"],
                "incorrect_label_rows": internal_audit["incorrect_label_rows"],
                "sensitive_data_rows": internal_audit["sensitive_data_rows"],
                "independent_human_review": False,
                "release_gate_passed": False,
                "publication_authorized": False,
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
    parser.add_argument("--internal-audit", type=Path, required=True)
    parser.add_argument("--batch-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.processed,
                args.token_audit,
                args.internal_audit,
                args.batch_selection,
                args.output,
                args.checkpoint_output,
                args.experiment_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
