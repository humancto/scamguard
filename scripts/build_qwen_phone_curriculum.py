#!/usr/bin/env python3
"""Build the frozen training-only Qwen Stage 7 phone curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from scripts.audit_source_overlap import near_overlap_indices
from training.build_qwen_sft import convert_supported_rows

PARENT_KIND = "qwen_boundary_recovery_stage3_curriculum"
EXPERIMENT_KIND = "qwen_phone_generalization_stage7_curriculum"
PHONE_SOURCE = "shakeleoatmeal_phone_scam_synthetic"
PHONE_REPOSITORY = "shakeleoatmeal/phone-scam-detection-synthetic"
PHONE_REVISION = "27a1f6d0cf21dda995130d221383900b060dbcc9"
PHONE_LICENSE = "MIT"
QWEN_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def verdict(row: dict[str, Any]) -> str:
    return str(json.loads(row["messages"][-1]["content"])["verdict"])


def sft_as_text_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        content = str(row["messages"][-2]["content"])
        prefix = "Classify this message:\n<message>"
        suffix = "</message>"
        if not content.startswith(prefix) or not content.endswith(suffix):
            raise ValueError(f"row {row.get('id')} has an unexpected user-message contract")
        output.append({"text": content[len(prefix) : -len(suffix)]})
    return output


def validate_phone_manifest(manifest: dict[str, Any], train_path: Path) -> dict[str, Any]:
    artifact = manifest.get("artifacts", {}).get("train", {})
    policy = manifest.get("policy", {})
    if (
        manifest.get("source") != PHONE_SOURCE
        or manifest.get("repository") != PHONE_REPOSITORY
        or manifest.get("revision") != PHONE_REVISION
        or manifest.get("license_declared_by_publisher") != PHONE_LICENSE
        or artifact.get("sha256") != file_sha256(train_path)
        or artifact.get("rows") != sum(1 for line in train_path.open() if line.strip())
        or policy.get("synthetic") is not True
        or policy.get("direct_reddit_scrape") is not False
        or policy.get("source_metadata_used_as_model_input") is not False
        or policy.get("test_prediction_sealed_until_candidate_freeze") is not True
        or policy.get("variation_style_is_perfectly_label_confounded") is not True
    ):
        raise ValueError("phone supplement differs from its pinned rights and split contract")
    return artifact


def build(
    parent: Path,
    phone_manifest_path: Path,
    phone_train_path: Path,
    output: Path,
    *,
    overlap_references: tuple[Path, ...] = (),
    tokenizer: Any | None = None,
    max_length: int = 640,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite curriculum: {output}")

    parent_manifest_path = parent / "manifest.json"
    parent_sft = parent / "qwen_sft"
    parent_sft_manifest_path = parent_sft / "manifest.json"
    parent_train_path = parent_sft / "train.jsonl"
    parent_dev_path = parent_sft / "dev.jsonl"
    for path in (
        parent_manifest_path,
        parent_sft_manifest_path,
        parent_train_path,
        parent_dev_path,
        phone_manifest_path,
        phone_train_path,
    ):
        if not path.is_file():
            raise ValueError(f"missing input artifact: {path}")

    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_sft_manifest = json.loads(
        parent_sft_manifest_path.read_text(encoding="utf-8")
    )
    if (
        parent_manifest.get("experiment_kind") != PARENT_KIND
        or parent_manifest.get("release_eligible") is not False
        or parent_manifest.get("publication_authorized") is not False
        or parent_sft_manifest.get("input_manifest_sha256")
        != file_sha256(parent_manifest_path)
    ):
        raise ValueError("parent is not the frozen non-release Stage 3 curriculum")

    parent_train = read_jsonl(parent_train_path)
    parent_dev = read_jsonl(parent_dev_path)
    parent_ids = [str(row.get("id", "")) for row in parent_train]
    dev_ids = [str(row.get("id", "")) for row in parent_dev]
    parent_families = {str(row.get("family_id", "")) for row in parent_train}
    dev_families = {str(row.get("family_id", "")) for row in parent_dev}
    if (
        not all(parent_ids + dev_ids)
        or len(parent_ids) != len(set(parent_ids))
        or len(dev_ids) != len(set(dev_ids))
        or set(parent_ids) & set(dev_ids)
        or not all(parent_families | dev_families)
        or parent_families & dev_families
    ):
        raise ValueError("parent train/dev identity contract is invalid")

    phone_manifest = json.loads(phone_manifest_path.read_text(encoding="utf-8"))
    phone_artifact = validate_phone_manifest(phone_manifest, phone_train_path)
    phone_raw = read_jsonl(phone_train_path)
    if any(row.get("source") != PHONE_SOURCE for row in phone_raw):
        raise ValueError("phone training artifact contains an unexpected source")
    phone_sft, excluded = convert_supported_rows(phone_raw)
    over_length: list[dict[str, Any]] = []
    if tokenizer is not None:
        retained: list[dict[str, Any]] = []
        for row in phone_sft:
            complete = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            full_tokens = len(
                tokenizer.tokenizer(complete, add_special_tokens=False)["input_ids"]
            )
            if full_tokens > max_length:
                over_length.append({"id": str(row["id"]), "full_tokens": full_tokens})
            else:
                retained.append(row)
        phone_sft = retained
    if not phone_sft or any(verdict(row) not in {"SAFE", "SCAM"} for row in phone_sft):
        raise ValueError("phone supplement has an invalid converted verdict contract")

    supplement_ids = [str(row.get("id", "")) for row in phone_sft]
    supplement_families = {str(row.get("family_id", "")) for row in phone_sft}
    if (
        not all(supplement_ids)
        or len(supplement_ids) != len(set(supplement_ids))
        or set(supplement_ids) & (set(parent_ids) | set(dev_ids))
        or supplement_families & (parent_families | dev_families)
    ):
        raise ValueError("phone supplement crosses a parent identity or family")

    reference_contracts: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for path in overlap_references:
        rows = read_jsonl(path)
        reference_rows.extend(rows)
        reference_contracts.append(
            {"path": str(path), "rows": len(rows), "sha256": file_sha256(path)}
        )
    held_overlaps = near_overlap_indices(phone_raw, reference_rows, 6) if reference_rows else set()
    if held_overlaps:
        raise ValueError("phone supplement near-overlaps a held evaluation reference")
    parent_overlaps = near_overlap_indices(phone_raw, sft_as_text_rows(parent_train), 6)
    if parent_overlaps:
        raise ValueError("phone supplement near-overlaps the Stage 3 fitting corpus")

    train = parent_train + sorted(phone_sft, key=lambda row: str(row["id"]))
    output_sft = output / "qwen_sft"
    output_sft.mkdir(parents=True)
    train_path = output_sft / "train.jsonl"
    dev_path = output_sft / "dev.jsonl"
    write_jsonl(train_path, train)
    shutil.copy2(parent_dev_path, dev_path)

    excluded_ids = sorted(str(row["id"]) for row in excluded)
    over_length_ids = sorted(str(row["id"]) for row in over_length)
    verdict_counts = Counter(verdict(row) for row in train)
    source_counts = Counter(str(row.get("source")) for row in train)
    manifest: dict[str, Any] = {
        "artifact_schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "schema_version": 25,
        "release_eligible": False,
        "publication_authorized": False,
        "parent": {
            "directory": str(parent),
            "manifest_sha256": file_sha256(parent_manifest_path),
            "sft_manifest_sha256": file_sha256(parent_sft_manifest_path),
            "train_sha256": file_sha256(parent_train_path),
            "dev_sha256": file_sha256(parent_dev_path),
            "full_replay_rows": len(parent_train),
        },
        "phone_supplement": {
            "manifest_path": str(phone_manifest_path),
            "manifest_sha256": file_sha256(phone_manifest_path),
            "source": PHONE_SOURCE,
            "repository": PHONE_REPOSITORY,
            "revision": PHONE_REVISION,
            "license": PHONE_LICENSE,
            "raw_train_path": str(phone_train_path),
            "raw_train_sha256": phone_artifact["sha256"],
            "raw_train_rows": len(phone_raw),
            "converted_rows": len(phone_sft),
            "excluded_unsupported_scam_rows": len(excluded),
            "excluded_ids_sha256": hashlib.sha256(
                "\n".join(excluded_ids).encode("utf-8")
            ).hexdigest(),
            "token_length_filter": {
                "enforced": tokenizer is not None,
                "model": QWEN_MODEL if tokenizer is not None else None,
                "revision": QWEN_REVISION if tokenizer is not None else None,
                "max_length": max_length,
                "over_length_rows_excluded": len(over_length),
                "over_length_ids_sha256": hashlib.sha256(
                    "\n".join(over_length_ids).encode("utf-8")
                ).hexdigest(),
            },
            "validation_rows_used_for_fitting": 0,
            "test_rows_read": 0,
            "test_predictions_opened": False,
            "source_metadata_used_as_model_input": False,
            "near_overlap_radius": 6,
            "near_overlap_with_parent_rows": 0,
            "near_overlap_with_held_rows": 0,
            "overlap_references": reference_contracts,
        },
        "selection": {
            "policy": "full immutable Stage 3 replay plus supported phone training rows once",
            "held_rows_used_for_fitting": 0,
            "primary_test_rows_used_for_fitting": 0,
            "bothbosu_rows_used_for_fitting": 0,
            "phone_validation_rows_used_for_fitting": 0,
            "phone_test_rows_used_for_fitting": 0,
        },
        "splits": {
            "train": {
                "rows": len(train),
                "families": len({str(row["family_id"]) for row in train}),
                "sha256": file_sha256(train_path),
                "verdicts": dict(sorted(verdict_counts.items())),
                "sources": dict(sorted(source_counts.items())),
            },
            "dev": {
                "rows": len(parent_dev),
                "families": len(dev_families),
                "sha256": file_sha256(dev_path),
                "byte_identical_to_parent": file_sha256(dev_path)
                == file_sha256(parent_dev_path),
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sft_manifest = {
        "artifact_schema_version": 1,
        "input_directory": str(output),
        "input_manifest_sha256": file_sha256(manifest_path),
        "policy": {
            "continuation_replay_only": True,
            "held_rows_used_for_fitting": 0,
            "phone_validation_rows_used_for_fitting": 0,
            "phone_test_rows_used_for_fitting": 0,
        },
        "splits": manifest["splits"],
    }
    (output_sft / "manifest.json").write_text(
        json.dumps(sft_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--phone-manifest", type=Path, required=True)
    parser.add_argument("--phone-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-reference", type=Path, action="append", default=[])
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--revision", default=QWEN_REVISION)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.model != QWEN_MODEL or args.revision != QWEN_REVISION:
        raise ValueError("Stage 7 requires the exact pinned Qwen model and revision")
    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    print(
        json.dumps(
            build(
                args.parent,
                args.phone_manifest,
                args.phone_train,
                args.output,
                overlap_references=tuple(args.overlap_reference),
                tokenizer=tokenizer,
                max_length=args.max_length,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
