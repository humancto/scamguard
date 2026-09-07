#!/usr/bin/env python3
"""Normalize and contamination-control the pinned phone-scam dialogue source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from scamguard.metrics import file_sha256
from scamguard.privacy import CONTEXTUAL_PRIVACY_REVISION, mask_contextual_sensitive_values

try:
    from scripts.build_dataset import (
        family_skeleton,
        privacy_normalize_real_text,
        simhash64,
        simhash_bands,
    )
    from scripts.build_schema19_call_windows import read_jsonl, write_jsonl
    from scripts.build_schema23_evidence_compaction import remove_reference_overlap_families
    from scripts.fetch_phone_scam_synthetic import (
        LICENSE,
        REPOSITORY,
        REVISION,
        verify_files,
        verify_receipt,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_dataset import (  # type: ignore[no-redef]
        family_skeleton,
        privacy_normalize_real_text,
        simhash64,
        simhash_bands,
    )
    from build_schema19_call_windows import read_jsonl, write_jsonl  # type: ignore[no-redef]
    from build_schema23_evidence_compaction import (  # type: ignore[no-redef]
        remove_reference_overlap_families,
    )
    from fetch_phone_scam_synthetic import (  # type: ignore[no-redef]
        LICENSE,
        REPOSITORY,
        REVISION,
        verify_files,
        verify_receipt,
    )

SOURCE = "shakeleoatmeal_phone_scam_synthetic"
EXPECTED_COLUMNS = (
    "dialogue",
    "type",
    "label",
    "length_category",
    "variation_style",
    "target_exchanges",
    "batch_num",
)
EXPECTED_COUNTS = {"train": 1259, "validation": 361, "test": 180}
EXPECTED_LABEL_COUNTS = {
    "train": {0: 630, 1: 629},
    "validation": {0: 180, 1: 181},
    "test": {0: 90, 1: 90},
}
TYPE_TO_CATEGORY = {
    "refund": "FINANCIAL",
    "ssn": "CREDENTIAL_THEFT",
    "support": "CREDENTIAL_THEFT",
}
PROMPT_ARTIFACT_RE = re.compile(
    r"^(?:here is|below is|sure[,! ]|certainly[,! ])\s+",
    re.IGNORECASE,
)
SPLIT_PRIORITY = {"test": 0, "validation": 1, "train": 2}


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def read_receipt(source: Path) -> dict[str, object]:
    receipt_path = source / "download_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing phone-scam download receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not verify_receipt(source, receipt):
        raise RuntimeError("phone-scam source differs from its download receipt")
    return receipt


def source_row(record: dict[str, Any], split: str, index: int) -> dict[str, object] | None:
    dialogue = str(record["dialogue"]).strip()
    if PROMPT_ARTIFACT_RE.match(dialogue):
        return None
    broad_normalized = privacy_normalize_real_text(dialogue)
    normalized = mask_contextual_sensitive_values(broad_normalized)
    label = int(record["label"])
    dialogue_type = str(record["type"])
    if label not in {0, 1} or dialogue_type not in TYPE_TO_CATEGORY:
        raise ValueError("phone-scam source contains an unknown label or dialogue type")
    identity = f"{split}:{index}:{hashlib.sha256(dialogue.encode()).hexdigest()}"
    return {
        "id": "phone-scam-synthetic-" + short_hash(identity),
        "text": normalized.text,
        "label": "SCAM" if label else "SAFE",
        "category": TYPE_TO_CATEGORY[dialogue_type] if label else "NONE",
        "source": SOURCE,
        "source_label": f"publisher_binary:{label}",
        "license": LICENSE,
        "split": split,
        "family_id": "phone-scam-source-" + short_hash(family_skeleton(normalized.text)),
        "is_synthetic": True,
        "synthetic_method": "publisher_generated_multi_turn_phone_dialogue",
        "upstream_generation_model": "not_disclosed_by_publisher",
        "source_language": "English",
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "source_record_index": index,
        "source_dialogue_type": dialogue_type,
        "source_length_category": str(record["length_category"]),
        "source_variation_style": str(record["variation_style"]),
        "source_target_exchanges": str(record["target_exchanges"]),
        "source_batch_number": int(record["batch_num"]),
        "label_policy": "publisher_synthetic_binary_label",
        "privacy_normalization": CONTEXTUAL_PRIVACY_REVISION,
        "broad_privacy_values_replaced": broad_normalized != dialogue,
        "privacy_values_replaced": normalized.changed,
        "metadata_used_as_model_input": False,
    }


def read_splits(source: Path) -> tuple[dict[str, list[dict[str, object]]], dict[str, int]]:
    output: dict[str, list[dict[str, object]]] = {}
    artifact_rejections: dict[str, int] = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        frame = pd.read_parquet(source / f"{split}.parquet")
        if tuple(frame.columns) != EXPECTED_COLUMNS or len(frame) != expected_count:
            raise ValueError(f"phone-scam {split} schema or row count differs")
        if frame.isna().any().any():
            raise ValueError(f"phone-scam {split} contains null fields")
        label_counts = {
            int(key): int(value) for key, value in frame["label"].value_counts().items()
        }
        if label_counts != EXPECTED_LABEL_COUNTS[split]:
            raise ValueError(f"phone-scam {split} label counts differ")
        rows = []
        rejected = 0
        for index, record in enumerate(frame.to_dict(orient="records")):
            row = source_row(record, split, index)
            if row is None:
                rejected += 1
            else:
                rows.append(row)
        output[split] = rows
        artifact_rejections[split] = rejected
    return output, artifact_rejections


def one_per_near_family(
    rows_by_split: dict[str, list[dict[str, object]]], max_hamming: int = 6
) -> tuple[dict[str, list[dict[str, object]]], dict[str, int]]:
    rows = [row for split in ("test", "validation", "train") for row in rows_by_split[split]]
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    signatures: list[int] = []
    buckets: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        signature = simhash64(family_skeleton(str(row["text"])))
        signatures.append(signature)
        candidates: set[int] = set()
        for key in simhash_bands(signature, max_hamming=max_hamming):
            candidates.update(buckets[key])
        for candidate in candidates:
            if (signature ^ signatures[candidate]).bit_count() <= max_hamming:
                union(index, candidate)
        for key in simhash_bands(signature, max_hamming=max_hamming):
            buckets[key].append(index)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    kept: dict[str, list[dict[str, object]]] = {split: [] for split in rows_by_split}
    conflicts = 0
    collapsed = 0
    cross_split = 0
    for members in groups.values():
        labels = {str(rows[index]["label"]) for index in members}
        splits = {str(rows[index]["split"]) for index in members}
        if len(labels) > 1:
            conflicts += 1
            continue
        if len(splits) > 1:
            cross_split += 1
        representative = min(
            members,
            key=lambda index: (
                SPLIT_PRIORITY[str(rows[index]["split"])],
                str(rows[index]["id"]),
            ),
        )
        family_id = "phone-scam-near-" + min(
            short_hash(family_skeleton(str(rows[index]["text"]))) for index in members
        )
        selected = rows[representative] | {"family_id": family_id}
        kept[str(selected["split"])].append(selected)
        collapsed += len(members) - 1
    for split in kept:
        kept[split].sort(key=lambda row: str(row["id"]))
    return kept, {
        "near_hamming_max": max_hamming,
        "source_rows": len(rows),
        "near_families": len(groups),
        "mixed_label_families_quarantined": conflicts,
        "cross_split_near_families": cross_split,
        "same_label_rows_collapsed": collapsed,
        "retained_rows": sum(len(values) for values in kept.values()),
    }


def reference_rows(directories: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for directory in directories:
        for path in sorted(directory.glob("*.jsonl")):
            if "quarantine" not in path.name:
                rows.extend(
                    row for row in read_jsonl(path) if {"id", "family_id", "text"} <= set(row)
                )
    if not rows:
        raise ValueError("phone-scam overlap audit has no reference rows")
    return rows


def remove_reference_overlaps(
    rows_by_split: dict[str, list[dict[str, object]]], references: list[dict[str, object]]
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    retained: dict[str, list[dict[str, object]]] = {}
    reports: dict[str, object] = {}
    protected = list(references)
    for split in ("test", "validation", "train"):
        kept, report = remove_reference_overlap_families(rows_by_split[split], protected)
        retained[split] = sorted(kept, key=lambda row: str(row["id"]))
        reports[split] = report
        protected.extend(kept)
    return retained, reports


def build(source: Path, reference_directories: list[Path], output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite phone-scam derivative: {output}")
    receipt = read_receipt(source)
    source_rows, artifact_rejections = read_splits(source)
    clustered, near_stats = one_per_near_family(source_rows)
    controlled, overlap_stats = remove_reference_overlaps(
        clustered, reference_rows(reference_directories)
    )
    all_ids = [str(row["id"]) for rows in controlled.values() for row in rows]
    all_families = [str(row["family_id"]) for rows in controlled.values() for row in rows]
    if len(all_ids) != len(set(all_ids)) or len(all_families) != len(set(all_families)):
        raise ValueError("phone-scam derivative is not one row per ID and near family")

    output.mkdir(parents=True)
    artifacts: dict[str, object] = {}
    for split, rows in controlled.items():
        path = output / f"{split}.jsonl"
        write_jsonl(path, rows)
        artifacts[split] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": file_sha256(path),
            "labels": dict(Counter(str(row["label"]) for row in rows)),
        }
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "source": SOURCE,
        "repository": REPOSITORY,
        "revision": REVISION,
        "license_declared_by_publisher": LICENSE,
        "download_receipt_sha256": file_sha256(source / "download_receipt.json"),
        "source_files": verify_files(source),
        "policy": {
            "synthetic": True,
            "counted_as_real_call_data": False,
            "publisher_split_boundary_preserved": True,
            "train_used_for_fitting": True,
            "validation_may_inform_candidate_selection": True,
            "test_prediction_sealed_until_candidate_freeze": True,
            "source_metadata_used_as_model_input": False,
            "variation_style_is_perfectly_label_confounded": True,
            "validation_and_test_are_not_final_sota_evidence": True,
            "generation_prompt_artifacts_quarantined": True,
            "privacy_normalized_before_overlap_control": True,
            "broad_email_phone_and_account_values_normalized": True,
            "direct_reddit_scrape": False,
        },
        "artifact_rejections": artifact_rejections,
        "near_duplicate_control": near_stats,
        "reference_overlap_control": overlap_stats,
        "reference_directories": [str(path) for path in reference_directories],
        "artifacts": artifacts,
        "download_receipt": receipt,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/phone_scam_synthetic"))
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[
            Path("data/experiments/schema24-annotated-hard-negatives/processed"),
            Path("data/external/scam_dialogue"),
        ],
    )
    parser.add_argument("--output", type=Path, default=Path("data/external/phone_scam_synthetic"))
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.reference, args.output), indent=2))


if __name__ == "__main__":
    main()
