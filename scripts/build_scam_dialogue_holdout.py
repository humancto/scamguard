#!/usr/bin/env python3
"""Build a deduplicated external diagnostic from synthetic scam phone dialogues."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

try:
    from scripts.build_dataset import (
        URL_RE,
        cluster_near_duplicates,
        deduplicate,
        make_row,
        normalized,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than the repo on sys.path.
    from build_dataset import (  # type: ignore[no-redef]
        URL_RE,
        cluster_near_duplicates,
        deduplicate,
        make_row,
        normalized,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )

SOURCE_REVISION = "321b961b5ae353e19ed479b960658dcd223d5e06"
SOURCE_SHA256 = "fe8a8fa0aa2b8afb0b0a672fb7f9739b323cb6dd12064f786a68c2a1f49a4e0b"
TYPE_TO_CATEGORY = {
    "ssn": "IDENTITY_IMPERSONATION",
    "refund": "FINANCIAL",
    "support": "CREDENTIAL_THEFT",
    "reward": "OPPORTUNITY",
}
SAFE_TYPES = {"delivery", "insurance", "telemarketing", "wrong"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostic_partition(family_id: str) -> str:
    bucket = int(
        hashlib.sha256(f"scam-dialogue-diagnostic-v1:{family_id}".encode()).hexdigest()[:8],
        16,
    ) % 100
    return "validation" if bucket < 20 else "ood"


def read_dialogues(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["dialogue", "type", "label"]:
            raise ValueError(f"unexpected scam-dialogue header: {reader.fieldnames!r}")
        for index, source_row in enumerate(reader, start=1):
            source_label = str(source_row["label"]).strip()
            dialogue_type = str(source_row["type"]).strip().casefold()
            if source_label not in {"0", "1"}:
                raise ValueError(f"unexpected label at source row {index}: {source_label!r}")
            if dialogue_type not in TYPE_TO_CATEGORY and dialogue_type not in SAFE_TYPES:
                raise ValueError(f"unexpected type at source row {index}: {dialogue_type!r}")
            label = "SCAM" if source_label == "1" else "SAFE"
            if (label == "SCAM") != (dialogue_type in TYPE_TO_CATEGORY):
                raise ValueError(f"type/label conflict at source row {index}")
            text = URL_RE.sub("<URL>", str(source_row["dialogue"]))
            row = make_row(
                text=text,
                label=label,
                source="bothbosu_scam_dialogue",
                source_label=f"{source_label}:{dialogue_type}",
                license_name="Apache-2.0",
            )
            if row is None:
                continue
            row.update(
                {
                    "category": TYPE_TO_CATEGORY.get(dialogue_type, "NONE"),
                    "source_language": "English",
                    "source_record_id": str(index),
                    "source_dialogue_type": dialogue_type,
                    "upstream_generation_model": "meta-llama-3-70b-instruct",
                    "label_policy": "upstream_synthetic_binary_label",
                }
            )
            yield row


def reference_rows(data: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in (
        "train",
        "dev",
        "test",
        "ood_financial",
        "ood_wspr",
        "forum_validation",
        "ood_forum",
        "ood_azsc",
    ):
        rows.extend(read_jsonl(data / f"{split}.jsonl"))
    return rows


def build(source: Path, data: Path, output: Path) -> dict[str, object]:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("scam-dialogue source hash differs from the pinned publisher artifact")

    source_rows = list(read_dialogues(source))
    exact_rows, exact_dropped, exact_conflicts = deduplicate(source_rows)
    clustered, near_conflicts, near_stats = cluster_near_duplicates(exact_rows)
    representatives: dict[str, dict[str, object]] = {}
    for row in clustered:
        family_id = str(row["family_id"])
        candidate = row | {
            "split": "ood",
            "is_synthetic": True,
            "synthetic_method": "upstream_llama3_70b_multi_turn_generation",
        }
        current = representatives.get(family_id)
        if current is None or str(candidate["id"]) < str(current["id"]):
            representatives[family_id] = candidate

    candidates = list(representatives.values())
    references = reference_rows(data)
    reference_keys = {normalized(str(row["text"])) for row in references}
    candidates = [row for row in candidates if normalized(str(row["text"])) not in reference_keys]
    exact_overlaps_removed = len(representatives) - len(candidates)
    candidates, near_overlaps_removed = remove_near_overlaps(candidates, references)
    candidates.sort(key=lambda row: str(row["id"]))

    ids = [str(row["id"]) for row in candidates]
    families = [str(row["family_id"]) for row in candidates]
    if len(ids) != len(set(ids)) or len(families) != len(set(families)):
        raise ValueError("scam-dialogue diagnostic is not one-row-per-id-and-family")
    if {str(row["label"]) for row in candidates} != {"SAFE", "SCAM"}:
        raise ValueError("scam-dialogue diagnostic must retain both SAFE and SCAM rows")

    partitions = {
        "validation": [
            row | {"split": "validation"}
            for row in candidates
            if diagnostic_partition(str(row["family_id"])) == "validation"
        ],
        "ood": [
            row | {"split": "ood"}
            for row in candidates
            if diagnostic_partition(str(row["family_id"])) == "ood"
        ],
    }
    for split, split_rows in partitions.items():
        if {str(row["label"]) for row in split_rows} != {"SAFE", "SCAM"}:
            raise ValueError(f"scam-dialogue {split} must retain SAFE and SCAM rows")
    output.mkdir(parents=True, exist_ok=True)
    validation_artifact = output / "scam_dialogue_validation.jsonl"
    ood_artifact = output / "ood_scam_dialogue.jsonl"
    write_jsonl(validation_artifact, partitions["validation"])
    write_jsonl(ood_artifact, partitions["ood"])
    write_jsonl(output / "quarantine_label_conflicts.jsonl", exact_conflicts + near_conflicts)
    manifest: dict[str, object] = {
        "diagnostic_schema_version": 1,
        "purpose": (
            "post-schema-v9 multi-turn synthetic diagnostic; excluded from fitting and "
            "thresholding; may inform candidate selection"
        ),
        "source": {
            "repository": "https://huggingface.co/datasets/BothBosu/scam-dialogue",
            "revision": SOURCE_REVISION,
            "license": "Apache-2.0",
            "raw_sha256": SOURCE_SHA256,
            "generation_model_reported_by_publisher": "meta-llama-3-70b-instruct",
        },
        "policy": {
            "used_for_fitting": False,
            "used_for_threshold": False,
            "counted_as_real_data": False,
            "validation_may_inform_candidate_selection": True,
            "ood_prediction_sealed_until_candidate_freeze": True,
            "partition": "sha256(scam-dialogue-diagnostic-v1:family_id), 20/80",
            "privacy_normalization": "URLs and phone/account-like values replaced",
            "near_template_hamming_max": 6,
            "one_representative_per_family": True,
            "independent_human_label_review_complete": False,
        },
        "counts": {
            "source_rows": len(source_rows),
            "source_labels": dict(Counter(str(row["label"]) for row in source_rows)),
            "source_types": dict(Counter(str(row["source_dialogue_type"]) for row in source_rows)),
            "exact_duplicates_removed": exact_dropped,
            "exact_conflict_groups_quarantined": len(exact_conflicts),
            "near_conflict_groups_quarantined": len(near_conflicts),
            "family_representatives_before_overlap": len(representatives),
            "exact_overlaps_removed": exact_overlaps_removed,
            "near_overlaps_removed": near_overlaps_removed,
            "final_rows": len(candidates),
            "final_labels": dict(Counter(str(row["label"]) for row in candidates)),
            "final_types": dict(
                Counter(str(row["source_dialogue_type"]) for row in candidates)
            ),
            "validation_rows": len(partitions["validation"]),
            "validation_labels": dict(
                Counter(str(row["label"]) for row in partitions["validation"])
            ),
            "ood_rows": len(partitions["ood"]),
            "ood_labels": dict(Counter(str(row["label"]) for row in partitions["ood"])),
        },
        "near_template_stats": near_stats,
        "artifacts": {
            "validation": {
                "path": str(validation_artifact),
                "sha256": sha256(validation_artifact),
            },
            "ood": {"path": str(ood_artifact), "sha256": sha256(ood_artifact)},
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/scam_dialogue_all.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/external/scam_dialogue"))
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(
            f"missing scam-dialogue source; run scripts/fetch_datasets.py: {args.source}"
        )
    print(json.dumps(build(args.source, args.data, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
