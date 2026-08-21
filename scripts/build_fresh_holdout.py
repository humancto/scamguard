#!/usr/bin/env python3
"""Build the sealed schema-v8 holdout from a newly sourced real dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from scripts.build_dataset import (
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        clean_text,
        cluster_near_duplicates,
        deduplicate,
        make_row,
        normalized,
        privacy_normalize_real_text,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )
except ModuleNotFoundError:  # Direct `python scripts/build_fresh_holdout.py` execution.
    from build_dataset import (  # type: ignore[no-redef]
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        clean_text,
        cluster_near_duplicates,
        deduplicate,
        make_row,
        normalized,
        privacy_normalize_real_text,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_moz_smishing(path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    privacy_counts = Counter[str]()
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for source_row in csv.DictReader(handle):
            raw_text = clean_text(source_row.get("text", ""))
            source_label = clean_text(source_row.get("label", "")).casefold()
            if not raw_text or source_label not in {"legitimate", "smishing"}:
                continue
            privacy_counts["rows_with_email_before_normalization"] += bool(
                EMAIL_RE.search(raw_text)
            )
            privacy_counts["rows_with_phone_like_before_normalization"] += bool(
                PHONE_LIKE_RE.search(raw_text)
            )
            privacy_counts["rows_with_long_digits_before_normalization"] += bool(
                LONG_DIGIT_RE.search(raw_text)
            )
            text = privacy_normalize_real_text(raw_text)
            label = "SAFE" if source_label == "legitimate" else "SCAM"
            row = make_row(
                text=text,
                label=label,
                source="moz_smishing",
                source_label=source_label,
                license_name="CreativeML-OpenRAIL-M",
            )
            if row is None:
                continue
            row.update(
                {
                    "split": "test",
                    "category": "NONE" if label == "SAFE" else "FINANCIAL",
                    "label_policy": (
                        "source_crowdsourced_legitimate"
                        if label == "SAFE"
                        else "source_crowdsourced_mobile_money_smishing"
                    ),
                    "source_language": "Portuguese (Mozambique)",
                    "source_provenance": "crowdsourced_mobile_money_users",
                    "privacy_policy": "all rows normalized by ScamGuard before deduplication",
                }
            )
            rows.append(row)
    return rows, dict(privacy_counts)


def reference_files(directory: Path, output: Path) -> list[Path]:
    excluded = {output.name, f"{output.stem}_quarantine.jsonl"}
    return [
        path
        for path in sorted(directory.glob("*.jsonl"))
        if path.name not in excluded and "quarantine" not in path.name
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/moz_smishing.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/primary_test_v8.jsonl")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/primary_test_v8.manifest.json")
    )
    parser.add_argument("--reference-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    raw_rows, privacy_counts = read_moz_smishing(args.source)
    exact_deduped, exact_duplicates_removed, exact_conflicts = deduplicate(raw_rows)
    clustered, near_conflicts, near_stats = cluster_near_duplicates(exact_deduped)

    representatives: dict[str, dict[str, object]] = {}
    for row in clustered:
        family_id = str(row["family_id"])
        current = representatives.get(family_id)
        if current is None or str(row["id"]) < str(current["id"]):
            representatives[family_id] = row | {"split": "test"}
    candidate_rows = list(representatives.values())

    paths = reference_files(args.reference_dir, args.output)
    references = [row for path in paths for row in read_jsonl(path)]
    reference_keys = {normalized(str(row["text"])) for row in references}
    exact_overlap_rows = sum(
        normalized(str(row["text"])) in reference_keys for row in candidate_rows
    )
    candidate_rows = [
        row for row in candidate_rows if normalized(str(row["text"])) not in reference_keys
    ]
    candidate_rows, near_overlap_rows = remove_near_overlaps(candidate_rows, references)
    candidate_rows.sort(key=lambda row: str(row["id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_count = write_jsonl(args.output, candidate_rows)
    quarantine_path = args.output.with_name(f"{args.output.stem}_quarantine.jsonl")
    quarantine_count = write_jsonl(quarantine_path, exact_conflicts + near_conflicts)
    manifest = {
        "schema_version": 8,
        "benchmark_state": "SEALED_MODEL_PREDICTIONS_NOT_RUN",
        "purpose": "newly sourced primary robustness holdout after schema-v6 test observation",
        "source": {
            "key": "moz_smishing",
            "revision": "1092f9d9a545b29ae6be030ee9713b615fc2d987",
            "raw_sha256": sha256(args.source),
            "citation": "https://doi.org/10.18653/v1/2025.africanlp-1.23",
            "publisher_license_tag": "creativeml-openrail-m",
            "license_scope_status": "DATASET_SPECIFIC_CLARIFICATION_REQUIRED",
            "training_allowed_by_project": False,
            "public_redistribution_allowed_by_project": False,
            "local_evaluation_only": True,
        },
        "policy": {
            "candidate_predictions_used_during_build": False,
            "all_rows_privacy_normalized_before_ids_or_deduplication": True,
            "exact_and_near_template_conflicts_quarantined": True,
            "one_representative_per_near_template_family": True,
            "near_template_hamming_max": 6,
            "overlap_removed_against_every_existing_processed_benchmark": True,
        },
        "counts": {
            "source_rows": len(raw_rows),
            "exact_duplicates_removed": exact_duplicates_removed,
            "exact_conflicting_groups_quarantined": len(exact_conflicts),
            "near_conflicting_groups_quarantined": len(near_conflicts),
            "near_conflicting_rows_quarantined": near_stats[
                "near_template_rows_quarantined"
            ],
            "same_label_near_template_rows_collapsed": len(clustered) - len(representatives),
            "exact_overlap_rows_removed": exact_overlap_rows,
            "near_overlap_rows_removed_after_exact": near_overlap_rows,
            "quarantine_records": quarantine_count,
            "final_rows": output_count,
            "final_labels": dict(Counter(str(row["label"]) for row in candidate_rows)),
        },
        "privacy": privacy_counts,
        "reference_files": {
            path.name: {"sha256": sha256(path), "rows": sum(1 for _ in read_jsonl(path))}
            for path in paths
        },
        "artifact": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "quarantine_path": str(quarantine_path),
            "quarantine_sha256": sha256(quarantine_path),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
