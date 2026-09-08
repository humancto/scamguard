#!/usr/bin/env python3
"""Build a split-safe real robocall corpus from the FTC PPoNE release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

try:
    from scripts.build_dataset import (
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        category_for,
        clean_text,
        cluster_near_duplicates,
        has_strong_scam_evidence,
        make_row,
        normalized,
        privacy_normalize_real_text,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than the repo on sys.path.
    from build_dataset import (  # type: ignore[no-redef]
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        category_for,
        clean_text,
        cluster_near_duplicates,
        has_strong_scam_evidence,
        make_row,
        normalized,
        privacy_normalize_real_text,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )

from scamguard.metrics import file_sha256

SOURCE = "wspr_ncsu_ppone_robocalls"
SOURCE_REPOSITORY = "wspr-ncsu/robocall-audio-dataset"
SOURCE_REVISION = "5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272"
SOURCE_BLOB_SHA1 = "b06bd7f19af5e17c7ad915f40dbf12b3b321f140"
SOURCE_SHA256 = "8e253a4a652abfa92ac3766feff8da2bef16f45ab3f0f434f212a7aafc3effe5"
README_SHA256 = "c39a89a53b34a0e2736b0c8848c20e400d7b710894f5cc2f975759654c7c788b"
LICENSE_SHA256 = "8d4d526b30448a13ad7d5badaeae87b9453437f1aac9837e5eb4ba4593222e5e"
EXPECTED_HEADER = ["file_name", "language", "transcript", "case_details", "case_pdf"]
EXPECTED_ROWS = 1432
PARTITION_SALT = "scamguard-ppone-robocalls-v1"
LICENSE_NAME = "Public-Domain"


def partition(family_id: str) -> str:
    bucket = int(
        hashlib.sha256(f"{PARTITION_SALT}:{family_id}".encode()).hexdigest()[:8], 16
    ) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def reference_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in paths:
        candidates = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
        for path in candidates:
            if not path.is_file() or "quarantine" in path.name:
                continue
            rows.extend(row for row in read_jsonl(path) if isinstance(row.get("text"), str))
    return rows


def read_source(source: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("PPoNE metadata differs from the pinned repository revision")
    rows: list[dict[str, object]] = []
    privacy = Counter[str]()
    language_counts = Counter[str]()
    case_counts = Counter[str]()
    with source.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(f"unexpected PPoNE header: {reader.fieldnames!r}")
        for index, source_row in enumerate(reader, start=1):
            transcript = str(source_row["transcript"]).strip()
            filename = str(source_row["file_name"]).strip()
            language = str(source_row["language"]).strip().casefold()
            case = str(source_row["case_details"]).strip()
            if not transcript or not filename or not language or not case:
                raise ValueError(f"missing required PPoNE value at row {index}")
            language_counts[language] += 1
            case_counts[case] += 1
            if language != "en":
                continue
            privacy["rows_with_email"] += bool(EMAIL_RE.search(transcript))
            privacy["rows_with_phone_like"] += bool(PHONE_LIKE_RE.search(transcript))
            privacy["rows_with_long_digit_sequence"] += bool(LONG_DIGIT_RE.search(transcript))
            normalized_transcript = privacy_normalize_real_text(clean_text(transcript))
            label = (
                "SCAM"
                if has_strong_scam_evidence(normalized_transcript)
                else "UNCERTAIN"
            )
            row = make_row(
                text=normalized_transcript,
                label=label,
                source=SOURCE,
                source_label="publisher_suspected_illegal_robocall",
                license_name=LICENSE_NAME,
            )
            if row is None:
                raise ValueError(f"PPoNE transcript became empty at row {index}")
            row.update(
                {
                    "source_language": "English",
                    "source_repository": SOURCE_REPOSITORY,
                    "source_revision": SOURCE_REVISION,
                    "source_record_sha256": hashlib.sha256(filename.encode()).hexdigest(),
                    "source_case": case,
                    "provenance_class": "real_world_robocall_transcript",
                    "individual_scam_ground_truth": False,
                    "label_policy": (
                        "publisher_suspected_illegal_plus_strong_runtime_text_evidence"
                        if label == "SCAM"
                        else "publisher_suspected_illegal_without_strong_runtime_text_evidence"
                    ),
                    "privacy_normalization": "email_phone_and_long_digit_placeholders_v1",
                }
            )
            rows.append(row)
    total = sum(language_counts.values())
    if total != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} PPoNE rows, found {total}")
    return rows, {
        "source_rows": total,
        "language_counts": dict(sorted(language_counts.items())),
        "case_count": len(case_counts),
        "english_candidate_rows": len(rows),
        "privacy_findings_before_normalization": dict(privacy),
    }


def one_per_family(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["family_id"])].append(row)
    kept = [min(group, key=lambda row: str(row["id"])) for group in groups.values()]
    return kept, len(rows) - len(kept)


def deduplicate_and_relabel(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[normalized(str(row["text"]))].append(row)
    kept: list[dict[str, object]] = []
    for candidates in groups.values():
        representative = min(candidates, key=lambda row: str(row["id"]))
        text = str(representative["text"])
        label = "SCAM" if has_strong_scam_evidence(text) else "UNCERTAIN"
        kept.append(
            representative
            | {
                "label": label,
                "category": category_for(text) if label == "SCAM" else "NONE",
                "label_policy": (
                    "publisher_suspected_illegal_plus_strong_runtime_text_evidence"
                    if label == "SCAM"
                    else "publisher_suspected_illegal_without_strong_runtime_text_evidence"
                ),
            }
        )
    return kept, len(rows) - len(kept)


def build(
    source: Path,
    output: Path,
    report_path: Path | None = None,
    *,
    references: tuple[Path, ...] = (),
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PPoNE corpus: {output}")
    source_rows, source_stats = read_source(source)
    exact_rows, exact_dropped = deduplicate_and_relabel(source_rows)

    held_references = reference_rows(references)
    nonoverlap_rows, near_overlap = remove_near_overlaps(exact_rows, held_references)
    clustered, near_conflicts, near_stats = cluster_near_duplicates(nonoverlap_rows)
    conflict_rows = [
        row
        for conflict in near_conflicts
        for row in conflict["candidates"]  # type: ignore[index]
    ]
    representatives, repeated_template_rows = one_per_family(clustered)
    rows = [row | {"split": partition(str(row["family_id"]))} for row in representatives]
    rows.sort(key=lambda row: str(row["id"]))

    if any(
        EMAIL_RE.search(str(row["text"]))
        or PHONE_LIKE_RE.search(str(row["text"]))
        or LONG_DIGIT_RE.search(str(row["text"]))
        for row in rows + conflict_rows
    ):
        raise ValueError("privacy-like value survived PPoNE normalization")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("PPoNE admitted IDs are not unique")

    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    output.mkdir(parents=True)
    artifacts: dict[str, object] = {}
    for split, values in split_rows.items():
        path = output / f"ppone_{split}.jsonl"
        write_jsonl(path, values)
        artifacts[split] = {
            "path": str(path),
            "rows": len(values),
            "sha256": file_sha256(path),
        }
    quarantine_path = output / "ppone_near_label_conflicts_quarantine.jsonl"
    write_jsonl(quarantine_path, sorted(conflict_rows, key=lambda row: str(row["id"])))
    artifacts["near_label_conflicts_quarantine"] = {
        "path": str(quarantine_path),
        "rows": len(conflict_rows),
        "sha256": file_sha256(quarantine_path),
    }

    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "metadata_git_blob_sha1": SOURCE_BLOB_SHA1,
            "metadata_sha256": SOURCE_SHA256,
            "publisher_readme_sha256": README_SHA256,
            "repository_license_sha256": LICENSE_SHA256,
            "rights": (
                "publisher README at the pinned revision declares the data itself public domain; "
                "the descriptive document is CC-BY-ND-4.0"
            ),
            "collection": (
                "real-world automated or semi-automated calls published through FTC Project "
                "Point of No Entry enforcement materials"
            ),
        },
        "policy": {
            "positive_or_suspicious_source_only": True,
            "individual_scam_ground_truth": False,
            "scam_label_requires_strong_runtime_text_evidence": True,
            "otherwise_label": "UNCERTAIN",
            "safe_rows_supplied_by_this_source": 0,
            "one_representative_per_near_template_family": True,
            "partition": "near-template family SHA-256 salted 70/15/15",
            "validation_used_for_fitting_or_threshold": False,
            "test_prediction_sealed_until_candidate_freeze": True,
            "independent_human_label_review_complete": False,
            "raw_text_written_to_manifest": False,
            "non_english_rows_admitted": 0,
        },
        "counts": source_stats
        | {
            "exact_duplicate_rows_removed": exact_dropped,
            "near_overlaps_with_references_removed": near_overlap,
            "near_label_conflict_rows_quarantined": len(conflict_rows),
            "same_label_near_template_rows_removed": repeated_template_rows,
            "admitted_rows": len(rows),
            "admitted_labels": dict(Counter(str(row["label"]) for row in rows)),
            "partition_rows": {split: len(values) for split, values in split_rows.items()},
            "reference_rows": len(held_references),
        },
        "near_template_stats": near_stats,
        "artifacts": artifacts,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("data/raw/ppone_robocall_metadata.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/external/ppone_robocalls")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/source-audits/ppone-robocalls.json"),
    )
    parser.add_argument("--reference", type=Path, action="append", default=[])
    args = parser.parse_args()
    build(args.source, args.output, args.report, references=tuple(args.reference))


if __name__ == "__main__":
    main()
