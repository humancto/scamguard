#!/usr/bin/env python3
"""Audit and partition real scam-call transcripts without exposing report text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

try:
    from scripts.build_dataset import (
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
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
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        cluster_near_duplicates,
        deduplicate,
        make_row,
        normalized,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )

from scamguard.metrics import file_sha256

SOURCE_SHA256 = "3f67497736e9421c2f6e59efc46c129006419d40fc752cbb981042940384cedd"
SOURCE_VERSION = 2
EXPECTED_MEMBERS = {"FullTranscriptData.csv", "FullTranscriptData.xlsx"}
EXPECTED_HEADER = ["ID", "Source", "Content", "Char_Len"]
PARTITION_SALT = "scamguard-youtube-scam-calls-v1"
MAX_WINDOW_CHARS = 425


def archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member: {member.filename}")
        if member.is_dir():
            continue
        if path.name in members:
            raise ValueError(f"duplicate archive basename: {path.name}")
        members[path.name] = member
    if set(members) != EXPECTED_MEMBERS:
        raise ValueError(f"archive members differ from pinned version: {sorted(members)}")
    return members


def clip_prefix(text: str) -> str:
    if len(text) <= MAX_WINDOW_CHARS:
        return text
    candidate = text[:MAX_WINDOW_CHARS]
    return candidate.rsplit(" ", 1)[0].strip()


def clip_recent(text: str) -> str:
    if len(text) <= MAX_WINDOW_CHARS:
        return text
    candidate = text[-MAX_WINDOW_CHARS:]
    _, separator, remainder = candidate.partition(" ")
    return (remainder if separator else candidate).strip()


def transcript_windows(text: str) -> list[tuple[str, str]]:
    candidates = [("early", clip_prefix(text)), ("recent", clip_recent(text))]
    unique: dict[str, tuple[str, str]] = {}
    for kind, candidate in candidates:
        key = normalized(candidate)
        if len(candidate) >= 80 and key:
            unique.setdefault(key, (kind, candidate))
    return list(unique.values())


def reference_rows(data: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(data.glob("*.jsonl")):
        if "quarantine" not in path.name:
            rows.extend(read_jsonl(path))
    return rows


def partition(family_id: str) -> str:
    value = int(
        hashlib.sha256(f"{PARTITION_SALT}:{family_id}".encode()).hexdigest()[:8], 16
    ) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "validation"
    return "ood"


def connect_source_families(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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

    by_source: defaultdict[str, list[int]] = defaultdict(list)
    by_template: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_source[str(row["source_group_hash"])].append(index)
        by_template[str(row["family_id"])].append(index)
    for group in (*by_source.values(), *by_template.values()):
        for index in group[1:]:
            union(group[0], index)

    components: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    result = list(rows)
    for members in components.values():
        component_keys = sorted(
            {
                str(rows[index]["source_group_hash"])
                for index in members
            }
            | {str(rows[index]["family_id"]) for index in members}
        )
        family_id = "youtube-call-" + hashlib.sha256(
            "|".join(component_keys).encode()
        ).hexdigest()[:16]
        split = partition(family_id)
        for index in members:
            result[index] = result[index] | {"family_id": family_id, "split": split}
    return result


def read_source(source: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    privacy_before: Counter[str] = Counter()
    source_rows: list[dict[str, object]] = []
    raw_source_groups: set[str] = set()
    length_values: list[int] = []
    with zipfile.ZipFile(source) as archive:
        members = archive_members(archive)
        raw = archive.read(members["FullTranscriptData.csv"])
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(f"unexpected source header: {reader.fieldnames!r}")
        for index, source_row in enumerate(reader, start=1):
            source_id = str(source_row["ID"]).strip()
            source_group = str(source_row["Source"]).strip()
            raw_content = str(source_row["Content"])
            content = raw_content.strip()
            if not source_id or not source_group or not content:
                raise ValueError(f"missing required value at source row {index}")
            if int(str(source_row["Char_Len"])) != len(raw_content):
                raise ValueError(f"Char_Len mismatch at source row {index}")
            raw_source_groups.add(source_group)
            length_values.append(len(content))
            privacy_before["email_like_rows"] += bool(EMAIL_RE.search(content))
            privacy_before["phone_like_rows"] += bool(PHONE_LIKE_RE.search(content))
            privacy_before["long_digit_rows"] += bool(LONG_DIGIT_RE.search(content))
            source_group_hash = hashlib.sha256(source_group.casefold().encode()).hexdigest()[:16]
            for window_kind, window in transcript_windows(content):
                row = make_row(
                    text=window,
                    label="SCAM",
                    source="youtube_scam_calls_cc0",
                    source_label="publisher_scam_call",
                    license_name="CC0-1.0",
                )
                if row is None:
                    continue
                row.update(
                    {
                        "source_record_id": source_id,
                        "source_group_hash": source_group_hash,
                        "source_window": window_kind,
                        "source_language": "English",
                        "provenance_class": "real_scam_call_or_autodialer_transcript",
                        "naturally_occurring_scammer_language": True,
                        "scambaiter_interaction_possible": True,
                        "label_policy": "publisher_positive_only_scam_call_collection",
                        "privacy_normalization": (
                            "publisher PII removal plus ScamGuard email and phone normalization"
                        ),
                    }
                )
                source_rows.append(row)
    stats: dict[str, object] = {
        "source_rows": len(length_values),
        "unique_upstream_source_groups": len(raw_source_groups),
        "source_length_min": min(length_values),
        "source_length_max": max(length_values),
        "privacy_like_counts_before_normalization": dict(privacy_before),
        "candidate_windows": len(source_rows),
    }
    return source_rows, stats


def build(
    source: Path,
    data: Path,
    output: Path,
    report_path: Path | None = None,
) -> dict[str, object]:
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("YouTube scam-call archive differs from pinned Kaggle version 2")
    source_rows, source_stats = read_source(source)
    exact_rows, exact_dropped, conflicts = deduplicate(source_rows)
    references = reference_rows(data)
    reference_keys = {normalized(str(row["text"])) for row in references}
    exact_overlap = sum(normalized(str(row["text"])) in reference_keys for row in exact_rows)
    exact_rows = [
        row for row in exact_rows if normalized(str(row["text"])) not in reference_keys
    ]
    nonoverlap_rows, near_overlap = remove_near_overlaps(exact_rows, references)
    clustered, near_conflicts, near_stats = cluster_near_duplicates(nonoverlap_rows)
    rows = connect_source_families(clustered)
    rows.sort(key=lambda row: str(row["id"]))

    if conflicts or near_conflicts:
        raise ValueError("positive-only source unexpectedly produced a label conflict")
    if any(
        EMAIL_RE.search(str(row["text"]))
        or PHONE_LIKE_RE.search(str(row["text"]))
        or LONG_DIGIT_RE.search(str(row["text"]))
        for row in rows
    ):
        raise ValueError("privacy-like value survived transcript normalization")

    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "ood")
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, object] = {}
    filenames = {
        "train": "youtube_scam_train.jsonl",
        "validation": "youtube_scam_validation.jsonl",
        "ood": "youtube_scam_ood.jsonl",
    }
    for split, filename in filenames.items():
        path = output / filename
        write_jsonl(path, split_rows[split])
        artifacts[split] = {"path": str(path), "sha256": file_sha256(path)}

    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "source": {
            "dataset": "rivalcults/youtube-scam-phone-call-transcripts",
            "version": SOURCE_VERSION,
            "license": "CC0-1.0",
            "archive_sha256": SOURCE_SHA256,
            "collection": (
                "manually corrected partial YouTube transcripts of scammer/scambaiter calls "
                "and some autodialer messages"
            ),
        },
        "policy": {
            "positive_only": True,
            "counted_as_real_scam_call_derived": True,
            "counted_as_ordinary_victim_calls": False,
            "train_used_for_fitting": True,
            "validation_used_for_fitting_or_threshold": False,
            "validation_may_inform_candidate_selection": True,
            "ood_prediction_sealed_until_candidate_freeze": True,
            "partition": "source-and-near-template connected family, deterministic 70/15/15",
            "window_policy": (
                f"up to early and recent whitespace-complete {MAX_WINDOW_CHARS}-character windows"
            ),
            "independent_human_label_review_complete": False,
            "raw_text_written_to_manifest": False,
        },
        "counts": source_stats
        | {
            "exact_duplicate_windows_removed": exact_dropped,
            "exact_overlaps_with_existing_data_removed": exact_overlap,
            "near_overlaps_with_existing_data_removed": near_overlap,
            "admitted_windows": len(rows),
            "admitted_families": len({str(row["family_id"]) for row in rows}),
            "partition_windows": {split: len(values) for split, values in split_rows.items()},
            "partition_families": {
                split: len({str(row["family_id"]) for row in values})
                for split, values in split_rows.items()
            },
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
    parser.add_argument("--source", type=Path, default=Path("data/raw/youtube_scam_calls_v2.zip"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/external/youtube_scam_calls"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/source-audits/youtube-scam-calls.json"),
    )
    args = parser.parse_args()
    build(args.source, args.data, args.output, args.report)


if __name__ == "__main__":
    main()
