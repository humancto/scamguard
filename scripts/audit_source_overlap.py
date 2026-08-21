#!/usr/bin/env python3
"""Audit a candidate CSV source before it can enter ScamBench.

The report is intentionally model-free: it measures provenance-adjacent data
quality properties without observing candidate-model predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.build_dataset import (
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        clean_text,
        family_skeleton,
        normalized,
        privacy_normalize_real_text,
        simhash64,
        simhash_bands,
    )
except ModuleNotFoundError:  # Direct `python scripts/audit_source_overlap.py` execution.
    from build_dataset import (  # type: ignore[no-redef]
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        clean_text,
        family_skeleton,
        normalized,
        privacy_normalize_real_text,
        simhash64,
        simhash_bands,
    )


def read_candidate(path: Path, text_column: str, label_column: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for source_row in csv.DictReader(handle):
            raw_text = clean_text(source_row.get(text_column, ""))
            label = clean_text(source_row.get(label_column, ""))
            if raw_text and label:
                rows.append(
                    {
                        "raw_text": raw_text,
                        "text": privacy_normalize_real_text(raw_text),
                        "label": label,
                    }
                )
    return rows


def read_reference_rows(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.jsonl")):
        if "quarantine" in path.name:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict) and isinstance(row.get("text"), str):
                        rows.append(row | {"_audit_file": path.name})
    return rows


def near_overlap_indices(
    candidate_rows: list[dict[str, str]], reference_rows: list[dict[str, object]], radius: int
) -> set[int]:
    reference_signatures = [
        simhash64(family_skeleton(str(row["text"]))) for row in reference_rows
    ]
    buckets: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for index, signature in enumerate(reference_signatures):
        for key in simhash_bands(signature, max_hamming=radius):
            buckets[key].append(index)

    overlaps: set[int] = set()
    for candidate_index, row in enumerate(candidate_rows):
        signature = simhash64(family_skeleton(row["text"]))
        candidates: set[int] = set()
        for key in simhash_bands(signature, max_hamming=radius):
            candidates.update(buckets[key])
        if any(
            (signature ^ reference_signatures[index]).bit_count() <= radius
            for index in candidates
        ):
            overlaps.add(candidate_index)
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--reference-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--near-hamming", type=int, default=6)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    candidate_rows = read_candidate(args.candidate, args.text_column, args.label_column)
    reference_rows = read_reference_rows(args.reference_dir)
    normalized_groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        normalized_groups[normalized(row["text"])].append(row)

    exact_reference_keys = {normalized(str(row["text"])) for row in reference_rows}
    exact_overlap = sum(normalized(row["text"]) in exact_reference_keys for row in candidate_rows)
    near_overlaps = near_overlap_indices(candidate_rows, reference_rows, args.near_hamming)
    overlap_by_file: dict[str, dict[str, int]] = {}
    for filename in sorted({str(row["_audit_file"]) for row in reference_rows}):
        file_rows = [row for row in reference_rows if row["_audit_file"] == filename]
        file_keys = {normalized(str(row["text"])) for row in file_rows}
        overlap_by_file[filename] = {
            "exact_overlap_rows": sum(
                normalized(row["text"]) in file_keys for row in candidate_rows
            ),
            "near_overlap_rows_including_exact": len(
                near_overlap_indices(candidate_rows, file_rows, args.near_hamming)
            ),
        }
    exact_duplicate_rows = sum(len(rows) - 1 for rows in normalized_groups.values())
    exact_conflict_groups = sum(
        len({row["label"] for row in rows}) > 1 for rows in normalized_groups.values()
    )
    report = {
        "candidate": str(args.candidate),
        "candidate_rows": len(candidate_rows),
        "labels": dict(Counter(row["label"] for row in candidate_rows)),
        "empty_or_unlabeled_rows_dropped": 0,
        "privacy_findings_before_normalization": {
            "rows_with_email": sum(
                bool(EMAIL_RE.search(row["raw_text"])) for row in candidate_rows
            ),
            "rows_with_phone_like": sum(
                bool(PHONE_LIKE_RE.search(row["raw_text"])) for row in candidate_rows
            ),
            "rows_with_long_digit_sequence": sum(
                bool(LONG_DIGIT_RE.search(row["raw_text"])) for row in candidate_rows
            ),
        },
        "privacy_findings_after_normalization": {
            "rows_with_email": sum(bool(EMAIL_RE.search(row["text"])) for row in candidate_rows),
            "rows_with_phone_like": sum(
                bool(PHONE_LIKE_RE.search(row["text"])) for row in candidate_rows
            ),
            "rows_with_long_digit_sequence": sum(
                bool(LONG_DIGIT_RE.search(row["text"])) for row in candidate_rows
            ),
        },
        "within_candidate": {
            "exact_duplicate_rows": exact_duplicate_rows,
            "exact_conflicting_label_groups": exact_conflict_groups,
        },
        "reference": {
            "directory": str(args.reference_dir),
            "rows": len(reference_rows),
            "exact_overlap_rows": exact_overlap,
            "near_overlap_rows_including_exact": len(near_overlaps),
            "near_hamming_max": args.near_hamming,
            "overlap_by_file": overlap_by_file,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
