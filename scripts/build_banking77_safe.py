#!/usr/bin/env python3
"""Build split-safe BANKING77 legitimate financial-service artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.build_dataset import (
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        URL_RE,
        clean_text,
        cluster_near_duplicates,
        deduplicate,
        make_row,
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
        URL_RE,
        clean_text,
        cluster_near_duplicates,
        deduplicate,
        make_row,
        privacy_normalize_real_text,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )

from scamguard.metrics import file_sha256

SOURCE = "polyai_banking77"
SOURCE_REPOSITORY = "PolyAI-LDN/task-specific-datasets"
SOURCE_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_PAPER = "https://arxiv.org/abs/2003.04807"
LICENSE_NAME = "CC-BY-4.0"
EXPECTED_HEADER = ["text", "category"]
EXPECTED_ROWS = {"train": 10_003, "test": 3_080}
EXPECTED_CATEGORIES = 77
DEFAULT_SELECTION_PER_INTENT = 10
SELECTION_SEED = "scamguard-banking77-selection-v1"


@dataclass(frozen=True)
class SourceFile:
    filename: str
    url: str
    sha256: str


SOURCE_FILES = (
    SourceFile(
        "train.csv",
        (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{SOURCE_REVISION}/banking_data/train.csv"
        ),
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    ),
    SourceFile(
        "test.csv",
        (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{SOURCE_REVISION}/banking_data/test.csv"
        ),
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    ),
    SourceFile(
        "categories.json",
        (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{SOURCE_REVISION}/banking_data/categories.json"
        ),
        "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    ),
    SourceFile(
        "LICENSE",
        (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{SOURCE_REVISION}/LICENSE"
        ),
        "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
    ),
)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def reference_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in paths:
        candidates = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
        for path in candidates:
            if not path.is_file() or "quarantine" in path.name or "sealed" in path.name:
                continue
            rows.extend(row for row in read_jsonl(path) if isinstance(row.get("text"), str))
    return rows


def verify_sources(source_dir: Path) -> None:
    for source in SOURCE_FILES:
        path = source_dir / source.filename
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned BANKING77 source: {path}")
        if file_sha256(path) != source.sha256:
            raise ValueError(f"BANKING77 source differs from pinned revision: {path}")


def read_split(
    path: Path, *, official_split: str, categories: set[str]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    privacy = Counter[str]()
    intent_counts = Counter[str]()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(f"unexpected BANKING77 header in {path}: {reader.fieldnames}")
        source_count = 0
        for source_count, source_row in enumerate(reader, start=1):
            raw_text = str(source_row["text"] or "").strip()
            intent = str(source_row["category"] or "").strip()
            if not raw_text or intent not in categories:
                raise ValueError(f"invalid BANKING77 row {source_count} in {path}")
            privacy["rows_with_email"] += bool(EMAIL_RE.search(raw_text))
            privacy["rows_with_phone_like"] += bool(PHONE_LIKE_RE.search(raw_text))
            privacy["rows_with_long_digit_sequence"] += bool(LONG_DIGIT_RE.search(raw_text))
            privacy["rows_with_url"] += bool(URL_RE.search(raw_text))
            text = URL_RE.sub("<URL>", privacy_normalize_real_text(clean_text(raw_text)))
            row = make_row(
                text=text,
                label="SAFE",
                source=SOURCE,
                source_label=intent,
                license_name=LICENSE_NAME,
            )
            if row is None:
                privacy["rows_rejected_too_short"] += 1
                continue
            source_identity = f"{official_split}:{source_count}:{intent}:{raw_text}"
            row.update(
                {
                    "official_split": official_split,
                    "source_intent": intent,
                    "source_record_sha256": stable_hash(source_identity),
                    "provenance_class": "published_banking_customer_service_query",
                    "naturally_occurring_provenance_verified": False,
                    "individual_safe_ground_truth": False,
                    "label_policy": "legitimate_banking_support_domain_weak_safe",
                    "privacy_normalization": "url_email_phone_and_long_digit_placeholders_v1",
                }
            )
            rows.append(row)
            intent_counts[intent] += 1
    expected = EXPECTED_ROWS[official_split]
    if source_count != expected:
        raise ValueError(
            f"expected {expected} BANKING77 {official_split} rows, found {source_count}"
        )
    if set(intent_counts) != categories:
        raise ValueError(f"BANKING77 {official_split} does not contain every declared intent")
    return rows, {
        "source_rows": source_count,
        "candidate_rows": len(rows),
        "intent_rows_min": min(intent_counts.values()),
        "intent_rows_max": max(intent_counts.values()),
        "privacy_findings_before_normalization": dict(privacy),
    }


def one_per_family(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["family_id"])].append(row)
    representatives = [
        min(group, key=lambda row: stable_hash(f"{SELECTION_SEED}:{row['id']}"))
        for group in groups.values()
    ]
    representatives.sort(key=lambda row: str(row["id"]))
    return representatives, len(rows) - len(representatives)


def split_train_representatives(
    rows: list[dict[str, object]], *, selection_per_intent: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_intent: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_intent[str(row["source_intent"])].append(row)
    fit: list[dict[str, object]] = []
    selection: list[dict[str, object]] = []
    for intent, candidates in sorted(by_intent.items()):
        ordered = sorted(
            candidates,
            key=lambda row: stable_hash(f"{SELECTION_SEED}:{intent}:{row['id']}"),
        )
        if len(ordered) <= selection_per_intent:
            raise ValueError(f"too few independent BANKING77 families for intent {intent}")
        selection.extend(
            row | {"split": "banking77_validation"}
            for row in ordered[:selection_per_intent]
        )
        fit.extend(row | {"split": "banking77_fit"} for row in ordered[selection_per_intent:])
    fit.sort(key=lambda row: str(row["id"]))
    selection.sort(key=lambda row: str(row["id"]))
    return fit, selection


def build(
    source_dir: Path,
    output: Path,
    report_path: Path | None = None,
    *,
    references: tuple[Path, ...] = (),
    selection_per_intent: int = DEFAULT_SELECTION_PER_INTENT,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite BANKING77 artifacts: {output}")
    if selection_per_intent <= 0:
        raise ValueError("selection_per_intent must be positive")
    verify_sources(source_dir)
    categories_list = json.loads((source_dir / "categories.json").read_text(encoding="utf-8"))
    if not (
        isinstance(categories_list, list)
        and len(categories_list) == EXPECTED_CATEGORIES
        and len(set(categories_list)) == EXPECTED_CATEGORIES
        and all(isinstance(category, str) and category for category in categories_list)
    ):
        raise ValueError("unexpected BANKING77 category inventory")
    categories = set(categories_list)
    train_rows, train_stats = read_split(
        source_dir / "train.csv", official_split="train", categories=categories
    )
    test_rows, test_stats = read_split(
        source_dir / "test.csv", official_split="test", categories=categories
    )
    train_exact, train_exact_dropped, train_conflicts = deduplicate(train_rows)
    test_exact, test_exact_dropped, test_conflicts = deduplicate(test_rows)
    if train_conflicts or test_conflicts:
        raise ValueError("SAFE-only BANKING77 source produced an exact label conflict")

    held_references = reference_rows(references)
    test_nonoverlap, test_reference_overlap = remove_near_overlaps(
        test_exact, held_references
    )
    test_clustered, test_near_conflicts, test_near_stats = cluster_near_duplicates(
        test_nonoverlap
    )
    if test_near_conflicts:
        raise ValueError("SAFE-only BANKING77 test produced a near-template label conflict")
    test_representatives, test_family_dropped = one_per_family(test_clustered)
    test_representatives = [
        row | {"split": "banking77_test_sealed"} for row in test_representatives
    ]

    train_without_test_overlap, train_test_overlap = remove_near_overlaps(
        train_exact, test_exact
    )
    train_nonoverlap, train_reference_overlap = remove_near_overlaps(
        train_without_test_overlap, held_references
    )
    train_clustered, train_near_conflicts, train_near_stats = cluster_near_duplicates(
        train_nonoverlap
    )
    if train_near_conflicts:
        raise ValueError("SAFE-only BANKING77 train produced a near-template label conflict")
    train_representatives, train_family_dropped = one_per_family(train_clustered)
    fit, selection = split_train_representatives(
        train_representatives, selection_per_intent=selection_per_intent
    )

    artifacts = fit + selection + test_representatives
    if any(
        EMAIL_RE.search(str(row["text"]))
        or PHONE_LIKE_RE.search(str(row["text"]))
        or LONG_DIGIT_RE.search(str(row["text"]))
        or URL_RE.search(str(row["text"]))
        for row in artifacts
    ):
        raise ValueError("privacy-like value survived BANKING77 normalization")
    if len({str(row["id"]) for row in artifacts}) != len(artifacts):
        raise ValueError("BANKING77 artifact IDs are not unique")
    if {str(row["label"]) for row in artifacts} != {"SAFE"}:
        raise ValueError("BANKING77 artifacts must contain SAFE rows only")
    for rows in (fit, selection, test_representatives):
        if {str(row["source_intent"]) for row in rows} != categories:
            raise ValueError("a BANKING77 artifact lost one or more intents")

    output.mkdir(parents=True)
    artifact_rows = {
        "fit": (output / "banking77_fit.jsonl", fit),
        "validation": (output / "banking77_validation.jsonl", selection),
        "test_sealed": (output / "banking77_test_sealed.jsonl", test_representatives),
    }
    for path, rows in artifact_rows.values():
        write_jsonl(path, rows)

    manifest: dict[str, object] = {
        "dataset_schema_version": 1,
        "purpose": (
            "licensed human-authored financial-service SAFE boundary source; fitting pool remains "
            "unadmitted, validation is selection-only, and official test predictions stay sealed"
        ),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "paper": SOURCE_PAPER,
            "license": LICENSE_NAME,
            "files": [
                {"filename": source.filename, "sha256": source.sha256}
                for source in SOURCE_FILES
            ],
            "publisher_description": "online banking customer-service queries over 77 intents",
            "natural_or_crowdsourced_provenance_documented": False,
        },
        "policy": {
            "fit_artifact_used_for_fitting": False,
            "validation_used_for_fitting": False,
            "validation_used_for_threshold": False,
            "validation_used_for_candidate_selection": True,
            "official_test_predictions_opened": False,
            "official_test_used_for_fitting_or_selection": False,
            "commercial_use_allowed": True,
            "attribution_required": True,
            "rows_redistributed": False,
            "naturally_occurring_provenance_claimed": False,
            "individual_safe_ground_truth_claimed": False,
            "one_representative_per_near_template_family": True,
            "selection_examples_per_intent": selection_per_intent,
            "selection_seed": SELECTION_SEED,
            "raw_text_written_to_manifest": False,
        },
        "counts": {
            "declared_intents": len(categories),
            "train": train_stats
            | {
                "exact_duplicate_rows_removed": train_exact_dropped,
                "near_overlaps_with_official_test_removed": train_test_overlap,
                "near_overlaps_with_existing_references_removed": train_reference_overlap,
                "same_family_rows_removed": train_family_dropped,
                "independent_family_representatives": len(train_representatives),
                "fit_rows": len(fit),
                "selection_rows": len(selection),
            },
            "test": test_stats
            | {
                "exact_duplicate_rows_removed": test_exact_dropped,
                "near_overlaps_with_existing_references_removed": test_reference_overlap,
                "same_family_rows_removed": test_family_dropped,
                "sealed_rows": len(test_representatives),
            },
            "reference_rows": len(held_references),
        },
        "near_template_stats": {
            "train": train_near_stats,
            "test": test_near_stats,
        },
        "artifacts": {
            key: {
                "path": str(path),
                "rows": len(rows),
                "sha256": file_sha256(path),
            }
            for key, (path, rows) in artifact_rows.items()
        },
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
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--output", type=Path, default=Path("data/external/banking77"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/source-audits/banking77-safe.json"),
    )
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument(
        "--selection-per-intent", type=int, default=DEFAULT_SELECTION_PER_INTENT
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source,
                args.output,
                args.report,
                references=tuple(args.reference),
                selection_per_intent=args.selection_per_intent,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
