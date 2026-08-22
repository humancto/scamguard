#!/usr/bin/env python3
"""Fail closed on schema, duplicates, leakage, provenance, and split coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.build_dataset import family_skeleton, simhash64, simhash_bands
except ModuleNotFoundError:  # Direct `python scripts/validate_dataset.py` execution.
    from build_dataset import family_skeleton, simhash64, simhash_bands

from scamguard.signals import extract_signal_matches

REQUIRED = {
    "id",
    "text",
    "label",
    "category",
    "source",
    "source_label",
    "license",
    "split",
    "family_id",
    "is_synthetic",
}
LABELS = {"SAFE", "UNCERTAIN", "SCAM"}
CORE_CATEGORIES = {
    "CREDENTIAL_THEFT",
    "DELIVERY_TOLL",
    "FINANCIAL",
    "IDENTITY_IMPERSONATION",
    "OPPORTUNITY",
    "RELATIONSHIP",
}
CATEGORIES = CORE_CATEGORIES | {"NONE", "UNKNOWN"}
LICENSES = {"Apache-2.0", "CC-BY-4.0", "CC0-1.0", "MIT"}
REAL_PII = re.compile(
    r"(?:\d{10,}|\b\d{3}-\d{2}-\d{4}\b|\b(?:\d[ -]*?){13,16}\b|"
    r"\b[A-Za-z0-9._%+-]+@(?!example\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)"
)
FORUM_UNMASKED_PHONE = re.compile(r"(?<![A-Za-z0-9])\+?\d(?:[\d ()-]{5,}\d)(?![A-Za-z0-9])")
SCAM_EXCLUDED_POLICY_FRAGMENTS = (
    "commercial_offer_without_clear_fraud",
    "defensive_guidance",
    "standalone_authentication_notification",
    "without_strong_text_evidence",
)
SYNTHETIC_REFERENCE_PREFIXES = (
    "https://consumer.ftc.gov/",
    "https://www.ic3.gov/",
    "https://www.irs.gov/",
    "https://www.uspis.gov/",
)
SYNTHETIC_METHODS = {
    "call_action_state_counterfactual_advisory_grounded_original_copy",
    "deterministic_slot_filling_original_copy",
    "deterministic_service_dialogue_error_audit_grounded_original_copy",
    "paired_deterministic_slot_filling_original_advisory_grounded_copy",
    "paired_deterministic_slot_filling_error_audit_grounded_original_copy",
    "paired_call_structure_minimal_contrast_advisory_grounded_original_copy",
    "paired_call_evidence_action_counterfactual_advisory_grounded_original_copy",
    "minimal_final_turn_transformation_of_cc_by_human_call_v1",
}
TRUSTED_POSITIVE_ONLY_SOURCES = {
    "youtube_scam_calls_cc0": {
        "license": "CC0-1.0",
        "label_policy": "publisher_positive_only_scam_call_collection",
        "provenance_class": "real_scam_call_or_autodialer_transcript",
    }
}


def has_scam_label_evidence(row: dict[str, object]) -> bool:
    if extract_signal_matches(str(row["text"])):
        return True
    contract = TRUSTED_POSITIVE_ONLY_SOURCES.get(str(row.get("source")))
    if contract is None:
        return False
    return all(row.get(field) == expected for field, expected in contract.items()) and bool(
        str(row.get("source_record_id", "")).strip()
    )


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def has_excluded_scam_policy(row: dict[str, object]) -> bool:
    """Catch source-label regressions independently of the dataset builder."""
    policy = str(row.get("label_policy", ""))
    return row.get("label") == "SCAM" and any(
        fragment in policy for fragment in SCAM_EXCLUDED_POLICY_FRAGMENTS
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", " ".join(value.casefold().split()))


def validate_fresh_holdout(data: Path, errors: list[str]) -> tuple[int, Counter[str]]:
    path = data / "primary_test_v8.jsonl"
    manifest_path = data / "primary_test_v8.manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        errors.append("schema-v8 primary holdout or manifest is missing")
        return 0, Counter()

    rows = read_rows(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = manifest.get("counts", {})
    source = manifest.get("source", {})
    artifact = manifest.get("artifact", {})
    if manifest.get("schema_version") != 8:
        errors.append("primary_test_v8 manifest schema_version is not 8")
    if manifest.get("benchmark_state") != "SEALED_MODEL_PREDICTIONS_NOT_RUN":
        errors.append("primary_test_v8 is not prediction-sealed")
    if source.get("training_allowed_by_project") is not False:
        errors.append("primary_test_v8 source is not explicitly excluded from training")
    if source.get("public_redistribution_allowed_by_project") is not False:
        errors.append("primary_test_v8 source is not excluded from public row redistribution")
    if artifact.get("sha256") != sha256(path):
        errors.append("primary_test_v8 artifact hash differs from manifest")
    if counts.get("final_rows") != len(rows):
        errors.append("primary_test_v8 final row count differs from manifest")

    labels = Counter(str(row.get("label")) for row in rows)
    if counts.get("final_labels") != dict(labels):
        errors.append("primary_test_v8 label counts differ from manifest")
    if not {"SAFE", "SCAM"}.issubset(labels):
        errors.append(f"primary_test_v8 lacks SAFE or SCAM coverage: {dict(labels)}")

    ids: set[str] = set()
    families: set[str] = set()
    candidate_texts: set[str] = set()
    candidate_signatures: list[tuple[int, str]] = []
    candidate_buckets: defaultdict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"primary_test_v8:{index} missing {sorted(missing)}")
            continue
        if row["split"] != "test":
            errors.append(f"primary_test_v8:{index} split is not test")
        if row["source"] != "moz_smishing" or row["license"] != "CreativeML-OpenRAIL-M":
            errors.append(f"primary_test_v8:{index} unexpected source or license")
        if row["is_synthetic"] is not False:
            errors.append(f"primary_test_v8:{index} must be real-source")
        if row["label"] not in {"SAFE", "SCAM"}:
            errors.append(f"primary_test_v8:{index} unexpected label {row['label']!r}")
        if row["label"] == "SCAM" and row["category"] != "FINANCIAL":
            errors.append(f"primary_test_v8:{index} SCAM category is not FINANCIAL")
        if row["label"] == "SAFE" and row["category"] != "NONE":
            errors.append(f"primary_test_v8:{index} SAFE category is not NONE")
        if REAL_PII.search(str(row["text"])) or FORUM_UNMASKED_PHONE.search(str(row["text"])):
            errors.append(f"primary_test_v8:{index} retains a PII-like value")
        row_id = str(row["id"])
        family_id = str(row["family_id"])
        if row_id in ids:
            errors.append(f"primary_test_v8 duplicate id: {row_id}")
        if family_id in families:
            errors.append(f"primary_test_v8 duplicate family representative: {family_id}")
        ids.add(row_id)
        families.add(family_id)
        normalized = normalized_text(str(row["text"]))
        if normalized in candidate_texts:
            errors.append(f"primary_test_v8 duplicate normalized text: {row_id}")
        candidate_texts.add(normalized)
        signature = simhash64(family_skeleton(str(row["text"])))
        near_candidates: set[tuple[int, str]] = set()
        for key in simhash_bands(signature, max_hamming=6):
            near_candidates.update(candidate_buckets[key])
        for other_signature, other_id in near_candidates:
            if (signature ^ other_signature).bit_count() <= 6:
                errors.append(
                    f"primary_test_v8 retains same-source near templates: {row_id} and {other_id}"
                )
        for key in simhash_bands(signature, max_hamming=6):
            candidate_buckets[key].append((signature, row_id))
        candidate_signatures.append((signature, row_id))

    reference_rows: list[dict[str, object]] = []
    for filename, recorded in manifest.get("reference_files", {}).items():
        reference_path = data / filename
        if not reference_path.is_file():
            errors.append(f"primary_test_v8 reference is missing: {filename}")
            continue
        if recorded.get("sha256") != sha256(reference_path):
            errors.append(f"primary_test_v8 reference hash differs: {filename}")
        file_rows = read_rows(reference_path)
        if recorded.get("rows") != len(file_rows):
            errors.append(f"primary_test_v8 reference count differs: {filename}")
        reference_rows.extend(file_rows)

    reference_texts = {normalized_text(str(row["text"])) for row in reference_rows}
    exact_overlap = candidate_texts & reference_texts
    if exact_overlap:
        errors.append(f"primary_test_v8 has {len(exact_overlap)} exact reference overlaps")
    reference_buckets: defaultdict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for row in reference_rows:
        signature = simhash64(family_skeleton(str(row["text"])))
        for key in simhash_bands(signature, max_hamming=6):
            reference_buckets[key].append((signature, str(row["id"])))
    for signature, row_id in candidate_signatures:
        possible: set[tuple[int, str]] = set()
        for key in simhash_bands(signature, max_hamming=6):
            possible.update(reference_buckets[key])
        if any((signature ^ other_signature).bit_count() <= 6 for other_signature, _ in possible):
            errors.append(f"primary_test_v8 near-overlaps a reference row: {row_id}")

    raw_path = data.parent / "raw" / "moz_smishing.csv"
    if raw_path.is_file() and source.get("raw_sha256") != sha256(raw_path):
        errors.append("primary_test_v8 raw source hash differs from manifest")
    return len(rows), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=12,
        help="schema version required in the processed manifest",
    )
    parser.add_argument(
        "--sealed-data",
        type=Path,
        help=(
            "directory containing the prediction-sealed primary test; defaults to --data "
            "and allows isolated experiment directories to reuse the frozen sealed artifact"
        ),
    )
    args = parser.parse_args()

    split_names = [
        "train",
        "dev",
        "test",
        "ood_financial",
        "ood_wspr",
        "forum_validation",
        "ood_forum",
    ]
    if (args.data / "ood_azsc.jsonl").is_file():
        split_names.append("ood_azsc")
    if (args.data / "call_pair_validation.jsonl").is_file():
        split_names.append("call_pair_validation")
    if (args.data / "call_state_validation.jsonl").is_file():
        split_names.append("call_state_validation")
    if (args.data / "call_window_validation.jsonl").is_file():
        split_names.append("call_window_validation")
    if (args.data / "harper_call_validation.jsonl").is_file():
        split_names.append("harper_call_validation")
    if (args.data / "harper_state_validation.jsonl").is_file():
        split_names.append("harper_state_validation")
    rows_by_split = {split: read_rows(args.data / f"{split}.jsonl") for split in split_names}
    errors: list[str] = []
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != args.expected_schema_version:
        errors.append(
            "processed manifest schema_version is not "
            f"{args.expected_schema_version}: {manifest.get('schema_version')}"
        )
    ids: set[str] = set()
    texts: dict[str, str] = {}
    family_splits: defaultdict[str, set[str]] = defaultdict(set)
    near_buckets: defaultdict[tuple[int, int], list[tuple[int, str, str]]] = defaultdict(list)

    for split, rows in rows_by_split.items():
        split_name = (
            "ood"
            if split.startswith("ood_")
            else "validation"
            if split.endswith("_validation")
            else split
        )
        labels = Counter(str(row.get("label")) for row in rows)
        single_class_scam_splits = {"ood_wspr", "forum_validation", "ood_forum"}
        if split not in single_class_scam_splits | {
            "call_window_validation",
            "harper_call_validation",
        } and not {"SAFE", "SCAM"}.issubset(labels):
            errors.append(f"{split} lacks SAFE or SCAM coverage: {dict(labels)}")
        if split in single_class_scam_splits and "SCAM" not in labels:
            errors.append(f"{split} lacks SCAM coverage: {dict(labels)}")
        if split in {"call_window_validation", "harper_call_validation"} and set(
            labels
        ) != {"SAFE"}:
            errors.append(f"{split} is not the declared SAFE-only diagnostic: {dict(labels)}")
        for index, row in enumerate(rows, start=1):
            missing = REQUIRED - row.keys()
            if missing:
                errors.append(f"{split}:{index} missing {sorted(missing)}")
                continue
            if row["split"] != split_name:
                errors.append(f"{split}:{index} split field is {row['split']!r}")
            if row["label"] not in LABELS:
                errors.append(f"{split}:{index} invalid label {row['label']!r}")
            if row["category"] not in CATEGORIES:
                errors.append(f"{split}:{index} invalid category {row['category']!r}")
            if row["label"] == "SCAM" and row["category"] == "NONE":
                errors.append(f"{split}:{index} SCAM category is NONE")
            if has_excluded_scam_policy(row):
                errors.append(
                    f"{split}:{index} SCAM retains excluded label policy "
                    f"{row.get('label_policy')!r}"
                )
            if (
                split in {"train", "dev", "test"}
                and row["label"] == "SCAM"
                and not has_scam_label_evidence(row)
            ):
                errors.append(f"{split}:{index} SCAM has no extractive evidence signal")
            if row["label"] != "SCAM" and row["category"] != "NONE":
                errors.append(f"{split}:{index} non-SCAM category is {row['category']!r}")
            if row["license"] not in LICENSES:
                errors.append(f"{split}:{index} unexpected license {row['license']!r}")
            if not isinstance(row["is_synthetic"], bool):
                errors.append(f"{split}:{index} is_synthetic is not boolean")
            for field in ("id", "text", "source", "source_label", "license", "family_id"):
                if not isinstance(row[field], str) or not str(row[field]).strip():
                    errors.append(f"{split}:{index} empty or non-string {field}")
            if row["id"] in ids:
                errors.append(f"duplicate id: {row['id']}")
            ids.add(str(row["id"]))
            normalized = " ".join(str(row["text"]).casefold().split())
            if normalized in texts:
                errors.append(f"exact text overlap: {split}:{index} and {texts[normalized]}")
            texts[normalized] = f"{split}:{index}"
            family_splits[str(row["family_id"])].add(split_name)
            if row["is_synthetic"] and REAL_PII.search(str(row["text"])):
                errors.append(f"synthetic PII-like value: {split}:{index}")
            if row["is_synthetic"]:
                if row.get("synthetic_method") not in SYNTHETIC_METHODS:
                    errors.append(f"synthetic method missing or unexpected: {split}:{index}")
                reference = str(row.get("pattern_reference", ""))
                if not reference.startswith(SYNTHETIC_REFERENCE_PREFIXES):
                    errors.append(
                        f"synthetic pattern reference is not authoritative: {split}:{index}"
                    )
            if not row["is_synthetic"] and (
                REAL_PII.search(str(row["text"])) or FORUM_UNMASKED_PHONE.search(str(row["text"]))
            ):
                errors.append(f"real-source row retains PII-like value: {split}:{index}")
            if not row["is_synthetic"]:
                signature = simhash64(family_skeleton(str(row["text"])))
                candidates: set[tuple[int, str, str]] = set()
                for key in simhash_bands(signature, max_hamming=6):
                    candidates.update(near_buckets[key])
                for other_signature, other_split, other_id in candidates:
                    development_pair = {split, other_split} <= {
                        "train",
                        "dev",
                        "test",
                        "forum_validation",
                    }
                    ood_holdout_pair = any(
                        candidate.startswith("ood_") for candidate in (split, other_split)
                    ) and bool({split, other_split} & {"train", "dev", "test", "forum_validation"})
                    if (
                        (development_pair or ood_holdout_pair)
                        and split != other_split
                        and (signature ^ other_signature).bit_count() <= 6
                    ):
                        errors.append(
                            f"near-template leakage: {row['id']} ({split}) and "
                            f"{other_id} ({other_split})"
                        )
                for key in simhash_bands(signature, max_hamming=6):
                    near_buckets[key].append((signature, split, str(row["id"])))

    for family_id, splits in family_splits.items():
        development = splits & {"train", "dev", "test", "validation"}
        if len(development) > 1:
            errors.append(f"family leakage: {family_id} appears in {sorted(development)}")

    for split in ("train", "dev", "test"):
        scam_categories = {
            str(row["category"]) for row in rows_by_split[split] if row["label"] == "SCAM"
        }
        missing_categories = CORE_CATEGORIES - scam_categories
        if missing_categories:
            errors.append(f"{split} lacks core scam categories: {sorted(missing_categories)}")

    actual_counts = {split: len(rows) for split, rows in rows_by_split.items()}
    if manifest.get("counts") != actual_counts:
        errors.append(
            f"manifest counts differ: recorded={manifest.get('counts')} actual={actual_counts}"
        )
    development_rows = sum((rows_by_split[split] for split in ("train", "dev", "test")), start=[])
    actual_labels = dict(Counter(str(row["label"]) for row in development_rows))
    actual_sources = dict(Counter(str(row["source"]) for row in development_rows))
    if manifest.get("labels") != actual_labels:
        errors.append(
            f"manifest labels differ: recorded={manifest.get('labels')} actual={actual_labels}"
        )
    if manifest.get("sources") != actual_sources:
        errors.append(
            f"manifest sources differ: recorded={manifest.get('sources')} actual={actual_sources}"
        )

    fresh_count, fresh_labels = validate_fresh_holdout(args.sealed_data or args.data, errors)

    if errors:
        for error in errors[:100]:
            print(f"ERROR {error}")
        raise SystemExit(f"dataset validation failed with {len(errors)} error(s)")

    for split, rows in rows_by_split.items():
        print(f"{split}: {len(rows)} {dict(Counter(str(row['label']) for row in rows))}")
        if split in {"train", "dev", "test"}:
            categories = Counter(str(row["category"]) for row in rows if row["label"] == "SCAM")
            print(f"{split} scam categories: {dict(categories)}")
    print(f"primary_test_v8: {fresh_count} {dict(fresh_labels)} (sealed)")
    print(f"validation passed: {len(ids)} unique examples, no family leakage")


if __name__ == "__main__":
    main()
