#!/usr/bin/env python3
"""Freeze schema-26 by subtracting two unresolved schema-25 training sources.

This is a byte-preserving, research-only source filter, not label adjudication or
rights clearance. It reads only the five pinned development/provenance inputs;
it never opens a held-out/test corpus, adds examples, or resumes model weights.
Unexpected derivative ancestry fails closed so it must be resolved explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from scamguard.privacy import CONTEXTUAL_PRIVACY_REVISION, mask_contextual_sensitive_values

try:
    from scripts.build_dataset import EMAIL_RE, LONG_DIGIT_RE, PHONE_LIKE_RE
except ModuleNotFoundError:  # Direct script execution puts scripts/ on sys.path.
    from build_dataset import EMAIL_RE, LONG_DIGIT_RE, PHONE_LIKE_RE  # type: ignore[no-redef]

PARENT = "data/experiments/schema25-full-call-curriculum/processed"
OUTPUT = "data/experiments/schema26-ettin-source-filtered/processed"
REPORT = "reports/data/ettin-v3-corpus.json"
INPUT_PINS = {
    "train": (f"{PARENT}/train.jsonl",
              "d4e3e100e97a937a789aa984d4ce75b545592886099a0027dbd758bbf0b2e614"),
    "dev": (f"{PARENT}/dev.jsonl",
            "04201ee86325306cf1751dbf848ddca360386de707069c0c4bba0c9c28a21c07"),
    "parent_manifest": (f"{PARENT}/manifest.json",
                        "6c1096f1017edb10b690fde8968d9de892d2bff8ae49b9183c5da15e0380ad57"),
    "source_inventory": ("configs/ettin150m-release-provenance.json",
                         "791fb85a06a8594983104219c51ffb531989e038fb02f849451d10cbc56368c0"),
    "source_registry": ("data/raw/sources.json",
                        "bb27a2d89e98a7e53f009ca556d35cb4a62fcd79dd5827ffafec3f609ae99cea"),
}
PRIVACY_CODE_PINS = {
    "src/scamguard/privacy.py":
        "089f118c174cc1249589d3250992a3574b4b3c0c3bb5dcd24547e40c00aef0fa",
    "scripts/build_dataset.py":
        "1c6b8b1e86e640eced80b1fa10cb3957e6d1f44b27e6f7a72cca12fd27fe78fb",
}
ALLOWED_SOURCES = {
    "imc25_public_forum_smishing": "CC-BY-4.0",
    "mendeley_sms_phishing": "CC-BY-4.0",
    "multidogo_human_service_dialogues": "CDLA-Permissive-1.0",
    "scamguard_ftc_pattern_action_states_v1": "Apache-2.0",
    "scamguard_multidogo_grounded_evidence_states_v1": "CDLA-Permissive-1.0",
    "scamguard_synthetic_counterfactual_v1": "Apache-2.0",
    "scamguard_synthetic_dialogue_v2": "Apache-2.0",
    "scamguard_synthetic_long_call_action_states_v1": "Apache-2.0",
    "scamguard_synthetic_v5": "Apache-2.0",
    "taskmaster1_woz_dialogues": "CC-BY-4.0",
    "uci_sms_spam": "CC-BY-4.0",
    "wspr_sms_phishing": "MIT",
}
EXCLUDED_SOURCES = {
    "shakeleoatmeal_phone_scam_synthetic": "MIT",
    "youtube_scam_calls_cc0": "CC0-1.0",
}
EXCLUDED_MARKERS = (
    "shakeleoatmeal", "youtube_scam_calls", "youtube-call-",
    "phone-scam-synthetic", "phone-scam-near-", "phone-scam-source-",
    "phone-scam-detection-synthetic", "27a1f6d0cf21dda995130d221383900b060dbcc9",
)
LABELS = ("SAFE", "UNCERTAIN", "SCAM")
FAMILY_FIELDS = ("family_id", "near_template_family_id")
HASH_FIELDS = ("shared_context_sha256", "schema23_source_text_sha256")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def ids_digest(rows: list[dict[str, Any]]) -> str:
    return digest(("\n".join(sorted(row["id"] for row in rows)) + "\n").encode())


def families(rows: list[dict[str, Any]]) -> set[str]:
    return {row[key] for row in rows for key in FAMILY_FIELDS if row.get(key)}


def safe_path(root: Path, relative: str) -> Path:
    requested = Path(relative)
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError("paths must be repo-relative without traversal")
    path = root / requested
    if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
        raise ValueError(f"symlink input/output rejected: {relative}")
    if not path.resolve().is_relative_to(root):
        raise ValueError("path escaped repository root")
    return path


def read_rows(data: bytes, split: str) -> tuple[list[dict[str, Any]], list[bytes]]:
    if not data.endswith(b"\n"):
        raise ValueError(f"{split}: missing final newline")
    lines = data.splitlines(keepends=True)
    rows = []
    source_licenses = ALLOWED_SOURCES | EXCLUDED_SOURCES
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{split}: row must be an object")
        for key in ("id", "text", "family_id", "source", "label", "license", "category"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{split}: missing/non-string {key}")
        if row["label"] not in LABELS or row.get("split") != split:
            raise ValueError(f"{split}: invalid label or source split")
        if row["source"] not in source_licenses:
            raise ValueError(f"{split}: unknown source")
        if row["license"] != source_licenses[row["source"]]:
            raise ValueError(f"{split}: source license differs from frozen inventory")
        for key in (*FAMILY_FIELDS, *HASH_FIELDS):
            if key in row and (not isinstance(row[key], str) or not row[key]):
                raise ValueError(f"{split}: malformed ancestry field {key}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{split}: empty input")
    for key, values in (
        ("ID", [row["id"] for row in rows]),
        ("normalized text", [normalized_text(row["text"]) for row in rows]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"{split}: duplicate {key}")
    return rows, lines


def validate_inventory(
    inventory: dict[str, Any], rows: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    sources = inventory["sources"]
    source_map = {source["id"]: source for source in sources}
    expected = ALLOWED_SOURCES | EXCLUDED_SOURCES
    if len(sources) != len(source_map) or set(source_map) != set(expected):
        raise ValueError("source inventory must contain the exact frozen 14 sources")
    for key in ("train", "dev"):
        if inventory["data"][f"{key}_rows"] != len(rows[key]):
            raise ValueError(f"{key}: inventory row count differs")
        if inventory["data_sha256"][key] != INPUT_PINS[key][1]:
            raise ValueError(f"{key}: inventory input hash differs")
        if inventory["data"][key] != INPUT_PINS[key][0]:
            raise ValueError(f"{key}: inventory input path differs")
    if inventory["source_registry"] != {
        "path": INPUT_PINS["source_registry"][0], "sha256": INPUT_PINS["source_registry"][1]
    }:
        raise ValueError("source registry binding differs")
    if inventory["data_sha256"]["manifest"] != INPUT_PINS["parent_manifest"][1]:
        raise ValueError("parent manifest binding differs")
    for name, license_name in expected.items():
        source = source_map[name]
        if source["license_declared"] != license_name:
            raise ValueError("source inventory license drift")
        for split, count_field in (("train", "rows"), ("dev", "dev_rows")):
            selected = [row for row in rows[split] if row["source"] == name]
            if len(selected) != source[count_field]:
                raise ValueError(f"{split}: source inventory count drift")
            if split == "train" and ids_digest(selected) != source["selected_train_ids_sha256"]:
                raise ValueError("source inventory ID digest drift")
    return source_map


def check_disjoint(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> None:
    comparisons = {
        "IDs": ({row["id"] for row in train}, {row["id"] for row in dev}),
        "normalized text": (
            {normalized_text(row["text"]) for row in train},
            {normalized_text(row["text"]) for row in dev},
        ),
        "families": (families(train), families(dev)),
    }
    for name, (left, right) in comparisons.items():
        if left & right:
            raise ValueError(f"train/dev overlap in {name}")


def filter_rows(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    excluded = [i for i, row in enumerate(rows) if row["source"] in EXCLUDED_SOURCES]
    retained = [i for i, row in enumerate(rows) if row["source"] in ALLOWED_SOURCES]
    excluded_rows = [rows[i] for i in excluded]
    excluded_identity = {row["id"] for row in excluded_rows} | families(excluded_rows)
    excluded_hashes = {
        row[key] for row in excluded_rows for key in HASH_FIELDS if row.get(key)
    } | {digest(row["text"].encode()) for row in excluded_rows}
    for index in retained:
        row = rows[index]
        # Metadata values, including nested derivation/source records, are inspected.
        # Text remains untouched and is not itself a provenance attribution signal.
        metadata = {key: value for key, value in row.items() if key != "text"}
        serialized = json.dumps(metadata, sort_keys=True, ensure_ascii=False).casefold()
        if any(marker in serialized for marker in EXCLUDED_MARKERS):
            raise ValueError("retained row embeds excluded provenance; explicit review required")
        scalars = set(_strings(metadata))
        if scalars & (excluded_identity | excluded_hashes):
            raise ValueError("retained row shares excluded ancestry; explicit review required")
    return retained, excluded


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [part for child in value.values() for part in _strings(child)]
    if isinstance(value, list):
        return [part for child in value for part in _strings(child)]
    return []


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "families": len({row["family_id"] for row in rows}),
        "labels": {label: sum(row["label"] == label for row in rows) for label in LABELS},
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "source_table": {
            source: {
                "rows": len(selected),
                "families": len({row["family_id"] for row in selected}),
                "labels": dict(sorted(Counter(row["label"] for row in selected).items())),
            }
            for source in sorted({row["source"] for row in rows})
            if (selected := [row for row in rows if row["source"] == source])
        },
    }


def privacy_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Inspect with existing masking heuristics without modifying any input row."""
    counts = {"email": 0, "phone_like": 0, "long_digits": 0, "contextual_value": 0}
    for row in rows:
        for name, pattern in (
            ("email", EMAIL_RE), ("phone_like", PHONE_LIKE_RE), ("long_digits", LONG_DIGIT_RE)
        ):
            counts[name] += bool(pattern.search(row["text"]))
        counts["contextual_value"] += mask_contextual_sensitive_values(row["text"]).changed
    if any(counts.values()):
        raise ValueError("unmasked privacy-like value detected; explicit review required")
    return {"status": "passed_heuristic_preflight", "rows": len(rows), "findings": counts,
            "text_modified": False, "independent_privacy_review_completed": False}


def build(root: Path, *, output: str = OUTPUT, report: str = REPORT) -> dict[str, Any]:
    root = root.resolve(strict=True)
    output_path, report_path = safe_path(root, output), safe_path(root, report)
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing output or report")
    if report_path.is_relative_to(output_path):
        raise ValueError("report must be outside the corpus output directory")
    inputs = {}
    for key, (relative, expected_sha) in INPUT_PINS.items():
        path = safe_path(root, relative)
        if path == report_path or path.is_relative_to(output_path):
            raise ValueError("output would contain a pinned input")
        data = path.read_bytes()
        if digest(data) != expected_sha:
            raise ValueError(f"pinned input hash mismatch: {key}")
        inputs[key] = data
    for relative, expected_sha in PRIVACY_CODE_PINS.items():
        if digest(safe_path(root, relative).read_bytes()) != expected_sha:
            raise ValueError("privacy scanner code differs from frozen audit policy")
    rows, lines = {}, {}
    for split in ("train", "dev"):
        rows[split], lines[split] = read_rows(inputs[split], split)
    inventory = json.loads(inputs["source_inventory"])
    source_map = validate_inventory(inventory, rows)
    check_disjoint(rows["train"], rows["dev"])
    # Cross-split excluded ancestry is checked together, not just within each split.
    filter_rows(rows["train"] + rows["dev"])
    retained, excluded, output_bytes = {}, {}, {}
    for split in ("train", "dev"):
        keep, drop = filter_rows(rows[split])
        retained[split], excluded[split] = [rows[split][i] for i in keep], [
            rows[split][i] for i in drop
        ]
        output_bytes[split] = b"".join(lines[split][i] for i in keep)
        if not retained[split]:
            raise ValueError(f"filter produced empty {split}")
    check_disjoint(retained["train"], retained["dev"])
    privacy = {split: privacy_audit(retained[split]) for split in ("train", "dev")}
    receipt = {
        "artifact_schema_version": 1,
        "dataset_schema_version": 26,
        "purpose": "source-filtered fresh-base Ettin research experiment",
        "research_eligible": True,
        "data_integrity_preflight": "passed",
        "release_authorized": False,
        "production_ready": False,
        "rights_review_status": "review_required",
        "human_review_status": "pending_new_corpus_bound_audit",
        "requires_fresh_base_weights": True,
        "input_refs": {key: {"path": path, "sha256": sha}
                       for key, (path, sha) in INPUT_PINS.items()},
        "builder": {"path": "scripts/build_ettin_v3_corpus.py",
                    "sha256": digest(Path(__file__).read_bytes())},
        "privacy_audit": {
            "policy": CONTEXTUAL_PRIVACY_REVISION,
            "code_sha256": PRIVACY_CODE_PINS,
            "splits": privacy,
            "limitation": "Conservative regex/masking checks are not a privacy certificate; "
                          "names, addresses, context and missed patterns require human review.",
        },
        "filter": {
            "allowlist": ALLOWED_SOURCES,
            "excluded_sources": EXCLUDED_SOURCES,
            "unexpected_derivative_policy": "reject; explicit ancestry review required",
            "retained_excluded_ancestry_matches": 0,
            "retained_row_bytes_preserved": True,
            "row_order_preserved": True,
            "labels_or_text_modified": False,
            "new_rows_added": 0,
            "benchmark_or_review_rows_added": 0,
            "held_out_corpora_opened": False,
            "training_or_model_inference_performed": False,
            "train_dev_id_text_family_overlap": 0,
            "duplicate_ids_or_normalized_text": 0,
            "family_repetition_within_split": "preserved; counterfactual siblings are intentional",
            "semantic_or_pretraining_overlap_proven_absent": False,
        },
        "original": {split: summary(rows[split]) for split in ("train", "dev")},
        "excluded": {split: summary(excluded[split]) for split in ("train", "dev")},
        "splits": {
            split: summary(retained[split]) | {
                "path": f"{output}/{split}.jsonl", "sha256": digest(output_bytes[split]),
            }
            for split in ("train", "dev")
        },
        "remaining_source_reviews": {
            source: {
                "license_declared": source_map[source]["license_declared"],
                "rights_review_status": "review_required",
                "human_review_status": "pending",
            }
            for source in sorted(ALLOWED_SOURCES)
        },
        "limitations": [
            "Publisher license records are not independent upstream-rights or privacy clearance.",
            "Existing labels are preserved, not human-verified or corrected by the AI review.",
            "The old 635-row audit is not a completed or sufficient audit of this new corpus.",
            "No existing holdout or benchmark was filtered, relabeled, or replaced.",
            "Training inputs are restricted to the pinned parent corpus, never the 96-case audit.",
            "Source subtraction does not erase influence from prior fitted model weights.",
            "Passing data integrity is not a quality, production, mobile, or publication gate.",
        ],
    }
    # Validate completely before any output; exclusive creation forbids silent reuse.
    output_path.mkdir(parents=True, exist_ok=False)
    for split in ("train", "dev"):
        with (output_path / f"{split}.jsonl").open("xb") as handle:
            handle.write(output_bytes[split])
    manifest_data = json_bytes(receipt)
    with (output_path / "manifest.json").open("xb") as handle:
        handle.write(manifest_data)
    public_report = receipt | {
        "manifest": {"path": f"{output}/manifest.json", "sha256": digest(manifest_data)}
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("xb") as handle:
        handle.write(json_bytes(public_report))
    return public_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = build(args.root)
    print(json.dumps({"train_rows": report["splits"]["train"]["rows"],
                      "dev_rows": report["splits"]["dev"]["rows"],
                      "release_authorized": False, "report": REPORT}, indent=2))


if __name__ == "__main__":
    main()
