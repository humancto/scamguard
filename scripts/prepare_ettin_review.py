#!/usr/bin/env python3
"""Freeze the schema-25 supervision audit for sg-ettin150m-bounded-v1.

Reads only the pinned train/dev files, never evaluation labels or predictions.
The enriched sample is a label-quality audit, not a population accuracy estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

try:
    from scripts.audit_protocol import AUDIT_PROTOCOL_VERSION, audit_protocol_sha256
    from scripts.build_blind_audit_bundle import build_bundle
    from scripts.check_audit_completion import (
        IMMUTABLE_AUDIT_FIELDS,
        audit_ids_sha256,
        audit_input_sha256,
    )
    from scripts.create_audit_sample import audit_key, read_jsonl
except ModuleNotFoundError:
    from audit_protocol import (  # type: ignore[no-redef]
        AUDIT_PROTOCOL_VERSION,
        audit_protocol_sha256,
    )
    from build_blind_audit_bundle import build_bundle  # type: ignore[no-redef]
    from check_audit_completion import (  # type: ignore[no-redef]
        IMMUTABLE_AUDIT_FIELDS,
        audit_ids_sha256,
        audit_input_sha256,
    )
    from create_audit_sample import audit_key, read_jsonl  # type: ignore[no-redef]

EXPERIMENT_ID = "sg-ettin150m-bounded-v1"
SEED = "sg-ettin150m-bounded-v1-review-v1"
LABEL_TARGETS = {"SAFE": 212, "SCAM": 212, "UNCERTAIN": 211}
STRATUM_MINIMUM = 6
PHONE_MINIMUM_PER_LABEL = 24
PHONE_SOURCE = "shakeleoatmeal_phone_scam_synthetic"
PINNED_HASHES = {
    "manifest.json": "6c1096f1017edb10b690fde8968d9de892d2bff8ae49b9183c5da15e0380ad57",
    "train.jsonl": "d4e3e100e97a937a789aa984d4ce75b545592886099a0027dbd758bbf0b2e614",
    "dev.jsonl": "04201ee86325306cf1751dbf848ddca360386de707069c0c4bba0c9c28a21c07",
}
DECISION_FIELDS = ("auditor_label", "label_correct", "contains_sensitive_data", "notes")
Stratum = tuple[str, str, str, str]


def stratum(row: dict[str, Any]) -> Stratum:
    return (
        str(row["source"]),
        str(row.get("source_domain", "")),
        str(row["label"]),
        str(row["dataset_split"]),
    )


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anchor every source/domain/label/split, then balance within each label."""
    groups: dict[Stratum, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in seen:
            raise ValueError("frame contains an empty or duplicate ID")
        if row.get("dataset_split") not in {"train", "dev"}:
            raise ValueError("only train and dev may enter this prospective audit")
        if row.get("label") not in LABEL_TARGETS or not row.get("text") or not row.get("source"):
            raise ValueError("frame contains an invalid label, source, or empty text")
        seen.add(identifier)
        groups[stratum(row)].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: (audit_key(row, SEED), str(row["id"])))

    selected: dict[str, dict[str, Any]] = {}
    for key in sorted(groups):
        for row in groups[key][:STRATUM_MINIMUM]:
            selected[str(row["id"])] = row
    for label in ("SAFE", "SCAM"):
        phone = sorted(
            (row for row in rows if row["source"] == PHONE_SOURCE and row["label"] == label),
            key=lambda row: (audit_key(row, SEED), str(row["id"])),
        )
        if len(phone) < PHONE_MINIMUM_PER_LABEL:
            raise ValueError(f"insufficient phone-scam {label} examples")
        current = sum(
            row["source"] == PHONE_SOURCE and row["label"] == label for row in selected.values()
        )
        for row in phone:
            if current >= PHONE_MINIMUM_PER_LABEL:
                break
            if str(row["id"]) not in selected:
                selected[str(row["id"])] = row
                current += 1

    for label, target in LABEL_TARGETS.items():
        current = sum(row["label"] == label for row in selected.values())
        if current > target:
            raise ValueError(f"mandatory strata exceed {label} target")
        remaining = {
            key: iter(row for row in groups[key] if str(row["id"]) not in selected)
            for key in sorted(groups)
            if key[2] == label
        }
        while current < target:
            progressed = False
            for candidates in remaining.values():
                row = next(candidates, None)
                if row is None:
                    continue
                selected[str(row["id"])] = row
                current += 1
                progressed = True
                if current == target:
                    break
            if not progressed:
                raise ValueError(f"insufficient {label} examples for frozen target")
    return sorted(selected.values(), key=lambda row: (row["source"], row["label"], row["id"]))


def prepare_package(
    data: Path,
    output: Path,
    report: Path,
    *,
    expected_hashes: dict[str, str] = PINNED_HASHES,
) -> dict[str, Any]:
    if output.exists() or report.exists():
        raise ValueError("refusing to overwrite an existing review directory or report")
    hashes = {name: file_sha256(data / name) for name in PINNED_HASHES}
    if hashes != expected_hashes:
        raise ValueError("dataset files differ from the frozen schema-25 frame")
    if json.loads((data / "manifest.json").read_text())["schema_version"] != 25:
        raise ValueError("review requires schema 25")
    rows = [
        row | {"dataset_split": split}
        for split in ("train", "dev")
        for row in read_jsonl(data / f"{split}.jsonl")
    ]
    selected = select_rows(rows)
    frame_counts = Counter(stratum(row) for row in rows)
    sample_counts = Counter(stratum(row) for row in selected)
    source_files = {
        f"{split}.jsonl": {
            "path": str(data / f"{split}.jsonl"),
            "sha256": hashes[f"{split}.jsonl"],
            "rows": sum(row["dataset_split"] == split for row in rows),
        }
        for split in ("train", "dev")
    }
    selection = {
        "experiment_id": EXPERIMENT_ID,
        "sampling_frame": "schema25 train+dev only; no evaluation labels or predictions",
        "seed": SEED,
        "label_targets": LABEL_TARGETS,
        "minimum_per_source_domain_label_split": STRATUM_MINIMUM,
        "minimum_per_phone_label": PHONE_MINIMUM_PER_LABEL,
        "selection_method": (
            "SHA256(seed:id) rank; stratum anchors; phone anchors; label round-robin"
        ),
        "interpretation": "enriched supervision audit; not population-weighted accuracy evidence",
        "frame_rows": len(rows),
        "frame_sources": len({row["source"] for row in rows}),
        "selected_sources": len({row["source"] for row in selected}),
        "selected_labels": dict(sorted(Counter(row["label"] for row in selected).items())),
        "selected_splits": dict(sorted(Counter(row["dataset_split"] for row in selected).items())),
        "strata": [
            {
                "source": key[0],
                "domain": key[1],
                "label": key[2],
                "split": key[3],
                "available": count,
                "selected": sample_counts[key],
            }
            for key, count in sorted(frame_counts.items())
        ],
    }
    output.mkdir(parents=True)
    audit = output / "label-audit.csv"
    manifest_path = output / "label-audit.manifest.json"
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*IMMUTABLE_AUDIT_FIELDS, *DECISION_FIELDS))
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in IMMUTABLE_AUDIT_FIELDS}
            | dict.fromkeys(DECISION_FIELDS, "")
            for row in selected
        )
    manifest = {
        "artifact_schema_version": 2,
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "audit_protocol_sha256": audit_protocol_sha256(),
        "audit_path": str(audit),
        "audit_template_sha256": file_sha256(audit),
        "selected_inputs_sha256": audit_input_sha256(audit),
        "selected_ids_sha256": audit_ids_sha256(audit),
        "selected_rows": len(selected),
        "data_manifest_path": str(data / "manifest.json"),
        "data_manifest_sha256": hashes["manifest.json"],
        "source_files": source_files,
        "preparation_script_sha256": file_sha256(Path(__file__)),
        **selection,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    bundle = build_bundle(
        audit,
        manifest_path,
        Path(__file__).with_name("review_blind_audit.py"),
        output / "scamguard-ettin150m-v1-blind-audit.zip",
    )
    result = {
        "artifact_type": "prospective_independent_review_preparation",
        "artifact_schema_version": 1,
        **selection,
        "source_files": source_files,
        "data_manifest_sha256": hashes["manifest.json"],
        "canonical_audit": str(audit),
        "canonical_audit_sha256": file_sha256(audit),
        "canonical_manifest": str(manifest_path),
        "selected_inputs_sha256": manifest["selected_inputs_sha256"],
        "selected_ids_sha256": manifest["selected_ids_sha256"],
        "bundle": bundle,
        "independent_human_complete_rows": 0,
        "release_gate_passed": False,
        "reviewer_identity_attestation": "required separately; not verified by CSV importer",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/experiments/schema25-full-call-curriculum/processed"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/audit/ettin150m-v1"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/ettin150m-v1-review-preparation.json")
    )
    args = parser.parse_args()
    try:
        result = prepare_package(args.data, args.output, args.report)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"review preparation failed: {error}") from error
    print(json.dumps({key: value for key, value in result.items() if key != "strata"}, indent=2))


if __name__ == "__main__":
    main()
