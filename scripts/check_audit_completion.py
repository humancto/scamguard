#!/usr/bin/env python3
"""Fail closed until the stratified label audit has independent decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scamguard.metrics import file_sha256

try:
    from scripts.audit_protocol import AUDIT_PROTOCOL_VERSION, audit_protocol_sha256
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from audit_protocol import (  # type: ignore[no-redef]
        AUDIT_PROTOCOL_VERSION,
        audit_protocol_sha256,
    )

LABEL_ORDER = ("SAFE", "UNCERTAIN", "SCAM")
LABELS = set(LABEL_ORDER)
BOOLEANS = {"true": True, "false": False, "yes": True, "no": False}
IMMUTABLE_AUDIT_FIELDS = (
    "id",
    "dataset_split",
    "source",
    "source_label",
    "label",
    "category",
    "text",
)


def audit_input_sha256(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    canonical = [
        {field: str(row.get(field, "")) for field in IMMUTABLE_AUDIT_FIELDS}
        for row in sorted(rows, key=lambda item: str(item.get("id", "")))
    ]
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def audit_ids_sha256(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        identifiers = sorted(str(row.get("id", "")) for row in csv.DictReader(handle))
    return hashlib.sha256("\n".join(identifiers).encode()).hexdigest()


def wilson_lower_bound(successes: int, total: int, z: float = 1.959963984540054) -> float | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * total)) / total
    )
    return (center - margin) / denominator


def cohen_kappa(
    project_counts: Counter[str],
    auditor_counts: Counter[str],
    agreements: int,
    total: int,
) -> float | None:
    if total <= 0:
        return None
    observed = agreements / total
    expected = sum(
        project_counts[label] * auditor_counts[label] for label in LABEL_ORDER
    ) / (total * total)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1 - expected)


def grouped_agreement(
    totals: Counter[str], agreements: Counter[str]
) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "rows": totals[key],
            "agreements": agreements[key],
            "agreement": agreements[key] / totals[key],
            "agreement_wilson_95_lower": wilson_lower_bound(
                agreements[key], totals[key]
            ),
        }
        for key in sorted(totals)
    }


def audit_summary(path: Path) -> tuple[dict[str, object], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    complete = 0
    agreements = 0
    sensitive_rows = 0
    audited_labels: Counter[str] = Counter()
    project_labels: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    source_totals: Counter[str] = Counter()
    source_agreements: Counter[str] = Counter()
    project_label_agreements: Counter[str] = Counter()
    disagreement_ids: list[str] = []
    for index, row in enumerate(rows, start=2):
        auditor_label = str(row.get("auditor_label", "")).strip().upper()
        project_label = str(row.get("label", "")).strip().upper()
        correct_text = str(row.get("label_correct", "")).strip().casefold()
        sensitive_text = str(row.get("contains_sensitive_data", "")).strip().casefold()
        populated = bool(auditor_label or correct_text or sensitive_text)
        if not populated:
            continue
        if auditor_label not in LABELS:
            errors.append(f"row {index}: invalid auditor_label {auditor_label!r}")
            continue
        if project_label not in LABELS:
            errors.append(f"row {index}: invalid project label {project_label!r}")
            continue
        if correct_text not in BOOLEANS:
            errors.append(f"row {index}: invalid label_correct {correct_text!r}")
            continue
        if sensitive_text not in BOOLEANS:
            errors.append(
                f"row {index}: invalid contains_sensitive_data {sensitive_text!r}"
            )
            continue
        agrees = auditor_label == project_label
        if BOOLEANS[correct_text] != agrees:
            errors.append(
                f"row {index}: label_correct conflicts with label/auditor_label agreement"
            )
            continue
        complete += 1
        agreements += int(agrees)
        sensitive_rows += int(BOOLEANS[sensitive_text])
        audited_labels[auditor_label] += 1
        project_labels[project_label] += 1
        confusion[(project_label, auditor_label)] += 1
        source = str(row.get("source", "unknown")).strip() or "unknown"
        source_totals[source] += 1
        source_agreements[source] += int(agrees)
        project_label_agreements[project_label] += int(agrees)
        if not agrees:
            disagreement_ids.append(str(row.get("id", f"row-{index}")))
    total = len(rows)
    agreement = agreements / complete if complete else None
    return (
        {
            "path": str(path),
            "rows": total,
            "complete_rows": complete,
            "incomplete_rows": total - complete,
            "agreement": agreement,
            "agreement_wilson_95_lower": wilson_lower_bound(agreements, complete),
            "cohen_kappa": cohen_kappa(
                project_labels, audited_labels, agreements, complete
            ),
            "incorrect_label_rows": complete - agreements,
            "sensitive_data_rows": sensitive_rows,
            "auditor_labels": dict(audited_labels),
            "project_labels": dict(project_labels),
            "agreement_by_project_label": grouped_agreement(
                project_labels, project_label_agreements
            ),
            "agreement_by_source": grouped_agreement(
                source_totals, source_agreements
            ),
            "confusion_matrix": {
                project: {
                    auditor: confusion[(project, auditor)] for auditor in LABEL_ORDER
                }
                for project in LABEL_ORDER
            },
            "disagreement_ids": disagreement_ids,
            "release_gate_passed": (
                total > 0
                and complete == total
                and agreements == complete
                and sensitive_rows == 0
                and not errors
            ),
        },
        errors,
    )


def validate_audit_binding(
    audit_path: Path, manifest_path: Path
) -> tuple[dict[str, object], list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("artifact_schema_version") != 2:
        errors.append("audit manifest schema version is not 2")
    if manifest.get("audit_protocol_version") != AUDIT_PROTOCOL_VERSION:
        errors.append("audit protocol version differs from the frozen reviewer rubric")
    if manifest.get("audit_protocol_sha256") != audit_protocol_sha256():
        errors.append("audit protocol SHA-256 differs from the frozen reviewer rubric")
    if manifest.get("selected_inputs_sha256") != audit_input_sha256(audit_path):
        errors.append("audit immutable inputs differ from their manifest")
    if manifest.get("selected_ids_sha256") != audit_ids_sha256(audit_path):
        errors.append("audit selected IDs differ from their manifest")
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit_rows = sum(1 for _ in csv.DictReader(handle))
    if manifest.get("selected_rows") != audit_rows:
        errors.append("audit row count differs from its manifest")
    data_manifest = Path(str(manifest.get("data_manifest_path", "")))
    if not data_manifest.is_file():
        errors.append("audited dataset manifest is missing")
    elif manifest.get("data_manifest_sha256") != file_sha256(data_manifest):
        errors.append("audited dataset manifest hash differs")
    return manifest, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=Path("data/audit/label_audit.csv"))
    parser.add_argument("--audit-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary, errors = audit_summary(args.audit)
    manifest_path = args.audit_manifest or args.audit.with_suffix(".manifest.json")
    binding: dict[str, object] | None = None
    if manifest_path.is_file():
        binding, binding_errors = validate_audit_binding(args.audit, manifest_path)
        errors.extend(binding_errors)
    else:
        errors.append(f"audit manifest is missing: {manifest_path}")
    summary["release_gate_passed"] = bool(summary["release_gate_passed"] and not errors)
    result = {
        **summary,
        "audit_sha256": file_sha256(args.audit),
        "audit_manifest_path": str(manifest_path),
        "audit_manifest_sha256": file_sha256(manifest_path) if manifest_path.is_file() else None,
        "audit_protocol_version": binding.get("audit_protocol_version") if binding else None,
        "audit_protocol_sha256": binding.get("audit_protocol_sha256") if binding else None,
        "data_manifest_path": binding.get("data_manifest_path") if binding else None,
        "data_manifest_sha256": binding.get("data_manifest_sha256") if binding else None,
        "errors": errors[:20],
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not summary["release_gate_passed"]:
        raise SystemExit("independent label-audit release gate is incomplete")


if __name__ == "__main__":
    main()
