#!/usr/bin/env python3
"""Verify and join a returned blind audit with the sealed canonical answer key."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

try:
    from scripts.audit_protocol import audit_protocol, audit_protocol_sha256
    from scripts.blind_audit import (
        blind_ids_sha256,
        blind_inputs_sha256,
        blind_review_id,
        canonical_sha256,
        file_sha256,
        read_csv,
        validate_blind_rows,
        validate_bundle_manifest,
    )
    from scripts.check_audit_completion import completion_result, validate_audit_binding
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from audit_protocol import audit_protocol, audit_protocol_sha256  # type: ignore[no-redef]
    from blind_audit import (  # type: ignore[no-redef]
        blind_ids_sha256,
        blind_inputs_sha256,
        blind_review_id,
        canonical_sha256,
        file_sha256,
        read_csv,
        validate_blind_rows,
        validate_bundle_manifest,
    )
    from check_audit_completion import (  # type: ignore[no-redef]
        completion_result,
        validate_audit_binding,
    )

EXPECTED_MEMBERS = {
    "README.txt",
    "review.py",
    "scamguard_blind_audit.csv",
    "scamguard_blind_audit.manifest.json",
}


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_and_verify_bundle(bundle_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(bundle_path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)) or set(members) != EXPECTED_MEMBERS:
            raise ValueError("bundle members differ from the frozen four-file layout")
        payloads = {name: archive.read(name) for name in members}
    manifest = json.loads(payloads["scamguard_blind_audit.manifest.json"])
    if manifest.get("review_app_sha256") != _bytes_sha256(payloads["review.py"]):
        raise ValueError("review application differs from the bundle manifest")
    if manifest.get("readme_sha256") != _bytes_sha256(payloads["README.txt"]):
        raise ValueError("bundle README differs from the bundle manifest")
    template = payloads["scamguard_blind_audit.csv"]
    if manifest.get("review_csv_template_sha256") != _bytes_sha256(template):
        raise ValueError("blank reviewer template differs from the bundle manifest")
    text = template.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames, rows = list(reader.fieldnames or []), list(reader)
    errors = validate_blind_rows(fieldnames, rows, require_complete=False)
    if manifest.get("selected_rows") != len(rows):
        errors.append("blank reviewer template row count differs from the bundle manifest")
    if manifest.get("selected_ids_sha256") != blind_ids_sha256(rows):
        errors.append("blank reviewer template IDs differ from the bundle manifest")
    if manifest.get("blind_inputs_sha256") != blind_inputs_sha256(rows):
        errors.append("blank reviewer template inputs differ from the bundle manifest")
    if any(
        str(row.get(field, "")).strip()
        for row in rows
        for field in ("auditor_label", "contains_sensitive_data", "notes")
    ):
        errors.append("bundle reviewer template is not blank")
    if errors:
        raise ValueError("; ".join(errors))
    return manifest


def _atomic_write_csv(output_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def import_returned_audit(
    returned_csv: Path,
    bundle_path: Path,
    canonical_audit: Path,
    canonical_manifest_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    protected_inputs = {
        returned_csv.resolve(),
        bundle_path.resolve(),
        canonical_audit.resolve(),
        canonical_manifest_path.resolve(),
    }
    if output_path.resolve() in protected_inputs or report_path.resolve() in protected_inputs:
        raise ValueError("refusing to overwrite a blind-audit input")
    if output_path.resolve() == report_path.resolve():
        raise ValueError("imported audit and report outputs must be different paths")
    if output_path.exists() or report_path.exists():
        raise ValueError("refusing to overwrite an existing imported audit or report")
    bundle = load_and_verify_bundle(bundle_path)
    canonical_manifest, binding_errors = validate_audit_binding(
        canonical_audit, canonical_manifest_path
    )
    if binding_errors:
        raise ValueError("; ".join(binding_errors))
    if bundle.get("audit_protocol_sha256") != audit_protocol_sha256():
        raise ValueError("bundle protocol hash differs from the frozen project protocol")
    if canonical_sha256(bundle.get("protocol")) != audit_protocol_sha256():
        raise ValueError("bundle protocol body differs from the frozen project protocol")
    if bundle.get("protocol") != audit_protocol():
        raise ValueError("bundle protocol content differs from the frozen project protocol")
    if bundle.get("canonical_audit_manifest_sha256") != file_sha256(canonical_manifest_path):
        raise ValueError("canonical audit manifest differs from the bundle binding")
    if bundle.get("canonical_audit_inputs_sha256") != canonical_manifest.get(
        "selected_inputs_sha256"
    ):
        raise ValueError("canonical audit inputs differ from the bundle binding")
    if bundle.get("data_manifest_sha256") != canonical_manifest.get("data_manifest_sha256"):
        raise ValueError("audited dataset manifest differs from the bundle binding")

    returned_rows, returned_errors = validate_bundle_manifest(
        bundle, returned_csv, require_complete=True
    )
    if returned_errors:
        raise ValueError("; ".join(returned_errors))
    canonical_fields, canonical_rows = read_csv(canonical_audit)
    canonical_blind = [
        {"id": blind_review_id(row.get("id", "")), "text": row.get("text", "")}
        for row in canonical_rows
    ]
    if bundle.get("selected_ids_sha256") != blind_ids_sha256(canonical_blind):
        raise ValueError("canonical audit IDs differ from the bundle")
    if bundle.get("blind_inputs_sha256") != blind_inputs_sha256(canonical_blind):
        raise ValueError("canonical audit messages differ from the bundle")

    decisions = {row["id"]: row for row in returned_rows}
    if len(decisions) != len(canonical_rows):
        raise ValueError("returned decisions do not map one-to-one to the canonical audit")
    joined: list[dict[str, str]] = []
    for row in canonical_rows:
        review_id = blind_review_id(row["id"])
        decision = decisions.get(review_id)
        if decision is None:
            raise ValueError(f"returned audit is missing opaque review ID {review_id!r}")
        label = decision["auditor_label"].strip().upper()
        project_label = row["label"].strip().upper()
        updated = dict(row)
        updated["auditor_label"] = label
        updated["label_correct"] = "yes" if label == project_label else "no"
        updated["contains_sensitive_data"] = decision["contains_sensitive_data"].strip().casefold()
        updated["notes"] = decision["notes"].strip()
        joined.append(updated)
    _atomic_write_csv(output_path, canonical_fields, joined)
    result = completion_result(output_path, canonical_manifest_path)
    result["blind_bundle_path"] = str(bundle_path)
    result["blind_bundle_sha256"] = file_sha256(bundle_path)
    result["returned_blind_audit_path"] = str(returned_csv)
    result["returned_blind_audit_sha256"] = file_sha256(returned_csv)
    result["imported_from_blind_bundle"] = True
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--returned-audit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--canonical-audit", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = import_returned_audit(
            args.returned_audit,
            args.bundle,
            args.canonical_audit,
            args.canonical_manifest,
            args.output,
            args.report,
        )
    except (KeyError, OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        raise SystemExit(f"blind audit import failed: {error}") from error
    print(json.dumps(result, indent=2))
    if not result["release_gate_passed"]:
        raise SystemExit(
            "blind audit imported, but disagreements, sensitive data, or gate errors "
            "require adjudication"
        )


if __name__ == "__main__":
    main()
