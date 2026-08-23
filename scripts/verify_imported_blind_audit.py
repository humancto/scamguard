#!/usr/bin/env python3
"""Reconstruct and verify a blind-audit import without rewriting release evidence."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    from scamguard.metrics import file_sha256
    from scripts.import_blind_audit import import_returned_audit
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from import_blind_audit import import_returned_audit  # type: ignore[no-redef]

    from scamguard.metrics import file_sha256


def verify_imported_audit(
    *,
    returned_csv: Path,
    bundle_path: Path,
    canonical_audit: Path,
    canonical_manifest: Path,
    reviewed_audit: Path,
    report_path: Path,
) -> dict[str, object]:
    existing = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError("existing blind-audit report must be a JSON object")
    with tempfile.TemporaryDirectory(prefix="scamguard-audit-reimport-") as temporary:
        root = Path(temporary)
        reconstructed_audit = root / "reviewed.csv"
        reconstructed_report = root / "report.json"
        reconstructed = import_returned_audit(
            returned_csv,
            bundle_path,
            canonical_audit,
            canonical_manifest,
            reconstructed_audit,
            reconstructed_report,
        )
        if file_sha256(reconstructed_audit) != file_sha256(reviewed_audit):
            raise ValueError("reviewed audit differs from a fresh verified blind import")
    expected = dict(reconstructed)
    expected["path"] = str(reviewed_audit)
    if existing != expected:
        differing = sorted(
            key for key in set(existing) | set(expected) if existing.get(key) != expected.get(key)
        )
        raise ValueError(
            "existing blind-audit report differs from a fresh verified import: "
            + ", ".join(differing[:20])
        )
    if existing.get("release_gate_passed") is not True:
        raise ValueError("reconstructed blind-audit release gate is not passed")
    return {
        "artifact_schema_version": 1,
        "measurement_kind": "blind_audit_import_reconstruction_check",
        "passed": True,
        "contains_message_text": False,
        "rows": existing["rows"],
        "complete_rows": existing["complete_rows"],
        "incorrect_label_rows": existing["incorrect_label_rows"],
        "sensitive_data_rows": existing["sensitive_data_rows"],
        "bindings": {
            "bundle_sha256": file_sha256(bundle_path),
            "returned_blind_audit_sha256": file_sha256(returned_csv),
            "canonical_audit_sha256": file_sha256(canonical_audit),
            "canonical_manifest_sha256": file_sha256(canonical_manifest),
            "reviewed_audit_sha256": file_sha256(reviewed_audit),
            "completion_report_sha256": file_sha256(report_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returned-audit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--canonical-audit", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_imported_audit(
            returned_csv=args.returned_audit,
            bundle_path=args.bundle,
            canonical_audit=args.canonical_audit,
            canonical_manifest=args.canonical_manifest,
            reviewed_audit=args.reviewed_audit,
            report_path=args.report,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (json.JSONDecodeError, OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"blind-audit import verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
