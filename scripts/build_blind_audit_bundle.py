#!/usr/bin/env python3
"""Build a deterministic, answer-key-free label-audit handoff ZIP."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

try:
    from scripts.audit_protocol import audit_protocol, audit_protocol_sha256
    from scripts.blind_audit import (
        BLIND_AUDIT_SCHEMA_VERSION,
        BLIND_AUDIT_TYPE,
        BLIND_FIELDS,
        REVIEW_ID_SCHEME,
        REVIEW_ORDER,
        blind_ids_sha256,
        blind_inputs_sha256,
        blind_review_id,
        file_sha256,
    )
    from scripts.check_audit_completion import validate_audit_binding
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from audit_protocol import audit_protocol, audit_protocol_sha256  # type: ignore[no-redef]
    from blind_audit import (  # type: ignore[no-redef]
        BLIND_AUDIT_SCHEMA_VERSION,
        BLIND_AUDIT_TYPE,
        BLIND_FIELDS,
        REVIEW_ID_SCHEME,
        REVIEW_ORDER,
        blind_ids_sha256,
        blind_inputs_sha256,
        blind_review_id,
        file_sha256,
    )
    from check_audit_completion import validate_audit_binding  # type: ignore[no-redef]

REVIEW_CSV = "scamguard_blind_audit.csv"
MANIFEST = "scamguard_blind_audit.manifest.json"
REVIEW_APP = "review.py"
README = "README.txt"
ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BLIND_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_deterministic_zip(source: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.iterdir(), key=lambda item: item.name):
            info = zipfile.ZipInfo(path.name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.name == REVIEW_APP else 0o644) << 16
            archive.writestr(info, path.read_bytes())


def build_bundle(
    audit_path: Path,
    audit_manifest_path: Path,
    review_app_path: Path,
    output_path: Path,
    *,
    replace: bool = False,
) -> dict[str, object]:
    protected_inputs = {
        audit_path.resolve(),
        audit_manifest_path.resolve(),
        review_app_path.resolve(),
    }
    if output_path.resolve() in protected_inputs:
        raise ValueError("refusing to overwrite a bundle input")
    if output_path.exists() and not replace:
        raise ValueError(f"refusing to overwrite existing bundle: {output_path}")
    canonical_manifest, errors = validate_audit_binding(audit_path, audit_manifest_path)
    if errors:
        raise ValueError("; ".join(errors))
    with audit_path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("canonical audit contains no rows")
    decision_fields = ("auditor_label", "label_correct", "contains_sensitive_data", "notes")
    if any(str(row.get(field, "")).strip() for row in source_rows for field in decision_fields):
        raise ValueError("canonical audit is not a blank reviewer template")
    blind_rows = [
        {
            "id": blind_review_id(str(row.get("id", ""))),
            "text": str(row.get("text", "")),
            "auditor_label": "",
            "contains_sensitive_data": "",
            "notes": "",
        }
        for row in source_rows
    ]
    blind_rows.sort(key=lambda row: row["id"])
    if any(not row["id"] or not row["text"] for row in blind_rows):
        raise ValueError("canonical audit contains an empty ID or message")
    if len({row["id"] for row in blind_rows}) != len(blind_rows):
        raise ValueError("canonical audit contains duplicate IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scamguard-blind-audit-") as temporary:
        root = Path(temporary)
        review_csv = root / REVIEW_CSV
        _write_csv(review_csv, blind_rows)
        shutil.copyfile(review_app_path, root / REVIEW_APP)
        (root / REVIEW_APP).chmod(0o755)
        readme_text = (
            "ScamGuard independent blind label audit\n"
            "=======================================\n\n"
            "Requirements: Python 3.11+ and a modern browser. No packages or network access.\n\n"
            "1. Extract this ZIP into its own folder.\n"
            "2. Run: python3 review.py --open\n"
            "3. Review every message using only the frozen rubric in the page.\n"
            "4. Stop the server with Ctrl-C.\n"
            "5. Return scamguard_blind_audit.csv to the project owner.\n\n"
            "Do not seek the project labels or source metadata. The returned file is verified\n"
            "against this bundle before it is joined to the sealed answer key.\n"
        )
        (root / README).write_text(readme_text, encoding="utf-8")
        manifest: dict[str, object] = {
            "artifact_schema_version": BLIND_AUDIT_SCHEMA_VERSION,
            "artifact_type": BLIND_AUDIT_TYPE,
            "audit_protocol_version": audit_protocol()["version"],
            "audit_protocol_sha256": audit_protocol_sha256(),
            "protocol": audit_protocol(),
            "selected_rows": len(blind_rows),
            "selected_ids_sha256": blind_ids_sha256(blind_rows),
            "blind_inputs_sha256": blind_inputs_sha256(blind_rows),
            "blind_fields": list(BLIND_FIELDS),
            "review_id_scheme": REVIEW_ID_SCHEME,
            "review_order": REVIEW_ORDER,
            "review_csv_filename": REVIEW_CSV,
            "review_csv_template_sha256": file_sha256(review_csv),
            "review_app_filename": REVIEW_APP,
            "review_app_sha256": file_sha256(root / REVIEW_APP),
            "readme_filename": README,
            "readme_sha256": file_sha256(root / README),
            "canonical_audit_inputs_sha256": canonical_manifest["selected_inputs_sha256"],
            "canonical_audit_manifest_sha256": file_sha256(audit_manifest_path),
            "data_manifest_sha256": canonical_manifest["data_manifest_sha256"],
        }
        (root / MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
            _write_deterministic_zip(root, Path(temporary_name))
            os.replace(temporary_name, output_path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
    return {
        "bundle": str(output_path),
        "bundle_sha256": file_sha256(output_path),
        "rows": len(blind_rows),
        "blind_inputs_sha256": manifest["blind_inputs_sha256"],
        "canonical_audit_manifest_sha256": manifest["canonical_audit_manifest_sha256"],
        "contains_answer_key": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--review-app", type=Path, default=Path("scripts/review_blind_audit.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replace", action="store_true", help="Atomically replace an existing generated ZIP."
    )
    args = parser.parse_args()
    try:
        result = build_bundle(
            args.audit,
            args.audit_manifest,
            args.review_app,
            args.output,
            replace=args.replace,
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"blind audit bundle failed: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
