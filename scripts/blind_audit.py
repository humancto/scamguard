"""Shared, dependency-free primitives for blind label-audit handoff."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

BLIND_AUDIT_SCHEMA_VERSION: Final[int] = 1
BLIND_AUDIT_TYPE: Final[str] = "scamguard_blind_label_audit_bundle"
BLIND_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "text",
    "auditor_label",
    "contains_sensitive_data",
    "notes",
)
BLIND_IMMUTABLE_FIELDS: Final[tuple[str, ...]] = ("id", "text")
LABELS: Final[set[str]] = {"SAFE", "UNCERTAIN", "SCAM"}
BOOLEAN_DECISIONS: Final[set[str]] = {"yes", "no"}
MAX_NOTES_LENGTH: Final[int] = 2_000
REVIEW_ID_SCHEME: Final[str] = "sha256-domain-separated-128-v1"
REVIEW_ID_DOMAIN: Final[bytes] = b"scamguard-blind-review-id-v1\0"
REVIEW_ORDER: Final[str] = "opaque-review-id-lexicographic-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def blind_review_id(canonical_id: str) -> str:
    """Return a stable opaque ID without exposing source-bearing canonical identifiers."""

    digest = hashlib.sha256(REVIEW_ID_DOMAIN + canonical_id.encode()).hexdigest()
    return f"sg-{digest[:32]}"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def blind_ids_sha256(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(row.get("id", "")) for row in rows)).encode()
    ).hexdigest()


def blind_inputs_sha256(rows: list[dict[str, str]]) -> str:
    canonical = [
        {field: str(row.get(field, "")) for field in BLIND_IMMUTABLE_FIELDS}
        for row in sorted(rows, key=lambda item: str(item.get("id", "")))
    ]
    return canonical_sha256(canonical)


def validate_blind_rows(
    fieldnames: list[str], rows: list[dict[str, str]], *, require_complete: bool
) -> list[str]:
    errors: list[str] = []
    if tuple(fieldnames) != BLIND_FIELDS:
        errors.append(
            "blind CSV fields differ from the frozen schema: "
            f"expected {list(BLIND_FIELDS)!r}, got {fieldnames!r}"
        )
    if not rows:
        errors.append("blind CSV contains no rows")
    identifiers = [str(row.get("id", "")) for row in rows]
    if any(not identifier for identifier in identifiers):
        errors.append("blind CSV contains an empty ID")
    if len(set(identifiers)) != len(identifiers):
        errors.append("blind CSV contains duplicate IDs")
    for index, row in enumerate(rows, start=2):
        label = str(row.get("auditor_label", "")).strip().upper()
        sensitive = str(row.get("contains_sensitive_data", "")).strip().casefold()
        notes = str(row.get("notes", ""))
        populated = bool(label or sensitive or notes.strip())
        if require_complete or populated:
            if label not in LABELS:
                errors.append(f"row {index}: invalid auditor_label {label!r}")
            if sensitive not in BOOLEAN_DECISIONS:
                errors.append(f"row {index}: invalid contains_sensitive_data {sensitive!r}")
        if len(notes) > MAX_NOTES_LENGTH:
            errors.append(f"row {index}: notes exceed {MAX_NOTES_LENGTH} characters")
    return errors


def validate_bundle_manifest(
    manifest: dict[str, object], csv_path: Path, *, require_complete: bool
) -> tuple[list[dict[str, str]], list[str]]:
    fieldnames, rows = read_csv(csv_path)
    errors = validate_blind_rows(fieldnames, rows, require_complete=require_complete)
    if manifest.get("artifact_schema_version") != BLIND_AUDIT_SCHEMA_VERSION:
        errors.append("blind bundle schema version is unsupported")
    if manifest.get("artifact_type") != BLIND_AUDIT_TYPE:
        errors.append("blind bundle artifact type is invalid")
    if manifest.get("blind_fields") != list(BLIND_FIELDS):
        errors.append("blind bundle field declaration differs from the frozen schema")
    if manifest.get("review_id_scheme") != REVIEW_ID_SCHEME:
        errors.append("blind bundle review-ID scheme differs from the frozen schema")
    if manifest.get("review_order") != REVIEW_ORDER:
        errors.append("blind bundle review order differs from the frozen schema")
    if manifest.get("selected_rows") != len(rows):
        errors.append("blind CSV row count differs from its bundle manifest")
    if manifest.get("selected_ids_sha256") != blind_ids_sha256(rows):
        errors.append("blind CSV IDs differ from the bundle manifest")
    if manifest.get("blind_inputs_sha256") != blind_inputs_sha256(rows):
        errors.append("blind CSV immutable id/text inputs differ from the bundle manifest")
    return rows, errors
