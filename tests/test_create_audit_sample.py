from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from scripts.check_audit_completion import validate_audit_binding
from scripts.create_audit_sample import main

STANDARD_SPLITS = (
    "train",
    "dev",
    "test",
    "ood_financial",
    "ood_wspr",
    "forum_validation",
    "ood_forum",
)


def row(identifier: str, split: str, schema24: bool = False) -> dict[str, object]:
    return {
        "id": identifier,
        "text": f"Routine service example {identifier}.",
        "label": "SAFE",
        "category": "NONE",
        "source": "fixture",
        "source_label": "routine",
        "split": split,
        "family_id": f"family-{identifier}",
        "source_domain": "finance",
        "annotation_stratum": "sensitive_service",
        "schema24_admitted": schema24,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def test_schema24_audit_forces_increment_and_extra_splits(
    tmp_path: Path, monkeypatch
) -> None:
    data = tmp_path / "processed"
    data.mkdir()
    for split in STANDARD_SPLITS:
        rows = [row(f"standard-{split}", split)] if split == "train" else []
        write_jsonl(data / f"{split}.jsonl", rows)
    with (data / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row("schema24-train", "train", schema24=True)) + "\n")
    write_jsonl(
        data / "multidogo_annotation_dev.jsonl",
        [row("schema24-dev", "validation")],
    )
    write_jsonl(
        data / "multidogo_annotation_test.jsonl",
        [row("schema24-test", "validation")],
    )
    (data / "manifest.json").write_text('{"schema_version":24}', encoding="utf-8")
    audit = tmp_path / "audit.csv"
    audit_manifest = tmp_path / "audit.manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_audit_sample.py",
            "--data",
            str(data),
            "--output",
            str(audit),
            "--manifest-output",
            str(audit_manifest),
            "--per-stratum",
            "1",
            "--extra-split",
            "multidogo_annotation_dev",
            "--extra-split",
            "multidogo_annotation_test",
        ],
    )

    main()

    with audit.open(encoding="utf-8", newline="") as handle:
        ids = {item["id"] for item in csv.DictReader(handle)}
    assert {"schema24-train", "schema24-dev", "schema24-test"} <= ids
    manifest = json.loads(audit_manifest.read_text(encoding="utf-8"))
    assert len(manifest["schema24_strata"]) == 3
    _, errors = validate_audit_binding(audit, audit_manifest)
    assert errors == []

    with audit.open(encoding="utf-8", newline="") as handle:
        audited_rows = list(csv.DictReader(handle))
        fieldnames = list(audited_rows[0])
    audited_rows[0]["auditor_label"] = audited_rows[0]["label"]
    audited_rows[0]["label_correct"] = "yes"
    audited_rows[0]["contains_sensitive_data"] = "no"
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audited_rows)
    _, errors = validate_audit_binding(audit, audit_manifest)
    assert errors == []

    audited_rows[0]["text"] = "tampered audited text"
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audited_rows)
    _, errors = validate_audit_binding(audit, audit_manifest)
    assert "audit immutable inputs differ from their manifest" in errors
