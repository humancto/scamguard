from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.check_audit_completion import audit_ids_sha256, audit_input_sha256
from scripts.serve_label_audit import audit_state, update_audit_row

FIELDS = (
    "id",
    "dataset_split",
    "source",
    "source_label",
    "label",
    "category",
    "text",
    "auditor_label",
    "label_correct",
    "contains_sensitive_data",
    "notes",
)


def audit_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_manifest = tmp_path / "data-manifest.json"
    data_manifest.write_text('{"schema_version":24}\n', encoding="utf-8")
    audit = tmp_path / "audit.csv"
    rows = [
        {
            "id": "safe-1",
            "dataset_split": "dev",
            "source": "publisher",
            "source_label": "book_flight",
            "label": "SAFE",
            "category": "NONE",
            "text": "I can help change that reservation.",
            "auditor_label": "",
            "label_correct": "",
            "contains_sensitive_data": "",
            "notes": "",
        },
        {
            "id": "scam-1",
            "dataset_split": "test",
            "source": "synthetic",
            "source_label": "SCAM",
            "label": "SCAM",
            "category": "CREDENTIAL_THEFT",
            "text": "Send me the verification code now.",
            "auditor_label": "",
            "label_correct": "",
            "contains_sensitive_data": "",
            "notes": "",
        },
    ]
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = tmp_path / "audit.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "selected_inputs_sha256": audit_input_sha256(audit),
                "selected_ids_sha256": audit_ids_sha256(audit),
                "selected_rows": 2,
                "data_manifest_path": str(data_manifest),
                "data_manifest_sha256": file_sha256(data_manifest),
            }
        ),
        encoding="utf-8",
    )
    return audit, manifest


def test_state_hides_project_labels_and_source_metadata(tmp_path: Path) -> None:
    audit, manifest = audit_fixture(tmp_path)

    state = audit_state(audit, manifest, None)

    assert set(state["row"]) == {
        "id",
        "index",
        "text",
        "auditor_label",
        "contains_sensitive_data",
        "notes",
        "complete",
    }
    assert "label" not in state["row"]
    assert "source" not in state["row"]


def test_update_derives_agreement_and_advances_to_incomplete_row(tmp_path: Path) -> None:
    audit, manifest = audit_fixture(tmp_path)

    state = update_audit_row(audit, manifest, "safe-1", "safe", False, "looks routine")

    with audit.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["auditor_label"] == "SAFE"
    assert rows[0]["label_correct"] == "yes"
    assert rows[0]["contains_sensitive_data"] == "no"
    assert rows[0]["notes"] == "looks routine"
    assert state["row"]["id"] == "scam-1"
    assert state["complete"] == 1


def test_update_records_disagreement_without_revealing_expected_label(tmp_path: Path) -> None:
    audit, manifest = audit_fixture(tmp_path)

    update_audit_row(audit, manifest, "scam-1", "UNCERTAIN", True, "phone number")

    with audit.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["label_correct"] == "no"
    assert rows[1]["contains_sensitive_data"] == "yes"


def test_update_refuses_immutable_audit_tampering(tmp_path: Path) -> None:
    audit, manifest = audit_fixture(tmp_path)
    text = audit.read_text(encoding="utf-8").replace("change that reservation", "send a wire")
    audit.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="immutable inputs"):
        update_audit_row(audit, manifest, "safe-1", "SAFE", False, "")


def test_update_rejects_invalid_decisions(tmp_path: Path) -> None:
    audit, manifest = audit_fixture(tmp_path)

    with pytest.raises(ValueError, match="auditor_label"):
        update_audit_row(audit, manifest, "safe-1", "MAYBE", False, "")
    with pytest.raises(ValueError, match="boolean"):
        update_audit_row(audit, manifest, "safe-1", "SAFE", "no", "")  # type: ignore[arg-type]
