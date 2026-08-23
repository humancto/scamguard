from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from scamguard.metrics import file_sha256
from scripts.audit_protocol import AUDIT_PROTOCOL_VERSION, audit_protocol_sha256
from scripts.build_blind_audit_bundle import build_bundle
from scripts.check_audit_completion import audit_ids_sha256, audit_input_sha256
from scripts.import_blind_audit import import_returned_audit, load_and_verify_bundle

CANONICAL_FIELDS = (
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
BLIND_FIELDS = (
    "id",
    "text",
    "auditor_label",
    "contains_sensitive_data",
    "notes",
)


def canonical_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_manifest = tmp_path / "data-manifest.json"
    data_manifest.write_text('{"schema_version":24}\n', encoding="utf-8")
    audit = tmp_path / "canonical.csv"
    rows = [
        {
            "id": "safe-1",
            "dataset_split": "test",
            "source": "licensed-dialogue",
            "source_label": "book_flight",
            "label": "SAFE",
            "category": "NONE",
            "text": "I can move your reservation to Tuesday.",
            "auditor_label": "",
            "label_correct": "",
            "contains_sensitive_data": "",
            "notes": "",
        },
        {
            "id": "scam-1",
            "dataset_split": "ood_forum",
            "source": "licensed-forum",
            "source_label": "smishing",
            "label": "SCAM",
            "category": "CREDENTIAL_THEFT",
            "text": "Send the verification code now to stop suspension.",
            "auditor_label": "",
            "label_correct": "",
            "contains_sensitive_data": "",
            "notes": "",
        },
    ]
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = tmp_path / "canonical.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_schema_version": 2,
                "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
                "audit_protocol_sha256": audit_protocol_sha256(),
                "selected_inputs_sha256": audit_input_sha256(audit),
                "selected_ids_sha256": audit_ids_sha256(audit),
                "selected_rows": len(rows),
                "data_manifest_path": str(data_manifest),
                "data_manifest_sha256": file_sha256(data_manifest),
            }
        ),
        encoding="utf-8",
    )
    return audit, manifest


def build_fixture_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    audit, manifest = canonical_fixture(tmp_path)
    bundle = tmp_path / "blind-audit.zip"
    review_app = Path(__file__).parents[1] / "scripts" / "review_blind_audit.py"
    build_bundle(audit, manifest, review_app, bundle)
    return audit, manifest, bundle


def extract_bundle(bundle: Path, destination: Path) -> None:
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(destination)


def complete_returned_csv(path: Path, labels_by_text: dict[str, str]) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BLIND_FIELDS)
        writer.writeheader()
        for row in rows:
            row["auditor_label"] = labels_by_text[row["text"]]
            row["contains_sensitive_data"] = "no"
            writer.writerow(row)


def test_bundle_contains_no_answer_key_and_runs_with_isolated_python(tmp_path: Path) -> None:
    _, _, bundle = build_fixture_bundle(tmp_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    extract_bundle(bundle, extracted)

    with (extracted / "scamguard_blind_audit.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == BLIND_FIELDS
    assert all(set(row) == set(BLIND_FIELDS) for row in rows)
    assert all(re.fullmatch(r"sg-[0-9a-f]{32}", row["id"]) for row in rows)
    assert all("licensed" not in row["id"] and "safe" not in row["id"] for row in rows)
    assert all(not row["auditor_label"] for row in rows)
    assert "source" not in reader.fieldnames
    assert "label" not in reader.fieldnames
    assert load_and_verify_bundle(bundle)["selected_rows"] == 2

    checked = subprocess.run(
        [sys.executable, "-I", str(extracted / "review.py"), "--check"],
        cwd=extracted,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(checked.stdout)
    assert result == {
        "valid": True,
        "rows": 2,
        "complete_rows": 0,
        "remaining_rows": 2,
        "contains_answer_key": False,
    }


def test_isolated_reviewer_server_saves_and_resumes(tmp_path: Path) -> None:
    _, _, bundle = build_fixture_bundle(tmp_path)
    extracted = tmp_path / "server"
    extracted.mkdir()
    extract_bundle(bundle, extracted)
    process = subprocess.Popen(
        [sys.executable, "-I", "-u", str(extracted / "review.py"), "--port", "0"],
        cwd=extracted,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        url_line = process.stdout.readline().strip()
        assert url_line.startswith("Independent blind-audit UI: http://127.0.0.1:")
        url = url_line.removeprefix("Independent blind-audit UI: ")
        with urlopen(url, timeout=5) as response:  # noqa: S310 - fixture loopback URL.
            html = response.read().decode()
        token_match = re.search(r"const token=(\"(?:[^\"\\]|\\.)*\")", html)
        assert token_match is not None
        token = json.loads(token_match.group(1))
        with urlopen(f"{url}api/state", timeout=5) as response:  # noqa: S310
            initial = json.loads(response.read())
        request = Request(
            f"{url}api/row",
            data=json.dumps(
                {
                    "id": initial["row"]["id"],
                    "auditor_label": "SAFE",
                    "contains_sensitive_data": False,
                    "notes": "reviewed independently",
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "X-ScamGuard-Audit-Token": token,
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:  # noqa: S310
            saved = json.loads(response.read())
        assert saved["complete"] == 1
        assert saved["remaining"] == 1
    finally:
        process.terminate()
        process.wait(timeout=5)

    checked = subprocess.run(
        [sys.executable, "-I", str(extracted / "review.py"), "--check"],
        cwd=extracted,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(checked.stdout)["complete_rows"] == 1


def test_completed_return_is_verified_joined_and_passes_gate(tmp_path: Path) -> None:
    audit, manifest, bundle = build_fixture_bundle(tmp_path)
    extracted = tmp_path / "returned"
    extracted.mkdir()
    extract_bundle(bundle, extracted)
    returned = extracted / "scamguard_blind_audit.csv"
    complete_returned_csv(
        returned,
        {
            "I can move your reservation to Tuesday.": "SAFE",
            "Send the verification code now to stop suspension.": "SCAM",
        },
    )
    output = tmp_path / "reviewed.csv"
    report = tmp_path / "completion.json"

    result = import_returned_audit(returned, bundle, audit, manifest, output, report)

    assert result["release_gate_passed"] is True
    assert result["agreement"] == 1.0
    assert result["imported_from_blind_bundle"] is True
    with output.open(encoding="utf-8", newline="") as handle:
        joined = {row["id"]: row for row in csv.DictReader(handle)}
    assert joined["safe-1"]["label_correct"] == "yes"
    assert joined["scam-1"]["label_correct"] == "yes"


def test_import_rejects_message_tampering(tmp_path: Path) -> None:
    audit, manifest, bundle = build_fixture_bundle(tmp_path)
    extracted = tmp_path / "tampered"
    extracted.mkdir()
    extract_bundle(bundle, extracted)
    returned = extracted / "scamguard_blind_audit.csv"
    complete_returned_csv(
        returned,
        {
            "I can move your reservation to Tuesday.": "SAFE",
            "Send the verification code now to stop suspension.": "SCAM",
        },
    )
    returned.write_text(
        returned.read_text(encoding="utf-8").replace("move your reservation", "send a wire"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="immutable id/text inputs"):
        import_returned_audit(
            returned,
            bundle,
            audit,
            manifest,
            tmp_path / "reviewed.csv",
            tmp_path / "report.json",
        )


def test_import_preserves_disagreement_as_failed_release_gate(tmp_path: Path) -> None:
    audit, manifest, bundle = build_fixture_bundle(tmp_path)
    extracted = tmp_path / "disagreed"
    extracted.mkdir()
    extract_bundle(bundle, extracted)
    returned = extracted / "scamguard_blind_audit.csv"
    complete_returned_csv(
        returned,
        {
            "I can move your reservation to Tuesday.": "UNCERTAIN",
            "Send the verification code now to stop suspension.": "SCAM",
        },
    )

    result = import_returned_audit(
        returned,
        bundle,
        audit,
        manifest,
        tmp_path / "reviewed.csv",
        tmp_path / "report.json",
    )

    assert result["release_gate_passed"] is False
    assert result["incorrect_label_rows"] == 1
    assert result["disagreement_ids"] == ["safe-1"]


def test_import_refuses_to_overwrite_any_input(tmp_path: Path) -> None:
    audit, manifest, bundle = build_fixture_bundle(tmp_path)
    extracted = tmp_path / "protected"
    extracted.mkdir()
    extract_bundle(bundle, extracted)
    returned = extracted / "scamguard_blind_audit.csv"

    with pytest.raises(ValueError, match="overwrite a blind-audit input"):
        import_returned_audit(
            returned,
            bundle,
            audit,
            manifest,
            returned,
            tmp_path / "report.json",
        )
