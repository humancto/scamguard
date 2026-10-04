from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

import scripts.build_international_robocalls as international


def write_csv(rows: list[dict[str, str]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=international.EXPECTED_HEADER)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def write_fixture(path: Path) -> list[str]:
    us_rows = [
        {
            "cluster_id": "1",
            "campaign": "ROBOCALL: IRS/Tax Scam",
            "transcript": "Final notice. Call 212-555-0100 now or legal action begins.",
        },
        {
            "cluster_id": "2",
            "campaign": "ROBOCALL: Telemarketing (Generic)",
            "transcript": "Hello, this recorded call describes a local home services offer.",
        },
    ]
    international_rows = [
        {
            "cluster_id": "3",
            "campaign": "ROBOCALL: Legal Threat Scam",
            "transcript": "A warrant is pending. Call 415-555-0123 before the deadline.",
        }
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("text/", b"")
        archive.writestr("text/text/", b"")
        archive.writestr("text/text/us.csv", write_csv(us_rows))
        archive.writestr("text/text/int.csv", write_csv(international_rows))
    return [row["transcript"] for row in us_rows + international_rows]


def test_build_is_private_text_free_and_noncommercial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "text.zip"
    raw_messages = write_fixture(source)
    monkeypatch.setattr(
        international, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(international, "EXPECTED_ROWS", {"international": 1, "us": 2})
    output = tmp_path / "output"
    report_path = tmp_path / "report.json"

    report = international.build(source, output, report_path)

    assert report["policy"]["used_for_fitting"] is False
    assert report["policy"]["commercial_use_allowed"] is False
    assert report["policy"]["raw_text_written_to_manifest"] is False
    assert report["counts"]["final_labels"] == {"SCAM": 2, "UNCERTAIN": 1}
    serialized_report = report_path.read_text(encoding="utf-8")
    assert all(message not in serialized_report for message in raw_messages)
    rows = [
        json.loads(line)
        for line in (output / "international_robocalls_diagnostic.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["license"] == "CC-BY-NC-4.0" for row in rows)
    assert all(row["is_synthetic"] is False for row in rows)
    assert all("555" not in row["text"] for row in rows)
    assert {row["label"] for row in rows} == {"SCAM", "UNCERTAIN"}


def test_rejects_unpinned_source_and_unknown_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "text.zip"
    write_fixture(source)
    with pytest.raises(ValueError, match="pinned Zenodo artifact"):
        international.read_source(source)

    monkeypatch.setattr(
        international, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(international, "EXPECTED_ROWS", {"international": 1, "us": 2})
    assert international.label_for_campaign("ROBOCALL: Unclassified") == "UNCERTAIN"
    with pytest.raises(ValueError, match="unknown international robocall campaign"):
        international.label_for_campaign("ROBOCALL: New Unsupported Category")


def test_rejects_unsafe_zip_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../escape.csv", b"bad")
    monkeypatch.setattr(
        international, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="unsafe international robocall ZIP member"):
        international.read_source(source)
