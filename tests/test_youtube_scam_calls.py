from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

import scripts.build_youtube_scam_calls as youtube_calls


def write_fixture_archive(path: Path) -> list[str]:
    messages = [
        (
            "I am calling from the refund department and need to explain the unusual "
            "charge before we continue with the cancellation request."
        ),
        (
            "The technical support desk detected a serious computer problem and I will "
            "walk you through the repair procedure immediately."
        ),
        (
            "This is the awards office calling about a prize notification and the "
            "processing steps required before it can be released."
        ),
        (
            "Our billing team needs to discuss the account balance and confirm several "
            "details before the service can remain active."
        ),
        (
            "I represent the tax resolution unit and this recorded call concerns a "
            "pending case that requires your immediate attention."
        ),
        (
            "The delivery administration office has an undelivered parcel and needs to "
            "review the release procedure with you today."
        ),
        (
            "I am the senior security technician assigned to your device and we must "
            "complete a diagnostic session to resolve the warning."
        ),
        (
            "This is the compensation center following up on an approved refund and the "
            "remaining verification procedure for the claim."
        ),
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=("ID", "Source", "Content", "Char_Len"))
    writer.writeheader()
    for index, message in enumerate(messages):
        writer.writerow(
            {
                "ID": f"row-{index}",
                "Source": f"https://video.example/watch/{index}",
                "Content": message,
                "Char_Len": len(message),
            }
        )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("FullTranscriptData.csv", buffer.getvalue())
        archive.writestr("FullTranscriptData.xlsx", b"fixture workbook placeholder")
    return messages


def test_build_is_text_free_and_keeps_families_split_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "source.zip"
    messages = write_fixture_archive(archive_path)
    monkeypatch.setattr(
        youtube_calls,
        "SOURCE_SHA256",
        hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    )
    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    output = tmp_path / "external"
    report_path = tmp_path / "report.json"

    report = youtube_calls.build(archive_path, reference_dir, output, report_path)

    serialized_report = report_path.read_text(encoding="utf-8")
    assert all(message not in serialized_report for message in messages)
    assert report["policy"]["raw_text_written_to_manifest"] is False
    family_splits: dict[str, set[str]] = {}
    for split, filename in (
        ("train", "youtube_scam_train.jsonl"),
        ("validation", "youtube_scam_validation.jsonl"),
        ("ood", "youtube_scam_ood.jsonl"),
    ):
        for line in (output / filename).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            family_splits.setdefault(row["family_id"], set()).add(split)
            assert row["source"] == "youtube_scam_calls_cc0"
            assert row["license"] == "CC0-1.0"
            assert row["is_synthetic"] is False
    assert family_splits
    assert all(len(splits) == 1 for splits in family_splits.values())


def test_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../FullTranscriptData.csv", "bad")
        archive.writestr("FullTranscriptData.xlsx", "bad")

    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="unsafe"):
        youtube_calls.archive_members(archive)
