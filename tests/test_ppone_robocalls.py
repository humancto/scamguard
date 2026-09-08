from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import scripts.build_ppone_robocalls as ppone


def write_fixture(path: Path) -> list[str]:
    messages = [
        "Call our toll-free number 315-232-8257 now to cancel the suspicious order.",
        "Call our toll-free number 212-555-0100 now to cancel the suspicious order.",
        "This is a recorded message about a new home services promotion in your area.",
        "请按一与客服代表通话并了解账户状态。",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ppone.EXPECTED_HEADER)
        writer.writeheader()
        for index, message in enumerate(messages):
            writer.writerow(
                {
                    "file_name": f"audio/{index}.wav",
                    "language": "zh" if index == 3 else "en",
                    "transcript": message,
                    "case_details": f"case-{index}",
                    "case_pdf": f"case-{index}.pdf",
                }
            )
    return messages


def test_build_is_split_safe_private_and_text_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "metadata.csv"
    messages = write_fixture(source)
    monkeypatch.setattr(ppone, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(ppone, "EXPECTED_ROWS", len(messages))
    references = tmp_path / "references"
    references.mkdir()
    output = tmp_path / "output"
    report_path = tmp_path / "report.json"

    report = ppone.build(
        source,
        output,
        report_path,
        references=(references,),
    )

    assert report["counts"]["language_counts"] == {"en": 3, "zh": 1}
    assert report["counts"]["english_candidate_rows"] == 3
    assert report["counts"]["exact_duplicate_rows_removed"] == 1
    assert report["policy"]["raw_text_written_to_manifest"] is False
    serialized = report_path.read_text(encoding="utf-8")
    assert all(message not in serialized for message in messages)
    admitted = []
    for split in ("train", "validation", "test"):
        path = output / f"ppone_{split}.jsonl"
        admitted.extend(json.loads(line) for line in path.read_text().splitlines())
    assert len(admitted) == 2
    assert {row["label"] for row in admitted} == {"SCAM", "UNCERTAIN"}
    assert all("315-232-8257" not in row["text"] for row in admitted)
    assert all(row["license"] == "Public-Domain" for row in admitted)
    assert all(row["is_synthetic"] is False for row in admitted)


def test_rejects_unpinned_or_incomplete_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "metadata.csv"
    write_fixture(source)
    monkeypatch.setattr(ppone, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())

    with pytest.raises(ValueError, match="expected 1432"):
        ppone.read_source(source)
    monkeypatch.setattr(ppone, "SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="pinned repository revision"):
        ppone.read_source(source)
