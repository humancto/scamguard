from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_source_overlap import read_candidate


def test_candidate_reader_supports_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "candidate.jsonl"
    source.write_text(
        json.dumps({"text": "Call the official number 1234567890", "label": "SAFE"}) + "\n",
        encoding="utf-8",
    )
    rows = read_candidate(source, "text", "label")
    assert rows == [
        {
            "raw_text": "Call the official number 1234567890",
            "text": "Call the official number <PHONE_NUMBER>",
            "label": "SAFE",
        }
    ]


def test_candidate_reader_drops_missing_or_non_string_jsonl_fields(tmp_path: Path) -> None:
    source = tmp_path / "candidate.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps({"label": "SAFE"}),
                json.dumps({"text": "valid text", "label": None}),
                json.dumps({"text": 42, "label": "SCAM"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert read_candidate(source, "text", "label") == []


def test_candidate_reader_rejects_unknown_format(tmp_path: Path) -> None:
    source = tmp_path / "candidate.txt"
    source.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported candidate format"):
        read_candidate(source, "text", "label")
