from __future__ import annotations

import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.build_schema14_natural_dialogue import build


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def source_row(identifier: str, window: str) -> dict[str, object]:
    return {
        "id": identifier,
        "text": f"source dialogue window {identifier} with enough ordinary words for a fixture",
        "label": "SCAM",
        "category": "UNKNOWN",
        "source": "youtube_scam_calls_cc0",
        "source_label": "publisher_scam_call",
        "license": "CC0-1.0",
        "split": "train",
        "family_id": f"youtube-call-{identifier}",
        "is_synthetic": False,
        "source_window": window,
        "source_record_id": identifier,
        "label_policy": "publisher_positive_only_scam_call_collection",
        "provenance_class": "real_scam_call_or_autodialer_transcript",
    }


def test_schema14_adds_only_early_windows_and_preserves_evaluation_bytes(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    output = tmp_path / "output"
    parent_train = [
        {
            "id": "parent-1",
            "label": "SAFE",
            "family_id": "parent-family",
        }
    ]
    write_jsonl(parent / "train.jsonl", parent_train)
    write_jsonl(parent / "dev.jsonl", [{"id": "dev-1", "label": "SAFE"}])
    write_jsonl(parent / "test.jsonl", [{"id": "test-1", "label": "SCAM"}])
    (parent / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 13,
                "counts": {"train": 1, "dev": 1, "test": 1},
                "labels": {"SAFE": 2, "SCAM": 1},
                "sources": {"fixture": 1},
            }
        ),
        encoding="utf-8",
    )
    source_train = tmp_path / "youtube.jsonl"
    write_jsonl(
        source_train,
        [source_row("early-row", "early"), source_row("recent-row", "recent")],
    )
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {"artifacts": {"train": {"sha256": file_sha256(source_train)}}}
        ),
        encoding="utf-8",
    )

    manifest = build(parent, source_train, source_manifest, output)

    assert manifest["schema_version"] == 14
    assert manifest["schema14_increment"]["rows"] == 1
    output_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [row["id"] for row in output_rows] == ["parent-1", "early-row"]
    for filename in ("dev.jsonl", "test.jsonl"):
        assert (output / filename).read_bytes() == (parent / filename).read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build(parent, source_train, source_manifest, output)
