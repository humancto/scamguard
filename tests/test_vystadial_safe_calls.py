from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.build_vystadial_safe_calls as vystadial


def write_shard(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def fixture_row(index: int, text: str, speaker: str) -> dict[str, object]:
    identity = f"jurcic_{speaker}_120912_12{index:04d}_0001000_0002000"
    return {
        "audio_id": f"vystadial-telephony-en_{identity}",
        "license": "CC-BY-SA-3.0",
        "original_transcription": text,
        "processed_transcription": "",
        "transcription": text,
        "split": "test",
        "dur": 2.5,
        "sample_rate": 16000,
        "lang": "en",
        "src": "vystadial-openslr6-en",
        "channels": 1,
        "n_segments": 1,
        "build": "utt_full",
        "call_id": identity,
        "spk": speaker,
    }


def install_fixture_sources(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    messages = [
        "I would like a restaurant near 212-555-0100",
        "I would like a restaurant near 415-555-0100",
        "Please find a quiet hotel in the city centre",
        "What is the address of the Italian restaurant",
    ]
    first = source / "first.parquet"
    second = source / "second.parquet"
    write_shard(first, [fixture_row(1, messages[0], "001"), fixture_row(2, messages[2], "002")])
    write_shard(second, [fixture_row(3, messages[1], "001"), fixture_row(4, messages[3], "002")])
    monkeypatch.setattr(
        vystadial,
        "SOURCES",
        (
            vystadial.ParquetSource(
                "first.parquet",
                "https://example.invalid/first",
                hashlib.sha256(first.read_bytes()).hexdigest(),
                2,
            ),
            vystadial.ParquetSource(
                "second.parquet",
                "https://example.invalid/second",
                hashlib.sha256(second.read_bytes()).hexdigest(),
                2,
            ),
        ),
    )
    return messages


def test_build_is_private_balanced_and_evaluation_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    messages = install_fixture_sources(source, monkeypatch)
    references = tmp_path / "references"
    references.mkdir()
    output = tmp_path / "output"
    report_path = tmp_path / "report.json"

    report = vystadial.build(
        source,
        output,
        report_path,
        references=(references,),
        sample_size=3,
    )

    assert report["policy"]["used_for_fitting"] is False
    assert report["policy"]["used_for_threshold"] is False
    assert report["policy"]["commercial_use_allowed"] is True
    assert report["policy"]["share_alike_required"] is True
    assert report["policy"]["raw_text_written_to_manifest"] is False
    assert report["counts"]["source_rows"] == 4
    assert report["counts"]["unique_speakers"] == 2
    assert report["counts"]["exact_duplicate_rows_removed"] == 1
    assert report["counts"]["final_rows"] == 3
    serialized = report_path.read_text(encoding="utf-8")
    assert all(message not in serialized for message in messages)
    rows = [
        json.loads(line)
        for line in (output / "vystadial_safe_diagnostic.jsonl").read_text().splitlines()
    ]
    assert {row["label"] for row in rows} == {"SAFE"}
    assert all(row["split"] == "vystadial_safe" for row in rows)
    assert all("555" not in row["text"] for row in rows)
    assert all(row["license"] == "CC-BY-SA-3.0" for row in rows)


def test_rejects_unpinned_or_rewritten_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    install_fixture_sources(source, monkeypatch)
    first = source / "first.parquet"
    first.write_bytes(first.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="pinned revision"):
        vystadial.read_source(source)

    source = tmp_path / "rewritten"
    source.mkdir()
    row = fixture_row(1, "Find a restaurant", "001")
    row["transcription"] = "rewritten"
    rewritten = source / "rewritten.parquet"
    write_shard(rewritten, [row])
    monkeypatch.setattr(
        vystadial,
        "SOURCES",
        (
            vystadial.ParquetSource(
                "rewritten.parquet",
                "https://example.invalid/rewritten",
                hashlib.sha256(rewritten.read_bytes()).hexdigest(),
                1,
            ),
        ),
    )
    with pytest.raises(ValueError, match="differs from upstream"):
        vystadial.read_source(source)
