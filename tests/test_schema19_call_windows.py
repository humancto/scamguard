from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.metrics import file_sha256
from scripts import build_schema19_call_windows as schema19
from scripts.generate_call_evidence_pairs import generate


def test_long_pair_expansion_is_deterministic_balanced_and_action_only() -> None:
    source = generate()
    rows = schema19.expand_long_pairs(source)

    assert rows == schema19.expand_long_pairs(source)
    assert len(rows) == len(source)
    assert Counter(str(row["label"]) for row in rows) == {
        "SAFE": len(rows) // 2,
        "SCAM": len(rows) // 2,
    }
    assert {str(row["source"]) for row in rows} == {schema19.LONG_PAIR_SOURCE}
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    assert len(grouped) == len(rows) // 2
    for pair in grouped.values():
        assert {str(row["label"]) for row in pair} == {"SAFE", "SCAM"}
        assert len({str(row["shared_context_sha256"]) for row in pair}) == 1
        contexts = {str(row["text"]).rsplit("\n", 1)[0] for row in pair}
        assert len(contexts) == 1
        assert len(next(iter(contexts))) > 1_000
        assert all(row["external_benchmark_text_copied"] is False for row in pair)


def test_taskmaster_long_window_preserves_conversation_family(
    tmp_path: Path, monkeypatch
) -> None:
    conversation_id = "conversation-one"
    raw_path = tmp_path / "taskmaster.json"
    utterances = [
        {"speaker": "USER" if index % 2 == 0 else "ASSISTANT", "text": f"turn {index} " * 18}
        for index in range(12)
    ]
    raw_path.write_text(
        json.dumps([{"conversation_id": conversation_id, "utterances": utterances}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(schema19, "TASKMASTER_RAW_SHA256", file_sha256(raw_path))
    seed = {
        "id": "seed",
        "text": "USER: short request\nASSISTANT: short response",
        "label": "SAFE",
        "category": "NONE",
        "source": schema19.TASKMASTER_SOURCE,
        "source_label": "legitimate_task_dialogue",
        "license": "CC-BY-4.0",
        "split": "train",
        "family_id": f"taskmaster1:{conversation_id}",
        "is_synthetic": False,
    }

    rows = schema19.build_taskmaster_long_rows(raw_path, [seed], "validation")

    assert len(rows) == 1
    assert rows[0]["family_id"] == seed["family_id"]
    assert rows[0]["split"] == "validation"
    assert rows[0]["source_window"] == "recent_complete_turns_long"
    assert len(str(rows[0]["text"])) > len(str(seed["text"]))


def test_youtube_long_window_uses_only_seeded_train_record(
    tmp_path: Path, monkeypatch
) -> None:
    archive_path = tmp_path / "youtube.zip"
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=schema19.YOUTUBE_EXPECTED_HEADER)
    writer.writeheader()
    content = "This is a long scam call request about a payment and account review. " * 30
    writer.writerow(
        {"ID": "record-1", "Source": "source-a", "Content": content, "Char_Len": len(content)}
    )
    writer.writerow(
        {
            "ID": "not-admitted",
            "Source": "source-b",
            "Content": "another transcript " * 80,
            "Char_Len": len("another transcript " * 80),
        }
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("FullTranscriptData.csv", csv_buffer.getvalue())
        archive.writestr("FullTranscriptData.xlsx", b"fixture")
    monkeypatch.setattr(schema19, "YOUTUBE_RAW_SHA256", file_sha256(archive_path))
    seed = {
        "id": "seed",
        "text": content[:425],
        "label": "SCAM",
        "category": "FINANCIAL",
        "source": schema19.YOUTUBE_SOURCE,
        "source_label": "publisher_scam_call",
        "license": "CC0-1.0",
        "split": "train",
        "family_id": "youtube-call-family",
        "is_synthetic": False,
        "source_record_id": "record-1",
    }

    rows = schema19.build_youtube_long_rows(archive_path, [seed])

    assert len(rows) == 1
    assert rows[0]["source_record_id"] == "record-1"
    assert rows[0]["family_id"] == seed["family_id"]
    assert rows[0]["source_window"] == "recent_long"
    assert 425 < len(str(rows[0]["text"])) <= schema19.YOUTUBE_LONG_MAX_CHARS
