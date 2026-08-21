from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.build_apptek_callcenter as apptek
from scripts.fetch_apptek_callcenter import fetch

ACCENTS = ("en-AU", "en-CA", "en-GB", "en-IN", "en-SG", "en-US_General")
TOPICS = (
    "winter rail pass",
    "garden equipment return",
    "museum membership renewal",
    "coastal hotel reservation",
    "library accessibility request",
    "bicycle repair appointment",
    "theatre seating exchange",
    "grocery delivery substitution",
    "language course registration",
    "community pool schedule",
    "home energy consultation",
    "pet boarding reservation",
)


def fixture_segment(role: str, speaker: str, start: float, text: str) -> dict[str, object]:
    return {
        "end": start + 4.0,
        "gender": "female" if role == "agent" else "male",
        "role": role,
        "speaker_id": speaker,
        "start": start,
        "text": text,
    }


def write_metadata_fixture(raw: Path) -> dict[str, dict[str, object]]:
    raw.mkdir()
    source_files: dict[str, dict[str, object]] = {}
    for accent_index, accent in enumerate(ACCENTS):
        calls: list[dict[str, object]] = []
        for call_index in range(2):
            topic = TOPICS[accent_index * 2 + call_index]
            customer = f"customer-{accent_index}-{call_index}"
            calls.append(
                {
                    "accent": accent,
                    "domain": f"service-domain-{accent_index}",
                    "duration": 36.0 + call_index,
                    "file_name": f"call-{accent_index}-{call_index}.wav",
                    "segments": [
                        fixture_segment(
                            "agent",
                            f"agent-{accent_index}",
                            0.0,
                            f"Welcome to the {topic} desk. I can explain the available options "
                            "and help document what you would like changed during this call.",
                        ),
                        fixture_segment(
                            "customer",
                            customer,
                            5.0,
                            f"Thanks. I have a detailed question about my {topic}, because the "
                            "arrangement no longer matches what my household needs this month.",
                        ),
                        fixture_segment(
                            "agent",
                            f"agent-{accent_index}",
                            10.0,
                            f"That makes sense. For the {topic}, I can compare the standard "
                            "choices, explain the timing, and record the preference you select.",
                        ),
                        fixture_segment(
                            "customer",
                            customer,
                            15.0,
                            f"Please compare those {topic} choices and tell me whether a later "
                            "date would be possible without changing the rest of the booking.",
                        ),
                    ],
                }
            )
        payload = "".join(json.dumps(call, sort_keys=True) + "\n" for call in calls).encode()
        path = raw / f"{accent}.jsonl"
        path.write_bytes(payload)
        source_files[accent] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return source_files


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_build_is_text_free_and_speaker_disjoint(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    source_files = write_metadata_fixture(raw)
    output = tmp_path / "external"
    report_path = tmp_path / "report.json"

    report = apptek.build(
        raw,
        [],
        output,
        report_path,
        source_files=source_files,
        expected_calls=12,
    )

    selection = read_rows(output / "apptek_call_selection.jsonl")
    ood = read_rows(output / "apptek_call_ood.jsonl")
    selection_speakers = {
        speaker for row in selection for speaker in row["source_speaker_hashes"]
    }
    ood_speakers = {speaker for row in ood for speaker in row["source_speaker_hashes"]}
    report_text = report_path.read_text()
    assert selection and ood
    assert selection_speakers.isdisjoint(ood_speakers)
    assert all(row["label"] == "SAFE" for row in selection + ood)
    assert all(row["is_synthetic"] is False for row in selection + ood)
    assert all(row["license"] == "CC-BY-SA-4.0" for row in selection + ood)
    assert report["policy"]["used_for_fitting"] is False
    assert report["policy"]["ood_prediction_sealed_until_candidate_freeze"] is True
    assert "winter rail pass" not in report_text
    assert "source_speaker_hashes" not in report_text


def test_fetch_verifies_existing_metadata_and_rejects_mutation(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    source_files = write_metadata_fixture(raw)

    receipt = fetch(raw, source_files=source_files)

    assert receipt["audio_downloaded"] is False
    assert receipt["total_bytes"] == sum(path.stat().st_size for path in raw.glob("*.jsonl"))
    path = raw / f"{ACCENTS[0]}.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="byte count differs"):
        fetch(raw, source_files=source_files)


def test_recent_long_segment_preserves_role_prefix() -> None:
    segments = [
        fixture_segment(
            "customer",
            "speaker",
            0.0,
            "opening context " + "descriptive words " * 40 + "final request for assistance",
        )
    ]

    rendered = apptek.render_window(segments, recent=True)

    assert rendered.startswith("CUSTOMER: ")
    assert rendered.endswith("final request for assistance")
    assert len(rendered) <= apptek.MAX_WINDOW_CHARS


def test_source_call_count_is_pinned(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    source_files = write_metadata_fixture(raw)

    with pytest.raises(ValueError, match="expected 873 AppTek calls, found 12"):
        apptek.read_source(raw, source_files=source_files)
