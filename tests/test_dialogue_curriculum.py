from __future__ import annotations

from collections import Counter

from scamguard.signals import extract_signal_matches
from scripts.build_taskmaster_hard_negatives import (
    DOMAIN_PREFIXES,
    MAX_CHARS,
    domain_for,
    partition,
    render_latest_context,
    row_for,
)
from scripts.generate_dialogue_curriculum import (
    ROWS_PER_LABEL_PER_SCENARIO,
    SCENARIOS,
    generate,
)


def test_dialogue_curriculum_is_balanced_and_training_only() -> None:
    rows = generate()
    expected_per_label = len(SCENARIOS) * ROWS_PER_LABEL_PER_SCENARIO
    assert Counter(row["label"] for row in rows) == {
        "SAFE": expected_per_label,
        "SCAM": expected_per_label,
    }
    assert all(row["split"] == "train" for row in rows)
    assert len({row["id"] for row in rows}) == len(rows)


def test_dialogue_scam_rows_have_extractable_evidence_and_authoritative_provenance() -> None:
    rows = generate()
    scam_rows = [row for row in rows if row["label"] == "SCAM"]
    assert all(extract_signal_matches(str(row["text"])) for row in scam_rows)
    assert all(str(row["pattern_reference"]).startswith("https://") for row in rows)
    assert all(row["is_synthetic"] is True for row in rows)


def test_taskmaster_row_is_privacy_normalized_and_bounded() -> None:
    dialogue = {
        "conversation_id": "dialogue-1",
        "instruction_id": "movie-tickets-2",
        "utterances": [
            {"speaker": "ASSISTANT", "text": "How can I help today?"},
            {"speaker": "USER", "text": "My email is person@real.test."},
            {"speaker": "ASSISTANT", "text": "Can you confirm +1 (415) 555-1212?"},
            {"speaker": "USER", "text": "I need two tickets for Thursday."},
        ],
    }
    row = row_for(dialogue, "train")
    assert row is not None
    assert row["label"] == "SAFE"
    assert row["is_synthetic"] is False
    assert row["provenance_class"] == "human_crowdsourced_roleplay"
    assert "person@real.test" not in str(row["text"])
    assert "415" not in str(row["text"])
    assert "<EMAIL>" in str(row["text"])
    assert "<NUMBER>" in str(row["text"])
    assert len(str(row["text"])) <= MAX_CHARS


def test_taskmaster_domains_and_partition_are_deterministic() -> None:
    assert domain_for("auto-repair-appt-2") == "auto-repair"
    assert domain_for("unknown-1") is None
    assert set(DOMAIN_PREFIXES) == {
        "auto-repair",
        "coffee-ordering",
        "movie-tickets",
        "pizza-ordering",
        "restaurant-table",
        "uber-lyft",
    }
    assert partition("dlg-123") == partition("dlg-123")
    assert partition("dlg-123") in {"train", "validation"}


def test_taskmaster_context_keeps_latest_complete_turns() -> None:
    utterances = [
        {"speaker": "USER", "text": "oldest " + "a" * 45},
        {"speaker": "ASSISTANT", "text": "middle " + "b" * 35},
        {"speaker": "USER", "text": "latest answer"},
    ]
    rendered = render_latest_context(utterances, max_chars=70)
    assert "latest answer" in rendered
    assert "oldest" not in rendered
    assert not rendered.endswith("…")
