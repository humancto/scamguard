from __future__ import annotations

from scamguard.preprocessing import (
    EVIDENCE_RECENT_MAX_CHARS,
    canonicalize_dialogue_speakers,
    compact_dialogue_evidence_recent,
    parse_dialogue_turns,
    prepare_model_text,
)


def test_dialogue_speakers_are_neutralized_in_turn_order() -> None:
    source = (
        "TRANSCRIPT:\nCALLER: Hello. RECEIVER: Hi. CALLER: Send a code. "
        "RECEIVER: I will verify first."
    )
    assert canonicalize_dialogue_speakers(source) == (
        "A: Hello. B: Hi. A: Send a code. B: I will verify first."
    )


def test_short_message_with_one_role_word_is_not_rewritten() -> None:
    text = "Caller: please leave a message after the tone."
    assert canonicalize_dialogue_speakers(text) == text
    assert prepare_model_text(text, "none") == text


def test_dialogue_turn_parser_preserves_speaker_and_content() -> None:
    source = (
        "TRANSCRIPT:\nCALLER: Hello there. RECEIVER: Hi. "
        "CALLER: Send a code. RECEIVER: I will verify first."
    )
    assert parse_dialogue_turns(source) == [
        ("caller", "Hello there."),
        ("receiver", "Hi."),
        ("caller", "Send a code."),
        ("receiver", "I will verify first."),
    ]


def test_dialogue_turn_parser_accepts_one_sided_transcript() -> None:
    assert parse_dialogue_turns("USER: One. USER: Two.") == [
        ("user", "One."),
        ("user", "Two."),
    ]


def test_dialogue_turn_parser_rejects_unmarked_text() -> None:
    assert parse_dialogue_turns("Please leave a message after the tone.") == []


def test_evidence_recent_policy_retains_early_harm_and_delayed_refusal() -> None:
    source = (
        "CALLER: This is technical support. RECEIVER: What is wrong? "
        "CALLER: Install this remote support tool and give me the verification code. "
        "RECEIVER: Let me find a pen. CALLER: The case number is 42. "
        "RECEIVER: I will not do that and will call the verified support number."
    )
    compacted = compact_dialogue_evidence_recent(source)
    assert compacted.startswith("EVIDENCE:")
    assert "remote support tool" in compacted
    assert "verification code" in compacted
    assert "verified support number" in compacted
    assert "RECENT:" in compacted


def test_evidence_recent_policy_keeps_explicit_safe_navigation() -> None:
    source = (
        "AGENT: Welcome to account support. CUSTOMER: I saw an alert. "
        "AGENT: Never share a security code with me. CUSTOMER: What should I do? "
        "AGENT: Open the official app yourself and verify the request independently. "
        "CUSTOMER: I will use the app."
    )
    compacted = prepare_model_text(source, "speaker-neutral-evidence-recent-v2")
    assert "Never share a security code" in compacted
    assert "official app yourself" in compacted
    assert "verify the request independently" in compacted


def test_evidence_recent_policy_assigns_neutral_speakers_by_first_appearance() -> None:
    source = (
        "CALLER: Hello. RECEIVER: Send the gift card code now. "
        "CALLER: I will not. RECEIVER: Stay on the line."
    )
    compacted = compact_dialogue_evidence_recent(source)
    assert "B: Send the gift card code now." in compacted
    assert "A: I will not." in compacted


def test_evidence_recent_policy_is_bounded_and_speaker_neutral() -> None:
    filler = "ordinary account discussion " * 80
    source = (
        f"CALLER: {filler} RECEIVER: hello. "
        "CALLER: Send the gift card code immediately. RECEIVER: no. "
        f"CALLER: {filler} RECEIVER: I will hang up."
    )
    compacted = compact_dialogue_evidence_recent(source)
    assert len(compacted) <= EVIDENCE_RECENT_MAX_CHARS
    assert "CALLER:" not in compacted
    assert "RECEIVER:" not in compacted
    assert "gift card code" in compacted


def test_evidence_recent_policy_leaves_non_dialogue_unchanged() -> None:
    text = "Your package is waiting. Open the official carrier app yourself."
    assert compact_dialogue_evidence_recent(text) == text
