from __future__ import annotations

from scamguard.preprocessing import (
    canonicalize_dialogue_speakers,
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
