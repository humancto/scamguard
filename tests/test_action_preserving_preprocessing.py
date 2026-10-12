"""Original synthetic regression fixtures; no copied benchmark dialogue text."""

from __future__ import annotations

import pytest

from scamguard.preprocessing import (
    DIALOGUE_POLICY_ACTION_PRESERVING_V3,
    EVIDENCE_RECENT_MAX_CHARS,
    compact_dialogue_action_preserving,
    compact_dialogue_evidence_recent,
    prepare_model_text,
)


def access_dialogue(request: str) -> str:
    return (
        "CALLER: This is Northstar software support. RECEIVER: What is the problem? "
        f"CALLER: {request} RECEIVER: What account, password, and billing details do you mean? "
        "CALLER: Please confirm the security information for your account and billing address. "
        "RECEIVER: I want to check this. CALLER: Our case identifier is DEMO-42. "
        "RECEIVER: I will independently contact the published support number."
    )


@pytest.mark.parametrize("action_text", [
    "Grant our technician access to your cloud management console.",
    "Please allow me administrative permissions on the hosted workspace.",
    "Give the caller control of the business dashboard.",
    "Authorize our team to have access to the project administration page.",
    "Never give a caller access to your workspace.",
])
def test_access_request_is_first_and_retains_safety_context(action_text: str) -> None:
    prepared = compact_dialogue_action_preserving(access_dialogue(action_text))
    assert prepared.startswith("EVIDENCE: A: " + action_text)
    assert "independently contact the published support number" in prepared
    assert "B:" in prepared
    assert len(prepared) <= EVIDENCE_RECENT_MAX_CHARS


def test_old_policy_is_not_silently_changed() -> None:
    text = access_dialogue("Grant me access to the project console.")
    old = compact_dialogue_evidence_recent(text)
    assert prepare_model_text(text, "speaker-neutral-evidence-recent-v2") == old
    assert prepare_model_text(text) == text
    assert "Grant me access" not in old  # Reproduces the missing-action defect.
    assert "Grant me access" in prepare_model_text(text, DIALOGUE_POLICY_ACTION_PRESERVING_V3)


def test_earlier_account_request_is_not_replaced_by_reassurance() -> None:
    text = (
        "AGENT: Please provide your bank account number to receive this adjustment. "
        "CUSTOMER: Is that required? AGENT: We have a record of your order. "
        "CUSTOMER: I need time. AGENT: Our reference is DEMO-61. "
        "CUSTOMER: I will verify the request independently."
    )
    prepared = compact_dialogue_action_preserving(text)
    assert prepared.startswith("EVIDENCE: A: Please provide your bank account number")
    assert "verify the request independently" in prepared


def test_negation_and_legitimate_sensitive_requests_are_not_rewritten() -> None:
    text = (
        "AGENT: Never share your password or code with anyone. CUSTOMER: Understood. "
        "AGENT: Enter your email address in the official app yourself. CUSTOMER: Thank you."
    )
    prepared = compact_dialogue_action_preserving(text)
    assert "Never share your password or code with anyone." in prepared
    assert "Enter your email address in the official app yourself." in prepared
    assert "SCAM" not in prepared and "SAFE" not in prepared


def test_concrete_account_request_outranks_identity_reassurance() -> None:
    text = (
        "CALLER: We are following up about an adjustment. RECEIVER: What is this about? "
        "CALLER: Confirm your account number to locate the entry. "
        "RECEIVER: Why is an account number necessary for the refund? "
        "CALLER: We can investigate the issue. RECEIVER: I am not convinced. "
        "CALLER: You can verify my identity using the account and billing reference. "
        "RECEIVER: I will think about that."
    )
    prepared = compact_dialogue_action_preserving(text)
    assert prepared.startswith("EVIDENCE: A: Confirm your account number to locate the entry.")
    assert "verify my identity" in prepared


def test_long_turn_keeps_request_anchor_and_character_bound() -> None:
    request = "Please grant me access to the project dashboard."
    filler = "Ordinary discussion of the weather and a scheduled appointment. " * 30
    prepared = compact_dialogue_action_preserving(access_dialogue(filler + request + filler))
    assert request in prepared
    assert len(prepared) <= EVIDENCE_RECENT_MAX_CHARS


def test_request_from_second_speaker_keeps_identity() -> None:
    text = (
        "CALLER: Hello. RECEIVER: Give me access to your work console. "
        "CALLER: Never share credentials with strangers. RECEIVER: Here is a reference number."
    )
    prepared = compact_dialogue_action_preserving(text)
    assert prepared.startswith("EVIDENCE: B: Give me access to your work console.")
    assert "A: Never share credentials with strangers." in prepared


def test_no_request_uses_unchanged_v2_fallback() -> None:
    text = "AGENT: Welcome. CUSTOMER: Hello. AGENT: Your visit is tomorrow. CUSTOMER: Thanks."
    assert compact_dialogue_action_preserving(text) == compact_dialogue_evidence_recent(text)


@pytest.mark.parametrize("text", ["A short message.", "CALLER: One. RECEIVER: Two.", ""])
def test_non_dialogue_is_unchanged(text: str) -> None:
    assert compact_dialogue_action_preserving(text) == text


def test_transform_is_deterministic_and_idempotent() -> None:
    text = access_dialogue("Grant me access to the cloud console.")
    prepared = compact_dialogue_action_preserving(text)
    assert compact_dialogue_action_preserving(text) == prepared
    assert compact_dialogue_action_preserving(prepared) == prepared
