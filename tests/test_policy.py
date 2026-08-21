from scamguard.policy import POLICY_VERSION, deterministic_override
from scamguard.signals import extract_signal_matches
from scamguard.taxonomy import Verdict


def apply(text: str):
    signals = tuple(match.signal for match in extract_signal_matches(text))
    return deterministic_override(text, signals)


def test_policy_version_is_explicit() -> None:
    assert POLICY_VERSION == "trusted-channel-v1"


def test_blocked_verification_plus_credential_request_is_scam() -> None:
    decision = apply(
        "Detective Miller opened case 441. Do not call the station; send your ID and banking "
        "details here."
    )

    assert decision is not None
    assert decision.verdict is Verdict.SCAM
    assert decision.rule_id == "blocked_verification_with_credential_request"


def test_defensive_code_message_does_not_trigger_scam_override() -> None:
    assert apply(
        "Your verification code is 442901. Do not share it with anyone. If this was not you, "
        "contact support."
    ) is None


def test_explicit_official_app_guidance_is_safe() -> None:
    decision = apply(
        "River Bank declined a transaction. Open the official app or call the number on your "
        "card; we will never ask for a code."
    )

    assert decision is not None
    assert decision.verdict is Verdict.SAFE
    assert decision.rule_id == "explicit_trusted_channel_guidance"


def test_known_contact_before_transfer_is_safe() -> None:
    decision = apply(
        "Could we discuss the $325 transfer tomorrow? Call me on the number you already have "
        "before sending anything."
    )

    assert decision is not None
    assert decision.verdict is Verdict.SAFE


def test_official_app_language_does_not_veto_a_link_or_credential_request() -> None:
    assert apply(
        "Open the official app at https://account-review.example and enter your password now."
    ) is None
