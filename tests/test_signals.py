"""Fine-grained categories remain stable for subtle scam language."""

import pytest

from scamguard.signals import extract_signal_matches, infer_category
from scamguard.taxonomy import Category, Signal


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A refundable onboarding fee reserves your position.", Category.JOB_OPPORTUNITY),
        ("Your mailbox session expired; sign in to view the contract.", Category.CREDENTIAL_MFA),
        ("Fraud desk: reply with the 6-digit code.", Category.CREDENTIAL_MFA),
        (
            "Detective Miller needs ID for this case; do not call the station.",
            Category.GOVERNMENT_LEGAL,
        ),
        (
            "Payroll needs your salary account and routing details.",
            Category.FINANCIAL_IMPERSONATION,
        ),
        (
            "I have been planning our future; the inheritance needs a fee.",
            Category.ROMANCE_RELATIONSHIP,
        ),
        (
            "I am in a confidential board meeting and will approve the paperwork.",
            Category.FAMILY_EXECUTIVE,
        ),
        ("Hi Auntie, my phone broke and this is my temporary number.", Category.FAMILY_EXECUTIVE),
    ],
)
def test_subtle_scam_categories(text: str, expected: Category) -> None:
    assert infer_category(text, ()) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Visit account-review.example/help", Signal.SUSPICIOUS_LINK),
        ("Call 08081560665 to claim", Signal.CONTACT_DIVERSION),
        ("Reply with the 6-digit code", Signal.OTP_REQUEST),
        ("Settle £75 using a payment app", Signal.PAYMENT_REQUEST),
        ("Community Bank fraud . desk asks you to reply", Signal.AUTHORITY_IMPERSONATION),
        ("u\u200br\u200bg\u200be\u200bn\u200bt: act now", Signal.ARTIFICIAL_URGENCY),
        ("Su cuenta está bloqueada. Actualice aquí: <URL>", Signal.SUSPICIOUS_LINK),
        ("Rufen Sie sofort <PHONE_NUMBER> an", Signal.CONTACT_DIVERSION),
        ("Votre compte est bloqué", Signal.THREAT_OR_COERCION),
        ("Ingrese su contraseña inmediatamente", Signal.CREDENTIAL_REQUEST),
    ],
)
def test_extracts_verbatim_evidence_for_non_url_scam_cues(text: str, expected: Signal) -> None:
    matches = extract_signal_matches(text)

    assert expected in {match.signal for match in matches}
    assert all(
        match.evidence.text == text[match.evidence.start : match.evidence.end] for match in matches
    )


def test_measurement_abbreviation_is_not_a_url() -> None:
    matches = extract_signal_matches(
        "Residential plots size 90,112,200,400sq.mtr, rate 5000 per sq.mtr."
    )

    assert Signal.SUSPICIOUS_LINK not in {match.signal for match in matches}
