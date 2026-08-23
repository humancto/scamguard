from __future__ import annotations

import pytest

from scamguard.privacy import mask_contextual_sensitive_values


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "9158 is your One Time Verification (OTP) code to confirm your phone.",
            "<ACCESS_CODE> is your One Time Verification (OTP) code to confirm your phone.",
        ),
        ("Your verification code is: 443088", "Your verification code is: <ACCESS_CODE>"),
        (
            "Please enter your OTP 4984 to continue.",
            "Please enter your OTP <ACCESS_CODE> to continue.",
        ),
        ("PIN-6661", "PIN-<ACCESS_CODE>"),
        (
            "356280 is your Amazon OTP. Do not share it.",
            "<ACCESS_CODE> is your Amazon OTP. Do not share it.",
        ),
        (
            "The last four of my social are 2596.",
            "The last four of my social are <ACCOUNT_NUMBER>.",
        ),
        (
            "The last four digits of my account number are 9950.",
            "The last four digits of my account number are <ACCOUNT_NUMBER>.",
        ),
        ("My zip code is 12010.", "My zip code is <POSTAL_CODE>."),
        ("My password is line6@8768", "My password is <CREDENTIAL>"),
    ],
)
def test_contextual_sensitive_values_are_masked(text: str, expected: str) -> None:
    result = mask_contextual_sensitive_values(text)

    assert result.text == expected
    assert result.changed is True


@pytest.mark.parametrize(
    "text",
    [
        "The order total is 4984 dollars.",
        "We launched in 2024 and offer 200 free spins.",
        "Never share your password with anyone.",
        "Your package arrives on 2026-09-01.",
    ],
)
def test_unrelated_short_numbers_are_preserved(text: str) -> None:
    result = mask_contextual_sensitive_values(text)

    assert result.text == text
    assert result.changed is False
