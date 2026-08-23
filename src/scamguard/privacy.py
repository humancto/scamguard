"""Context-aware masking for short sensitive values missed by broad PII patterns."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

CONTEXTUAL_PRIVACY_REVISION = "contextual_sensitive_values_v1"

ACCESS_CODE_TERM = (
    r"(?:otp|one[- ]time(?: verification)?(?:\s*\(otp\))? "
    r"(?:pin|password|passcode|code)|verification code|security code|"
    r"login code|sign[- ]in code|passcode|pin)"
)
ACCESS_CODE_AFTER_RE = re.compile(
    rf"(?P<prefix>\b{ACCESS_CODE_TERM}\b"
    r"(?:\s*(?:(?:is|was)\s*[:=]?|[:=-])\s*|\s+))"
    r"[\"']?(?P<value>\d{4,8})[\"']?\b",
    re.IGNORECASE,
)
ACCESS_CODE_BEFORE_RE = re.compile(
    rf"(?<!\d)(?P<value>\d{{4,8}})(?!\d)(?=\s+(?:is\s+)?(?:your\s+)?"
    rf"(?:[A-Za-z][\w.-]*\s+){{0,3}}{ACCESS_CODE_TERM}\b)",
    re.IGNORECASE,
)
LEADING_ACCESS_CODE_RE = re.compile(
    rf"^(?P<prefix>\s*)\d{{4,8}}(?=.{{0,120}}\b{ACCESS_CODE_TERM}\b)",
    re.IGNORECASE | re.DOTALL,
)
LAST_FOUR_RE = re.compile(
    r"(?P<prefix>\b(?:last\s+four(?:\s+digits)?(?:\s+of)?(?:\s+(?:my|your|the))?\s+"
    r"(?:ssn|social(?:\s+security)?|account(?:\s+number)?|card(?:\s+number)?)|"
    r"(?:ssn|social(?:\s+security)?|account(?:\s+number)?|card(?:\s+number)?)\s+"
    r"(?:ending|ends\s+in))\D{0,12})\d{4}\b",
    re.IGNORECASE,
)
POSTAL_CODE_RE = re.compile(
    r"(?P<prefix>\b(?:zip|postal)\s+code\D{0,12})\d{4,10}\b",
    re.IGNORECASE,
)
CREDENTIAL_VALUE_RE = re.compile(
    r"(?P<prefix>\b(?:password|credential)\s+(?:is\s+)?)(?P<value>[^\s,;]{4,32})",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PrivacyNormalization:
    text: str
    replacement_counts: dict[str, int]

    @property
    def changed(self) -> bool:
        return bool(self.replacement_counts)


def _substitute(
    pattern: re.Pattern[str],
    replacement: str,
    text: str,
    counts: Counter[str],
    category: str,
) -> str:
    def replace(match: re.Match[str]) -> str:
        counts[category] += 1
        return f"{match.groupdict().get('prefix', '')}{replacement}"

    return pattern.sub(replace, text)


def mask_contextual_sensitive_values(text: str) -> PrivacyNormalization:
    """Mask short codes only when surrounding language establishes a sensitive role."""

    counts: Counter[str] = Counter()
    normalized = _substitute(
        ACCESS_CODE_AFTER_RE, "<ACCESS_CODE>", text, counts, "access_code"
    )
    normalized = _substitute(
        ACCESS_CODE_BEFORE_RE, "<ACCESS_CODE>", normalized, counts, "access_code"
    )
    normalized = _substitute(
        LEADING_ACCESS_CODE_RE, "<ACCESS_CODE>", normalized, counts, "access_code"
    )
    normalized = _substitute(
        LAST_FOUR_RE, "<ACCOUNT_NUMBER>", normalized, counts, "account_fragment"
    )
    normalized = _substitute(
        POSTAL_CODE_RE, "<POSTAL_CODE>", normalized, counts, "postal_code"
    )

    def replace_credential(match: re.Match[str]) -> str:
        value = match.group("value")
        if not any(character.isdigit() for character in value):
            return match.group(0)
        counts["credential"] += 1
        return f"{match.group('prefix')}<CREDENTIAL>"

    normalized = CREDENTIAL_VALUE_RE.sub(replace_credential, normalized)
    return PrivacyNormalization(normalized, dict(sorted(counts.items())))
