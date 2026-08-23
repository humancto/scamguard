"""Frozen independent-review instructions for ScamGuard label audits."""

from __future__ import annotations

import hashlib
import json
from typing import Final

AUDIT_PROTOCOL_VERSION: Final[int] = 1
LABEL_RUBRIC: Final[dict[str, str]] = {
    "SAFE": (
        "Ordinary legitimate communication, defensive safety guidance, or a standalone "
        "authentication notice with no risky external action."
    ),
    "UNCERTAIN": (
        "Suspicious, ambiguous, generic spam, or an external-action request without enough "
        "message-local evidence to conclude fraud."
    ),
    "SCAM": (
        "The message itself contains strong fraud evidence, such as deceptive impersonation "
        "paired with credential, payment, remote-access, secrecy, or account-transfer pressure."
    ),
}
REVIEW_PRINCIPLES: Final[tuple[str, ...]] = (
    "Judge only the displayed message; do not infer a verdict from where it may have come from.",
    "Generic advertising or spam is not automatically a scam.",
    "A source-reported positive is SCAM only when the message itself contains strong fraud "
    "evidence.",
    "When fraud evidence is insufficient or the context could reasonably be legitimate, "
    "choose UNCERTAIN.",
)
SENSITIVE_DATA_RUBRIC: Final[str] = (
    "Mark sensitive data when the message contains a real-looking email address, phone number, "
    "account or card number, access code, credential, or other person-identifying value that "
    "should be quarantined. Obvious placeholders and .example domains do not count."
)


def audit_protocol() -> dict[str, object]:
    return {
        "version": AUDIT_PROTOCOL_VERSION,
        "labels": LABEL_RUBRIC,
        "principles": list(REVIEW_PRINCIPLES),
        "sensitive_data": SENSITIVE_DATA_RUBRIC,
    }


def audit_protocol_sha256() -> str:
    payload = json.dumps(
        audit_protocol(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()
