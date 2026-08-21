"""High-precision deterministic overrides layered on a calibrated model decision."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .taxonomy import Signal, Verdict

POLICY_VERSION = "trusted-channel-v1"


@dataclass(frozen=True, slots=True)
class PolicyOverride:
    verdict: Verdict
    rule_id: str


_INDEPENDENT_VERIFICATION_BLOCK_RE = re.compile(
    r"\b(?:do not|don't|don’t|cannot|can't|can’t|must not)\s+"
    r"(?:try\s+to\s+)?(?:call|contact|verify|check|ring)\b",
    re.I,
)

# Original phrases covering an already-known channel, an app launched independently, or the
# published number on a physical card. These are policy cues, not a list of brands or domains.
_TRUSTED_CHANNEL_RE = re.compile(
    r"(?:"
    r"official app|aplicaci[oó]n oficial|application officielle|offizielle app|"
    r"offici[eë]le app|app ufficiale|aplica[cç][aã]o oficial|aplikasi resmi|"
    r"number (?:on|printed on) (?:your|the) card|"
    r"n[uú]mero (?:de|en|que figura en) (?:tu |la )?tarjeta|"
    r"num[eé]ro (?:sur|figurant sur) (?:votre |la )?carte|"
    r"nummer (?:auf|op) (?:ihrer|uw|de) (?:karte|kaart)|"
    r"numero (?:sulla|riportato sulla) carta|"
    r"n[uú]mero (?:no|indicado no) cart[aã]o|"
    r"nomor (?:pada|yang tertera pada) kartu|"
    r"number you already have|saved contact|known contact|regular call|normal contact"
    r")",
    re.I,
)

_EXPLICIT_SAFETY_RE = re.compile(
    r"(?:"
    r"before sending anything|never (?:ask|asks|request|requests) (?:you )?(?:for )?"
    r"(?:a )?code|no link|asks? for no credentials|not requesting money|"
    r"nunca (?:te )?pediremos un c[oó]digo|nunca pediremos um c[oó]digo|"
    r"ne demanderons jamais de code|fragen nie nach einem code|"
    r"vragen nooit om een code|non chiederemo mai un codice|"
    r"tidak akan pernah meminta kode"
    r")",
    re.I,
)

_SAFE_VETO_BLOCKERS = {
    Signal.ARTIFICIAL_URGENCY,
    Signal.CREDENTIAL_REQUEST,
    Signal.OTP_REQUEST,
    Signal.UNUSUAL_PAYMENT_METHOD,
    Signal.SUSPICIOUS_LINK,
    Signal.SHORTENED_URL,
    Signal.CONTACT_DIVERSION,
    Signal.SECRECY_ISOLATION,
    Signal.GUARANTEED_RETURN,
    Signal.ADVANCE_FEE,
    Signal.REMOTE_ACCESS_REQUEST,
    Signal.THREAT_OR_COERCION,
    Signal.TRUST_ACCELERATION,
    Signal.TOO_GOOD_TO_BE_TRUE,
    Signal.OFF_PLATFORM_REQUEST,
}


def deterministic_override(text: str, signals: tuple[Signal, ...]) -> PolicyOverride | None:
    """Return only overrides with an explicit, extractive trust-boundary rationale."""

    active = set(signals)
    if (
        Signal.CREDENTIAL_REQUEST in active
        and _INDEPENDENT_VERIFICATION_BLOCK_RE.search(text)
    ):
        return PolicyOverride(Verdict.SCAM, "blocked_verification_with_credential_request")

    if not _TRUSTED_CHANNEL_RE.search(text) or active & _SAFE_VETO_BLOCKERS:
        return None
    if Signal.PAYMENT_REQUEST not in active or _EXPLICIT_SAFETY_RE.search(text):
        return PolicyOverride(Verdict.SAFE, "explicit_trusted_channel_guidance")
    return None
