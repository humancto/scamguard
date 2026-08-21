"""Deterministic signals and extractive evidence; no generated reasoning."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import EvidenceSpan
from .taxonomy import Category, RecommendedAction, Signal


@dataclass(frozen=True, slots=True)
class SignalMatch:
    signal: Signal
    evidence: EvidenceSpan


_WEB_TLDS = (
    "ae|ai|app|ar|au|bank|bd|be|biz|br|bw|bz|ca|cc|ch|cl|click|cloud|cn|co|com|de|"
    "dev|email|es|example|finance|fr|ga|gg|gl|gr|gs|help|hu|icu|id|in|info|io|it|jp|"
    "kr|lc|life|link|live|lv|ly|me|ml|mobi|mx|net|news|nl|no|nz|online|org|ph|pk|pl|"
    "pro|pt|pw|ro|ru|se|sg|shop|site|st|store|su|support|tech|tk|to|today|top|tr|tv|"
    "ty|uk|us|vg|vip|website|work|world|ws|xy|xyz|za"
)
_PATTERNS: tuple[tuple[Signal, re.Pattern[str]], ...] = (
    (
        Signal.ARTIFICIAL_URGENCY,
        re.compile(
            r"\b(?:act now|immediately|urgent|within \d+ hours?|final notice|final chance|"
            r"last chance|today only|urgente|inmediatamente|immédiatement|dringend|"
            r"onmiddellijk|sofort|immediatamente|imediatamente)\b",
            re.I,
        ),
    ),
    (
        Signal.AUTHORITY_IMPERSONATION,
        re.compile(
            r"\b(?:irs|police|court|customs|government|tax department|fbi|cbi|"
            r"fraud desk|customer care|fedex|apple pay|paytm|westpac|hdfc|anz)\b",
            re.I,
        ),
    ),
    (
        Signal.CREDENTIAL_REQUEST,
        re.compile(
            r"\b(?:password|login details?|log in|sign in|credentials?|personal details?|"
            r"bank(?:ing)? details?|routing details?|salary account|social security|"
            r"account info(?:rmation)?|billing address|email address|authentication information|"
            r"seed phrase|recovery phrase|pan card|kyc|photo id|contraseña|mot de passe|"
            r"wachtwoord|passwort|senha|c[oó]digo de verificaci[oó]n|code de vérification|"
            r"verificatiecode)\b",
            re.I,
        ),
    ),
    (
        Signal.OTP_REQUEST,
        re.compile(
            r"\b(?:otp|one[- ]time (?:code|password)|verification code|mfa code|"
            r"\d+[- ]digit code)\b",
            re.I,
        ),
    ),
    (
        Signal.PAYMENT_REQUEST,
        re.compile(
            r"\b(?:pay|payment|transfer|send|process|settle|deposit|buy|use|return)\b.{0,50}"
            r"(?:\$|£|€|usd|bdt|taka|inr|rs\.?|dollars?|fee|money|funds?|invoice|account|gift cards?|"
            r"crypto(?:currency)?|equipment|charge|toll|payment app)",
            re.I,
        ),
    ),
    (
        Signal.UNUSUAL_PAYMENT_METHOD,
        re.compile(
            r"\b(?:gift cards?|bitcoin|crypto(?:currency)?|wire transfer|western union|moneygram)\b",
            re.I,
        ),
    ),
    (Signal.SHORTENED_URL, re.compile(r"\b(?:bit\.ly|tinyurl\.com|t\.co|is\.gd|rb\.gy)/\S+", re.I)),
    (
        Signal.SUSPICIOUS_LINK,
        re.compile(
            r"<URL>|(?:\bhttps?\s*:\s*/{0,2}|\bhtt\s*ps?\s*:\s*/{0,2}|\bwww\.)[^\s<>]+|"
            rf"\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:{_WEB_TLDS})(?:/[^\s<>]*)?|"
            rf"\b[a-z0-9-]+\[\.\](?:{_WEB_TLDS})(?:/[^\s<>]*)?|"
            r"\b(?:click|tap|open|follow|visit|track).{0,24}\b(?:link|here)\b|"
            r"\b(?:haga clic aqu[ií]|clique aqui|clique ici|klik hier|klicken sie hier|"
            r"lien ci-dessous)\b|\bdownload\b.{0,40}\.apk\b",
            re.I,
        ),
    ),
    (
        Signal.CONTACT_DIVERSION,
        re.compile(
            r"<PHONE_NUMBER>|<US_BANK_NUMBER>|"
            r"\b(?:call|dial|text|contact|reply|calnumber)\b.{0,45}(?:this number|below number|"
            r"whatsapp|telegram|freefone|freephone|customer care|live operator|"
            r"\+?\d[\d *#-]{5,})|\b\+?\d[\d *#-]{6,}\d\b|\bcontact now\b",
            re.I,
        ),
    ),
    (
        Signal.SECRECY_ISOLATION,
        re.compile(
            r"\b(?:do not tell|keep this secret|don't tell|between us|do not contact)\b", re.I
        ),
    ),
    (
        Signal.GUARANTEED_RETURN,
        re.compile(
            r"\b(?:guaranteed|risk[- ]free)\b.{0,25}\b(?:profits?|returns?|income|investment)\b",
            re.I,
        ),
    ),
    (
        Signal.ADVANCE_FEE,
        re.compile(
            r"\b(?:processing|release|activation|customs|clearance|application|onboarding|"
            r"recovery|refundable)\s+(?:fee|bond)\b",
            re.I,
        ),
    ),
    (
        Signal.REMOTE_ACCESS_REQUEST,
        re.compile(
            r"\b(?:anydesk|teamviewer|remote desktop|remote support|screen sharing|remote access)\b",
            re.I,
        ),
    ),
    (
        Signal.THREAT_OR_COERCION,
        re.compile(
            r"\b(?:arrest|warrant|legal action|disconnect(?:ed|ion)?|"
            r"(?:account|atm|card|wallet|paytm|access).{0,35}(?:closed|suspended|block(?:ed)?|locked|"
            r"frozen|de-?activated|restricted)|account hacked|failed login|temporarily disabled|"
            r"expired identity|restore access|storage reached|suspendid[ao]|bloquead[ao]|"
            r"gesperrt|geblokkeerd|bloqu[eé]e?|sospes[ao]|disattivat[ao])\b",
            re.I,
        ),
    ),
    (
        Signal.TRUST_ACCELERATION,
        re.compile(r"\b(?:soulmate|destiny|trust me|only you understand|future together)\b", re.I),
    ),
    (
        Signal.TOO_GOOD_TO_BE_TRUE,
        re.compile(
            r"\b(?:winner|won|prize|lottery|free money|selected to receive|reward|awarded|"
            r"lucky draw|free gift|await collection|gained the sum|entitled to|shopping spree|"
            r"discount vouchers?|cashback|bonus|giveaway|eligible to claim|free smartphone|"
            r"free mobile|double (?:your )?(?:money|investment)|earn .{0,16} per day)\b|"
            r"(?:জিতেছেন|পুরস্কার|লটারি|বোনাস|ভাউচার)",
            re.I,
        ),
    ),
    (
        Signal.OFF_PLATFORM_REQUEST,
        re.compile(
            r"\b(?:move|continue|message me)\b.{0,30}\b(?:telegram|whatsapp|signal)\b", re.I
        ),
    ),
)


def _searchable_text(text: str) -> tuple[str, list[int]]:
    """Normalize evasive separators while retaining a map to original offsets."""
    characters: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text):
        surrounded_punctuation = (
            character in ".:;|"
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isspace()
            and text[index + 1].isspace()
        )
        if character == "\u200b" or surrounded_punctuation:
            continue
        normalized = " " if character.isspace() else character
        if normalized == " " and characters and characters[-1] == " ":
            continue
        characters.append(normalized)
        offsets.append(index)
    return "".join(characters), offsets


def extract_signal_matches(text: str) -> tuple[SignalMatch, ...]:
    searchable, offsets = _searchable_text(text)
    matches: list[SignalMatch] = []
    seen: set[Signal] = set()
    for signal, pattern in _PATTERNS:
        match = pattern.search(searchable)
        if match and signal not in seen:
            start = offsets[match.start()]
            end = offsets[match.end() - 1] + 1
            matches.append(
                SignalMatch(
                    signal,
                    EvidenceSpan(text[start:end], start, end),
                )
            )
            seen.add(signal)
    return tuple(matches)


def infer_category(text: str, signals: tuple[Signal, ...]) -> Category:
    value = text.casefold()
    rules = (
        (Category.DELIVERY_TOLL_PARKING, ("package", "parcel", "delivery", "toll", "parking")),
        (
            Category.CREDENTIAL_MFA,
            (
                "password",
                "otp",
                "verification code",
                "digit code",
                "fraud desk",
                "login",
                "sign in",
                "mailbox session",
                "mfa",
            ),
        ),
        (
            Category.GOVERNMENT_LEGAL,
            (
                "irs",
                "police",
                "court",
                "customs",
                "government",
                "tax",
                "detective",
                "warrant",
                "station",
            ),
        ),
        (
            Category.TECH_SUPPORT,
            ("tech support", "microsoft support", "virus", "anydesk", "teamviewer"),
        ),
        (
            Category.JOB_OPPORTUNITY,
            (
                "job",
                "recruiter",
                "hiring",
                "hired",
                "work from home",
                "interview",
                "onboarding fee",
                "position",
            ),
        ),
        (
            Category.FAMILY_EXECUTIVE,
            (
                "mom",
                "mum",
                "dad",
                "auntie",
                "grandpa",
                "son",
                "daughter",
                "ceo",
                "boss",
                "board meeting",
                "approve the paperwork",
                "temporary number",
                "phone broke",
            ),
        ),
        (
            Category.ROMANCE_RELATIONSHIP,
            (
                "romance",
                "soulmate",
                "future together",
                "planning our future",
                "love",
                "inheritance",
                "only you",
            ),
        ),
        (
            Category.INVESTMENT_CRYPTO,
            ("investment", "bitcoin", "crypto", "guaranteed return", "forex"),
        ),
        (Category.PRIZE_LOTTERY, ("winner", "prize", "lottery", "sweepstakes")),
        (Category.MARKETPLACE, ("marketplace", "buyer", "seller", "shipping agent", "listing")),
        (Category.PAYMENT_INVOICE, ("invoice", "payment", "wire transfer")),
        (
            Category.FINANCIAL_IMPERSONATION,
            (
                "bank",
                "card",
                "account frozen",
                "fraud department",
                "payroll",
                "routing details",
                "direct-deposit",
                "salary account",
            ),
        ),
        (Category.CHARITY, ("charity", "donation", "fundraiser")),
    )
    for category, keywords in rules:
        if any(keyword in value for keyword in keywords):
            return category
    return Category.OTHER_SCAM if signals else Category.UNKNOWN


def choose_action(signals: tuple[Signal, ...]) -> RecommendedAction:
    values = set(signals)
    if Signal.OTP_REQUEST in values or Signal.CREDENTIAL_REQUEST in values:
        return RecommendedAction.DO_NOT_SHARE_CODE
    if Signal.REMOTE_ACCESS_REQUEST in values:
        return RecommendedAction.DO_NOT_INSTALL_SOFTWARE
    if Signal.UNUSUAL_PAYMENT_METHOD in values or Signal.PAYMENT_REQUEST in values:
        return RecommendedAction.DO_NOT_PAY
    if Signal.SUSPICIOUS_LINK in values or Signal.SHORTENED_URL in values:
        return RecommendedAction.DO_NOT_OPEN_LINK
    if values:
        return RecommendedAction.VERIFY_OFFICIAL_CHANNEL
    return RecommendedAction.NO_ACTION
