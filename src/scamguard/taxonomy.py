"""Frozen ScamGuard v0 taxonomy."""

from __future__ import annotations

from enum import StrEnum


class Verdict(StrEnum):
    SAFE = "SAFE"
    UNCERTAIN = "UNCERTAIN"
    SCAM = "SCAM"


class Category(StrEnum):
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"
    DELIVERY_TOLL_PARKING = "DELIVERY_TOLL_PARKING"
    FINANCIAL_IMPERSONATION = "FINANCIAL_IMPERSONATION"
    PAYMENT_INVOICE = "PAYMENT_INVOICE"
    GOVERNMENT_LEGAL = "GOVERNMENT_LEGAL"
    FAMILY_EXECUTIVE = "FAMILY_EXECUTIVE"
    TECH_SUPPORT = "TECH_SUPPORT"
    JOB_OPPORTUNITY = "JOB_OPPORTUNITY"
    INVESTMENT_CRYPTO = "INVESTMENT_CRYPTO"
    PRIZE_LOTTERY = "PRIZE_LOTTERY"
    MARKETPLACE = "MARKETPLACE"
    ROMANCE_RELATIONSHIP = "ROMANCE_RELATIONSHIP"
    CREDENTIAL_MFA = "CREDENTIAL_MFA"
    CHARITY = "CHARITY"
    OTHER_SCAM = "OTHER_SCAM"


class Signal(StrEnum):
    ARTIFICIAL_URGENCY = "artificial_urgency"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    CREDENTIAL_REQUEST = "credential_request"
    OTP_REQUEST = "otp_request"
    PAYMENT_REQUEST = "payment_request"
    UNUSUAL_PAYMENT_METHOD = "unusual_payment_method"
    SUSPICIOUS_LINK = "suspicious_link"
    SHORTENED_URL = "shortened_url"
    CONTACT_DIVERSION = "contact_diversion"
    SECRECY_ISOLATION = "secrecy_isolation"
    GUARANTEED_RETURN = "guaranteed_return"
    ADVANCE_FEE = "advance_fee"
    REMOTE_ACCESS_REQUEST = "remote_access_request"
    THREAT_OR_COERCION = "threat_or_coercion"
    TRUST_ACCELERATION = "trust_acceleration"
    TOO_GOOD_TO_BE_TRUE = "too_good_to_be_true"
    OFF_PLATFORM_REQUEST = "off_platform_request"


class RecommendedAction(StrEnum):
    NO_ACTION = "NO_ACTION"
    VERIFY_OFFICIAL_CHANNEL = "VERIFY_OFFICIAL_CHANNEL"
    DO_NOT_OPEN_LINK = "DO_NOT_OPEN_LINK"
    DO_NOT_REPLY = "DO_NOT_REPLY"
    DO_NOT_PAY = "DO_NOT_PAY"
    DO_NOT_SHARE_CODE = "DO_NOT_SHARE_CODE"
    DO_NOT_INSTALL_SOFTWARE = "DO_NOT_INSTALL_SOFTWARE"
    ESCALATE_TRUSTED_PERSON = "ESCALATE_TRUSTED_PERSON"
