#!/usr/bin/env python3
"""Generate original action-state dialogues grounded in FTC-described scam patterns.

No FTC robocall transcript is an input to this generator. The pattern summaries and URLs below
ground scenario selection; every dialogue line is original ScamGuard copy under Apache-2.0.
"""

# ruff: noqa: E501 -- reviewed dialogue turns stay intact for line-by-line audit.

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scamguard.metrics import file_sha256
from scamguard.signals import extract_signal_matches

try:
    from scripts.generate_call_action_states import CONTRAST_STATES, TARGET_KEYS
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from generate_call_action_states import CONTRAST_STATES, TARGET_KEYS  # type: ignore[no-redef]

GENERATOR_VERSION = 1
SOURCE = "scamguard_ftc_pattern_action_states_v1"
LICENSE = "Apache-2.0"
METHOD = "original_counterfactual_dialogues_grounded_in_ftc_authored_pattern_descriptions_v1"
FTC_PATTERN_PAGE = "https://consumer.ftc.gov/features/robocall-scam-examples"
FTC_WEBSITE_POLICY = "https://www.ftc.gov/policy-notices/website-policy"
VALIDATION_PATTERNS = ("student_loan_relief", "utility_rebate")

STRUCTURES = (
    (
        "live_inbound",
        "CALLER: Good afternoon, this is the {desk}.\n"
        "RECIPIENT: What is this about?\n"
        "CALLER: {claim}.\n"
        "RECIPIENT: I did not expect that.\n"
        "CALLER: {action}.\n"
        "RECIPIENT: I need a moment.\n"
        "CALLER: I can remain available while you review the notice.",
    ),
    (
        "automated_callback",
        "CALLER: This automated message is from the {desk}.\n"
        "RECIPIENT: The message was answered.\n"
        "CALLER: {claim}.\n"
        "RECIPIENT: What is the next step?\n"
        "CALLER: {action}.\n"
        "RECIPIENT: I will review it first.\n"
        "CALLER: The notice remains on file while you decide.",
    ),
    (
        "transferred_specialist",
        "AGENT: You have reached the {desk}.\n"
        "CUSTOMER: I was transferred here.\n"
        "AGENT: {claim}.\n"
        "CUSTOMER: I want to understand the request.\n"
        "AGENT: {action}.\n"
        "CUSTOMER: I have not agreed to anything.\n"
        "AGENT: Take time to review the details.",
    ),
)

COMMON_ROUTINE = (
    "This is only a general notice; no payment, password, code, or account detail is requested",
    "The information can be reviewed later and nothing needs to be completed during this call",
    "No response by phone is required; the ordinary account record will remain unchanged",
    "This informational message does not ask you to move money or disclose private information",
)
COMMON_VERIFIED = (
    "Do not share any code or payment with me; end this call and open the official app yourself",
    "Never provide private account details here; independently call the published service number",
    "Do not follow instructions from this call; verify the notice using the number on your card",
    "Keep every credential private and navigate to the known official website yourself",
)
COMMON_UNRESOLVED = (
    "A follow-up action is mentioned, but this call does not verify its destination or authority",
    "The file lists another step, although the channel and recipient have not been confirmed",
    "A response may be requested later, but the identity behind this message is still unverified",
    "The notice suggests action could be needed, while the account and contact path remain unclear",
)

PATTERNS: dict[str, dict[str, object]] = {
    "marketplace_charge": {
        "desk": "online-order review desk",
        "reference": "https://www.ftc.gov/media/87483",
        "category": "IDENTITY_IMPERSONATION",
        "claims": (
            "a high-value purchase appears under your profile",
            "an unfamiliar electronics order is waiting for confirmation",
            "a charge from an online marketplace has been placed on hold",
            "the ordering system marked a recent purchase as unusual",
        ),
        "harmful": (
            "Read me the one-time verification code so I can cancel the purchase",
            "Provide your banking details and one-time verification code to reverse the charge",
            "Install this remote support tool so I can remove the order",
            "Open the link I send and log in there while I remain on the line",
        ),
        "irreversible": False,
        "pressure": False,
    },
    "legal_enforcement": {
        "desk": "federal case notification unit",
        "reference": "https://www.ftc.gov/media/87480",
        "category": "IDENTITY_IMPERSONATION",
        "claims": (
            "a legal complaint has been associated with your identity",
            "the system lists a court matter under your name",
            "an enforcement file shows an unresolved identity issue",
            "a case record is marked for immediate contact",
        ),
        "harmful": (
            "Immediately transfer funds to the safe account I provide to avoid an arrest warrant",
            "Buy gift cards today and read the codes to settle the legal action",
            "Send the processing fee by wire transfer before the case is escalated",
            "Keep this secret and provide your social security and bank details now",
        ),
        "irreversible": True,
        "pressure": True,
    },
    "border_package": {
        "desk": "border parcel review center",
        "reference": "https://www.ftc.gov/media/87470",
        "category": "FINANCIAL",
        "claims": (
            "a parcel linked to your name has been stopped for review",
            "customs records show a held shipment using your address",
            "an incoming package is listed with an unpaid border charge",
            "a seized delivery has been connected to your contact information",
        ),
        "harmful": (
            "Pay the customs clearance fee in bitcoin immediately to release the package",
            "Wire the release fee to the account I give you before the parcel is destroyed",
            "Buy a prepaid gift card and provide its code to settle the customs fee",
            "Transfer the refundable bond now and do not contact anyone else",
        ),
        "irreversible": True,
        "pressure": True,
    },
    "technical_support": {
        "desk": "device security support desk",
        "reference": FTC_PATTERN_PAGE,
        "category": "CREDENTIAL_THEFT",
        "claims": (
            "your device account generated a security warning",
            "unusual activity was reported for a computer on your profile",
            "the support system detected a possible account compromise",
            "a device registered to you is showing a critical alert",
        ),
        "harmful": (
            "Install the remote support application and give me the connection code",
            "Start screen sharing and read me the one-time password from your phone",
            "Download this remote desktop tool so I can control the device",
            "Log in through the link I send and provide the verification code",
        ),
        "irreversible": False,
        "pressure": False,
    },
    "tax_debt_relief": {
        "desk": "tax settlement enrollment desk",
        "reference": FTC_PATTERN_PAGE,
        "category": "FINANCIAL",
        "claims": (
            "your old tax balance may qualify for a new settlement",
            "a tax-debt reduction program appears available for your account",
            "the enrollment system marked your tax file as eligible",
            "a limited tax hardship option has been attached to your record",
        ),
        "harmful": (
            "Pay the processing fee today and give me your bank account information",
            "Send the activation fee by wire transfer so I can enroll you",
            "Provide your social security and card details to process the application fee",
            "Transfer the refundable fee immediately to secure the settlement",
        ),
        "irreversible": True,
        "pressure": True,
    },
    "social_security_suspension": {
        "desk": "benefit identity review line",
        "reference": FTC_PATTERN_PAGE,
        "category": "CREDENTIAL_THEFT",
        "claims": (
            "suspicious activity has been associated with your benefit number",
            "your identity record is marked for possible suspension",
            "the benefits system reports an unresolved security event",
            "access to a government benefit profile may be restricted",
        ),
        "harmful": (
            "Give me your social security number and verification code immediately",
            "Read your bank details to me now to prevent the account from being suspended",
            "Provide your password and one-time code before the benefit record is frozen",
            "Keep this confidential and confirm your full social security and card number",
        ),
        "irreversible": False,
        "pressure": True,
    },
    "interest_reduction": {
        "desk": "card-rate review center",
        "reference": FTC_PATTERN_PAGE,
        "category": "FINANCIAL",
        "claims": (
            "your payment history may qualify for a lower card rate",
            "a reduced-interest offer has been attached to your profile",
            "the system shows a possible card-rate adjustment",
            "a credit account is listed for a promotional rate review",
        ),
        "harmful": (
            "Provide the card number and pay the activation fee during this final call",
            "Read me your security code and bank details to process the rate reduction",
            "Send the enrollment fee now and confirm the one-time password",
            "Transfer the processing fee today before the offer expires",
        ),
        "irreversible": True,
        "pressure": True,
    },
    "auto_warranty": {
        "desk": "vehicle coverage review line",
        "reference": FTC_PATTERN_PAGE,
        "category": "FINANCIAL",
        "claims": (
            "the coverage record for your vehicle may be out of date",
            "a vehicle service plan is listed as approaching expiration",
            "the system shows a possible gap in manufacturer coverage",
            "your vehicle profile is eligible for a coverage review",
        ),
        "harmful": (
            "Pay the activation fee now and provide the card number to extend coverage",
            "Send the processing fee today and read me your bank account details",
            "Transfer the enrollment payment during this final chance to keep coverage",
            "Give me the one-time verification code and banking details to start the plan",
        ),
        "irreversible": True,
        "pressure": True,
    },
    "student_loan_relief": {
        "desk": "education debt enrollment center",
        "reference": FTC_PATTERN_PAGE,
        "category": "CREDENTIAL_THEFT",
        "claims": (
            "a new repayment option may be available for your education debt",
            "your loan profile appears eligible for a hardship program",
            "the enrollment system found a possible payment-reduction plan",
            "an alternative student-debt arrangement is listed for review",
        ),
        "harmful": (
            "Pay the application fee and give me your federal student account password",
            "Provide your social security and bank details to process the enrollment fee",
            "Send the activation payment today and read me the verification code",
            "Log in through my link and transfer the processing fee while I stay on the line",
        ),
        "irreversible": True,
        "pressure": False,
    },
    "utility_rebate": {
        "desk": "energy billing adjustment desk",
        "reference": FTC_PATTERN_PAGE,
        "category": "FINANCIAL",
        "claims": (
            "a possible overcharge was found on your energy account",
            "the billing system lists a rebate for an earlier utility payment",
            "a supplier adjustment may have created a refund on your account",
            "your service profile is marked for a billing credit review",
        ),
        "harmful": (
            "Pay the processing fee and provide your bank account to receive the rebate",
            "Read me the card number and one-time code so I can release the refund",
            "Transfer the activation fee now to the account I provide",
            "Install the remote support tool and log in to confirm the billing credit",
        ),
        "irreversible": True,
        "pressure": False,
    },
}


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def action_targets(state: str, *, irreversible: bool, pressure: bool) -> dict[str, bool]:
    values = {
        "sensitive_action_language": state != "routine_safe",
        "requested_disclosure_or_transfer": state == "harmful_scam",
        "caller_controls_target": state == "harmful_scam",
        "official_self_navigation": state == "verified_safe",
        "independent_verification": state == "verified_safe",
        "pressure_or_secrecy": state == "harmful_scam" and pressure,
        "irreversible_action": state == "harmful_scam" and irreversible,
    }
    if tuple(values) != TARGET_KEYS:
        raise RuntimeError("FTC pattern target order changed")
    return values


def generate() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    for pattern, spec in PATTERNS.items():
        claims = tuple(spec["claims"])
        harmful = tuple(spec["harmful"])
        for structure, template in STRUCTURES:
            for variant in range(4):
                family_key = f"v{GENERATOR_VERSION}:{pattern}:{structure}:{variant}"
                contrast_id = "ftc-pattern-state-" + short_hash(family_key)
                common_frame = template.format(
                    desk=spec["desk"], claim=claims[variant], action="<DECISIVE_ACTION>"
                )
                common_hash = hashlib.sha256(common_frame.encode("utf-8")).hexdigest()
                endings = {
                    "routine_safe": COMMON_ROUTINE[variant],
                    "verified_safe": COMMON_VERIFIED[variant],
                    "unresolved": COMMON_UNRESOLVED[variant],
                    "harmful_scam": harmful[variant],
                }
                destination = validation if pattern in VALIDATION_PATTERNS else train
                split = "validation" if pattern in VALIDATION_PATTERNS else "train"
                for state in CONTRAST_STATES:
                    label = {
                        "routine_safe": "SAFE",
                        "verified_safe": "SAFE",
                        "unresolved": "UNCERTAIN",
                        "harmful_scam": "SCAM",
                    }[state]
                    text = template.format(
                        desk=spec["desk"], claim=claims[variant], action=endings[state]
                    )
                    row = {
                        "id": f"{contrast_id}-{state}",
                        "text": text,
                        "label": label,
                        "category": spec["category"] if label == "SCAM" else "NONE",
                        "source": SOURCE,
                        "source_label": f"ftc_pattern_grounded:{pattern}:{state}",
                        "license": LICENSE,
                        "split": split,
                        "family_id": f"synthetic:ftc_pattern:{pattern}:{structure}:{variant}:v1",
                        "contrast_id": contrast_id,
                        "contrast_state": state,
                        "is_synthetic": True,
                        "synthetic_method": METHOD,
                        "source_language": "English",
                        "scenario": pattern,
                        "dialogue_structure": structure,
                        "action_targets": action_targets(
                            state,
                            irreversible=bool(spec["irreversible"]),
                            pressure=bool(spec["pressure"]),
                        ),
                        "action_verdict_weight": 0.25,
                        "shared_context_sha256": common_hash,
                        "minimal_contrast_field": "decisive_caller_action",
                        "decisive_action_precedes_shared_continuation": True,
                        "pattern_reference": spec["reference"],
                        "rights_reference": FTC_WEBSITE_POLICY,
                        "source_grounding": "FTC-authored fraud-pattern description only",
                        "external_transcript_text_copied": False,
                        "external_benchmark_text_copied": False,
                        "generator_version": GENERATOR_VERSION,
                    }
                    if state == "harmful_scam" and not extract_signal_matches(text):
                        raise ValueError(f"harmful FTC pattern lacks deterministic evidence: {row['id']}")
                    destination.append(row)
    return (
        sorted(train, key=lambda row: str(row["id"])),
        sorted(validation, key=lambda row: str(row["id"])),
    )


def validate(train: list[dict[str, object]], validation: list[dict[str, object]]) -> None:
    rows = train + validation
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("FTC pattern generator produced duplicate IDs")
    texts = [" ".join(str(row["text"]).casefold().split()) for row in rows]
    if len(texts) != len(set(texts)):
        raise ValueError("FTC pattern generator produced duplicate texts")
    train_patterns = {str(row["scenario"]) for row in train}
    validation_patterns = {str(row["scenario"]) for row in validation}
    if train_patterns & validation_patterns or validation_patterns != set(VALIDATION_PATTERNS):
        raise ValueError("FTC pattern split is not scenario-disjoint")
    for rows_for_split in (train, validation):
        families = {str(row["contrast_id"]) for row in rows_for_split}
        for family in families:
            states = {
                str(row["contrast_state"])
                for row in rows_for_split
                if row["contrast_id"] == family
            }
            if states != set(CONTRAST_STATES):
                raise ValueError(f"incomplete FTC pattern contrast: {family}")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/generated/ftc_pattern_action_states_v1")
    )
    args = parser.parse_args()
    train, validation = generate()
    validate(train, validation)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "source": SOURCE,
        "license": LICENSE,
        "method": METHOD,
        "pattern_reference": FTC_PATTERN_PAGE,
        "rights_reference": FTC_WEBSITE_POLICY,
        "external_transcript_text_copied": False,
        "pattern_count": len(PATTERNS),
        "train_patterns": sorted(set(PATTERNS) - set(VALIDATION_PATTERNS)),
        "validation_patterns": list(VALIDATION_PATTERNS),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_families": len({str(row["contrast_id"]) for row in train}),
        "validation_families": len({str(row["contrast_id"]) for row in validation}),
        "labels": dict(Counter(str(row["label"]) for row in train + validation)),
        "action_target_keys": list(TARGET_KEYS),
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
