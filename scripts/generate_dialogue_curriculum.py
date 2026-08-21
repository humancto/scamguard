#!/usr/bin/env python3
"""Generate paired multi-turn scam and legitimate dialogue curricula."""

# ruff: noqa: E501 -- keeping each dialogue turn intact makes label review safer.

from __future__ import annotations

import argparse
import hashlib
import json
import string
from collections import Counter
from pathlib import Path

GENERATOR_VERSION = 2
ROWS_PER_LABEL_PER_SCENARIO = 64

VALUES = {
    "amount": ["$75", "$240", "$890", "$1,600"],
    "bank": ["Harbor Bank", "Union Credit", "First Metro", "Community Bank"],
    "carrier": ["postal service", "courier", "parcel desk", "shipping service"],
    "code": ["four-digit code", "six-digit code", "verification code", "security code"],
    "day": ["today", "Monday", "Thursday", "August 28"],
    "employer": ["Northstar", "Pioneer", "Citywide", "Contoso"],
    "fee": ["$19", "$45", "$120", "$275"],
    "item": ["laptop", "phone", "tablet", "printer"],
    "payment": ["gift cards", "cryptocurrency", "a wire transfer", "a payment app"],
    "relative": ["grandson", "niece", "brother", "cousin"],
    "role": ["support specialist", "data coordinator", "reviewer", "remote assistant"],
    "ticket": ["A-1048", "Q-219", "R-774", "C-530"],
    "vendor": ["Northstar Shop", "Pioneer Services", "Citywide Market", "Contoso Help"],
}

SCENARIOS = {
    "remote_support": {
        "category": "CREDENTIAL_THEFT",
        "reference": "https://www.ic3.gov/PSA/2023/PSA230929",
        "scam": (
            "CALLER: I am with {vendor} security. Your {item} is sending us infection alerts.\n"
            "RECIPIENT: I did not schedule a support call.\n"
            "CALLER: This is an emergency case {ticket}. Open https://support.example/session and install the remote tool.\n"
            "RECIPIENT: Can I call the company using its official website?\n"
            "CALLER: No. The case will close {day}; stay on this call and read me the {code}."
        ),
        "safe": (
            "AGENT: Hello, this is the {vendor} appointment you scheduled for ticket {ticket}.\n"
            "CUSTOMER: Yes, I am calling through the support page saved in my account.\n"
            "AGENT: Great. Please open the diagnostic menu already built into your {item}.\n"
            "CUSTOMER: Will you need my password or a remote-access code?\n"
            "AGENT: No. We will never request a password, money transfer, or hidden remote session."
        ),
    },
    "government_case": {
        "category": "IDENTITY_IMPERSONATION",
        "reference": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
        "scam": (
            "CALLER: I am an investigator assigned to government case {ticket}.\n"
            "RECIPIENT: What is this about?\n"
            "CALLER: Your identity was connected to a crime. Do not call the public office because the file is sealed.\n"
            "RECIPIENT: How do I verify you?\n"
            "CALLER: Send a photo of your ID and bank statement here, then pay {fee} by {payment} {day}."
        ),
        "safe": (
            "CLERK: This is a general notice about case {ticket}; no identity or payment information is requested in this chat.\n"
            "RESIDENT: How can I confirm the notice?\n"
            "CLERK: End this conversation and find the agency number in the official public directory.\n"
            "RESIDENT: Should I send a copy of my ID first?\n"
            "CLERK: No. Do not send documents or pay anyone who contacts you unexpectedly."
        ),
    },
    "refund": {
        "category": "FINANCIAL",
        "reference": "https://consumer.ftc.gov/consumer-alerts/2024/03/never-move-your-money-protect-it-thats-scam",
        "scam": (
            "AGENT: {vendor} accidentally refunded {amount} too much to your account.\n"
            "CUSTOMER: I cannot see that refund in the official app.\n"
            "AGENT: The display is delayed. Install our screen-sharing tool from https://refund.example/help.\n"
            "CUSTOMER: What should I do after that?\n"
            "AGENT: While I watch, return the excess today using {payment}; do not contact your bank."
        ),
        "safe": (
            "CUSTOMER: I am checking refund ticket {ticket} for my {item}.\n"
            "AGENT: I found it in your signed-in {vendor} account. The refund is {amount}.\n"
            "CUSTOMER: Do I need to provide a bank code?\n"
            "AGENT: No. It returns automatically to the original payment method by {day}.\n"
            "CUSTOMER: Thanks. I will check the official app later."
        ),
    },
    "bank_fraud": {
        "category": "FINANCIAL",
        "reference": "https://consumer.ftc.gov/consumer-alerts/2024/03/never-move-your-money-protect-it-thats-scam",
        "scam": (
            "CALLER: {bank} fraud desk detected a transfer for {amount}.\n"
            "CUSTOMER: I do not recognize it.\n"
            "CALLER: To reverse it, read me the {code} that just arrived.\n"
            "CUSTOMER: The message says not to share it.\n"
            "CALLER: Ignore that warning or move the money to our protected account by {payment} {day}."
        ),
        "safe": (
            "BANK: {bank} declined a transfer for {amount}. This conversation will not ask for a security code.\n"
            "CUSTOMER: I do not recognize the transfer.\n"
            "BANK: End the chat and use the official app or the number printed on your card.\n"
            "CUSTOMER: Should I move my balance to protect it?\n"
            "BANK: No. Bank employees will never direct you to a protected account or request the {code}."
        ),
    },
    "delivery": {
        "category": "DELIVERY_TOLL",
        "reference": "https://www.uspis.gov/news/scam-article/smishing-package-tracking-text-scams",
        "scam": (
            "COURIER: Your {carrier} parcel under ticket {ticket} cannot be released.\n"
            "RECIPIENT: I am expecting a package.\n"
            "COURIER: Confirm your home address and card at https://parcel.example/confirm.\n"
            "RECIPIENT: Is there a charge?\n"
            "COURIER: Pay the {fee} redelivery fee {day} or the parcel will be destroyed."
        ),
        "safe": (
            "COURIER: Your opted-in {carrier} update for parcel {ticket} is available.\n"
            "RECIPIENT: Can I arrange pickup?\n"
            "COURIER: Yes. Open the official app you already installed and choose the staffed location.\n"
            "RECIPIENT: Is an online redelivery fee required?\n"
            "COURIER: No. This chat contains no payment link and asks for no card or identity details."
        ),
    },
    "job": {
        "category": "OPPORTUNITY",
        "reference": "https://consumer.ftc.gov/consumer-alerts/2025/08/how-spot-avoid-task-scams",
        "scam": (
            "RECRUITER: {employer} selected you for a remote {role} position without another interview.\n"
            "APPLICANT: What happens next?\n"
            "RECRUITER: Deposit our check, keep {amount}, and buy a {item} from our supplier.\n"
            "APPLICANT: Is there any fee?\n"
            "RECRUITER: Pay the refundable {fee} activation by {payment} {day} so work can begin."
        ),
        "safe": (
            "RECRUITER: {employer} scheduled your interview for the {role} role under ticket {ticket}.\n"
            "APPLICANT: Do I need to buy equipment before the call?\n"
            "RECRUITER: No. We do not send checks, charge an activation fee, or require a supplier.\n"
            "APPLICANT: When would payroll details be collected?\n"
            "RECRUITER: Only after a signed offer, through the company portal you reach independently."
        ),
    },
    "wrong_number": {
        "category": "RELATIONSHIP",
        "reference": "https://consumer.ftc.gov/articles/what-know-about-romance-scams",
        "scam": (
            "SENDER: Hi, is this Daniel? I may have saved the wrong number.\n"
            "RECIPIENT: No, sorry.\n"
            "SENDER: You seem kind. I run investments and would like to keep talking.\n"
            "RECIPIENT: I do not know you.\n"
            "SENDER: I can teach you a guaranteed strategy. Start with {amount} in cryptocurrency at https://invest.example/start."
        ),
        "safe": (
            "SENDER: Hi, is this Daniel? I may have saved the wrong number.\n"
            "RECIPIENT: No, this is not Daniel.\n"
            "SENDER: Sorry about that. I will remove the number.\n"
            "RECIPIENT: No problem.\n"
            "SENDER: Take care."
        ),
    },
    "insurance": {
        "category": "IDENTITY_IMPERSONATION",
        "reference": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
        "scam": (
            "CALLER: I am the claims officer for case {ticket}. Your {amount} benefit is ready.\n"
            "CUSTOMER: Which insurer are you calling from?\n"
            "CALLER: The national claims unit. Send your policy ID and bank details in this chat.\n"
            "CUSTOMER: Is there a processing cost?\n"
            "CALLER: Pay {fee} using {payment} {day}, and do not call the number on your card."
        ),
        "safe": (
            "AGENT: I am following up on the {vendor} claim you opened under case {ticket}.\n"
            "CUSTOMER: I can see that case in my saved account.\n"
            "AGENT: Good. No processing fee, gift card, bank transfer, or security code is required.\n"
            "CUSTOMER: Where should I upload the requested repair receipt?\n"
            "AGENT: Sign in through the official app you already use; do not follow a link from this chat."
        ),
    },
    "tax_collection": {
        "category": "IDENTITY_IMPERSONATION",
        "reference": "https://www.irs.gov/help/tax-scams",
        "scam": (
            "CALLER: I am with the tax enforcement unit about case {ticket}.\n"
            "RECIPIENT: I was not expecting a call.\n"
            "CALLER: You owe {amount}. Stay on the line or officers will arrive {day}.\n"
            "RECIPIENT: Can I check my official tax account first?\n"
            "CALLER: No. Pay {fee} now using {payment} and give me the receipt number."
        ),
        "safe": (
            "AGENT: I am returning the appointment you requested about notice {ticket}.\n"
            "TAXPAYER: I can see that notice in my official online account.\n"
            "AGENT: Please compare the notice there; this call will not demand immediate payment.\n"
            "TAXPAYER: Should I buy a gift card or stay on the line?\n"
            "AGENT: No. Use the payment options in your signed-in account after reviewing the notice."
        ),
    },
    "family_emergency": {
        "category": "IDENTITY_IMPERSONATION",
        "reference": "https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money",
        "scam": (
            "CALLER: It is your {relative}. I had an accident and need help.\n"
            "RECIPIENT: You sound different. What happened?\n"
            "CALLER: I am in custody. Do not tell the family or call anyone else.\n"
            "RECIPIENT: I want to verify this with you directly.\n"
            "CALLER: There is no time. Send {amount} for bail by {payment} {day}."
        ),
        "safe": (
            "CALLER: It is your {relative}. I am confirming our family dinner {day}.\n"
            "RECIPIENT: You sound different on this connection.\n"
            "CALLER: No problem. Call me back using the number already saved in your contacts.\n"
            "RECIPIENT: Is there an emergency or any money needed?\n"
            "CALLER: No. I am safe, there is no payment, and everyone else knows about the plan."
        ),
    },
    "investment": {
        "category": "RELATIONSHIP",
        "reference": "https://consumer.ftc.gov/articles/investment-scams",
        "scam": (
            "SENDER: I enjoyed our conversation and can show you my private investment method.\n"
            "RECIPIENT: What kind of investment is it?\n"
            "SENDER: A guaranteed cryptocurrency return with no risk; my screenshots prove it works.\n"
            "RECIPIENT: Can I research the company independently?\n"
            "SENDER: The window closes {day}. Deposit {amount} at https://invest.example/private now."
        ),
        "safe": (
            "ADVISER: This is the investment-review meeting you booked under {ticket}.\n"
            "CUSTOMER: Are any returns guaranteed?\n"
            "ADVISER: No. Every investment can lose value, and you should review the written risks.\n"
            "CUSTOMER: Do I need to transfer money during this chat?\n"
            "ADVISER: No. Research the registration independently and decide later through your account."
        ),
    },
    "marketplace_overpayment": {
        "category": "FINANCIAL",
        "reference": "https://consumer.ftc.gov/articles/how-spot-avoid-and-report-fake-check-scams",
        "scam": (
            "BUYER: I want your {item} and sent a check for {amount} more than the price.\n"
            "SELLER: Why is the amount too high?\n"
            "BUYER: My mover made a mistake. Deposit it and the bank will make the funds available.\n"
            "SELLER: What should I do with the extra amount?\n"
            "BUYER: Send it to the mover by {payment} {day} before the check finishes clearing."
        ),
        "safe": (
            "BUYER: Is your {item} still available through the marketplace listing?\n"
            "SELLER: Yes. The listed price and pickup details are in the platform.\n"
            "BUYER: I will pay exactly that amount through the platform after inspection.\n"
            "SELLER: Will you send a check or ask me to pay a mover?\n"
            "BUYER: No. There is no overpayment, refund, outside payment, or third-party courier."
        ),
    },
}

NEUTRAL_SUFFIXES = (
    "",
    "PARTICIPANT: I understand.",
    "PARTICIPANT: I will review the conversation.",
    "PARTICIPANT: I need a moment to think.",
    "PARTICIPANT: Thanks for explaining.",
    "PARTICIPANT: I will check my records.",
    "PARTICIPANT: Let me consider that.",
    "PARTICIPANT: I am going to end this chat now.",
    "PARTICIPANT: I will decide after I verify the details.",
    "PARTICIPANT: I have written down what you said.",
    "PARTICIPANT: I will not make a decision on this call.",
    "PARTICIPANT: I need to speak with someone I trust first.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def field_names(template: str) -> list[str]:
    return sorted({field for _, field, _, _ in string.Formatter().parse(template) if field})


def render(template: str, scenario: str, label: str, index: int) -> str:
    values = {}
    for field in field_names(template):
        options = VALUES[field]
        choice = int(
            short_hash(f"dialogue-v{GENERATOR_VERSION}:{scenario}:{label}:{index}:{field}", 8),
            16,
        )
        values[field] = options[choice % len(options)]
    text = template.format(**values)
    # Harmless discourse variation reduces reliance on a fixed opening token.
    prefixes = ("", "TRANSCRIPT:\n", "CHAT LOG:\n", "Recent conversation:\n")
    prefix = prefixes[index % len(prefixes)]
    suffix = NEUTRAL_SUFFIXES[(index // len(prefixes)) % len(NEUTRAL_SUFFIXES)]
    return prefix + text + (f"\n{suffix}" if suffix else "")


def generate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, spec in SCENARIOS.items():
        for label_key, label in (("scam", "SCAM"), ("safe", "SAFE")):
            template = str(spec[label_key])
            for index in range(ROWS_PER_LABEL_PER_SCENARIO):
                text = render(template, scenario, label, index)
                row_id = "dialogue-" + short_hash(f"{scenario}:{label}:{index}:{text}")
                rows.append(
                    {
                        "id": row_id,
                        "text": text,
                        "label": label,
                        "category": str(spec["category"]) if label == "SCAM" else "NONE",
                        "source": f"scamguard_synthetic_dialogue_v{GENERATOR_VERSION}",
                        "source_label": label.casefold(),
                        "license": "Apache-2.0",
                        "split": "train",
                        "family_id": (
                            f"synthetic:dialogue:{scenario}:{label.casefold()}:"
                            f"v{GENERATOR_VERSION}"
                        ),
                        "is_synthetic": True,
                        "synthetic_method": (
                            "paired_deterministic_slot_filling_original_advisory_grounded_copy"
                        ),
                        "pattern_reference": str(spec["reference"]),
                        "source_language": "English",
                        "scenario": scenario,
                        "generator_version": GENERATOR_VERSION,
                        "context_policy": "five_turn_dialogue_under_mobile_256_token_window",
                    }
                )
    return sorted(rows, key=lambda row: str(row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("data/generated/dialogue_curriculum.jsonl")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/generated/dialogue_curriculum_manifest.json")
    )
    args = parser.parse_args()
    rows = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "license": "Apache-2.0",
        "method": "paired deterministic multi-turn scenario grammars with original copy",
        "used_for_fitting": True,
        "used_for_threshold": False,
        "selection_or_test_rows_modified": False,
        "rows": len(rows),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "scenarios": dict(Counter(str(row["scenario"]) for row in rows)),
        "sha256": sha256(args.output),
        "pattern_references": sorted({str(row["pattern_reference"]) for row in rows}),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
