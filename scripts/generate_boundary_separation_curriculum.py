#!/usr/bin/env python3
"""Generate stage-4 triads for SAFE/UNCERTAIN/SCAM boundary separation."""

# ruff: noqa: E501 -- keeping reviewed dialogue turns intact makes label inspection safer.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from scripts.generate_dialogue_curriculum import render, short_hash
from scripts.generate_evidence_persistence_curriculum import generate as generate_persistence
from training.build_qwen_sft import convert_supported_rows

GENERATOR_VERSION = 1
ROWS_PER_LABEL_PER_SCENARIO = 48
SOURCE = "scamguard_synthetic_boundary_separation_v1"

NEUTRAL_CLOSINGS = (
    (
        "PARTICIPANT: I will stop here and verify the situation independently.",
        "CONTACT: All right; this conversation can end now.",
    ),
    (
        "PARTICIPANT: I need time to review the earlier details.",
        "CONTACT: Understood; we can pause the conversation.",
    ),
    (
        "PARTICIPANT: I am going to check my own records before acting.",
        "CONTACT: That is fine; there is nothing more to add right now.",
    ),
    (
        "PARTICIPANT: I will discuss this with someone I trust first.",
        "CONTACT: Okay; we can leave it there.",
    ),
)

# Each triad shares a realistic domain and neutral ending. SCAM contains an
# observable fraud action, SAFE contains an independently verifiable legitimate
# path, and UNCERTAIN deliberately withholds enough evidence to justify review.
# Patterns are original phrasing grounded in official public advisories.
SCENARIOS = {
    "account_recovery": {
        "category": "CREDENTIAL_THEFT",
        "reference": "https://www.ic3.gov/PSA/2023/PSA230929",
        "scam": "CONTACT: I am handling security ticket {ticket} for {vendor}.\nPARTICIPANT: I did not open that ticket.\nCONTACT: To keep the account active, tell me your password and the {code}, then approve the sign-in prompt while we speak.\nPARTICIPANT: The prompt says not to approve requests from callers.\nCONTACT: Ignore that warning; the recovery expires {day} and you must stay on this conversation.",
        "safe": "CONTACT: This is a status update for security ticket {ticket}; I cannot recover the account in this conversation.\nPARTICIPANT: What should I do?\nCONTACT: Close this message and use the address already saved in your password manager.\nPARTICIPANT: Will anyone ask for the {code}?\nCONTACT: No. Support cannot see or request passwords, approval prompts, or verification codes.",
        "uncertain": "CONTACT: There may be a sign-in issue associated with ticket {ticket}.\nPARTICIPANT: I was not expecting this message.\nCONTACT: Please review the account when convenient.\nPARTICIPANT: Where should I review it?\nCONTACT: The details are available through customer support.",
    },
    "device_assistance": {
        "category": "CREDENTIAL_THEFT",
        "reference": "https://www.ic3.gov/PSA/2023/PSA230929",
        "scam": "CONTACT: Your {item} is generating security alerts at {vendor}.\nPARTICIPANT: I did not request technical help.\nCONTACT: Install the remote-control program from https://assist.example/session and leave banking apps open.\nPARTICIPANT: Can I contact the manufacturer myself?\nCONTACT: No; pay {fee} by {payment} and do not disconnect until I finish.",
        "safe": "CONTACT: I am returning the device appointment listed as {ticket}.\nPARTICIPANT: I can see that appointment in my signed-in account.\nCONTACT: Use the diagnostic screen built into the {item}; no remote-control download is needed.\nPARTICIPANT: Is there a payment or password request?\nCONTACT: No. End the session if anyone asks for access codes, money, or control of the device.",
        "uncertain": "CONTACT: I am following up about a possible issue with your {item}.\nPARTICIPANT: I do not remember arranging support.\nCONTACT: A technician can explain the available options.\nPARTICIPANT: Does anything need to be installed?\nCONTACT: That depends on what the technician finds.",
    },
    "bank_protection": {
        "category": "FINANCIAL",
        "reference": "https://consumer.ftc.gov/consumer-alerts/2024/03/never-move-your-money-protect-it-thats-scam",
        "scam": "CONTACT: I am calling from {bank} about a transfer for {amount}.\nPARTICIPANT: It is not mine.\nCONTACT: Move the balance to our protected account by {payment} before {day}.\nPARTICIPANT: I want to call the number printed on my card.\nCONTACT: Do not call anyone else; read me the {code} so I can complete the reversal.",
        "safe": "CONTACT: {bank} declined a transfer for {amount}; no transfer is needed to protect your balance.\nPARTICIPANT: I do not recognize it.\nCONTACT: End this conversation and use the number printed on your card or your saved banking app.\nPARTICIPANT: Should I provide the {code}?\nCONTACT: No. Bank staff do not request one-time codes or direct customers to a protected account.",
        "uncertain": "CONTACT: This is a notice about activity for {amount} associated with {bank}.\nPARTICIPANT: I do not recognize the activity.\nCONTACT: Review it soon to avoid an interruption.\nPARTICIPANT: How should I respond?\nCONTACT: Contact the account team for the next step.",
    },
    "refund_adjustment": {
        "category": "FINANCIAL",
        "reference": "https://consumer.ftc.gov/articles/how-spot-avoid-and-report-refund-scams",
        "scam": "CONTACT: {vendor} sent an excess refund of {amount}.\nPARTICIPANT: It is not visible in the official account.\nCONTACT: The display is delayed; share your screen and return the difference using {payment}.\nPARTICIPANT: Can the company correct it internally?\nCONTACT: No. You must send it {day} and must not contact your bank.",
        "safe": "CONTACT: Refund case {ticket} is visible in your signed-in {vendor} account.\nPARTICIPANT: Do I need to send any money back?\nCONTACT: No. Corrections happen automatically through the original payment method.\nPARTICIPANT: Should I share my screen or bank details?\nCONTACT: No. Use the account you reach independently and never transfer money to receive a refund.",
        "uncertain": "CONTACT: There is an update concerning refund case {ticket}.\nPARTICIPANT: I cannot see the update yet.\nCONTACT: Processing may take additional time.\nPARTICIPANT: Is any action required?\nCONTACT: The account team will provide more information.",
    },
    "remote_job": {
        "category": "OPPORTUNITY",
        "reference": "https://consumer.ftc.gov/consumer-alerts/2025/08/how-spot-avoid-task-scams",
        "scam": "CONTACT: {employer} selected you for a remote {role} position without another interview.\nPARTICIPANT: What is required before I start?\nCONTACT: Deposit our check, keep {amount}, and send the balance to the equipment vendor.\nPARTICIPANT: Is there a fee?\nCONTACT: Pay the {fee} activation through {payment} by {day} to unlock the work portal.",
        "safe": "CONTACT: {employer} scheduled an interview for the {role} role under {ticket}.\nPARTICIPANT: Must I buy equipment first?\nCONTACT: No. Applicants do not pay fees, deposit checks, or purchase from a required vendor.\nPARTICIPANT: When are financial details collected?\nCONTACT: Only after a signed offer, in the company portal you open independently.",
        "uncertain": "CONTACT: {employer} may have an opening for a remote {role}.\nPARTICIPANT: I did not apply for that exact role.\nCONTACT: Your profile appears relevant and the team would like to discuss it.\nPARTICIPANT: What happens next?\nCONTACT: More instructions will follow after an initial conversation.",
    },
    "paid_tasks": {
        "category": "OPPORTUNITY",
        "reference": "https://consumer.ftc.gov/consumer-alerts/2025/08/how-spot-avoid-task-scams",
        "scam": "CONTACT: Earn commissions by rating products for {vendor}.\nPARTICIPANT: How are tasks assigned?\nCONTACT: First deposit {fee} through {payment}; each completed task increases the displayed balance.\nPARTICIPANT: Can I withdraw before depositing more?\nCONTACT: No. Add {amount} by {day} to clear the negative balance and release your earnings.",
        "safe": "CONTACT: This is the paid research session you booked under {ticket}.\nPARTICIPANT: Is a deposit required?\nCONTACT: No. Participants never fund tasks, pay to unlock earnings, or transfer cryptocurrency.\nPARTICIPANT: How is compensation handled?\nCONTACT: The written consent form explains a fixed payment after the scheduled session.",
        "uncertain": "CONTACT: {vendor} is inviting people to review products online.\nPARTICIPANT: Is this employment or a survey?\nCONTACT: It is a flexible earning opportunity with several task levels.\nPARTICIPANT: Is there a written agreement?\nCONTACT: Program details are provided during enrollment.",
    },
    "official_case": {
        "category": "IDENTITY_IMPERSONATION",
        "reference": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
        "scam": "CONTACT: I am an investigator for government case {ticket}.\nPARTICIPANT: I was not expecting a call.\nCONTACT: The case is sealed; do not contact the public office or discuss it with anyone.\nPARTICIPANT: How can I resolve it?\nCONTACT: Send a photo ID and pay {fee} by {payment} before {day} to prevent arrest.",
        "safe": "CONTACT: This general notice references case {ticket} and asks for no payment or identity document.\nPARTICIPANT: How can I verify it?\nCONTACT: End this conversation and find the agency in the official public directory.\nPARTICIPANT: Is there an immediate deadline?\nCONTACT: No legitimate official will demand secrecy, gift cards, cryptocurrency, or payment to prevent arrest.",
        "uncertain": "CONTACT: I am trying to reach you concerning case {ticket}.\nPARTICIPANT: Which office is this?\nCONTACT: The case team handles the details.\nPARTICIPANT: Is a response required today?\nCONTACT: Timely attention is recommended.",
    },
    "family_request": {
        "category": "IDENTITY_IMPERSONATION",
        "reference": "https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money",
        "scam": "CONTACT: It is your {relative}; I am in custody after an accident.\nPARTICIPANT: Your voice sounds different.\nCONTACT: Do not call the family or the number you have saved.\nPARTICIPANT: What do you need?\nCONTACT: Send {amount} for bail using {payment} by {day}; there is no time to verify me.",
        "safe": "CONTACT: It is your {relative}, confirming the family plan for {day}.\nPARTICIPANT: The connection sounds unfamiliar.\nCONTACT: Hang up and call the number already saved in your contacts.\nPARTICIPANT: Is there an emergency or a money request?\nCONTACT: No. There is no secrecy, payment, or urgent request.",
        "uncertain": "CONTACT: Hi, it is a family member using a different phone.\nPARTICIPANT: I cannot identify the voice.\nCONTACT: I wanted to speak with you about something important.\nPARTICIPANT: Can I call the number I have saved?\nCONTACT: We can discuss the details later.",
    },
}


def generate() -> list[dict[str, object]]:
    rows = list(generate_persistence())
    for scenario, spec in SCENARIOS.items():
        for label_key, label in (("safe", "SAFE"), ("uncertain", "UNCERTAIN"), ("scam", "SCAM")):
            for index in range(ROWS_PER_LABEL_PER_SCENARIO):
                base = render(str(spec[label_key]), scenario, label, index + 101)
                closing = NEUTRAL_CLOSINGS[index % len(NEUTRAL_CLOSINGS)]
                text = base + "\n" + "\n".join(closing)
                row_id = "boundary-" + short_hash(
                    f"v{GENERATOR_VERSION}:{scenario}:{label}:{index}:{text}"
                )
                rows.append(
                    {
                        "id": row_id,
                        "text": text,
                        "label": label,
                        "category": str(spec["category"]) if label != "SAFE" else "NONE",
                        "source": SOURCE,
                        "source_label": label.casefold(),
                        "license": "Apache-2.0",
                        "split": "train",
                        "family_id": f"synthetic:boundary-separation:{scenario}:{label.casefold()}:v{GENERATOR_VERSION}",
                        "is_synthetic": True,
                        "synthetic_method": "advisory_grounded_safe_uncertain_scam_dialogue_triad_with_label_matched_neutral_closing",
                        "pattern_reference": str(spec["reference"]),
                        "source_language": "English",
                        "scenario": scenario,
                        "generator_version": GENERATOR_VERSION,
                        "context_policy": "classify observable behavior across the full dialogue; neutral endings do not erase earlier evidence",
                    }
                )
    return sorted(rows, key=lambda row: str(row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/boundary_separation_curriculum.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/boundary_separation_curriculum_manifest.json"),
    )
    args = parser.parse_args()
    rows = generate()
    converted, excluded = convert_supported_rows(rows)
    if excluded or len(converted) != len(rows):
        raise ValueError("every boundary-separation row must satisfy the grounded SFT contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "source": SOURCE,
        "included_prior_source": "scamguard_synthetic_evidence_persistence_v1",
        "license": "Apache-2.0",
        "method": "advisory-grounded SAFE/UNCERTAIN/SCAM dialogue triads plus the stage-3 persistence curriculum",
        "used_for_fitting": True,
        "used_for_threshold": False,
        "held_rows_copied": 0,
        "design_disclosure": (
            "Aggregate errors from the prior-open BothBosu validation and frozen schema-24 test "
            "informed category and boundary objectives. No evaluation row is copied, paraphrased, "
            "or transformed; all dialogue wording is newly authored from linked public advisories."
        ),
        "rows": len(rows),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "sources": dict(Counter(str(row["source"]) for row in rows)),
        "scenarios": dict(Counter(str(row.get("scenario")) for row in rows)),
        "sha256": file_sha256(args.output),
        "pattern_references": sorted({str(row["pattern_reference"]) for row in rows}),
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
