#!/usr/bin/env python3
"""Generate matched government-call evidence contrasts for Stage 10.

The prior-open Stage 9 promotion screen disclosed a missing boundary between coercive
government-identity robocalls, generic unresolved IVRs, and legitimate instructions to
navigate independently. No evaluation transcript is copied or paraphrased here. Every
message is original text grounded in the linked FTC government-impersonation guidance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from training.build_qwen_sft import convert_supported_rows

GENERATOR_VERSION = 1
SOURCE = "scamguard_synthetic_government_evidence_contrasts_v1"
LICENSE = "Apache-2.0"
REFERENCE = "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam"

DESKS = (
    "public-benefits record office",
    "civil-document review bureau",
    "national identity records desk",
    "retirement-services review unit",
    "resident-services case office",
    "public-insurance records team",
    "licensing status bureau",
    "citizen-records notification desk",
)
SUBJECTS = (
    "a discrepancy attached to your resident file",
    "an identity record marked for administrative review",
    "a benefits record that may require attention",
    "a document-status entry associated with your name",
    "an unresolved record in the agency system",
    "a case-reference mismatch in your profile",
    "a status review connected to your public record",
    "an identity-check entry that has not been resolved",
)
THREATS = (
    "legal action will begin and your account access will be suspended",
    "the case will be sent for enforcement and your benefits will be frozen",
    "proceedings will move forward and your public record will be restricted",
    "the matter will be escalated for legal action and your access will be blocked",
)
SAFE_PATHS = (
    "open the agency page from a bookmark you created yourself",
    "use the address printed on a letter already in your possession",
    "call the number in the official public directory after ending this call",
    "visit the government service through your saved account entry",
)
STRUCTURES = (
    (
        "automated_notice",
        "RECORDED CALL: This is the {desk}.\n"
        "RECIPIENT: The recorded message was answered.\n"
        "RECORDED CALL: We are trying to reach you about {subject}.\n"
        "RECIPIENT: What does the office need?\n"
        "RECORDED CALL: {action}.\n"
        "RECIPIENT: I have not agreed to continue.",
    ),
    (
        "live_followup",
        "CALLER: I am following up from the {desk}.\n"
        "RESIDENT: I was not expecting a call.\n"
        "CALLER: The file mentions {subject}.\n"
        "RESIDENT: How should I handle that?\n"
        "CALLER: {action}.\n"
        "RESIDENT: I will decide after this call ends.",
    ),
    (
        "transferred_case",
        "AGENT: You have reached the {desk}.\n"
        "PARTICIPANT: I was transferred from a recorded notice.\n"
        "AGENT: The notice concerns {subject}.\n"
        "PARTICIPANT: What is the next step?\n"
        "AGENT: {action}.\n"
        "PARTICIPANT: I need time to verify the office.",
    ),
    (
        "voicemail_callback",
        "VOICEMAIL: A message was left by the {desk}.\n"
        "LISTENER: The call was not requested.\n"
        "VOICEMAIL: The message refers to {subject}.\n"
        "LISTENER: No agency contact is saved with this number.\n"
        "VOICEMAIL: {action}.\n"
        "LISTENER: The message ends here.",
    ),
)


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def action_for(state: str, index: int) -> str:
    digit = 2 + (index % 7)
    if state == "coercive_scam":
        return (
            f"Press {digit} now and remain on the line. If you disconnect, "
            f"{THREATS[index % len(THREATS)]}. Have your social security number ready"
        )
    if state == "unresolved_ivr":
        return (
            f"Press {digit} to hear more about the record. This message does not identify "
            "the handling office or explain whether a response is required"
        )
    if state == "verified_safe":
        return (
            "Do not press a key, provide a code, or disclose private information on this call. "
            f"Hang up and {SAFE_PATHS[index % len(SAFE_PATHS)]}. There is no immediate deadline"
        )
    raise ValueError(f"unknown contrast state: {state}")


def generate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    states = (
        ("coercive_scam", "SCAM"),
        ("unresolved_ivr", "UNCERTAIN"),
        ("verified_safe", "SAFE"),
    )
    for structure_index, (structure, template) in enumerate(STRUCTURES):
        for lexical_index in range(len(DESKS)):
            pair_index = structure_index * len(DESKS) + lexical_index
            contrast_group = f"v{GENERATOR_VERSION}:{structure}:{lexical_index}"
            frame = template.format(
                desk=DESKS[lexical_index],
                subject=SUBJECTS[lexical_index],
                action="<DECISIVE_ACTION>",
            )
            frame_sha256 = hashlib.sha256(frame.encode("utf-8")).hexdigest()
            for state, label in states:
                text = template.format(
                    desk=DESKS[lexical_index],
                    subject=SUBJECTS[lexical_index],
                    action=action_for(state, pair_index),
                )
                identifier = "government-evidence-" + short_hash(
                    f"{contrast_group}:{state}:{text}"
                )
                rows.append(
                    {
                        "id": identifier,
                        "text": text,
                        "label": label,
                        "category": "GOVERNMENT_LEGAL" if label == "SCAM" else "NONE",
                        "source": SOURCE,
                        "source_label": state,
                        "license": LICENSE,
                        "split": "train",
                        "family_id": f"synthetic:government-evidence:{contrast_group}",
                        "is_synthetic": True,
                        "synthetic_method": (
                            "matched_original_government_call_evidence_triad_v1"
                        ),
                        "pattern_reference": REFERENCE,
                        "source_language": "English",
                        "generator_version": GENERATOR_VERSION,
                        "contrast_state": state,
                        "contrast_group": contrast_group,
                        "common_frame_sha256": frame_sha256,
                        "design_disclosure": (
                            "Stage 9 open promotion errors informed the abstract boundary; no "
                            "evaluation row was copied, paraphrased, or transformed"
                        ),
                    }
                )
    return sorted(rows, key=lambda row: str(row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/government_evidence_contrasts.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/government_evidence_contrasts_manifest.json"),
    )
    args = parser.parse_args()
    rows = generate()
    converted, excluded = convert_supported_rows(rows)
    if excluded or len(converted) != len(rows):
        raise ValueError("every government evidence contrast must satisfy the SFT contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "artifact_schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "source": SOURCE,
        "license": LICENSE,
        "method": "matched original government-call SAFE/UNCERTAIN/SCAM evidence triads",
        "pattern_references": [REFERENCE],
        "used_for_fitting": True,
        "used_for_threshold": False,
        "held_rows_copied": 0,
        "evaluation_rows_read_by_generator": 0,
        "design_disclosure": (
            "Stage 9 open promotion errors informed the abstract evidence boundary; no "
            "evaluation row was copied, paraphrased, or transformed"
        ),
        "rows": len(rows),
        "families": len({str(row["family_id"]) for row in rows}),
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "states": dict(
            sorted(Counter(str(row["contrast_state"]) for row in rows).items())
        ),
        "sha256": file_sha256(args.output),
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
