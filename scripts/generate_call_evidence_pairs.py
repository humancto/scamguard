#!/usr/bin/env python3
"""Generate diverse call pairs that separate suspicious framing from harmful actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.signals import extract_signal_matches

try:
    from scripts.generate_legitimate_call_openings import SCENARIOS, STRUCTURES
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from generate_legitimate_call_openings import SCENARIOS, STRUCTURES  # type: ignore[no-redef]

GENERATOR_VERSION = 2
SOURCE = "scamguard_synthetic_call_evidence_pairs_v2"
SYNTHETIC_METHOD = "paired_call_evidence_action_counterfactual_advisory_grounded_original_copy"
PATTERN_REFERENCE = "https://consumer.ftc.gov/articles/how-avoid-scam"
HOLDOUT_SCENARIOS = (
    "financial_planning",
    "health_scheduling",
    "parcel_service",
    "technology_service",
)
CONTEXT_FRAMES = {
    "neutral": "I have the reference information here and want to understand the next step",
    "unexpected": "I was not expecting a call, so I want to understand each step before continuing",
    "skeptical": (
        "I received a suspicious message earlier, but I do not know if it relates to this request"
    ),
    "privacy": "I prefer to keep personal details limited to what is needed for this request",
}
SAFE_ACTION_STYLES = (
    "{scenario_action}",
    (
        "I can note the question, and you can verify the request with the service desk "
        "before continuing"
    ),
    "I can explain the status already shown in the request and leave the next step to you",
    "I can document the issue and arrange a standard follow-up at a time you choose",
)
FORBIDDEN_SAFE_ENDING_CUES = (
    "verification code",
    "security code",
    "remote access",
    "remote support",
    "screen sharing",
    "wire transfer",
    "protection account",
    "cryptocurrency",
    "gift card",
    "processing fee",
    "do not tell",
    "keep this secret",
    "between us",
    "do not contact",
    "sign in with",
)
RISK_MECHANISMS: dict[str, dict[str, object]] = {
    "credential_code": {
        "category": "CREDENTIAL_THEFT",
        "actions": (
            "read me the verification code sent to your phone so I can finish the request",
            "tell me the one-time code that just arrived so I can approve the request",
            "share the 6-digit code from the message so I can continue",
            "give me the login verification code while we are on this call",
        ),
    },
    "remote_access": {
        "category": "CREDENTIAL_THEFT",
        "actions": (
            "install the remote support tool and give me screen sharing access",
            "open a remote desktop session so I can control the device",
            "download the screen sharing application and accept my access request",
            "enable remote access and let me take control while we complete the request",
        ),
    },
    "protection_transfer": {
        "category": "FINANCIAL",
        "actions": (
            "move the balance by wire transfer to a temporary protection account",
            "send the available funds by wire transfer to the safe account I provide",
            "transfer the money to a protection account so it cannot be taken",
            "send the account balance by wire to the secure holding account now",
        ),
    },
    "crypto_fee": {
        "category": "FINANCIAL",
        "actions": (
            "settle the service fee with cryptocurrency at a kiosk",
            "buy bitcoin and send it to the wallet address I provide",
            "pay the required fee in crypto before the request can continue",
            "use a cryptocurrency ATM to complete the payment while I stay on the line",
        ),
    },
    "secrecy_isolation": {
        "category": "IDENTITY_IMPERSONATION",
        "actions": (
            "do not tell the service desk or anyone else about this call",
            "keep this secret from your family and the staff while I finish the request",
            "keep this between us and do not discuss the call with anyone",
            "do not contact the service desk until I say the request is complete",
        ),
    },
    "login_link": {
        "category": "CREDENTIAL_THEFT",
        "actions": (
            "open the link I just sent and sign in with your account details",
            "click the login link in my message and enter the account password",
            "follow the link I sent and sign in so I can validate the request",
            "visit the link in the text message and log in while I wait",
        ),
    },
    "gift_card": {
        "category": "FINANCIAL",
        "actions": (
            "buy a gift card and read the number to me to settle the charge",
            "purchase gift cards and send me the codes before we continue",
            "pay the balance with a gift card and give me the PIN",
            "get a store gift card for the fee and read the code over this call",
        ),
    },
    "advance_fee": {
        "category": "FINANCIAL",
        "actions": (
            "pay a processing fee before the request can be released",
            "send the refundable activation fee so I can unlock the request",
            "cover the clearance fee now before the service can continue",
            "pay the release fee first and I will complete the request afterward",
        ),
    },
}


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_context(
    spec: dict[str, tuple[str, ...]],
    structure: str,
    context_variant: int,
    mechanism_variant: int,
) -> tuple[str, str, str]:
    frame_name = tuple(CONTEXT_FRAMES)[context_variant]
    frame = CONTEXT_FRAMES[frame_name]
    variant = context_variant + mechanism_variant
    desk = spec["desk"][(variant + len(structure)) % 4]
    topic = spec["topic"][(variant * 3 + len(structure)) % 4]
    request = spec["request"][variant % 4]
    detail = spec["detail"][(variant // 2) % 4]
    scenario_action = spec["action"][(variant * 3 + context_variant) % 4]
    if structure == "inbound":
        context = (
            f"AGENT: Good morning, you've reached the {desk}. What can I help with?\n"
            f"CUSTOMER: {request}. {frame}.\n"
            f"AGENT: I have opened the existing service request about {topic}.\n"
            f"CUSTOMER: Thanks. {detail}."
        )
    elif structure == "callback":
        context = (
            f"AGENT: Hello, I'm returning the callback requested about {topic}.\n"
            f"CUSTOMER: Yes, now is a good time. {detail}. {frame}.\n"
            f"AGENT: I have the existing note from the {desk}.\n"
            f"CUSTOMER: Great. {request}."
        )
    elif structure == "transfer":
        context = (
            f"AGENT: Hello, you've reached the {desk}. How can I help?\n"
            f"CUSTOMER: I was transferred about {topic}. {request}.\n"
            "AGENT: I have the existing service note in front of me.\n"
            f"CUSTOMER: Thanks. {detail}. {frame}."
        )
    elif structure == "outbound_update":
        context = (
            f"AGENT: Hi, this is the {desk} with the requested update about {topic}.\n"
            f"CUSTOMER: Thanks for getting back to me. {detail}.\n"
            "AGENT: I have the existing service request open now.\n"
            f"CUSTOMER: Good. {request}. {frame}."
        )
    else:
        raise ValueError(f"unknown dialogue structure: {structure}")
    safe_action = SAFE_ACTION_STYLES[(context_variant + mechanism_variant) % 4].format(
        scenario_action=scenario_action
    )
    return context, topic, safe_action


def generate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, spec in SCENARIOS.items():
        for structure in STRUCTURES:
            for context_variant, frame_name in enumerate(CONTEXT_FRAMES):
                for mechanism_variant, (mechanism, risk) in enumerate(
                    RISK_MECHANISMS.items()
                ):
                    context, topic, safe_action = render_context(
                        spec, structure, context_variant, mechanism_variant
                    )
                    action = risk["actions"][context_variant]
                    pair_id = "call-evidence-pair-" + short_hash(
                        f"v{GENERATOR_VERSION}:{scenario}:{structure}:{frame_name}:"
                        f"{mechanism}:{context}"
                    )
                    family_id = (
                        f"synthetic:call_evidence_pair:{scenario}:{structure}:{frame_name}:"
                        f"{mechanism}:v{GENERATOR_VERSION}"
                    )
                    endings = {
                        "SAFE": f"AGENT: For {topic}, {safe_action}.",
                        "SCAM": f"AGENT: For {topic}, {action}.",
                    }
                    for label, ending in endings.items():
                        if label == "SAFE" and any(
                            cue in ending.casefold() for cue in FORBIDDEN_SAFE_ENDING_CUES
                        ):
                            raise ValueError(f"risk action leaked into SAFE ending: {pair_id}")
                        text = f"{context}\n{ending}"
                        signal_names = sorted(
                            {match.signal.value for match in extract_signal_matches(text)}
                        )
                        if label == "SCAM" and not signal_names:
                            raise ValueError(f"SCAM evidence pair lacks a signal: {pair_id}")
                        rows.append(
                            {
                                "id": f"{pair_id}-{'safe' if label == 'SAFE' else 'scam'}",
                                "text": text,
                                "label": label,
                                "category": "NONE" if label == "SAFE" else risk["category"],
                                "source": SOURCE,
                                "source_label": (
                                    "synthetic_legitimate_call_evidence_pair"
                                    if label == "SAFE"
                                    else "synthetic_scam_call_evidence_pair"
                                ),
                                "license": "Apache-2.0",
                                "split": "train",
                                "family_id": family_id,
                                "pair_id": pair_id,
                                "pair_label": label,
                                "is_synthetic": True,
                                "synthetic_method": SYNTHETIC_METHOD,
                                "pattern_reference": PATTERN_REFERENCE,
                                "source_language": "English",
                                "scenario": scenario,
                                "dialogue_structure": structure,
                                "context_frame": frame_name,
                                "risk_mechanism": mechanism,
                                "evidence_signals": signal_names,
                                "generator_version": GENERATOR_VERSION,
                                "shared_context_sha256": hashlib.sha256(
                                    context.encode("utf-8")
                                ).hexdigest(),
                                "minimal_contrast_field": "final_agent_action",
                                "external_benchmark_text_copied": False,
                                "selection_signal": (
                                    "schema17 paired underfit and aggregate open benign-dialogue "
                                    "false positives; no benchmark text copied"
                                ),
                            }
                        )

    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("call evidence-pair generator produced a duplicate ID")
    if len({" ".join(str(row["text"]).casefold().split()) for row in rows}) != len(rows):
        raise ValueError("call evidence-pair generator produced duplicate text")
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {str(row["label"]) for row in pair} != {"SAFE", "SCAM"}:
            raise ValueError(f"invalid evidence pair: {pair_id}")
        if len({str(row["text"]).rsplit("\n", 1)[0] for row in pair}) != 1:
            raise ValueError(f"evidence-pair context differs: {pair_id}")
    return sorted(rows, key=lambda row: str(row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/call_evidence_pairs_v2.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/call_evidence_pairs_v2_manifest.json"),
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
        "source": SOURCE,
        "license": "Apache-2.0",
        "method": SYNTHETIC_METHOD,
        "pattern_reference": PATTERN_REFERENCE,
        "rows": len(rows),
        "pair_families": len({str(row["pair_id"]) for row in rows}),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "scenarios": dict(Counter(str(row["scenario"]) for row in rows)),
        "dialogue_structures": dict(
            Counter(str(row["dialogue_structure"]) for row in rows)
        ),
        "context_frames": dict(Counter(str(row["context_frame"]) for row in rows)),
        "risk_mechanisms": dict(
            Counter(str(row["risk_mechanism"]) for row in rows)
        ),
        "holdout_scenarios": list(HOLDOUT_SCENARIOS),
        "shared_context_turns": 4,
        "minimal_contrast_field": "final_agent_action",
        "used_for_fitting": True,
        "used_for_threshold": False,
        "external_benchmark_text_copied": False,
        "sha256": file_sha256(args.output),
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
