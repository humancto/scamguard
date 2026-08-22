#!/usr/bin/env python3
"""Generate structure-matched legitimate/scam call-opening minimal pairs."""

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

GENERATOR_VERSION = 1
SOURCE = "scamguard_synthetic_call_minimal_pairs_v1"
SYNTHETIC_METHOD = "paired_call_structure_minimal_contrast_advisory_grounded_original_copy"
PATTERN_REFERENCE = "https://consumer.ftc.gov/articles/how-avoid-scam"
HOLDOUT_SCENARIOS = (
    "financial_planning",
    "health_scheduling",
    "parcel_service",
    "technology_service",
)
FORBIDDEN_SAFE_CUES = (
    "never ask",
    "will not ask",
    "do not share",
    "official app",
    "gift card",
    "security code",
    "this is not a scam",
    "verification code",
    "remote access",
    "wire transfer",
    "cryptocurrency",
    "do not tell",
    "sign in",
)
RISK_MECHANISMS: dict[str, dict[str, str]] = {
    "credential_code": {
        "category": "CREDENTIAL_THEFT",
        "action": (
            "read me the six-digit verification code sent to your phone so I can "
            "complete the request"
        ),
    },
    "remote_access": {
        "category": "CREDENTIAL_THEFT",
        "action": (
            "install the remote support tool and give me screen sharing access so I can "
            "complete the request"
        ),
    },
    "protection_transfer": {
        "category": "FINANCIAL",
        "action": (
            "move the account balance by wire transfer to a temporary protection account "
            "before we complete the request"
        ),
    },
    "crypto_fee": {
        "category": "FINANCIAL",
        "action": (
            "settle the service fee with cryptocurrency at a kiosk before I complete "
            "the request"
        ),
    },
    "secrecy_isolation": {
        "category": "IDENTITY_IMPERSONATION",
        "action": (
            "do not tell the service desk or anyone else about this call while I complete "
            "the request"
        ),
    },
    "login_link": {
        "category": "CREDENTIAL_THEFT",
        "action": (
            "open the link I just sent and sign in with your account details to complete "
            "the request"
        ),
    },
}


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_context(
    spec: dict[str, tuple[str, ...]], structure: str, variant: int
) -> tuple[str, str, str]:
    desk = spec["desk"][(variant + len(structure)) % 4]
    topic = spec["topic"][(variant * 3 + len(structure)) % 4]
    request = spec["request"][variant % 4]
    detail = spec["detail"][(variant // 2) % 4]
    safe_action = spec["action"][(variant * 3 + variant // 2) % 4]
    if structure == "inbound":
        context = (
            f"AGENT: Good morning, you've reached the {desk}. What can I help with?\n"
            f"CUSTOMER: {request}.\n"
            f"AGENT: I have opened the existing service request about {topic}.\n"
            f"CUSTOMER: Thanks. {detail}."
        )
    elif structure == "callback":
        context = (
            f"AGENT: Hello, I'm returning the callback requested about {topic}.\n"
            f"CUSTOMER: Yes, now is a good time. {detail}.\n"
            f"AGENT: I have the existing note from the {desk}.\n"
            f"CUSTOMER: Great. {request}."
        )
    elif structure == "transfer":
        context = (
            f"AGENT: Hello, you've reached the {desk}. How can I help?\n"
            f"CUSTOMER: Hi, I was transferred about {topic}. {request}.\n"
            "AGENT: Yes, I have the existing service note in front of me.\n"
            f"CUSTOMER: Thanks. {detail}."
        )
    elif structure == "outbound_update":
        context = (
            f"AGENT: Hi, this is the {desk} with the update requested about {topic}.\n"
            f"CUSTOMER: Thanks for getting back to me. {detail}.\n"
            "AGENT: I have the existing service request open now.\n"
            f"CUSTOMER: Good. {request}."
        )
    else:
        raise ValueError(f"unknown dialogue structure: {structure}")
    return context, topic, safe_action


def generate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, spec in SCENARIOS.items():
        for structure in STRUCTURES:
            for variant, (mechanism, risk) in enumerate(RISK_MECHANISMS.items()):
                context, topic, safe_action = render_context(spec, structure, variant)
                pair_id = "call-pair-" + short_hash(
                    f"v{GENERATOR_VERSION}:{scenario}:{structure}:{mechanism}:{context}"
                )
                family_id = (
                    f"synthetic:call_minimal_pair:{scenario}:{structure}:{mechanism}:"
                    f"v{GENERATOR_VERSION}"
                )
                endings = {
                    "SAFE": f"AGENT: For {topic}, {safe_action}.",
                    "SCAM": f"AGENT: For {topic}, {risk['action']}.",
                }
                for label, ending in endings.items():
                    text = f"{context}\n{ending}"
                    if label == "SAFE" and any(
                        cue in text.casefold() for cue in FORBIDDEN_SAFE_CUES
                    ):
                        raise ValueError(f"explicit safety cue leaked into {pair_id}")
                    signal_names = sorted(
                        match.signal.value for match in extract_signal_matches(text)
                    )
                    if label == "SCAM" and not signal_names:
                        raise ValueError(f"SCAM minimal pair lacks extractive evidence: {pair_id}")
                    row_id = f"{pair_id}-{'safe' if label == 'SAFE' else 'scam'}"
                    rows.append(
                        {
                            "id": row_id,
                            "text": text,
                            "label": label,
                            "category": "NONE" if label == "SAFE" else risk["category"],
                            "source": SOURCE,
                            "source_label": (
                                "synthetic_legitimate_call_minimal_pair"
                                if label == "SAFE"
                                else "synthetic_scam_call_minimal_pair"
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
                            "risk_mechanism": mechanism,
                            "evidence_signals": signal_names,
                            "generator_version": GENERATOR_VERSION,
                            "shared_context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                            "minimal_contrast_field": "final_agent_action",
                            "external_benchmark_text_copied": False,
                            "selection_signal": (
                                "aggregate AppTek early-window and open balanced-dialogue errors"
                            ),
                        }
                    )

    normalized = {" ".join(str(row["text"]).casefold().split()) for row in rows}
    if len(normalized) != len(rows) or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("call minimal-pair generator produced a duplicate row")
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_id"])].append(row)
    for family_id, pair in grouped.items():
        if len(pair) != 2 or {str(row["label"]) for row in pair} != {"SAFE", "SCAM"}:
            raise ValueError(f"invalid minimal-pair family: {family_id}")
        contexts = {str(row["text"]).rsplit("\n", 1)[0] for row in pair}
        if len(contexts) != 1:
            raise ValueError(f"minimal-pair context differs: {family_id}")
    return sorted(rows, key=lambda row: str(row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/call_minimal_pairs.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/call_minimal_pairs_manifest.json"),
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
        "pair_families": len({str(row["family_id"]) for row in rows}),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "scenarios": dict(Counter(str(row["scenario"]) for row in rows)),
        "dialogue_structures": dict(
            Counter(str(row["dialogue_structure"]) for row in rows)
        ),
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
