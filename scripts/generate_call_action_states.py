#!/usr/bin/env python3
"""Generate action-state call contrasts for separate evidence and context supervision."""

# ruff: noqa: E501 -- full dialogue turns remain intact for label review.

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.signals import extract_signal_matches

try:
    from scripts.generate_call_evidence_pairs import (
        CONTEXT_FRAMES,
        HOLDOUT_SCENARIOS,
        RISK_MECHANISMS,
        render_context,
    )
    from scripts.generate_legitimate_call_openings import SCENARIOS, STRUCTURES
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from generate_call_evidence_pairs import (  # type: ignore[no-redef]
        CONTEXT_FRAMES,
        HOLDOUT_SCENARIOS,
        RISK_MECHANISMS,
        render_context,
    )
    from generate_legitimate_call_openings import (  # type: ignore[no-redef]
        SCENARIOS,
        STRUCTURES,
    )

GENERATOR_VERSION = 1
SOURCE = "scamguard_synthetic_call_action_states_v1"
SYNTHETIC_METHOD = "call_action_state_counterfactual_advisory_grounded_original_copy"
PATTERN_REFERENCE = "https://consumer.ftc.gov/articles/how-avoid-scam"
CONTRAST_STATES = ("routine_safe", "verified_safe", "unresolved", "harmful_scam")
TARGET_KEYS = (
    "sensitive_action_language",
    "requested_disclosure_or_transfer",
    "caller_controls_target",
    "official_self_navigation",
    "independent_verification",
    "pressure_or_secrecy",
    "irreversible_action",
)


VERIFIED_SAFE_ACTIONS: dict[str, tuple[str, ...]] = {
    "credential_code": (
        "do not read any verification code to me; open the official app yourself and review the request there",
        "keep every one-time code private and independently check the request in the official app",
        "I do not need the code sent to your phone; use the official account page you open yourself",
        "never give me a login code; end this call and use the verified service number if you need help",
    ),
    "remote_access": (
        "do not install a remote support tool; use the help page inside the official app instead",
        "keep screen sharing disabled and arrange an in-person or official-app support appointment",
        "I do not need remote control of your device; follow the local settings guide you open yourself",
        "decline any remote desktop request and call the verified service number if the issue continues",
    ),
    "protection_transfer": (
        "do not move money to a protection account; independently call the number on your card",
        "leave the balance where it is and review any alert in the official banking app you open yourself",
        "no safe account transfer is required; end the call and contact your bank through a verified channel",
        "do not wire funds anywhere from this call; ask the branch to review the account independently",
    ),
    "crypto_fee": (
        "do not pay a service fee in cryptocurrency; verify any charge in the official account portal",
        "no bitcoin payment is needed; independently review the ordinary invoice before paying anything",
        "avoid every crypto kiosk instruction and call the verified billing desk about the stated fee",
        "keep cryptocurrency out of this request and use only the payment options in the official app",
    ),
    "secrecy_isolation": (
        "you may discuss this call with family and independently verify it with the service desk",
        "do not keep the request secret; pause and ask a trusted person or the verified office to review it",
        "there is no reason to hide this conversation; confirm the case through an official channel",
        "please tell anyone you trust and contact the published service number before taking another step",
    ),
    "login_link": (
        "do not sign in through a link from this call; open the official app or type the known address yourself",
        "ignore login links in messages and independently navigate to the verified account page",
        "I will not ask you to use a link I send; use the official application already on your device",
        "avoid the text-message login link and call the published service number if the account needs review",
    ),
    "gift_card": (
        "do not buy a gift card for this request; verify any balance in the official billing portal",
        "gift-card codes are not accepted; independently call the published billing number",
        "keep every gift card and PIN private and review the ordinary invoice through the official app",
        "no store card payment is required; end the call and ask the verified service desk about the charge",
    ),
    "advance_fee": (
        "do not pay a processing fee on this call; check the published fee schedule independently",
        "leave any activation fee unpaid until the official portal shows it and you verify the request",
        "no release payment is needed now; call the published service number before paying anything",
        "pause before any clearance fee and confirm it through an official channel you choose yourself",
    ),
}


UNRESOLVED_ACTIONS: dict[str, tuple[str, ...]] = {
    "credential_code": (
        "a verification prompt may arrive next, but first confirm who initiated this request",
        "the next screen mentions a one-time code, so pause until the source of the request is clear",
        "account verification is the next step, but I cannot confirm the channel from this note alone",
        "there may be a login-code step after this call; verify the contact before continuing",
    ),
    "remote_access": (
        "the case note mentions remote support, but confirm the support channel before installing anything",
        "screen sharing may be an option later, though the current record does not verify who would connect",
        "a remote session is listed as a possible next step; pause until the service identity is confirmed",
        "device access might be requested later, but this call alone is not enough to validate it",
    ),
    "protection_transfer": (
        "the note mentions moving funds after review, but the destination and authority are not yet verified",
        "an account transfer is listed as a possible step; pause until the receiving account is confirmed",
        "the balance may need attention, though this conversation does not establish where money should move",
        "a transfer instruction could follow, but first determine whether the request is genuinely authorized",
    ),
    "crypto_fee": (
        "the record mentions a cryptocurrency fee, but the billing source has not been verified",
        "a crypto payment option appears in the note; pause until the charge and recipient are confirmed",
        "the next instruction may involve bitcoin, though this call does not establish that the fee is valid",
        "cryptocurrency is mentioned as a payment step, but its destination and purpose remain unclear",
    ),
    "secrecy_isolation": (
        "the note says the matter is confidential, but it is unclear whether that restriction is appropriate",
        "privacy is requested for the case, though this call does not explain why outside review is discouraged",
        "the next step is described as private; pause until the reason for limiting contact is verified",
        "the record discourages sharing details, but the authority for that instruction is still unclear",
    ),
    "login_link": (
        "a login link may be sent after this call, but the destination has not been independently verified",
        "the note mentions signing in from a message; pause until the address and sender are confirmed",
        "an account link is listed as the next step, though this conversation does not establish its origin",
        "a sign-in message could follow, but first determine whether the channel belongs to the service",
    ),
    "gift_card": (
        "the note mentions a gift-card payment, but neither the charge nor recipient has been verified",
        "a store card appears as a possible payment method; pause until the billing authority is confirmed",
        "gift-card details may be requested later, though the reason and destination remain unclear",
        "the case refers to paying by gift card, but this call does not establish that the request is valid",
    ),
    "advance_fee": (
        "the record lists a processing fee, but the official schedule and recipient have not been verified",
        "an activation payment may be requested next; pause until the charge is independently confirmed",
        "the note mentions a release fee, though this conversation does not establish that it is legitimate",
        "a clearance charge appears in the case, but its authority and payment channel remain unclear",
    ),
}


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def targets_for(state: str, mechanism: str) -> dict[str, bool]:
    sensitive = state != "routine_safe"
    harmful = state == "harmful_scam"
    return {
        "sensitive_action_language": sensitive,
        "requested_disclosure_or_transfer": harmful
        and mechanism in {
            "credential_code",
            "protection_transfer",
            "crypto_fee",
            "login_link",
            "gift_card",
            "advance_fee",
        },
        "caller_controls_target": harmful,
        "official_self_navigation": state == "verified_safe",
        "independent_verification": state == "verified_safe",
        "pressure_or_secrecy": harmful and mechanism == "secrecy_isolation",
        "irreversible_action": harmful
        and mechanism
        in {"protection_transfer", "crypto_fee", "gift_card", "advance_fee"},
    }


def validate_action_state_rows(rows: list[dict[str, object]]) -> None:
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("action-state generator produced a duplicate ID")
    if len({" ".join(str(row["text"]).casefold().split()) for row in rows}) != len(rows):
        raise ValueError("action-state generator produced duplicate text")
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["contrast_id"])].append(row)
    for contrast_id, contrast in grouped.items():
        states = {str(row["contrast_state"]) for row in contrast}
        if len(contrast) != len(CONTRAST_STATES) or states != set(CONTRAST_STATES):
            raise ValueError(f"invalid action-state contrast: {contrast_id}")
        if len({str(row["shared_context_sha256"]) for row in contrast}) != 1:
            raise ValueError(f"action-state context hash differs: {contrast_id}")
        if len({str(row["text"]).rsplit("\n", 1)[0] for row in contrast}) != 1:
            raise ValueError(f"action-state context text differs: {contrast_id}")
        for row in contrast:
            targets = row.get("action_targets")
            if not isinstance(targets, dict) or tuple(targets) != TARGET_KEYS:
                raise ValueError(f"invalid action targets: {row['id']}")
            if not all(isinstance(value, bool) for value in targets.values()):
                raise ValueError(f"non-boolean action target: {row['id']}")
        harmful = next(row for row in contrast if row["contrast_state"] == "harmful_scam")
        if harmful["label"] != "SCAM" or not extract_signal_matches(str(harmful["text"])):
            raise ValueError(f"harmful contrast lacks SCAM evidence: {contrast_id}")


def generate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, spec in SCENARIOS.items():
        for structure in STRUCTURES:
            for context_variant, frame_name in enumerate(CONTEXT_FRAMES):
                for mechanism_variant, (mechanism, risk) in enumerate(
                    RISK_MECHANISMS.items()
                ):
                    context, topic, routine_action = render_context(
                        spec, structure, context_variant, mechanism_variant
                    )
                    contrast_id = "call-action-state-" + short_hash(
                        f"v{GENERATOR_VERSION}:{scenario}:{structure}:{frame_name}:"
                        f"{mechanism}:{context}"
                    )
                    variant = context_variant
                    endings = {
                        "routine_safe": f"AGENT: For {topic}, {routine_action}.",
                        "verified_safe": (
                            f"AGENT: For {topic}, {VERIFIED_SAFE_ACTIONS[mechanism][variant]}."
                        ),
                        "unresolved": (
                            f"AGENT: For {topic}, {UNRESOLVED_ACTIONS[mechanism][variant]}."
                        ),
                        "harmful_scam": (
                            f"AGENT: For {topic}, {risk['actions'][variant]}."
                        ),
                    }
                    labels = {
                        "routine_safe": "SAFE",
                        "verified_safe": "SAFE",
                        "unresolved": "UNCERTAIN",
                        "harmful_scam": "SCAM",
                    }
                    for state in CONTRAST_STATES:
                        label = labels[state]
                        text = f"{context}\n{endings[state]}"
                        rows.append(
                            {
                                "id": f"{contrast_id}-{state}",
                                "text": text,
                                "label": label,
                                "category": "NONE" if label != "SCAM" else risk["category"],
                                "source": SOURCE,
                                "source_label": f"synthetic_call_action_state_{state}",
                                "license": "Apache-2.0",
                                "split": "train",
                                "family_id": (
                                    f"synthetic:call_action_state:{scenario}:{structure}:"
                                    f"{frame_name}:{mechanism}:v{GENERATOR_VERSION}"
                                ),
                                "contrast_id": contrast_id,
                                "contrast_state": state,
                                "is_synthetic": True,
                                "synthetic_method": SYNTHETIC_METHOD,
                                "pattern_reference": PATTERN_REFERENCE,
                                "source_language": "English",
                                "scenario": scenario,
                                "dialogue_structure": structure,
                                "context_frame": frame_name,
                                "risk_mechanism": mechanism,
                                "action_targets": targets_for(state, mechanism),
                                "evidence_signals": sorted(
                                    {
                                        match.signal.value
                                        for match in extract_signal_matches(text)
                                    }
                                ),
                                "generator_version": GENERATOR_VERSION,
                                "shared_context_sha256": hashlib.sha256(
                                    context.encode("utf-8")
                                ).hexdigest(),
                                "minimal_contrast_field": "final_agent_action_state",
                                "external_benchmark_text_copied": False,
                                "selection_signal": (
                                    "schema19 action ranking passed while real-dialogue absolute "
                                    "calibration failed; aggregate error types only"
                                ),
                            }
                        )
    rows = sorted(rows, key=lambda row: str(row["id"]))
    validate_action_state_rows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/call_action_states_v1.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/call_action_states_v1_manifest.json"),
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
        "contrast_families": len({str(row["contrast_id"]) for row in rows}),
        "contrast_states": list(CONTRAST_STATES),
        "action_target_keys": list(TARGET_KEYS),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "scenarios": dict(Counter(str(row["scenario"]) for row in rows)),
        "risk_mechanisms": dict(Counter(str(row["risk_mechanism"]) for row in rows)),
        "holdout_scenarios": list(HOLDOUT_SCENARIOS),
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
