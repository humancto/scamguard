#!/usr/bin/env python3
"""Build call-disjoint real SAFE and human-grounded action-state artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scamguard.metrics import file_sha256

try:
    from scripts.build_schema19_call_windows import write_jsonl
    from scripts.build_taskmaster_hard_negatives import privacy_normalize
    from scripts.fetch_harper_valley import LICENSE_SHA256, REVISION, verify_repository
    from scripts.generate_call_action_states import CONTRAST_STATES, TARGET_KEYS
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_schema19_call_windows import write_jsonl  # type: ignore[no-redef]
    from build_taskmaster_hard_negatives import privacy_normalize  # type: ignore[no-redef]
    from fetch_harper_valley import (  # type: ignore[no-redef]
        LICENSE_SHA256,
        REVISION,
        verify_repository,
    )
    from generate_call_action_states import (  # type: ignore[no-redef]
        CONTRAST_STATES,
        TARGET_KEYS,
    )


SOURCE = "harper_valley_bank_human_calls"
STATE_SOURCE = "scamguard_harper_grounded_action_states_v1"
LICENSE = "CC-BY-4.0"
MAX_CONTEXT_CHARS = 1800
HOLDOUT_TASKS = {"get branch hours", "replace card"}
SENSITIVE_TASKS = {
    "check balance",
    "order checks",
    "pay bill",
    "replace card",
    "reset password",
    "transfer money",
}
IRREVERSIBLE_TASKS = {"pay bill", "transfer money"}
DATA_ACTS = {
    "gridspace_confirm_data",
    "gridspace_data_communication",
    "gridspace_data_question",
}

ROUTINE_ENDINGS = (
    "AGENT: That completes the request. You do not need to provide anything else on this call.",
    "AGENT: The ordinary request is complete, and there is no additional payment or security step.",
    "AGENT: Everything requested is now recorded. No password or one-time code is needed.",
    "AGENT: I have finished the normal service request. There is nothing else you need to do.",
)
VERIFIED_ENDINGS = (
    (
        "AGENT: To review this later, open the official bank app yourself or call the number "
        "printed on your card. Do not use a link or number supplied in this call."
    ),
    (
        "AGENT: If you want to confirm the result, end this call and sign in through the bank "
        "site you normally use. Never read a password or one-time code to a caller."
    ),
    (
        "AGENT: You can verify the request independently in the official app or at a branch. "
        "I will not ask you to transfer money or disclose a security code."
    ),
    (
        "AGENT: For an independent check, contact the bank through the number on your "
        "statement. Do not install software or share a verification code with me."
    ),
)
UNRESOLVED_ENDINGS = (
    (
        "AGENT: I see an account-security note, but I have not verified what it concerns. "
        "No transfer or disclosure is authorized yet."
    ),
    (
        "AGENT: There may be another issue attached to the account. I cannot confirm it on "
        "this call, so no action should be taken yet."
    ),
    (
        "AGENT: A separate review appears to be pending, but its purpose is unclear. Do not "
        "send money or provide a code until it is verified."
    ),
    (
        "AGENT: I found an unconfirmed alert. I can only say that it needs review; I am not "
        "asking you to act on it now."
    ),
)
HARMFUL_ENDINGS = {
    "check balance": (
        "AGENT: To protect the balance, transfer all of it to the safe account number I give "
        "you and read me the one-time code right now."
    ),
    "get branch hours": (
        "AGENT: Before I give the hours, buy a gift card for the verification fee and tell me "
        "the code while I stay on the line."
    ),
    "order checks": (
        "AGENT: To release the checks, tell me the full account number, online password, and "
        "the one-time code sent to your phone."
    ),
    "pay bill": (
        "AGENT: The payment failed, so pay it again with gift cards and read each card number "
        "to me before we disconnect."
    ),
    "replace card": (
        "AGENT: To replace the card, read me the full card number, PIN, security code, and the "
        "one-time code from the text message."
    ),
    "reset password": (
        "AGENT: Install the remote-control app I name, give me access to the screen, and read "
        "me the verification code immediately."
    ),
    "schedule appointment": (
        "AGENT: The appointment requires a refundable gift card deposit. Buy the card now and "
        "tell me its code on this call."
    ),
    "transfer money": (
        "AGENT: Move the money to the secure account I specify and read me the one-time code "
        "so I can complete the transfer now."
    ),
}


def render_context(segments: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
    lines: list[tuple[str, dict[str, object]]] = []
    for segment in segments:
        utterance = str(segment.get("human_transcript") or segment.get("transcript") or "").strip()
        if not utterance:
            continue
        role = "AGENT" if segment.get("speaker_role") == "agent" else "CUSTOMER"
        lines.append((f"{role}: {privacy_normalize(utterance)}", segment))
    selected: list[tuple[str, dict[str, object]]] = []
    characters = 0
    for line, segment in reversed(lines):
        addition = len(line) + (1 if selected else 0)
        if selected and characters + addition > MAX_CONTEXT_CHARS:
            break
        selected.append((line, segment))
        characters += addition
    selected.reverse()
    if not selected:
        raise ValueError("HarperValleyBank call has no usable transcript")
    return "\n".join(line for line, _ in selected), [segment for _, segment in selected]


def action_targets(task: str, segments: list[dict[str, object]]) -> dict[str, bool]:
    acts = {str(act) for segment in segments for act in segment.get("dialog_acts", [])}
    agent_data_question = any(
        segment.get("speaker_role") == "agent"
        and "gridspace_data_question" in segment.get("dialog_acts", [])
        for segment in segments
    )
    targets = {
        "sensitive_action_language": task in SENSITIVE_TASKS or bool(acts & DATA_ACTS),
        "requested_disclosure_or_transfer": agent_data_question or task in IRREVERSIBLE_TASKS,
        "caller_controls_target": False,
        "official_self_navigation": False,
        "independent_verification": False,
        "pressure_or_secrecy": False,
        "irreversible_action": task in IRREVERSIBLE_TASKS,
    }
    if tuple(targets) != TARGET_KEYS:
        raise RuntimeError("HarperValleyBank action target schema changed")
    return targets


def state_targets(state: str, base: dict[str, bool], task: str) -> dict[str, bool]:
    if state == "routine_safe":
        return dict(base)
    if state == "verified_safe":
        return {
            "sensitive_action_language": True,
            "requested_disclosure_or_transfer": False,
            "caller_controls_target": False,
            "official_self_navigation": True,
            "independent_verification": True,
            "pressure_or_secrecy": False,
            "irreversible_action": False,
        }
    if state == "unresolved":
        return {
            "sensitive_action_language": True,
            "requested_disclosure_or_transfer": False,
            "caller_controls_target": False,
            "official_self_navigation": False,
            "independent_verification": False,
            "pressure_or_secrecy": False,
            "irreversible_action": False,
        }
    if state == "harmful_scam":
        return {
            "sensitive_action_language": True,
            "requested_disclosure_or_transfer": True,
            "caller_controls_target": True,
            "official_self_navigation": False,
            "independent_verification": False,
            "pressure_or_secrecy": True,
            "irreversible_action": task != "reset password",
        }
    raise ValueError(f"unknown state: {state}")


def build(repository: Path, output: Path) -> dict[str, object]:
    source_manifest = verify_repository(repository)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite HarperValleyBank output: {output}")
    transcript_paths = sorted((repository / "data" / "transcript").glob("*.json"))
    real_train: list[dict[str, object]] = []
    state_train: list[dict[str, object]] = []
    real_validation: list[dict[str, object]] = []
    state_validation: list[dict[str, object]] = []
    task_counts: Counter[str] = Counter()
    fallback_segments = 0
    for transcript_path in transcript_paths:
        sid = transcript_path.stem
        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
        metadata = json.loads(
            (repository / "data" / "metadata" / f"{sid}.json").read_text(encoding="utf-8")
        )
        tasks = {str(item["task_type"]) for item in metadata.get("tasks", [])}
        if len(tasks) != 1:
            raise ValueError(f"HarperValleyBank call does not have one task: {sid}")
        task = next(iter(tasks))
        if task not in HARMFUL_ENDINGS:
            raise ValueError(f"unknown HarperValleyBank task: {task}")
        task_counts[task] += 1
        context, selected_segments = render_context(segments)
        fallback_segments += sum(
            not str(segment.get("human_transcript", "")).strip() for segment in selected_segments
        )
        split = "validation" if task in HOLDOUT_TASKS else "train"
        family_id = f"harper-valley:{sid}"
        base_targets = action_targets(task, selected_segments)
        real_row: dict[str, object] = {
            "id": f"hvb-real-{sid}",
            "text": context,
            "label": "SAFE",
            "category": "NONE",
            "source": SOURCE,
            "source_label": f"legitimate_banking_call:{task}",
            "license": LICENSE,
            "split": split,
            "family_id": family_id,
            "is_synthetic": False,
            "source_language": "English",
            "source_record_id": sid,
            "source_task": task,
            "source_revision": REVISION,
            "context_policy": f"latest_complete_turns_capped_at_{MAX_CONTEXT_CHARS}_characters",
            "action_targets": base_targets,
            "action_label_method": "source_task_and_dialog_act_weak_supervision_v1",
            "action_verdict_weight": 1.0,
        }
        (real_validation if split == "validation" else real_train).append(real_row)
        variant = int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % len(ROUTINE_ENDINGS)
        contrast_id = "hvb-action-state-" + hashlib.sha256(f"v1:{sid}".encode()).hexdigest()[:16]
        endings = {
            "routine_safe": ROUTINE_ENDINGS[variant],
            "verified_safe": VERIFIED_ENDINGS[variant],
            "unresolved": UNRESOLVED_ENDINGS[variant],
            "harmful_scam": HARMFUL_ENDINGS[task],
        }
        for state in CONTRAST_STATES:
            label = {
                "routine_safe": "SAFE",
                "verified_safe": "SAFE",
                "unresolved": "UNCERTAIN",
                "harmful_scam": "SCAM",
            }[state]
            state_row: dict[str, object] = {
                "id": f"{contrast_id}-{state}",
                "text": f"{context}\n{endings[state]}",
                "label": label,
                "category": "CREDENTIAL_THEFT"
                if state == "harmful_scam"
                and task in {"replace card", "reset password", "order checks"}
                else ("FINANCIAL" if state == "harmful_scam" else "NONE"),
                "source": STATE_SOURCE,
                "source_label": f"human_grounded_action_state:{state}:{task}",
                "license": LICENSE,
                "split": split,
                "family_id": family_id,
                "contrast_id": contrast_id,
                "contrast_state": state,
                "is_synthetic": True,
                "synthetic_method": "minimal_final_turn_transformation_of_cc_by_human_call_v1",
                "source_language": "English",
                "source_record_id": sid,
                "source_task": task,
                "source_revision": REVISION,
                "action_targets": state_targets(state, base_targets, task),
                "shared_context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "minimal_contrast_field": "final_agent_action_state",
                "human_grounded": True,
                "external_benchmark_text_copied": False,
                "pattern_reference": "https://consumer.ftc.gov/articles/how-avoid-scam",
            }
            (state_validation if split == "validation" else state_train).append(state_row)

    output.mkdir(parents=True)
    artifacts = {
        "real_train": (
            output / "harper_real_train.jsonl",
            sorted(real_train, key=lambda row: str(row["id"])),
        ),
        "state_train": (
            output / "harper_state_train.jsonl",
            sorted(state_train, key=lambda row: str(row["id"])),
        ),
        "call_validation": (
            output / "harper_call_validation.jsonl",
            sorted(real_validation, key=lambda row: str(row["id"])),
        ),
        "state_validation": (
            output / "harper_state_validation.jsonl",
            sorted(state_validation, key=lambda row: str(row["id"])),
        ),
    }
    for _, (path, rows) in artifacts.items():
        write_jsonl(path, rows)
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "source": SOURCE,
        "state_source": STATE_SOURCE,
        "repository": source_manifest["repository"],
        "revision": REVISION,
        "license": LICENSE,
        "license_sha256": LICENSE_SHA256,
        "transcript_tree_sha256": source_manifest["transcript_tree_sha256"],
        "metadata_tree_sha256": source_manifest["metadata_tree_sha256"],
        "citation": source_manifest["citation"],
        "source_calls": len(transcript_paths),
        "task_counts": dict(sorted(task_counts.items())),
        "asr_fallback_segments_in_selected_windows": fallback_segments,
        "partition": f"task-disjoint holdout: {sorted(HOLDOUT_TASKS)}",
        "policy": {
            "source_calls_are_human_human_roleplay": True,
            "real_rows_counted_separately_from_synthetic_derivatives": True,
            "speaker_ids_removed_from_model_rows": True,
            "call_family_cross_split": False,
            "audio_downloaded": False,
            "action_labels_on_real_rows": "weak supervision from task and segment dialog acts",
            "state_variants": "final-turn transformations grounded in the same human call context",
        },
        "artifacts": {
            name: {"path": str(path), "rows": len(rows), "sha256": file_sha256(path)}
            for name, (path, rows) in artifacts.items()
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository", type=Path, default=Path("data/raw/harper_valley/repository")
    )
    parser.add_argument("--output", type=Path, default=Path("data/external/harper_valley"))
    args = parser.parse_args()
    print(json.dumps(build(args.repository, args.output), indent=2))


if __name__ == "__main__":
    main()
